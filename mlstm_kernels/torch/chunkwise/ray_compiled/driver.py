import os
from typing import Tuple

import torch

try:
    import ray
except Exception as _e:
    ray = None  # type: ignore

from ...recurrent.metal.compiled import mlstm_recurrent_step__metal
from ...monitoring.memory import MemoryMonitor, MemoryPressureAbort


def _ensure_ray() -> bool:
    """Ensure Ray is initialized. Returns True if this call started Ray.

    In local_mode=1 (default), Ray runs tasks/actors in-process and should not
    spawn external daemons. If local_mode=0, a local head (raylet/GCS) is
    launched and must be cleaned up via ray.shutdown().
    """
    if ray is None:
        raise RuntimeError(
            "Ray is not installed. Install ray or select a non-ray chunkwise backend."
        )
    if not ray.is_initialized():
        local_mode = os.environ.get("XLSTM_RAY_LOCAL_MODE", "1") == "1"
        # Dashboard requested implies non-local mode
        dash = os.environ.get("XLSTM_RAY_DASHBOARD", "0") == "1"
        include_dashboard = dash and not local_mode
        dash_port = int(os.environ.get("XLSTM_RAY_DASHBOARD_PORT", "8265"))
        # Avoid extra background noise unless explicitly requested
        kwargs = dict(ignore_reinit_error=True, local_mode=local_mode)
        if not local_mode:
            kwargs.update(dict(include_dashboard=include_dashboard, dashboard_port=dash_port))
        ray.init(**kwargs)  # type: ignore[arg-type]
        return True
    return False


def _gpu_only_guard(t: torch.Tensor):
    if t.device.type != "mps":
        raise RuntimeError("Ray compiled chunkwise requires MPS device; CPU/CUDA not allowed.")


@ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)  # type: ignore[misc]
class HeadBandWorker:
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        hs: int,
        he: int,
        eps: float,
    ):
        # Store references (in local_mode these are shared, avoiding copies)
        self.q = q
        self.k = k
        self.v = v
        self.i = i
        self.f = f
        self.hs = hs
        self.he = he
        self.eps = eps

    def run(
        self,
        c_state: torch.Tensor | None,
        n_state: torch.Tensor | None,
        m_state: torch.Tensor | None,
        s_start: int,
        s_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hs, he = self.hs, self.he
        B, NH, S, DHQK = self.q.shape
        DHHV = self.v.shape[-1]

        # Initialize per-band states
        C = (
            c_state
            if c_state is not None
            else torch.zeros((B, he - hs, DHQK, DHHV), device=self.q.device, dtype=torch.float32)
        )
        N = (
            n_state
            if n_state is not None
            else torch.zeros((B, he - hs, DHQK), device=self.q.device, dtype=torch.float32)
        )
        M = (
            m_state
            if m_state is not None
            else torch.zeros((B, he - hs, 1), device=self.q.device, dtype=torch.float32)
        )
        H_out = torch.empty(
            (B, he - hs, s_end - s_start, DHHV), device=self.q.device, dtype=self.q.dtype
        )
        # CfC-hybrid experimental override
        cfc_on = os.environ.get("XLSTM_CFC_HYBRID", "0") == "1"
        cfc_dt = float(os.environ.get("XLSTM_CFC_DT", "0.01"))
        cfc_alpha = float(os.environ.get("XLSTM_CFC_ALPHA", "1.0"))
        cfc_act = os.environ.get("XLSTM_CFC_ACT", "sigmoid")
        h_cfc = None
        if cfc_on:
            h_cfc = torch.zeros((B, he - hs, DHHV), device=self.q.device, dtype=self.q.dtype)
        # Step through sequence slice
        for t in range(s_start, s_end):
            H, (C, N, M) = mlstm_recurrent_step__metal(
                q=self.q[:, hs:he, t],
                k=self.k[:, hs:he, t],
                v=self.v[:, hs:he, t],
                i=self.i[:, hs:he, t : t + 1],
                f=self.f[:, hs:he, t : t + 1],
                c=C,
                n=N,
                m=M,
                eps=self.eps,
                dtype_state=torch.float32,
            )
            if cfc_on:
                ff = torch.sigmoid(H) if cfc_act == "sigmoid" else (1.7159 * torch.tanh(0.666 * H))
                try:
                    i_t = self.i[:, hs:he, t : t + 1]
                    f_t = self.f[:, hs:he, t : t + 1]
                    lam = torch.sigmoid(cfc_alpha * (i_t + f_t)).squeeze(-1)
                except Exception:
                    lam = torch.zeros_like(h_cfc)
                denom = 1.0 + cfc_dt * lam
                h_cfc = (h_cfc + cfc_dt * ff) / denom
                H = h_cfc
            H_out[:, :, t - s_start] = H
        return H_out, C, N, M

    def mem_stats(self) -> Tuple[float, float | None, float | None]:
        """Return (rss_mb, mps_alloc_mb, mps_reserved_mb) for this worker process.
        In local_mode=1, this reports the driver's process stats.
        """
        rss_mb = 0.0
        try:
            try:
                import psutil  # type: ignore
                rss_mb = float(psutil.Process().memory_info().rss) / (1024 * 1024)
            except Exception:
                rss_mb = 0.0
            a = r = None
            try:
                a = float(torch.mps.current_allocated_memory()) / (1024 * 1024)  # type: ignore[attr-defined]
                r = float(torch.mps.current_reserved_memory()) / (1024 * 1024)  # type: ignore[attr-defined]
            except Exception:
                a = r = None
            return rss_mb, a, r
        except Exception:
            return rss_mb, None, None


def mlstm_chunkwise__ray_compiled_steps(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None,  # (B, NH, DHQK)
    m_initial: torch.Tensor = None,  # (B, NH, 1)
    chunk_size: int = 32,
    return_last_states: bool = True,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-6,
    **kwargs,
):
    _gpu_only_guard(q)
    _ray_started_here = _ensure_ray()

    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]

    # Warm-up a tiny step compile to avoid mid-run overhead
    if os.environ.get("XLSTM_MPS_WARMUP", "1") != "0":
        try:
            _ = mlstm_recurrent_step__metal(
                q[:, :1, 0], k[:, :1, 0], v[:, :1, 0], i[:, :1, 0:1], f[:, :1, 0:1],
                c_initial[:, :1] if c_initial is not None else None,
                n_initial[:, :1] if n_initial is not None else None,
                m_initial[:, :1] if m_initial is not None else None,
                eps=eps, dtype_state=torch.float32,
            )
        except Exception:
            pass

    h_out = torch.empty(B, NH, S, DHHV, device=q.device, dtype=q.dtype)

    # Partition by head bands, then run per-band actors over the full sequence in chunk tiles
    heads_per_band = int(os.environ.get("XLSTM_MPS_HEADS_PER_BAND", "4"))
    bands = []
    for hs in range(0, NH, max(1, heads_per_band)):
        he = min(NH, hs + heads_per_band)
        bands.append((hs, he))

    # Create actors (propagate critical env vars when multi-process)
    actors = []
    # Prepare per-actor runtime env to enforce GPU-only and quiet tokenizers
    _actor_runtime_env = {
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "TOKENIZERS_PARALLELISM",
                "XLSTM_MEM_WATCHDOG",
                "XLSTM_MEM_POLL_MS",
                "XLSTM_MEM_SOFT_PCT",
                "XLSTM_MEM_HARD_PCT",
                "XLSTM_MEM_SOFT_MB",
                "XLSTM_MEM_HARD_MB",
                "XLSTM_MEM_ACTION",
            )
        }
    }
    for hs, he in bands:
        try:
            actor = HeadBandWorker.options(runtime_env=_actor_runtime_env).remote(  # type: ignore[union-attr]
                q, k, v, i, f, hs, he, eps
            )
        except Exception:
            actor = HeadBandWorker.remote(q, k, v, i, f, hs, he, eps)  # type: ignore[union-attr]
        actors.append((hs, he, actor))

    # Initial per-band states
    band_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}
    for hs, he, _ in actors:
        if c_initial is None:
            band_states[(hs, he)] = None
        else:
            band_states[(hs, he)] = (
                c_initial[:, hs:he], n_initial[:, hs:he], m_initial[:, hs:he]
            )

    # Memory watchdog (no runtime chunk-size changes to preserve canonical behavior)
    monitor: MemoryMonitor | None = None
    if os.environ.get("XLSTM_MEM_WATCHDOG", "1") == "1":
        monitor = MemoryMonitor().start()

    # Dispatch incrementally: at most one inflight chunk per actor, reschedule on completion.
    pending: list[tuple[int, int, int, int, object]] = []
    next_start: dict[tuple[int, int], int] = {(hs, he): 0 for hs, he, _ in actors}
    # Seed one task per actor
    CHUNK = max(1, int(chunk_size))
    for hs, he, actor in actors:
        s = next_start[(hs, he)]
        if s < S:
            e = min(s + CHUNK, S)
            C0, N0, M0 = band_states[(hs, he)] if band_states[(hs, he)] is not None else (None, None, None)
            ref = actor.run.remote(C0, N0, M0, s, e)  # type: ignore[attr-defined]
            pending.append((hs, he, s, e, ref))
            band_states[(hs, he)] = None
            next_start[(hs, he)] = e

    # Gather results as they complete; resubmit next chunk for that band
    last_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    try:
        # Extract refs and metas for ray.wait loop
        refs = [ref for *_, ref in pending]
        metas = [(hs, he, s, e) for hs, he, s, e, _ in pending]
        while refs:
            if monitor is not None:
                monitor.check()
            done, not_done = ray.wait(refs, num_returns=1)  # type: ignore[union-attr]
            idx = refs.index(done[0])
            hs, he, s, e = metas[idx]
            Hband, Cb, Nb, Mb = ray.get(done[0])  # type: ignore[union-attr]
            h_out[:, hs:he, s:e] = Hband
            last_states[(hs, he)] = (Cb, Nb, Mb)
            # remove this ref/meta
            refs.pop(idx)
            metas.pop(idx)
            # schedule next chunk for this band if any
            s2 = next_start[(hs, he)]
            if s2 < S:
                if monitor is not None:
                    monitor.check()
                e2 = min(s2 + CHUNK, S)
                C0, N0, M0 = (Cb, Nb, Mb)  # continue from last state
                ref2 = [act for hss, hee, act in actors if hss == hs and hee == he][0].run.remote(C0, N0, M0, s2, e2)  # type: ignore[attr-defined]
                refs.append(ref2)
                metas.append((hs, he, s2, e2))
                next_start[(hs, he)] = e2
    except MemoryPressureAbort as e:
        if monitor is not None:
            monitor.stop()
        # Ensure all actors are terminated to free GPU memory even on abort
        try:
            for _, _, actor in actors:
                try:
                    ray.get(actor.__ray_terminate__.remote())  # type: ignore[attr-defined]
                except Exception:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
        except Exception:
            pass
        raise RuntimeError(f"Aborted due to unified memory pressure: {e}")
    finally:
        # Always terminate actors we created to free GPU memory, regardless of Ray lifecycle
        try:
            for _, _, actor in actors:
                try:
                    ray.get(actor.__ray_terminate__.remote())  # type: ignore[attr-defined]
                except Exception:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
        except Exception:
            pass
        # Auto-shutdown Ray if we started it here (opt-out via env)
        if _ray_started_here and os.environ.get("XLSTM_RAY_AUTOSHUTDOWN", "1") == "1":
            try:
                ray.shutdown()
            except Exception:
                pass

    if return_last_states:
        Cf = torch.empty(B, NH, DHQK, DHHV, device=q.device, dtype=torch.float32)
        Nf = torch.empty(B, NH, DHQK, device=q.device, dtype=torch.float32)
        Mf = torch.empty(B, NH, 1, device=q.device, dtype=torch.float32)
        for hs, he in bands:
            Cb, Nb, Mb = last_states[(hs, he)]
            Cf[:, hs:he] = Cb
            Nf[:, hs:he] = Nb
            Mf[:, hs:he] = Mb
        if monitor is not None:
            monitor.stop()
        return h_out, (Cf, Nf, Mf)
    else:
        if monitor is not None:
            monitor.stop()
        return h_out
