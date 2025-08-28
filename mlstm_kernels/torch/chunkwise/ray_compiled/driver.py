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
            H_out[:, :, t - s_start] = H
        return H_out, C, N, M


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

    # Create actors
    actors = []
    for hs, he in bands:
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

    # Memory watchdog and dynamic chunk-size shrink
    min_chunk = int(os.environ.get("XLSTM_MIN_CHUNK", "8"))
    shrink_on_soft = os.environ.get("XLSTM_SHRINK_ON_SOFT", "1") != "0"
    chunk_ref = [max(min_chunk, int(chunk_size))]
    monitor: MemoryMonitor | None = None
    if os.environ.get("XLSTM_MEM_WATCHDOG", "1") == "1":
        def _on_soft(_st):
            if not shrink_on_soft:
                return
            cur = int(chunk_ref[0])
            if cur > min_chunk:
                new = max(min_chunk, cur // 2)
                if new < cur:
                    print(f"[xLSTM][mem] Shrinking chunk_size {cur} -> {new}", flush=True)
                    chunk_ref[0] = new

        monitor = MemoryMonitor(on_soft=_on_soft).start()

    # Dispatch incrementally: at most one inflight chunk per actor, reschedule on completion.
    pending: list[tuple[int, int, int, int, object]] = []
    next_start: dict[tuple[int, int], int] = {(hs, he): 0 for hs, he, _ in actors}
    # Seed one task per actor
    for hs, he, actor in actors:
        s = next_start[(hs, he)]
        if s < S:
            e = min(s + int(chunk_ref[0]), S)
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
                e2 = min(s2 + int(chunk_ref[0]), S)
                C0, N0, M0 = (Cb, Nb, Mb)  # continue from last state
                ref2 = [act for hss, hee, act in actors if hss == hs and hee == he][0].run.remote(C0, N0, M0, s2, e2)  # type: ignore[attr-defined]
                refs.append(ref2)
                metas.append((hs, he, s2, e2))
                next_start[(hs, he)] = e2
    except MemoryPressureAbort as e:
        if monitor is not None:
            monitor.stop()
        raise RuntimeError(f"Aborted due to unified memory pressure: {e}")
    finally:
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
        # If keeping dashboard alive, terminate actors to free GPU mem
        if os.environ.get("XLSTM_RAY_AUTOSHUTDOWN", "1") != "1":
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
        return h_out, (Cf, Nf, Mf)
    else:
        if monitor is not None:
            monitor.stop()
        if os.environ.get("XLSTM_RAY_AUTOSHUTDOWN", "1") != "1":
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
        return h_out
