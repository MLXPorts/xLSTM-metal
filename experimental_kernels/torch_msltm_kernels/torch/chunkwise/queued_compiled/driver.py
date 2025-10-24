import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from ...monitoring.memory import MemoryMonitor, MemoryPressureAbort
from ...recurrent.metal.compiled import mlstm_recurrent_step__metal


def _process_head_band(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    head_start: int,
    head_end: int,
    chunk_size: int,
    eps: float,
    h_out_ref: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    stream: "torch.mps.Stream | None" = None,
    chunk_size_ref: list | None = None,
    monitor: MemoryMonitor | None = None,
):
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    hs = head_start
    he = head_end

    # Local states for this head band
    C = c_state[:, hs:he] if c_state is not None else torch.zeros(
        (B, he - hs, DHQK, DHHV), device=q.device, dtype=torch.float32
    )
    N = n_state[:, hs:he] if n_state is not None else torch.zeros(
        (B, he - hs, DHQK), device=q.device, dtype=torch.float32
    )
    M = m_state[:, hs:he] if m_state is not None else torch.zeros(
        (B, he - hs, 1), device=q.device, dtype=torch.float32
    )

    pos = 0
    steps_processed = 0
    t_start = time.perf_counter()
    while pos < S:
        if monitor is not None:
            monitor.check()
        # allow dynamic shrink via shared ref
        cur_chunk = chunk_size_ref[0] if chunk_size_ref is not None else chunk_size
        seg = min(int(cur_chunk), S - pos)
        # step along this chunk
        # Use provided MPS stream to encourage concurrency across bands
        if stream is not None:
            ctx = torch.mps.stream(stream)
        else:
            # no-op context manager
            class _Null:
                def __enter__(self_):
                    return None
                def __exit__(self_, exc_type, exc, tb):
                    return False
            ctx = _Null()
        with ctx:
            # CfC-hybrid experimental override
            cfc_on = os.environ.get("XLSTM_CFC_HYBRID", "0") == "1"
            cfc_dt = float(os.environ.get("XLSTM_CFC_DT", "0.01"))
            cfc_alpha = float(os.environ.get("XLSTM_CFC_ALPHA", "1.0"))
            cfc_act = os.environ.get("XLSTM_CFC_ACT", "sigmoid")
            h_cfc = None
            if cfc_on:
                h_cfc = torch.zeros((B, he - hs, DHHV), device=q.device, dtype=q.dtype)
            for t in range(pos, pos + seg):
                if monitor is not None:
                    monitor.check()
                H, (C, N, M) = mlstm_recurrent_step__metal(
                        q=q[:, hs:he, t],
                        k=k[:, hs:he, t],
                        v=v[:, hs:he, t],
                        i=i[:, hs:he, t : t + 1],
                        f=f[:, hs:he, t : t + 1],
                        c=C,
                        n=N,
                        m=M,
                        eps=eps,
                        dtype_state=torch.float32,
                    )
                if cfc_on:
                    ff = torch.sigmoid(H) if cfc_act == "sigmoid" else (1.7159 * torch.tanh(0.666 * H))
                    try:
                        i_t = i[:, hs:he, t : t + 1]
                        f_t = f[:, hs:he, t : t + 1]
                        lam = torch.sigmoid(cfc_alpha * (i_t + f_t)).squeeze(-1)
                    except Exception:
                        lam = torch.zeros_like(h_cfc)
                    denom = 1.0 + cfc_dt * lam
                    h_cfc = (h_cfc + cfc_dt * ff) / denom
                    H = h_cfc
                # write output slice
                h_out_ref[:, hs:he, t] = H
                steps_processed += 1
        pos += seg
    t_total = time.perf_counter() - t_start
    return C, N, M, steps_processed, t_total


def mlstm_chunkwise__queued_compiled_steps(
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
    """
    GPU-only queued chunkwise prefill using compiled step kernels.
    - Splits heads into bands and sequences into small chunks.
    - Uses a CPU thread pool to enqueue many small compiled step kernels.
    - All math runs on GPU (MPS); CPU threads only coordinate.
    """
    assert q.device.type == "mps", "Queued compiled chunkwise requires MPS device"
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]

    # Optional warm-up to compile tiny step kernel variants (one-time)
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

    # Output allocation
    h_out = torch.empty(B, NH, S, DHHV, dtype=q.dtype, device=q.device)

    # Threading parameters
    heads_per_band = int(os.environ.get("XLSTM_MPS_HEADS_PER_BAND", "4"))
    num_workers = int(os.environ.get("XLSTM_MPS_WORKERS", "6"))
    num_streams = int(os.environ.get("XLSTM_MPS_STREAMS", str(num_workers)))
    autoscale = os.environ.get("XLSTM_MPS_AUTOSCALE", "0") == "1"
    heads_per_band = max(1, min(NH, heads_per_band))
    num_workers = max(1, num_workers)

    # Memory watchdog (no runtime chunk-size changes to preserve canonical behavior)
    monitor: MemoryMonitor | None = None
    if os.environ.get("XLSTM_MEM_WATCHDOG", "1") == "1":
        monitor = MemoryMonitor().start()

    # Optional micro autoscale: adjust heads_per_band based on a small probe
    if autoscale and NH >= 4:
        probe_hpb = max(1, min(heads_per_band, NH))
        t0 = time.perf_counter()
        _ = _process_head_band(
            q[:, :probe_hpb, : min(8, S)],
            k[:, :probe_hpb, : min(8, S)],
            v[:, :probe_hpb, : min(8, S)],
            i[:, :probe_hpb, : min(8, S)],
            f[:, :probe_hpb, : min(8, S)],
            0,
            probe_hpb,
            min(8, chunk_size),
            eps,
            torch.empty(B, probe_hpb, min(8, S), DHHV, device=q.device, dtype=q.dtype),
            c_initial[:, :probe_hpb] if c_initial is not None else None,
            n_initial[:, :probe_hpb] if n_initial is not None else None,
            m_initial[:, :probe_hpb] if m_initial is not None else None,
            None,
        )
        t_probe = time.perf_counter() - t0
        # Heuristic: if probe is slow, reduce heads per band to increase parallelism
        if t_probe > 0.010 and heads_per_band > 2:
            heads_per_band = max(2, heads_per_band // 2)

    bands = []
    for hs in range(0, NH, heads_per_band):
        he = min(NH, hs + heads_per_band)
        bands.append((hs, he))

    # Create dedicated MPS streams to enable true concurrency
    # Create streams if supported by this torch build; otherwise fall back to None
    streams: list = []
    try:
        _Stream = getattr(torch.mps, "Stream", None)
        if _Stream is not None:
            streams = [_Stream() for _ in range(max(1, num_streams))]
    except Exception:
        streams = []
    results = []
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for hs, he in bands:
                stream = streams[(hs // heads_per_band) % len(streams)] if len(streams) > 0 else None
                fut = ex.submit(
                    _process_head_band,
                    q,
                    k,
                    v,
                    i,
                    f,
                    hs,
                    he,
                    chunk_size,
                    eps,
                    h_out,
                    c_initial,
                    n_initial,
                    m_initial,
                    stream,
                    chunk_ref,
                    monitor,
                )
                results.append((hs, he, fut))
    except MemoryPressureAbort as e:
        # Propagate a cleaner message
        if monitor is not None:
            monitor.stop()
        raise RuntimeError(f"Aborted due to unified memory pressure: {e}")

    # Aggregate final states
    if return_last_states:
        Cf = torch.empty(
            (B, NH, DHQK, DHHV), device=q.device, dtype=torch.float32
        )
        Nf = torch.empty((B, NH, DHQK), device=q.device, dtype=torch.float32)
        Mf = torch.empty((B, NH, 1), device=q.device, dtype=torch.float32)
        total_steps = 0
        total_time = 0.0
        for hs, he, fut in results:
            C, N, M, steps, t_time = fut.result()
            Cf[:, hs:he] = C
            Nf[:, hs:he] = N
            Mf[:, hs:he] = M
            total_steps += steps
            total_time += t_time
        # Attach simple telemetry for callers (attrs on tensor)
        try:
            h_out._mps_steps = total_steps  # type: ignore[attr-defined]
            h_out._mps_time = total_time    # type: ignore[attr-defined]
        except Exception:
            pass
        if monitor is not None:
            monitor.stop()
        return h_out, (Cf, Nf, Mf)
    else:
        # ensure completion
        for _, _, fut in results:
            _ = fut.result()
        if monitor is not None:
            monitor.stop()
        return h_out
