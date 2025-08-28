import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import torch

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
    while pos < S:
        seg = min(chunk_size, S - pos)
        # step along this chunk
        for t in range(pos, pos + seg):
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
            # write output slice
            h_out_ref[:, hs:he, t] = H
        pos += seg

    return C, N, M


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

    # Output allocation
    h_out = torch.empty(B, NH, S, DHHV, dtype=q.dtype, device=q.device)

    # Threading parameters
    heads_per_band = int(os.environ.get("XLSTM_MPS_HEADS_PER_BAND", "4"))
    num_workers = int(os.environ.get("XLSTM_MPS_WORKERS", "6"))
    heads_per_band = max(1, min(NH, heads_per_band))
    num_workers = max(1, num_workers)

    bands = []
    for hs in range(0, NH, heads_per_band):
        he = min(NH, hs + heads_per_band)
        bands.append((hs, he))

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for hs, he in bands:
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
            )
            results.append((hs, he, fut))

    # Aggregate final states
    if return_last_states:
        Cf = torch.empty(
            (B, NH, DHQK, DHHV), device=q.device, dtype=torch.float32
        )
        Nf = torch.empty((B, NH, DHQK), device=q.device, dtype=torch.float32)
        Mf = torch.empty((B, NH, 1), device=q.device, dtype=torch.float32)
        for hs, he, fut in results:
            C, N, M = fut.result()
            Cf[:, hs:he] = C
            Nf[:, hs:he] = N
            Mf[:, hs:he] = M
        return h_out, (Cf, Nf, Mf)
    else:
        # ensure completion
        for _, _, fut in results:
            _ = fut.result()
        return h_out

