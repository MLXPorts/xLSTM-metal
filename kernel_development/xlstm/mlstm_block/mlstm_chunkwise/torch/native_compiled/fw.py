import torch

from kernel_development.xlstm.mlstm_block import mlstm_recurrent_sequence__native_fw

try:
    # Optional import for forcing smaller compiled regions if needed
    from torch._dynamo import graph_break  # type: ignore
except Exception:  # pragma: no cover
    def graph_break():
        """

        :return:
        """
        return None


def _chunk_step_eager(
        q_chunk: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        i_chunk: torch.Tensor,
        f_chunk: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float,
):
    # Single-chunk compiled inner: returns h_seg and updated states
    return mlstm_recurrent_sequence__native_fw(q=q_chunk, k=k_chunk, v=v_chunk, i=i_chunk, f=f_chunk, c_initial=c_state,
                                               n_initial=n_state, m_initial=m_state, return_last_states=True, eps=eps)


try:
    _chunk_step_compiled = torch.compile(_chunk_step_eager, mode="default")
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for chunkwise inner compiled kernel: {e}. No fallback allowed."
    )


def _chunkwise_native_compiled_autograd_eager(
        q: torch.Tensor,  # (B, NH, S, DHQK)
        k: torch.Tensor,  # (B, NH, S, DHQK)
        v: torch.Tensor,  # (B, NH, S, DHHV)
        i: torch.Tensor,  # (B, NH, S)
        f: torch.Tensor,  # (B, NH, S)
        c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
        n_initial: torch.Tensor = None,  # (B, NH, DHQK)
        m_initial: torch.Tensor = None,  # (B, NH, 1)
        chunk_size: int = 64,
        return_last_states: bool = True,
        autocast_kernel_dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        **kwargs,
):
    # Strict device check: compiled path must run on GPU
    if q.device.type == 'cpu':
        raise RuntimeError("mLSTM chunkwise compiled requires GPU (MPS/CUDA); CPU not allowed.")
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    device = q.device

    # Initialize states if needed
    if c_initial is None:
        c_state = torch.zeros(B, NH, DHQK, DHHV, dtype=torch.float32, device=device)
        n_state = torch.zeros(B, NH, DHQK, dtype=torch.float32, device=device)
        m_state = torch.zeros(B, NH, 1, dtype=torch.float32, device=device)
    else:
        c_state, n_state, m_state = c_initial, n_initial, m_initial

    # List-accumulate to avoid many out_ptrN slice stores in a single fused kernel
    h_segments = []
    pos = 0
    while pos < S:
        seg = min(chunk_size, S - pos)
        h_seg, (c_state, n_state, m_state) = _chunk_step_compiled(
            q[:, :, pos:pos + seg, :].contiguous(),
            k[:, :, pos:pos + seg, :].contiguous(),
            v[:, :, pos:pos + seg, :].contiguous(),
            i[:, :, pos:pos + seg].contiguous(),
            f[:, :, pos:pos + seg].contiguous(),
            c_state,
            n_state,
            m_state,
            eps,
        )
        h_segments.append(h_seg)
        # Optional graph break to limit fusion breadth per chunk (kept cheap on eager)
        graph_break()
        pos += seg

    matH = torch.cat(h_segments, dim=2) if len(h_segments) > 1 else h_segments[0]

    if return_last_states:
        return matH, (c_state, n_state, m_state)
    else:
        return matH


# Expose the orchestrator as a regular eager function that calls a compiled inner
mlstm_chunkwise__native_compiled_autograd = _chunkwise_native_compiled_autograd_eager

# Alias for an XL-chunk style variant; functionally identical for now
mlstm_chunkwise__native_compiled_xl = mlstm_chunkwise__native_compiled_autograd
