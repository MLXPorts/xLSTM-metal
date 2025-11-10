"""
sLSTM Kernel - MLX Implementation

Implements the scalar LSTM (sLSTM) recurrence with exponential gating.
Based on xLSTM paper (https://arxiv.org/pdf/2405.04517) Appendix A, equations 3-9.

Key differences from mLSTM:
- Scalar cell state c_t (vector, not matrix)
- Scalar normalizer state n_t (vector, not matrix)
- True recurrence with block-diagonal R matrices
- Exponential gating: i_t = exp(ĩ_t), f_t = exp(f̃_t)
"""

from typing import Tuple, Optional

import mlx.core as mx


def slstm_recurrent_step(
        z: mx.array,
        i_preact: mx.array,
        f_preact: mx.array,
        o_preact: mx.array,
        c_state: mx.array,
        n_state: mx.array,
        m_state: mx.array,
        eps: float = 1e-6
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Single-step sLSTM recurrence with exponential gating.

    Implements the canonical sLSTM equations from xLSTM paper Appendix A:
    - c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t     (cell state, eq 3)
    - n_t = f_t ⊙ n_{t-1} + i_t            (normalizer state, eq 4)
    - h̃_t = c_t ⊙ n_t^{-1}                (normalized hidden, eq 5)
    - h_t = o_t ⊙ h̃_t                     (output gated hidden, eq 5)
    - i_t = exp(ĩ_t - m_t)                (input gate, eq 7)
    - f_t = exp(f̃_t + m_{t-1} - m_t)      (forget gate, eq 8)
    - o_t = σ(õ_t)                        (output gate, eq 9)
    - m_t = max(f̃_t + m_{t-1}, ĩ_t)       (stabilizer, eq 15)

    Args:
        z: Cell input (post-activation) [B, NH, H]
        i_preact: Input gate pre-activation [B, NH] (already soft-capped)
        f_preact: Forget gate pre-activation [B, NH] (already soft-capped)
        o_preact: Output gate pre-activation [B, NH] (already soft-capped)
        c_state: Cell state [B, NH, H]
        n_state: Normalizer state [B, NH, H]
        m_state: Stabilizer (running max) [B, NH]
        eps: Small constant for numerical stability

    Returns:
        h: Hidden state [B, NH, H]
        c_new: Updated cell state [B, NH, H]
        n_new: Updated normalizer [B, NH, H]
        m_new: Updated stabilizer [B, NH]
    """
    B, NH, H = z.shape

    # Compute stabilizer: m_t = max(f̃_t + m_{t-1}, ĩ_t)  (eq 15 from paper)
    # This ensures numerical stability by keeping exponentials bounded
    m_new = mx.maximum(f_preact + m_state, i_preact)  # [B, NH]

    # Normalized exponential gates (stabilized by m_t)
    # i_t = exp(ĩ_t - m_t)  (eq 16)
    # f_t = exp(f̃_t + m_{t-1} - m_t)  (eq 17)
    i_gate = mx.exp(i_preact - m_new)  # [B, NH]
    f_gate = mx.exp(f_preact + m_state - m_new)  # [B, NH]

    # Output gate: o_t = σ(õ_t)  (eq 14)
    o_gate = mx.sigmoid(o_preact)  # [B, NH]

    # Expand gates for broadcasting with head dimension
    i_expanded = i_gate[:, :, None]  # [B, NH, 1]
    f_expanded = f_gate[:, :, None]  # [B, NH, 1]
    o_expanded = o_gate[:, :, None]  # [B, NH, 1]

    # Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t  (eq 8)
    c_new = f_expanded * c_state + i_expanded * z  # [B, NH, H]

    # Update normalizer: n_t = f_t ⊙ n_{t-1} + i_t  (eq 9)
    n_new = f_expanded * n_state + i_expanded  # [B, NH, H]

    # Compute normalized hidden state: h̃_t = c_t / (n_t + eps)  (eq 10)
    h_tilde = c_new / (n_new + eps)  # [B, NH, H]

    # Apply output gate: h_t = o_t ⊙ h̃_t  (eq 10)
    h = o_expanded * h_tilde  # [B, NH, H]

    return h, c_new, n_new, m_new


def slstm_sequential(
    z: mx.array,
    i_preact: mx.array,
    f_preact: mx.array,
    o_preact: mx.array,
    c_initial: Optional[mx.array] = None,
    n_initial: Optional[mx.array] = None,
    m_initial: Optional[mx.array] = None,
    eps: float = 1e-6,
    return_last_states: bool = True
) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
    """
    Sequential sLSTM processing for inference.

    Processes sequence step-by-step using recurrent updates.

    Args:
        z: Cell inputs [B, NH, S, H]
        i_preact: Input gate pre-activations [B, NH, S]
        f_preact: Forget gate pre-activations [B, NH, S]
        o_preact: Output gate pre-activations [B, NH, S]
        c_initial: Initial cell state [B, NH, H] or None
        n_initial: Initial normalizer [B, NH, H] or None
        m_initial: Initial stabilizer [B, NH] or None
        eps: Numerical stability constant
        return_last_states: Whether to return final states

    Returns:
        h: Hidden states [B, NH, S, H]
        state: Tuple (c_final, n_final, m_final) if return_last_states else None
    """
    B, NH, S, H = z.shape

    # Initialize states
    if c_initial is None:
        c_state = mx.zeros((B, NH, H), dtype=z.dtype)
    else:
        c_state = c_initial

    if n_initial is None:
        n_state = mx.zeros((B, NH, H), dtype=z.dtype)
    else:
        n_state = n_initial

    if m_initial is None:
        m_state = mx.zeros((B, NH), dtype=z.dtype)
    else:
        m_state = m_initial

    # Process sequence step-by-step
    h_list = []
    for t in range(S):
        z_t = z[:, :, t, :]  # [B, NH, H]
        i_t = i_preact[:, :, t]  # [B, NH]
        f_t = f_preact[:, :, t]  # [B, NH]
        o_t = o_preact[:, :, t]  # [B, NH]

        h_t, c_state, n_state, m_state = slstm_recurrent_step(
            z_t, i_t, f_t, o_t,
            c_state, n_state, m_state,
            eps=eps
        )

        h_list.append(h_t)

    # Stack outputs
    h = mx.stack(h_list, axis=2)  # [B, NH, S, H]

    if return_last_states:
        return h, (c_state, n_state, m_state)
    else:
        return h, None
