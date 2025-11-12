"""Canonical xLSTM reference utilities used for parity tests.

These functions mirror the math described in the Hugging Face xLSTM
implementation (see quarantine/xlstm/modeling_xlstm.py). They are kept
simple and framework-agnostic so we can feed numpy arrays captured from
MLX runs and verify numerical agreement component-by-component.
"""

from __future__ import annotations

import numpy as np


def _logsigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def canonical_soft_cap(values: np.ndarray, cap: float | None) -> np.ndarray:
    """Apply the canonical soft-cap function (cap * tanh(x / cap))."""
    if cap is None:
        return values
    cap = float(cap)
    return cap * np.tanh(values / cap)


def canonical_rmsnorm(values: np.ndarray, eps: float, weight: np.ndarray | None = None) -> np.ndarray:
    """Reference RMSNorm with optional learned weight."""
    eps = float(eps)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    normalized = values / np.sqrt(variance + eps)
    if weight is not None:
        normalized = normalized * weight
    return normalized


def canonical_multihead_rmsnorm(
    values: np.ndarray,
    eps: float,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Per-head RMSNorm followed by flattening.

    Args:
        values: Array shaped [B, S, NH, DH].
        eps: Numerical stability epsilon.
    Returns:
        Flattened array [B, S, NH*DH] after per-head RMS normalization.
    """
    eps = float(eps)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    normalized = values / np.sqrt(variance + eps)
    B, S, NH, DH = normalized.shape
    flattened = normalized.reshape(B, S, NH * DH)
    if weight is not None:
        flattened = flattened * weight
    return flattened


def canonical_mlstm_recurrent_sequence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    i_preact: np.ndarray,
    f_preact: np.ndarray,
    eps: float,
    state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reference mLSTM recurrent kernel (sequential) following canonical formulas."""

    B, NH, S, DHQK = q.shape
    DHV = v.shape[-1]

    if state is None:
        c_state = np.zeros((B, NH, DHQK, DHV), dtype=np.float32)
        n_state = np.zeros((B, NH, DHQK), dtype=np.float32)
        m_state = np.zeros((B, NH, 1), dtype=np.float32)
    else:
        c_state, n_state, m_state = state

    h_steps = []
    scale = 1.0 / np.sqrt(DHQK)

    for t in range(S):
        igate = i_preact[:, :, t][:, :, None]
        fgate = f_preact[:, :, t][:, :, None]
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]

        f_log = _logsigmoid(fgate)
        m_new = np.maximum(f_log + m_state, igate)
        f_act = np.exp(f_log + m_state - m_new)
        i_act = np.exp(igate - m_new)

        kv_outer = k_t[:, :, :, None] * v_t[:, :, None, :]
        f_expand = f_act.reshape(B, NH, 1, 1)
        i_expand = i_act.reshape(B, NH, 1, 1)
        c_state = f_expand * c_state + i_expand * kv_outer
        n_state = f_act.reshape(B, NH, 1) * n_state + i_act.reshape(B, NH, 1) * k_t
        m_state = m_new

        q_scaled = q_t * scale
        h_num = np.einsum('bhd,bhdv->bhv', q_scaled, c_state)
        qn_dot = np.einsum('bhd,bhd->bh', q_scaled, n_state)
        max_val = np.exp(-m_state.squeeze(-1))
        h_denom = np.maximum(np.abs(qn_dot), max_val) + eps
        h_t = h_num / h_denom[:, :, None]
        h_steps.append(h_t)

    h = np.stack(h_steps, axis=2)
    return h, (c_state, n_state, m_state)
