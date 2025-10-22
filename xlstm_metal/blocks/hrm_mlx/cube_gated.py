#!/usr/bin/env python
"""
Cube-Gated Block for Memory-Augmented Predictions (MLX).

Blends memory cube predictions with current activations using learned confidence gates.
Ported from src/lnn_hrm/cube_gated_block.py (PyTorch) to MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from .memory_cube import MemoryCubeMLX


class CubeGatedBlockMLX(nn.Module):
    """Attach a memory cube and a gate to a block boundary.

    Given input h_in [B,L,D], computes key projections, queries the cube for residual
    predictions, computes a confidence-weighted gate α, and blends with a teacher
    output if provided.

    Args:
        d_in: Input dimension
        d_key: Key dimension for memory cube (defaults to d_in)
        d_val: Value dimension for memory cube (defaults to d_in)
        fuse_phase_keys: Whether to fuse temporal phase encodings into keys
        k_5ht: Serotonin modulation strength (default: 0.5)
        gain_floor: Minimum gain under neuromodulation (default: 0.3)
        max_items: Maximum items in memory cube
        topk: Number of top-k retrievals

    Example:
        >>> block = CubeGatedBlockMLX(d_in=512, fuse_phase_keys=True)
        >>> h_in = mx.random.normal((2, 10, 512))  # (B, L, D)
        >>> times = mx.arange(10).reshape(1, -1).broadcast_to((2, 10))
        >>> y_out, alpha_mean, conf_mean = block(h_in, times=times)
    """

    def __init__(
        self,
        d_in: int,
        d_key: int | None = None,
        d_val: int | None = None,
        fuse_phase_keys: bool = True,
        k_5ht: float = 0.5,
        gain_floor: float = 0.3,
        max_items: int = 65536,
        topk: int = 8
    ):
        super().__init__()
        d_key = d_key or d_in
        d_val = d_val or d_in
        self.d_in = d_in
        self.d_key = d_key
        self.d_val = d_val
        self.fuse_phase_keys = fuse_phase_keys
        self.k_5ht = float(k_5ht)
        self.gain_floor = float(gain_floor)

        # Key projection
        self.key_proj = nn.Linear(d_in, d_key)

        # Optional phase fusion: map [key || phase] -> key_dim
        if fuse_phase_keys:
            self.phase_proj = nn.Linear(d_key + 8, d_key)

        # Alpha gate head: takes [h_in || pred || conf] -> gate value
        self.alpha_fc1 = nn.Linear(d_in + d_val + 1, d_in)
        self.alpha_fc2 = nn.Linear(d_in, 1)

        # Memory cube
        self.cube = MemoryCubeMLX(
            d_key=d_key,
            d_val=d_val,
            max_items=max_items,
            topk=topk
        )

        # Layer norms for stability
        self.ln_in = nn.LayerNorm(d_in)
        self.ln_pred = nn.LayerNorm(d_val)

    def __call__(
        self,
        h_in: mx.array,
        state: dict | None = None
    ) -> tuple[mx.array, dict]:
        """Forward pass through cube-gated block (MAD interface).

        Args:
            h_in: Input activations (B, L, D)
            state: Optional state dict with:
                - 'times': Time steps (B, L) for phase-key fusion and Z5 scheduler
                - 'mod_5ht': Serotonin modulation (B, L) or (B, L, 1)
                - 'train': Whether in training mode (enables cube updates)
                - 'y_teacher': Teacher output for training (B, L, D)

        Returns:
            y_out: Gated output (B, L, D)
            new_state: Dict with telemetry {'alpha_mean', 'conf_mean', 'energy_pre', 'energy_post'}
        """
        # Extract from state dict
        if state is None:
            state = {}

        times = state.get('times', None)
        mod_5ht = state.get('mod_5ht', None)
        train = state.get('train', False)
        y_teacher = state.get('y_teacher', None)
        allow_commit = None
        B, L, D = h_in.shape

        # Compute neuromodulation gain if provided
        gain = None
        if mod_5ht is not None:
            g = mod_5ht
            if g.ndim == 2:
                g = mx.expand_dims(g, -1)
            # Divisive gain: higher 5-HT -> lower gain
            gain = mx.clip(mx.exp(-self.k_5ht * g), self.gain_floor, 1.0)

        # Project to keys
        keys = self.key_proj(h_in)  # (B, L, d_key)

        # Fuse phase keys if enabled
        if self.fuse_phase_keys and times is not None:
            assert times.shape[:2] == h_in.shape[:2], "times must be (B,L) aligning with h_in"

            t = times.astype(mx.float32)

            # Three cos/sin phases with periods 1, 3, 9
            phase_1 = mx.cos(2 * mx.pi * t / 1.0)
            phase_3 = mx.cos(2 * mx.pi * t / 3.0)
            phase_9 = mx.cos(2 * mx.pi * t / 9.0)
            sin_1 = mx.sin(2 * mx.pi * t / 1.0)
            sin_3 = mx.sin(2 * mx.pi * t / 3.0)
            sin_9 = mx.sin(2 * mx.pi * t / 9.0)

            phi = mx.stack([phase_1, phase_3, phase_9, sin_1, sin_3, sin_9], axis=-1)  # (B, L, 6)

            # Z5 slot one-hot (5 classes)
            slots = (t.astype(mx.int32) % 5)
            # Create one-hot encoding manually
            z5 = (mx.expand_dims(slots, -1) == mx.arange(5)).astype(mx.float32)

            # Concatenate phase features: (B, L, 6+5=11)
            phase_full = mx.concatenate([phi, z5], axis=-1)

            # Select first 8 features and apply tanh for stability
            phase_feats = mx.tanh(phase_full[..., :8])  # (B, L, 8)

            # Concatenate with keys and project
            k_cat = mx.concatenate([keys, phase_feats], axis=-1)  # (B, L, d_key+8)
            keys = self.phase_proj(k_cat)  # (B, L, d_key)

        # Reshape for cube query: (B*L, d_key)
        keys_flat = keys.reshape(B * L, -1)

        # Query memory cube
        pred, conf = self.cube.query(keys_flat)

        # Reshape back: (B, L, d_val) and (B, L)
        pred = pred.reshape(B, L, -1)
        conf = conf.reshape(B, L, 1)

        # Compute gate α from [normalized_input || normalized_pred || conf]
        feats = mx.concatenate([self.ln_in(h_in), self.ln_pred(pred), conf], axis=-1)
        alpha_hidden = nn.silu(self.alpha_fc1(feats))
        alpha = mx.sigmoid(self.alpha_fc2(alpha_hidden))  # (B, L, 1)
        alpha = mx.clip(alpha, 0.0, 1.0)

        # Apply neuromodulation if present
        if gain is not None:
            pred = pred * gain
            alpha = alpha * gain

        # Compute residual-augmented output
        y_resid = h_in + pred

        # Blend with teacher or direct input
        if y_teacher is None:
            y_out = (1 - alpha) * h_in + alpha * y_resid
        else:
            y_out = (1 - alpha) * y_teacher + alpha * y_resid

        # Update cube if in training mode
        if train and y_teacher is not None:
            # Compute delta (residuals to store)
            delta = (y_teacher - h_in).reshape(B * L, -1)

            if allow_commit is not None:
                assert allow_commit.shape[:2] == h_in.shape[:2], "allow_commit must be (B,L)"
                # Only update for allowed positions (Z5 boundary commits)
                mask = allow_commit.reshape(B * L)
                if mx.sum(mask.astype(mx.float32)) > 0:
                    # Filter keys and deltas by mask
                    keys_masked = keys_flat[mask]
                    delta_masked = delta[mask]
                    self.cube.update(keys_masked, delta_masked)
            else:
                # Update all positions
                self.cube.update(keys_flat, delta)

        # Return output and telemetry in state dict (MAD interface)
        energy_pre = float(mx.mean(mx.sum(h_in ** 2, axis=-1)).item())
        energy_post = float(mx.mean(mx.sum(y_out ** 2, axis=-1)).item())

        new_state = {
            'alpha_mean': float(mx.mean(alpha).item()),
            'conf_mean': float(mx.mean(conf).item()),
            'energy_pre_gate': energy_pre,
            'energy_post_gate': energy_post
        }

        return y_out, new_state
