#!/usr/bin/env python
"""
HRM-Enhanced xLSTM Block for MLX.

Wraps a standard xLSTM block with HRM+ memory augmentation including:
- Cube-gated memory predictions
- Z5 boundary-based memory commits
- ACT halting telemetry
- Optional neuromodulation

This enables the two-timescale HRM+ architecture:
- Fast (mLSTM): Parallel processing within chunks
- Slow (Cube): Memory updates at temporal boundaries
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from dataclasses import dataclass

from ..mlstm_mlx.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .cube_gated import CubeGatedBlockMLX
from .act_halting import ACTHaltingHeadMLX
from .scheduler import boundary_commit_mask


@dataclass
class HRMxLSTMConfig:
    """Configuration for HRM-enhanced xLSTM block.

    Args:
        xlstm_config: Configuration for base xLSTM block
        enable_hrm: Whether to enable HRM memory augmentation
        enable_act: Whether to enable ACT halting telemetry
        fuse_phase_keys: Whether to fuse temporal phase encodings
        cube_max_items: Maximum items in memory cube
        cube_topk: Number of top-k retrievals
        k_5ht: Serotonin modulation strength
        gain_floor: Minimum gain under neuromodulation
        act_threshold: ACT halting threshold
    """
    xlstm_config: xLSTMBlockConfig
    enable_hrm: bool = True
    enable_act: bool = False
    fuse_phase_keys: bool = True
    cube_max_items: int = 65536
    cube_topk: int = 8
    k_5ht: float = 0.5
    gain_floor: float = 0.3
    act_threshold: float = 0.5


class HRMxLSTMBlockMLX(nn.Module):
    """HRM-enhanced xLSTM block combining mLSTM with memory augmentation.

    Forward pass:
    1. Apply standard xLSTM block (norm -> mLSTM -> FFN with residuals)
    2. If HRM enabled: Apply cube-gated memory blending
    3. If ACT enabled: Compute halting probabilities
    4. Return output with telemetry

    Args:
        config: HRM-enhanced xLSTM configuration

    Example:
        >>> from ..mlstm_mlx.xlstm_block import xLSTMBlockConfig
        >>> xlstm_cfg = xLSTMBlockConfig(embedding_dim=512, num_heads=8)
        >>> hrm_cfg = HRMxLSTMConfig(xlstm_config=xlstm_cfg, enable_hrm=True, enable_act=True)
        >>> block = HRMxLSTMBlockMLX(hrm_cfg)
        >>>
        >>> x = mx.random.normal((2, 10, 512))  # (B, L, D)
        >>> times = mx.arange(10).reshape(1, -1).broadcast_to((2, 10))
        >>> output, state, telemetry = block(x, times=times)
        >>> print(f"Alpha: {telemetry['alpha_mean']:.3f}, Conf: {telemetry['conf_mean']:.3f}")
    """

    def __init__(self, config: HRMxLSTMConfig):
        super().__init__()
        self.config = config

        # Base xLSTM block
        self.xlstm_block = xLSTMBlock(config.xlstm_config)

        # HRM cube-gated block
        if config.enable_hrm:
            self.cube_gate = CubeGatedBlockMLX(
                d_in=config.xlstm_config.embedding_dim,
                fuse_phase_keys=config.fuse_phase_keys,
                k_5ht=config.k_5ht,
                gain_floor=config.gain_floor,
                max_items=config.cube_max_items,
                topk=config.cube_topk
            )

        # ACT halting head
        if config.enable_act:
            self.act_head = ACTHaltingHeadMLX(
                d_model=config.xlstm_config.embedding_dim,
                threshold=config.act_threshold
            )

    def __call__(
        self,
        x: mx.array,
        state: Optional[tuple] = None,
        times: Optional[mx.array] = None,
        mod_5ht: Optional[mx.array] = None,
        train: bool = False
    ) -> tuple[mx.array, Optional[tuple], dict]:
        """Forward pass through HRM-enhanced xLSTM block.

        Args:
            x: Input of shape (B, L, D)
            state: Optional mLSTM hidden state
            times: Optional time steps (B, L) for Z5 scheduler and phase keys
            mod_5ht: Optional serotonin modulation (B, L) or (B, L, 1)
            train: Whether in training mode (enables cube updates)

        Returns:
            output: Block output (B, L, D)
            new_state: Updated mLSTM hidden state
            telemetry: Dictionary with HRM and ACT metrics
        """
        telemetry = {}

        # 1. Apply base xLSTM block (mLSTM + FFN) - this computes the teacher signal
        xlstm_out, new_state = self.xlstm_block(x, state)

        # 2. Apply HRM cube gating if enabled
        if self.config.enable_hrm:
            # Compute Z5 boundary commit mask if times provided
            allow_commit = None
            if times is not None:
                allow_commit = boundary_commit_mask(times)

            # Apply cube-gated blending
            # Per-block architecture: cube learns the residual transformation that xLSTM applies
            # Teacher y_T = xlstm_block(x), stored residual Δy = y_T - x
            # Cube gate receives original input x, blends with memory-augmented prediction
            cube_state = {
                'times': times,
                'mod_5ht': mod_5ht,
                'train': train,
                'y_teacher': xlstm_out if train else None,  # Teacher is xLSTM output
                'allow_commit': allow_commit  # Z5 boundary commit mask
            }
            hrm_out, hrm_telemetry = self.cube_gate(x, cube_state)

            output = hrm_out
            # Collect telemetry from cube gate
            telemetry.update(hrm_telemetry)
        else:
            output = xlstm_out

        # 3. Apply ACT halting telemetry if enabled
        if self.config.enable_act:
            halt_probs, halt_mask, act_stats = self.act_head(output)
            telemetry.update(act_stats)
            telemetry['halt_probs'] = halt_probs
            telemetry['halt_mask'] = halt_mask

        return output, new_state, telemetry
