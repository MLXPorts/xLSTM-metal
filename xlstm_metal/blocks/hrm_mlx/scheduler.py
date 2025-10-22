#!/usr/bin/env python
"""
Z5 Scheduler for temporal discretization.

Base-5 carry policy for boundary-commit memory updates.
Ported from src/lnn_hrm/scheduler.py (PyTorch) to MLX.
"""

import mlx.core as mx


def z5_slots(times: mx.array) -> mx.array:
    """Return Z5 envelope slot indices in {0,1,2,3,4} for given time steps.

    Args:
        times: (B, L) array of monotonically increasing time steps

    Returns:
        Slot indices (B, L) in range [0, 4]

    Example:
        >>> times = mx.array([[0, 1, 2, 3, 4, 5, 6]])
        >>> z5_slots(times)
        array([[0, 1, 2, 3, 4, 0, 1]], dtype=int64)
    """
    return times.astype(mx.int64) % 5


def boundary_commit_mask(times: mx.array) -> mx.array:
    """Return a boolean mask (B, L) that is True on boundary commit steps.

    We define slot 4 → 0 rollover as the control carry. For simple sequences,
    we mark steps whose slot equals 4 as the commit boundary.

    Args:
        times: (B, L) array of monotonically increasing time steps

    Returns:
        Boolean mask (B, L) where True indicates commit boundary (slot == 4)

    Example:
        >>> times = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        >>> boundary_commit_mask(times)
        array([[False, False, False, False, True, False, False, False, False, True]])
    """
    slots = z5_slots(times)
    return slots == 4
