#!/usr/bin/env python
"""
Content-Addressable Memory Cube for MLX.

Per-block associative memory for residual predictions using cosine similarity retrieval.
Ported from src/lnn_hrm/memory_cube.py (PyTorch) to MLX.
"""

import mlx.core as mx
import mlx.nn as nn


class MemoryCubeMLX(nn.Module):
    """Per-block associative memory for residual predictions.

    Stores (keys, values) with cosine similarity retrieval. Values typically encode
    residuals Δy for memory-augmented predictions.

    Args:
        d_key: Dimension of keys
        d_val: Dimension of values
        max_items: Maximum number of (key, value) pairs to store (ring buffer eviction)
        topk: Number of top-k similar keys to retrieve for weighted averaging

    Example:
        >>> cube = MemoryCubeMLX(d_key=512, d_val=512, max_items=1024, topk=8)
        >>> q = mx.random.normal((16, 512))  # Query 16 keys
        >>> pred, conf = cube.query(q)  # Returns predictions and confidence
        >>> # Update with new memories
        >>> k_new = mx.random.normal((16, 512))
        >>> v_new = mx.random.normal((16, 512))
        >>> cube.update(k_new, v_new)
    """

    def __init__(
        self,
        d_key: int,
        d_val: int,
        max_items: int = 65536,
        topk: int = 8
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.max_items = int(max_items)
        self.topk = int(topk)

        # Initialize empty buffers
        # MLX nn.Module uses __dict__ to avoid parameter registration issues
        self.__dict__['keys'] = mx.zeros((0, d_key))
        self.__dict__['vals'] = mx.zeros((0, d_val))

    def query(self, q: mx.array) -> tuple[mx.array, mx.array]:
        """Query keys with q [Q, d_key]; return (pred [Q, d_val], conf [Q]).

        Args:
            q: Query keys of shape (Q, d_key)

        Returns:
            pred: Predicted values (Q, d_val) as weighted average of top-k values
            conf: Confidence scores (Q,) as mean of top-k cosine similarities
        """
        # Handle empty cube
        if self.keys.size == 0:
            pred = mx.zeros((q.shape[0], self.d_val))
            conf = mx.zeros((q.shape[0],))
            return pred, conf

        # Normalize keys and queries for cosine similarity
        k_norm = self.keys / (mx.linalg.norm(self.keys, axis=-1, keepdims=True) + 1e-8)
        q_norm = q / (mx.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

        # Compute cosine similarities: (Q, d_key) @ (d_key, K) -> (Q, K)
        sims = q_norm @ k_norm.T

        # Top-k retrieval
        k_actual = min(self.topk, sims.shape[1])
        topk_indices = mx.argpartition(-sims, kth=k_actual-1, axis=-1)[:, :k_actual]

        # Gather top-k values and scores
        # For each query, gather the top-k indices
        batch_indices = mx.arange(q.shape[0])[:, None]
        topk_sims = sims[batch_indices, topk_indices]  # (Q, k)
        topk_vals = self.vals[topk_indices]  # (Q, k, d_val)

        # Softmax weights over top-k
        weights = mx.softmax(topk_sims, axis=-1)  # (Q, k)

        # Weighted average: (Q, k) @ (Q, k, d_val) -> (Q, d_val)
        pred = mx.einsum("qk,qkd->qd", weights, topk_vals)

        # Confidence: mean of top-k similarities, clamped to [0, 1]
        conf = mx.clip(mx.mean(topk_sims, axis=-1), 0.0, 1.0)

        return pred, conf

    def update(self, k_new: mx.array, v_new: mx.array):
        """Update memory with new (key, value) pairs.

        Uses ring buffer eviction: if total items exceed max_items,
        keeps only the most recent max_items entries.

        Args:
            k_new: New keys of shape (N, d_key)
            v_new: New values of shape (N, d_val)
        """
        # Concatenate with existing
        if self.keys.size > 0:
            keys = mx.concatenate([self.keys, k_new], axis=0)
            vals = mx.concatenate([self.vals, v_new], axis=0)
        else:
            keys = k_new
            vals = v_new

        # Ring buffer eviction: keep most recent max_items
        if keys.shape[0] > self.max_items:
            keys = keys[-self.max_items:]
            vals = vals[-self.max_items:]

        # Update via __dict__ to avoid parameter registration
        self.__dict__['keys'] = keys
        self.__dict__['vals'] = vals

    def __call__(self, q: mx.array) -> tuple[mx.array, mx.array]:
        """Alias for query method to support nn.Module interface."""
        return self.query(q)
