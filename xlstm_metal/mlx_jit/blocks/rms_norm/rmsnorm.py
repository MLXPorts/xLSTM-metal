"""Metal-accelerated RMSNorm modules."""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_RMS_TEMPLATE = r"""
    using scalar_t = @SCALAR@;
    using accum_t = @ACCUM@;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    uint rows = params[0];
    uint cols = params[1];
    uint tg_size = params[2];

    if (row >= rows) {
        return;
    }

    uint base_idx = row * cols;

    threadgroup accum_t partial_sums[256];
    accum_t local_sum = accum_t(0.0);
    for (uint idx = tid; idx < cols; idx += tg_size) {
        accum_t val = accum_t(inp[base_idx + idx]);
        local_sum += val * val;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        accum_t total = accum_t(0.0);
        for (uint i = 0; i < tg_size; ++i) {
            total += partial_sums[i];
        }
        partial_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    accum_t sum = partial_sums[0];
    accum_t mean = sum / accum_t(cols);
    accum_t eps = accum_t(eps_param[0]);
    accum_t rms_inv = accum_t(rsqrt((float)(mean + eps)));

    for (uint idx = tid; idx < cols; idx += tg_size) {
        accum_t val = accum_t(inp[base_idx + idx]) * rms_inv;
        val = val * accum_t(weight[idx]);
        out[base_idx + idx] = scalar_t(val);
    }
"""


class RMSNormMetalKernel(nn.Module):
    """Shared Metal kernel for RMSNorm."""

    def __init__(self) -> None:
        super().__init__()
        # cache keyed by (dtype, force_float32)
        self._kernels: dict[tuple[mx.Dtype, bool], Optional[mx.fast.metal_kernel]] = {}

    def build(self, dtype: mx.Dtype, force_float32: bool) -> mx.fast.metal_kernel:
        key = (dtype, force_float32)
        if key in self._kernels and self._kernels[key] is not None:
            return self._kernels[key]

        if dtype == mx.float32:
            scalar = "float"
        elif dtype == mx.bfloat16:
            scalar = "bfloat"
        else:
            raise TypeError(f"RMSNorm kernel only supports float32/bfloat16, got {dtype}")

        accum = "float" if force_float32 else scalar
        source = _RMS_TEMPLATE.replace("@SCALAR@", scalar).replace("@ACCUM@", accum)

        dtype_tag = 'fp32' if dtype == mx.float32 else 'bf16'
        kernel = mx.fast.metal_kernel(
            name=f"rms_norm_{dtype_tag}_{int(force_float32)}",
            input_names=["inp", "weight", "eps_param", "params"],
            output_names=["out"],
            header=_HEADER,
            source=source,
        )
        self._kernels[key] = kernel
        return kernel

    def __call__(self, dtype: mx.Dtype, force_float32: bool) -> mx.fast.metal_kernel:
        return self.build(dtype, force_float32)

    def apply(
            self,
            inputs_2d: mx.array,
            weight: mx.array,
            eps: mx.array,
            force_float32: bool,
    ) -> mx.array:
        rows, cols = inputs_2d.shape
        tg = min(256, cols if cols > 0 else 1)
        params = mx.array([rows, cols, tg], dtype=mx.uint32)
        kernel = self(inputs_2d.dtype, force_float32)
        threadgroup = (tg, 1, 1)
        (out,) = kernel(
            inputs=[inputs_2d, weight, eps, params],
            output_shapes=[inputs_2d.shape],
            output_dtypes=[inputs_2d.dtype],
            grid=(rows, 1, 1),
            threadgroup=threadgroup,
        )
        return out


class RMSNormCell(nn.Module):
    """NCPS-style RMSNorm using the Metal kernel."""

    def __init__(
            self,
            dims: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            force_float32_reductions: bool = True,
            kernel: Optional[RMSNormMetalKernel] = None,
            debug_compare: Optional[bool] = None,
            param_dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.force_float32 = force_float32_reductions
        self.use_weight = use_weight
        dtype = mx.float32 if self.force_float32 else param_dtype
        self._eps = mx.array([eps], dtype=dtype)
        self.kernel = kernel or RMSNormMetalKernel()
        if use_weight:
            self.weight = mx.ones((dims,), dtype=dtype)
        env_flag = os.environ.get("XLSTM_DEBUG_RMSNORM")
        self.debug_compare = debug_compare if debug_compare is not None else bool(int(env_flag)) if env_flag else False

    def build(self, dtype: mx.Dtype) -> None:
        self.kernel.build(dtype, self.force_float32)

    def precompile(self, dtypes: tuple[mx.Dtype, ...]) -> None:
        for dtype in dtypes:
            self.build(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        last_dim = orig_shape[-1]
        x_arr = mx.array(x, dtype=mx.float32 if self.force_float32 else x.dtype)
        rows = int(x_arr.size // last_dim)
        x_2d = x_arr.reshape(rows, last_dim)
        if self.use_weight:
            weight = mx.array(self.weight, dtype=x_arr.dtype)
        else:
            weight = mx.ones((last_dim,), dtype=x_arr.dtype)
        eps = mx.array(self._eps, dtype=x_arr.dtype)
        out_2d = self.kernel.apply(x_2d, weight, eps, force_float32=self.force_float32)
        out = out_2d.reshape(orig_shape).astype(x.dtype)

        if mx.any(mx.isnan(out)).item():
            raise ValueError(
                "RMSNormCell produced NaNs; consider enabling norm_reduction_force_float32"
            )

        if self.debug_compare:
            self._compare_with_torch(x, out)

        return out

    def _compare_with_torch(self, x: mx.array, out: mx.array) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return

        x_np = x.to_numpy()
        w_np = self.weight.to_numpy() if self.use_weight else None
        x_t = torch.from_numpy(x_np).to(torch.float32)
        ref = x_t * torch.rsqrt(torch.mean(x_t * x_t, dim=-1, keepdim=True) + float(self.eps.item()))
        if w_np is not None:
            ref = ref * torch.from_numpy(w_np).to(torch.float32)
        ref_np = ref.numpy().astype(out.dtype)
        diff = mx.max(mx.abs(out - mx.array(ref_np, dtype=out.dtype))).item()
        assert diff < 1e-3, f"RMSNorm mismatch (max abs diff {diff})"


class MultiHeadRMSNormCell(nn.Module):
    """Multi-head RMSNorm using the RMSNormCell kernel."""

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            force_float32_reductions: bool = True,
            use_weight: bool = True,
            debug_compare: Optional[bool] = None,
            param_dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_weight = use_weight
        self.inner = RMSNormCell(
            dims=head_dim,
            eps=eps,
            use_weight=False,
            force_float32_reductions=force_float32_reductions,
            debug_compare=debug_compare,
            param_dtype=param_dtype,
        )
        if use_weight:
            dtype = mx.float32 if force_float32_reductions else param_dtype
            self.weight = mx.ones((num_heads * head_dim,), dtype=dtype)

    def build(self, dtype: mx.Dtype) -> None:
        self.inner.build(dtype)

    def precompile(self, dtypes: tuple[mx.Dtype, ...]) -> None:
        for dtype in dtypes:
            self.build(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, NH, DH = x.shape
        if NH != self.num_heads or DH != self.head_dim:
            raise ValueError("MultiHeadRMSNormCell received mismatched dimensions")
        y = self.inner(x)
        y_flat = y.reshape(B, S, -1)
        if self.use_weight:
            w = mx.array(self.weight, dtype=y_flat.dtype)
            y_flat = mx.multiply(w, y_flat)
        return y_flat


__all__ = [
    'RMSNormMetalKernel',
    'RMSNormCell',
    'MultiHeadRMSNormCell',
]
