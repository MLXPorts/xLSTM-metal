"""Variable quantization using MLX Metal kernels.

Inspired by GEMM tiling patterns:
- Body-only Metal source (MLX generates signatures)
- Dynamic shapes via buffer to avoid recompiles
- Cooperative loading with unique writers
- Double barrier synchronization (after load, after compute)
- Optional double buffering for latency hiding
- FMA instructions for throughput
"""

import mlx.core as mx
import os
from typing import Optional

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_QUANTIZE_TILED_KERNEL = r"""
    // Tiled quantization kernel with cooperative loading
    // Following GEMM patterns: grid=threads, threadgroup=tile_size
    
    const uint TILE_SIZE = 256;  // Threads per threadgroup
    const uint VEC_WIDTH = 4;    // float4 vectorization
    
    uint local_tid = thread_position_in_threadgroup.x;
    uint tg_id = threadgroup_position_in_grid.x;
    uint tg_size = threads_per_threadgroup.x;
    
    int size = int(shape[0]);
    float scale = params[0];
    float inv_scale = 1.0f / scale;  // Pre-computed for FMA
    
    // Threadgroup shared memory (1KB for 256 floats)
    threadgroup float tile[TILE_SIZE];
    
    // Each threadgroup processes TILE_SIZE elements
    uint tile_start = tg_id * TILE_SIZE;
    uint tile_end = min(tile_start + TILE_SIZE, uint(size));
    uint tile_len = tile_end - tile_start;
    
    // Cooperative load (coalesced, unique writers)
    if (local_tid < tile_len) {
        tile[local_tid] = inp[tile_start + local_tid];
    }
    
    // Barrier 1: Ensure all loads complete before compute
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process in shared memory with vectorization
    // Each thread processes 4 elements if possible
    uint vec_count = tile_len / VEC_WIDTH;
    uint my_vec = local_tid;
    
    if (my_vec < vec_count) {
        uint base = my_vec * VEC_WIDTH;
        
        // Vectorized load from shared memory
        float4 x_vec = float4(tile[base], tile[base+1], tile[base+2], tile[base+3]);
        
        // Quantization with FMA pattern
        x_vec = clamp(x_vec, -1.0f, 1.0f);
        x_vec = fma(x_vec, scale, 0.0f);  // x * scale + 0
        x_vec = round(x_vec);
        x_vec = fma(x_vec, inv_scale, 0.0f);  // x * inv_scale + 0
        
        // Store back to shared memory
        tile[base] = x_vec.x;
        tile[base+1] = x_vec.y;
        tile[base+2] = x_vec.z;
        tile[base+3] = x_vec.w;
    }
    
    // Handle tail elements
    uint tail_start = vec_count * VEC_WIDTH;
    uint tail_idx = tail_start + local_tid;
    if (tail_idx < tile_len) {
        float x = tile[tail_idx];
        x = clamp(x, -1.0f, 1.0f);
        x = fma(round(x * scale), inv_scale, 0.0f);
        tile[tail_idx] = x;
    }
    
    // Barrier 2: Ensure all compute complete before store
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooperative store back to global memory (coalesced)
    if (local_tid < tile_len) {
        out[tile_start + local_tid] = tile[local_tid];
    }
"""

_QUANTIZE_TILED_KERNEL_DB = r"""
    // Double-buffered variant for latency hiding
    // Two tiles: ping-pong between them while processing
    
    const uint TILE_SIZE = 256;
    const uint VEC_WIDTH = 4;
    
    uint local_tid = thread_position_in_threadgroup.x;
    uint tg_id = threadgroup_position_in_grid.x;
    uint tg_size = threads_per_threadgroup.x;
    
    int size = int(shape[0]);
    float scale = params[0];
    float inv_scale = 1.0f / scale;
    
    // Double buffers (2KB total for 2x256 floats)
    threadgroup float tile0[TILE_SIZE];
    threadgroup float tile1[TILE_SIZE];
    
    uint num_tiles = (uint(size) + TILE_SIZE - 1) / TILE_SIZE;
    if (num_tiles == 0) return;
    
    // Prefetch first tile into buffer 0
    uint tile0_start = 0 * TILE_SIZE;
    uint tile0_end = min(tile0_start + TILE_SIZE, uint(size));
    uint tile0_len = tile0_end - tile0_start;
    
    if (local_tid < tile0_len) {
        tile0[local_tid] = inp[tile0_start + local_tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    bool use_buf0 = true;
    for (uint t = 0; t < num_tiles; ++t) {
        uint current_tile = tg_id * num_tiles + t;
        if (current_tile >= num_tiles) break;
        
        uint tile_start = current_tile * TILE_SIZE;
        uint tile_end = min(tile_start + TILE_SIZE, uint(size));
        uint tile_len = tile_end - tile_start;
        
        // Prefetch NEXT tile into other buffer
        if (t + 1 < num_tiles) {
            uint next_start = (current_tile + 1) * TILE_SIZE;
            uint next_end = min(next_start + TILE_SIZE, uint(size));
            uint next_len = next_end - next_start;
            
            if (local_tid < next_len) {
                if (use_buf0) {
                    tile1[local_tid] = inp[next_start + local_tid];
                } else {
                    tile0[local_tid] = inp[next_start + local_tid];
                }
            }
        }
        
        // Process CURRENT tile
        uint vec_count = tile_len / VEC_WIDTH;
        uint my_vec = local_tid;
        
        if (use_buf0) {
            if (my_vec < vec_count) {
                uint base = my_vec * VEC_WIDTH;
                float4 x = float4(tile0[base], tile0[base+1], tile0[base+2], tile0[base+3]);
                x = clamp(x, -1.0f, 1.0f);
                x = fma(round(fma(x, scale, 0.0f)), inv_scale, 0.0f);
                tile0[base] = x.x; tile0[base+1] = x.y; tile0[base+2] = x.z; tile0[base+3] = x.w;
            }
            uint tail_idx = vec_count * VEC_WIDTH + local_tid;
            if (tail_idx < tile_len) {
                tile0[tail_idx] = fma(round(tile0[tail_idx] * scale), inv_scale, 0.0f);
            }
        } else {
            if (my_vec < vec_count) {
                uint base = my_vec * VEC_WIDTH;
                float4 x = float4(tile1[base], tile1[base+1], tile1[base+2], tile1[base+3]);
                x = clamp(x, -1.0f, 1.0f);
                x = fma(round(fma(x, scale, 0.0f)), inv_scale, 0.0f);
                tile1[base] = x.x; tile1[base+1] = x.y; tile1[base+2] = x.z; tile1[base+3] = x.w;
            }
            uint tail_idx = vec_count * VEC_WIDTH + local_tid;
            if (tail_idx < tile_len) {
                tile1[tail_idx] = fma(round(tile1[tail_idx] * scale), inv_scale, 0.0f);
            }
        }
        
        // Barrier after compute
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Store current tile back
        if (local_tid < tile_len) {
            if (use_buf0) {
                out[tile_start + local_tid] = tile0[local_tid];
            } else {
                out[tile_start + local_tid] = tile1[local_tid];
            }
        }
        
        // Barrier before next prefetch overwrites
        if (t + 1 < num_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            use_buf0 = !use_buf0;
        }
    }
"""


def _select_use_double_buffer() -> bool:
    """Determine if double buffering should be used."""
    if os.environ.get("XLSTM_QUANT_DB", "0") == "1":
        return True
    return False  # Default: single buffer (simpler, less shared memory)


class VariableQuantizationMLXKernel:
    """Apply variable quantization using GEMM-inspired tiling patterns.

    Architecture:
    - Threadgroup shared memory for tile processing
    - Cooperative loading with unique writers
    - Double barrier synchronization (after load, after compute)
    - Optional double buffering for latency hiding
    - FMA instructions for throughput
    - MLX grid semantics: grid=total_threads, threadgroup=tile_size
    """

    def __init__(self, use_double_buffer: bool = False):
        """Initialize the kernel.

        Args:
            use_double_buffer: If True, use double buffering for latency hiding.
                             Costs 2x shared memory but can improve performance.
        """
        self._kernel: Optional[mx.fast.metal_kernel] = None
        self._use_double_buffer = use_double_buffer

    def compile(self) -> mx.fast.metal_kernel:
        """
        Compile the Metal kernel and return it.

        Following GEMM pattern: compile early to avoid recompilation overhead.

        Returns:
            mx.fast.metal_kernel: The compiled Metal kernel.
        """
        if self._kernel is None:
            # Select kernel variant
            use_db = self._use_double_buffer or _select_use_double_buffer()
            kernel_source = _QUANTIZE_TILED_KERNEL_DB if use_db else _QUANTIZE_TILED_KERNEL

            self._kernel = mx.fast.metal_kernel(
                name="variable_quantization_tiled",
                input_names=["inp", "params", "shape"],
                output_names=["out"],
                header=_HEADER,
                source=kernel_source,
                ensure_row_contiguous=True,
            )
        return self._kernel

    def __call__(self, x: mx.array, bits: int) -> mx.array:
        """
        Apply variable quantization to the input array.

        Parameters:
            x (mx.array): The input array (values should be in [-1, 1] range).
            bits (int): The number of bits for quantization (e.g., 4, 8).

        Returns:
            mx.array: The quantized array with same shape as input.
        """
        if not isinstance(bits, int) or bits < 1 or bits > 16:
            raise ValueError(f"bits must be an integer between 1 and 16, but got {bits}")

        # Prepare inputs
        x_flat = x.reshape(-1).astype(mx.float32)
        scale = float(2 ** (bits - 1) - 1)
        params_arr = mx.array([scale], dtype=mx.float32)
        shape_arr = mx.array([x_flat.size], dtype=mx.uint32)

        # Grid configuration following GEMM pattern
        # Grid = TOTAL THREADS (not threadgroups!)
        # Threadgroup = tile size
        TILE_SIZE = 256
        num_tiles = (x_flat.size + TILE_SIZE - 1) // TILE_SIZE

        # Grid: total threads = num_tiles * TILE_SIZE
        grid = (num_tiles * TILE_SIZE, 1, 1)
        threadgroup = (TILE_SIZE, 1, 1)

        # Call kernel
        kernel = self.compile()
        (out_flat,) = kernel(
            inputs=[x_flat, params_arr, shape_arr],
            output_shapes=[x_flat.shape],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
        )

        return out_flat.reshape(x.shape)


# Global instance (single buffer by default)
variable_quantization = VariableQuantizationMLXKernel(use_double_buffer=False)


def quantize(x: mx.array, bits: int) -> mx.array:
    """
    Quantize array to specified bit precision.

    Uses GEMM-inspired tiling with:
    - Threadgroup shared memory
    - Cooperative loading
    - Double barrier synchronization
    - FMA instructions

    Args:
        x: Input array with values in [-1, 1] range
        bits: Number of bits for quantization (1-16)

    Returns:
        Quantized array with same shape as input
    """
    return variable_quantization(x, bits)


def set_double_buffer(enable: bool) -> None:
    """Enable or disable double buffering.

    Double buffering trades 2x shared memory for potential latency hiding.
    Resets the kernel to force recompilation with new variant.

    Args:
        enable: True to use double buffering, False for single buffer
    """
    global variable_quantization
    variable_quantization = VariableQuantizationMLXKernel(use_double_buffer=enable)
    """
    Quantize array to specified bit precision.

    Args:
        x: Input array with values in [-1, 1] range
        bits: Number of bits for quantization (1-16)

    Returns:
        Quantized array with same shape as input
    """
    return variable_quantization(x, bits)

