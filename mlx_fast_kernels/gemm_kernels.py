"""
Tiled GEMM kernels (MLX + Metal)

Kernels
- `gemm_av`: B = A (m,n) × V (n,k) -> (m,k)
- `gemm_at_b`: Z = Aᵀ (n,m) × B (m,k) -> (n,k)

Design
- Body-only kernel sources with includes in `header` match the
  `mx.fast.metal_kernel` contract (no function signatures in `source`).
- Parameters (m, n, k) are passed in a small `shape` buffer to avoid
  recompilation across calls.
- 2D tiling with threadgroup shared memory enables coalesced loads and high
  arithmetic intensity; barriers synchronize phases.

Tile selection (hardware-aware)
- Tile sizes are chosen at import using `mlx.core.metal.device_info()` and can
  be overridden via env:
  - `XLSTM_GEMM_TILE_AV="TMxT"` (e.g., `32x8`) – TN and TK are set to T.
  - `XLSTM_GEMM_TILE_ATB="TNxTK"` (e.g., `8x32`).
  - Defaults: M3 → AV(32×8), AT_B(8×32); otherwise AV(16×16), AT_B(16×16).

Notes
- Adapted for xLSTM from MetalFaiss (faissmlx.kernels.gemm_kernels) to provide
  a minimal, dependency-free MLX fast-kernel toolbox.
"""

from __future__ import annotations
from typing import Tuple
import os
import mlx.core as mx
try:
    import mlx.core.metal as metal
except Exception:  # pragma: no cover
    metal = None  # type: ignore

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""


def _detect_device_name() -> str:
    try:
        if metal is None:
            return ""
        info = metal.device_info()
        return str(info.get("device_name", ""))
    except Exception:
        return ""


def _select_tile_av() -> Tuple[int, int]:
    """Select (TM, T) for AV kernel where TN=TK=T.

    Env override: XLSTM_GEMM_TILE_AV="TMxT" (e.g., 32x8).
    Defaults: M3 → (32, 8); otherwise (16, 16).
    """
    env = os.environ.get("XLSTM_GEMM_TILE_AV") or os.environ.get("XLSTM_GEMM_TILE")
    if env:
        try:
            tm_s, t_s = env.lower().split("x")
            tm, t = int(tm_s), int(t_s)
            if tm * t <= 1024 and tm > 0 and t > 0:
                return tm, t
        except Exception:
            pass
    name = _detect_device_name().lower()
    if "m3" in name:
        return 32, 8
    return 16, 16


def _select_tile_atb() -> Tuple[int, int, int]:
    """Select (TN, TI, TK) for AT_B kernel.

    Env override: XLSTM_GEMM_TILE_ATB="TNxTK"; TI fixed at 16.
    Defaults: M3 → (8, 16, 32); otherwise (16, 16, 16).
    """
    env = os.environ.get("XLSTM_GEMM_TILE_ATB")
    if env:
        try:
            tn_s, tk_s = env.lower().split("x")
            tn, tk = int(tn_s), int(tk_s)
            if tn * tk <= 1024 and tn > 0 and tk > 0:
                return tn, 16, tk
        except Exception:
            pass
    name = _detect_device_name().lower()
    if "m3" in name:
        return 8, 16, 32
    return 16, 16, 16


def _format_av_source_square(T: int) -> str:
    from string import Template
    tpl = Template(r"""
    // Square thread-tiled GEMM: C = A * V
    // Shapes via shape buffer: [m, n, k]
    const uint T = $T;  // tile size along m, n, k

    threadgroup float Asub[T][T];
    threadgroup float Vsub[T][T];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x; // 0..T-1 (local col)
    uint ty = thread_position_in_threadgroup.y; // 0..T-1 (local row)

    // Global output coordinates
    int col = int(thread_position_in_grid.x);
    int row = int(thread_position_in_grid.y);

    float acc = 0.0f;

    int ntiles = (n + int(T) - 1) / int(T);
    for (int t = 0; t < ntiles; ++t) {
        int a_col = t * int(T) + int(tx);
        float a_val = 0.0f;
        if (row < m && a_col < n) {
            a_val = A[row * n + a_col]; // A[row, a_col]
        }
        Asub[ty][tx] = a_val;

        int v_row = t * int(T) + int(ty);
        float v_val = 0.0f;
        if (v_row < n && col < k) {
            v_val = V[v_row * k + col]; // V[v_row, col]
        }
        Vsub[ty][tx] = v_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < T; ++p) {
            acc = fma(Asub[ty][p], Vsub[p][tx], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < k) {
        C[row * k + col] = acc;
    }
""")
    return tpl.substitute(T=T)


def _format_at_b_source_square(T: int) -> str:
    from string import Template
    tpl = Template(r"""
    // Threadgroup-tiled GEMM for Z = A^T * B
    // Shapes: A (m,n), B (m,k), Z (n,k), shape=[m,n,k]
    const uint T = $T; // square tile size

    threadgroup float Atile[T][T]; // tile of A over (i, n)
    threadgroup float Btile[T][T]; // tile of B over (i, k)

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint tx = thread_position_in_threadgroup.x; // 0..T-1 local col
    uint ty = thread_position_in_threadgroup.y; // 0..T-1 local row

    // Global output coordinates (n, k)
    int colK = int(thread_position_in_grid.x);
    int rowN = int(thread_position_in_grid.y);

    float acc = 0.0f;

    int itiles = (m + int(T) - 1) / int(T); // walk shared dimension m
    for (int t = 0; t < itiles; ++t) {
        int i0 = t * int(T);
        // Load a tile from A along i (columns of A) and rowN (rows of A^T)
        int ai = i0 + int(tx);
        float a_val = 0.0f;
        if (ai < m && rowN < n) {
            a_val = A[ai * n + rowN]; // A[i, rowN]
        }
        Atile[ty][tx] = a_val;

        // Load a tile from B along i (rows) and colK (cols)
        int bi = i0 + int(ty);
        float b_val = 0.0f;
        if (bi < m && colK < k) {
            b_val = B[bi * k + colK]; // B[i, colK]
        }
        Btile[ty][tx] = b_val;

        // Barrier required: tiles are consumed by multiple SIMD groups
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over T (shared dimension)
        for (uint p = 0; p < T; ++p) {
            acc = fma(Atile[ty][p], Btile[p][tx], acc);
        }

        // Barrier before next iteration's loads
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowN < n && colK < k) {
        Z[rowN * k + colK] = acc;
    }
""")
    return tpl.substitute(T=T)


_KERNEL_AV = None
_KERNEL_AT_B = None
_TILES_AV: Tuple[int, int] | None = None
_TILES_ATB: Tuple[int, int, int] | None = None


def _build_av_kernel():
    """Create the tiled kernel for B = A × V."""
    global _TILES_AV
    if _TILES_AV is None:
        _TILES_AV = _select_tile_av()
    TM, Tsel = _TILES_AV
    T = int(min(TM, Tsel))
    return mx.fast.metal_kernel(
        name="gemm_av_tiled",
        input_names=["A", "V", "shape"],
        output_names=["C"],
        header=_HEADER,
        source=_format_av_source_square(T),
        ensure_row_contiguous=True,
    )


def _build_at_b_kernel():
    """Create the tiled kernel for Z = Aᵀ × B."""
    global _TILES_ATB
    if _TILES_ATB is None:
        _TILES_ATB = _select_tile_atb()
    TN, TI, TK = _TILES_ATB
    T = int(min(TN, TI, TK))
    return mx.fast.metal_kernel(
        name="gemm_at_b_tiled",
        input_names=["A", "B", "shape"],
        output_names=["Z"],
        header=_HEADER,
        source=_format_at_b_source_square(T),
        ensure_row_contiguous=True,
    )



def set_gemm_tiles(av: str | Tuple[int, int] | None = None,
                   atb: str | Tuple[int, int] | None = None) -> None:
    """Override tile sizes and reset kernels.

    Args
    - av: "TMxT" or (TM, T) for AV kernel (TN=TK=T). If None, keep current.
    - atb: "TNxTK" or (TN, TK) for AT_B kernel (TI fixed at 16). If None, keep current.
    """
    global _TILES_AV, _TILES_ATB, _KERNEL_AV, _KERNEL_AT_B
    if av is not None:
        if isinstance(av, str) and "x" in av:
            tm_s, t_s = av.lower().split("x")
            _TILES_AV = (int(tm_s), int(t_s))
        elif isinstance(av, tuple):
            _TILES_AV = (int(av[0]), int(av[1]))
        _KERNEL_AV = None
    if atb is not None:
        if isinstance(atb, str) and "x" in atb:
            tn_s, tk_s = atb.lower().split("x")
            _TILES_ATB = (int(tn_s), 16, int(tk_s))
        elif isinstance(atb, tuple):
            _TILES_ATB = (int(atb[0]), 16, int(atb[1]))
        _KERNEL_AT_B = None


def get_gemm_tiles() -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
    """Return the current tile sizes: (AV(TM,T), AT_B(TN,TI,TK))."""
    av = _TILES_AV or _select_tile_av()
    atb = _TILES_ATB or _select_tile_atb()
    return av, atb


def gemm_av(A: mx.array, V: mx.array) -> mx.array:
    """Compute B = A @ V with a shared‑memory tiled Metal kernel."""
    global _KERNEL_AV
    if _KERNEL_AV is None:
        _KERNEL_AV = _build_av_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    # Shape check: A (m,n) @ V (n,k)
    if int(V.shape[0]) != n:
        raise ValueError(f"gemm_av: incompatible shapes A{tuple(A.shape)} and V{tuple(V.shape)}")
    shape = mx.array([m, n, k], dtype=mx.uint32)

    TM, Tsel = _TILES_AV or _select_tile_av()
    T = int(min(TM, Tsel))
    # MLX `grid` is threads, not threadgroups. Launch one thread per output element.
    gx = ((k + T - 1) // T) * T
    gy = ((m + T - 1) // T) * T
    grid = (gx, gy, 1)
    threadgroup = (T, T, 1)

    (B,) = _KERNEL_AV(
        inputs=[A, V, shape],
        output_shapes=[(m, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return B


def gemm_at_b(A: mx.array, B: mx.array) -> mx.array:
    """Compute Z = A.T @ B with a shared‑memory tiled Metal kernel."""
    global _KERNEL_AT_B
    if _KERNEL_AT_B is None:
        _KERNEL_AT_B = _build_at_b_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(B.shape[1])
    # Shape check: (A.T) (n,m) @ B (m,k)
    if int(B.shape[0]) != m:
        raise ValueError(f"gemm_at_b: incompatible shapes A{tuple(A.shape)} and B{tuple(B.shape)}")
    shape = mx.array([m, n, k], dtype=mx.uint32)

    TN, TI, TK = _TILES_ATB or _select_tile_atb()
    T = int(min(TN, TI, TK))
    gx = ((k + T - 1) // T) * T
    gy = ((n + T - 1) // T) * T
    grid = (gx, gy, 1)
    threadgroup = (T, T, 1)

    (Z,) = _KERNEL_AT_B(
        inputs=[A, B, shape],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Z
