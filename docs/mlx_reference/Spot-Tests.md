<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

Spot Tests: Teaching Microbenchmarks for MLX + Metal

Purpose

These small, focused tests demonstrate why certain optimization patterns matter. Each test includes: what to change, how to measure, and what to look for. Run them on your machine and compare wall time; optionally capture one representative run in Xcode to inspect counters.

How to run

- Use MLX’s `mx.fast.metal_kernel` as shown. Each snippet compiles a kernel once and launches it repeatedly.
- Measure with `time.perf_counter()` and force evaluation via `mx.eval(...)`.
- Prefer a size that is large enough to be compute-bound but quick to iterate (e.g., 1–10M elements on your GPU).

1) Avoid Integer Division/Modulus by Non-Constants

Idea
- Re-map 1D thread IDs to a 2D grid using `/` and `%` is slow when the divisor isn’t a compile-time constant. Prefer 2D grids or precomputed strides.

Kernels
```python
import mlx.core as mx, time

header = """#include <metal_stdlib>\nusing namespace metal;\n"""

# Slow: 1D with / and % by runtime k
src_div = r"""
  uint gid = thread_position_in_grid.x;
  uint n = shape[0];
  uint k = shape[1];
  uint total = n * k;
  if (gid >= total) return;
  uint col = gid % k;     // non-constant modulus
  uint row = gid / k;     // non-constant division
  out[gid] = in0[row * k + col] * 2.0f;
"""

# Fast: 2D grid, no division/modulus
src_2d = r"""
  uint2 g = thread_position_in_grid.xy;
  uint n = shape[0];
  uint k = shape[1];
  if (g.x >= k || g.y >= n) return;
  uint idx = g.y * k + g.x;
  out[idx] = in0[idx] * 2.0f;
"""

def bench_div_vs_2d(n=2048, k=2048, reps=5):
  arr = mx.random.normal((n*k,)).astype(mx.float32)
  shape = mx.array([n,k], dtype=mx.uint32)
  ker_div = mx.fast.metal_kernel(name="divmap", input_names=["in0","shape"], output_names=["out"], header=header, source=src_div, ensure_row_contiguous=True)
  ker_2d  = mx.fast.metal_kernel(name="twod",   input_names=["in0","shape"], output_names=["out"], header=header, source=src_2d,  ensure_row_contiguous=True)
  # Warm
  (y,) = ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)); mx.eval(y)
  (y,) = ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=( (k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)); mx.eval(y)
  def timeit(f):
    ts=[]
    for _ in range(reps):
      t0=time.perf_counter(); (y,)=f(); mx.eval(y); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]
  t_div = timeit(lambda: ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)))
  t_2d  = timeit(lambda: ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=( (k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)))
  print(f"int-div map: {t_div:.4f}s; 2D map: {t_2d:.4f}s")
```

What to expect
- The 2D version should be faster when k is not a compile-time literal. If you must stick to 1D, precompute strides or choose k as a function constant.

2) Dynamic Stack Arrays vs Compile-Time Constant Arrays

Idea
- Dynamically-indexed, non-constant stack arrays can be catastrophic. If you can, restructure to constant-sized arrays or unrolled loops.

Kernels (illustrative)
```cpp
// Bad: dynamic index into non-constant array (don’t do this in hot loops)
int foo(int a, int b, int c) {
  int tmp[2] = { a, b };
  return tmp[c];
}
// Okay: fixed-size loop; compiler can unroll
int sum3(int a, int b, int c) {
  int tmp3[3] = { a, b, c };
  int s = 0; for (int i=0;i<3;++i) s += tmp3[i]; return s;
}
```

How to measure
- If you have a real kernel that uses dynamic stack arrays, replace the pattern with a constant-sized array or a small unrolled loop and re-measure. Expect substantial speedups if this was on a hot path.

3) Barriers: threadgroup_barrier vs simdgroup_barrier

Idea
- Use the smallest scope that is correct. For warp-only reductions that never touch threadgroup memory across warps, `simdgroup_barrier` can be cheaper.

Sketch
```cpp
// Warp-only reduction
float x = ...;
float r = simd_sum(x);
// Synchronize lanes within the warp (no TG memory used across warps)
simdgroup_barrier(mem_flags::mem_none);
```

Caution
- If you stage data in `threadgroup` memory across multiple warps (e.g., 16×16 tiles), you must use `threadgroup_barrier(mem_flags::mem_threadgroup)`.

4) half I/O with float Accumulation

Idea
- Reduce bandwidth with `half` for loads/stores, but keep accumulators in `float`.

Kernel snippet
```cpp
half ha = in_h[i];
float a = float(ha);
acc = fma(a, b, acc);
out_h[i] = half(acc);
```

How to measure
- Compare float32 end-to-end vs half I/O + float accumulate in a bandwidth-sensitive kernel (elementwise ops or GEMM with small arithmetic intensity). Validate error bounds are acceptable.

5) Branchless Select vs if/else

Idea
- Prefer fast ternary/select in simple guards.

Kernel snippet
```cpp
// Branchless
float y = cond ? a : b;
```

How to measure
- On a kernel with simple clamping/guards, compare if/else vs ternary. The difference may be small in isolation; the main benefit is avoiding divergence and enabling better instruction scheduling.

Appendix: Xcode GPU Capture

- For any test, capture one run and check: shader time, memory transactions, and occupancy. Confirm qualitative expectations (fewer scalar loads, no unexpected barriers).

