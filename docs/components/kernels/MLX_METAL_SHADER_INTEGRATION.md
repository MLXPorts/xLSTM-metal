# MLX + Metal Shader Integration

This note explains how to leverage existing `.metal` kernels in this repo with MLX, and how to port them into the `mx.fast.metal_kernel` pattern that runs directly from Python.

## What We Have

- Native Metal shader file in this repo (archived prototype location):
  - `research_archive/metal_prototypes/kernels_metal/shaders/mlstm_kernels.metal`
    - `soft_cap_kernel`, `memcpy_kernel`, `mlstm_step_full_kernel`
- A PyTorch Objective‑C++ bridge that compiles and launches those shaders:
  - `research_archive/metal_prototypes/kernels_metal/pytorch_ext/mlstm_metal_backend.mm`

## Two Ways to Use Metal

- PyTorch bridge (existing): call from C++/PyTorch; good for our MPS path.
- MLX `mx.fast.metal_kernel` (recommended for MLX): compile a kernel at runtime from a header + body source and launch it from Python.

> Practical point: `mx.fast.metal_kernel` expects body‑only kernel text with a specific input/output contract. Using a full `.metal` function definition verbatim often fails to compile (as we observed in MetalFaiss). The reliable path is to port kernels to the body‑only style.

## Porting a `.metal` Kernel to MLX (Body‑Only)

1) Identify the operation, inputs, and outputs.
2) Express the kernel in terms of MLX’s input/output names and a small extra `shape` buffer.
3) Use `thread_position_in_threadgroup` and `threadgroup_position_in_grid` to avoid runtime integer division/mod in hot loops.
4) Stage tiles in `threadgroup` memory if the op benefits from reuse; fence with `threadgroup_barrier` between phases.

Example (see working GEMM tiles):
- Source: `mlx_fast_kernels/gemm_kernels.py` — functions `_format_av_source(...)` and `_format_at_b_source(...)` return body‑only Metal.
- Launch: `mx.fast.metal_kernel(name=..., input_names=[...], output_names=[...], header=..., source=..., ensure_row_contiguous=True)`.

## Adapting `soft_cap_kernel`

Original signature (in `.metal`):
```metal
kernel void soft_cap_kernel(device float* input, device float* output,
                            constant float& cap_value, constant uint& size,
                            uint id [[thread_position_in_grid]])
```
Port to MLX body‑only pattern:
- Inputs: `input`, `cap` (scalar via a 1‑element buffer or pack into `shape`)
- Output: `out`
- Body computes the elementwise cap; guards for `id >= size`.

We suggest following the GEMM template and building a simple helper:
```python
import mlx.core as mx
HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""
BODY = r"""
  int size = int(shape[0]);
  uint id = thread_position_in_grid.x;
  if (id >= size) return;
  float val = input[id];
  out[id] = cap * tanh(val / cap);
"""
ker = mx.fast.metal_kernel(
  name="softcap", input_names=["input","shape"], output_names=["out"],
  header=HEADER, source=BODY, ensure_row_contiguous=True)
```
Launch with `shape=[N]` and pass a pre‑scaled cap into the body or add a second scalar buffer if you prefer.

## Adapting `mlstm_step_full_kernel`

This kernel updates per‑head covariance `C`, normalizer `N`, and hidden output `H` given gates and q/k/v. It’s non‑trivial but follows patterns you see in our GEMM tiles:
- Flatten (b,h,dh) to 2D/3D index math using grid positions; avoid inner `/` and `%`.
- Use `fma` in inner loops and keep per‑thread live state small.
- Consider tiling by head and dimension to coalesce loads.

Recommendation: keep the PyTorch bridge path for this complex kernel in the PyTorch/MPS stack, and use the MLX path for projection and other GEMM‑like ops unless we decide to port the entire recurrent step to MLX.

## Notes and Pitfalls

- Avoid dynamic stack arrays in kernels; prefer fixed tiles or unrolled small loops.
- Watch for integer division/modulus in hot loops; reframe with 2D grid math.
- Keep barriers to `threadgroup_barrier` between phases; only use `simdgroup_barrier` when reductions are warp‑local.
- Accumulate in float32 even if inputs are float16/bfloat16 to preserve accuracy.

## Where to Learn by Example

- Tiled GEMM (working): `mlx_fast_kernels/gemm_kernels.py`
- Ported MLX docs: `docs/mlx_reference/Kernel-Guide.md`, `.../Comprehensive-MLX-Metal-Guide.md`, `.../Streams-Guide.md`, and `.../Spot-Tests.md`
