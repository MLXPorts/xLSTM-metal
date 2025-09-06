<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

MLX Streams: Practical Overlap And Concurrency (Plain-Speech Guide)

Why streams matter
- Streams let you run independent work in parallel (CPU and GPU) instead of queueing everything on the default stream. That overlap is how you cut idle time and lift throughput.

Core habits
- Put independent tasks on their own streams. If you skip the `stream=` argument, ops run on the default stream and can serialize.
- Keep a small, consistent set of streams per device (e.g., one or two for CPU work, one or two for GPU-heavy kernels). Predictability beats churning streams.
- Synchronize only at clear boundaries (e.g., logging, checkpoints, timed sections). Don’t fence after every op.

Essentials
- Every op (including RNG) accepts a `stream` (or a `device`). Passing a `device` runs on that device’s default stream; pass a `stream` for fine-grained control.
- MLX tracks cross-stream dependencies for you: if a result from stream A is used on stream B, MLX inserts the minimal wait.
- Use context managers to set scoped defaults so code stays readable.

Simple example: CPU–GPU overlap with automatic handoff
```python
import mlx.core as mx

# Create one CPU stream and one GPU stream
s_cpu = mx.new_stream(mx.cpu)
s_gpu = mx.new_stream(mx.gpu)

with mx.stream(s_cpu):
    x = mx.add(a, b)          # runs on CPU stream

with mx.stream(s_gpu):
    y = mx.multiply(x, b)     # runs on GPU stream; MLX waits only if x isn’t ready

# Synchronize at a boundary (e.g., before reading y for logging)
mx.synchronize(s_gpu)
```

Callbacks without blocking the UI or other streams
- Wait on a specific stream in the background, then fire a host callback. This scopes the wait and keeps other streams running.

```python
from concurrent.futures import ThreadPoolExecutor
from python.metalfaiss.utils.streams import on_stream_complete

s_gpu = mx.new_stream(mx.gpu)

def on_done():
    print("gpu stream finished a step")

# Wait in a worker, then call the callback
on_stream_complete(s_gpu, on_done, executor=ThreadPoolExecutor(max_workers=1))
```

Asyncio integration
- If you use an async app loop, do the wait in `run_in_executor` to avoid blocking the event loop.

```python
import asyncio
from python.metalfaiss.utils.streams import on_stream_complete_async

await on_stream_complete_async(s_gpu, lambda: print("step done"))
```

Trigger by array evaluation
- When the trigger is "these results are ready", evaluate them in a worker thread, then call the callback.

```python
from python.metalfaiss.utils.streams import after_eval

future = after_eval([y], lambda: print("y ready"))
```

Pipelines and prefetch
- Combine compute streams with MLX Data streams to keep the GPU fed while the CPU decodes and augments batches.
- Treat the data path as a stream early, then prefetch.

Sketch
```python
# Pseudocode: use mlx-data to stream batches; overlap with GPU compute stream
ds = buffer.to_stream()         # turn buffer into a data stream
ds = ds.batch(32).prefetch(8, 4)

s_gpu = mx.new_stream(mx.gpu)
for batch in ds:
    with mx.stream(s_gpu):
        logits = model_forward(batch)
        loss = loss_fn(logits, batch.labels)
    # synchronize here only if you need to log/step synchronously
```

Synchronization strategy
- Prefer stream-scoped `mx.synchronize(s)` at boundaries. Avoid global `mx.synchronize()` unless you truly want to stall the default device’s default stream.
- Keep dependent steps in the same stream to benefit from in‑stream ordering. Split only truly independent work across streams.

Determinism and RNG
- RNG ops honor the `stream` argument. Keep RNG on a stable stream mapping for reproducibility.
- If RNG values cross streams, MLX still orders the dependency; just avoid reshuffling streams mid‑run if you care about exact repeatability.

Note
- This repository does not include an MLX Swift implementation. All examples and APIs here refer to MLX Python.

Checklist
- Define a small, fixed set of streams per device.
- Use data streams with prefetch to keep compute streams busy.
- Synchronize only where results cross program boundaries.
- Rely on MLX’s cross‑stream dependency tracking rather than adding global fences.

Where to find the helpers
- `tools/mlx_streams.py` provides `on_stream_complete`, `on_stream_complete_async`, and `after_eval` so you don’t have to re-write the patterns.

Common pitfalls
- Letting everything fall onto the default stream (lost overlap, hidden contention).
- Sprinkling `synchronize()` everywhere (kills throughput).
- Mixing defaults and custom streams haphazardly (surprising waits). Use scoped contexts.

References
- MLX Streams (Python): usage, devices and streams, `mx.synchronize`
- MLX‑Data: buffers, streams, and samples
