"""
MLX Streams Overlap Demo

Shows using tools.mlx_streams to wait on a stream in the background and
trigger a callback when compute completes, without blocking other work.
"""

import time
from concurrent.futures import ThreadPoolExecutor
import mlx.core as mx
from tools.mlx_streams import on_stream_complete


def heavy_op(n=1_000_000):
    a = mx.random.normal((n,))
    b = mx.random.normal((n,))
    return mx.sum(a * b)


def log_done(tag):
    print(f"[{time.strftime('%H:%M:%S')}] Stream '{tag}' completed.")


if __name__ == "__main__":
    s = mx.gpu  # default GPU stream
    # Launch heavy op and register callback
    y = heavy_op()
    t = on_stream_complete(s, log_done, "demo")
    # Do some CPU work while GPU runs
    for i in range(3):
        time.sleep(0.5)
        print("CPU doing other work...")
    mx.eval(y)  # ensure completion
    t.join()

