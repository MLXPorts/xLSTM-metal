"""
MLX stream utilities: background waiters and evaluation-triggered callbacks.

Adapted from the MetalFaiss project (python/metalfaiss/utils/streams.py) to
provide stream-scoped synchronization helpers for MLX code in this repo.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx


def on_stream_complete(
        stream: mx.core.Stream,
        callback: Callable[..., Any],
        *args: Any,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs: Any,
) -> threading.Thread | Future:
    """Wait for `stream` to complete in the background, then run `callback`.

    Scopes synchronization to the specified stream so other streams continue.
    Returns a daemon Thread (default) or a Future if an executor is provided.
    """

    def wait_and_call() -> Any:
        """

        :return:
        """
        mx.synchronize(stream)
        return callback(*args, **kwargs)

    if executor is not None:
        return executor.submit(wait_and_call)
    t = threading.Thread(target=wait_and_call, daemon=True)
    t.start()
    return t


async def on_stream_complete_async(
        stream: mx.core.Stream,
        callback: Callable[..., Any],
        *args: Any,
        loop=None,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs: Any,
) -> Any:
    """Async variant: wait on `stream` off-loop and then run `callback`."""
    if loop is None:
        import asyncio

        loop = asyncio.get_running_loop()

    def wait_and_call() -> Any:
        """

        :return:
        """
        mx.synchronize(stream)
        return callback(*args, **kwargs)

    return await loop.run_in_executor(executor, wait_and_call)


def after_eval(
        arrays: Sequence[mx.core.array],
        callback: Callable[..., Any],
        *args: Any,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs: Any,
) -> Future:
    """Evaluate `arrays` in a worker thread, then run `callback`.

    Useful when the trigger is data readiness rather than stream completion.
    Returns a Future from the provided or internal executor.
    """
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1)

    def wait_and_call() -> Any:
        """

        :return:
        """
        mx.eval(*arrays)
        return callback(*args, **kwargs)

    return executor.submit(wait_and_call)
