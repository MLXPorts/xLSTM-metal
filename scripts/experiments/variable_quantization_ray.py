
from __future__ import annotations

import argparse, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import ray
    from ray.util.metrics import Gauge  # type: ignore
    try:
        from ray.dag import InputNode as _RayInputNode
    except Exception:
        _RayInputNode = None  # type: ignore
except Exception:
    ray = None  # type: ignore
    Gauge = None  # type: ignore
    _RayInputNode = None  # type: ignore


@dataclass
class Config:
    size: int
    dtype: str
    num_shards: int
    block_size: int
    bits: int
    method: str
    verify: bool
    seed: int
    stats_log: Optional[str]
    use_async: bool
    use_compiled_graph: bool
    ray_local_mode: int


def load_config(path: str) -> Config:
    cfg = json.loads(Path(path).read_text())
    return Config(
        size=int(cfg.get("size", 11000000)),
        dtype=str(cfg.get("dtype", "float32")),
        num_shards=int(cfg.get("num_shards", 8)),
        block_size=int(cfg.get("block_size", 4096)),
        bits=int(cfg.get("bits", 4)),
        method=str(cfg.get("method", "floating_point")),
        verify=bool(cfg.get("verify", True)),
        seed=int(cfg.get("seed", 42)),
        stats_log=cfg.get("stats_log"),
        use_async=bool(cfg.get("use_async", True)),
        use_compiled_graph=bool(cfg.get("use_compiled_graph", True)),
        ray_local_mode=int(cfg.get("ray_local_mode", 1)),
    )


def _make_runtime_env() -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in (
            "TOKENIZERS_PARALLELISM",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "RAY_CGRAPH_ENABLE_TORCH_PROFILING",
            "RAY_CGRAPH_ENABLE_NVTX_PROFILING",
        )
    }
    out = {"env_vars": env}
    nsight = os.environ.get("XLSTM_RAY_NSIGHT")
    if nsight:
        out["nsight"] = nsight  # type: ignore[index]
    return out


class _NoopGauge:
    def set(self, *a, **k):
        return None


def _make_gauges(prefix: str = "varquant"):
    if Gauge is None:
        return {"blocks_sec": _NoopGauge(), "rss_mb": _NoopGauge()}
    return {
        "blocks_sec": Gauge(f"{prefix}_blocks_sec", "Blocks per second (quantize)"),
        "rss_mb": Gauge(f"{prefix}_rss_mb", "RSS memory in MB"),
    }


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return 0.0


def _quantize_block(block: np.ndarray, bits: int, method: str) -> np.ndarray:
    if method == "quantile":
        num_levels = 2 ** bits
        sorted_data = np.sort(block)
        boundaries = np.percentile(sorted_data, np.linspace(0, 100, num_levels + 1)[1:-1])
        idx = np.searchsorted(boundaries, block)
        # For verification, map back to bin mean
        bins = np.array_split(sorted_data, num_levels)
        reps = np.array([np.mean(b) if len(b) else 0.0 for b in bins], dtype=np.float32)
        return reps[idx]
    # floating_point scaling
    max_val = 2 ** bits - 1
    abs_max = np.max(np.abs(block)) if block.size else 1.0
    scale = abs_max / (max_val / 2) if abs_max > 0 else 1.0
    scaled = np.clip(block / scale, -max_val / 2, max_val / 2)
    q = np.round(scaled + max_val / 2).astype(np.uint8)
    deq = (q.astype(np.float32) - max_val / 2) * scale
    return deq.astype(np.float32)


def _split_indices(n: int, k: int) -> List[Tuple[int, int]]:
    base = n // k
    rem = n % k
    out = []
    s = 0
    for i in range(k):
        e = s + base + (1 if i < rem else 0)
        out.append((s, e))
        s = e
    return out


try:
    _CG = {"compute": 1, "ctrl": 8}
    _remote_kwargs = dict(num_cpus=1, max_restarts=0, max_task_retries=0, concurrency_groups=_CG)
except Exception:
    _remote_kwargs = dict(num_cpus=1, max_restarts=0, max_task_retries=0)


@ray.remote(**_remote_kwargs)  # type: ignore[misc]
class VarQuantShard:
    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self._data_ref = None
        self._start = 0
        self._end = 0
        self._gauges = _make_gauges(prefix=f"varquant_{shard_id}")

    def attach_source(self, data_ref, start: int, end: int) -> str:
        self._data_ref = data_ref
        self._start, self._end = int(start), int(end)
        return "ok"

    def quantize(self, block_size: int, bits: int, method: str) -> "np.ndarray":
        arr = self._data_ref
        if not isinstance(arr, np.ndarray):
            arr = ray.get(self._data_ref)
        view = arr[self._start:self._end]
        n = view.size
        bs = int(block_size)
        out_blocks = []
        t0 = time.perf_counter()
        for s in range(0, n, bs):
            e = min(n, s + bs)
            out_blocks.append(_quantize_block(view[s:e], bits, method))
        dt = time.perf_counter() - t0
        blocks = (n + bs - 1) // bs
        bps = blocks / max(dt, 1e-9)
        try:
            self._gauges["blocks_sec"].set(bps)
            self._gauges["rss_mb"].set(_rss_mb())
        except Exception:
            pass
        return np.concatenate(out_blocks) if out_blocks else np.zeros((0,), dtype=np.float32)

    def verify(self, quantized: "np.ndarray") -> dict:
        arr = self._data_ref
        if not isinstance(arr, np.ndarray):
            arr = ray.get(self._data_ref)
        view = arr[self._start:self._end]
        expected = np.square(view.astype(np.float32))
        got = np.square(quantized.astype(np.float32))
        mse = float(np.mean((expected - got) ** 2))
        return {"shard": self.shard_id, "mse": mse, "n": int(view.size)}

    def metrics(self) -> dict:
        return {"rss_mb": _rss_mb(), "start": self._start, "end": self._end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/varquant_default.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if ray is None:
        raise RuntimeError("Ray not installed. Install ray to run this experiment.")

    # Ray init
    ray.init(ignore_reinit_error=True, local_mode=bool(cfg.ray_local_mode))
    try:
        # Data
        np.random.seed(cfg.seed)
        data = np.random.rand(cfg.size).astype(cfg.dtype)
        data_ref = ray.put(data)

        # Shards
        shards = _split_indices(cfg.size, cfg.num_shards)
        runtime_env = _make_runtime_env()
        Actor = VarQuantShard
        actors = []
        for sid, (s, e) in enumerate(shards):
            try:
                a = Actor.options(runtime_env=runtime_env).remote(sid)  # type: ignore
            except Exception:
                a = Actor.remote(sid)
            actors.append((sid, s, e, a))

        # Attach sources
        for sid, s, e, a in actors:
            ray.get(a.attach_source.remote(data_ref, s, e))

        # Optional compiled graph (beta)
        compiled = {}
        # Disable compiled graph in local_mode; only safe in multi-process mode
        use_cg = (cfg.ray_local_mode == 0) and cfg.use_compiled_graph and (_RayInputNode is not None)
        if use_cg:
            try:
                for sid, s, e, a in actors:
                    with _RayInputNode() as inp:  # type: ignore
                        dag = a.quantize.bind(inp[0], inp[1], inp[2])
                    compiled[sid] = dag.experimental_compile()  # type: ignore
            except Exception:
                compiled = {}
                use_cg = False

        # Quantize per shard
        refs = []
        t0 = time.perf_counter()
        for sid, s, e, a in actors:
            if use_cg and sid in compiled:
                try:
                    refs.append(compiled[sid].execute((cfg.block_size, cfg.bits, cfg.method)))  # type: ignore
                    continue
                except Exception:
                    pass
            refs.append(a.quantize.remote(cfg.block_size, cfg.bits, cfg.method))

        quants: List[np.ndarray] = ray.get(refs)
        dt = time.perf_counter() - t0

        # Aggregate and verify
        out = np.concatenate(quants)
        if cfg.verify:
            expected = np.square(data.astype(np.float32))
            mse = float(np.mean((expected - out) ** 2))
        else:
            mse = float("nan")

        # Stats log for xltop compatibility (inst/avg)
        if cfg.stats_log:
            p = Path(cfg.stats_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                f.write("step,dt_ms,cum_ms,inst_tok_s,avg_tok_s\n")
                blocks_total = sum(((e - s) + cfg.block_size - 1) // cfg.block_size for (_, s, e, _) in actors)
                bps = blocks_total / max(dt, 1e-9)
                f.write(f"1,{dt*1000.0:.3f},{dt*1000.0:.3f},{bps:.3f},{bps:.3f}\n")

        print(f"Done: shards={cfg.num_shards} blocks={sum(((e - s) + cfg.block_size - 1) // cfg.block_size for (_, s, e, _) in actors)} time={dt:.3f}s mse={mse:.6e}")
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
