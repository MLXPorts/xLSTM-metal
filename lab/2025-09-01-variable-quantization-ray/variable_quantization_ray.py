"""An experiment for exploring variable quantization with Ray."""

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
    """Configuration for the variable quantization experiment."""
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
    """Loads the configuration for the variable quantization experiment from a JSON file."""
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
    """Creates a Ray runtime environment from the current environment variables."""
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
    """A no-op gauge for when Ray is not installed."""
    def set(self, *a, **k):
        return None


def _make_gauges(prefix: str = "varquant"):
    """Creates a dictionary of Ray gauges for monitoring the experiment."""
    return {"rss_mb": _NoopGauge()} if Gauge is None else {"rss_mb": Gauge("rss_mb")}


def _rss_mb() -> float:
    """Returns the resident set size (RSS) of the current process in megabytes."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return 0.0


def _quantize_block(block: np.ndarray, bits: int, method: str) -> np.ndarray:
    """Quantizes a block of data using the specified method."""
    if method == "quantile":
        num_levels = 2 ** bits
        sorted_data = np.sort(block)
        boundaries = np.percentile(sorted_data, np.linspace(0, 100, num_levels + 1)[1:-1])
        indices = np.digitize(block, boundaries)
        return indices.astype(np.uint8)
    else:
        # floating_point: naive cast to reduced precision emulation
        scale = np.float32(2 ** (bits - 1) - 1)
        clipped = np.clip(block, -1.0, 1.0)
        return (np.round(clipped * scale) / scale).astype(block.dtype)


def main():
    cfg = Config(
        size=11_000_000, dtype="float32", num_shards=8, block_size=4096, bits=4,
        method="floating_point", verify=True, seed=42, stats_log=None,
        use_async=True, use_compiled_graph=True, ray_local_mode=1,
    )
    print("Variable quantization experiment config:", cfg)
    rng = np.random.default_rng(cfg.seed)
    data = rng.standard_normal(cfg.size, dtype=np.float32)
    t0 = time.time()
    out = _quantize_block(data[: cfg.block_size], cfg.bits, cfg.method)
    dt = time.time() - t0
    print(f"Quantized one block of {cfg.block_size} elements in {dt*1e3:.2f} ms; RSS≈{_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()

