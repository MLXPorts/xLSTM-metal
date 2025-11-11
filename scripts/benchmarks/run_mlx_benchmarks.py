"""
MLX xLSTM Benchmarks

Runs prefill/decode benchmarks for the MLX implementation across realistic
profiles and tile settings, and writes CSVs (and optional PNG charts) under
`runs/benchmarks/mlx/<timestamp>/`.

Usage (examples)
  PYTHONPATH=. python scripts/benchmarks/run_mlx_benchmarks.py --profiles medium large \
    --tiles "16x16,32x8,8x32" --seq-len 2048 --new-tokens 256 --repeats 3 \
    --outdir runs/benchmarks/mlx

  # Generate charts if matplotlib is available
  PYTHONPATH=. python scripts/benchmarks/run_mlx_benchmarks.py --profiles medium \
    --make-charts 1

Notes
- Set XLSTM_MLX_FAST_HEAD=1 to benchmark the tiled GEMM path for the final
  projection. This is the default; you can toggle per-run via --fast-head.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx

# Import model and kernels (no fallbacks)
from src.mlx_impl.xlstm_mlx import create_xlstm_model

from mlx_src.mlx_fast_kernels.gemm_kernels import set_gemm_tiles, get_gemm_tiles
from tools.mlx_runtime import configure_gemm, configure_qr, configure_ivf


def profiles() -> Dict[str, Dict[str, int]]:
    """Return named profiles approximating production-ish scales.

    You can override via CLI. These are intended as starting points:
    - small: quick sanity
    - medium: moderate (single-SoC friendly)
    - large: closer to production-ish compute, may take time
    """
    return {
        # quick sanity
        "small": {"layers": 6, "model_dim": 512, "head_dim": 64, "heads": 8, "vocab": 256},
        # moderate scale
        "medium": {"layers": 16, "model_dim": 1536, "head_dim": 128, "heads": 12, "vocab": 32000},
        # heavier scale (adjust to your hardware)
        "large": {"layers": 24, "model_dim": 3072, "head_dim": 128, "heads": 24, "vocab": 50257},
    }


def ensure_outdir(base: Path) -> Path:
    """

    :param base:
    :return:
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def time_prefill_and_decode(model, seq_len: int, new_tokens: int) -> Tuple[float, float, float]:
    """

    :param model:
    :param seq_len:
    :param new_tokens:
    :return:
    """
    # Synthetic byte tokens for simplicity; tokenizer choice doesn't affect compute scale much
    tokens = mx.random.randint(0, 256, (1, seq_len))
    # Prefill
    t0 = time.time()
    logits, state = model(tokens, return_hidden=True)
    mx.eval(logits)
    t1 = time.time()
    # Decode
    last_logits = logits[:, -1, :]
    t_decode_start = time.time()
    for _ in range(new_tokens):
        # sample argmax for deterministic timing
        next_id = int(mx.argmax(last_logits[0]))
        step_in = mx.array([[next_id]], dtype=mx.int32)
        logits, state = model(step_in, hidden_states=state, return_hidden=True)
        last_logits = logits[:, -1, :]
    mx.eval(last_logits)
    t_decode_end = time.time()
    return (t1 - t0), (t_decode_end - t_decode_start), (t_decode_end - t0)


def maybe_make_charts(outdir: Path, csv_path: Path):
    """

    :param outdir:
    :param csv_path:
    :return:
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found, skipping chart generation.")
        return
    # Simple chart: decode tok/s by tile for each profile
    rows: List[Dict[str, str]] = []
    import csv as _csv
    with open(csv_path, newline="") as f:
        r = _csv.DictReader(f)
        rows = [row for row in r]
    # Group by profile
    from collections import defaultdict
    by_profile = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)
    for prof, rws in by_profile.items():
        labels = []
        values = []
        for row in rws:
            labels.append(f"AV={row['tiles_av']}\nATB={row['tiles_atb']}")
            values.append(float(row["decode_tok_s"]))
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels, rotation=30, ha='right')
        plt.ylabel("Decode tok/s")
        plt.title(f"MLX xLSTM Decode Throughput — {prof}")
        plt.tight_layout()
        png = outdir / f"bench_decode_{prof}.png"
        plt.savefig(png)
        plt.close()


def main():
    """The main function of the script."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["medium"], help="Profiles to run (small, medium, large)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tiles", type=str, default="16x16,32x8,8x32")
    ap.add_argument("--fast-head", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="runs/benchmarks/mlx")
    # Runtime config (no envs)
    ap.add_argument("--gemm-pad", type=int, default=None)
    ap.add_argument("--gemm-align-execw", type=int, default=None)
    ap.add_argument("--gemm-double-buffer", type=int, default=None)
    ap.add_argument("--qr-dot-mode", type=str, default=None, choices=["auto", "simd", "simple"])
    ap.add_argument("--ivf-tpb", type=int, default=None)
    ap.add_argument("--make-charts", type=int, default=1)
    args = ap.parse_args()

    # Apply runtime config
    configure_gemm(pad=bool(args.gemm_pad) if args.gemm_pad is not None else None,
                   align_execw=bool(args.gemm_align_execw) if args.gemm_align_execw is not None else None,
                   double_buffer=bool(args.gemm_double_buffer) if args.gemm_double_buffer is not None else None)
    if args.qr_dot_mode is not None:
        configure_qr(dot_mode=args.qr_dot_mode)
    if args.ivf_tpb is not None:
        configure_ivf(tpb=int(args.ivf_tpb))

    base = Path(args.outdir)
    outdir = ensure_outdir(base)
    csv_path = outdir / "mlx_benchmarks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "profile", "layers", "model_dim", "head_dim", "heads", "vocab",
            "seq_len", "new_tokens", "tiles_av", "tiles_atb", "fast_head",
            "prefill_s", "prefill_tok_s", "decode_s", "decode_tok_s", "total_s"
        ])

        # Iterate profiles
        profs = profiles()
        for prof in args.profiles:
            if prof not in profs:
                raise ValueError(f"Unknown profile: {prof}")
            cfg = profs[prof].copy()
            layers = cfg["layers"]
            model_dim = cfg["model_dim"]
            head_dim = cfg["head_dim"]
            heads = cfg["heads"]
            vocab = cfg["vocab"]

            # Construct model
            model = create_xlstm_model(
                vocab_size=int(vocab),
                num_layers=int(layers),
                signature=(1, 1),  # pattern cycles; not critical for timing
                inp_dim=int(model_dim),
                head_dim=int(head_dim),
                head_num=int(heads),
                dropout=0.0,
            )

            # Fast head toggle
            os.environ["XLSTM_MLX_FAST_HEAD"] = "1" if args.fast_head else "0"

            # Tiles to test
            tile_list = [t.strip() for t in args.tiles.split(",") if t.strip()]
            for tile in tile_list:
                # AV uses TMxT; AT_B uses TNxTK. We map the same pair for both for comparability.
                try:
                    tm_s, t_s = tile.split("x", 1)
                    tm, t = int(tm_s), int(t_s)
                except ValueError:
                    raise ValueError(f"Invalid tile format (expected AxB with integer values): {tile}") from None
                set_gemm_tiles(av=(tm, t), atb=(t, t))
                # Warm-up
                _ = time_prefill_and_decode(model, seq_len=int(args.seq_len), new_tokens=4)
                # Repeats
                prefill_times = []
                decode_times = []
                total_times = []
                for _ in range(int(args.repeats)):
                    p_s, d_s, tot_s = time_prefill_and_decode(model, seq_len=int(args.seq_len),
                                                              new_tokens=int(args.new_tokens))
                    prefill_times.append(p_s)
                    decode_times.append(d_s)
                    total_times.append(tot_s)
                # Median
                import statistics as stats
                p = float(stats.median(prefill_times))
                d = float(stats.median(decode_times))
                tot = float(stats.median(total_times))
                prefill_tok_s = int(args.seq_len) / max(1e-9, p)
                decode_tok_s = int(args.new_tokens) / max(1e-9, d)
                w.writerow([
                    prof, layers, model_dim, head_dim, heads, vocab,
                    int(args.seq_len), int(args.new_tokens), tile, tile, int(bool(args.fast_head)),
                    f"{p:.6f}", f"{prefill_tok_s:.2f}", f"{d:.6f}", f"{decode_tok_s:.2f}", f"{tot:.6f}"
                ])

    if int(args.make_charts):
        maybe_make_charts(outdir, csv_path)
    print(f"Wrote: {csv_path}")
    if int(args.make_charts):
        print(f"Charts under: {outdir}")


if __name__ == "__main__":
    main()
