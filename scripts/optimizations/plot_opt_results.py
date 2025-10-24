
"""
Plot optimization results from runs/mps_opt/<run>/summary.csv.
Usage:
  python scripts/plot_opt_results.py --run runs/mps_opt/ray_YYYYMMDD_HHMMSS_tag
Generates simple PNGs for decode tok/s vs chunk_size, heads_per_band, workers.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    csv_path = run_dir / "summary.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    # Basic plots
    for col in ["chunk_size", "heads_per_band", "workers"]:
        if col in df.columns and df[col].notna().any():
            plt.figure()
            df.groupby(col)["decode_tok_s"].max().plot(kind="bar", title=f"Max decode tok/s by {col}")
            plt.ylabel("decode tok/s (max)")
            plt.tight_layout()
            out = run_dir / f"max_decode_by_{col}.png"
            plt.savefig(out)
            plt.close()

    # Scatter of prefill vs decode
    plt.figure()
    plt.scatter(df["prefill_tok_s"], df["decode_tok_s"], alpha=0.6)
    plt.xlabel("prefill tok/s")
    plt.ylabel("decode tok/s")
    plt.title("Prefill vs Decode")
    plt.tight_layout()
    plt.savefig(run_dir / "prefill_vs_decode.png")
    plt.close()


if __name__ == "__main__":
    main()

