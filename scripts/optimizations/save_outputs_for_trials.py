
"""
Regenerate and save outputs for each trial in an optimizer run.

Reads runs/mps_opt/<run>/summary.csv and, for each row, re-runs a greedy decode
with the recorded parameters, saving the text to an outputs directory with a
filename that encodes the parameter settings.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    python scripts/save_outputs_for_trials.py \
      --run runs/mps_opt/<run_dir> \
      --model_path /path/to/xlstm_7b_model \
      --prompt-file /path/to/long_prompt.txt \
      --new 32 \
      --outputs runs/mps_opt/<run_dir>/outputs
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from xlstm_official_full.xlstm_large.model import xLSTMLarge
from scripts.run_local_xlstm_mps import load_local_config, load_local_weights


@torch.no_grad()
def greedy_gen(model: xLSTMLarge, x: torch.Tensor, max_len: int) -> torch.Tensor:
    device = x.device
    state = None
    B = x.size(0)
    gen = torch.empty((B, max_len), dtype=torch.long, device=device)

    logits, state = model(x, state)
    next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
    gen[:, 0:1] = next_tok

    for i in range(1, max_len):
        logits, state = model(next_tok, state)
        next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
        gen[:, i:i+1] = next_tok
    return gen


def set_chunk_size(model: xLSTMLarge, chunk_size: int) -> None:
    for blk in model.backbone.blocks:
        try:
            blk.mlstm_layer.mlstm_backend.config.chunk_size = int(chunk_size)
        except Exception:
            pass


def make_input(tok: AutoTokenizer, prompt: str) -> torch.Tensor:
    x = tok(prompt, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Optional JSON config to override CLI")
    ap.add_argument("--run", type=str, required=True, help="Optimizer run directory with summary.csv")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt-file", type=str, required=True)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--outputs", type=str, required=True)
    args = ap.parse_args()

    assert torch.backends.mps.is_available(), "MPS not available"

    # Load JSON config (if provided) to override args
    if args.config:
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text())
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        for sect in ("runner", "optimizer"):
            if isinstance(cfg.get(sect), dict):
                for k, v in cfg[sect].items():
                    if hasattr(args, k):
                        setattr(args, k, v)
    run_dir = Path(args.run)
    csv_path = run_dir / "summary.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build model once; we’ll adjust chunk size and env per row.
    os.environ.setdefault("XLSTM_CHUNKWISE_BACKEND", "ray_compiled_steps")
    mcfg = load_local_config(Path(args.model_path) / "config.json")
    model = xLSTMLarge(mcfg).to("mps").eval()
    sd = load_local_weights(Path(args.model_path))
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(str(args.model_path))

    prompt_text = Path(args.prompt_file).read_text()
    x = make_input(tok, prompt_text)

    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        backend = str(row.get("backend"))
        hpb = int(row.get("heads_per_band")) if not pd.isna(row.get("heads_per_band")) else None
        ck = int(row.get("chunk_size")) if not pd.isna(row.get("chunk_size")) else None
        w = int(row.get("workers")) if not pd.isna(row.get("workers")) else None

        # Set environment per row
        if backend == "ray":
            os.environ["XLSTM_CHUNKWISE_BACKEND"] = "ray_compiled_steps"
            os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
            if hpb is not None:
                os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(hpb)
        else:
            os.environ["XLSTM_CHUNKWISE_BACKEND"] = "queued_compiled_steps"
            if w is not None:
                os.environ["XLSTM_MPS_WORKERS"] = str(w)
            if hpb is not None:
                os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(hpb)
            os.environ["XLSTM_MPS_STREAMS"] = "0"
        if ck is not None:
            set_chunk_size(model, ck)

        # Generate and save
        gen = greedy_gen(model, x, args.new)
        text = tok.decode(gen[0], skip_special_tokens=True)
        parts = [f"b={backend}"]
        if hpb is not None:
            parts.append(f"h={hpb}")
        if ck is not None:
            parts.append(f"ck={ck}")
        if w is not None:
            parts.append(f"w={w}")
        parts.append(f"i={idx}")
        name = "__".join(parts)
        (outputs_dir / f"{name}.txt").write_text(text)
        print(f"Saved {name}.txt")


if __name__ == "__main__":
    main()
