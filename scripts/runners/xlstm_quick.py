#!/usr/bin/env python
"""
Minimal, batteries‑included runner for local xLSTM on Apple MPS.

Goals
- Zero required tuning flags; sensible defaults.
- Ray backend by default (in‑process/local_mode), compiled MPS step/sequence.
- Auto‑discover local HF checkpoint (./xlstm_7b_model) unless --model_path is given.

Usage
  PYTHONPATH=. python scripts/runners/xlstm_quick.py \
    --prompt "The capital of France is" --new 16

Notes
- Uses the same JSON‑first config loader as run_local_xlstm_mps.py.
"""
import os
import sys
import time
from pathlib import Path
import argparse

import torch
from transformers import AutoTokenizer

from scripts.run_local_xlstm_mps import load_local_config
from xlstm_official_full.xlstm_large.model import xLSTMLarge


def find_model_dir(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not (p.is_dir() and (p / "config.json").exists()):
            raise FileNotFoundError(f"Model path invalid: {p}")
        return p
    # Heuristics: ./xlstm_7b_model, then any immediate subdir with config.json
    d = Path.cwd() / "xlstm_7b_model"
    if d.is_dir() and (d / "config.json").exists():
        return d
    for c in Path.cwd().iterdir():
        if c.is_dir() and (c / "config.json").exists():
            return c
    raise FileNotFoundError(
        "No local HF checkpoint found. Place a model dir with config.json (e.g., ./xlstm_7b_model) or pass --model_path."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="Path to HF checkpoint (config.json + model-*.safetensors)")
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text; if omitted, will prompt interactively")
    ap.add_argument("--prompt-file", type=str, default=None, help="Path to a prompt text file (long context)")
    ap.add_argument("--new", "--max_new_tokens", type=int, default=16, dest="max_new_tokens")
    ap.add_argument("--stats-log", type=str, default=None, help="Optional CSV for per-step decode stats")
    ap.add_argument("--auto-best", action="store_true", help="Load last runs/mps_opt/*/best.json and apply heads_per_band/chunk_size")
    ap.add_argument("--aggressive", action="store_true", help="Use large chunk/band settings and max-autotune compile to maximize GPU use")
    ap.add_argument("--heads-per-band", type=int, default=None, help="Override heads per band (Ray/queued)")
    ap.add_argument("--chunk-size", type=int, default=None, help="Override chunk size for prefill")
    args = ap.parse_args()

    # Strong defaults for Apple MPS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    assert torch.backends.mps.is_available(), "MPS not available; requires Apple Silicon + torch with MPS."

    model_dir = find_model_dir(args.model_path)

    cfg = load_local_config(model_dir / "config.json")
    # Apply best.json if requested
    if args.auto_best:
        try:
            runs = sorted((Path("runs/mps_opt").glob("*/best.json")), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                import json
                best = json.loads(runs[0].read_text())
                hpb = best.get("heads_per_band") or best.get("hpb")
                csz = best.get("chunk_size") or best.get("ck")
                if hpb is not None:
                    os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(int(hpb))
                if csz is not None:
                    cfg.chunk_size = int(csz)
                print(f"[auto-best] Using heads_per_band={os.environ.get('XLSTM_MPS_HEADS_PER_BAND','?')}, chunk_size={cfg.chunk_size}")
        except Exception:
            pass
    # Aggressive mode: prioritize throughput over memory
    if args.aggressive:
        os.environ.setdefault("XLSTM_COMPILE_MODE", "max-autotune")
        # Prefer larger chunk and bands; respect explicit CLI overrides later
        try:
            # Use up to 8 heads per band by default (or num_heads if smaller)
            # We'll write to env only if not set by user
            import json as _json
            num_heads = cfg.num_heads if hasattr(cfg, 'num_heads') else None
            default_hpb = str(min(8, num_heads or 8))
            os.environ.setdefault("XLSTM_MPS_HEADS_PER_BAND", default_hpb)
        except Exception:
            os.environ.setdefault("XLSTM_MPS_HEADS_PER_BAND", "8")
        # Prefer larger chunk sizes for prefill
        if getattr(cfg, 'chunk_size', 0) < 64:
            cfg.chunk_size = 64

    # CLI overrides
    if args.heads_per_band is not None:
        os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(args.heads_per_band)
    if args.chunk_size is not None:
        cfg.chunk_size = int(args.chunk_size)

    torch.set_float32_matmul_precision("high")
    model = xLSTMLarge(cfg).to("mps").eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        prompt_text = args.prompt
        if prompt_text is None:
            try:
                prompt_text = input("Prompt: ")
            except EOFError:
                prompt_text = "The capital of France is"

    inputs = tok(prompt_text, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=inputs.dtype)
        inputs = torch.cat([bos, inputs], dim=1)

    stats_fp = None
    if args.stats_log:
        stats_path = Path(args.stats_log)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_fp = open(stats_path, "w")
        stats_fp.write("step,dt_ms,cum_ms,inst_tok_s,avg_tok_s\n")

    @torch.no_grad()
    def greedy_gen(prefill_tokens: torch.Tensor, max_len: int):
        state = None
        B = prefill_tokens.size(0)
        gen = torch.empty((B, max_len), dtype=torch.long, device=prefill_tokens.device)
        t0 = time.time()
        logits, state = model(prefill_tokens, state)
        t1 = time.time()
        next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
        gen[:, 0:1] = next_tok
        decode_time = 0.0
        cum = 0.0
        for i in range(1, max_len):
            td0 = time.time()
            logits, state = model(next_tok, state)
            td1 = time.time()
            dt = td1 - td0
            decode_time += dt
            cum += dt
            next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
            gen[:, i:i+1] = next_tok
            if stats_fp:
                inst = 1.0 / max(dt, 1e-9)
                avg = i / max(cum, 1e-9)
                stats_fp.write(f"{i},{dt*1000.0:.3f},{cum*1000.0:.3f},{inst:.3f},{avg:.3f}\n")
        if stats_fp:
            stats_fp.close()
        return gen, (t1 - t0), decode_time

    # One-time warmup to stabilize compile paths (full prompt)
    _ = model(inputs)
    tokens, t_prefill, t_decode = greedy_gen(inputs, args.max_new_tokens)
    text = tok.decode(tokens[0], skip_special_tokens=True)
    print("\nOutput:\n", text)
    print(f"\nTiming: total={t_prefill + t_decode:.2f}s (prefill={t_prefill:.2f}s, decode={t_decode:.2f}s)")
    print(f"Throughput: prefill={inputs.shape[1]/max(t_prefill,1e-9):.1f} tok/s, decode={(args.max_new_tokens-1)/max(t_decode,1e-9):.1f} tok/s")


if __name__ == "__main__":
    main()
