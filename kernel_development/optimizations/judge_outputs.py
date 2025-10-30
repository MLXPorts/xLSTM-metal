
"""
Judge outputs produced by the optimizer runs using the same xLSTM model as a heuristic scorer.

For each output file in an optimizer run's outputs directory, compute:
- avg_logprob of the continuation under the model conditioned on the prompt
- perplexity (exp(-avg_logprob))
- distinct-2 and distinct-3 ratios (diversity proxy)
- length of continuation (tokens)

Writes ratings.jsonl and ratings.csv next to the outputs for downstream analysis.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    python scripts/judge_outputs.py \
      --model_path /path/to/xlstm_7b_model \
      --prompt-file /path/to/long_prompt.txt \
      --outputs runs/mps_opt/<run_dir>/outputs
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Tuple
import json

import torch
from transformers import AutoTokenizer

from xlstm_official_full.xlstm_large.model import xLSTMLarge
from scripts.run_local_xlstm_mps import load_local_config, load_local_weights


def make_input(tok: AutoTokenizer, text: str) -> torch.Tensor:
    """

    :param tok:
    :param text:
    :return:
    """
    x = tok(text, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)
    return x


@torch.no_grad()
def score_continuation(
    model: xLSTMLarge,
    tok: AutoTokenizer,
    prompt_text: str,
    cont_text: str,
) -> Tuple[float, float, int]:
    """Teacher-forced scoring of continuation tokens given a prompt.

    Returns (avg_logprob, ppl, num_tokens_cont).
    """
    # Condition on prompt
    prompt_ids = make_input(tok, prompt_text)
    state = None
    logits, state = model(prompt_ids, state)

    # Tokenize continuation (no BOS prepend here)
    cont_ids = tok(cont_text, return_tensors="pt")["input_ids"].to("mps")
    # Score token by token using teacher forcing
    total_logprob = 0.0
    T = cont_ids.shape[1]
    for t in range(T):
        logits, state = model(cont_ids[:, t:t+1], state)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        token_logprob = log_probs.gather(1, cont_ids[:, t:t+1]).squeeze(1)
        total_logprob += float(token_logprob.mean().item())
    avg_logprob = total_logprob / max(T, 1)
    ppl = math.exp(-avg_logprob) if T > 0 else float("inf")
    return avg_logprob, ppl, T


def distinct_ngrams(text: str, n: int) -> float:
    """

    :param text:
    :param n:
    :return:
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def parse_params_from_name(name: str) -> dict:
    """

    :param name:
    :return:
    """
    # Example: b=ray__h=4__ck=32__w=6__i=7
    params = {}
    for part in name.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params


def main():
    """

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Optional JSON config to override CLI")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt-file", type=str, required=True)
    ap.add_argument("--outputs", type=str, required=True, help="Directory with *.txt outputs from saver")
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

    # Build model
    os.environ.setdefault("XLSTM_CHUNKWISE_BACKEND", "ray_compiled_steps")
    mcfg = load_local_config(Path(args.model_path) / "config.json")
    model = xLSTMLarge(mcfg).to("mps").eval()
    sd = load_local_weights(Path(args.model_path))
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(str(args.model_path))

    prompt_text = Path(args.prompt_file).read_text()

    out_dir = Path(args.outputs)
    files = sorted(out_dir.glob("*.txt"))
    assert files, f"No outputs found in {out_dir}"

    ratings_jsonl = out_dir / "ratings.jsonl"
    ratings_csv = out_dir / "ratings.csv"

    csvf = open(ratings_csv, "w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=[
        "file", "backend", "heads_per_band", "chunk_size", "workers",
        "avg_logprob", "ppl", "len_tokens", "distinct2", "distinct3"
    ])
    writer.writeheader()

    with open(ratings_jsonl, "w") as jf:
        for fp in files:
            name = fp.stem
            params = parse_params_from_name(name)
            text = fp.read_text()
            avg_logprob, ppl, T = score_continuation(model, tok, prompt_text, text)
            d2 = distinct_ngrams(text, 2)
            d3 = distinct_ngrams(text, 3)
            rec = {
                "file": fp.name,
                "backend": params.get("b"),
                "heads_per_band": params.get("h"),
                "chunk_size": params.get("ck"),
                "workers": params.get("w"),
                "avg_logprob": avg_logprob,
                "ppl": ppl,
                "len_tokens": T,
                "distinct2": d2,
                "distinct3": d3,
            }
            jf.write(f"{rec}\n")
            writer.writerow(rec)
            csvf.flush()
            print(f"Scored {fp.name}: ppl={ppl:.2f}, d3={d3:.3f}")

    csvf.close()


if __name__ == "__main__":
    main()
