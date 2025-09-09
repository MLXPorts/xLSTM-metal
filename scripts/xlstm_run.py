
"""
Unified runner for xLSTM on Apple MPS.

Examples:
  - Local checkpoint:
    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
      python scripts/xlstm_run.py --local /path/to/xlstm_7b_model \
      --prompt "The capital of France is" --new 64 --workers 6 --heads 4

  - Hugging Face model id:
    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
      XLSTM_MODEL_ID=NX-AI/xLSTM-7b \
      python scripts/xlstm_run.py --hf --prompt "…" --new 64
"""
import argparse
import os
import sys
from pathlib import Path
import time

import torch


def run_local(model_path: str, prompt: str, new_tokens: int, workers: int, heads: int):
    from xlstm_official_full.xlstm_large.from_pretrained import load_from_pretrained
    from transformers import AutoTokenizer

    m = load_from_pretrained(
        model_path,
        backend_mode="inference",
        return_last_states=True,
        # Apple defaults kick in internally
    ).to("mps").eval()

    tok = AutoTokenizer.from_pretrained(model_path)
    x = tok(prompt, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)

    os.environ["XLSTM_MPS_WORKERS"] = str(workers)
    os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(heads)

    t0 = time.time()
    with torch.no_grad():
        y, _ = m.generate(prefill_tokens=x, max_length=new_tokens, sampling_type="greedy")
    dt = time.time() - t0

    print(tok.decode(y[0], skip_special_tokens=True))
    print(f"\nTiming: {dt:.2f}s for {new_tokens} tokens")


def run_hf(model_id: str, prompt: str, new_tokens: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.step_kernel = "metal"
    cfg.sequence_kernel = "native_sequence__metal"
    cfg.chunkwise_kernel = "chunkwise--queued_compiled_steps"
    cfg.mode = "inference"
    cfg.return_last_states = True

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("mps").eval()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    x = tok(prompt, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)

    t0 = time.time()
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    print(tok.decode(y[0], skip_special_tokens=True))
    print(f"\nTiming: {dt:.2f}s for {new_tokens} tokens")


def main():
    assert torch.backends.mps.is_available(), "MPS not available; requires Apple Silicon."

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", type=str, help="Path to local HF checkpoint directory")
    g.add_argument("--hf", action="store_true", help="Use HF hub model id (env XLSTM_MODEL_ID)")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--new", type=int, default=64, help="Max new tokens")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    if args.local:
        run_local(args.local, args.prompt, args.new, args.workers, args.heads)
    else:
        mid = os.environ.get("XLSTM_MODEL_ID", "NX-AI/xLSTM-7b")
        run_hf(mid, args.prompt, args.new)


if __name__ == "__main__":
    main()

