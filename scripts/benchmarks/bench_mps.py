
"""
Benchmark xLSTM on Apple MPS across chunkwise backends and tuning sweeps.

Examples:
  # Ray backend sweep
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. XLSTM_RAY_LOCAL_MODE=1 \
    python scripts/bench_mps.py \
      --backend ray \
      --model_path /path/to/xlstm_7b_model \
      --prompt "The capital of France is" \
      --new 64 \
      --heads 2 4 6 \
      --chunks 16 32 64 \
      --repeats 1

  # Queued backend sweep
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    python scripts/bench_mps.py \
      --backend queued \
      --model_path /path/to/xlstm_7b_model \
      --prompt "The capital of France is" \
      --new 64 \
      --workers 4 6 8 \
      --heads 2 4 6 \
      --chunks 16 32 64 \
      --repeats 1
"""
import argparse
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer

from xlstm_official_full.xlstm_large.model import xLSTMLarge
from scripts.run_local_xlstm_mps import load_local_config, load_local_weights


@torch.no_grad()
def greedy_gen_timed(model: xLSTMLarge, prefill_tokens: torch.Tensor, max_len: int):
    device = prefill_tokens.device
    state = None
    B = prefill_tokens.size(0)
    gen = torch.empty((B, max_len), dtype=torch.long, device=device)

    t0 = time.time()
    logits, state = model(prefill_tokens, state)
    t1 = time.time()

    next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
    gen[:, 0:1] = next_tok

    decode_time = 0.0
    for i in range(1, max_len):
        td0 = time.time()
        logits, state = model(next_tok, state)
        td1 = time.time()
        decode_time += (td1 - td0)
        next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
        gen[:, i:i+1] = next_tok

    return (t1 - t0), decode_time


def set_chunk_size(model: xLSTMLarge, chunk_size: int):
    for blk in model.backbone.blocks:
        try:
            blk.mlstm_layer.mlstm_backend.config.chunk_size = chunk_size
        except Exception:
            pass


def build_model(model_path: str, chunkwise_backend: str, chunk_size: int) -> tuple[xLSTMLarge, AutoTokenizer]:
    model_dir = Path(model_path)
    os.environ["XLSTM_CHUNKWISE_BACKEND"] = chunkwise_backend
    mcfg = load_local_config(model_dir / "config.json")
    mcfg.chunk_size = chunk_size
    model = xLSTMLarge(mcfg).to("mps").eval()
    sd = load_local_weights(model_dir)
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, choices=["ray", "queued"], required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=[6])
    ap.add_argument("--heads", type=int, nargs="+", default=[4])
    ap.add_argument("--chunks", type=int, nargs="+", default=[32])
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    assert torch.backends.mps.is_available(), "MPS not available"
    if args.backend == "ray":
        os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
        chunkwise_key = "ray_compiled_steps"
    else:
        chunkwise_key = "queued_compiled_steps"

    # Build model once with initial chunk size
    model, tok = build_model(args.model_path, chunkwise_key, args.chunks[0])

    # Prepare input
    x = tok(args.prompt, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)

    results = []
    for ck in args.chunks:
        set_chunk_size(model, ck)
        if args.backend == "ray":
            # Sweep heads only
            for hpb in args.heads:
                os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(hpb)
                # Warm-up
                _ = greedy_gen_timed(model, x, 1)
                pref_times = []
                dec_times = []
                for _ in range(args.repeats):
                    pt, dt = greedy_gen_timed(model, x, args.new)
                    pref_times.append(pt)
                    dec_times.append(dt)
                pt = sum(pref_times) / len(pref_times)
                dt = sum(dec_times) / len(dec_times)
                prompt_len = x.shape[1]
                prefill_tps = prompt_len / max(pt, 1e-9)
                decode_tps = max(args.new - 1, 1) / max(dt, 1e-9)
                results.append({
                    "backend": args.backend,
                    "workers": None,
                    "heads_per_band": hpb,
                    "chunk_size": ck,
                    "prefill_tok_s": prefill_tps,
                    "decode_tok_s": decode_tps,
                    "prefill_s": pt,
                    "decode_s": dt,
                })
        else:
            # Queued: sweep workers x heads
            for w in args.workers:
                os.environ["XLSTM_MPS_WORKERS"] = str(w)
                for hpb in args.heads:
                    os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(hpb)
                    os.environ["XLSTM_MPS_STREAMS"] = "0"
                    # Warm-up
                    _ = greedy_gen_timed(model, x, 1)
                    pref_times = []
                    dec_times = []
                    for _ in range(args.repeats):
                        pt, dt = greedy_gen_timed(model, x, args.new)
                        pref_times.append(pt)
                        dec_times.append(dt)
                    pt = sum(pref_times) / len(pref_times)
                    dt = sum(dec_times) / len(dec_times)
                    prompt_len = x.shape[1]
                    prefill_tps = prompt_len / max(pt, 1e-9)
                    decode_tps = max(args.new - 1, 1) / max(dt, 1e-9)
                    results.append({
                        "backend": args.backend,
                        "workers": w,
                        "heads_per_band": hpb,
                        "chunk_size": ck,
                        "prefill_tok_s": prefill_tps,
                        "decode_tok_s": decode_tps,
                        "prefill_s": pt,
                        "decode_s": dt,
                    })

    results.sort(key=lambda r: r["decode_tok_s"], reverse=True)
    print("\n=== Backend Sweep Results (backend=%s) ===" % args.backend)
    for r in results:
        wstr = f"w={r['workers']} " if r['workers'] is not None else ""
        print(
            f"{wstr}hpb={r['heads_per_band']:>2} ck={r['chunk_size']:>3} "
            f"prefill={r['prefill_tok_s']:.1f} tok/s ({r['prefill_s']:.2f}s) "
            f"decode={r['decode_tok_s']:.1f} tok/s ({r['decode_s']:.2f}s)"
        )


if __name__ == "__main__":
    main()

