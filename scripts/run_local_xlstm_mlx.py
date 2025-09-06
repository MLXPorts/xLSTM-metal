#!/usr/bin/env python
"""
Run a local xLSTM (MLX backend) on Apple Silicon with MLX/Metal kernels.

- No Ray dependency. Executes entirely with MLX on GPU if available.
- Uses a lightweight byte-level tokenizer by default to avoid large vocab.
- Optional: set XLSTM_MLX_FAST_HEAD=1 to use tiled Metal GEMM for final head.

Examples
  conda run -n base python scripts/run_local_xlstm_mlx.py \
      --prompt "The capital of France is" --max_new_tokens 32 \
      --layers 6 --model-dim 512 --head-dim 64 --heads 8

Optional (use HF tokenizer):
  conda run -n base python scripts/run_local_xlstm_mlx.py \
      --prompt "Hello" --hf-tokenizer gpt2 --vocab-size 50257
"""

import argparse
import os
import sys
from typing import Optional

import mlx.core as mx

try:
    from implementations.mlx.xlstm_mlx import create_xlstm_model
except Exception:
    # Allow running from repo root with PYTHONPATH=.
    sys.path.append(".")
    from implementations.mlx.xlstm_mlx import create_xlstm_model


class ByteTokenizer:
    """Simple byte-level tokenizer (vocab=256)."""
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes([int(x) for x in ids]).decode("utf-8", errors="ignore")


def _load_hf_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    # Ensure pad token exists for single-batch convenience
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def softmax_sample(logits: mx.array, temperature: float = 1.0, top_k: int = 0) -> int:
    x = logits.astype(mx.float32)
    if temperature != 1.0:
        x = x / max(1e-6, float(temperature))
    if top_k and top_k > 0 and top_k < int(x.shape[-1]):
        # Mask all but top_k
        kth = mx.topk(x, k=top_k, axis=-1)
        thresh = kth.values[..., -1]
        mask = x < mx.expand_dims(thresh, -1)
        x = mx.where(mask, -1e30, x)
    x = x - mx.max(x, axis=-1, keepdims=True)
    p = mx.exp(x)
    p = p / mx.sum(p, axis=-1, keepdims=True)
    # Sample from categorical
    u = mx.random.uniform(shape=p.shape, low=0.0, high=1.0, dtype=p.dtype)
    c = mx.cumsum(p, axis=-1)
    idx = mx.argmax(u <= c, axis=-1)
    return int(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="Hello, world!")
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)

    # Model dims
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--model-dim", type=int, default=512)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--signature", type=str, default="1,1", help="mLSTM,sLSTM pattern (e.g., 1,1)")
    ap.add_argument("--dropout", type=float, default=0.0)

    # Tokenizer
    ap.add_argument("--hf-tokenizer", type=str, default=None, help="Optional HF tokenizer name (e.g., gpt2)")
    # Runtime config (no envs)
    ap.add_argument("--gemm-pad", type=int, default=None)
    ap.add_argument("--gemm-align-execw", type=int, default=None)
    ap.add_argument("--gemm-double-buffer", type=int, default=None)
    ap.add_argument("--qr-dot-mode", type=str, default=None, choices=["auto","simd","simple"]) 
    # Benchmarking / stats
    ap.add_argument("--stats-log", type=str, default=None, help="Optional CSV file to log per-step decode timing (step,elapsed_s,tok_s)")
    ap.add_argument("--stats-every", type=int, default=1, help="Log every N decode steps (default 1)")
    args = ap.parse_args()

    # Apply runtime config
    try:
        from tools.mlx_runtime import configure_gemm, configure_qr
        configure_gemm(pad=bool(args.gemm_pad) if args.gemm_pad is not None else None,
                       align_execw=bool(args.gemm_align_execw) if args.gemm_align_execw is not None else None,
                       double_buffer=bool(args.gemm_double_buffer) if args.gemm_double_buffer is not None else None)
        if args.qr_dot_mode is not None:
            configure_qr(dot_mode=args.qr_dot_mode)
    except Exception:
        pass

    # Resolve prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

    # Tokenizer
    tok = None
    if args.hf_tokenizer:
        tok = _load_hf_tokenizer(args.hf_tokenizer)
        vocab_size = int(tok.vocab_size)
    else:
        tok = ByteTokenizer()
        vocab_size = int(args.vocab_size)

    # Model
    sig = tuple(int(x) for x in args.signature.split(","))
    model = create_xlstm_model(
        vocab_size=vocab_size,
        num_layers=int(args.layers),
        signature=sig,  # pattern cycles across layers
        inp_dim=int(args.model_dim),
        head_dim=int(args.head_dim),
        head_num=int(args.heads),
        dropout=float(args.dropout),
    )

    # Encode prompt
    if args.hf_tokenizer:
        ids = tok.encode(prompt, return_tensors=None)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        ids = tok.encode(prompt)
    ids = ids[:4096]  # keep things bounded
    input_ids = mx.array([ids], dtype=mx.int32)

    # Streams: route compute to a dedicated GPU stream when available
    use_streams = True
    try:
        s_gpu = mx.new_stream(mx.gpu)
    except Exception:
        s_gpu = None
        use_streams = False

    # Prefill on compute stream
    import time, csv
    t_prefill_start = time.time()
    if use_streams:
        with mx.stream(s_gpu):
            logits, state = model(input_ids, return_hidden=True)
            last_logits = logits[:, -1, :]
    else:
        logits, state = model(input_ids, return_hidden=True)
        last_logits = logits[:, -1, :]
    t_prefill = time.time() - t_prefill_start

    # Decode
    out_ids = list(ids)
    # Stats setup
    stats_path = args.stats_log
    stats_every = max(1, int(args.stats_every))
    csv_f = None; csv_w = None
    if stats_path:
        csv_f = open(stats_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["step", "elapsed_s", "tok_s"])  # header

    t_decode_start = time.time()
    for step in range(1, int(args.max_new_tokens) + 1):
        t_step_start = time.time()
        if use_streams:
            with mx.stream(s_gpu):
                next_id = softmax_sample(last_logits[0], args.temperature, args.top_k)
                out_ids.append(int(next_id))  # host read is the sync point per step
                step_in = mx.array([[int(next_id)]], dtype=mx.int32)
                logits, state = model(step_in, hidden_states=state, return_hidden=True)
                last_logits = logits[:, -1, :]
        else:
            next_id = softmax_sample(last_logits[0], args.temperature, args.top_k)
            out_ids.append(int(next_id))
            step_in = mx.array([[int(next_id)]], dtype=mx.int32)
            logits, state = model(step_in, hidden_states=state, return_hidden=True)
            last_logits = logits[:, -1, :]
        # Per-step timing (host-side sync via sampling)
        if (step % stats_every) == 0 and (csv_w is not None):
            elapsed = time.time() - t_step_start
            tok_s = 1.0 / max(1e-9, elapsed)
            csv_w.writerow([step, f"{elapsed:.6f}", f"{tok_s:.2f}"])

    # Decode and print
    # Synchronize compute stream before host decode
    if use_streams:
        mx.synchronize(s_gpu)

    t_decode = time.time() - t_decode_start
    total_new = int(args.max_new_tokens)
    overall_tok_s = (total_new / max(1e-9, t_decode)) if total_new > 0 else 0.0
    if csv_f is not None:
        csv_f.flush(); csv_f.close()

    if args.hf_tokenizer:
        try:
            text = tok.decode(out_ids)
        except Exception:
            text = str(out_ids)
    else:
        text = tok.decode(out_ids)
    print(text)
    # Print benchmark summary
    print("\n--- Benchmark Summary (MLX) ---")
    print(f"Prefill time: {t_prefill*1000:.1f} ms")
    print(f"Decode tokens: {total_new}")
    print(f"Decode time: {t_decode:.3f} s")
    print(f"Decode throughput: {overall_tok_s:.1f} tok/s")


if __name__ == "__main__":
    # Enable fast tiled GEMM head by default (validated)
    os.environ.setdefault("XLSTM_MLX_FAST_HEAD", "1")
    main()
