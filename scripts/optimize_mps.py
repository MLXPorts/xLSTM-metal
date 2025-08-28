#!/usr/bin/env python
"""
Optimize xLSTM MPS performance via random search or a simple genetic algorithm.

Targets Apple Silicon (MPS) and maximizes decode tokens/sec (default objective).
Works with:
  - Ray backend (chunkwise--ray_compiled_steps): tune heads_per_band, chunk_size
  - Queued backend (chunkwise--queued_compiled_steps): tune workers, heads_per_band, chunk_size

Example (GA, ray):
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. XLSTM_RAY_LOCAL_MODE=1 \
    python scripts/optimize_mps.py \
      --backend ray \
      --model_path /path/to/xlstm_7b_model \
      --prompt "The capital of France is" \
      --new 64 \
      --mode ga --generations 5 --population 10 --repeats 1

Example (random, queued):
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    python scripts/optimize_mps.py \
      --backend queued \
      --model_path /path/to/xlstm_7b_model \
      --prompt "The capital of France is" \
      --new 64 \
      --mode random --trials 20 --repeats 1
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer

from xlstm_official_full.xlstm_large.model import xLSTMLarge
from scripts.run_local_xlstm_mps import load_local_config, load_local_weights


@torch.no_grad()
def greedy_gen_timed(model: xLSTMLarge, prefill_tokens: torch.Tensor, max_len: int) -> Tuple[float, float]:
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


def set_chunk_size(model: xLSTMLarge, chunk_size: int) -> None:
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


def make_input(tok: AutoTokenizer, prompt: str) -> torch.Tensor:
    x = tok(prompt, return_tensors="pt")["input_ids"].to("mps")
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], device="mps", dtype=x.dtype)
        x = torch.cat([bos, x], dim=1)
    return x


def eval_params(
    model: xLSTMLarge,
    tok: AutoTokenizer,
    prompt: str,
    new_tokens: int,
    backend: str,
    params: Dict[str, int],
    repeats: int,
) -> Dict[str, float]:
    # Apply env knobs based on backend
    if backend == "ray":
        os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(params["heads_per_band"])
        set_chunk_size(model, params["chunk_size"])
        os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
    elif backend == "queued":
        os.environ["XLSTM_MPS_WORKERS"] = str(params["workers"])
        os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(params["heads_per_band"])
        os.environ["XLSTM_MPS_STREAMS"] = "0"
        set_chunk_size(model, params["chunk_size"])

    x = make_input(tok, prompt)
    # Warmup
    _ = greedy_gen_timed(model, x, 1)
    pref_times, dec_times = [], []
    for _ in range(repeats):
        pt, dt = greedy_gen_timed(model, x, new_tokens)
        pref_times.append(pt)
        dec_times.append(dt)
    pt = sum(pref_times) / len(pref_times)
    dt = sum(dec_times) / len(dec_times)
    prompt_len = x.shape[1]
    prefill_tps = prompt_len / max(pt, 1e-9)
    decode_tps = max(new_tokens - 1, 1) / max(dt, 1e-9)
    return {"prefill_tok_s": prefill_tps, "decode_tok_s": decode_tps, "prefill_s": pt, "decode_s": dt}


@dataclass
class Bounds:
    heads_min: int
    heads_max: int
    chunks_min: int
    chunks_max: int
    workers_min: int | None = None
    workers_max: int | None = None


def random_params(backend: str, b: Bounds) -> Dict[str, int]:
    p = {
        "heads_per_band": random.randint(b.heads_min, b.heads_max),
        "chunk_size": random.choice([b.chunks_min, (b.chunks_min + b.chunks_max)//2, b.chunks_max])
    }
    if backend == "queued":
        assert b.workers_min is not None and b.workers_max is not None
        p["workers"] = random.randint(b.workers_min, b.workers_max)
    return p


def mutate(backend: str, p: Dict[str, int], b: Bounds, mut_prob: float) -> Dict[str, int]:
    q = dict(p)
    if random.random() < mut_prob:
        q["heads_per_band"] = max(b.heads_min, min(b.heads_max, q["heads_per_band"] + random.choice([-2, -1, 1, 2])))
    if random.random() < mut_prob:
        step = max(1, (b.chunks_max - b.chunks_min)//4)
        q["chunk_size"] = max(b.chunks_min, min(b.chunks_max, q["chunk_size"] + random.choice([-step, step])))
    if backend == "queued" and random.random() < mut_prob:
        q["workers"] = max(b.workers_min, min(b.workers_max, q["workers"] + random.choice([-1, 1])))  # type: ignore[arg-type]
    return q


def crossover(backend: str, a: Dict[str, int], c: Dict[str, int]) -> Dict[str, int]:
    r = {
        "heads_per_band": random.choice([a["heads_per_band"], c["heads_per_band"]]),
        "chunk_size": random.choice([a["chunk_size"], c["chunk_size"]]),
    }
    if backend == "queued":
        r["workers"] = random.choice([a["workers"], c["workers"]])
    return r


def optimize_ga(
    backend: str,
    model: xLSTMLarge,
    tok: AutoTokenizer,
    prompt: str,
    new_tokens: int,
    bounds: Bounds,
    generations: int,
    population: int,
    repeats: int,
    elite_frac: float = 0.3,
    mut_prob: float = 0.4,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    pop = [random_params(backend, bounds) for _ in range(population)]
    best_p, best_m = None, None
    for g in range(generations):
        scored = []
        for p in pop:
            m = eval_params(model, tok, prompt, new_tokens, backend, p, repeats)
            scored.append((p, m))
        scored.sort(key=lambda pm: pm[1]["decode_tok_s"], reverse=True)
        if best_m is None or scored[0][1]["decode_tok_s"] > best_m["decode_tok_s"]:
            best_p, best_m = scored[0]
        # Elitism
        keep = max(1, int(elite_frac * population))
        elites = [p for p, _ in scored[:keep]]
        # Breed new population
        new_pop = elites.copy()
        while len(new_pop) < population:
            a, c = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = crossover(backend, a, c)
            child = mutate(backend, child, bounds, mut_prob)
            new_pop.append(child)
        pop = new_pop
        print(f"[gen {g+1}/{generations}] best decode={best_m['decode_tok_s']:.2f} tok/s params={best_p}")
    return best_p, best_m  # type: ignore[return-value]


def optimize_random(
    backend: str,
    model: xLSTMLarge,
    tok: AutoTokenizer,
    prompt: str,
    new_tokens: int,
    bounds: Bounds,
    trials: int,
    repeats: int,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    best_p, best_m = None, None
    for t in range(trials):
        p = random_params(backend, bounds)
        m = eval_params(model, tok, prompt, new_tokens, backend, p, repeats)
        if best_m is None or m["decode_tok_s"] > best_m["decode_tok_s"]:
            best_p, best_m = p, m
        print(f"[trial {t+1}/{trials}] decode={m['decode_tok_s']:.2f} tok/s params={p}")
    return best_p, best_m  # type: ignore[return-value]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, choices=["ray", "queued"], required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--mode", type=str, choices=["ga", "random"], default="ga")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1)
    # Bounds
    ap.add_argument("--heads-min", type=int, default=2)
    ap.add_argument("--heads-max", type=int, default=8)
    ap.add_argument("--chunks-min", type=int, default=16)
    ap.add_argument("--chunks-max", type=int, default=64)
    ap.add_argument("--workers-min", type=int, default=4)
    ap.add_argument("--workers-max", type=int, default=8)
    args = ap.parse_args()

    assert torch.backends.mps.is_available(), "MPS not available"
    if args.backend == "ray":
        os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
        chunkwise_key = "ray_compiled_steps"
    else:
        chunkwise_key = "queued_compiled_steps"

    model, tok = build_model(args.model_path, chunkwise_key, args.chunks_min)

    b = Bounds(
        heads_min=args.heads_min, heads_max=args.heads_max,
        chunks_min=args.chunks_min, chunks_max=args.chunks_max,
        workers_min=(args.workers_min if args.backend == "queued" else None),
        workers_max=(args.workers_max if args.backend == "queued" else None),
    )

    if args.mode == "ga":
        best_p, best_m = optimize_ga(
            args.backend, model, tok, args.prompt, args.new, b,
            generations=args.generations, population=args.population, repeats=args.repeats,
        )
    else:
        best_p, best_m = optimize_random(
            args.backend, model, tok, args.prompt, args.new, b,
            trials=args.trials, repeats=args.repeats,
        )

    print("\n=== Best Configuration ===")
    print(best_p)
    print(best_m)


if __name__ == "__main__":
    main()

