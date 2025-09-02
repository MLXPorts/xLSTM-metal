#!/usr/bin/env python
"""
Ray Serve deployment for xLSTM inference (Apple/MPS).

Usage (requires ray[serve] installed):
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    conda run -n base python scripts/serve_xlstm.py --model_path ./xlstm_7b_model

Then send JSON:
  curl -X POST http://127.0.0.1:8000/infer -H 'Content-Type: application/json' \
       -d '{"prompt": "The capital of France is", "max_new_tokens": 16}'
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from ray import serve
except Exception as e:  # pragma: no cover
    serve = None  # type: ignore

from scripts.run_local_xlstm_mps import load_local_config, load_local_weights
from xlstm_official_full.xlstm_large.model import xLSTMLarge


def build_model(model_path: str) -> tuple[xLSTMLarge, Any]:
    mp = Path(model_path)
    mcfg = load_local_config(mp / "config.json")
    model = xLSTMLarge(mcfg).to("mps").eval()
    sd = load_local_weights(mp)
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(model_path)
    # GPU-only if supported
    try:
        torch._C._set_mps_fallback_enabled(False)  # type: ignore[attr-defined]
    except Exception:
        pass
    return model, tok


if serve is not None:
    @serve.deployment(
        ray_actor_options={"num_cpus": 1},
        autoscaling_config={"min_replicas": 1, "max_replicas": 2, "target_num_ongoing_requests_per_replica": 8},
    )
    class XLSTMService:
        def __init__(self, model_path: str):
            os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
            self.model, self.tok = build_model(model_path)

        @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.01)  # type: ignore[attr-defined]
        async def handle_batch(self, requests):
            # Parse JSON bodies in parallel
            try:
                import asyncio
                payloads = await asyncio.gather(*[r.json() for r in requests])
            except Exception:
                payloads = [await r.json() for r in requests]
            prompts = [p.get("prompt", "") for p in payloads]
            max_new_tokens = [int(p.get("max_new_tokens", 16)) for p in payloads]
            return [{"text": self.generate(prompts[i], max_new_tokens[i])} for i in range(len(prompts))]

        async def __call__(self, request):
            # Route single requests through the batcher
            return await self.handle_batch([request])

        def generate(self, prompt: str, max_new: int) -> str:
            device = torch.device("mps")
            x = self.tok(prompt, return_tensors="pt")["input_ids"].to(device)
            if self.tok.bos_token_id is not None:
                bos = torch.tensor([[self.tok.bos_token_id]], device=device, dtype=x.dtype)
                x = torch.cat([bos, x], dim=1)
            with torch.no_grad():
                logits, state = self.model(x, None)
                next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
                out = [next_tok]
                for _ in range(max_new - 1):
                    logits, state = self.model(next_tok, state)
                    next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
                    out.append(next_tok)
            y = torch.cat(out, dim=1)
            return self.tok.decode(y[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()
    if serve is None:
        raise RuntimeError("ray[serve] is not installed. Install Ray Serve to use this script.")
    serve.run(XLSTMService.bind(args.model_path))  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
