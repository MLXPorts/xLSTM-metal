#!/usr/bin/env python
"""
Run NX-AI/xLSTM-7b via Hugging Face transformers on Apple Silicon (MPS) using our Metal kernels.

Important:
- Ensure this repository path is on sys.path before transformers imports so that
  imports of `mlstm_kernels.*` resolve to our copy (with Metal kernels) rather than
  the one shipped with the site-packages xlstm.

Setup (one-time):
  pip install xlstm accelerate 'transformers @ git+https://github.com/huggingface/transformers.git@main'

Usage:
  python scripts/run_hf_xlstm_metal.py
  XLSTM_MODEL_ID=NX-AI/xLSTM-7b XLSTM_PROMPT="Your prompt" python scripts/run_hf_xlstm_metal.py
"""
import os
import sys
import time
from pathlib import Path

# Prepend repo root to sys.path to shadow site-packages mlstm_kernels
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    assert torch.backends.mps.is_available(), "MPS not available; Metal kernels require Apple Silicon GPUs"

    model_id = os.environ.get("XLSTM_MODEL_ID", "NX-AI/xLSTM-7b")

    print(f"Loading config for {model_id} ...")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Select our Metal kernels
    cfg.step_kernel = "metal"
    # Use a parallel native kernel for chunkwise prefill for stability and speed on MPS
    cfg.chunkwise_kernel = "chunkwise--native_compiled_autograd"
    cfg.sequence_kernel = "native_sequence__metal"      # sequence loop that calls Metal step
    cfg.mode = "inference"
    cfg.return_last_states = True

    print("Loading model (this may take a while)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("mps").eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    prompt = os.environ.get("XLSTM_PROMPT", "Explain quantum computing in simple terms.")
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("mps")
    # Prepend BOS if available
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device="mps", dtype=inputs.dtype)
        inputs = torch.cat([bos, inputs], dim=1)

    print("Generating (Metal kernels) ...")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nOutput:")
    print(text)


if __name__ == "__main__":
    main()
