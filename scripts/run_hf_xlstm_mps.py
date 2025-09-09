
"""
Run NX-AI/xLSTM-7b via Hugging Face transformers on Apple Silicon (MPS).

Two modes:
- Native kernels (no Triton): sets step/sequence/chunkwise to native.
- Optional: if you want to try our Metal kernels, ensure this repo is on PYTHONPATH
  before importing transformers so their xlstm module resolves our mlstm_kernels.
"""
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    assert torch.backends.mps.is_available(), "MPS not available; this script targets Apple Silicon GPUs"

    model_id = os.environ.get("XLSTM_MODEL_ID", "NX-AI/xLSTM-7b")

    print(f"Loading config for {model_id} ...")
    cfg = AutoConfig.from_pretrained(model_id)

    # If we can't or don't want to use Triton on MPS, stick to native kernels.
    # This aligns with official guidance when Triton is unavailable.
    cfg.step_kernel = "native"
    cfg.chunkwise_kernel = "chunkwise--native_autograd"
    cfg.sequence_kernel = "native_sequence__native"

    print("Loading model (this may take a while)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        device_map="auto",  # HF will place on MPS if configured; we ensure below
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Ensure MPS execution
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

    print("Generating ...")
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

