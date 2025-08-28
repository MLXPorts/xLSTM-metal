#!/usr/bin/env python
"""
Run a local xLSTM HF checkpoint on Apple Silicon (MPS) using the compiled backends.

- Loads HF-style sharded safetensors from a local directory (config.json + model-*.safetensors).
- Instantiates xlstm_official_full xLSTMLarge with compiled kernels on MPS.
- Maps known key differences (backbone.embeddings.weight -> embedding.weight).
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from xlstm_official_full.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig


def load_local_config(config_path: Path) -> xLSTMLargeConfig:
    cfg = json.loads(config_path.read_text())
    # Construct our model config
    mcfg = xLSTMLargeConfig(
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        vocab_size=cfg["vocab_size"],
        use_bias=cfg.get("use_bias", False),
        norm_eps=cfg.get("norm_eps", 1e-6),
        norm_reduction_force_float32=cfg.get("norm_reduction_force_float32", True),
        add_out_norm=cfg.get("add_out_norm", True),
        qk_dim_factor=cfg.get("qk_dim_factor", 0.5),
        v_dim_factor=cfg.get("v_dim_factor", 1.0),
        gate_soft_cap=cfg.get("gate_soft_cap", 15.0),
        output_logit_soft_cap=cfg.get("output_logit_soft_cap", 30.0),
        weight_mode=cfg.get("weight_mode", "single"),
        # Backend defaults will be overridden below for MPS
    )
    # Apple defaults for compiled kernels
    if torch.backends.mps.is_available():
        # Use compiled step+sequence and queued compiled-steps for chunkwise
        mcfg.chunkwise_kernel = "chunkwise--queued_compiled_steps"
        mcfg.sequence_kernel = "native_sequence__metal"
        mcfg.step_kernel = "metal"
        mcfg.mode = "inference"
        mcfg.return_last_states = True
        mcfg.autocast_kernel_dtype = "bfloat16"
        mcfg.inference_state_dtype = "float32"
    return mcfg


def load_local_weights(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text())
    files = sorted({f for f in idx["weight_map"].values()})
    state = {}
    for f in files:
        fp = model_dir / f
        with safe_open(str(fp), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)
    # Key mapping
    if "backbone.embeddings.weight" in state:
        state["embedding.weight"] = state.pop("backbone.embeddings.weight")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./xlstm_7b_model")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6, help="CPU coordinator threads (MPS queued backend)")
    ap.add_argument("--heads-per-band", type=int, default=4, help="Heads per task (MPS queued backend)")
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    assert model_dir.is_dir(), f"Not a directory: {model_dir}"
    assert torch.backends.mps.is_available(), "MPS not available; requires Apple Silicon."

    # Load config and model
    mcfg = load_local_config(model_dir / "config.json")
    model = xLSTMLarge(mcfg)
    # Load weights
    print("Loading weights ...")
    sd = load_local_weights(model_dir)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (showing up to 10):")
        for k in missing[:10]:
            print("  ", k)
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} (showing up to 10):")
        for k in unexpected[:10]:
            print("  ", k)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Device setup
    device = torch.device("mps")
    model = model.to(device).eval()

    # Optional: enforce GPU-only if available in this PyTorch build
    try:
        torch._C._set_mps_fallback_enabled(False)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Prepare input
    inputs = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=inputs.dtype)
        inputs = torch.cat([bos, inputs], dim=1)

    # Set queued backend tuning if selected
    os.environ["XLSTM_MPS_WORKERS"] = str(args.workers)
    os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(args.heads_per_band)

    print(f"Generating (workers={args.workers}, heads_per_band={args.heads_per_band}) ...")
    import time
    t0 = time.time()
    with torch.no_grad():
        tokens, _ = model.generate(
            prefill_tokens=inputs,
            max_length=args.max_new_tokens,
            sampling_type="greedy",
        )
    dt = time.time() - t0
    print("\nOutput:")
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(text)
    print(f"\nTiming: {dt:.2f}s total for {args.max_new_tokens} new tokens")


if __name__ == "__main__":
    main()
