#!/usr/bin/env python
"""
Capture GPU trace during fault for Xcode inspection
"""

import mlx.core as mx
from mlx.core import metal
from transformers import AutoTokenizer
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner

def main():
    print("="*80)
    print("GPU Capture Test - Creating .gputrace for Xcode inspection")
    print("="*80)

    # 1. Load model
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
    print("   ✓ Tokenizer loaded")

    print("\n2. Creating xLSTM-7B runner...")
    runner = xLSTM7BRunner(
        embedding_dim=4096,
        num_heads=8,
        num_blocks=32,
        vocab_size=50304,
        output_logit_soft_cap=30.0
    )
    print("   ✓ Runner created")

    print("\n3. Loading weights...")
    runner.load_weights("xlstm_7b_model")
    print("   ✓ Weights loaded")

    # 2. Prepare prompt
    prompt = "Hello"
    print(f"\n4. Preparing prompt: '{prompt}'")
    input_ids = tokenizer.encode(prompt)
    print(f"   Input IDs: {input_ids} ({len(input_ids)} tokens)")

    # 3. Start GPU capture
    capture_path = "/tmp/xlstm_gpu_fault.gputrace"
    print(f"\n5. Starting GPU capture to: {capture_path}")
    metal.start_capture(capture_path)
    print("   ✓ GPU capture started")

    # 4. Run generation until fault (or max 10 tokens)
    print("\n6. Running generation (will capture GPU fault)...")
    try:
        output_ids = runner.generate(
            input_ids=[input_ids],
            max_tokens=10,
            temperature=0.0,
        )
        print(f"   ✓ Generated: {output_ids}")
        print("   ✓ No fault occurred!")
    except Exception as e:
        print(f"   ✗ GPU fault occurred: {e}")
    finally:
        # 5. Stop capture
        print("\n7. Stopping GPU capture...")
        metal.stop_capture()
        print("   ✓ GPU capture stopped")
        print(f"\n   Open in Xcode: {capture_path}")
        print(f"   Command: open {capture_path}")

    print("\n" + "="*80)
    print("GPU capture complete!")
    print("="*80)

if __name__ == "__main__":
    main()
