#!/usr/bin/env python3
"""
Download xLSTM-7B model from HuggingFace Hub

Simple script to download the official xLSTM-7B model weights.
No PyTorch required - downloads raw safetensors files.

Usage:
    python scripts/downloads/download_model.py
    python scripts/downloads/download_model.py --output ./my_model_dir
"""

import argparse
import sys
from pathlib import Path


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description="Download xLSTM-7B model from HuggingFace")
    parser.add_argument(
        "--output", "-o",
        default="./xlstm_7b_model",
        help="Output directory for model files (default: ./xlstm_7b_model)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install it with: pip install huggingface_hub")
        return 1

    print("=" * 70)
    print("xLSTM-7B Model Download")
    print("=" * 70)
    print(f"\nModel: NX-AI/xLSTM-7b")
    print(f"Output: {args.output}")
    print(f"Size: ~14GB (6 safetensor files + config)")
    print("\nThis may take several minutes depending on your connection speed...")
    print("-" * 70)

    try:
        local_dir = Path(args.output)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id="NX-AI/xLSTM-7b",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=args.resume or True,
            max_workers=4
        )
        
        print("\n" + "=" * 70)
        print("✓ Download Complete!")
        print("=" * 70)
        
        # List downloaded files
        files = sorted(local_dir.iterdir())
        print(f"\nDownloaded {len(files)} files to {local_dir}:")
        
        total_size = 0
        for f in files:
            if f.is_file():
                size_gb = f.stat().st_size / (1024**3)
                total_size += size_gb
                print(f"  {f.name:<40} {size_gb:>6.2f} GB")
        
        print(f"\nTotal size: {total_size:.2f} GB")
        print(f"\nYou can now run inference with:")
        print(f"  python generate.py --model {local_dir} --prompt \"Hello world\"")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Run with --resume to continue.")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())