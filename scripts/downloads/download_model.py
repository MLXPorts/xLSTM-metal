#!/usr/bin/env python
"""
Direct download of xLSTM-7B model
"""

import os
from huggingface_hub import snapshot_download

print("Downloading xLSTM-7B model from HuggingFace...")
print("This will download ~14GB of model files")
print("-" * 60)

# Download the model
local_dir = "./xlstm_7b_model"
try:
    snapshot_download(
        repo_id="NX-AI/xLSTM-7b",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,  # Resume if interrupted
        max_workers=4  # Parallel downloads
    )
    print(f"\n✓ Model downloaded to {local_dir}")
    
    # List downloaded files
    import os
    files = os.listdir(local_dir)
    print(f"\nDownloaded {len(files)} files:")
    for f in sorted(files)[:10]:
        size = os.path.getsize(os.path.join(local_dir, f)) / (1024**3)
        print(f"  {f}: {size:.2f}GB")
        
except Exception as e:
    print(f"Error: {e}")