
"""
Download xLSTM-7B model with better resume capability
"""

import os
from huggingface_hub import hf_hub_download
from pathlib import Path

model_id = "NX-AI/xLSTM-7b"
local_dir = "./xlstm_7b_model"

# List of files we need
required_files = [
    "model-00001-of-00006.safetensors",
    "model-00002-of-00006.safetensors", 
    "model-00003-of-00006.safetensors",
    "model-00004-of-00006.safetensors",
    "model-00005-of-00006.safetensors",
    "model-00006-of-00006.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]

print("Downloading xLSTM-7B model files...")
print("=" * 60)

# Check what we already have
existing_files = set()
for file in required_files:
    if os.path.exists(os.path.join(local_dir, file)):
        size_mb = os.path.getsize(os.path.join(local_dir, file)) / (1024*1024)
        print(f"✓ Already have: {file} ({size_mb:.1f} MB)")
        existing_files.add(file)

# Download missing files
missing_files = [f for f in required_files if f not in existing_files]
if missing_files:
    print(f"\nNeed to download {len(missing_files)} files:")
    for file in missing_files:
        print(f"  - {file}")
    
    print("\nDownloading...")
    for i, file in enumerate(missing_files, 1):
        print(f"\n[{i}/{len(missing_files)}] Downloading {file}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=file,
                local_dir=local_dir,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            size_mb = os.path.getsize(downloaded_path) / (1024*1024)
            print(f"  ✓ Downloaded: {file} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Error downloading {file}: {e}")
            print("  Will continue with other files...")
else:
    print("\n✓ All files already downloaded!")

# Final check
print("\n" + "=" * 60)
print("Final status:")
for file in required_files[:6]:  # Just check safetensor files
    path = os.path.join(local_dir, file)
    if os.path.exists(path):
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  ✓ {file}: {size_gb:.2f} GB")
    else:
        print(f"  ✗ {file}: MISSING")

total_size = sum(
    os.path.getsize(os.path.join(local_dir, f)) 
    for f in os.listdir(local_dir) 
    if f.endswith('.safetensors')
) / (1024**3)
print(f"\nTotal safetensors size: {total_size:.1f} GB")