#!/usr/bin/env python3
import os
import shutil
from huggingface_hub import hf_hub_download

# Target folder (final location of files)
target_dir = "./wan_models/Wan2.1-T2V-1.3B"
os.makedirs(target_dir, exist_ok=True)

repo_id = "Wan-AI/Wan2.1-T2V-1.3B"

files = [
    "Wan2.1_VAE.pth",
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
]

# OPTIONAL: put HF cache somewhere else (not inside your repo)
# cache_dir = os.path.expanduser("~/.cache/huggingface")
cache_dir = None  # keep default HF cache

for filename in files:
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,   # keeps cache OUTSIDE target_dir
        resume_download=True,
    )

    out_path = os.path.join(target_dir, filename)
    shutil.copy2(cached_path, out_path)
    print(f"Downloaded {filename} → {out_path}")

print("✅ All requested files are in", target_dir)
