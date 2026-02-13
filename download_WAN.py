#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download

# Target folder
target_dir = "./wan_models/Wan2.1-T2V-1.3B"
os.makedirs(target_dir, exist_ok=True)

# Repo ID
repo_id = "Wan-AI/Wan2.1-T2V-1.3B"

# Files to fetch
files = [
    "Wan2.1_VAE.pth",
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
]

for filename in files:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_dir)
    print(f"Downloaded {filename} → {local_path}")

print("✅ All requested files are in", target_dir)
