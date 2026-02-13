#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Repo and subfolder
repo_id = "tianweiy/CausVid"
subfolder = "bidirectional_checkpoint2"

# Where to save
target_dir = "./models"

# Download everything in that subfolder
snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    allow_patterns=[f"{subfolder}/*"],
    local_dir_use_symlinks=False,
)

print("✅ Downloaded all files from", repo_id, "/", subfolder, "→", target_dir)
