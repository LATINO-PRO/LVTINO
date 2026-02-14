# Official repository for **LVTINO**

> **L**Atent **V**ideo consis**T**ency **IN**verse s**O**lver for High Definition Video Restoration

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=LATINO-PRO.LVTINO)
[![GitHub stars](https://img.shields.io/github/stars/LATINO-PRO/LVTINO.svg?style=social&label=Stars)](https://github.com/<USERNAME>/<REPO>/stargazers)
---

This repository contains the official implementation of **LVTINO**, a plug-and-play / zero-shot inverse solver for **high-definition video restoration** using priors encoded by **Video Consistency Models (VCMs)**.

- [**Paper (arXiv)**](https://arxiv.org/abs/2510.01339)
- [**Web-Page**](https://latino-pro.github.io/LVTINO/)

| Observed | LVTINO |
|---|---|
| [![obs](https://github.com/LATINO-PRO/LVTINO/blob/web-page/static/thumbs/video1_obs.gif)](https://latino-pro.github.io/LVTINO/static/videos/video1/obs.mp4) | [![out](https://github.com/LATINO-PRO/LVTINO/blob/web-page/static/thumbs/video1_out.gif)](https://latino-pro.github.io/LVTINO/static/videos/video1/out.mp4) |
| [![obs](https://github.com/LATINO-PRO/LVTINO/blob/web-page/static/thumbs/video2_obs.gif)](https://latino-pro.github.io/LVTINO/static/videos/video2/obs.mp4) | [![out](https://github.com/LATINO-PRO/LVTINO/blob/web-page/static/thumbs/video2_out.gif)](https://latino-pro.github.io/LVTINO/static/videos/video2/out.mp4) |


---

## Overview

LVTINO solves a variety of video inverse problems by combining:
1. a **video generative prior** (VCM; here based on **Wan2.1-1.3B** and a **CausVid** fine-tuned checkpoint),
2. a **measurement-consistency step** (proximal / data-fidelity operator),
3. optional **warm-starting** using **VIDUE** for the most challenging setting (**Problem C** in the paper).

This codebase is based on, and adapts components from:
- **CausVid** (VCM backbone): https://github.com/tianweiy/CausVid  
- **VIDUE** (warm-start backbone for Problem C): https://github.com/shangwei5/VIDUE  

> **Note:** This repository is a *minimized* codebase tailored to the experiments and usage in the LVTINO paper.

---

## What’s included

### Minimal inference scripts
- `minimal_inference/LVTINO.py`  
  Solves **Problem A** and **Problem B**, and also supports the **Noise + JPEG compression** setting.

- `minimal_inference/LVTINO_VIDUE.py`  
  Solves **Problem C** using **VIDUE** as a warm-start.

### VIDUE checkpoints (Git LFS)
We provide (via **Git LFS**) the two VIDUE checkpoints (GoPro and Adobe), as in the official VIDUE repository.
The code has been minimized to match our usage.

### Optuna hyperparameter tuning
We provide two Optuna-based versions of the scripts (Optuna is **not** included in `requirements.txt`) that can automatically tune solver hyperparameters on a dataset of your choice, optimizing one of:
- **PSNR**
- **LPIPS**
- **SSIM**

---

## Installation

### 1) Clone the repo (with LFS)
VIDUE checkpoints are stored with Git LFS:
```bash
git lfs install
git clone https://github.com/LATINO-PRO/LVTINO.git
cd LVTINO
git lfs pull
```

### 2) Create an environment (recommended)
Example with `venv`:
```bash
python -m venv lvtino
source lvtino/bin/activate
pip install -U pip setuptools wheel
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Install this repo as a package
```bash
pip install -e .
```

---

## Model weights (required)

### Wan2.1 base models
Download the Wan base models from:
https://github.com/Wan-Video/Wan2.1

Save them under:
```
wan_models/Wan2.1-T2V-1.3B/
```

We provide an automated script:
```bash
python download_WAN.py
```

### CausVid fine-tuned checkpoint
Download the CausVid fine-tuned model from the official CausVid repository:
https://github.com/tianweiy/CausVid

We provide an automated script:
```bash
python download_CausVid.py
```

> **Important:** We do not redistribute third-party model weights. Please ensure you comply with the upstream licenses/terms.

---


## Running inference

Check the available CLI flags:
```bash
python minimal_inference/LVTINO.py --help
python minimal_inference/LVTINO_VIDUE.py --help
```

### Problem A / B / Noise+JPEG
Example:
```bash
python minimal_inference/LVTINO.py \
  --config_path "./configs/wan_bidirectional_dmd_from_scratch.yaml" \
  --checkpoint_folder "./models/bidirectional_checkpoint2" \
  --output_folder "./outputs/lvtino_blur_sr" \
  --prompt_file_path "./prompt.txt" \
  --frames_root "./adobe_frames" \
  --operator "tblur_sr"
```

### Problem C with VIDUE warm-start
Example:
```bash
python minimal_inference/LVTINO_VIDUE.py \
  --config_path "./configs/wan_bidirectional_dmd_from_scratch_VIDUE.yaml" \
  --checkpoint_folder "./models/bidirectional_checkpoint2" \
  --output_folder "./outputs/lvtino_SR8x8" \
  --prompt_file_path "./prompt.txt" \
  --frames_root "./adobe_frames"
```
## Hyperparameter tuning (Optuna)

Install Optuna separately:
```bash
pip install optuna
```

Then run the Optuna scripts (see `--help` for full options). You can select the metric to optimize via `--optuna_objective` (e.g. `psnr`, `lpips`, `ssim`).

### LVTINO + Optuna
Example:
```bash
python minimal_inference/LVTINO_optuna.py \
  --config_path "./configs/wan_bidirectional_dmd_from_scratch.yaml" \
  --checkpoint_folder "./models/bidirectional_checkpoint2" \
  --output_folder "./outputs/lvtino_blur_sr_optuna" \
  --prompt_file_path "./prompt.txt" \
  --frames_root "./adobe_frames" \
  --operator "tblur_sr" \
  --optuna \
  --optuna_trials 500 \
  --optuna_storage "sqlite:////path_to_study/study.db" \
  --optuna_videos 1 \
  --optuna_T 25 \
  --optuna_objective "lpips" \
  --optuna_only
```

### LVTINO + VIDUE + Optuna
Example:
```bash
python minimal_inference/LVTINO_VIDUE_optuna.py \
  --config_path "./configs/wan_bidirectional_dmd_from_scratch_VIDUE.yaml" \
  --checkpoint_folder "./models/bidirectional_checkpoint2" \
  --output_folder "./outputs/lvtino_SR8x8_optuna" \
  --prompt_file_path "./prompt.txt" \
  --frames_root "./adobe_frames" \
  --optuna \
  --optuna_trials 500 \
  --optuna_storage "sqlite:////path_to_study/study.db" \
  --optuna_videos 1 \
  --optuna_T 25 \
  --optuna_objective "lpips" \
  --optuna_only
```

## Acknowledgements

This codebase is based on:
- CausVid: https://github.com/tianweiy/CausVid  
- VIDUE: https://github.com/shangwei5/VIDUE  
- Wan2.1: https://github.com/Wan-Video/Wan2.1  

We thank the authors for releasing their code and models.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{spagnoletti2025lvtinolatentvideoconsistency,
      title={LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration}, 
      author={Alessio Spagnoletti and Andrés Almansa and Marcelo Pereyra},
      year={2025},
      eprint={2510.01339},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.01339}, 
}
```
