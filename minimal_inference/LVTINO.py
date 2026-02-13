#!/usr/bin/env python3
import os
import argparse
from typing import Tuple
import torch
import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

from causvid.models.wan.bidirectional_LVTINO import PipelineLVTINO
from causvid.data import TextDataset

import torchmetrics
import csv

from utils import TemporalSRThenSROp, TemporalBlurThenSROp, GaussianNoiseJPEGOp, load_video_from_frames, save_video_tensor

import random
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download

import lpips  # <-- LPIPS (we’ll use VGG backbone)

# -------------------------
# LPIPS helpers
# -------------------------
def _resize_tensor_chw(x: torch.Tensor, size_wh: Tuple[int, int]) -> torch.Tensor:
    """Resize [C,H,W] float tensor to (W,H) using area interpolation."""
    Wt, Ht = size_wh
    x4 = x.unsqueeze(0)
    x4 = torch.nn.functional.interpolate(x4, size=(Ht, Wt), mode="area", align_corners=None)
    return x4.squeeze(0)

@torch.inference_mode()
def lpips_video_mean(
    loss_fn: lpips.LPIPS,
    ref: torch.Tensor,       # [T,C,H,W] in [0,1]
    pred: torch.Tensor,      # [T,C,H,W] in [0,1]
    device: torch.device,
    batch_size: int = 8,
) -> float:
    """Stream LPIPS over frames, average. Resizes pred to ref size if needed."""
    T = min(ref.shape[0], pred.shape[0])
    ref = ref[:T]
    pred = pred[:T]

    # Match resolution to ref
    Cr, Hr, Wr = ref.shape[1:]
    Cp, Hp, Wp = pred.shape[1:]
    if (Hr, Wr) != (Hp, Wp):
        resized = []
        for t in range(T):
            resized.append(_resize_tensor_chw(pred[t], (Wr, Hr)))
        pred = torch.stack(resized, dim=0)

    # Normalize to [-1,1]
    ref = ref * 2.0 - 1.0
    pred = pred * 2.0 - 1.0

    total = 0.0
    done = 0
    for s in range(0, T, batch_size):
        e = min(T, s + batch_size)
        a = pred[s:e].to(device, non_blocking=True)  # [b,C,H,W]
        b = ref[s:e].to(device, non_blocking=True)
        vals = loss_fn(a, b).view(-1)               # [b]
        total += float(vals.sum().item())
        done += (e - s)
        del a, b, vals
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return total / max(1, done)


# -------------------------
# Operator builder
# -------------------------
def build_forward_model(args, device: torch.device):
    """
    Returns an instance of the chosen forward operator.
    """
    op = args.operator

    if op == "tblur_sr":
        # Temporal blur then spatial SR
        return TemporalBlurThenSROp(
            kernel_size_t=args.kernel_size_t,
            kernel_type=args.kernel_type,
            sigma_t=args.sigma_t,
            factor=args.blur_sr_factor,
            filter=args.filter,
            padding=args.padding,
            device=str(device),
            dtype=torch.float32,
        )

    if op == "tsr_sr":
        # Temporal SR then spatial SR
        return TemporalSRThenSROp(
            factor_t=args.factor_t,
            factor=args.sr_factor,
            filter=args.filter,
            padding=args.padding,
        )

    if op == "noise_jpeg":
        return GaussianNoiseJPEGOp(
            sigma=args.noise_sigma,
            jpeg_quality=args.jpeg_quality,
        )

    raise ValueError(f"Unknown operator: {op}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--T", type=int, default=25)  # we’ll use first 25 frames
    parser.add_argument("--frames_root", type=str, required=True,
                        help="Root directory containing numbered subfolders (1..8702), each with 25 frames.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--lpips_batch", type=int, default=8, help="Frames per LPIPS forward to limit VRAM.")
    
    # Selection of videos
    parser.add_argument("--mod", type=int, default=1, help="Process only video_id %% mod == 0.")
    parser.add_argument("--min_id", type=int, default=0, help="Process only video_id >= min_id.")

    # -------------------------
    # NEW: operator selection
    # -------------------------
    parser.add_argument(
        "--operator",
        type=str,
        default="tblur_sr",
        choices=["tblur_sr", "tsr_sr", "noise_jpeg"],
        help=(
            "Forward operator to apply to GT video.\n"
            "  tblur_sr  : TemporalBlurThenSROp (default)\n"
            "  tsr_sr    : TemporalSRThenSROp\n"
            "  noise_jpeg: GaussianNoiseJPEGOp"
        ),
    )

    # Shared operator knobs
    parser.add_argument("--filter", type=str, default="bicubic", help="Resampling filter (if applicable).")
    parser.add_argument("--padding", type=str, default="reflect", help="Padding mode (if applicable).")
    parser.add_argument("--sr_factor", type=int, default=4)

    # TemporalBlurThenSROp knobs
    parser.add_argument("--kernel_size_t", type=int, default=7)
    parser.add_argument("--kernel_type", type=str, default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--sigma_t", type=float, default=2.0)
    parser.add_argument("--blur_sr_factor", type=int, default=8, help="Spatial downsampling factor for tblur_sr.")

    # TemporalSRThenSROp knobs
    parser.add_argument("--factor_t", type=int, default=4, help="Temporal SR factor for tsr_sr.")

    # GaussianNoiseJPEGOp knobs
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--jpeg_quality", type=int, default=10)

    # Extra noise added AFTER applying operator
    parser.add_argument("--add_noise_std", type=float, default=0.001, help="Extra Gaussian noise added to y after A(x).")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_folder, "metrics.csv")
    csv_exists = os.path.isfile(metrics_csv_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dtype = torch.bfloat16

    # Set global random seeds for full reproducibility
    seed = 42
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior for CUDA (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load stable diffusion
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
    unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device, weights_only=True))
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe_img = DiffusionPipeline.from_pretrained(
        base_model_id, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16", guidance_scale=0
    ).to(device)
    pipe_img.scheduler = LCMScheduler.from_config(pipe_img.scheduler.config)

    prompt = "a high resolution photo"

    # Encode text to conditioning
    text_embeddings, _, pooled_text_embeds, _ = pipe_img.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

    # Assuming desired resolution of 768x1280
    image_height = 768
    image_width = 1280

    # Get time_ids automatically based on the image resolution
    time_ids = pipe_img._get_add_time_ids(
        original_size=(image_height, image_width),
        crops_coords_top_left=(0, 0),
        target_size=(image_height, image_width),
        dtype=torch.float16,
        text_encoder_projection_dim=1280
    ).to(device)

    # Additional conditioning required for SDXL
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds,
        "time_ids": time_ids
    }

    # Define the number of inference steps and set timesteps
    num_inference_steps = 8
    pipe_img.scheduler.set_timesteps(num_inference_steps, device=device)
    custom_timesteps = torch.tensor([499, 374, 249, 124, 124], device=device, dtype=torch.long)
    pipe_img.scheduler.timesteps = custom_timesteps

    # ---- Build forward model ONCE (selected by CLI)
    forward_model = build_forward_model(args, device=device)

    # ---- Load config & pipeline
    config = OmegaConf.load(args.config_path)
    config.forward_operator = args.operator
    pipe = PipelineLVTINO(config, device=str(device))
    state = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipe.generator.load_state_dict(state)
    pipe = pipe.to(device=device, dtype=work_dtype)

    # ---- Prompts
    dataset = TextDataset(args.prompt_file_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in {args.prompt_file_path}")
    prompt = dataset[0]

    # ---- Metrics (build ONCE)
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device).eval()
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()  # <-- VGG LPIPS

    # ---- Prepare CSV (append mode)
    if not csv_exists:
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "frames_used", "psnr", "ssim", "lpips", "output_path"])

    # ---- Enumerate subfolders
    def _is_valid_subdir(name):
        return name.isdigit() and (int(name) % args.mod == 0) and (int(name) >= args.min_id)

    subdirs = [d for d in os.listdir(args.frames_root) if _is_valid_subdir(d)]
    subdirs.sort(key=lambda s: int(s))

    for name in tqdm(subdirs, desc="Processing videos"):
        vid_id = int(name)
        frames_dir = os.path.join(args.frames_root, name)
        out_dir = os.path.join(args.output_folder, f"{vid_id:05d}")
        os.makedirs(out_dir, exist_ok=True)

        # ---- Load GT video (T frames)
        x_gt = load_video_from_frames(frames_dir, device=device, dtype=work_dtype, T=args.T)  # [T,3,H,W] in [0,1]
        save_video_tensor(x_gt.float(), os.path.join(out_dir, "gt.mp4"), fps=args.fps)

        # ---- Forward model: y = A(x_gt)
        y = forward_model(x_gt.to(dtype=torch.float32)).to(dtype=work_dtype)

        # Optional: add slight noise AFTER operator
        if args.add_noise_std and args.add_noise_std > 0:
            y = y + float(args.add_noise_std) * torch.randn_like(y)

        save_video_tensor(y.float(), os.path.join(out_dir, "observed_y.mp4"), fps=args.fps)

        # ---- Simple initializer via prox (HR length)
        x0_hat = forward_model.prox_l2(torch.zeros_like(x_gt), y=y, gamma=200).unsqueeze(0)
        save_video_tensor(x0_hat.squeeze(0).float(), os.path.join(out_dir, "init.mp4"), fps=args.fps)

        # Scale to [-1,1] for VAE encode
        x0_hat = 2 * x0_hat - 1
        y_n = 2 * y - 1

        # ---- Prepare noisy latents for pipeline
        z0_hat = pipe.vae.encode_from_pixel(x0_hat)
        next_timestep = pipe.denoising_step_list[0] * torch.ones(
            z0_hat.shape[:2], dtype=torch.long, device=z0_hat.device)

        noisy_image_or_video = pipe.scheduler.add_noise(
            z0_hat.flatten(0, 1),
            torch.randn_like(z0_hat.flatten(0, 1)),
            next_timestep.flatten(0, 1)
        ).unflatten(0, z0_hat.shape[:2])

        # ---- Run inference once per video
        with torch.no_grad():
            out = pipe.inference(
                pipe_img=pipe_img,
                x_gt=x_gt,
                text_embeddings=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                noise=noisy_image_or_video,
                y=y_n,
                forward_model=forward_model,
                output_folder=out_dir,
                fps=args.fps,
                text_prompts=[prompt],
            )  # (B,T,C,H,W)

        # Save restored video
        restored = out[0]  # [T,3,H,W] in [0,1]
        export_to_video(
            restored.permute(0, 2, 3, 1).detach().cpu().numpy(),
            os.path.join(out_dir, "output.mp4"),
            fps=args.fps
        )

        # ---- Metrics
        Tm = min(x_gt.shape[0], restored.shape[0])
        x_ref = x_gt[:Tm].to(device, dtype=torch.float32)
        y_ref = restored[:Tm].to(device, dtype=torch.float32)

        psnr = psnr_loss(x_ref, y_ref).item()
        ssim = ssim_loss(x_ref, y_ref).item()
        lp = lpips_video_mean(lpips_loss, x_ref, y_ref, device=device, batch_size=args.lpips_batch)

        with open(metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([vid_id, Tm, f"{psnr:.6f}", f"{ssim:.6f}", f"{lp:.6f}",
                             os.path.join(out_dir, "output.mp4")])

        del x_gt, y, x0_hat, z0_hat, noisy_image_or_video, out, restored, x_ref, y_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
