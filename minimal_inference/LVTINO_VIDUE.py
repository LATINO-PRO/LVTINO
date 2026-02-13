#!/usr/bin/env python3
import os
import re
import csv
import glob
import random
import argparse
import subprocess
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import deepinv as dinv
import imageio
import lpips

from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from huggingface_hub import hf_hub_download

from causvid.models.wan.bidirectional_LVTINO_VIDUE import (
    PipelineLVTINO,
)
from causvid.data import TextDataset

from utils import (
    TemporalSRThenSROp,
    load_video_from_frames,
    save_video_tensor,
)

# -------------------------
# LPIPS helpers
# -------------------------
def _resize_tensor_chw(x: torch.Tensor, size_wh: Tuple[int, int]) -> torch.Tensor:
    """Resize [C,H,W] float tensor to (W,H) using area interpolation."""
    _, _, _ = x.shape
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
    _, Hr, Wr = ref.shape[1:]
    _, Hp, Wp = pred.shape[1:]
    if (Hr, Wr) != (Hp, Wp):
        pred = torch.stack([_resize_tensor_chw(pred[t], (Wr, Hr)) for t in range(T)], dim=0)

    # Normalize to [-1,1]
    ref = ref * 2.0 - 1.0
    pred = pred * 2.0 - 1.0

    total = 0.0
    done = 0
    for s in range(0, T, batch_size):
        e = min(T, s + batch_size)
        a = pred[s:e].to(device, non_blocking=True)
        b = ref[s:e].to(device, non_blocking=True)
        vals = loss_fn(a, b).view(-1)
        total += float(vals.sum().item())
        done += (e - s)
        del a, b, vals
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total / max(1, done)


# -------------------------
# Interp helpers
# -------------------------
def tensor_to_uint8_img(t3chw: torch.Tensor) -> np.ndarray:
    """t: (3,H,W) in [0,1] -> uint8 HWC"""
    x = t3chw.detach().float().cpu().clamp(0, 1) * 255.0
    x = x.round().byte().permute(1, 2, 0).numpy()
    return x

def save_png(path: str, np_img_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, np_img_uint8)

def numeric_key(p: str):
    stem = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else stem


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # Core I/O
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True,
                        help="Root dir containing numbered subfolders, each with frames.")
    parser.add_argument("--T", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--lpips_batch", type=int, default=16)

    # SR setup (SR8x8 = temporal x8 + spatial x8)
    parser.add_argument("--sr_factor", type=int, default=8)
    parser.add_argument("--temporal_factor", type=int, default=8)

    # Selection of videos
    parser.add_argument("--mod", type=int, default=1, help="Process only video_id %% mod == 0.")
    parser.add_argument("--min_id", type=int, default=0, help="Process only video_id >= min_id.")

    # VIDUE interpolation (external script)
    parser.add_argument("--interp_input_root", type=str,
                        default="./temp/interp_input",
                        help="Where to write sparse observation frames (per-video subfolders).")
    parser.add_argument("--interp_result_root", type=str,
                        default="./temp/interp_result",
                        help="Where VIDUE writes interpolated output frames (per-video subfolders).")
    parser.add_argument("--infer_script", type=str,
                        default="./VIDUE/code/inference_vidue_worsu.py")
    parser.add_argument("--infer_model_path", type=str,
                        default="./VIDUE/pretrained_model/model_best_gopro.pt")
    parser.add_argument("--infer_default_data", type=str, default="Adobe")
    parser.add_argument("--infer_n_outputs", type=int, default=8)
    parser.add_argument("--infer_border", action="store_true", default=True)
    parser.add_argument("--infer_m", type=int, default=7)
    parser.add_argument("--infer_n", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_folder, "metrics.csv")
    csv_exists = os.path.isfile(metrics_csv_path)

    # Repro + determinism
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    work_dtype = torch.bfloat16

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -------------------------
    # Load SDXL+DMD2 (image prior)
    # -------------------------
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

    unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(
        torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device, weights_only=True)
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe_img = DiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        guidance_scale=0,
    ).to(device)
    pipe_img.scheduler = LCMScheduler.from_config(pipe_img.scheduler.config)

    prompt_for_embed = "a high resolution photo"
    text_embeddings, _, pooled_text_embeds, _ = pipe_img.encode_prompt(
        prompt_for_embed, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
    )

    image_height, image_width = 768, 1280
    time_ids = pipe_img._get_add_time_ids(
        original_size=(image_height, image_width),
        crops_coords_top_left=(0, 0),
        target_size=(image_height, image_width),
        dtype=torch.float16,
        text_encoder_projection_dim=1280,
    ).to(device)

    added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": time_ids}

    pipe_img.scheduler.set_timesteps(8, device=device)
    pipe_img.scheduler.timesteps = torch.tensor([499, 249, 63], device=device, dtype=torch.long)

    # -------------------------
    # Load video pipeline
    # -------------------------
    config = OmegaConf.load(args.config_path)
    pipe = PipelineLVTINO(config, device=str(device))
    state = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipe.generator.load_state_dict(state)
    pipe = pipe.to(device=device, dtype=work_dtype)

    # Prompts
    dataset = TextDataset(args.prompt_file_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in {args.prompt_file_path}")
    prompt = dataset[0]

    # Metrics
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device).eval()
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()

    if not csv_exists:
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "frames_used", "psnr", "ssim", "lpips", "output_path"])

    # -------------------------
    # Select videos
    # -------------------------
    def is_valid_subdir(name: str) -> bool:
        return name.isdigit() and (int(name) % args.mod == 0) and (int(name) >= args.min_id)

    subdirs = sorted([d for d in os.listdir(args.frames_root) if is_valid_subdir(d)], key=lambda s: int(s))

    # -------------------------
    # Forward model: SR8x8 (temporal SR then spatial SR)
    # -------------------------
    forward_model = TemporalSRThenSROp(
        factor_t=args.temporal_factor,
        factor=args.sr_factor,
        filter="bicubic",
        padding="reflect",
    )

    for name in tqdm(subdirs, desc="Processing videos"):
        vid_id = int(name)
        frames_dir = os.path.join(args.frames_root, name)
        out_dir = os.path.join(args.output_folder, f"{vid_id:05d}")
        os.makedirs(out_dir, exist_ok=True)

        # ---- Load GT
        x_gt = load_video_from_frames(frames_dir, device=device, dtype=work_dtype, T=args.T)  # (T,3,H,W)
        save_video_tensor(x_gt.float(), os.path.join(out_dir, "gt.mp4"), fps=args.fps)

        # ---- Observe y = A(x)
        y = forward_model(x_gt.to(dtype=torch.float32)).to(dtype=work_dtype)
        save_video_tensor(y.float(), os.path.join(out_dir, "observed_y.mp4"), fps=2)

        # ---- 1) Write sparse observation frames for VIDUE
        vid_dir_sparse = os.path.join(args.interp_input_root, f"{vid_id:05d}")
        os.makedirs(vid_dir_sparse, exist_ok=True)

        # For SR8x8:
        for t in range(y.shape[0]):
            np_img = tensor_to_uint8_img(y[t])
            save_png(os.path.join(vid_dir_sparse, f"{t * args.temporal_factor:06d}.png"), np_img)

        # ---- 2) Run VIDUE interpolation
        cmd = [
            "python", args.infer_script,
            "--default_data", args.infer_default_data,
            "--data_path", args.interp_input_root,
            "--model_path", args.infer_model_path,
            "--result_path", args.interp_result_root,
            "--n_outputs", str(args.infer_n_outputs),
            "--m", str(args.infer_m),
            "--n", str(args.infer_n),
        ]
        if args.infer_border:
            cmd += ["--border"]

        print("Running interpolation script:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # ---- 3) Load interpolated frames as x_init
        vid_dir_frames = os.path.join(args.interp_result_root, f"{vid_id:05d}")
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        all_paths = [
            p for p in glob.glob(os.path.join(vid_dir_frames, "*"))
            if os.path.splitext(p)[1].lower() in img_exts
        ]
        frame_paths = sorted(all_paths, key=numeric_key)

        if len(frame_paths) < (args.T - 1):
            raise RuntimeError(
                f"Expected at least {args.T - 1} frames in {vid_dir_frames}, found {len(frame_paths)}."
            )

        frame_paths_Tm1 = frame_paths[: args.T - 1]

        x_init_frames: List[np.ndarray] = []
        for p in frame_paths_Tm1:
            img = imageio.imread(p)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            x_init_frames.append(img)

        # Duplicate last frame to reach T
        x_init_frames.append(x_init_frames[-1].copy())

        x_init = torch.stack(
            [
                torch.from_numpy(im).permute(2, 0, 1).to(device=device, dtype=work_dtype) / 255.0
                for im in x_init_frames
            ],
            dim=0,
        )  # (T,3,H,W)

        save_video_tensor(x_init.float(), os.path.join(out_dir, "x_init_lowres.mp4"), fps=args.fps)
        torch.save(x_init, os.path.join(out_dir, "x_init_lowres.pt"))

        # Optional cleanup of sparse inputs
        try:
            import shutil
            if os.path.isdir(vid_dir_sparse):
                shutil.rmtree(vid_dir_sparse)
        except Exception as e:
            print(f"[WARN] Could not remove {vid_dir_sparse}: {e}")

        # ---- 4) Spatial upsample to GT resolution for initialization
        x_init_up = F.interpolate(
            x_init.to(dtype=torch.float32),
            size=(x_gt.shape[2], x_gt.shape[3]),
            mode="bilinear",
            align_corners=False,
        )  # (T,3,H,W) fp32
        save_video_tensor(x_init_up.float(), os.path.join(out_dir, "x_init_up.mp4"), fps=args.fps)

        # ---- Prepare inputs for pipeline
        x_init_lat = 2 * x_init_up.unsqueeze(0).clamp(0, 1) - 1  # (1,T,3,H,W) in [-1,1]
        y_n = 2 * y - 1

        z0_hat = pipe.vae.encode_from_pixel(x_init_lat)
        next_timestep = pipe.denoising_step_list[0] * torch.ones(
            z0_hat.shape[:2], dtype=torch.long, device=z0_hat.device
        )
        noisy_image_or_video = pipe.scheduler.add_noise(
            z0_hat.flatten(0, 1),
            torch.randn_like(z0_hat.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, z0_hat.shape[:2])

        # ---- Inference
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
            )

        restored = out[0]
        export_to_video(
            restored.permute(0, 2, 3, 1).detach().cpu().numpy(),
            os.path.join(out_dir, "output.mp4"),
            fps=args.fps,
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

        # Cleanup
        del x_gt, y, x_init, x_init_up, x_init_lat, z0_hat, noisy_image_or_video, out, restored, x_ref, y_ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
