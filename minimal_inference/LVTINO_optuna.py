#!/usr/bin/env python3
import os
import argparse
from typing import Tuple, Dict, Any, Optional, List
import json
import random

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

try:
    import optuna
except Exception:
    optuna = None  # optuna is optional unless --optuna is used

try:
    from causvid.models.wan.bidirectional_LVTINO_optuna import PipelineLVTINO
except Exception:
    from bidirectional_LVTINO_optuna import PipelineLVTINO

from causvid.data import TextDataset

import torchmetrics
import csv

from utils import (
    TemporalSRThenSROp,
    TemporalBlurThenSROp,
    GaussianNoiseJPEGOp,
    load_video_from_frames,
    save_video_tensor,
)

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download

import lpips


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
# Operator builder
# -------------------------
def build_forward_model(args, device: torch.device):
    """
    Returns an instance of the chosen forward operator.
    """
    op = args.operator

    if op == "tblur_sr":
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
# Optuna: hyperparameter space
# -------------------------
def suggest_hparams(trial, operator: str) -> Dict[str, Any]:
    """
    Operator-aware hyperparameter space.
    Only suggests parameters that are actually used for the chosen operator.

    operator in {"tblur_sr", "tsr_sr", "noise_jpeg"}
    """
    hp: Dict[str, Any] = {}

    if operator == "tblur_sr":
        # TemporalBlurThenSROp (SRx*)
        hp["tblur_gamma_switch"] = trial.suggest_int("tblur_gamma_switch", 2, 6)
        hp["tblur_gamma_hi"] = trial.suggest_float("tblur_gamma_hi", 1e2, 2e4, log=True)
        hp["tblur_gamma_lo"] = trial.suggest_float("tblur_gamma_lo", 5e1, 2e3, log=True)

        hp["tblur_latino_gamma_switch"] = trial.suggest_int("tblur_latino_gamma_switch", 2, 6)
        hp["tblur_latino_gamma_hi"] = trial.suggest_float("tblur_latino_gamma_hi", 1e3, 1e5, log=True)
        hp["tblur_latino_gamma_lo"] = trial.suggest_float("tblur_latino_gamma_lo", 5e1, 5e3, log=True)

        # safety: lo <= hi
        hp["tblur_gamma_lo"] = min(hp["tblur_gamma_lo"], hp["tblur_gamma_hi"])
        hp["tblur_latino_gamma_lo"] = min(hp["tblur_latino_gamma_lo"], hp["tblur_latino_gamma_hi"])

    elif operator == "tsr_sr":
        # TemporalSRThenSROp (SR4x4 / SR8x8 / ...)
        hp["tsr_tv_lam"] = trial.suggest_float("tsr_tv_lam", 1e-4, 5e-2, log=True)
        hp["tsr_cg_steps"] = trial.suggest_int("tsr_cg_steps", 3, 30)
        hp["tsr_cg_epsilon"] = trial.suggest_categorical(
            "tsr_cg_epsilon", [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
        )

    elif operator == "noise_jpeg":
        # JPEG
        hp["jpeg_gamma"] = trial.suggest_float("jpeg_gamma", 0.5, 5.0, log=True)
        hp["jpeg_latino_gamma"] = trial.suggest_float("jpeg_latino_gamma", 0.5, 5.0, log=True)

    else:
        raise ValueError(f"Unknown operator='{operator}'")

    # Optional: record operator in the trial params for clarity
    hp["operator"] = operator

    return hp



def list_video_ids(frames_root: str, predicate) -> List[int]:
    subdirs = [d for d in os.listdir(frames_root) if predicate(d)]
    subdirs.sort(key=lambda s: int(s))
    return [int(d) for d in subdirs]


# -------------------------
# Single-video runner (used by Optuna + main)
# -------------------------
@torch.no_grad()
def run_one_video(
    vid_id: int,
    frames_root: str,
    out_dir: str,
    pipe: PipelineLVTINO,
    pipe_img,
    forward_model,
    device: torch.device,
    work_dtype: torch.dtype,
    text_embeddings,
    added_cond_kwargs,
    prompt: str,
    T: int,
    fps: int,
    lpips_loss,
    psnr_loss,
    ssim_loss,
    lpips_batch: int,
    add_noise_std: float,
    save_outputs: bool,
):
    frames_dir = os.path.join(frames_root, f"{vid_id}")
    if not os.path.isdir(frames_dir):
        # zero-pad fallback
        frames_dir = os.path.join(frames_root, f"{vid_id:05d}")

    os.makedirs(out_dir, exist_ok=True)

    # ---- Load GT
    x_gt = load_video_from_frames(frames_dir, device=device, dtype=work_dtype, T=T)  # [T,3,H,W] in [0,1]
    if save_outputs:
        save_video_tensor(x_gt.float(), os.path.join(out_dir, "gt.mp4"), fps=fps)

    # ---- Measurements
    y = forward_model(x_gt.to(dtype=torch.float32)).to(dtype=work_dtype)
    if add_noise_std and add_noise_std > 0:
        y = y + float(add_noise_std) * torch.randn_like(y)

    if save_outputs:
        save_video_tensor(y.float(), os.path.join(out_dir, "observed_y.mp4"), fps=fps)

    # ---- Init
    x0_hat = forward_model.prox_l2(torch.zeros_like(x_gt), y=y, gamma=200).unsqueeze(0)
    if save_outputs:
        save_video_tensor(x0_hat.squeeze(0).float(), os.path.join(out_dir, "init.mp4"), fps=fps)

    # ---- Scale to [-1,1]
    x0_hat = 2 * x0_hat - 1
    y_n = 2 * y - 1

    # ---- Noisy latents for pipeline
    z0_hat = pipe.vae.encode_from_pixel(x0_hat)
    next_timestep = pipe.denoising_step_list[0] * torch.ones(
        z0_hat.shape[:2], dtype=torch.long, device=z0_hat.device
    )

    noisy_image_or_video = pipe.scheduler.add_noise(
        z0_hat.flatten(0, 1),
        torch.randn_like(z0_hat.flatten(0, 1)),
        next_timestep.flatten(0, 1)
    ).unflatten(0, z0_hat.shape[:2])

    # ---- Inference
    out = pipe.inference(
        pipe_img=pipe_img,
        x_gt=x_gt,
        text_embeddings=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
        noise=noisy_image_or_video,
        y=y_n,
        forward_model=forward_model,
        output_folder=out_dir,
        fps=fps,
        text_prompts=[prompt],
    )  # (B,T,C,H,W)

    restored = out[0]  # [T,3,H,W] in [0,1]
    if save_outputs:
        export_to_video(
            restored.permute(0, 2, 3, 1).detach().cpu().numpy(),
            os.path.join(out_dir, "output.mp4"),
            fps=fps
        )

    # ---- Metrics
    Tm = min(x_gt.shape[0], restored.shape[0])
    x_ref = x_gt[:Tm].to(device, dtype=torch.float32)
    y_ref = restored[:Tm].to(device, dtype=torch.float32)

    psnr = psnr_loss(x_ref, y_ref).item()
    ssim = ssim_loss(x_ref, y_ref).item()
    lp = lpips_video_mean(lpips_loss, x_ref, y_ref, device=device, batch_size=lpips_batch)

    # cleanup
    del x_gt, y, x0_hat, z0_hat, noisy_image_or_video, out, restored, x_ref, y_ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return psnr, ssim, lp


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--T", type=int, default=25)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--lpips_batch", type=int, default=8)
    
    # Selection of videos
    parser.add_argument("--mod", type=int, default=1, help="Process only video_id %% mod == 0.")
    parser.add_argument("--min_id", type=int, default=0, help="Process only video_id >= min_id.")

    # Operator selection (unchanged)
    parser.add_argument("--operator", type=str, default="tblur_sr",
                        choices=["tblur_sr", "tsr_sr", "noise_jpeg"])
    parser.add_argument("--filter", type=str, default="bicubic")
    parser.add_argument("--padding", type=str, default="reflect")
    parser.add_argument("--sr_factor", type=int, default=4)

    # TemporalBlurThenSROp knobs
    parser.add_argument("--kernel_size_t", type=int, default=7)
    parser.add_argument("--kernel_type", type=str, default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--sigma_t", type=float, default=2.0)
    parser.add_argument("--blur_sr_factor", type=int, default=8)

    # TemporalSRThenSROp knobs
    parser.add_argument("--factor_t", type=int, default=4)

    # GaussianNoiseJPEGOp knobs
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--jpeg_quality", type=int, default=10)

    parser.add_argument("--add_noise_std", type=float, default=0.001)

    # -------------------------
    # Optuna
    # -------------------------
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search before full restoration.")
    parser.add_argument("--optuna_trials", type=int, default=20)
    parser.add_argument("--optuna_videos", type=int, default=2,
                        help="How many videos (in order) to use for the Optuna objective.")
    parser.add_argument("--optuna_T", type=int, default=12,
                        help="Frames used during Optuna only (smaller = faster).")
    parser.add_argument("--optuna_objective", type=str, default="psnr",
                        choices=["psnr", "ssim", "lpips", "combo"],
                        help="Objective to optimize during Optuna.")
    parser.add_argument("--optuna_combo_w_psnr", type=float, default=1.0)
    parser.add_argument("--optuna_combo_w_ssim", type=float, default=10.0)
    parser.add_argument("--optuna_combo_w_lpips", type=float, default=5.0)
    parser.add_argument("--optuna_seed", type=int, default=0)
    parser.add_argument("--optuna_only", action="store_true",
                        help="Only run Optuna and save best params, do not run full restoration.")
    parser.add_argument("--optuna_storage", type=str, default="",
                        help="Optional Optuna storage URL (e.g., sqlite:///study.db) to resume.")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_folder, "metrics.csv")
    csv_exists = os.path.isfile(metrics_csv_path)

    # Determinism / seeds
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dtype = torch.bfloat16

    seed = 42
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # SDXL
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

    base_prompt = "a high resolution photo"
    text_embeddings, _, pooled_text_embeds, _ = pipe_img.encode_prompt(
        base_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

    image_height = 768
    image_width = 1280
    time_ids = pipe_img._get_add_time_ids(
        original_size=(image_height, image_width),
        crops_coords_top_left=(0, 0),
        target_size=(image_height, image_width),
        dtype=torch.float16,
        text_encoder_projection_dim=1280
    ).to(device)

    added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": time_ids}

    num_inference_steps = 8
    pipe_img.scheduler.set_timesteps(num_inference_steps, device=device)
    pipe_img.scheduler.timesteps = torch.tensor([499, 374, 249, 124, 124], device=device, dtype=torch.long)

    # Forward model
    forward_model = build_forward_model(args, device=device)

    # Load config & pipeline
    config = OmegaConf.load(args.config_path)
    config.forward_operator = args.operator
    config.save_intermediates = True

    pipe = PipelineLVTINO(config, device=str(device))
    state = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipe.generator.load_state_dict(state)
    pipe = pipe.to(device=device, dtype=work_dtype)

    # Prompt dataset
    dataset = TextDataset(args.prompt_file_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in {args.prompt_file_path}")
    prompt = dataset[0]

    # Metrics
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device).eval()
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()

    # CSV
    if not csv_exists:
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "frames_used", "psnr", "ssim", "lpips", "output_path"])

    def _is_valid_subdir(name: str) -> bool:
        return name.isdigit() and (int(name) % args.mod == 0) and (int(name) >= args.min_id)


    video_ids = list_video_ids(args.frames_root, _is_valid_subdir)
    if args.start > 0:
        video_ids = [v for v in video_ids if v >= args.start]

    # -------------------------
    # Optuna search
    # -------------------------
    best_hparams: Optional[Dict[str, Any]] = None
    if args.optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed. Please `pip install optuna` or disable --optuna.")

        tune_ids = video_ids[: max(1, int(args.optuna_videos))]
        if len(tune_ids) == 0:
            raise RuntimeError("No videos found for Optuna tuning (check frames_root / predicate).")

        direction = "maximize"
        if args.optuna_objective == "lpips":
            direction = "minimize"

        sampler = optuna.samplers.TPESampler(seed=int(args.optuna_seed))
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name="lvtino_hparam_search_SLURM_JPEG",
            storage=args.optuna_storage if args.optuna_storage else None,
            load_if_exists=bool(args.optuna_storage),
        )

        # Disable intermediate saving during tuning
        pipe.save_intermediates = False

        def objective(trial):
            hp = suggest_hparams(trial, operator=args.operator)
            pipe.hparams = hp  # <-- pipeline consumes this during inference/LATINO_img

            scores = []
            for vid in tune_ids:
                out_dir = os.path.join(args.output_folder, "_optuna_tmp", f"trial_{trial.number:04d}", f"{vid:05d}")
                # We don't save mp4 during tuning
                psnr, ssim, lp = run_one_video(
                    vid_id=vid,
                    frames_root=args.frames_root,
                    out_dir=out_dir,
                    pipe=pipe,
                    pipe_img=pipe_img,
                    forward_model=forward_model,
                    device=device,
                    work_dtype=work_dtype,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    prompt=prompt,
                    T=int(args.optuna_T),
                    fps=args.fps,
                    lpips_loss=lpips_loss,
                    psnr_loss=psnr_loss,
                    ssim_loss=ssim_loss,
                    lpips_batch=args.lpips_batch,
                    add_noise_std=args.add_noise_std,
                    save_outputs=True,
                )

                if args.optuna_objective == "psnr":
                    scores.append(psnr)
                elif args.optuna_objective == "ssim":
                    scores.append(ssim)
                elif args.optuna_objective == "lpips":
                    scores.append(lp)
                else:
                    # combo: maximize (w_psnr*psnr + w_ssim*ssim - w_lpips*lpips)
                    scores.append(
                        float(args.optuna_combo_w_psnr) * psnr
                        + float(args.optuna_combo_w_ssim) * ssim
                        - float(args.optuna_combo_w_lpips) * lp
                    )

            return float(np.mean(scores))

        study.optimize(objective, n_trials=int(args.optuna_trials), gc_after_trial=True)

        best_hparams = dict(study.best_params)
        # Add any keys that are not in best_params but required by the pipeline.
        # (suggest_hparams returns all keys, but Optuna only stores suggested ones.)
        # We'll rebuild the full dict from the best trial.
        best_hparams = suggest_hparams(study.best_trial, operator=args.operator)

        best_path = os.path.join(args.output_folder, "optuna_best_hparams.json")
        with open(best_path, "w") as f:
            json.dump(
                {
                    "operator": args.operator,
                    "objective": args.optuna_objective,
                    "best_value": float(study.best_value),
                    "best_hparams": best_hparams,
                },
                f,
                indent=2
            )

        print(f"[OPTUNA] Best value: {study.best_value:.6f}")
        print(f"[OPTUNA] Best params saved to: {best_path}")

        if args.optuna_only:
            return

    # If we ran Optuna, use the best params for full run
    if best_hparams is not None:
        pipe.hparams = best_hparams

    # Re-enable intermediate saving for the final run
    pipe.save_intermediates = True

    # -------------------------
    # Full restoration
    # -------------------------
    for vid in tqdm(video_ids, desc="Processing videos"):
        out_dir = os.path.join(args.output_folder, f"{vid:05d}")
        psnr, ssim, lp = run_one_video(
            vid_id=vid,
            frames_root=args.frames_root,
            out_dir=out_dir,
            pipe=pipe,
            pipe_img=pipe_img,
            forward_model=forward_model,
            device=device,
            work_dtype=work_dtype,
            text_embeddings=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
            prompt=prompt,
            T=int(args.T),
            fps=args.fps,
            lpips_loss=lpips_loss,
            psnr_loss=psnr_loss,
            ssim_loss=ssim_loss,
            lpips_batch=args.lpips_batch,
            add_noise_std=args.add_noise_std,
            save_outputs=True,
        )

        with open(metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([vid, int(args.T), f"{psnr:.6f}", f"{ssim:.6f}", f"{lp:.6f}",
                             os.path.join(out_dir, "output.mp4")])

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
