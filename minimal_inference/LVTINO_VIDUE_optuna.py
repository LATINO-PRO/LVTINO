#!/usr/bin/env python3
import os
import re
import csv
import glob
import json
import random
import argparse
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import lpips
import optuna
import imageio.v2 as imageio  # avoid imageio v3 warning

import sys
import types
import importlib.util
import shutil

from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from huggingface_hub import hf_hub_download

try:
    from causvid.models.wan.bidirectional_LVTINO_VIDUE_optuna import PipelineLVTINO
except Exception:
    from bidirectional_LVTINO_VIDUE_optuna import PipelineLVTINO  # local fallback

from causvid.data import TextDataset
from utils import TemporalSRThenSROp, load_video_from_frames, save_video_tensor


# -------------------------
# VIDUE in-process runner
# -------------------------
def _load_module_from_path(module_name: str, path: str) -> types.ModuleType:
    """Dynamically import a python file as a module and ensure its package imports work."""
    path = os.path.abspath(path)
    mod_dir = os.path.dirname(path)               # e.g. .../VIDUE/code
    mod_root = os.path.dirname(mod_dir)           # e.g. .../VIDUE

    # Make both code dir and VIDUE root importable (covers both layouts)
    for p in (mod_dir, mod_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_vidue_interpolation_inprocess(
    *,
    infer_script: str,
    default_data: str,
    data_path: str,      # should point to a root that contains per-video subfolders
    model_path: str,
    result_path: str,
    n_outputs: int,
    m: int,
    n: int,
    border: bool = True,
    save_image: bool = True,
    n_sequence: int = 4,
    n_gpus: int = 1,
    blur_deg: int = 1,
) -> None:
    """Run VIDUE interpolation by calling its Inference class directly."""
    vidue = _load_module_from_path("_vidue_infer", infer_script)

    # Match the script's default per-dataset settings
    gt_path = ""
    if default_data == "GOPRO":
        gt_path = "./gopro"
        n_sequence = 4
        n_gpus = 1
        blur_deg = 1
    elif default_data == "Adobe":
        gt_path = "./adobe"
        n_sequence = 4
        n_gpus = 1
        blur_deg = 1

    args = argparse.Namespace(
        save_image=save_image,
        border=border,
        default_data=default_data,
        data_path=data_path,
        model_path=model_path,
        result_path=result_path,
        m=int(m),
        n=int(n),
        n_outputs=int(n_outputs),
        gt_path=gt_path,
        n_sequence=int(n_sequence),
        n_GPUs=int(n_gpus),
        blur_deg=int(blur_deg),
        submodel="unet_18",
        joinType="concat",
        upmode="transpose",
    )

    if not hasattr(vidue, "Inference"):
        raise RuntimeError(f"VIDUE module at {infer_script} has no Inference class.")

    Infer = vidue.Inference(args)
    if not hasattr(Infer, "infer"):
        raise RuntimeError("VIDUE Inference object has no `.infer()` method.")
    Infer.infer()


# -------------------------
# Metrics helpers
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
    T = min(ref.shape[0], pred.shape[0])
    ref = ref[:T]
    pred = pred[:T]

    _, Hr, Wr = ref.shape[1:]
    _, Hp, Wp = pred.shape[1:]
    if (Hr, Wr) != (Hp, Wp):
        pred = torch.stack([_resize_tensor_chw(pred[t], (Wr, Hr)) for t in range(T)], dim=0)

    # normalize to [-1, 1]
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


def tensor_to_uint8_img(t3chw: torch.Tensor) -> np.ndarray:
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
# Optuna search space (SR8x8 only)
# -------------------------
def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    hp["tv_lam_switch"] = trial.suggest_int("tv_lam_switch", 1, 2)
    hp["tv_lam_hi"] = trial.suggest_float("tv_lam_hi", 1e-7, 5e-4, log=True)
    hp["tv_lam_lo"] = trial.suggest_float("tv_lam_lo", 1e-8, 5e-5, log=True)
    hp["tv_iters"] = trial.suggest_int("tv_iters", 30, 180)
    hp["tv_lr"] = trial.suggest_float("tv_lr", 1e-4, 2e-2, log=True)
    hp["latino_cg_steps"] = trial.suggest_int("latino_cg_steps", 2, 25)
    hp["latino_cg_epsilon"] = trial.suggest_categorical(
        "latino_cg_epsilon", [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
    )
    hp["tv_lam_lo"] = min(hp["tv_lam_lo"], hp["tv_lam_hi"])
    return hp


def _ensure_sqlite_parent_dir(storage_uri: str) -> None:
    """
    If storage_uri is sqlite file-based, ensure the parent directory exists.
    Accepts:
      - sqlite:////abs/path/to.db
      - sqlite:///rel/path/to.db
    """
    if not storage_uri.startswith("sqlite:"):
        return

    if storage_uri.startswith("sqlite:////"):
        db_path = storage_uri[len("sqlite:////"):]
        db_path = os.path.abspath(db_path)
    elif storage_uri.startswith("sqlite:///"):
        db_path = storage_uri[len("sqlite:///"):]
        db_path = os.path.abspath(db_path)
    else:
        # other sqlite forms (e.g., :memory:) not handled
        return

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def _write_sparse_frames_for_vidue(
    *,
    y: torch.Tensor,                # [T_lr,3,h,w] in [0,1]
    vid_id: int,
    temporal_factor: int,
    interp_input_root: str,
) -> Tuple[str, str]:
    """
    Writes sparse frames into a sandbox:
        interp_input_root/_current_vidue/<vid_id>/
    Returns:
      (data_path_root, vid_dir_sparse)
    where data_path_root is what should be passed to VIDUE as --data_path.
    """
    data_path_root = os.path.join(interp_input_root, "_current_vidue")
    vid_dir_sparse = os.path.join(data_path_root, f"{vid_id:05d}")

    # clean per-video folder to avoid VIDUE mixing runs
    shutil.rmtree(vid_dir_sparse, ignore_errors=True)
    os.makedirs(vid_dir_sparse, exist_ok=True)

    for t in range(y.shape[0]):
        np_img = tensor_to_uint8_img(y[t])
        save_png(os.path.join(vid_dir_sparse, f"{t * temporal_factor:06d}.png"), np_img)

    return data_path_root, vid_dir_sparse


@torch.no_grad()
def build_cache_for_video(
    vid_id: int,
    frames_root: str,
    out_dir: str,
    pipe: PipelineLVTINO,
    pipe_img,
    forward_model: TemporalSRThenSROp,
    device: torch.device,
    work_dtype: torch.dtype,
    text_embeddings,
    added_cond_kwargs,
    prompt: str,
    T: int,
    fps: int,
    lpips_batch: int,
    interp_input_root: str,
    interp_result_root: str,
    infer_script: str,
    infer_model_path: str,
    infer_default_data: str,
    infer_n_outputs: int,
    infer_border: bool,
    infer_m: int,
    infer_n: int,
    temporal_factor: int,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    frames_dir = os.path.join(frames_root, str(vid_id))
    if not os.path.isdir(frames_dir):
        frames_dir = os.path.join(frames_root, f"{vid_id:05d}")

    x_gt = load_video_from_frames(frames_dir, device=device, dtype=work_dtype, T=T)
    y = forward_model(x_gt.to(dtype=torch.float32)).to(dtype=work_dtype)

    # Write sparse observations into a sandbox that contains ONLY this video
    data_path_root, vid_dir_sparse = _write_sparse_frames_for_vidue(
        y=y, vid_id=vid_id, temporal_factor=temporal_factor, interp_input_root=interp_input_root
    )

    # Run VIDUE interpolation in-process
    run_vidue_interpolation_inprocess(
        infer_script=infer_script,
        default_data=infer_default_data,
        data_path=data_path_root,
        model_path=infer_model_path,
        result_path=interp_result_root,
        n_outputs=infer_n_outputs,
        m=infer_m,
        n=infer_n,
        border=infer_border,
        save_image=True,
    )

    # Read results
    vid_dir_frames = os.path.join(interp_result_root, f"{vid_id:05d}")
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    all_paths = [p for p in glob.glob(os.path.join(vid_dir_frames, "*")) if os.path.splitext(p)[1].lower() in img_exts]
    frame_paths = sorted(all_paths, key=numeric_key)

    if len(frame_paths) < (T - 1):
        raise RuntimeError(f"Expected at least {T - 1} frames in {vid_dir_frames}, found {len(frame_paths)}.")

    frame_paths_Tm1 = frame_paths[: T - 1]

    x_init_frames: List[np.ndarray] = []
    for p in frame_paths_Tm1:
        img = imageio.imread(p)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        x_init_frames.append(img)
    x_init_frames.append(x_init_frames[-1].copy())

    x_init = torch.stack(
        [torch.from_numpy(im).permute(2, 0, 1).to(device=device, dtype=work_dtype) / 255.0 for im in x_init_frames],
        dim=0,
    )

    # Clean sparse input folder
    shutil.rmtree(vid_dir_sparse, ignore_errors=True)

    # Upsample temporally-interpolated low-res init to HR
    x_init_up = F.interpolate(
        x_init.to(dtype=torch.float32),
        size=(x_gt.shape[2], x_gt.shape[3]),
        mode="bilinear",
        align_corners=False,
    )

    # Prepare inference inputs
    x_init_lat = 2 * x_init_up.unsqueeze(0).clamp(0, 1) - 1
    y_n = 2 * y - 1

    z0_hat = pipe.vae.encode_from_pixel(x_init_lat)
    next_timestep = pipe.denoising_step_list[0] * torch.ones(z0_hat.shape[:2], dtype=torch.long, device=z0_hat.device)
    noisy_image_or_video = pipe.scheduler.add_noise(
        z0_hat.flatten(0, 1),
        torch.randn_like(z0_hat.flatten(0, 1)),
        next_timestep.flatten(0, 1),
    ).unflatten(0, z0_hat.shape[:2])

    return {"vid_id": vid_id, "x_gt": x_gt, "y_n": y_n, "noisy": noisy_image_or_video, "prompt": prompt, "out_dir": out_dir}


@torch.no_grad()
def run_one_cached(
    cache: Dict[str, Any],
    pipe: PipelineLVTINO,
    pipe_img,
    forward_model: TemporalSRThenSROp,
    device: torch.device,
    work_dtype: torch.dtype,
    text_embeddings,
    added_cond_kwargs,
    fps: int,
    lpips_loss,
    psnr_loss,
    ssim_loss,
    lpips_batch: int,
    save_outputs: bool,
) -> Tuple[float, float, float]:
    x_gt = cache["x_gt"]
    y_n = cache["y_n"]
    noisy = cache["noisy"]
    prompt = cache["prompt"]
    out_dir = cache["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    out = pipe.inference(
        pipe_img=pipe_img,
        x_gt=x_gt,
        text_embeddings=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
        noise=noisy,
        y=y_n,
        forward_model=forward_model,
        output_folder=out_dir,
        fps=fps,
        text_prompts=[prompt],
        save_intermediates=save_outputs,
    )
    restored = out[0]

    if save_outputs:
        export_to_video(
            restored.permute(0, 2, 3, 1).detach().cpu().numpy(),
            os.path.join(out_dir, "output.mp4"),
            fps=fps,
        )

    Tm = min(x_gt.shape[0], restored.shape[0])
    x_ref = x_gt[:Tm].to(device, dtype=torch.float32)
    y_ref = restored[:Tm].to(device, dtype=torch.float32)

    psnr = psnr_loss(x_ref, y_ref).item()
    ssim = ssim_loss(x_ref, y_ref).item()
    lp = lpips_video_mean(lpips_loss, x_ref, y_ref, device=device, batch_size=lpips_batch)
    return psnr, ssim, lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--T", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--lpips_batch", type=int, default=16)

    parser.add_argument("--sr_factor", type=int, default=8)
    parser.add_argument("--temporal_factor", type=int, default=8)

    parser.add_argument("--mod", type=int, default=1)
    parser.add_argument("--min_id", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)

    parser.add_argument("--interp_input_root", type=str, default="./temp/interp_input")
    parser.add_argument("--interp_result_root", type=str, default="./temp/interp_result")
    parser.add_argument("--infer_script", type=str, default="./VIDUE/code/inference_vidue_worsu.py")
    parser.add_argument("--infer_model_path", type=str, default="./VIDUE/pretrained_model/model_best_gopro.pt")
    parser.add_argument("--infer_default_data", type=str, default="Adobe")
    parser.add_argument("--infer_n_outputs", type=int, default=8)
    parser.add_argument("--infer_border", action="store_true", default=True)
    parser.add_argument("--infer_m", type=int, default=7)
    parser.add_argument("--infer_n", type=int, default=1)

    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--optuna_only", action="store_true")
    parser.add_argument("--optuna_trials", type=int, default=50)
    parser.add_argument("--optuna_storage", type=str, default="")
    parser.add_argument("--optuna_study", type=str, default="lvtino_vidue")
    parser.add_argument("--optuna_seed", type=int, default=0)
    parser.add_argument("--optuna_objective", type=str, default="lpips", choices=["psnr", "ssim", "lpips", "combo"])
    parser.add_argument("--optuna_combo_w_psnr", type=float, default=1.0)
    parser.add_argument("--optuna_combo_w_ssim", type=float, default=100.0)
    parser.add_argument("--optuna_combo_w_lpips", type=float, default=10.0)
    parser.add_argument("--optuna_videos", type=int, default=1)
    parser.add_argument("--optuna_T", type=int, default=25)
    parser.add_argument("--optuna_save_videos", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_folder, "metrics.csv")
    csv_exists = os.path.isfile(metrics_csv_path)

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

    # SDXL backbone for embeddings
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

    config = OmegaConf.load(args.config_path)
    pipe = PipelineLVTINO(config, device=str(device))
    state = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipe.generator.load_state_dict(state)
    pipe = pipe.to(device=device, dtype=work_dtype)

    dataset = TextDataset(args.prompt_file_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in {args.prompt_file_path}")
    prompt = dataset[0]

    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device).eval()
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()

    if not csv_exists:
        with open(metrics_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["video_id", "frames_used", "psnr", "ssim", "lpips", "output_path"])

    def is_valid_subdir(name: str) -> bool:
        return name.isdigit() and (int(name) % args.mod == 0) and (int(name) >= args.min_id)

    subdirs = sorted([d for d in os.listdir(args.frames_root) if is_valid_subdir(d)], key=lambda s: int(s))
    video_ids = [int(d) for d in subdirs]
    if args.max_videos and args.max_videos > 0:
        video_ids = video_ids[: int(args.max_videos)]

    forward_model = TemporalSRThenSROp(
        factor_t=args.temporal_factor, factor=args.sr_factor, filter="bicubic", padding="reflect"
    )

    # -------------------------
    # Optuna
    # -------------------------
    if args.optuna:
        tune_ids = video_ids[: int(args.optuna_videos)]
        if len(tune_ids) == 0:
            raise RuntimeError("No tune videos selected.")

        caches: List[Dict[str, Any]] = []
        for vid in tune_ids:
            caches.append(
                build_cache_for_video(
                    vid_id=vid,
                    frames_root=args.frames_root,
                    out_dir=os.path.join(args.output_folder, "_optuna_cache", f"{vid:05d}"),
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
                    lpips_batch=args.lpips_batch,
                    interp_input_root=args.interp_input_root,
                    interp_result_root=args.interp_result_root,
                    infer_script=args.infer_script,
                    infer_model_path=args.infer_model_path,
                    infer_default_data=args.infer_default_data,
                    infer_n_outputs=args.infer_n_outputs,
                    infer_border=args.infer_border,
                    infer_m=args.infer_m,
                    infer_n=args.infer_n,
                    temporal_factor=args.temporal_factor,
                )
            )

        def objective(trial: optuna.Trial) -> float:
            hp = suggest_hparams(trial)
            hp["save_intermediates"] = bool(args.optuna_save_videos)
            pipe.hparams = hp

            scores: List[float] = []
            for cache in caches:
                vid = int(cache["vid_id"])
                cache_local = dict(cache)
                cache_local["out_dir"] = os.path.join(
                    args.output_folder, "_optuna_tmp", f"trial_{trial.number:04d}", f"{vid:05d}"
                )

                psnr, ssim, lp = run_one_cached(
                    cache=cache_local,
                    pipe=pipe,
                    pipe_img=pipe_img,
                    forward_model=forward_model,
                    device=device,
                    work_dtype=work_dtype,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    fps=args.fps,
                    lpips_loss=lpips_loss,
                    psnr_loss=psnr_loss,
                    ssim_loss=ssim_loss,
                    lpips_batch=args.lpips_batch,
                    save_outputs=bool(args.optuna_save_videos),
                )

                if args.optuna_objective == "psnr":
                    scores.append(psnr)
                elif args.optuna_objective == "ssim":
                    scores.append(ssim)
                elif args.optuna_objective == "lpips":
                    scores.append(-lp)  # maximize => minimize lpips
                else:
                    scores.append(
                        float(args.optuna_combo_w_psnr) * psnr
                        + float(args.optuna_combo_w_ssim) * ssim
                        - float(args.optuna_combo_w_lpips) * lp
                    )

            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=int(args.optuna_seed))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

        kwargs = dict(
            study_name=str(args.optuna_study),
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        if args.optuna_storage:
            _ensure_sqlite_parent_dir(str(args.optuna_storage))
            kwargs["storage"] = str(args.optuna_storage)

        study = optuna.create_study(**kwargs)
        study.optimize(objective, n_trials=int(args.optuna_trials), gc_after_trial=True)

        best_hp = dict(study.best_trial.params)
        best_hp["save_intermediates"] = True
        best_path = os.path.join(args.output_folder, "optuna_best_hparams.json")
        with open(best_path, "w") as f:
            json.dump(best_hp, f, indent=2)

        print(f"[OPTUNA] Best value: {study.best_value}")
        print(f"[OPTUNA] Best params saved to: {best_path}")

        if args.optuna_only:
            return

        pipe.hparams = best_hp

    # -------------------------
    # Full run
    # -------------------------
    for vid_id in tqdm(video_ids, desc="Processing videos"):
        frames_dir = os.path.join(args.frames_root, str(vid_id))
        if not os.path.isdir(frames_dir):
            frames_dir = os.path.join(args.frames_root, f"{vid_id:05d}")
        out_dir = os.path.join(args.output_folder, f"{vid_id:05d}")
        os.makedirs(out_dir, exist_ok=True)

        x_gt = load_video_from_frames(frames_dir, device=device, dtype=work_dtype, T=args.T)
        save_video_tensor(x_gt.float(), os.path.join(out_dir, "gt.mp4"), fps=args.fps)

        y = forward_model(x_gt.to(dtype=torch.float32)).to(dtype=work_dtype)
        save_video_tensor(y.float(), os.path.join(out_dir, "observed_y.mp4"), fps=2)

        # per-video sandbox inputs for VIDUE
        data_path_root, vid_dir_sparse = _write_sparse_frames_for_vidue(
            y=y, vid_id=vid_id, temporal_factor=args.temporal_factor, interp_input_root=args.interp_input_root
        )

        run_vidue_interpolation_inprocess(
            infer_script=args.infer_script,
            default_data=args.infer_default_data,
            data_path=data_path_root,
            model_path=args.infer_model_path,
            result_path=args.interp_result_root,
            n_outputs=args.infer_n_outputs,
            m=args.infer_m,
            n=args.infer_n,
            border=args.infer_border,
            save_image=True,
        )

        vid_dir_frames = os.path.join(args.interp_result_root, f"{vid_id:05d}")
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        all_paths = [p for p in glob.glob(os.path.join(vid_dir_frames, "*")) if os.path.splitext(p)[1].lower() in img_exts]
        frame_paths = sorted(all_paths, key=numeric_key)

        if len(frame_paths) < (args.T - 1):
            raise RuntimeError(f"Expected at least {args.T - 1} frames in {vid_dir_frames}, found {len(frame_paths)}.")

        frame_paths_Tm1 = frame_paths[: args.T - 1]

        x_init_frames: List[np.ndarray] = []
        for p in frame_paths_Tm1:
            img = imageio.imread(p)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            x_init_frames.append(img)
        x_init_frames.append(x_init_frames[-1].copy())

        x_init = torch.stack(
            [torch.from_numpy(im).permute(2, 0, 1).to(device=device, dtype=work_dtype) / 255.0 for im in x_init_frames],
            dim=0,
        )
        save_video_tensor(x_init.float(), os.path.join(out_dir, "x_init_lowres.mp4"), fps=args.fps)

        shutil.rmtree(vid_dir_sparse, ignore_errors=True)

        x_init_up = F.interpolate(
            x_init.to(dtype=torch.float32),
            size=(x_gt.shape[2], x_gt.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        save_video_tensor(x_init_up.float(), os.path.join(out_dir, "x_init_up.mp4"), fps=args.fps)

        x_init_lat = 2 * x_init_up.unsqueeze(0).clamp(0, 1) - 1
        y_n = 2 * y - 1

        z0_hat = pipe.vae.encode_from_pixel(x_init_lat)
        next_timestep = pipe.denoising_step_list[0] * torch.ones(z0_hat.shape[:2], dtype=torch.long, device=z0_hat.device)
        noisy_image_or_video = pipe.scheduler.add_noise(
            z0_hat.flatten(0, 1),
            torch.randn_like(z0_hat.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, z0_hat.shape[:2])

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
                save_intermediates=True,
            )

        restored = out[0]
        export_to_video(
            restored.permute(0, 2, 3, 1).detach().cpu().numpy(),
            os.path.join(out_dir, "output.mp4"),
            fps=args.fps,
        )

        Tm = min(x_gt.shape[0], restored.shape[0])
        x_ref = x_gt[:Tm].to(device, dtype=torch.float32)
        y_ref = restored[:Tm].to(device, dtype=torch.float32)

        psnr = psnr_loss(x_ref, y_ref).item()
        ssim = ssim_loss(x_ref, y_ref).item()
        lp = lpips_video_mean(lpips_loss, x_ref, y_ref, device=device, batch_size=args.lpips_batch)

        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([vid_id, Tm, f"{psnr:.6f}", f"{ssim:.6f}", f"{lp:.6f}", os.path.join(out_dir, "output.mp4")])

        del x_gt, y, x_init, x_init_up, x_init_lat, z0_hat, noisy_image_or_video, out, restored, x_ref, y_ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
