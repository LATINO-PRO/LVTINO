from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
)
from typing import List
import torch
import os
from diffusers.utils import export_to_video

from utils import TemporalSRThenSROp


# -------------------------
# IO helpers
# -------------------------
def save_video_tensor(tvid: torch.Tensor, out_path: str, fps: int = 16):
    """
    tvid: (T, C, H, W), values in [0,1]. Saves MP4.
    """
    tv = tvid.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()  # (T,H,W,C)
    export_to_video(tv, out_path, fps=fps)


# -------------------------
# CG utilities (LATINO_img)
# -------------------------
def _flatten_for_dot(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:      # (T,C,H,W)
        return x.reshape(1, -1)
    elif x.dim() == 5:    # (B,T,C,H,W)
        B = x.shape[0]
        return x.reshape(B, -1)
    else:
        return x.reshape(1, -1)


@torch.no_grad()
def cg_solve_normal_eq(
    forward_model: TemporalSRThenSROp,
    x_init: torch.Tensor,         # (T,C,H,W) in latent-decoded space [-1,1] typically
    y: torch.Tensor,              # measurement (T_lr,C,h,w)
    n_inner: int = 5,
    epsilon: float = 0.0,
    T_out: int = None,
    eps_stop: float = 1e-6,
) -> torch.Tensor:
    """
    Solve via CG:
        argmin_x 0.5||A x - y||^2 + (epsilon/2)||x - x_init||^2
    Normal equations:
        (A^T A + epsilon I) x = A^T y + epsilon x_init

    Returns: fp32 tensor (T,C,H,W).
    """
    # A: HR -> LR
    A_lr = lambda z: forward_model.A(z.to(torch.float32))

    # A^T: LR -> HR (for SR8x8 temporal op, we must pass T_out)
    if T_out is None:
        raise ValueError("For SR8x8 (TemporalSRThenSROp), T_out must be provided to A_adjoint.")
    AT_hr = lambda z: forward_model.A_adjoint(z.to(torch.float32), T_out=T_out)

    # A^T A: HR -> HR
    A_cg = lambda z: AT_hr(A_lr(z))

    def dot(u, v):
        uf = _flatten_for_dot(u).to(torch.float64)
        vf = _flatten_for_dot(v).to(torch.float64)
        return (uf * vf).sum(dim=-1)

    with torch.cuda.amp.autocast(enabled=False):
        x = x_init.to(torch.float32)
        b = AT_hr(y).to(torch.float32)
        if epsilon != 0.0:
            b = b + float(epsilon) * x

        def Apply(z):
            out = A_cg(z)
            if epsilon != 0.0:
                out = out + float(epsilon) * z
            return out.to(torch.float32)

        r = b - Apply(x)
        p = r.clone()
        rsold = dot(r, r)

        for _ in range(n_inner):
            Ap = Apply(p)
            denom = dot(p, Ap)
            bad = (~torch.isfinite(denom)) | (denom.abs() < 1e-20)
            if bad.any():
                break

            alpha = rsold / denom
            view = [-1] + [1] * (x.dim() - 1)
            x = x + alpha.view(view) * p
            r = r - alpha.view(view) * Ap

            rsnew = dot(r, r)
            if not torch.isfinite(rsnew).all():
                break
            if torch.sqrt(rsnew.max()) < eps_stop:
                break

            beta = rsnew / rsold
            p = r + beta.view(view) * p
            rsold = rsnew

        return x


# -------------------------
# LATINO step (SR8x8 only)
# -------------------------
@torch.no_grad()
def LATINO_img(
    video: torch.Tensor,          # (T,C,H,W) GT only used for T_out
    pipe,                         # SDXL pipe (unet, vae, scheduler)
    text_embeddings,
    added_cond_kwargs,
    zt: torch.Tensor,             # (T,4,h/8,w/8)
    x0_hat: torch.Tensor,         # (1,T,3,H,W) not used directly here, kept for API compatibility
    measurement: torch.Tensor,    # y in measurement space (T_lr,C,h,w)
    timestep: int,
    index: int,
    forward_model: TemporalSRThenSROp,
    CG_step: int = 5,
    epsilon: float = 0.0,
):
    """
    SR8x8 LATINO update:
      1) decode per-frame x0_t from z_t
      2) data-consistency via CG on normal equations using forward_model (TemporalSRThenSROp)
         with A_adjoint(..., T_out=video.shape[0])
    """
    forward_model = forward_model.to(dtype=torch.float32)

    # Decode x0_t frames from z_t
    t = timestep
    at = pipe.scheduler.alphas_cumprod[t]

    x0t_frames = []
    shared_noise = torch.randn_like(zt[0])  # fixed noise for all frames

    for i in range(video.shape[0]):
        zt[i] = pipe.scheduler.add_noise(
            zt[i],
            noise=shared_noise,
            timesteps=torch.tensor([t], device=zt.device)
        )

    for frame in range(video.shape[0]):
        zt_frame = zt[frame].unsqueeze(0)
        noise_pred = pipe.unet(
            zt_frame,
            timestep,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        z0t = (zt_frame - (1 - at).sqrt() * noise_pred) / at.sqrt()
        decoded = pipe.vae.decode(z0t / pipe.vae.config.scaling_factor).sample
        x0t_frames.append(decoded)

    x0t_frames = torch.cat(x0t_frames, dim=0)  # (T,3,H,W) in [-1,1]

    # Data-consistency (CG normal equations)
    x_dc = cg_solve_normal_eq(
        forward_model=forward_model,
        x_init=x0t_frames,
        y=measurement,
        n_inner=CG_step,
        epsilon=epsilon,
        T_out=video.shape[0],
    )

    return x_dc.to(dtype=x0t_frames.dtype)


# -------------------------
# Main pipeline (SR8x8 only)
# -------------------------
class PipelineLVTINO(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.generator_model_name = getattr(args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device
        )

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(
        self,
        pipe_img,
        x_gt,
        text_embeddings,
        added_cond_kwargs,
        noise: torch.Tensor,
        text_prompts: List[str],
        y,
        forward_model: TemporalSRThenSROp,
        output_folder,
        fps
    ) -> torch.Tensor:
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        noisy_image_or_video = noise
        timesteps_img = pipe_img.scheduler.timesteps

        for index, current_timestep in enumerate(self.denoising_step_list):

            if index > 0:
                zt = torch.zeros(
                    (x0_hat.shape[1], 4, x0_hat.shape[3] // 8, x0_hat.shape[4] // 8),
                    device=x_gt.device,
                    dtype=torch.float16
                )
                for i in range(x0_hat.shape[1]):
                    zt[i] = pipe_img.vae.encode(
                        x0_hat.squeeze(0)[i].unsqueeze(0).to(dtype=torch.float16).clip(-1, 1)
                    ).latent_dist.mean * pipe_img.vae.config.scaling_factor

                x0_LATINO = LATINO_img(
                    video=x_gt,
                    pipe=pipe_img,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    zt=zt,
                    x0_hat=x0_hat,
                    measurement=y,
                    timestep=timesteps_img[index],
                    index=index,
                    forward_model=forward_model,
                    CG_step=23,
                    epsilon=0.0,
                )

                save_video_tensor(
                    x0_LATINO.float() * 0.5 + 0.5,
                    os.path.join(output_folder, f"LATINO_step_{timesteps_img[index]}.mp4"),
                    fps=fps
                )

                x0_hat = x0_LATINO.unsqueeze(0).to(dtype=torch.bfloat16)

                z0_hat = self.vae.encode_from_pixel(x0_hat)
                next_timestep = self.denoising_step_list[index] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device
                )

                noisy_image_or_video = self.scheduler.add_noise(
                    z0_hat.flatten(0, 1),
                    torch.randn_like(z0_hat.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])

            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )

            x0 = self.vae.decode_to_pixel(pred_image_or_video)
            save_video_tensor(
                (x0.squeeze(0).float() * 0.5 + 0.5).clamp(0, 1),
                os.path.join(output_folder, f"x0_{current_timestep}.mp4"),
                fps=fps
            )

            # Data-consistency update used in this SR8x8 pipeline:
            # TV3-regularized solve
            forward_model = forward_model.to(dtype=torch.float32)


            with torch.enable_grad():
                x0_hat = forward_model.solve_l2_tv3_adam_autograd(
                    x_init=x0[0].to(dtype=torch.float32),
                    y=y.to(dtype=torch.float32),
                    lam=6.178523568938361e-07 if index < 1 else 1.5583416542691956e-06,
                    iters=55,
                    lr=0.019848,
                    log_every=0     # do not log
                ).unsqueeze(0).to(dtype=x0.dtype)
            save_video_tensor(
                (x0_hat.squeeze(0).float() * 0.5 + 0.5).clamp(0, 1),
                os.path.join(output_folder, f"x0_hat_{current_timestep}.mp4"),
                fps=fps
            )

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video
