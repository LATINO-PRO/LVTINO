from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch
import os
from diffusers.utils import export_to_video


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
# Small utils
# -------------------------
def _infer_forward_op_name(forward_model) -> str:
    """
    Best-effort operator identifier from class name.

    Returns one of:
      - 'tblur_sr'   : TemporalBlurThenSROp  (SRx*)
      - 'tsr_sr'     : TemporalSRThenSROp    (SR4x4)
      - 'noise_jpeg' : GaussianNoiseJPEGOp  (JPEG)
      - 'unknown'
    """
    cls = forward_model.__class__.__name__.lower()
    if "temporalblurthensrop" in cls or ("blur" in cls and "sr" in cls):
        return "tblur_sr"
    if "temporalsrthensrop" in cls or ("temporalsr" in cls):
        return "tsr_sr"
    if "gaussiannoisejpegop" in cls or ("jpeg" in cls):
        return "noise_jpeg"
    return "unknown"


def _flatten_for_dot(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:      # (T,C,H,W)
        return x.reshape(1, -1)
    elif x.dim() == 5:    # (B,T,C,H,W)
        B = x.shape[0]
        return x.reshape(B, -1)
    else:
        return x.reshape(1, -1)


@torch.no_grad()
def CG_safe(A_fn, b, x, n_inner=5, eps=1e-6, lam=0.0, diagnostics=False):
    """
    Conjugate Gradient for SPD systems: (A_fn + lam*I) x = b.

    - Forces fp32 in the loop; fp64 for dot products.
    - 'lam' is added inside the matvec to stabilize.
    """
    with torch.cuda.amp.autocast(enabled=False):
        dev = b.device
        x = x.to(dev, torch.float32)
        b = b.to(dev, torch.float32)

        def Apply(z):
            z2 = A_fn(z.to(torch.float32))
            if lam != 0.0:
                z2 = z2 + float(lam) * z
            return z2.to(torch.float32)

        Ax0 = Apply(x)
        if Ax0.shape != b.shape or Ax0.shape != x.shape:
            raise RuntimeError(f"Shape mismatch: A(x)={Ax0.shape}, b={b.shape}, x={x.shape}")

        r = b - Ax0
        p = r.clone()

        def dot(u, v):
            uf = _flatten_for_dot(u).to(torch.float64)
            vf = _flatten_for_dot(v).to(torch.float64)
            return (uf * vf).sum(dim=-1)  # [B] or [1]

        rsold = dot(r, r)
        if not torch.isfinite(rsold).all():
            raise RuntimeError("CG: non-finite initial residual. Check A_fn/b inputs.")

        for i in range(n_inner):
            Ap = Apply(p)
            denom = dot(p, Ap)

            bad = (~torch.isfinite(denom)) | (denom.abs() < 1e-20)
            if bad.any():
                if lam == 0.0:
                    # try tiny regularization
                    lam = 1e-6
                    Ap = Apply(p)
                    denom = dot(p, Ap)
                    bad = (~torch.isfinite(denom)) | (denom.abs() < 1e-20)
                if bad.any():
                    if diagnostics:
                        print(f"CG breakdown at iter {i}: denom={denom}")
                    break

            alpha = rsold / denom
            view = [-1] + [1] * (x.dim() - 1)
            x = x + alpha.view(view) * p
            r = r - alpha.view(view) * Ap

            rsnew = dot(r, r)
            if not torch.isfinite(rsnew).all():
                if diagnostics:
                    print(f"CG encountered non-finite rsnew at iter {i}.")
                break

            if torch.sqrt(rsnew.max()) < eps:
                break

            beta = rsnew / rsold
            p = r + beta.view(view) * p
            rsold = rsnew

        return x  # fp32


def data_consistency_cg(forward_model, x_init, measurement, op_name, T_out, n_inner=5, epsilon=0.0):
    """
    Solve: argmin_x 0.5||A x - y||^2 + (epsilon/2)||x - x_init||^2
    Normal eq.: (A^T A + epsilon I) x = A^T y + epsilon x_init
    """
    # Expect forward_model provides A and A_adjoint
    A_lr  = lambda z: forward_model.A(z.to(torch.float32))
    AT_hr = lambda z: forward_model.A_adjoint(z.to(torch.float32), T_out=T_out) if op_name == "tsr_sr" else lambda z: forward_model.A_adjoint(z.to(torch.float32))
    A_cg  = lambda z: AT_hr(A_lr(z))  # HR -> HR

    x0_hr = x_init.to(torch.float32)
    y_hr  = measurement

    b = AT_hr(y_hr).to(torch.float32)
    if epsilon != 0.0:
        b = b + float(epsilon) * x0_hr

    x_cg = CG_safe(A_cg, b, x0_hr, n_inner=n_inner, eps=1e-6, lam=epsilon)
    return x_cg


def _apply_data_fidelity_inference(forward_model, x0_f32, y_f32, op_name: str, index: int) -> torch.Tensor:
    """
    Data-fidelity step used in PipelineLVTINO.inference():
      - SRx*   (tblur_sr): prox_l2 with gamma schedule
      - JPEG   (noise_jpeg): prox_l2 gamma=1
      - SR4x4  (tsr_sr): solve_l2_tv with lam=0.005
    """
    if op_name == "tblur_sr":
        gamma = 3539 if index < 2 else 1363
        return forward_model.prox_l2(x0_f32, y=y_f32, gamma=gamma)

    if op_name == "noise_jpeg":
        return forward_model.prox_l2(x0_f32, y=y_f32, gamma=0.6026421242101478)

    if op_name == "tsr_sr":
        return forward_model.solve_l2_tv(x0_f32, y=y_f32, lam=0.0002873789)
    
    # fallback
    if hasattr(forward_model, "prox_l2"):
        return forward_model.prox_l2(x0_f32, y=y_f32, gamma=200)
    return x0_f32


def _apply_data_fidelity_latino(forward_model, x0t_frames_f32, measurement_f32, op_name: str, index: int,
                                data_consistency_cg_step: int = 10, epsilon: float = 0.0, T_out: int = None,) -> torch.Tensor:
    """
    Data-fidelity step used inside LATINO_img:
      - SR4x4 (tsr_sr): CG
      - SRx*  (tblur_sr): prox_l2 with larger gamma schedule (20000/200)
      - JPEG  (noise_jpeg): prox_l2 gamma=1
    """
    if op_name == "tsr_sr":
        data_consistency_cg_step = 19
        epsilon = 1e-6
        return data_consistency_cg(forward_model, x0t_frames_f32, measurement_f32, op_name, T_out, n_inner=data_consistency_cg_step, epsilon=epsilon)

    if op_name == "tblur_sr":
        gamma = 16868 if index < 3 else 2721
        return forward_model.prox_l2(x0t_frames_f32, y=measurement_f32, gamma=gamma)

    if op_name == "noise_jpeg":
        return forward_model.prox_l2(x0t_frames_f32, y=measurement_f32, gamma=0.518751616460249)
    
    # fallback
    if hasattr(forward_model, "prox_l2"):
        return forward_model.prox_l2(x0t_frames_f32, y=measurement_f32, gamma=200)
    return x0t_frames_f32


# -------------------------
# LATINO step
# -------------------------
@torch.no_grad()
def LATINO_img(
        video,
        pipe,
        text_embeddings,
        added_cond_kwargs,
        zt,
        x0_hat,
        measurement,
        timestep,
        index,
        forward_model,              # <-- pass the SAME forward_model from caller
        forward_operator: Optional[str] = None,  # <-- optional explicit name
        data_consistency_cg_step=10,
        epsilon=0.0,                # regularization
):
    """
    LATINO single iteration returning updated_x0t.

    IMPORTANT: forward_model is passed in. Data-fidelity is chosen automatically based on operator.
    """

    op_name = forward_operator or _infer_forward_op_name(forward_model)

    # Make sure we run operators in fp32
    forward_model = forward_model.to(dtype=torch.float32)

    # pseudo-batch consistent sampling
    t = timestep
    at = pipe.scheduler.alphas_cumprod[t]
    x0t_frames = []

    noise = torch.randn_like(zt[0])  # fixed noise for all frames
    for i in range(video.shape[0]):
        zt[i] = pipe.scheduler.add_noise(zt[i], noise=noise, timesteps=torch.tensor([t], device=zt.device))

    for frame in range(video.shape[0]):
        zt_frame = zt[frame].unsqueeze(0)
        noise_pred = pipe.unet(
            zt_frame,
            timestep,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        z0t = (zt_frame - (1 - at).sqrt() * noise_pred) / at.sqrt()
        decoded_z = pipe.vae.decode(z0t / pipe.vae.config.scaling_factor).sample
        x0t_frames.append(decoded_z)

    x0t_frames = torch.cat(x0t_frames, dim=0)  # (T,3,H,W) in [-1,1] domain (SD decode space)

    # -------------------------
    # Data-fidelity step
    # -------------------------
    x0t_f32 = x0t_frames.to(torch.float32)
    meas_f32 = measurement.to(torch.float32)

    updated_x0t = _apply_data_fidelity_latino(
        forward_model=forward_model,
        x0t_frames_f32=x0t_f32,
        measurement_f32=meas_f32,
        op_name=op_name,
        index=index,
        data_consistency_cg_step=data_consistency_cg_step,
        epsilon=epsilon,
        T_out=video.shape[0],
    ).to(dtype=x0t_frames.dtype)

    return updated_x0t


# -------------------------
# Main pipeline
# -------------------------
class PipelineLVTINO(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device
        )

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Optional explicit operator name coming from CLI/config:
        # 'tblur_sr' (SRx*), 'tsr_sr' (SR4x4), 'noise_jpeg' (JPEG)
        self.forward_operator = getattr(args, "forward_operator", None)

    def inference(self, pipe_img, x_gt, text_embeddings, added_cond_kwargs,
                  noise: torch.Tensor, text_prompts: List[str],
                  y, forward_model, output_folder, fps) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        """
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        noisy_image_or_video = noise
        timesteps_img = pipe_img.scheduler.timesteps

        # determine operator once
        op_name = self.forward_operator or _infer_forward_op_name(forward_model)

        for index, current_timestep in enumerate(self.denoising_step_list):

            # LATINO iteration
            if index > 0:
                zt = torch.zeros(
                    (x0_hat.shape[1], 4, x0_hat.shape[3] // 8, x0_hat.shape[4] // 8),
                    device=x_gt.device, dtype=torch.float16
                )  # (T,4,h/8,w/8)

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
                    forward_model=forward_model,         # <-- same operator instance
                    forward_operator=op_name,            # <-- explicit name
                )

                save_video_tensor(
                    x0_LATINO.float() * 0.5 + 0.5,
                    os.path.join(output_folder, f"LATINO_step_{timesteps_img[index]}.mp4"),
                    fps=fps
                )

                x0_hat = x0_LATINO.unsqueeze(0).to(dtype=torch.bfloat16)  # (1,T,3,H,W)

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
            )  # [B, F, C, H, W]

            x0 = self.vae.decode_to_pixel(pred_image_or_video)
            save_video_tensor(
                (x0.squeeze(0).float() * 0.5 + 0.5).clamp(0, 1),
                os.path.join(output_folder, f"x0_{current_timestep}.mp4"),
                fps=fps
            )

            # -------------------------
            # Data-consistency step
            # -------------------------
            forward_model = forward_model.to(dtype=torch.float32)
            x0_f32 = x0[0].to(dtype=torch.float32)
            y_f32 = y.to(dtype=torch.float32)

            x0_hat = _apply_data_fidelity_inference(
                forward_model=forward_model,
                x0_f32=x0_f32,
                y_f32=y_f32,
                op_name=op_name,
                index=index
            ).unsqueeze(0).to(dtype=x0.dtype)

            save_video_tensor(
                (x0_hat.squeeze(0).float() * 0.5 + 0.5).clamp(0, 1),
                os.path.join(output_folder, f"x0_hat_{current_timestep}.mp4"),
                fps=fps
            )

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video
