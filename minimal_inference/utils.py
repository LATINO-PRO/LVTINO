import os
import argparse
from typing import Optional, List, Literal, Union, Tuple
import torch
from PIL import Image
import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

import deepinv as dinv

import torch.nn.functional as F
from torch import nn
import torchmetrics
import csv

from torchvision import io as tv_io


#########################
#       Operators       #
#########################

def uniform_kernel_1d(kernel_size: int, dtype=torch.float32):
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    k = torch.ones(kernel_size, dtype=dtype)
    return k / k.sum()

def gaussian_kernel_1d(kernel_size: int, sigma: float, dtype=torch.float32):
    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    return k / k.sum()

def _temporal_h_for_fft(kernel_1d: torch.Tensor, T: int) -> torch.Tensor:
    """
    Build length-T cyclic kernel h so that (h ⊛_circ x)[t] == conv with center kernel index K//2.
    """
    k = kernel_1d
    K = k.numel()
    center = K // 2
    h = torch.zeros(T, dtype=k.dtype, device=k.device)
    for i in range(K):
        h[(i - center) % T] = k[i]
    return h  # length T

class TemporalBlurThenSROp(nn.Module):
    """
    A = SR ∘ TemporalBlur.  Works on x: (T,C,H,W) -> y: (T,C,H/f,W/f)
    - Temporal blur: circular conv along time with uniform/gaussian kernel.
    - SR: DeepInv Downsampling (factor, filter, padding).

    Provides:
      A(x), A_adjoint(u), prox_l2(x0, y, gamma), solve_l2_tv(x_init, y, lam, ...)
    """
    def __init__(
        self,
        kernel_size_t: int,
        kernel_type: Literal["uniform", "gaussian"] = "uniform",
        sigma_t: Optional[float] = None,
        factor: int = 8,
        filter: str = "bicubic",
        padding: str = "reflect",
        noise_model=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if kernel_type == "gaussian" and (sigma_t is None or sigma_t <= 0):
            raise ValueError("sigma_t must be provided and > 0 for gaussian kernel.")
        self.kernel_size_t = int(kernel_size_t)
        self.kernel_type   = kernel_type
        self.sigma_t       = sigma_t
        self.factor        = int(factor)
        self.filter        = filter
        self.padding       = padding
        self.noise_model   = noise_model

        self.device_ = device
        self.dtype_  = dtype

        # SR operator is created lazily on first call (needs H,W)
        self._sr = None   # DeepInv Downsampling instance bound to (C,H,W)
        self._hr_size = None  # <- cache (C,H,W)

    # ---------- small utilities ----------
    def _kernel(self, dtype, device):
        if self.kernel_type == "uniform":
            return uniform_kernel_1d(self.kernel_size_t, dtype=dtype).to(device)
        else:
            return gaussian_kernel_1d(self.kernel_size_t, sigma=self.sigma_t, dtype=dtype).to(device)

    @staticmethod
    def _grad_t(x: torch.Tensor) -> torch.Tensor:
        """Forward difference along time: (T-1,C,H,W)"""
        return x[1:] - x[:-1]

    @staticmethod
    def _div_t(p: torch.Tensor) -> torch.Tensor:
        """Adjoint of grad_t (backward diff with Neumann BC)."""
        Tm1, C, H, W = p.shape
        T = Tm1 + 1
        div = torch.zeros((T, C, H, W), device=p.device, dtype=p.dtype)
        div[0]      = -p[0]
        if T > 2:
            div[1:-1] = p[:-1] - p[1:]
        div[-1]     =  p[-1]
        return div

    def _ensure_sr(self, C, H, W, device):
        """(Re)build SR for this exact (C,H,W) and cache size."""
        import deepinv as dinv
        if (self._sr is None) or (self._hr_size is None) or self._hr_size != (C, H, W):
            self._sr = dinv.physics.Downsampling(
                img_size=(C, H, W),    # your DeepInv build wants img_size
                factor=self.factor,
                device=device,
                noise_model=self.noise_model,
                filter=self.filter,
                padding=self.padding,
            )
            self._hr_size = (C, H, W) # cache it ourselves
        return self._sr

    @torch.no_grad()
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T,C,H,W)  ->  y: (T,C,H/f,W/f)
        Steps: temporal FFT blur, then framewise SR downsampling.
        """
        assert x.dim() == 4, f"Expected (T,C,H,W), got {x.shape}"
        dev = self.device_ or x.device
        dty = self.dtype_  or x.dtype
        T, C, H, W = x.shape
        x = x.to(dev, dty)

        # temporal blur via FFT along time (no spatial mixing)
        h  = _temporal_h_for_fft(self._kernel(dty, dev), T)  # (T,)
        Hf = torch.fft.rfft(h)                               # (F,)
        X  = x.permute(1,2,3,0).reshape(C*H*W, T)            # (N,T)
        FX = torch.fft.rfft(X)
        FY = FX * Hf
        Yt = torch.fft.irfft(FY, n=T).view(C, H, W, T).permute(3,0,1,2)  # (T,C,H,W)

        T, C, H, W = x.shape
        dev = self.device_ or x.device
        dty = self.dtype_ or x.dtype

        # ensure SR exists for this (C,H,W) and cache
        self._ensure_sr(C, H, W, dev)

        # framewise SR (batched if supported)
        try:
            y = self._sr.A(Yt.to(dev, dty))                # (T,C,h,w) if batched
        except Exception:
            ys = [self._sr.A(Yt[t]).unsqueeze(0) for t in range(T)]
            y = torch.cat(ys, dim=0)
        return y.to(dty)

    @torch.no_grad()
    def A_adjoint(self, u: torch.Tensor) -> torch.Tensor:
        """u: (T,C,h,w) -> v: (T,C,H,W) then temporal adjoint."""
        assert u.dim() == 4, f"Expected (T,C,h,w), got {u.shape}"
        dev = self.device_ or u.device
        dty = self.dtype_ or u.dtype
        T, C, h, w = u.shape
        u = u.to(dev, dty)

        # if HR size unknown, infer from LR and factor, then build & cache
        if self._hr_size is None:
            H = h * self.factor
            W = w * self.factor
            self._ensure_sr(C, H, W, dev)
        else:
            C_hr, H, W = self._hr_size
            if C_hr != C:
                # Rebuild SR if channels differ (e.g., switching grayscale/RGB)
                H = h * self.factor
                W = w * self.factor
                self._ensure_sr(C, H, W, dev)

        # SR adjoint per frame (batched if supported)
        try:
            vs = self._sr.A_adjoint(u)   # (T,C,H,W) if batched
        except Exception:
            vs = torch.cat([self._sr.A_adjoint(u[t]).unsqueeze(0) for t in range(T)], dim=0)

        # temporal adjoint (time-reversed kernel) via FFT — as before
        k  = self._kernel(dty, dev)
        h_tr = _temporal_h_for_fft(torch.flip(k, dims=(0,)), T)
        Hf = torch.fft.rfft(h_tr)

        V  = vs.permute(1,2,3,0).reshape(C*H*W, T)
        FV = torch.fft.rfft(V)
        FZ = FV * Hf
        Z  = torch.fft.irfft(FZ, n=T).view(C, H, W, T).permute(3, 0, 1, 2)
        return Z.to(dty)

    # ---------- operator API ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(x)

    # ---------- prox_{(gamma/2)||A x - y||^2} via CG ----------
    @torch.no_grad()
    def prox_l2(
        self,
        x0: torch.Tensor,             # (T,C,H,W)
        y:  torch.Tensor,             # (T,C,h,w)  <-- LR measurement space
        gamma: Union[float, torch.Tensor],  # scalar or tensor (broadcastable to x0)
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Solve (I + gamma A^T A) x = x0 + gamma A^T y  via CG.
        (Composition with SR breaks the simple temporal FFT closed-form.)
        """
        assert x0.dim() == 4 and y.dim() == 4
        dev = self.device_ or x0.device
        dty = self.dtype_  or x0.dtype
        x0 = x0.to(dev, dty)
        y  = y.to(dev, dty)

        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=dev, dtype=dty)
        gamma = gamma.to(dev, dty)
        if gamma.dim() == 0:
            gamma = gamma.view(1,1,1,1).expand_as(x0)
        else:
            gamma = gamma.expand_as(x0)

        def M(z):
            return z + gamma * self.A_adjoint(self.A(z))
        b = x0 + gamma * self.A_adjoint(y)

        x = x0.clone()
        r = b - M(x)
        p = r.clone()
        rs = (r*r).sum()
        for _ in range(max_iter):
            Mp = M(p)
            a = rs / ((p*Mp).sum() + 1e-20)
            x = x + a * p
            r = r - a * Mp
            rs_new = (r*r).sum()
            if torch.sqrt(rs_new) < tol:
                break
            p = r + (rs_new/rs) * p
            rs = rs_new
        return x.clamp(-1.0, 1.0)

    # ---------- solve 0.5||A x - y||^2 + lam ||D_t x||_1 (time-TV) ----------
    @torch.no_grad()
    def solve_l2_tv(
        self,
        x_init: torch.Tensor,   # (T,C,H,W)
        y: torch.Tensor,        # (T,C,h,w)  (LR measurement space)
        lam: float,
        iters: int = 200,
        tau: float = 0.25,
        sigma: float = 0.25,
        theta: float = 1.0,
        clamp_m11: bool = True,
        ) -> torch.Tensor:
        """
        PDHG (Chambolle–Pock) with two duals:
          p for f(z)=0.5||z - y||^2  (z in LR space),
          q for g(w)=λ||w||_1 with w = D_t x  (time TV).
        """
        assert x_init.dim() == 4 and y.dim() == 4
        dev = x_init.device
        dty = x_init.dtype

        # shapes
        T, C, H, W = x_init.shape
        _, _, h, w = y.shape

        x  = x_init.clone()
        x_bar = x.clone()

        # Duals
        p = torch.zeros((T, C, h, w), device=dev, dtype=dty)   # dual for Ax in LR space
        q = torch.zeros((T-1, C, H, W), device=dev, dtype=dty) # dual for D_t x in HR space

        for _ in range(iters):
            # p <- prox_{σ f*}(p + σ A x̄) with f(z)=0.5||z - y||^2
            Axbar = self.A(x_bar)                 # (T,C,h,w)
            p = (p + sigma * Axbar - sigma * y) / (1.0 + sigma)

            # q <- prox_{σ g*}(q + σ D_t x̄) with g(w)=λ||w||_1 => clip to [-λ, λ]
            Dt_xbar = self._grad_t(x_bar)         # (T-1,C,H,W)
            q = torch.clamp(q + sigma * Dt_xbar, min=-lam, max=lam)

            # x <- x - τ (A^T p + D_t^T q)
            ATp = self.A_adjoint(p)               # (T,C,H, W)
            DtTq = self._div_t(q)                 # (T,C,H, W)
            x_new = x - tau * (ATp + DtTq)

            # extrapolation
            x_bar = x_new + theta * (x_new - x)
            x = x_new

        if clamp_m11:
            x = x.clamp(-1.0, 1.0)
        return x

def gaussian_kernel_1d(kernel_size: int, sigma: float, dtype=torch.float32, device=None):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive and odd (e.g., 5, 7, 9).")
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    return k  # (K,)

def _temporal_conv_same_fp32(
    x: torch.Tensor,           # (T,C,H,W), any dtype/device
    k1d: torch.Tensor,         # (K,), any dtype/device
    pad_mode: str = "replicate"
) -> torch.Tensor:
    """
    1-D 'same' conv along time with padding, computed in FP32 regardless of input dtype.
    Returns result in x.dtype.
    """
    assert x.dim() == 4, f"expected (T,C,H,W), got {x.shape}"
    T, C, H, W = x.shape
    orig_dtype = x.dtype
    pad = (k1d.numel() - 1) // 2

    # Flatten over (C,H,W) -> batch dimension for conv1d
    x32 = x.to(dtype=torch.float32)
    X   = x32.permute(1, 2, 3, 0).reshape(C * H * W, 1, T)      # (N,1,T)

    k32 = k1d.to(device=x.device, dtype=torch.float32).view(1, 1, -1)  # (1,1,K)

    # Safe padding
    if pad > 0:
        if pad_mode == "reflect":
            # reflect requires T >= 2 and pad < T
            if T >= 2 and pad < T:
                # Manual reflect (avoid CUDA pad kernel pitfalls)
                left  = X[:, :, 1:pad+1].flip(-1)           # reflect from index 1..pad
                right = X[:, :, -pad-1:-1].flip(-1)         # reflect excluding last
                Xp = torch.cat([left, X, right], dim=-1)
            else:
                # fall back to replicate
                left  = X[:, :, :1].expand(-1, -1, pad)
                right = X[:, :, -1:].expand(-1, -1, pad)
                Xp = torch.cat([left, X, right], dim=-1)
        elif pad_mode == "replicate":
            left  = X[:, :, :1].expand(-1, -1, pad)
            right = X[:, :, -1:].expand(-1, -1, pad)
            Xp = torch.cat([left, X, right], dim=-1)
        elif pad_mode == "constant":
            left  = torch.zeros_like(X[:, :, :1]).expand(-1, -1, pad)
            right = torch.zeros_like(X[:, :, :1]).expand(-1, -1, pad)
            Xp = torch.cat([left, X, right], dim=-1)
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")
    else:
        Xp = X

    # Conv in FP32, then restore shape and dtype
    with torch.cuda.amp.autocast(enabled=False):
        Y = torch.nn.functional.conv1d(Xp, k32, stride=1, padding=0)   # (N,1,T)
    y32 = Y.view(C, H, W, T).permute(3, 0, 1, 2)                       # (T,C,H,W)
    return y32.to(orig_dtype)

class TemporalGaussianSRThenSROp(nn.Module):
    """
    A = SpatialSR ∘ TemporalGaussianSR.
      x: (T,C,H,W) -> y: (ceil(T/ft), C, H/f, W/f)

    Temporal SR:
      - Convolve along time with Gaussian kernel (reflect pad, 'same' length)
      - Decimate by ft (take every ft-th sample)
      - Tail padded to T_pad = ceil(T/ft)*ft using replicate; adjoint folds tail back

    Spatial SR: DeepInv Downsampling (factor, filter, padding) in fp32 (safe for kernels).
    """

    def __init__(
        self,
        factor_t: int = 4,
        kernel_size_t: int = 7,
        sigma_t: float = 2.0,
        factor: int = 4,                         # spatial SR factor
        filter: str = "bicubic",
        padding: str = "reflect",
        time_conv_pad_mode: str = "replicate",
        noise_model=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        time_pad_mode: Literal["replicate", "reflect", "constant"] = "replicate",
        constant_pad_value: float = 0.0,
    ):
        super().__init__()
        assert factor_t > 0 and factor > 0
        if kernel_size_t % 2 == 0:
            raise ValueError("kernel_size_t must be odd.")
        if sigma_t <= 0:
            raise ValueError("sigma_t must be > 0.")

        self.factor_t = int(factor_t)
        self.kernel_size_t = int(kernel_size_t)
        self.sigma_t = float(sigma_t)

        self.factor = int(factor)
        self.filter = filter
        self.padding = padding
        self.noise_model = noise_model

        self.device_ = device
        self.dtype_ = dtype

        self.time_conv_pad_mode = time_conv_pad_mode
        self._sr_dtype = torch.float32

        self.time_pad_mode = time_pad_mode
        self.constant_pad_value = float(constant_pad_value)

        # Spatial SR operator is created lazily on first call (needs C,H,W)
        self._sr = None
        self._hr_size = None

    # ---------- helpers ----------
    def _t_pad_info(self, T: int):
        T_lr = (T + self.factor_t - 1) // self.factor_t
        T_pad = T_lr * self.factor_t
        return T_lr, T_pad, (T_pad - T)

    def _pad_time_end(self, x: torch.Tensor, T_pad: int) -> torch.Tensor:
        T, C, H, W = x.shape
        if T_pad == T:
            return x
        pad_t = T_pad - T
        if self.time_pad_mode == "replicate":
            tail = x[-1:].expand(pad_t, C, H, W)
            return torch.cat([x, tail], dim=0)
        elif self.time_pad_mode == "reflect":
            idxs = torch.arange(T-2, T-2-pad_t, -1, device=x.device).clamp_min(0)
            return torch.cat([x, x[idxs]], dim=0)
        elif self.time_pad_mode == "constant":
            tail = torch.full((pad_t, C, H, W), self.constant_pad_value, device=x.device, dtype=x.dtype)
            return torch.cat([x, tail], dim=0)
        else:
            raise ValueError(f"Unsupported time_pad_mode: {self.time_pad_mode}")

    def _ensure_sr(self, C, H, W, device):
        import deepinv as dinv
        if (self._sr is None) or (self._hr_size is None) or self._hr_size != (C, H, W):
            self._sr = dinv.physics.Downsampling(
                img_size=(C, H, W),
                factor=self.factor,
                device=device,
                noise_model=self.noise_model,
                filter=self.filter,
                padding=self.padding,
            )
            self._hr_size = (C, H, W)
        return self._sr

    def _temporal_gauss_conv_same(self, x: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
        """
        x: (T,C,H,W) real; k1d: (K,) normalized
        Return: (T,C,H,W), 'same' length using reflect pad.
        """
        T, C, H, W = x.shape
        pad = self.kernel_size_t // 2

        # flatten spatial+channels into batch for conv1d
        X = x.permute(1, 2, 3, 0).reshape(C * H * W, 1, T)  # (N,1,T)
        K = k1d.to(device=x.device, dtype=x.dtype).view(1, 1, -1)      # (1,1,K)

        Xp = F.pad(X, (pad, pad), mode='reflect')            # (N,1,T+2p)
        Y = F.conv1d(Xp, K, stride=1, padding=0)             # (N,1,T)
        y = Y.view(C, H, W, T).permute(3, 0, 1, 2)           # (T,C,H,W)
        return y

    # ---------- A: HR -> LR ----------
    @torch.no_grad()
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal Gaussian low-pass (same) -> decimate by factor_t -> spatial SR (DeepInv).
        x: (T,C,H,W)  →  y: (ceil(T/factor_t), C, H/factor, W/factor)
        """
        assert x.dim() == 4, f"Expected (T,C,H,W), got {x.shape}"
        dev = self.device_ or x.device
        dty = self.dtype_  or x.dtype
        x = x.to(dev, dty)

        T, C, H, W = x.shape
        # pad time to multiple of factor_t
        T_lr  = (T + self.factor_t - 1) // self.factor_t
        T_pad = T_lr * self.factor_t
        if T_pad > T:
            tail = x[-1:].expand(T_pad - T, C, H, W)
            x_pad = torch.cat([x, tail], dim=0)
        else:
            x_pad = x

        # temporal Gaussian low-pass in FP32 (safe), then decimate
        k = gaussian_kernel_1d(self.kernel_size_t, self.sigma_t, dtype=torch.float32, device=dev)
        z = _temporal_conv_same_fp32(x_pad, k, pad_mode=getattr(self, "time_conv_pad_mode", "replicate"))  # (T_pad,C,H,W)
        y_t = z[::self.factor_t]                                                                           # (T_lr,C,H,W)

        # spatial SR (DeepInv) done in fp32 to match its weights, then cast back
        self._ensure_sr(C, H, W, dev)
        y32 = None
        try:
            y32 = self._sr.A(y_t.to(self._sr_dtype))                  # (T_lr,C,h,w)
        except Exception:
            ys = [self._sr.A(y_t[t].to(self._sr_dtype)).unsqueeze(0) for t in range(T_lr)]
            y32 = torch.cat(ys, dim=0)
        return y32.to(dty)

    @torch.no_grad()
    def A_adjoint(self, u: torch.Tensor, T_out: Optional[int] = None) -> torch.Tensor:
        """
        Spatial adjoint (DeepInv fp32) → temporal adjoint of (Gaussian + decimate):
        zero-insert upsample to T_pad, then convolve with time-reversed Gaussian (same).
        u: (T_lr,C,h,w)  →  v: (T_out,C,H,W), with T_out usually original T.
        """
        assert u.dim() == 4, f"Expected (T_lr,C,h,w), got {u.shape}"
        dev = self.device_ or u.device
        dty = self.dtype_  or u.dtype
        u = u.to(dev, dty)

        T_lr, C, h, w = u.shape
        H, W = h * self.factor, w * self.factor
        T_pad = T_lr * self.factor_t

        # spatial adjoint (fp32), then back to working dtype
        if self._hr_size is None or self._hr_size != (C, H, W):
            self._ensure_sr(C, H, W, dev)
        try:
            vs32 = self._sr.A_adjoint(u.to(self._sr_dtype))           # (T_lr,C,H,W)
        except Exception:
            vs32 = torch.cat([self._sr.A_adjoint(u[t].to(self._sr_dtype)).unsqueeze(0) for t in range(T_lr)], dim=0)
        vs = vs32.to(dty)

        # temporal adjoint: zero-insertion upsample → time-reversed Gaussian conv (FP32 helper)
        z_up = torch.zeros(T_pad, C, H, W, device=dev, dtype=dty)
        z_up[::self.factor_t] = vs

        k = gaussian_kernel_1d(self.kernel_size_t, self.sigma_t, dtype=torch.float32, device=dev)
        k_tr = torch.flip(k, dims=(0,))
        x_pad = _temporal_conv_same_fp32(z_up, k_tr, pad_mode=getattr(self, "time_conv_pad_mode", "replicate"))  # (T_pad,C,H,W)

        # fold padded tail back to last real frame if T_out provided and < T_pad
        if T_out is None or T_out == T_pad:
            return x_pad
        if T_out > T_pad:
            raise ValueError(f"T_out={T_out} cannot exceed T_pad={T_pad}")
        if T_pad > T_out:
            tail = x_pad[T_out:].sum(dim=0)
            x = x_pad[:T_out].clone()
            x[-1] += tail
            return x
        return x_pad[:T_out]


    # ---------- module API ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(x)

    # ---------- prox_{(gamma/2)||A x - y||^2} via CG (HR domain) ----------
    @torch.no_grad()
    def prox_l2(
        self,
        x0: torch.Tensor,             # (T,C,H,W) HR
        y:  torch.Tensor,             # (ceil(T/ft),C,h,w) LR
        gamma: Union[float, torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        assert x0.dim() == 4 and y.dim() == 4
        dev = self.device_ or x0.device
        dty = self.dtype_  or x0.dtype
        x0 = x0.to(dev, dty)
        y  = y.to(dev, dty)
        T  = x0.shape[0]

        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=dev, dtype=dty)
        gamma = gamma.to(dev, dty)
        if gamma.dim() == 0:
            gamma = gamma.view(1,1,1,1).expand_as(x0)
        else:
            gamma = gamma.expand_as(x0)

        def M(z):
            return z + gamma * self.A_adjoint(self.A(z), T_out=T)
        b = x0 + gamma * self.A_adjoint(y, T_out=T)

        x = x0.clone()
        r = b - M(x)
        p = r.clone()
        rs = (r*r).sum()
        for _ in range(max_iter):
            Mp = M(p)
            a = rs / ((p*Mp).sum() + 1e-20)
            x = x + a * p
            r = r - a * Mp
            rs_new = (r*r).sum()
            if torch.sqrt(rs_new) < tol:
                break
            p = r + (rs_new/rs) * p
            rs = rs_new
        return x.clamp(-1.0, 1.0)

class TemporalSRThenSROp(nn.Module):
    """
    A = SpatialSR ∘ TemporalSR.
    Works on x: (T,C,H,W) -> y: (ceil(T/ft), C, H/f, W/f)

    - Temporal SR: average pooling over time with factor ft (pad tail if T % ft != 0).
    - Spatial SR: DeepInv Downsampling with 'factor', 'filter', 'padding'.

    Provides:
      A(x), A_adjoint(u, T_out=None), prox_l2(x0, y, gamma), solve_l2_tv(x_init, y, lam, ...)

    Notes:
      * A_adjoint(u, T_out): if T_out is given (original T), we fold the padded tail back
        to the last real frame to return exactly length T_out. If None, returns length T_pad.
    """

    def __init__(
        self,
        factor_t: int = 4,
        factor: int = 8,                 # spatial factor
        filter: str = "bicubic",
        padding: str = "reflect",
        noise_model=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        time_pad_mode: Literal["replicate", "reflect", "constant"] = "replicate",
        constant_pad_value: float = 0.0,
    ):
        super().__init__()
        assert factor_t > 0 and factor > 0
        self.factor_t     = int(factor_t)
        self.factor       = int(factor)
        self.filter       = filter
        self.padding      = padding
        self.noise_model  = noise_model
        self.device_      = device
        self.dtype_       = dtype
        self._sr_dtype = torch.float32   # DeepInv SR runs in fp32 for safety
        self.time_pad_mode = time_pad_mode
        self.constant_pad_value = float(constant_pad_value)

        # Spatial SR operator is created lazily on first call (needs H,W,C)
        self._sr = None                 # DeepInv Downsampling instance bound to (C,H,W)
        self._hr_size = None            # cache (C,H,W)

    # ---------- helpers ----------
    def _t_pad_info(self, T: int):
        T_lr  = (T + self.factor_t - 1) // self.factor_t  # ceil(T/ft)
        T_pad = T_lr * self.factor_t
        pad_t = T_pad - T
        return T_lr, T_pad, pad_t

    def _pad_time_end(self, x: torch.Tensor, T_pad: int) -> torch.Tensor:
        """Pad time at end to length T_pad."""
        T, C, H, W = x.shape
        if T_pad == T:
            return x
        pad_t = T_pad - T
        if self.time_pad_mode == "replicate":
            tail = x[-1:].expand(pad_t, C, H, W)
            return torch.cat([x, tail], dim=0)
        elif self.time_pad_mode == "reflect":
            idxs = torch.arange(T-2, T-2-pad_t, -1, device=x.device).clamp_min(0)
            return torch.cat([x, x[idxs]], dim=0)
        elif self.time_pad_mode == "constant":
            tail = torch.full((pad_t, C, H, W), self.constant_pad_value, device=x.device, dtype=x.dtype)
            return torch.cat([x, tail], dim=0)
        else:
            raise ValueError(f"Unsupported time_pad_mode: {self.time_pad_mode}")

    def _ensure_sr(self, C, H, W, device):
        import deepinv as dinv
        if (self._sr is None) or (self._hr_size is None) or self._hr_size != (C, H, W):
            self._sr = dinv.physics.Downsampling(
                img_size=(C, H, W),
                factor=self.factor,
                device=device,
                noise_model=self.noise_model,
                filter=self.filter,
                padding=self.padding,
            )
            self._hr_size = (C, H, W)
        return self._sr

    @torch.no_grad()
    def A(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        dev = self.device_ or x.device
        dty = self.dtype_  or x.dtype
        x   = x.to(dev, dty)

        T, C, H, W = x.shape
        T_lr, T_pad, _ = self._t_pad_info(T)
        x_pad = self._pad_time_end(x, T_pad)                 # (T_pad,C,H,W)

        # temporal SR (avg over time)
        x5    = x_pad.permute(1,0,2,3).unsqueeze(0)          # (1,C,T_pad,H,W)
        y5_t  = F.avg_pool3d(x5, kernel_size=(self.factor_t,1,1), stride=(self.factor_t,1,1))
        y_t   = y5_t.squeeze(0).permute(1,0,2,3)             # (T_lr,C,H,W)

        # spatial SR in fp32, then cast back
        self._ensure_sr(C, H, W, dev)
        y_t32 = y_t.to(dev, self._sr_dtype)
        try:
            y32 = self._sr.A(y_t32)                          # (T_lr,C,h,w), fp32
        except Exception:
            ys  = [self._sr.A(y_t32[t]).unsqueeze(0) for t in range(T_lr)]
            y32 = torch.cat(ys, dim=0)
        return y32.to(dty)                                   # back to original dtype

    @torch.no_grad()
    def A_adjoint(self, u: torch.Tensor, T_out: Optional[int] = None) -> torch.Tensor:
        assert u.dim() == 4
        dev = self.device_ or u.device
        dty = self.dtype_  or u.dtype
        T_lr, C, h, w = u.shape
        H, W = h * self.factor, w * self.factor
        T_pad = T_lr * self.factor_t

        # spatial adjoint in fp32, then cast
        if self._hr_size is None or self._hr_size != (C, H, W):
            self._ensure_sr(C, H, W, dev)
        u32 = u.to(dev, self._sr_dtype)
        try:
            vs32 = self._sr.A_adjoint(u32)                   # (T_lr,C,H,W), fp32
        except Exception:
            vs32 = torch.cat([self._sr.A_adjoint(u32[t]).unsqueeze(0) for t in range(T_lr)], dim=0)
        vs = vs32.to(dty)

        # temporal adjoint of avg (nearest upsample / factor_t)
        vs5   = vs.permute(1,0,2,3).unsqueeze(0)             # (1,C,T_lr,H,W)
        z5pad = F.interpolate(vs5, scale_factor=(self.factor_t,1,1), mode='nearest') / self.factor_t
        zpad  = z5pad.squeeze(0).permute(1,0,2,3)            # (T_pad,C,H,W)

        if T_out is None or T_out == T_pad:
            return zpad
        if T_out > T_pad:
            raise ValueError(f"T_out={T_out} cannot exceed T_pad={T_pad}")
        tail = zpad[T_out:].sum(dim=0) if T_pad > T_out else 0
        z    = zpad[:T_out].clone()
        if T_pad > T_out:
            z[-1] += tail
        return z

    # ---------- module API ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(x)

    def _check_SSt_identity_once(self, y_sample: torch.Tensor) -> bool:
        """
        Cheap one-time probe to check whether S S^T ≈ I on the LR grid.
        Caches the result in self._SSt_is_identity.
        """
        if hasattr(self, "_SSt_is_identity"):
            return self._SSt_is_identity
        try:
            # random small LR tensor with same shape as y
            u = torch.randn_like(y_sample) * 1e-3
            Su = self.A_adjoint(u, T_out=None)      # HR
            SSt_u = self.A(Su)                      # back to LR: S S^T u
            ok = torch.allclose(SSt_u, u, rtol=1e-3, atol=1e-5)
        except Exception:
            ok = False
        self._SSt_is_identity = ok
        return ok

    # ---------- prox_{(gamma/2)||A x - y||^2} via CG ----------
    @torch.no_grad()
    def prox_l2(
        self,
        x0: torch.Tensor,             # (T,C,H,W)  HR
        y:  torch.Tensor,             # (T_lr,C,h,w) LR
        gamma: Union[float, torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Solve (I + gamma A^T A) x = x0 + gamma A^T y via CG in HR space.
        Works for any T (handles temporal padding internally).
        """
        assert x0.dim() == 4 and y.dim() == 4
        dev = self.device_ or x0.device
        dty = self.dtype_  or x0.dtype
        x0 = x0.to(dev, dty)
        y  = y.to(dev, dty)
        T  = x0.shape[0]

        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=dev, dtype=dty)
        gamma = gamma.to(dev, dty)
        if gamma.dim() == 0:
            gamma = gamma.view(1,1,1,1).expand_as(x0)
        else:
            gamma = gamma.expand_as(x0)

        def M(z):
            return z + gamma * self.A_adjoint(self.A(z), T_out=T)
        b = x0 + gamma * self.A_adjoint(y, T_out=T)

        x = x0.clone()
        r = b - M(x)
        p = r.clone()
        rs = (r*r).sum()
        for _ in range(max_iter):
            Mp = M(p)
            a = rs / ((p*Mp).sum() + 1e-20)
            x = x + a * p
            r = r - a * Mp
            rs_new = (r*r).sum()
            if torch.sqrt(rs_new) < tol:
                break
            p = r + (rs_new/rs) * p
            rs = rs_new
        return x.clamp(-1.0, 1.0)

    # ---------- solve 0.5||A x - y||^2 + lam ||D_t x||_1 (time-TV) ----------
    @torch.no_grad()
    def solve_l2_tv(
        self,
        x_init: torch.Tensor,   # (T,C,H,W) HR
        y: torch.Tensor,        # (T_lr,C,h,w) LR (T_lr=ceil(T/ft))
        lam: float,
        iters: int = 200,
        tau: float = 0.25,
        sigma: float = 0.25,
        theta: float = 1.0,
        clamp_m11: bool = True,
        ) -> torch.Tensor:
        """
        PDHG with:
          p for f(z)=0.5||z - y||^2 in LR space (T_lr frames),
          q for g(w)=λ||w||_1 with w = D_t x  (time TV in HR space).
        """
        assert x_init.dim() == 4 and y.dim() == 4
        dev = x_init.device
        dty = x_init.dtype

        # shapes
        T, C, H, W = x_init.shape
        T_lr, _, h, w = y.shape

        x  = x_init.clone()
        x_bar = x.clone()

        # Duals
        p = torch.zeros((T_lr, C, h, w), device=dev, dtype=dty)  # LR dual
        q = torch.zeros((T-1, C, H, W), device=dev, dtype=dty)   # time-TV dual

        for _ in range(iters):
            # p <- prox_{σ f*}(p + σ A x̄) with f(z)=0.5||z - y||^2
            Axbar = self.A(x_bar)                 # (T_lr,C,h,w)
            p = (p + sigma * Axbar - sigma * y) / (1.0 + sigma)

            # q <- prox_{σ g*}(q + σ D_t x̄) with g(w)=λ||w||_1 => clip to [-λ, λ]
            Dt_xbar = x_bar[1:] - x_bar[:-1]      # (T-1,C,H,W)
            q = torch.clamp(q + sigma * Dt_xbar, min=-lam, max=lam)

            # x <- x - τ (A^T p + D_t^T q)
            ATp = self.A_adjoint(p, T_out=T)      # (T,C,H,W)
            # div_t (adjoint of forward diff with Neumann BC)
            Tm1 = q.shape[0]
            div = torch.zeros_like(x)
            div[0]      = -q[0]
            if Tm1 > 1:
                div[1:-1] = q[:-1] - q[1:]
            div[-1]     =  q[-1]

            x_new = x - tau * (ATp + div)

            # extrapolation
            x_bar = x_new + theta * (x_new - x)
            x = x_new

        if clamp_m11:
            x = x.clamp(-1.0, 1.0)
        return x

    # ---------- differentiable A (no @torch.no_grad) ----------
    def A_autograd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable version of A(x): (T,C,H,W) -> (T_lr,C,h,w).
        Same logic as A(), just no torch.no_grad and keeps graph.
        """
        assert x.dim() == 4
        dev = self.device_ or x.device
        dty = self.dtype_  or x.dtype
        x   = x.to(dev, dty)

        T, C, H, W = x.shape
        T_lr, T_pad, _ = self._t_pad_info(T)
        x_pad = self._pad_time_end(x, T_pad)                 # (T_pad,C,H,W)

        # temporal SR (avg over time)
        x5    = x_pad.permute(1,0,2,3).unsqueeze(0)          # (1,C,T_pad,H,W)
        y5_t  = F.avg_pool3d(x5, kernel_size=(self.factor_t,1,1),
                            stride=(self.factor_t,1,1))
        y_t   = y5_t.squeeze(0).permute(1,0,2,3)             # (T_lr,C,H,W)

        # spatial SR in fp32, then cast back (autograd-safe)
        self._ensure_sr(C, H, W, dev)
        y_t32 = y_t.to(dev, self._sr_dtype)
        try:
            y32 = self._sr.A(y_t32)                          # (T_lr,C,h,w), fp32
        except Exception:
            ys  = [self._sr.A(y_t32[t]).unsqueeze(0) for t in range(T_lr)]
            y32 = torch.cat(ys, dim=0)
        return y32.to(dty)

    # ---------- 3D forward diffs (t,y,x) ----------
    def _grad3(self, x: torch.Tensor):
        gt = torch.zeros_like(x); gy = torch.zeros_like(x); gx = torch.zeros_like(x)
        gt[:-1] = x[1:] - x[:-1]
        gy[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        return gt, gy, gx

    # ---------- isotropic TV_3D with eps smoothing (autograd) ----------
    def _tv3_isotropic(self, x: torch.Tensor, eps_tv: float = 1e-3, alpha: float = 1e0) -> torch.Tensor:
        gt, gy, gx = self._grad3(x)
        tv_vox = torch.sqrt(alpha*gt*gt + gy*gy + gx*gx + eps_tv*eps_tv)
        return tv_vox.sum()  # or .mean(), but keep consistent with lambda scale

    def solve_l2_tv3_adam_autograd(
        self,
        x_init: torch.Tensor,        # (T,C,H,W)
        y: torch.Tensor,             # (T_lr,C,h,w)
        lam: float,
        iters: int = 300,
        lr: float = 1e-3,
        eps_tv: float = 1e-2,
        alpha: float = 1e0,
        weight_decay: float = 0.0,   # optional L2 on x
        clamp_m11: bool = True,
        stop_tol: float = 0.0,       # relative change early stop; 0 = off
        log_every: int = 10,          # 0 = no logs
        grad_clip: float | None = None,
    ):
        """
        Pure-autograd Adam on: 0.5||A(x)-y||^2 + lam * TV3D(x).
        No use of A^T or prox; backprop goes through A_autograd and TV.
        """
        assert x_init.dim() == 4 and y.dim() == 4
        dev = self.device_ or x_init.device
        # work in fp32 for stability
        x_param = torch.nn.Parameter(x_init.to(dev, torch.float32).clone())
        y = y.to(dev, torch.float32)

        opt = torch.optim.Adam([{"params": [x_param], "weight_decay": weight_decay}], lr=lr)

        prev_x = x_param.data.clone()
        for k in range(1, iters + 1):
            opt.zero_grad(set_to_none=True)

            Ax = self.A_autograd(x_param)                # differentiable forward model
            data_loss = 0.5 * torch.sum((Ax - y) ** 2)   # sum or mean; keep λ consistent
            tv_loss   = self._tv3_isotropic(x_param, eps_tv=eps_tv, alpha=alpha) * float(lam)
            loss = data_loss + tv_loss

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([x_param], max_norm=grad_clip)

            opt.step()

            if clamp_m11:
                x_param.data.clamp_(-1.0, 1.0)

            if stop_tol > 0.0 and (k % 5 == 0):
                num = (x_param.data - prev_x).pow(2).sum().sqrt()
                den = prev_x.pow(2).sum().sqrt().clamp_min(1e-12)
                if (num / den).item() < stop_tol:
                    break
                prev_x.copy_(x_param.data)

            if log_every and (k % log_every == 0):
                with torch.no_grad():
                    # a cheap TV metric for logging (per-voxel mean)
                    gt, gy, gx = self._grad3(x_param.data)
                    tv_mean = torch.sqrt(alpha*gt*gt + gy*gy + gx*gx + eps_tv*eps_tv).mean().item()
                    print(f"[adam {k:04d}] loss={loss.item():.4e} data={data_loss.item():.4e} "
                        f"tv*lam={(tv_loss).item():.4e} tv_mean={tv_mean:.4e}")

        out = x_param.detach().to(x_init.dtype if x_init.dtype != torch.float32 else torch.float32)
        return out


## JPEG
class GaussianNoiseJPEGOp(nn.Module):
    """
    Forward operator for data simulation: per-frame Gaussian noise + JPEG compression.

        x: (T, C, H, W) in [-1, 1]
        y = A(x): (T, C, H, W) in [-1, 1]

    Inverse side:
      - For the proximal step we use ONLY the deterministic JPEG part as A_lin.
      - The Gaussian noise is treated as part of the measurement y.

    All heavy work (JPEG) is done on CPU by default to save GPU VRAM.
    """

    def __init__(
        self,
        sigma: float = 0.01,
        jpeg_quality: int = 50,
        device: Optional[torch.device] = torch.device("cpu"),  # physics device
        dtype: Optional[torch.dtype] = torch.float32,
        n_prox_iter: int = 10,
        prox_step: Optional[float] = None,
    ):
        super().__init__()
        assert sigma >= 0.0
        assert 1 <= jpeg_quality <= 100

        self.sigma = float(sigma)
        self.jpeg_quality = int(jpeg_quality)

        self.device_ = device     # "physics" device (CPU by default)
        self.dtype_ = dtype

        self.n_prox_iter = int(n_prox_iter)
        self.prox_step = prox_step  # if None we set a conservative default at run time

    # ---------- helpers ----------

    @staticmethod
    def _to_uint8(x: torch.Tensor) -> torch.Tensor:
        """Map [-1,1] float tensor -> [0,255] uint8."""
        x = x.clamp(-1.0, 1.0)
        x01 = (x + 1.0) * 0.5          # [0,1]
        x255 = torch.round(x01 * 255.0)
        return x255.to(torch.uint8)

    @staticmethod
    def _from_uint8(xu: torch.Tensor, device, dtype) -> torch.Tensor:
        """Map [0,255] uint8 tensor -> [-1,1] float tensor."""
        x = xu.to(device=device, dtype=torch.float32) / 255.0  # [0,1]
        x = x * 2.0 - 1.0                                      # [-1,1]
        return x.to(dtype)

    def _jpeg_compress_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        JPEG compress a single frame (C,H,W) in [-1,1], return (C,H,W) in [-1,1].
        JPEG encode/decode happen on CPU.
        """
        dev = frame.device
        dty = frame.dtype

        frame_u8_cpu = self._to_uint8(frame).cpu()
        jpeg_bytes = tv_io.encode_jpeg(frame_u8_cpu, quality=self.jpeg_quality)
        dec_u8 = tv_io.decode_jpeg(jpeg_bytes)  # uint8, CHW on CPU

        return self._from_uint8(dec_u8, device=dev, dtype=dty)

    def _jpeg_compress_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG compression frame-wise.
        x: (T,C,H,W) -> y: (T,C,H,W)
        """
        T, C, H, W = x.shape
        frames = [self._jpeg_compress_frame(x[t]) for t in range(T)]
        return torch.stack(frames, dim=0)

    # ---------- A: used for *data simulation* ----------
    @torch.no_grad()
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise (sigma) and then JPEG-compress each frame.

        x: (T,C,H,W) in [-1,1] (can be on GPU)
        y: (T,C,H,W) in [-1,1], returned on the input device.
        """
        assert x.dim() == 4, f"Expected (T,C,H,W), got {x.shape}"

        orig_dev = x.device
        orig_dtype = x.dtype

        dev = self.device_ or torch.device("cpu")
        dty = self.dtype_ or torch.float32

        x = x.to(dev, dty)

        # add noise ONCE when simulating measurements
        if self.sigma > 0:
            x = x + self.sigma * torch.randn_like(x)
            x.clamp_(-1.0, 1.0)

        y = self._jpeg_compress_video(x)

        return y.to(orig_dev, orig_dtype)

    # deterministic part used in the prox (no noise)
    @torch.no_grad()
    def A_det(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic JPEG-only operator used inside prox_l2.

        x,y share the same semantics as for A, but here NO noise is added.
        """
        assert x.dim() == 4
        orig_dev = x.device
        orig_dtype = x.dtype

        dev = self.device_ or torch.device("cpu")
        dty = self.dtype_ or torch.float32

        x = x.to(dev, dty)
        y = self._jpeg_compress_video(x)

        return y.to(orig_dev, orig_dtype)

    # ---------- module API ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias so the module can be used like a usual nn.Module."""
        return self.A(x)

    # ---------- approximate adjoint ----------
    @torch.no_grad()
    def A_adjoint(self, u: torch.Tensor) -> torch.Tensor:
        """
        Approximate adjoint of A_det.

        JPEG is non-linear and non-invertible, so we use the identity map:
            A_adjoint(u) ≈ u
        """
        return u

    # ---------- approx prox_{(gamma/2)||A_det(x)-y||^2} ----------
    @torch.no_grad()
    def prox_l2(
        self,
        x0: torch.Tensor,             # (T,C,H,W), initial HR (can be on GPU)
        y:  torch.Tensor,             # (T,C,H,W), observed (noise+JPEG) video
        gamma: Union[float, torch.Tensor],
        max_iter: Optional[int] = None,
        step: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Approximate proximal operator

            prox_{gamma/2 ||A_det(x)-y||^2}(x0)

        using Landweber iterations with approximate gradient:

            f(x) = 0.5||x - x0||^2 + gamma/2 ||A_det(x) - y||^2

        grad f(x) ≈ (x - x0) + gamma * (A_det(x) - y).

        All JPEG calls happen on CPU by default; result is returned on x0.device.
        """
        assert x0.dim() == 4 and y.dim() == 4

        if max_iter is None:
            max_iter = self.n_prox_iter

        orig_dev = x0.device
        orig_dtype = x0.dtype

        dev = self.device_ or torch.device("cpu")
        dty = self.dtype_ or torch.float32

        x = x0.to(dev, dty)
        y_ = y.to(dev, dty)
        x0_ = x0.to(dev, dty)

        gamma = torch.as_tensor(gamma, device=dev, dtype=dty)

        # conservative default step if not given
        if step is None:
            step = self.prox_step if self.prox_step is not None else 1.0 / (1.0 + float(gamma))

        for _ in range(max_iter):
            # deterministic JPEG-only forward on physics device
            Ax = self.A_det(x.to(orig_dev, orig_dtype)).to(dev, dty)

            # residual in measurement space
            r = Ax - y_

            # approximate gradient
            grad = (x - x0_) + gamma * r

            x = x - step * grad
            x.clamp_(-1.0, 1.0)

        return x.to(orig_dev, orig_dtype)



# -------------------------
# I/O helpers
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def list_frames(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMG_EXTS]
    # natural-ish sort
    def key(p):
        base = os.path.basename(p)
        head = "".join(ch if not ch.isdigit() else " " for ch in base).split()
        nums = [int("".join(filter(str.isdigit, s))) for s in base.split() if any(c.isdigit() for c in s)]
        return (head, nums, base)
    return sorted(files, key=lambda p: (os.path.dirname(p), os.path.basename(p)))

def load_video_from_frames(
    frames_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    T: Optional[int] = None,
    target_size: tuple[int, int] = (1280, 768),  # (W, H) (1280, 768)
) -> torch.Tensor:
    """
    Load up to the first T frames from a folder of images and resize each to target_size (W,H).

    Returns:
        Tensor of shape (T_loaded, 3, 480, 832) in [0, 1] (given the default target_size).
    """
    paths = list_frames(frames_dir)
    if not paths:
        raise FileNotFoundError(
            f"No frames found in {frames_dir} (supported: {sorted(IMG_EXTS)})"
        )

    if T is not None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        paths = paths[:min(T, len(paths))]

    target_w, target_h = target_size

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((target_w, target_h), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        t = torch.from_numpy(arr).permute(2, 0, 1)        # (3, H, W)
        frames.append(t)

    vid = torch.stack(frames, dim=0)  # (T_loaded, 3, target_h, target_w)
    return vid.to(device=device, dtype=dtype)

def save_video_tensor(tvid: torch.Tensor, out_path: str, fps: int = 16):
    """
    tvid: (T, C, H, W), values in [0,1]. Saves MP4.
    """
    tv = tvid.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()  # (T,H,W,C)
    export_to_video(tv, out_path, fps=fps)