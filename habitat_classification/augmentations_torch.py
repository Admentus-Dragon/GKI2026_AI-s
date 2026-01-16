# habitat_classification/augmentations_torch.py
import torch
import math
from dataclasses import dataclass

@dataclass
class AugConfig:
    enable: bool = True

    # Geometry
    random_rot90: bool = True
    random_flip: bool = True

    # Spectral
    spectral_noise_std: float = 0.0      # e.g. 0.01
    spectral_mult_jitter: float = 0.0    # e.g. 0.05 (mult factor in [1-j, 1+j])
    band_dropout_p: float = 0.0          # e.g. 0.10
    band_dropout_max_bands: int = 2      # drop up to N spectral bands

@dataclass
class AspectConfig:
    mode: str = "raw"  # "raw" | "sincos" | "drop"
    # raw   : keep aspect as channel 14
    # sincos: replace aspect (1 ch) with sin/cos (2 ch) -> +1 channel overall
    # drop  : remove aspect entirely -> -1 channel overall

def aspect_to_sincos(x: torch.Tensor, aspect_ch: int = 14) -> torch.Tensor:
    """
    x: (C, H, W) float tensor
    Aspect assumed degrees [0,360) or similar. Converts to sin/cos.
    Returns tensor with aspect channel replaced by two channels: sin, cos.
    """
    aspect_deg = x[aspect_ch]  # (H, W)
    aspect_rad = aspect_deg * (math.pi / 180.0)
    asp_sin = torch.sin(aspect_rad)
    asp_cos = torch.cos(aspect_rad)

    # keep everything except aspect_ch, then append sin/cos where aspect was
    before = x[:aspect_ch]
    after = x[aspect_ch + 1 :]
    return torch.cat([before, asp_sin.unsqueeze(0), asp_cos.unsqueeze(0), after], dim=0)

def drop_channel(x: torch.Tensor, ch: int) -> torch.Tensor:
    return torch.cat([x[:ch], x[ch+1:]], dim=0)

def apply_aspect_transform(x: torch.Tensor, cfg: AspectConfig, aspect_ch: int = 14) -> torch.Tensor:
    if cfg.mode == "raw":
        return x
    if cfg.mode == "sincos":
        return aspect_to_sincos(x, aspect_ch=aspect_ch)
    if cfg.mode == "drop":
        return drop_channel(x, ch=aspect_ch)
    raise ValueError(f"Unknown AspectConfig.mode={cfg.mode}")

def _rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (C,H,W)
    return torch.rot90(x, k=k, dims=(1, 2))

def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(2,))  # flip W

def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(1,))  # flip H

def apply_geometry_aug(x: torch.Tensor, cfg: AugConfig, g: torch.Generator) -> torch.Tensor:
    if not cfg.enable:
        return x

    if cfg.random_rot90:
        k = int(torch.randint(0, 4, (1,), generator=g).item())
        if k:
            x = _rot90(x, k)

    if cfg.random_flip:
        if bool(torch.randint(0, 2, (1,), generator=g).item()):
            x = _hflip(x)
        if bool(torch.randint(0, 2, (1,), generator=g).item()):
            x = _vflip(x)

    return x

def apply_spectral_aug(x: torch.Tensor, cfg: AugConfig, g: torch.Generator, spectral_channels: int = 12) -> torch.Tensor:
    """
    Applies augmentations ONLY to spectral channels [0:spectral_channels).
    Terrain channels (elev/slope/aspect or sincos) are left untouched.
    """
    if not cfg.enable:
        return x

    xs = x[:spectral_channels]

    # multiplicative jitter
    if cfg.spectral_mult_jitter and cfg.spectral_mult_jitter > 0:
        j = cfg.spectral_mult_jitter
        scale = (1.0 - j) + (2.0 * j) * torch.rand((spectral_channels, 1, 1), generator=g, device=x.device)
        xs = xs * scale

    # additive noise
    if cfg.spectral_noise_std and cfg.spectral_noise_std > 0:
        noise = torch.randn(xs.shape, generator=g, device=xs.device, dtype=xs.dtype)
        xs = xs + cfg.spectral_noise_std * noise


    # band dropout (drop whole bands)
    if cfg.band_dropout_p and cfg.band_dropout_p > 0 and cfg.band_dropout_max_bands > 0:
        if torch.rand((), generator=g).item() < cfg.band_dropout_p:
            n_drop = int(torch.randint(1, cfg.band_dropout_max_bands + 1, (1,), generator=g).item())
            drop_idx = torch.randperm(spectral_channels, generator=g)[:n_drop]
            xs[drop_idx] = 0.0

    x = torch.cat([xs, x[spectral_channels:]], dim=0)
    return x
