import torch
import torch.nn as nn
import torch.nn.functional as F


class HistConvResidual(nn.Module):
    """
    Predict normalized residuals r_hat_norm: (B,H,S)

    Inputs:
      h_idx: (B,H) long
      s_idx: (B,H,S) long
      weather_in: (B,H,W) float
      r_hist_norm:
         - (B,T,S) or
         - (B,T,S,C) if using multi-channel history
      yb_s_norm: (B,H,S) optional per-sensor baseline feature (normalized)
    """

    def __init__(
        self,
        n_sensors: int,
        w_dim: int,
        h_dim: int = 32,
        ctx_dim: int = 32,
        t_len: int = 168,
        rh_channels: int = 1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.h_dim = h_dim
        self.ctx_dim = ctx_dim
        self.rh_channels = rh_channels

        # embeddings
        self.h_embed = nn.Embedding(72, h_dim)
        self.s_embed = nn.Embedding(n_sensors, h_dim)

        # weather -> h_dim
        self.w_proj = nn.Linear(w_dim, h_dim)

        # NEW: per-sensor baseline scalar -> h_dim
        self.yb_proj = nn.Linear(1, h_dim)

        # history encoder: Conv1D over time per sensor
        self.conv1 = nn.Conv1d(rh_channels, 8, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ctx_fc = nn.Linear(16, ctx_dim)

        self.ctx_to_h = nn.Linear(ctx_dim, h_dim)

        # per-sensor bias
        self.b = nn.Parameter(torch.zeros(n_sensors))

    def encode_hist(self, r_hist_norm: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          (B,T,S) or (B,T,S,C)
        returns:
          ctx: (B,S,ctx_dim)
        """
        if r_hist_norm.dim() == 3:
            r_hist_norm = r_hist_norm.unsqueeze(-1)  # (B,T,S,1)

        B, T, S, C = r_hist_norm.shape
        if C != self.rh_channels:
            raise ValueError(
                f"Expected rh_channels={self.rh_channels}, got C={C} from shape {tuple(r_hist_norm.shape)}"
            )

        x = r_hist_norm.permute(0, 2, 3, 1).contiguous()  # (B,S,C,T)
        x = x.view(B * S, C, T)                           # (B*S,C,T)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)                      # (B*S,16)
        x = self.ctx_fc(x)                                # (B*S,ctx_dim)
        x = x.view(B, S, self.ctx_dim)                    # (B,S,ctx_dim)
        return x

    def forward(self, h_idx, s_idx, weather_in, r_hist_norm, yb_s_norm=None):
        """
        returns r_hat_norm: (B,H,S)
        """
        B, H = h_idx.shape
        _, _, S = s_idx.shape

        h_vec = self.h_embed(h_idx)       # (B,H,h_dim)
        s_vec = self.s_embed(s_idx)       # (B,H,S,h_dim)
        w_vec = self.w_proj(weather_in)   # (B,H,h_dim)

        ctx = self.encode_hist(r_hist_norm)   # (B,S,ctx_dim)
        ctx_h0 = self.ctx_to_h(ctx)           # (B,S,h_dim)

        # Phase 2: horizon-aware context
        ctx_h = ctx_h0[:, None, :, :] + h_vec[:, :, None, :]  # (B,H,S,h_dim)

        hw = (h_vec + w_vec)[:, :, None, :].expand(B, H, S, self.h_dim)  # (B,H,S,h_dim)

        # NEW: per-sensor baseline injection
        if yb_s_norm is not None:
            # (B,H,S) -> (B,H,S,1) -> (B,H,S,h_dim)
            yb_vec = self.yb_proj(yb_s_norm[..., None])
            hw = hw + yb_vec

        out = (hw + ctx_h) * s_vec
        out = out.sum(dim=-1) + self.b[None, None, :]  # (B,H,S)
        return out