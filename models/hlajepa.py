"""
H-LAJEPA: Hierarchical Latent Action JEPA

Full forward pass:
  1. Encode frames with frozen backbone  → (B, T, N, D)
  2. Infer latent action code between frame pairs
  3. Low-level prediction:  z_t  + a      → ẑ_{t+1}   (frame level)
  4. High-level prediction: s_t  + a_seg  → ŝ_{t+T}   (segment level)
  5. JEPA losses — no pixels, no decoder

forward() accepts either:
  - raw frames  (B, T, C, H, W)  → passes through frozen encoder
  - pre-encoded tokens (B, T, N, D)  → skips encoder (feature cache mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass

from .encoder import FrozenEncoder
from .action_inferrer import ActionInferrer
from .predictor import LowLevelPredictor, HighLevelPredictor


@dataclass
class HLAJEPAOutput:
    loss: torch.Tensor
    loss_low: torch.Tensor
    loss_high: torch.Tensor
    loss_vq: torch.Tensor
    action_indices: torch.Tensor   # (B, T-1) frame-level codebook indices


class HLAJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = FrozenEncoder(cfg)
        self.action_inferrer = ActionInferrer(cfg)
        self.low_predictor = LowLevelPredictor(cfg)
        self.high_predictor = HighLevelPredictor(cfg)

        w = cfg.training.loss_weights
        self.w_low    = w.low_level
        self.w_high   = w.high_level
        self.w_commit = w.vq_commit

    def _get_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Accept either raw frames (B,T,C,H,W) or pre-encoded tokens (B,T,N,D).
        Returns tokens (B, T, N, D).
        """
        if frames.dim() == 5:
            # raw video — run through frozen encoder
            with torch.no_grad():
                return self.encoder(frames)
        elif frames.dim() == 4:
            # pre-encoded features (B, T, N, D) — skip encoder
            return frames
        else:
            raise ValueError(f"Expected 4D or 5D input, got {frames.dim()}D")

    def forward(self, frames: torch.Tensor) -> HLAJEPAOutput:
        """
        Args:
            frames: (B, T, C, H, W)  raw video  OR
                    (B, T, N, D)     pre-encoded tokens from feature cache
        Returns:
            HLAJEPAOutput
        """
        tokens = self._get_tokens(frames)   # (B, T, N, D)
        B, T, N, D = tokens.shape
        assert T >= 4, "Need at least 4 frames for hierarchical prediction"

        # ── 1. Low-level: predict frame t+1 from frame t ─────────────────────
        z_t   = tokens[:, :-1]              # (B, T-1, N, D)
        z_tp1 = tokens[:, 1:]               # (B, T-1, N, D)

        # flatten time into batch for parallel inference
        z_t_flat   = rearrange(z_t,   "b t n d -> (b t) n d")
        z_tp1_flat = rearrange(z_tp1, "b t n d -> (b t) n d")

        code, indices, vq_loss = self.action_inferrer(z_t_flat, z_tp1_flat)
        # code: (B*(T-1), bottleneck_dim)

        pred_tp1 = self.low_predictor(z_t_flat, code)   # (B*(T-1), N, D)
        loss_low = F.mse_loss(pred_tp1, z_tp1_flat.detach())

        # ── 2. High-level: predict next segment from current segment ──────────
        seg_len = self.cfg.high_level_predictor.segment_len
        n_segs  = T // seg_len
        if n_segs >= 2:
            # pool frames + patches into segment vectors
            seg_tokens = tokens[:, : n_segs * seg_len]   # (B, n_segs*L, N, D)
            seg_tokens = rearrange(
                seg_tokens, "b (s l) n d -> b s l n d", s=n_segs, l=seg_len
            )
            seg_repr = seg_tokens.mean(dim=(2, 3))        # (B, n_segs, D)

            seg_t   = seg_repr[:, :-1]                    # (B, n_segs-1, D)
            seg_tp1 = seg_repr[:, 1:]                     # (B, n_segs-1, D)

            # flatten segments into batch
            seg_t_flat   = rearrange(seg_t,   "b s d -> (b s) d")
            seg_tp1_flat = rearrange(seg_tp1, "b s d -> (b s) d")

            # action inferrer expects (B, N, D) — treat segment as single "patch"
            seg_code, _, seg_vq = self.action_inferrer(
                seg_t_flat.unsqueeze(1),    # (B*s, 1, D)
                seg_tp1_flat.unsqueeze(1),  # (B*s, 1, D)
            )

            # high predictor expects (B, 1, D)
            pred_seg = self.high_predictor(
                seg_t_flat.unsqueeze(1), seg_code
            )                                              # (B*s, 1, D)
            loss_high = F.mse_loss(
                pred_seg, seg_tp1_flat.unsqueeze(1).detach()
            )
            vq_loss = vq_loss + seg_vq
        else:
            loss_high = torch.tensor(0.0, device=tokens.device)

        # ── 3. Total loss ─────────────────────────────────────────────────────
        loss = (
            self.w_low    * loss_low
            + self.w_high   * loss_high
            + self.w_commit * vq_loss
        )

        return HLAJEPAOutput(
            loss=loss,
            loss_low=loss_low,
            loss_high=loss_high,
            loss_vq=vq_loss,
            action_indices=indices.reshape(B, T - 1),
        )
