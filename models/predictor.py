"""
Hierarchical predictors — both operate entirely in latent space (no pixels).

LowLevelPredictor:   (z_t, a_k) → ẑ_{t+1}    frame-level
HighLevelPredictor:  (s_t, a_seg) → ŝ_{t+T}   segment-level
"""

import torch
import torch.nn as nn
from einops import rearrange


class _TransformerPredictor(nn.Module):
    """Shared backbone for both predictor levels."""

    def __init__(self, in_dim: int, cond_dim: int, hidden: int,
                 n_layers: int, n_heads: int, out_dim: int, dropout: float):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)

        # learnable [PRED] query token
        self.pred_token = nn.Parameter(torch.randn(1, 1, hidden))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    (B, N, D_in)  context tokens  (N=1 for segment level)
        cond: (B, D_cond)   action code
        Returns:
            pred: (B, N, D_out)  predicted future tokens
        """
        B, N, _ = z.shape
        ctx   = self.in_proj(z)                                # (B, N, H)
        cond_ = self.cond_proj(cond).unsqueeze(1)              # (B, 1, H)
        pred_tok = self.pred_token.expand(B, N, -1)            # (B, N, H)

        # sequence: [context | action_cond | pred_query]
        seq = torch.cat([ctx, cond_, pred_tok], dim=1)         # (B, 2N+1, H)
        out = self.transformer(seq)                            # (B, 2N+1, H)

        pred = self.out_proj(out[:, -N:])                      # (B, N, D_out)
        return pred


class LowLevelPredictor(nn.Module):
    """
    Predicts the next frame's patch tokens given current tokens + action code.
    Operates at per-patch level: N patch tokens per frame.
    """

    def __init__(self, cfg):
        super().__init__()
        feat_dim  = cfg.backbone.feat_dim
        cond_dim  = cfg.vq.code_dim
        hidden    = cfg.low_level_predictor.hidden_dim
        n_layers  = cfg.low_level_predictor.num_layers
        n_heads   = cfg.low_level_predictor.num_heads
        dropout   = cfg.low_level_predictor.dropout

        self.predictor = _TransformerPredictor(
            in_dim=feat_dim, cond_dim=cond_dim,
            hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            out_dim=feat_dim, dropout=dropout,
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z_t:    (B, N, D)  current frame patch tokens
        action: (B, code_dim)
        Returns:
            z_pred: (B, N, D)  predicted next frame patch tokens
        """
        return self.predictor(z_t, action)


class HighLevelPredictor(nn.Module):
    """
    Predicts future segment representation given current segment + segment action.
    Segment = mean-pooled frame tokens over segment_len frames → (B, 1, D).
    """

    def __init__(self, cfg):
        super().__init__()
        feat_dim  = cfg.backbone.feat_dim
        cond_dim  = cfg.vq.code_dim
        hidden    = cfg.high_level_predictor.hidden_dim
        n_layers  = cfg.high_level_predictor.num_layers
        n_heads   = cfg.high_level_predictor.num_heads
        dropout   = cfg.high_level_predictor.dropout
        self.segment_len = cfg.high_level_predictor.segment_len

        self.predictor = _TransformerPredictor(
            in_dim=feat_dim, cond_dim=cond_dim,
            hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            out_dim=feat_dim, dropout=dropout,
        )

    def _segment_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, N, D)  — T frames, N patches each
        Returns: (B, 1, D)    — single segment vector
        """
        return tokens.mean(dim=(1, 2)).unsqueeze(1)  # mean over T and N

    def forward(self, seg_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        seg_t:  (B, 1, D)  current segment token
        action: (B, code_dim)
        Returns:
            seg_pred: (B, 1, D)
        """
        return self.predictor(seg_t, action)
