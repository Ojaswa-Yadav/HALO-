"""
Action Inferrer: given (z_t, z_{t+k}) → latent action code a_k.

Architecture:
  1. Pool patch tokens to frame-level CLS  (mean pool)
  2. Concatenate [z_t; z_{t+k}] along sequence dim
  3. Small transformer → single [ACTION] token
  4. Linear proj to bottleneck_dim
  5. VQ to discrete code

The bottleneck is intentionally narrow to force disentanglement:
the code captures *what changed*, not what the scene looks like.
"""

import torch
import torch.nn as nn
from einops import rearrange
from .vq import VectorQuantizer


class ActionInferrer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.backbone.feat_dim
        hidden = cfg.action_inferrer.hidden_dim
        bottleneck = cfg.action_inferrer.bottleneck_dim
        n_layers = cfg.action_inferrer.num_layers
        n_heads = cfg.action_inferrer.num_heads
        dropout = cfg.action_inferrer.dropout

        # project encoder features to action inferrer dimension
        self.input_proj = nn.Linear(feat_dim, hidden)

        # learnable [ACTION] query token
        self.action_token = nn.Parameter(torch.randn(1, 1, hidden))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # project to VQ bottleneck dimension
        self.to_bottleneck = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
        )

        self.vq = VectorQuantizer(cfg)

    def _pool_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean pool patch tokens → single frame vector. (B, N, D) → (B, D)"""
        return tokens.mean(dim=1)

    def forward(self, z_t: torch.Tensor, z_tk: torch.Tensor):
        """
        Args:
            z_t:  (B, N, D)  patch tokens at time t
            z_tk: (B, N, D)  patch tokens at time t+k
        Returns:
            code:    (B, bottleneck_dim)  quantized action code (same dim as bottleneck)
            indices: (B,)                 codebook index
            vq_loss: scalar
        """
        # pool to frame-level
        f_t  = self._pool_frame(z_t)    # (B, D)
        f_tk = self._pool_frame(z_tk)   # (B, D)

        # project and build sequence: [action_token, f_t, f_{t+k}]
        B = f_t.shape[0]
        seq = torch.stack([f_t, f_tk], dim=1)       # (B, 2, D)
        seq = self.input_proj(seq)                   # (B, 2, H)
        action_tok = self.action_token.expand(B, -1, -1)  # (B, 1, H)
        seq = torch.cat([action_tok, seq], dim=1)    # (B, 3, H)

        out = self.transformer(seq)                  # (B, 3, H)
        action_repr = out[:, 0]                      # (B, H)  — [ACTION] position

        bottleneck = self.to_bottleneck(action_repr) # (B, bottleneck_dim)
        code, indices, vq_loss = self.vq(bottleneck) # (B, code_dim), (B,), scalar

        return code, indices, vq_loss
