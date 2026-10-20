"""
Vector Quantizer with EMA codebook updates.
Straight-through estimator for gradients.

Input:  continuous vector z  (..., D)
Output: quantized vector z_q (..., D), indices (...,), commitment loss scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        K = cfg.vq.codebook_size
        D = cfg.vq.code_dim
        self.K = K
        self.D = D
        self.commitment_cost = cfg.vq.commitment_cost
        self.use_ema = cfg.vq.ema_update
        self.decay = cfg.vq.ema_decay

        # codebook embeddings
        embed = torch.randn(K, D)
        nn.init.uniform_(embed, -1 / K, 1 / K)

        if self.use_ema:
            self.register_buffer("embed", embed)
            self.register_buffer("ema_cluster_size", torch.ones(K))
            self.register_buffer("ema_embed_sum", embed.clone())
        else:
            self.embed = nn.Parameter(embed)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., D)  continuous bottleneck vector
        Returns:
            z_q:        (..., D)  quantized (straight-through on backward)
            indices:    (...)    codebook indices
            loss:       scalar   commitment + (optionally) codebook loss
        """
        flat = z.reshape(-1, self.D)                          # (N, D)

        # L2 distances to codebook
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )                                                      # (N, K)
        indices = dist.argmin(1)                               # (N,)
        z_q_flat = self.embed[indices]                         # (N, D)

        if self.training and self.use_ema:
            self._ema_update(flat, indices)

        # commitment loss: push encoder output towards codebook
        loss = self.commitment_cost * F.mse_loss(z_q_flat.detach(), flat)
        if not self.use_ema:
            # also update codebook (non-EMA path)
            loss = loss + F.mse_loss(z_q_flat, flat.detach())

        # straight-through: copy gradients from z_q back to z
        z_q = z_q_flat.reshape(z.shape)
        z_q = z + (z_q - z).detach()

        indices = indices.reshape(z.shape[:-1])
        return z_q, indices, loss

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.K).float()          # (N, K)
        self.ema_cluster_size = (
            self.ema_cluster_size * self.decay
            + (1 - self.decay) * one_hot.sum(0)
        )
        self.ema_embed_sum = (
            self.ema_embed_sum * self.decay
            + (1 - self.decay) * one_hot.t() @ flat
        )
        # Laplace smoothing to avoid dead codes
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        self.embed = self.ema_embed_sum / smoothed.unsqueeze(1)

    @property
    def codebook_utilization(self) -> float:
        """Fraction of codebook entries used in the last forward pass."""
        if self.use_ema:
            return (self.ema_cluster_size > 1.0).float().mean().item()
        return 0.0
