"""
Frozen video encoder wrapper.
Supports V-JEPA (ViT-B/16) loaded from HuggingFace or local checkpoint.
Outputs per-frame patch tokens: (B, T, N, D) where N = num_patches, D = feat_dim.
"""

import torch
import torch.nn as nn
from einops import rearrange


class FrozenEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = cfg.backbone.feat_dim
        self._build_backbone()
        self._freeze()

    def _build_backbone(self):
        name = self.cfg.backbone.name

        if name == "vjepa":
            # V-JEPA ships as a plain ViT; load via timm or HuggingFace
            # We wrap it to handle temporal input (B, T, C, H, W)
            import timm
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,          # drop classification head
                global_pool="",         # return all patch tokens
            )

        elif name == "dinov2":
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            )

        elif name == "videomae":
            from transformers import VideoMAEModel
            self.backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base"
            )

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        # keep backbone frozen regardless of module train mode
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)  float32 in [0, 1]
        Returns:
            tokens: (B, T, N, D)  per-frame patch tokens
        """
        B, T, C, H, W = frames.shape
        # encode each frame independently (works for image ViTs)
        flat = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.cfg.backbone.name == "videomae":
            # VideoMAE expects (B, T, C, H, W) directly
            out = self.backbone(frames).last_hidden_state  # (B, T*N, D)
            N = out.shape[1] // T
            tokens = rearrange(out, "b (t n) d -> b t n d", t=T, n=N)
        else:
            tokens = self.backbone.forward_features(flat)  # (B*T, N, D)
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(1)  # add N dim if pooled
            tokens = rearrange(tokens, "(b t) n d -> b t n d", b=B, t=T)

        return tokens  # (B, T, N, D)
