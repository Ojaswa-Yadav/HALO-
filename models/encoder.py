"""
Frozen video encoder wrapper.

Supports:
  - V-JEPA ViT-B/16  (Meta, loaded from local checkpoint downloaded by setup script)
  - DINOv2 ViT-B/14  (Meta, via torch.hub)
  - VideoMAE ViT-B   (MCG-NJU, via HuggingFace)

V-JEPA weights are NOT on HuggingFace — Meta releases them as raw .pth files.
Download script: scripts/download_vjepa.sh
Checkpoint path set in config: backbone.checkpoint_path

Output shape: (B, T, N, D)
  B = batch, T = frames, N = num_patches (196 for ViT-B/16 @ 224px), D = 768
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path


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
            self.backbone = self._load_vjepa()

        elif name == "dinov2":
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            )

        elif name == "videomae":
            from transformers import VideoMAEModel
            self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def _load_vjepa(self):
        """
        Load V-JEPA ViT-B/16 encoder from Meta's official checkpoint.

        Meta's V-JEPA checkpoint structure:
          {
            'encoder': { state_dict of ViT },
            'predictor': { ... }   ← we ignore this
          }

        Checkpoint is downloaded by scripts/download_vjepa.sh to:
          backbone.checkpoint_path (set in configs/base.yaml)
        """
        import timm

        # Build ViT-B/16 with no head, return all patch tokens
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,       # we load weights manually below
            num_classes=0,
            global_pool="",
        )

        ckpt_path = self.cfg.backbone.get("checkpoint_path", "")

        if ckpt_path and Path(ckpt_path).exists():
            print(f"Loading V-JEPA weights from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # Meta checkpoint may have 'encoder' or 'target_encoder' key
            if "encoder" in ckpt:
                state = ckpt["encoder"]
            elif "target_encoder" in ckpt:
                state = ckpt["target_encoder"]
            else:
                state = ckpt  # assume raw state dict

            # strip any 'module.' prefix from DDP training
            state = {k.replace("module.", ""): v for k, v in state.items()}

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
            print("V-JEPA weights loaded.")
        else:
            print(
                "WARNING: V-JEPA checkpoint not found. "
                "Run scripts/download_vjepa.sh first. "
                "Using random weights for now (fine for smoke tests)."
            )

        return model

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        # backbone stays frozen regardless of module train/eval mode
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)  float32, ImageNet-normalized
        Returns:
            tokens: (B, T, N, D)     patch tokens per frame
        """
        B, T, C, H, W = frames.shape

        if self.cfg.backbone.name == "videomae":
            # VideoMAE processes full clip at once
            out = self.backbone(frames).last_hidden_state  # (B, T*N, D)
            N = out.shape[1] // T
            return rearrange(out, "b (t n) d -> b t n d", t=T, n=N)

        # For image ViTs (V-JEPA, DINOv2): encode each frame independently
        flat = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.cfg.backbone.name == "dinov2":
            # DINOv2 returns dict — patch tokens are under 'x_norm_patchtokens'
            out = self.backbone.forward_features(flat)
            tokens = out["x_norm_patchtokens"]              # (B*T, N, 768)
        else:
            tokens = self.backbone.forward_features(flat)   # (B*T, N, D)
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(1)

        return rearrange(tokens, "(b t) n d -> b t n d", b=B, t=T)
