#!/bin/bash
# Download Meta's official V-JEPA ViT-B/16 encoder checkpoint
# Source: https://github.com/facebookresearch/jepa
#
# Usage: bash scripts/download_vjepa.sh

CKPT_DIR="/workspace/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Downloading V-JEPA ViT-B/16 checkpoint..."

# Meta hosts V-JEPA weights on their JEPA GitHub releases
wget -c \
  "https://dl.fbaipublicfiles.com/jepa/vith16.pth.tar" \
  -O "$CKPT_DIR/vjepa_vitb16.pth"

echo "Saved to $CKPT_DIR/vjepa_vitb16.pth"
