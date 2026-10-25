#!/bin/bash
# HALO — RunPod Setup Script
# Run this once after spinning up a fresh pod.
# Everything persists on the Network Volume at /workspace
#
# Usage:
#   bash scripts/setup_runpod.sh
#
# Before running:
#   1. Create a Network Volume (50GB) in RunPod dashboard
#   2. Attach it to your pod at /workspace
#   3. SSH into the pod and run this script

set -e  # exit on any error

echo "============================================"
echo "  HALO RunPod Setup"
echo "============================================"

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/HALO"
SSV2_DIR="$WORKSPACE/ssv2"
CKPT_DIR="$WORKSPACE/checkpoints"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg unzip wget git

# ── 2. Clone HALO repo ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Cloning HALO repo..."
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/Ojaswa-Yadav/HALO.git "$REPO_DIR"
fi

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    torch torchvision \
    einops timm \
    omegaconf \
    wandb tqdm rich \
    decord \
    scikit-learn \
    pytest

# ── 4. Download V-JEPA weights ────────────────────────────────────────────────
echo ""
echo "[4/5] Downloading V-JEPA ViT-B/16 weights..."
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_DIR/vjepa_vitb16.pth" ]; then
    echo "  Weights already downloaded, skipping."
else
    wget -q -c \
        "https://dl.fbaipublicfiles.com/jepa/vith16.pth.tar" \
        -O "$CKPT_DIR/vjepa_vitb16.pth"
    echo "  Saved to $CKPT_DIR/vjepa_vitb16.pth"
fi

# ── 5. Download Something-Something-v2 ───────────────────────────────────────
echo ""
echo "[5/5] Setting up SSv2..."
mkdir -p "$SSV2_DIR/videos"
mkdir -p "$SSV2_DIR/labels"

if [ "$(ls -A $SSV2_DIR/videos 2>/dev/null | wc -l)" -gt "1000" ]; then
    echo "  SSv2 videos already present ($(ls $SSV2_DIR/videos | wc -l) files), skipping."
else
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │  SSv2 requires manual download (registration required)  │"
    echo "  │                                                         │"
    echo "  │  1. Register at:                                        │"
    echo "  │     https://developer.qualcomm.com/software/ai-datasets │"
    echo "  │     /something-something                                │"
    echo "  │                                                         │"
    echo "  │  2. Get download links from the dataset page            │"
    echo "  │                                                         │"
    echo "  │  3. Run:                                                │"
    echo "  │     bash scripts/download_ssv2.sh <your_download_url>  │"
    echo "  └─────────────────────────────────────────────────────────┘"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Download SSv2 (see instructions above)"
echo "  2. Run smoke tests:"
echo "     cd $REPO_DIR && pytest tests/test_shapes.py -v"
echo "  3. Start training:"
echo "     cd $REPO_DIR && python scripts/train.py \\"
echo "       --config configs/base.yaml \\"
echo "       data.root=$SSV2_DIR \\"
echo "       backbone.checkpoint_path=$CKPT_DIR/vjepa_vitb16.pth \\"
echo "       logging.ckpt_dir=$CKPT_DIR/halo"
echo "============================================"
