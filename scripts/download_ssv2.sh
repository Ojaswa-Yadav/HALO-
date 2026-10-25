#!/bin/bash
# Download Something-Something-v2
# URLs from: https://developer.qualcomm.com/software/ai-datasets/something-something
#
# Usage:
#   bash scripts/download_ssv2.sh

set -e

SSV2_DIR="/workspace/ssv2"
mkdir -p "$SSV2_DIR/videos"
mkdir -p "$SSV2_DIR/labels"

# ── 1. Download labels ────────────────────────────────────────────────────────
echo "[1/3] Downloading labels..."
wget -c \
  "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip" \
  -O "$SSV2_DIR/labels.zip"

echo "  Extracting labels..."
unzip -o "$SSV2_DIR/labels.zip" -d "$SSV2_DIR/labels/"
echo "  Labels ready: $(ls $SSV2_DIR/labels/)"

# ── 2. Download video parts ───────────────────────────────────────────────────
echo ""
echo "[2/3] Downloading video parts (~19.4GB)..."

wget -c \
  "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00" \
  -O "$SSV2_DIR/20bn-something-something-v2-00"

wget -c \
  "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01" \
  -O "$SSV2_DIR/20bn-something-something-v2-01"

# ── 3. Extract videos ─────────────────────────────────────────────────────────
echo ""
echo "[3/3] Extracting videos (this takes a few minutes)..."

# Per the official instructions: unzip parts first, then cat | tar
cd "$SSV2_DIR"
unzip "20bn-something-something-v2-??"  2>/dev/null || true
cat 20bn-something-something-v2-?? | tar -xvzf - -C "$SSV2_DIR/videos/"

echo ""
echo "Done!"
echo "  Videos: $(ls $SSV2_DIR/videos/ | wc -l)"
echo "  Labels: $(ls $SSV2_DIR/labels/)"
