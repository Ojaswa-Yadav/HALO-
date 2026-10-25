#!/bin/bash
# Download Something-Something-v2
# You get the download URL after registering at:
# https://developer.qualcomm.com/software/ai-datasets/something-something
#
# Usage:
#   bash scripts/download_ssv2.sh

SSV2_DIR="/workspace/ssv2"
mkdir -p "$SSV2_DIR/videos"
mkdir -p "$SSV2_DIR/labels"

echo "Downloading SSv2 label files..."
# Labels (~50MB) — paste your label download URLs here
# wget -c "<LABEL_URL>" -O "$SSV2_DIR/labels/something-something-v2-labels.json"
# wget -c "<TRAIN_URL>"  -O "$SSV2_DIR/labels/something-something-v2-train.json"
# wget -c "<VAL_URL>"    -O "$SSV2_DIR/labels/something-something-v2-validation.json"

echo ""
echo "Downloading SSv2 video parts (~20GB total, 20 parts)..."
# Paste your video download URLs here — one per line
# wget -c "<PART_00_URL>" -O "$SSV2_DIR/20bn-something-something-v2-00"
# wget -c "<PART_01_URL>" -O "$SSV2_DIR/20bn-something-something-v2-01"
# ... (repeat for all 20 parts)

echo ""
echo "Extracting videos..."
for part in "$SSV2_DIR"/20bn-something-something-v2-*; do
    if [ -f "$part" ]; then
        echo "  Extracting $(basename $part)..."
        tar -xf "$part" -C "$SSV2_DIR/videos/" 2>/dev/null || \
        cat "$part" | tar zx -C "$SSV2_DIR/videos/"
    fi
done

echo ""
echo "Done! Total videos: $(ls $SSV2_DIR/videos/ | wc -l)"
echo "Labels: $(ls $SSV2_DIR/labels/)"
