"""
One-time feature extraction: run frozen encoder over all SSv2 videos
and save per-video .pt files to feat_cache directory.

Run this ONCE before training. On 1× A100 takes ~4 hours for full SSv2.
On Kaggle T4 takes ~10-12 hours (run overnight).

Usage:
    python scripts/extract_features.py --config configs/base.yaml
    python scripts/extract_features.py --config configs/base.yaml --split validation
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import FrozenEncoder
from data.ssv2 import SSv2Dataset


def extract(cfg, split: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split: {split}")

    # Force raw mode for extraction
    feat_dir = Path(cfg.data.feat_cache) / split
    feat_dir.mkdir(parents=True, exist_ok=True)

    # temporarily disable feat_cache so dataset uses raw mode
    cfg_raw = OmegaConf.merge(cfg, {"data": {"feat_cache": ""}})
    dataset = SSv2Dataset(cfg_raw, split=split)
    loader  = DataLoader(
        dataset,
        batch_size=32,             # encode 32 clips at once
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    encoder = FrozenEncoder(cfg).to(device)
    encoder.eval()

    skipped = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split}"):
            frames    = batch["frames"].to(device)   # (B, T, C, H, W)
            video_ids = batch["video_id"]

            tokens = encoder(frames)                 # (B, T, N, D)
            tokens = tokens.cpu()

            for i, vid_id in enumerate(video_ids):
                out_path = feat_dir / f"{vid_id}.pt"
                if out_path.exists():
                    skipped += 1
                    continue
                torch.save(tokens[i], out_path)      # (T, N, D)

    print(f"Done. Saved to {feat_dir}. Skipped {skipped} existing.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation", "both"])
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    splits = ["train", "validation"] if args.split == "both" else [args.split]
    for split in splits:
        extract(cfg, split)


if __name__ == "__main__":
    main()
