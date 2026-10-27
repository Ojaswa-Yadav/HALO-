"""
Something-Something-v2 dataset loader.

Two modes:
  1. RAW mode:    reads video files directly with decord  (slower, flexible)
  2. FEAT mode:   loads precomputed encoder features from .pt files (fast, recommended)

Directory structure expected (raw):
    <root>/
        videos/          # .webm files named {video_id}.webm
        labels/
            train.json
            validation.json

Directory structure expected (feat):
    <feat_cache>/
        train/{video_id}.pt    # tensor (T, N, D)
        validation/{video_id}.pt
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


# ── Transforms ───────────────────────────────────────────────────────────────

def build_video_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class SSv2Dataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        assert split in ("train", "validation")
        self.cfg   = cfg
        self.split = split

        root       = Path(cfg.data.root)
        feat_cache = Path(cfg.data.feat_cache) if cfg.data.feat_cache else None

        # detect mode
        if feat_cache and (feat_cache / split).exists():
            self.mode = "feat"
            self.feat_dir = feat_cache / split
        else:
            assert DECORD_AVAILABLE, (
                "decord not installed and no feature cache found. "
                "Either install decord or run scripts/extract_features.py first."
            )
            self.mode = "raw"
            # SSv2 extracts into a subdirectory — handle both layouts
            video_dir = root / "videos" / "20bn-something-something-v2"
            if not video_dir.exists():
                video_dir = root / "videos"
            self.video_dir = video_dir
            self.transform = build_video_transform(cfg.data.img_size)

        # load metadata — handle nested labels/ folder from zip extraction
        label_file = root / "labels" / "labels" / f"{split}.json"
        if not label_file.exists():
            label_file = root / "labels" / f"{split}.json"
        with open(label_file) as f:
            meta = json.load(f)

        self.samples: List[Dict] = meta  # list of {"id": ..., "label": ..., "template": ...}
        self.num_frames   = cfg.data.num_frames
        self.frame_stride = cfg.data.frame_stride

    def __len__(self) -> int:
        return len(self.samples)

    def _load_feat(self, video_id: str) -> torch.Tensor:
        path = self.feat_dir / f"{video_id}.pt"
        tokens = torch.load(path, map_location="cpu")  # (T_total, N, D)
        return self._sample_frames_feat(tokens)

    def _sample_frames_feat(self, tokens: torch.Tensor) -> torch.Tensor:
        T_total = tokens.shape[0]
        indices = self._sample_indices(T_total)
        return tokens[indices]  # (num_frames, N, D)

    def _load_raw(self, video_id: str) -> torch.Tensor:
        path = self.video_dir / f"{video_id}.webm"
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
            T_total = len(vr)
            indices = self._sample_indices(T_total)
            frames = vr.get_batch(indices).asnumpy()   # (T, H, W, C) uint8
        except Exception:
            # corrupted / undecodable video — return blank frames
            H = W = self.cfg.data.img_size
            frames = np.zeros((self.num_frames, H, W, 3), dtype=np.uint8)

        frames_t = []
        for f in frames:
            img = Image.fromarray(f)
            frames_t.append(self.transform(img))
        return torch.stack(frames_t)               # (T, C, H, W)

    def _sample_indices(self, T_total: int) -> List[int]:
        """Sample num_frames evenly spaced indices."""
        stride    = self.frame_stride
        available = list(range(0, T_total, stride))
        if len(available) >= self.num_frames:
            # uniform subsample
            step = len(available) / self.num_frames
            return [available[int(i * step)] for i in range(self.num_frames)]
        else:
            # repeat last frame if video is too short
            pad = self.num_frames - len(available)
            return available + [available[-1]] * pad

    def __getitem__(self, idx: int) -> Dict:
        sample   = self.samples[idx]
        video_id = str(sample["id"])
        raw_label = sample.get("label", -1)
        try:
            label = int(raw_label)
        except (ValueError, TypeError):
            label = -1  # text labels used during pretraining — ignored

        if self.mode == "feat":
            frames = self._load_feat(video_id)    # (T, N, D)  — pre-encoded tokens
        else:
            frames = self._load_raw(video_id)     # (T, C, H, W)  — raw frames

        # HLAJEPA.forward() handles both shapes:
        #   (B, T, N, D) → skips encoder (feat mode)
        #   (B, T, C, H, W) → runs encoder on-the-fly (raw mode)
        return {
            "frames":   frames,
            "label":    label,
            "video_id": video_id,
        }
