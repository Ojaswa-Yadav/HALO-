"""
Smoke tests — verify tensor shapes through the full forward pass.
No GPU or data required.

Run: pytest tests/test_shapes.py -v
"""

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

# ── Minimal config for testing ────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return OmegaConf.create({
        "backbone": {"name": "vjepa", "feat_dim": 768, "patch_size": 16, "frozen": True},
        "data": {"num_frames": 16, "img_size": 224},
        "vq": {"codebook_size": 512, "code_dim": 256, "commitment_cost": 0.25,
               "ema_update": True, "ema_decay": 0.99},
        "action_inferrer": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4,
                            "bottleneck_dim": 256, "dropout": 0.0},
        "low_level_predictor": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4, "dropout": 0.0},
        "high_level_predictor": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4,
                                  "segment_len": 4, "dropout": 0.0},
        "training": {"loss_weights": {"low_level": 1.0, "high_level": 0.5, "vq_commit": 0.25}},
    })


# ── VQ ────────────────────────────────────────────────────────────────────────

def test_vq_shapes(cfg):
    from models.vq import VectorQuantizer
    vq = VectorQuantizer(cfg)
    z = torch.randn(8, 256)
    z_q, idx, loss = vq(z)
    assert z_q.shape == (8, 256)
    assert idx.shape == (8,)
    assert loss.ndim == 0


def test_vq_straight_through(cfg):
    from models.vq import VectorQuantizer
    vq = VectorQuantizer(cfg)
    z = torch.randn(4, 256, requires_grad=True)
    z_q, _, _ = vq(z)
    z_q.sum().backward()
    assert z.grad is not None, "Straight-through gradient must flow"


# ── Action Inferrer ───────────────────────────────────────────────────────────

def test_action_inferrer_shapes(cfg):
    from models.action_inferrer import ActionInferrer
    ai = ActionInferrer(cfg)
    B, N, D = 4, 196, 768
    z_t  = torch.randn(B, N, D)
    z_tk = torch.randn(B, N, D)
    code, idx, loss = ai(z_t, z_tk)
    assert code.shape == (B, 256)
    assert idx.shape  == (B,)
    assert loss.ndim  == 0


# ── Predictors ────────────────────────────────────────────────────────────────

def test_low_level_predictor_shapes(cfg):
    from models.predictor import LowLevelPredictor
    pred = LowLevelPredictor(cfg)
    B, N, D = 4, 196, 768
    z = torch.randn(B, N, D)
    a = torch.randn(B, 256)
    out = pred(z, a)
    assert out.shape == (B, N, D)


def test_high_level_predictor_shapes(cfg):
    from models.predictor import HighLevelPredictor
    pred = HighLevelPredictor(cfg)
    B, D = 4, 768
    seg = torch.randn(B, 1, D)
    a   = torch.randn(B, 256)
    out = pred(seg, a)
    assert out.shape == (B, 1, D)


# ── Full model (mock encoder) ─────────────────────────────────────────────────

def test_hlajepa_forward(cfg):
    from models.hlajepa import HLAJEPA

    model = HLAJEPA(cfg)

    # mock the frozen encoder to avoid loading real weights
    B, T, N, D = 2, 16, 196, 768
    fake_tokens = torch.randn(B, T, N, D)
    with patch.object(model.encoder, "forward", return_value=fake_tokens):
        frames = torch.randn(B, T, 3, 224, 224)
        out = model(frames)

    assert out.loss.ndim == 0
    assert out.loss_low.ndim  == 0
    assert out.loss_high.ndim == 0
    assert out.action_indices.shape == (B, T - 1)


def test_hlajepa_loss_positive(cfg):
    from models.hlajepa import HLAJEPA
    model = HLAJEPA(cfg)
    B, T, N, D = 2, 16, 196, 768
    fake_tokens = torch.randn(B, T, N, D)
    with patch.object(model.encoder, "forward", return_value=fake_tokens):
        out = model(torch.randn(B, T, 3, 224, 224))
    assert out.loss.item() > 0
