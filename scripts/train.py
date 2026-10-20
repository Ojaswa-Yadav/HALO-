"""
Training entry point for H-LAJEPA.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml training.batch_size=128
"""

import os
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HLAJEPA
from data.ssv2 import SSv2Dataset

console = Console()


# ── Optimizer / scheduler ────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    # only optimize non-frozen params
    params = [p for p in model.parameters() if p.requires_grad]
    console.log(f"Trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    return torch.optim.AdamW(
        params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )


def cosine_lr_schedule(step: int, cfg) -> float:
    warmup = cfg.training.warmup_steps
    total  = cfg.training.max_steps
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ── Main training loop ───────────────────────────────────────────────────────

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = HLAJEPA(cfg).to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = SSv2Dataset(cfg, split="train")
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_lr_schedule(s, cfg)
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.logging.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────────
    step = 0
    model.train()

    with Progress(SpinnerColumn(), *Progress.get_default_columns(),
                  TimeElapsedColumn()) as progress:
        task = progress.add_task("Training", total=cfg.training.max_steps)

        while step < cfg.training.max_steps:
            for batch in train_loader:
                if step >= cfg.training.max_steps:
                    break

                frames = batch["frames"].to(device, non_blocking=True)  # (B,T,C,H,W)

                optimizer.zero_grad(set_to_none=True)
                out = model(frames)
                out.loss.backward()

                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.training.grad_clip,
                )
                optimizer.step()
                scheduler.step()

                # ── Logging ──────────────────────────────────────────────────
                if step % cfg.logging.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    util = model.action_inferrer.vq.codebook_utilization
                    log = {
                        "loss/total":    out.loss.item(),
                        "loss/low":      out.loss_low.item(),
                        "loss/high":     out.loss_high.item(),
                        "loss/vq":       out.loss_vq.item(),
                        "vq/utilization": util,
                        "lr":            lr,
                        "step":          step,
                    }
                    wandb.log(log, step=step)
                    progress.print(
                        f"[bold]step {step:6d}[/bold]  "
                        f"loss={out.loss.item():.4f}  "
                        f"low={out.loss_low.item():.4f}  "
                        f"high={out.loss_high.item():.4f}  "
                        f"vq_util={util:.2f}  lr={lr:.2e}"
                    )

                # ── Checkpoint ───────────────────────────────────────────────
                if step % cfg.logging.ckpt_every == 0 and step > 0:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    torch.save({
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "cfg": OmegaConf.to_container(cfg),
                    }, ckpt_path)
                    console.log(f"Saved checkpoint: {ckpt_path}")

                step += 1
                progress.advance(task)

    wandb.finish()
    console.log("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    console.log(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
