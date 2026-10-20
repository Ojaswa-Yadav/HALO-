# HALO — Hierarchical Action Latent Observer

HALO is a research framework for learning latent world models from unlabeled video,
without rewards, environment access, or action labels.

It builds on a frozen video encoder (V-JEPA ViT-B/16) and learns to:
1. **Discover** discrete latent action codes from consecutive frame pairs
2. **Predict** future frame representations conditioned on those codes (low level)
3. **Predict** future segment representations at a coarser timescale (high level)

All prediction happens in latent space — no pixel decoder, no reconstruction loss.

---

## Architecture

```
Frozen V-JEPA Encoder  (never updated)
         │
         ▼
  Action Inferrer  ──→  VQ Codebook (K=512 discrete action codes)
         │
    ┌────┴─────────────────────┐
    ▼                          ▼
Low-Level Predictor     High-Level Predictor
(frame → frame+1)       (segment → segment+T)
    │                          │
    └────────┬─────────────────┘
             ▼
       JEPA Loss (latent space only)
```

---

## Setup

```bash
git clone https://github.com/Ojaswa-Yadav/HALO.git
cd HALO
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Extract features (run once)**
```bash
python scripts/extract_features.py --config configs/base.yaml --split both
```

**Step 2 — Train**
```bash
python scripts/train.py --config configs/base.yaml
```

**Step 3 — Run smoke tests**
```bash
pytest tests/test_shapes.py -v
```

---

## Project Structure

```
HALO/
├── configs/base.yaml          # all hyperparameters
├── data/ssv2.py               # SSv2 dataset loader (raw + feature cache modes)
├── models/
│   ├── encoder.py             # frozen backbone wrapper
│   ├── vq.py                  # EMA vector quantizer
│   ├── action_inferrer.py     # (z_t, z_{t+k}) → discrete action code
│   ├── predictor.py           # low-level + high-level predictors
│   └── hlajepa.py             # full model + loss
├── scripts/
│   ├── extract_features.py    # one-time feature extraction
│   └── train.py               # training loop
└── tests/test_shapes.py       # shape smoke tests
```

---

## Key Design Choices

- **Frozen backbone** — encoder is never updated; all novel computation is in the action inferrer and predictors (~50M params total)
- **VQ bottleneck** — discrete action codes are interpretable and visualizable
- **No pixel decoder** — following the JEPA principle; avoids wasting capacity on texture
- **Hierarchical prediction** — low level models frame transitions, high level models intent across segments

---

## Datasets

| Dataset | Use | Size |
|---|---|---|
| Something-Something-v2 | Pretraining + probing | ~20GB |
| Ego4D (subset) | Generalization | ~100h |
| RoboMimic | Robot manipulation eval | — |
| BridgeData V2 | Visual planning eval | — |

---

## Citation

> Work in progress. Paper forthcoming.
