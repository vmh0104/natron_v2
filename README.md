# Natron Transformer V2

Natron Transformer V2 is an end-to-end multi-task trading research pipeline that learns market structure from OHLCV candles, produces actionable trade signals, and serves them in real time. The system targets Linux GPU servers (Ubuntu/Debian, Python 3.10+, PyTorch 2.x CUDA) and is organized into three learning phases:

1. **Masked & Contrastive Pretraining** – captures latent market grammar from unlabeled sequences.
2. **Supervised Multi-Task Fine-Tuning** – predicts buy/sell, 3-way direction, and 6-state regimes.
3. **Optional PPO Reinforcement Learning** – optimizes policy for profit, turnover, and drawdown.

A Flask HTTP service and TCP socket bridge deliver sub-50 ms predictions ready for MetaTrader 5 or custom execution stacks.

---

## Repository Layout

```
natron/
  api/                # Flask REST app, socket bridge, reusable predictor
  configs/            # YAML configuration presets
  data/               # Data module orchestrating features, labels, splits
  datasets/           # Sequence datasets and scalers
  features/           # ~100 engineered technical indicators
  labels/             # Bias-reduced institutional labeling logic
  models/             # Transformer encoder with multi-task heads
  rl/                 # PPO agent + trading environment
  scripts/            # CLI entrypoints for training phases
  training/           # Shared optimizers, schedulers, loops
  utils/              # Logging, metrics, seeding helpers
```

Artifacts (checkpoints, logs, scalers) default to `artifacts/`. Adjust any path via `natron/configs/natron_base.yaml`.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install torch with your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core scientific stack
pip install pandas numpy scikit-learn pyyaml flask
```

Optional: `tmux`, `docker`, `systemd` for deployment automation.

> **GPU Note:** Set `training.device`/`api.device` to `"cuda"` in YAML. Scripts auto-fallback to CPU when CUDA is unavailable.

---

## Data Requirements

- CSV path (default `data/data_export.csv`).
- Columns: `time, open, high, low, close, volume`.
- `time` must be parseable by `pandas.to_datetime`.
- Provide ≥ 96 + max rolling window (~200) candles for stable features.

Update `data.data_path` in the YAML if your files live elsewhere.

---

## Phase 1 — Pretraining (Masked + Contrastive)

```bash
python -m natron.scripts.pretrain \
  --config natron/configs/natron_base.yaml \
  --output artifacts/model/pretrain.pt
```

- Masks ~15 % tokens and reconstructs features (MSE loss).
- Runs InfoNCE between clean/augmented sequences (SimCLR-style).
- Persists weights (`pretrain.pt`) and feature scaler (`pretrain.scaler.npz`).

---

## Phase 2 — Supervised Multi-Task Fine-Tuning

```bash
python -m natron.scripts.train \
  --config natron/configs/natron_base.yaml \
  --pretrained artifacts/model/pretrain.pt \
  --output artifacts/model/natron_v2.pt
```

- Optimizes BCE + CE losses for buy/sell/direction/regime heads.
- Prints validation AUC/accuracy metrics per task.
- Saves fine-tuned weights (`natron_v2.pt`) and scaler (`natron_v2.scaler.npz`).

---

## Phase 3 — PPO Reinforcement Learning (Optional)

Enable RL via `rl.enabled: true` in the config and run:

```bash
python -m natron.scripts.rl_train \
  --config natron/configs/natron_base.yaml \
  --pretrained artifacts/model/natron_v2.pt \
  --iterations 200
```

- Uses frozen transformer embeddings by default (set `train_backbone=True` to fine-tune).
- Discrete actions: short / flat / long.
- Reward: `profit - α × turnover - β × drawdown` (set α, β in YAML).
- Outputs PPO head parameters (`natron_rl.pt`).

---

## Inference Services

### HTTP API

```bash
export FLASK_APP=natron.api.app:build_app
flask run --host 0.0.0.0 --port 8080
```

Request body: last 96 candles (`time`, `open`, `high`, `low`, `close`, `volume`).

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [...96 latest bars...]}'
```

Typical response:

```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_neutral": 0.12,
  "direction_down": 0.19,
  "regime": "BULL_WEAK",
  "confidence": 0.82,
  "regime_probs": {
    "BULL_STRONG": 0.18,
    "BULL_WEAK": 0.33,
    "RANGE": 0.22,
    "BEAR_WEAK": 0.12,
    "BEAR_STRONG": 0.09,
    "VOLATILE": 0.06
  }
}
```

### TCP Socket Bridge (MQL5)

```bash
python -m natron.api.socket_server \
  --config natron/configs/natron_base.yaml \
  --host 0.0.0.0 --port 5555
```

- Receives JSON payloads from MetaTrader 5 Expert Advisors.
- Returns compact JSON with probabilities, regime label, confidence.
- Designed for <50 ms round-trip latency on LAN.

---

## Configuration Highlights (`natron_base.yaml`)

- `data`: CSV path, sequence length (96), train/val/test splits, loader workers.
- `features`: rolling periods, MACD/RSI/ATR settings, profile bins, noise floor.
- `labels`: institutional buy/sell logic, balance targets, regime thresholds, stochastic jitter.
- `model`: transformer depth, attention heads, feed-forward width, dropout, masking ratio, projection size.
- `training`: epochs, batch size, learning-rate scheduling, AMP, clipping.
- `rl`: PPO hyperparameters, reward weights, rollout sizes.
- `api`: inferencing host/port, model artifact path, device.

Clone the base config, tweak hyperparameters per instrument/timeframe, and pass via `--config`.

---

## Development & Testing Notes

- Feature engine exports ~100 indicators across moving averages, momentum, volatility, volume, price patterns, statistical moments, support/resistance, smart money concepts, and market profile metrics.
- Label generator prints distribution diagnostics on each run—monitor for class imbalance before training.
- `FeatureScaler` stores train-split mean/std and is persisted alongside each checkpoint.
- Training scripts use gradient clipping, mixed precision (optional), and `ReduceLROnPlateau`.
- PPO trainer logs policy/value/entropy losses every iteration for monitoring.

---

## Roadmap Ideas

- Integrate experiment tracking (Weights & Biases, MLflow).
- Add multi-symbol RL portfolios with position sizing.
- Provide MetaTrader 5 EA template and visualization dashboards.
- Package Docker images with GPU base layers for reproducible deployment.

Natron’s philosophy:

> **“Natron doesn’t just predict Buy/Sell — it learns the grammar of the market.”**

Use the three-phase pipeline to expose the model to structure, signals, and behavior that reflect your trading thesis. Test thoroughly and manage risk responsibly.