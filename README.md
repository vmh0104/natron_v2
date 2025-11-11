# Natron V2 — Multi-Task Financial Transformer

Natron V2 is an end-to-end trading intelligence stack that learns the "grammar of the market" across three phases:
masked-modeling pretraining, multi-task supervised fine-tuning, and optional policy-gradient reinforcement learning.
The system ingests 96-candle OHLCV sequences, generates ~100 engineered factors, and produces actionable
buy/sell probabilities, directional outlook, regime classification, and RL-aligned trade actions with sub-50 ms runtime targets.

## Key Features

- **FeatureEngine**: ~100 institutional-grade signals (trend, momentum, volatility, SMC, market profile, etc.) computed from raw OHLCV.
- **LabelGeneratorV2**: Balanced buy/sell signals, tri-class direction (up/down/neutral), six-state regimes, adaptive thresholds, diagnostics.
- **SequenceCreator**: Sliding-window tensors `(96 × 100)` aligned with label endpoints, ready for GPU training.
- **Transformer Core**: Shared encoder + multi-task heads for buy, sell, direction, and regime outputs.
- **Phase 1 (Pretrain)**: Masked token reconstruction + InfoNCE contrastive loss for latent structure discovery.
- **Phase 2 (Fine-Tune)**: Multi-task supervision with BCE + CE losses, LR scheduler, gradient clipping, checkpointing.
- **Phase 3 (RL, optional)**: Contextual policy-gradient bandit (long/flat/short) with turnover & drawdown penalties.
- **Inference Stack**: `NatronPredictor` (batch or streaming), Flask REST API, async TCP bridge for MetaTrader 5 / custom clients.
- **Model Bundle**: Single checkpoint (`model/natron_v2.pt`) containing weights, scaler stats, feature ordering, RL policy.

## Repository Layout

```
configs/                YAML configs (defaults in `natron.yaml`)
natron/
  data/                 Feature engineering, labeling, sequencing, data module
  models/               Transformer encoder & pretraining heads
  pipelines/            Training orchestration + inference utilities
  serving/              Flask API + async socket bridge
  training/             Phase-specific training loops (pretrain, finetune, RL)
  utils/                Config and logging helpers
scripts/                CLI entry points (training, API, socket server)
requirements.txt        Python dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Python**: 3.10+ recommended. **PyTorch**: install the CUDA build matching your GPU (`pip install torch==<version>+cu118`).

## Data Requirements

Prepare `data/data_export.csv` (or update `configs/natron.yaml`) with columns:
`time, open, high, low, close, volume`. Each row is a 15m/1h candle.
The pipeline automatically:

1. Sorts by `time`
2. Computes ~100 features (forward/backward fill of gaps)
3. Generates labels with adaptive balancing
4. Prints distribution diagnostics

Example diagnostics (auto-emitted during `LabelGeneratorV2.transform`):

```
=== 📊 Label Distribution Summary ===
▶ BUY distribution:
0.0    0.36
1.0    0.34
...
```

## Training Pipeline

1. Configure `configs/natron.yaml` (paths, device, batch sizes, LR, RL toggles).
2. Launch the orchestrator:

```bash
python scripts/run_training.py --config configs/natron.yaml
```

Artifacts:

- Checkpoints per phase in `checkpoints/` (`*_pretrain.pt`, `*_finetune.pt`, `*_rl.pt`)
- Final bundle in `model/natron_v2.pt`
- Logs in `logs/`

### Phase Summary

| Phase | Objective | Key Losses |
|-------|-----------|------------|
| Pretrain | Masked reconstruction + InfoNCE | MSE, contrastive CrossEntropy |
| Fine-Tune | Multi-task prediction | BCE (buy/sell) + CE (direction/regime) |
| RL (optional) | Long/flat/short policy refinement | Policy gradient w/ turnover & drawdown penalization |

Disable any phase via `phases.<name>.enabled` in the YAML.

## Inference & Serving

### Python Predictor

```python
from pathlib import Path
import pandas as pd
from natron.pipelines.inference import load_predictor

predictor = load_predictor(Path("model/natron_v2.pt"), device="cuda")
recent_candles = pd.read_csv("latest_candles.csv")
result = predictor.predict(recent_candles.tail(96))
print(result)
```

Example response:

```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction": "UP",
  "direction_probs": {"DOWN": 0.16, "UP": 0.69, "NEUTRAL": 0.15},
  "regime": "BULL_WEAK",
  "regime_probs": {"BULL_STRONG": 0.18, ..., "VOLATILE": 0.04},
  "confidence": 0.82,
  "timestamp": "2024-01-05T12:00:00"
}
```

If RL is enabled, payload also includes `rl_action` (`LONG/FLAT/SHORT`) and probabilities.

### REST API

```bash
python scripts/serve_api.py --model model/natron_v2.pt --host 0.0.0.0 --port 8000
```

Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [... last 96 OHLCV objects ...]}'
```

Health check: `GET /health`

### Async TCP Bridge (MQL5 / custom clients)

```bash
python scripts/run_socket_server.py --model model/natron_v2.pt --port 9000
```

Protocol: newline-delimited JSON.

| Request | Description |
|---------|-------------|
| `{"type": "ping"}` | Heartbeat |
| `{"type": "predict", "candles": [...]}` | Returns `{ "type": "prediction", "data": {...} }` |

Ideal for MetaTrader 5 Expert Advisor via sockets.

## Checkpoint Contents (`model/natron_v2.pt`)

```python
{
  "model_state": Transformer weights,
  "model_config": dataclass dict,
  "feature_columns": ordered feature names,
  "scaler_mean" / "scaler_scale": StandardScaler stats,
  "rl_policy_state": optional RL policy weights,
  "window_length": sequence length (default 96),
  "phases": original YAML phase config
}
```

Load with `NatronModelBundle.load(path)` to recreate the predictor stack.

## Configuration Highlights (`configs/natron.yaml`)

- `pretraining.masking_ratio`: fraction of tokens masked
- `finetuning.lr`, `finetuning.scheduler`: AdamW + ReduceLROnPlateau parameters
- `reinforcement.reward.alpha/beta`: turnover & drawdown penalties
- `reinforcement.train_encoder`: set `true` to fine-tune encoder during RL

## Development Notes

- Default feature dimensionality: 100. Adjustments require updating label config & model.
- FeatureEngine fills NaNs via forward/backward fill; validate against exchange outages.
- LabelGenerator prints distributions each run — monitor for drift.
- Training loops use `tqdm` for epoch progress and clip gradients to 1.0.
- Estimated GPU memory (d_model=256, batch=64): ~8 GB during fine-tuning.

## Next Steps / Extensions

1. Swap RL phase with PPO/SAC for multi-step trajectories (plug in Stable-Baselines3).
2. Add real-time feature cache to avoid recomputation in live trading.
3. Integrate experiment tracking (Weights & Biases) via callback hooks in training loops.
4. Build dashboards on top of REST API (`/model-info`, `/predict`).

Natron V2 is built for research + production parity: feature parity, consistent scaling, and unified model artifact
ensure the same intelligence trains offline and executes live.
