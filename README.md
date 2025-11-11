# Natron Transformer V2

Multi-task Transformer pipeline for institutional-grade financial trading. The system ingests raw OHLCV data, engineers ~100 features, generates balanced multi-head labels, pretrains a Transformer encoder on GPUs, fine-tunes for trading signals, and optionally performs reinforcement learning rollouts. A Flask + socket interface serves low-latency predictions for MetaTrader 5 integrations.

---

## Key Capabilities
- **Feature Engine**: ~118 engineered indicators spanning moving averages, momentum, volatility, volume, market profile, and smart-money concepts.
- **Bias-Reduced Labeling V2**: Balanced buy/sell, adaptive thresholds, 3-class direction, 6 regime states, and automatic distribution reporting.
- **Sequence Builder**: Sliding 96×100 tensors feeding the Natron Transformer.
- **Transformer Core**:
  - Phase 1: Masked token reconstruction + InfoNCE contrastive pretraining.
  - Phase 2: Multi-task supervised fine-tuning (buy/sell, direction, regime).
  - Phase 3: PPO reinforcement learning head (optional).
- **Deployment**: Flask REST API (`/predict`) and asyncio socket bridge for sub-50 ms inference in production.
- **Tooling**: End-to-end scripts for data prep, training, validation, RL, and serving.

---

## Project Layout

```
natron/
  api/                # Flask app & socket bridge
  config/             # Default configuration helpers
  data/               # Feature engine, labels, sequences, pipeline
  evaluation/         # Metrics helpers
  models/             # Transformer backbone and pretraining utilities
  rl/                 # PPO agent and trading environment
  training/           # Training orchestration (pretrain, fine-tune)
  utils/              # Logging, config, seed, torch helpers
configs/
  natron.yaml         # Sample configuration overrides
scripts/
  prepare_data.py     # CSV → features → labels → sequences
  train_pretrain.py   # Phase 1: masked + contrastive
  train_supervised.py # Phase 2: multi-task supervised head
  run_ppo.py          # Phase 3: PPO (optional)
  run_api.py          # Launch Flask inference API
  train_all.py        # Convenience wrapper (Phases 1 & 2)
model/                # Saved checkpoints (`natron_encoder.pt`, `natron_v2.pt`, `natron_ppo.pt`)
artifacts/            # Generated features, labels, sequences, stats
requirements.txt      # Python dependencies (PyTorch 2.x + CUDA stack)
```

---

## Environment Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Ensure CUDA-enabled PyTorch (`torch>=2.2.0`) is installed for GPU training. Adjust the wheel if necessary.

---

## Data Preparation

1. Place your exported candles at `data/data_export.csv` with columns:
   ```
   time, open, high, low, close, volume
   ```
2. Run feature, label, and sequence generation:
   ```bash
   python scripts/prepare_data.py --config configs/natron.yaml
   ```
3. Outputs (in `artifacts/`):
   - `features.parquet` (~100 engineered features)
   - `labels.parquet` (`buy`, `sell`, `direction`, `regime`)
   - `sequences.npy` (N × 96 × 100 tensor)
   - `targets.npz` (aligned label arrays)
   - `close_prices.csv` (needed for PPO)
   - Console report: `"=== 📊 Label Distribution Summary ==="` ensuring healthy balance.

---

## Training Pipeline

### Phase 1 – Transformer Pretraining
```bash
python scripts/train_pretrain.py --config configs/natron.yaml
```
Uses masked reconstruction + InfoNCE with stochastic augmentations. Outputs `model/natron_encoder.pt`.

### Phase 2 – Supervised Fine-Tuning
```bash
python scripts/train_supervised.py --config configs/natron.yaml
```
- Loads pretrained encoder (if available).
- Trains multi-task heads with AdamW + ReduceLROnPlateau.
- Evaluates on temporal hold-out split, printing per-head metrics.
- Saves `model/natron_v2.pt`.

### Phase 3 – Reinforcement Learning (Optional)
```bash
python scripts/run_ppo.py --config configs/natron.yaml
```
- Wraps supervised model inside an actor-critic PPO agent.
- Uses `TradingEnv` reward: `profit − α·turnover − β·drawdown`.
- Saves policy weights at `model/natron_ppo.pt`.

### One-Command Pipeline
```bash
python scripts/train_all.py --config configs/natron.yaml
```
Runs Phases 1 & 2 sequentially (Phase 3 stays manual).

---

## Inference & Serving

### Flask REST API
```bash
python scripts/run_api.py --host 0.0.0.0 --port 8000
```
Request example:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"candles": [... last 96 OHLCV entries ...]}'
```
Response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

### Low-Latency Socket Bridge
```bash
python -m natron.api.socket_server \
       --host 0.0.0.0 --port 9000 \
       --model model/natron_v2.pt
```
*MetaTrader 5* EAs can stream candle JSON and receive trading signals with sub-50 ms latency. See `natron/api/socket_server.py` for protocol details (newline-delimited JSON).

---

## MetaTrader 5 Integration Workflow

1. **Composer 1**: Build EA to stream the latest 96 candles to the Python socket and parse responses.
2. **Opus 4.1**: Deploy the socket/Flask servers (Docker/systemd/tmux recommended) on the GPU VM.
3. **Sonnet 4.5**: Adjust YAML specifications & strategy parameters if market regimes change.
4. **GPT-5 Codex**: Update training scripts & model versions, regenerate checkpoints, monitor real-time performance.

---

## Configuration Notes (`configs/natron.yaml`)
- **paths**: Customize data/model directories.
- **data**: Sequence length, label neutral buffer, split ratios.
- **pretraining**: Epochs, batch size, masking probability, InfoNCE temperature.
- **supervised**: Learning rates, scheduler patience, per-head class weights.
- **model**: Transformer depth, width, heads, feed-forward dim, dropout.
- **rl**: PPO horizon and reward penalties.

Override any setting with additional YAML files or via CLI `--config`.

---

## Testing & Validation
- Training scripts automatically log epoch losses.
- Supervised phase prints classification reports for each head (macro metrics).
- Manual smoke tests:
  ```bash
  python scripts/prepare_data.py
  python scripts/train_pretrain.py
  python scripts/train_supervised.py
  python scripts/run_api.py  # send sample request
  ```
- Use `artifacts/features.parquet` and `labels.parquet` for exploratory analysis in notebooks.

---

## Production Checklist
- [ ] GPU drivers + CUDA-compatible PyTorch installed.
- [ ] `data/data_export.csv` refreshed at desired cadence.
- [ ] Run Phase 1 + 2 training on updated data.
- [ ] Validate metrics & label distribution.
- [ ] Launch Flask or socket server (consider Gunicorn + Nginx for REST, systemd for sockets).
- [ ] Monitor latency (`<50 ms` target) and equity/drawdown via PPO logs.

---

Natron learns the *grammar of the market*: structure (pretraining), signal recognition (supervised), and behavioural adaptation (reinforcement). Tune, retrain, and deploy as regimes evolve. Good trading. 🚀