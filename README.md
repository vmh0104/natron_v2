# 🧠 Natron Transformer V2 – Multi-Task Financial Trading Model

Natron V2 is an end-to-end GPU-ready trading research stack that learns the “grammar of the market.” The pipeline ingests raw OHLCV candles, engineers ~100 technical factors, applies institutional-grade labeling, pretrains a transformer encoder, fine-tunes multi-task heads, optionally trains a PPO policy, and serves real-time signals over REST or sockets for MetaTrader 5 integrations.

---

## 🚀 Capabilities
- **Multi-task supervision**: joint Buy/Sell, direction (up/down/neutral), and 6-state regime classification.
- **Masked modeling + contrastive pretraining**: self-supervised phase builds temporal market understanding.
- **Optional PPO reinforcement learning**: optimizes reward = profit − α·turnover − β·drawdown.
- **GPU-first implementation**: PyTorch 2.x, Lightning 2.x, BF16-friendly, CUDA-aware.
- **Realtime inference**: Flask `/predict` API plus low-latency TCP socket bridge for MQL5 Expert Advisors.

---

## 📦 Project Layout
```
natron/
  config/              # YAML and loaders
  data/                # Feature engineering, labeling, sequencing, Lightning DataModule
  models/              # Transformer encoder + multitask heads
  training/            # Losses, metrics, pretraining & supervised Lightning modules
  rl/                  # PPO trainer & trading environment
  serving/             # Inference service, Flask API, socket bridge
  scripts/             # CLI entrypoints for training & serving
  utils/               # Logging and reproducibility helpers
requirements.txt
```

---

## ⚙️ Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required GPU libraries:
- CUDA 11.8+ with compatible PyTorch 2.x build
- Optional: `stable-baselines3`, `gymnasium` for RL phase

---

## 📂 Data Expectations
- **Input CSV**: `data_export.csv` (path configurable in `natron/config/defaults.yaml`)
- Columns: `time`, `open`, `high`, `low`, `close`, `volume`
- Each row: one candle (M15/H1 recommended)
- Sequence length: `96` consecutive candles per sample (configurable)

> The pipeline prints **mandatory label distribution diagnostics** after label generation to ensure balanced classes.

---

## 🛠️ Feature & Label Generation

The `FeatureEngine` synthesizes ~100 indicators spanning:
- Moving averages, slope, ratios, crossovers
- Momentum (RSI/ROC/CCI/Stochastic/MACD)
- Volatility (ATR/Bollinger/Keltner/StdDev)
- Volume analytics (OBV/VWAP/MFI ratios)
- Price patterns (gaps, shadows, doji, engulfing)
- Returns & drawdowns
- Trend strength (ADX/DI/Aroon)
- Statistical moments & Hurst exponent
- Support/Resistance distances
- Smart Money Concepts proxies (swing H/L, BOS, CHoCH)
- Market profile (POC, VAH/VAL, entropy)

`LabelGeneratorV2` implements the bias-reduced institutional logic described in the spec, including adaptive class balancing and stochastic jitter, with balanced Buy/Sell ratios (~0.3–0.4).

---

## 🔄 Training Pipeline (CLI)

### 1. Pretraining → Supervised Fine-tuning
```bash
python -m natron.scripts.train \
  --config natron/config/defaults.yaml \
  --experiment-name natron_v2_experiment
```

Steps executed:
1. Feature extraction & label generation (cached to Parquet if configured)
2. **Phase 1**: `NatronPretrainModule` – masked token reconstruction + contrastive InfoNCE pretraining
3. **Phase 2**: `NatronSupervisedModule` – multitask heads (Buy/Sell sigmoid, Direction softmax(3), Regime softmax(6))
4. Optional evaluation on validation/test splits
5. Model checkpoint saved to `model/natron_v2.pt`

Trainer configuration is controlled via `defaults.yaml`:
- Optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`
- Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`
- Precision: `bf16` (auto-downgraded to fp32 on CPU)

### 2. Optional PPO Reinforcement Learning
Enable in YAML:
```yaml
rl:
  enabled: true
  total_steps: 200000
  learning_rate: 3.0e-5
  reward:
    turnover_penalty: 0.001
    drawdown_penalty: 0.005
```
When enabled, the training CLI:
1. Generates transformer embeddings for each sequence
2. Spins up `TradingEnv` with PPO (Stable-Baselines3)
3. Saves policy zip to `runs/<experiment>/rl/ppo_policy.zip`

Reward: `R = pnl − α·turnover − β·drawdown`

---

## 🧪 Natron DataModule
- Wraps feature, label, and sequence creation
- Caches features/labels via Parquet (configurable)
- Splits into train/val/test with reproducible seeds
- Exposes Lightning-ready `NatronDataset` returning `(sequence_tensor, target_dict)`

---

## 🌐 Inference & Serving

### Flask REST API
```bash
python -m natron.scripts.serve --mode api \
  --model-path model/natron_v2.pt \
  --config-path natron/config/defaults.yaml \
  --host 0.0.0.0 --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `POST /predict` with JSON body:
  ```json
  {
    "candles": [
      {"time": "2024-01-01T00:00:00Z", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "volume": 12345},
      ...
      (96 candles total)
    ]
  }
  ```
- Response:
  ```json
  {
    "buy_prob": 0.71,
    "sell_prob": 0.24,
    "direction_up": 0.69,
    "direction_probs": [0.18, 0.69, 0.13],
    "regime": "BULL_WEAK",
    "regime_probs": [0.05, 0.62, 0.12, 0.09, 0.06, 0.06],
    "confidence": 0.71
  }
  ```

### TCP Socket Bridge (MQL5 Ready)
```bash
python -m natron.scripts.serve --mode socket \
  --host 0.0.0.0 --socket-port 8765 \
  --model-path model/natron_v2.pt
```
- Accepts newline-delimited JSON `{ "candles": [...] }`
- Streams JSON responses per request
- Achieves sub-50 ms latency on GPU instances (depends on hardware/network)

### Combined Mode
```bash
python -m natron.scripts.serve --mode both
```
Runs Flask API and socket server concurrently.

---

## 🧩 Configuration
- Base config: `natron/config/defaults.yaml`
- Override runtime settings with `--config`, environment vars (`NATRON_MODEL_PATH`, `NATRON_CONFIG_PATH`), or YAML merges.
- Key knobs:
  - `data.sequence_length`, caching paths, batch sizes
  - Feature window lists
  - Label balancing targets & neutral buffer
  - Transformer depth, width, dropout, heads
  - Pretraining/supervised hyperparameters
  - RL toggles and penalties
  - Serving host/port/model path

---

## 📈 Monitoring & Logs
- Structured logging via `natron/utils/logging_utils.py`
- Default log streams to stdout; optionally configure file path
- Lightning integrates TensorBoard-compatible logs under `runs/<experiment>/`

---

## 🛡️ Operational Notes
- Ensure enough historical data (>96 + warmup) before inference to avoid cold-start indicators.
- Pretraining and supervised phases rely on GPU memory; adjust `batch_size` and `d_model` for your device.
- RL phase uses Stable-Baselines3 PPO with vectorized env; tune `total_steps`, `batch_size`, `policy_hidden_sizes`.
- Always inspect the printed label distribution summary to maintain balanced targets.

---

## 🤝 MetaTrader 5 Integration Blueprint
```
MQL5 Expert Advisor  ⇄  Python Socket Server  ⇄  Natron Inference Service (GPU)
```
1. EA collects latest 96 candles, sends JSON string over TCP (see example payload above).
2. Socket server returns Natron predictions.
3. EA executes trade logic using `{buy_prob, sell_prob, direction_probs, regime, confidence}`.

Latency target: < 50 ms (achievable on modern GPU VMs; ensure persistent connections and lightweight JSON).

---

## 🧪 Testing & Validation
- `python3 -m compileall natron` – quick syntax check (already integrated in dev workflow)
- Add unit/integration tests as you expand (pytest recommended)
- Validate inference on held-out data slices; confirm label balance and accuracy metrics from Lightning logs.

---

## 📄 License
MIT-style or proprietary—adapt to your organization. Add legal text as needed.

---

Natron’s philosophy: **“Learn the market’s grammar, then speak in trades.”**  
Happy building. 🧠📈