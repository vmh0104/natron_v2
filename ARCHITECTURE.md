# Natron Transformer Architecture Documentation

## 🏗️ System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     NATRON TRANSFORMER V2                        │
│                 Multi-Task Financial Trading Model                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Raw OHLCV   │ ───> │   Feature    │ ───> │    Label     │
│  (96 candles)│      │   Engine     │      │  Generator   │
│              │      │ (~100 feat.) │      │     V2       │
└──────────────┘      └──────────────┘      └──────────────┘
                             │                      │
                             └──────────┬───────────┘
                                        │
                             ┌──────────▼──────────┐
                             │  Sequence Dataset   │
                             │   (96, 100) → 4     │
                             └──────────┬──────────┘
                                        │
┌───────────────────────────────────────┴────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Pretraining (Unsupervised)                           │
│  ├─ Masked Token Reconstruction                                │
│  ├─ Contrastive Learning (InfoNCE)                             │
│  └─ Output: Pretrained Encoder                                 │
│                                                                 │
│  Phase 2: Supervised Fine-Tuning                               │
│  ├─ Multi-Task Learning                                        │
│  ├─ Buy/Sell + Direction + Regime                              │
│  └─ Output: Trained Model                                      │
│                                                                 │
│  Phase 3: Reinforcement Learning (Optional)                    │
│  ├─ PPO/SAC Algorithm                                          │
│  ├─ Custom Reward: Profit - Turnover - Drawdown               │
│  └─ Output: RL-Optimized Model                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Trained Model  │
                    │  natron_v2.pt   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Flask API     │
                    │   (REST)        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
      ┌───────▼──────┐  ┌───▼────┐  ┌─────▼──────┐
      │  HTTP Client │  │  MQL5  │  │  Custom    │
      │  (Python)    │  │   EA   │  │  App       │
      └──────────────┘  └────────┘  └────────────┘
```

---

## 🧩 Component Details

### 1. Feature Engine (`feature_engine.py`)

**Purpose:** Extract comprehensive technical indicators from raw OHLCV data.

**Categories (100 features total):**

1. **Moving Averages (13)**
   - SMA: 5, 10, 20, 50
   - EMA: 5, 20, 50
   - Slopes, ratios, crossovers

2. **Momentum Indicators (13)**
   - RSI: 7, 14, 21
   - ROC: 5, 10
   - CCI, Stochastic, MACD family

3. **Volatility (15)**
   - ATR: 14, 20
   - Bollinger Bands (20)
   - Keltner Channels
   - Historical volatility

4. **Volume (9)**
   - OBV, VWAP, MFI
   - Volume ratios and trends

5. **Price Patterns (8)**
   - Candle body, shadows
   - Doji, gaps
   - Position in range

6. **Returns (8)**
   - Log returns (1, 5, 10)
   - Intraday, cumulative

7. **Trend Strength (6)**
   - ADX, +DI, -DI
   - Aroon Up/Down

8. **Statistical (6)**
   - Skewness, Kurtosis
   - Z-score, Hurst exponent

9. **Support/Resistance (4)**
   - Distance to high/low (20, 50)

10. **Smart Money Concepts (6)**
    - Swing high/low
    - Break of Structure (BOS)
    - Change of Character (CHoCH)

11. **Market Profile (10)**
    - POC, VAH, VAL
    - Volume entropy

**Output:** DataFrame (N, 100)

---

### 2. Label Generator V2 (`label_generator.py`)

**Purpose:** Generate balanced, bias-reduced labels for multi-task learning.

#### Label Types:

**A. Buy Signal (Binary)**
- Threshold: ≥2 of 6 conditions
- Conditions:
  1. Price above MA trend
  2. RSI momentum confirmation
  3. Bollinger position + slope
  4. Volume spike
  5. High position in range
  6. MACD histogram positive & rising

**B. Sell Signal (Binary)**
- Threshold: ≥2 of 6 conditions
- Inverse logic of buy conditions
- Focus on bearish confirmations

**C. Direction (3-class)**
- 0: DOWN (future < current - buffer)
- 1: UP (future > current + buffer)
- 2: NEUTRAL (within buffer zone)
- Buffer: 0.1% by default

**D. Regime (6-class)**
| ID | Name | Condition |
|----|------|-----------|
| 0 | BULL_STRONG | Trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < Trend ≤ +2% |
| 2 | RANGE | Lateral (default) |
| 3 | BEAR_WEAK | -2% ≤ Trend < 0 |
| 4 | BEAR_STRONG | Trend < -2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile |

#### Adaptive Balancing

- Monitors class distribution
- Downsamples over-represented classes
- Adds stochastic perturbation
- Target: balanced ratios (~0.3-0.4 for binary)

**Output:** DataFrame (N, 4) → [buy, sell, direction, regime]

---

### 3. Transformer Model (`model.py`)

#### NatronTransformer Architecture

```python
NatronTransformer(
    num_features=100,
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_length=96
)
```

**Components:**

1. **Input Projection**
   - Linear: (num_features) → (d_model)
   - Projects raw features to model dimension

2. **Positional Encoding**
   - Sinusoidal encoding
   - Adds temporal information

3. **[CLS] Token**
   - Learnable token prepended to sequence
   - Aggregates sequence information

4. **Transformer Encoder**
   - 6 layers
   - 8 attention heads per layer
   - Layer normalization (pre-norm)
   - GELU activation

5. **Multi-Task Heads**

   a. **Buy Head**
      - FC(256 → 512 → 128 → 1)
      - Sigmoid activation
      - Outputs: P(Buy) ∈ [0, 1]

   b. **Sell Head**
      - FC(256 → 512 → 128 → 1)
      - Sigmoid activation
      - Outputs: P(Sell) ∈ [0, 1]

   c. **Direction Head**
      - FC(256 → 512 → 128 → 3)
      - Softmax activation
      - Outputs: [P(Down), P(Up), P(Neutral)]

   d. **Regime Head**
      - FC(256 → 512 → 128 → 6)
      - Softmax activation
      - Outputs: [P(regime_0), ..., P(regime_5)]

**Total Parameters:** ~5-10M (depends on config)

---

### 4. Training Pipeline (`train.py`)

#### Phase 1: Pretraining (Unsupervised)

**Objective:** Learn latent market structure without labels.

**Methods:**
1. **Masked Token Reconstruction**
   - Randomly mask 15% of sequence tokens
   - Reconstruct original values
   - Loss: MSE between reconstructed and original

2. **Contrastive Learning (InfoNCE)**
   - Project [CLS] embeddings to lower dimension
   - Compute similarity matrix
   - Maximize similarity of related sequences
   - Loss: Cross-entropy with temperature scaling

**Hyperparameters:**
- Epochs: 50
- Batch size: 128
- Learning rate: 1e-4
- Mask ratio: 0.15
- Temperature: 0.07

**Output:** Pretrained encoder weights

---

#### Phase 2: Supervised Fine-Tuning

**Objective:** Learn to predict Buy/Sell/Direction/Regime.

**Loss Function:**
```
L_total = w_buy·BCE(y_buy, ŷ_buy) +
          w_sell·BCE(y_sell, ŷ_sell) +
          w_dir·CE(y_dir, ŷ_dir) +
          w_reg·CE(y_reg, ŷ_reg)
```

**Default Weights:**
- w_buy = 1.0
- w_sell = 1.0
- w_dir = 1.5 (emphasize direction)
- w_reg = 1.2

**Hyperparameters:**
- Epochs: 100
- Batch size: 64
- Learning rate: 1e-4
- Gradient clipping: 1.0
- Early stopping: 15 epochs patience

**Metrics:**
- Buy/Sell accuracy
- Direction accuracy (3-class)
- Regime accuracy (6-class)
- Per-class F1 scores

**Output:** Trained multi-task model

---

#### Phase 3: Reinforcement Learning (Optional)

**Status:** Placeholder for future development

**Algorithm:** PPO (Proximal Policy Optimization) or SAC

**Environment:** Custom trading gym

**Reward Function:**
```
R = profit - α·turnover - β·drawdown

where:
  profit = realized P&L
  turnover = frequency penalty
  drawdown = max equity drop
  α = 0.1 (default)
  β = 0.2 (default)
```

**Goal:** Maximize cumulative reward over trading episodes

---

### 5. Inference API (`api.py`)

**Framework:** Flask + CORS

**Endpoints:**

#### `GET /health`
Health check.

**Response:**
```json
{"status": "healthy", "model_loaded": true}
```

---

#### `GET /info`
Model information.

**Response:**
```json
{
  "model": "Natron Transformer V2",
  "sequence_length": 96,
  "features": 100,
  "d_model": 256,
  "layers": 6,
  "heads": 8,
  "device": "cuda"
}
```

---

#### `POST /predict`
Main prediction endpoint.

**Request:**
```json
{
  "candles": [
    {
      "time": "2024-01-01 00:00:00",
      "open": 100.0,
      "high": 101.0,
      "low": 99.0,
      "close": 100.5,
      "volume": 1000
    },
    ... // 96 total
  ]
}
```

**Response:**
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction": "UP",
  "direction_probs": {
    "DOWN": 0.15,
    "UP": 0.69,
    "NEUTRAL": 0.16
  },
  "regime": "BULL_WEAK",
  "regime_probs": {
    "BULL_STRONG": 0.12,
    "BULL_WEAK": 0.45,
    "RANGE": 0.18,
    "BEAR_WEAK": 0.10,
    "BEAR_STRONG": 0.05,
    "VOLATILE": 0.10
  },
  "confidence": 0.82
}
```

**Latency:** <50ms on GPU, <200ms on CPU

---

## 🔬 Technical Specifications

### Data Flow

1. **Input:** 96 OHLCV candles
2. **Feature Extraction:** 100 technical indicators
3. **Normalization:** StandardScaler (fitted on training data)
4. **Sequence:** (96, 100) tensor
5. **Forward Pass:** Through Transformer
6. **Output:** 4 prediction tasks
7. **Post-processing:** Probabilities + confidence score

### Memory Requirements

- **Training:**
  - Minimum: 8GB RAM, 4GB VRAM
  - Recommended: 16GB RAM, 8GB VRAM

- **Inference:**
  - Minimum: 4GB RAM, 2GB VRAM
  - Recommended: 8GB RAM, 4GB VRAM

### Performance

- **Training Time:**
  - Phase 1 (50 epochs): ~2-4 hours (GPU)
  - Phase 2 (100 epochs): ~4-8 hours (GPU)
  
- **Inference:**
  - Single prediction: <50ms (GPU)
  - Batch (32): ~100-200ms (GPU)

---

## 🧪 Testing & Validation

### Unit Tests

Each module has standalone test:
```bash
python src/feature_engine.py    # Test feature extraction
python src/label_generator.py   # Test label generation
python src/dataset.py            # Test dataset creation
python src/model.py              # Test model forward pass
```

### Integration Test

```bash
python main.py --mode test       # Full inference test
```

### Validation Metrics

- **Training:** Loss curves, gradient norms
- **Validation:** Multi-task accuracies
- **Test:** Out-of-sample performance

---

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python3", "main.py", "--mode", "api"]
```

### Systemd Service

```ini
[Unit]
Description=Natron Transformer API
After=network.target

[Service]
Type=simple
User=natron
WorkingDirectory=/opt/natron
ExecStart=/usr/bin/python3 main.py --mode api
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📊 Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
```

Metrics tracked:
- Training/validation loss
- Per-task accuracy
- Learning rate
- Gradient norms

### API Logging

- Request count
- Response time
- Error rate
- Prediction distribution

---

## 🔐 Security Considerations

1. **API Authentication:** Add token-based auth for production
2. **Rate Limiting:** Implement request throttling
3. **Input Validation:** Strict schema validation
4. **Model Integrity:** Checksum verification
5. **SSL/TLS:** HTTPS for external access

---

## 📚 References

### Academic Papers

1. Vaswani et al. (2017) - *Attention Is All You Need*
2. Devlin et al. (2018) - *BERT: Pre-training of Deep Bidirectional Transformers*
3. Chen et al. (2020) - *A Simple Framework for Contrastive Learning*
4. Schulman et al. (2017) - *Proximal Policy Optimization*

### Libraries

- PyTorch 2.x
- Transformers
- Pandas, NumPy
- Scikit-learn
- Flask

---

**End of Architecture Documentation**

*For usage instructions, see README.md*
