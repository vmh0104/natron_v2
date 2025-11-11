# 🧠 Natron Transformer – Multi-Task Financial Trading Model

**End-to-End GPU-Accelerated Deep Learning Pipeline for Financial Trading**

---

## 📋 Overview

Natron Transformer is a state-of-the-art multi-task Transformer model designed for financial trading. It learns market structure through unsupervised pretraining and provides actionable predictions for:

- **Buy/Sell Classification** (Binary signals)
- **Direction Prediction** (Up/Down/Neutral)
- **Market Regime Classification** (6 regimes: Bull Strong/Weak, Range, Bear Weak/Strong, Volatile)

### 🎯 Key Features

✅ **~100 Technical Indicators** automatically extracted  
✅ **Bias-Reduced Labeling** with institutional logic  
✅ **Three-Phase Training**: Pretraining → Supervised → RL (optional)  
✅ **GPU-Optimized** for PyTorch 2.x + CUDA  
✅ **REST API** for real-time inference (<50ms latency)  
✅ **MQL5 Integration Ready** for MetaTrader 5  

---

## 🏗️ Architecture

```
Input (96 OHLCV candles)
    ↓
Feature Engine (~100 indicators)
    ↓
Transformer Encoder (6 layers, 8 heads, d_model=256)
    ↓
Multi-Task Heads:
    ├─ Buy Head (Sigmoid)
    ├─ Sell Head (Sigmoid)
    ├─ Direction Head (Softmax 3-class)
    └─ Regime Head (Softmax 6-class)
```

### 📊 Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Moving Averages | 13 | MA, EMA, slopes, crossovers |
| Momentum | 13 | RSI, MACD, CCI, Stochastic |
| Volatility | 15 | ATR, Bollinger Bands, Keltner |
| Volume | 9 | OBV, VWAP, MFI |
| Price Patterns | 8 | Doji, gaps, shadows |
| Returns | 8 | Log returns, cumulative |
| Trend Strength | 6 | ADX, Aroon, DI |
| Statistical | 6 | Skewness, Kurtosis, Hurst |
| Support/Resistance | 4 | Distance to highs/lows |
| Smart Money | 6 | Swing HL, BOS, CHoCH |
| Market Profile | 10 | POC, VAH, VAL, entropy |

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone repository
git clone <your-repo>
cd natron-transformer

# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### 2️⃣ Prepare Data

Place your OHLCV data in `data_export.csv` with columns:
```
time, open, high, low, close, volume
```

Or let the system generate synthetic data for testing.

### 3️⃣ Train Model

**Full Pipeline (Recommended):**
```bash
python main.py --mode train
```

**Individual Phases:**
```bash
# Phase 1: Pretraining only
python main.py --mode pretrain

# Phase 2: Supervised training only
python main.py --mode supervised
```

### 4️⃣ Start API Server

```bash
python main.py --mode api
```

Server runs at `http://localhost:5000`

### 5️⃣ Test Inference

```bash
python main.py --mode test
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model architecture
model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 6
  dim_feedforward: 1024

# Training parameters
pretrain:
  epochs: 50
  batch_size: 128
  learning_rate: 0.0001

supervised:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0001
  early_stopping_patience: 15
```

---

## 📡 API Usage

### Endpoints

#### `POST /predict`

Send 96 OHLCV candles as JSON:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "2024-01-01 00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
      ...
    ]
  }'
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

#### `GET /health`

Health check endpoint.

#### `GET /info`

Returns model architecture information.

---

## 🎓 Training Pipeline Details

### Phase 1: Unsupervised Pretraining

**Objective:** Learn latent market structure

**Methods:**
- Masked Token Reconstruction (15% masking)
- Contrastive Learning (InfoNCE)

**Duration:** ~50 epochs

**Output:** Pretrained encoder weights

### Phase 2: Supervised Fine-Tuning

**Objective:** Predict Buy/Sell/Direction/Regime

**Loss Function:**
```
L_total = w_buy·L_buy + w_sell·L_sell + w_dir·L_direction + w_reg·L_regime
```

**Metrics:**
- Buy/Sell Accuracy
- Direction Accuracy (3-class)
- Regime Accuracy (6-class)

**Duration:** ~100 epochs with early stopping

### Phase 3: Reinforcement Learning (Optional)

**Objective:** Optimize trading performance

**Algorithm:** PPO or SAC

**Reward:**
```
R = profit - α·turnover - β·drawdown
```

**Status:** Placeholder (requires trading environment)

---

## 📊 Label Generation V2

### Buy Signal (≥2 conditions)
1. close > MA20 > MA50
2. RSI > 50 or crossed up from <30
3. close > BB mid and MA20 slope > 0
4. volume > 1.5× rolling20
5. position_in_range ≥ 0.7
6. MACD_hist > 0 and rising

### Sell Signal (≥2 conditions)
1. close < MA20 < MA50
2. RSI < 50 or turned down from >70
3. close < BB mid and MA20 slope < 0
4. volume > 1.5× rolling20 and position ≤ 0.3
5. MACD_hist < 0 and falling
6. minus_DI > plus_DI

### Direction (3-class)
- **UP**: future_close > current + buffer
- **DOWN**: future_close < current - buffer
- **NEUTRAL**: otherwise

### Regime Classification (6 states)
| ID | Regime | Condition |
|----|--------|-----------|
| 0 | BULL_STRONG | trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < trend ≤ +2% |
| 2 | RANGE | Lateral market (default) |
| 3 | BEAR_WEAK | -2% ≤ trend < 0 |
| 4 | BEAR_STRONG | trend < -2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile |

---

## 🔌 MQL5 Integration

### Architecture

```
MetaTrader 5 (MQL5 EA)
    ↓ Socket
Python Socket Server
    ↓ REST API
Natron Transformer (GPU)
```

### Example Integration (Coming Soon)

See `examples/mql5_integration.mq5` for MetaTrader 5 Expert Advisor template.

---

## 📁 Project Structure

```
natron-transformer/
├── main.py                 # Main orchestration script
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── README.md              # This file
├── src/
│   ├── feature_engine.py  # Feature extraction (~100 indicators)
│   ├── label_generator.py # Label generation V2
│   ├── dataset.py         # PyTorch datasets
│   ├── model.py           # Transformer architecture
│   ├── train.py           # Training loops (3 phases)
│   └── api.py             # Flask inference API
├── data/
│   └── data_export.csv    # Input OHLCV data
├── model/
│   ├── natron_v2.pt       # Trained model
│   └── scaler.pkl         # Feature scaler
└── logs/                  # TensorBoard logs
```

---

## 🧪 Testing Individual Modules

### Test Feature Engine
```bash
python src/feature_engine.py
```

### Test Label Generator
```bash
python src/label_generator.py
```

### Test Dataset
```bash
python src/dataset.py
```

### Test Model
```bash
python src/model.py
```

---

## 📈 Monitoring Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

Navigate to `http://localhost:6006`

---

## 🔬 Research & Development

### Natron Philosophy

> "Natron doesn't just predict Buy/Sell — it learns the grammar of the market."

**Three-Stage Learning:**
1. **Structure Understanding** (Pretrain) - Temporal dependencies
2. **Signal Recognition** (Supervised) - Trade setups
3. **Behavioral Adaptation** (RL) - Real-world optimization

---

## ⚙️ System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- CPU with AVX2 support

### Recommended
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- Ubuntu 20.04+ / Debian 11+

---

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.yaml`:
```yaml
supervised:
  batch_size: 32  # Reduce from 64
```

### Model Not Found
Ensure training completed successfully and check:
```bash
ls -lh model/natron_v2.pt
```

### API Connection Error
Verify server is running:
```bash
curl http://localhost:5000/health
```

---

## 📚 References

- Vaswani et al. (2017) - Attention Is All You Need
- Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
- Schulman et al. (2017) - Proximal Policy Optimization

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: [Your contact]

---

## 🎉 Acknowledgments

Built with:
- PyTorch 2.x
- Transformers
- Pandas, NumPy, Scikit-learn
- Flask
- TensorBoard

---

**Natron Transformer V2** - *Institutional-Grade AI for Financial Trading*

*"Where Deep Learning Meets Market Microstructure"*
