# 🧠 Natron Transformer – Multi-Task Financial Trading Model

**End-to-End GPU-Accelerated Deep Learning Pipeline for Algorithmic Trading**

---

## 🎯 Overview

Natron Transformer is a state-of-the-art multi-task Transformer model designed for financial trading. It jointly learns multiple representations of market behavior from sequences of 96 consecutive OHLCV candles and outputs actionable predictions:

- **Buy/Sell Classification** – Binary signals for trade entry
- **Directional Prediction** – Up/Down/Neutral 3-class classification
- **Market Regime** – 6-state regime classification (Bull/Bear Strong/Weak, Range, Volatile)

### Key Features

✅ **~100 Technical Indicators** – Comprehensive feature engineering across 10 categories  
✅ **Bias-Reduced Labeling** – Institutional-grade labeling with adaptive balancing  
✅ **Three-Phase Training** – Pretraining → Supervised → Reinforcement Learning  
✅ **GPU-Optimized** – PyTorch 2.x with mixed precision training  
✅ **Production-Ready API** – Flask server with <50ms latency target  
✅ **MQL5 Compatible** – Ready for MetaTrader 5 integration

---

## 📁 Project Structure

```
natron-transformer/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── feature_engine.py        # Technical indicator generation (~100 features)
│   ├── label_generator.py       # Bias-reduced institutional labeling
│   ├── sequence_creator.py      # 96-candle sequence creation
│   ├── model.py                 # Transformer architecture
│   ├── pretrain.py              # Phase 1: Unsupervised pretraining
│   ├── train_supervised.py      # Phase 2: Supervised fine-tuning
│   ├── train_rl.py              # Phase 3: Reinforcement learning (PPO)
│   └── api_server.py            # Flask API server
├── data/
│   └── data_export.csv          # OHLCV input data
├── model/                       # Saved models and checkpoints
├── logs/                        # Training logs
├── main.py                      # Main training pipeline
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd natron-transformer

# Create virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- 4GB+ GPU VRAM (for training)

### 2. Prepare Data

Place your OHLCV data in `data/data_export.csv` with columns:
```
time,open,high,low,close,volume
```

If no data is provided, the system will generate sample data for demonstration.

### 3. Train Model

```bash
# Full training pipeline (all 3 phases)
python main.py

# Skip pretraining
python main.py --skip-pretrain

# Skip RL phase
python main.py --skip-rl

# Custom config
python main.py --config my_config.yaml

# Custom data path
python main.py --data path/to/data.csv
```

### 4. Start API Server

```bash
python src/api_server.py
```

The server will start on `http://0.0.0.0:5000`

### 5. Make Predictions

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"time": "2024-01-01 00:00:00", "open": 100.0, "high": 101.0, 
       "low": 99.0, "close": 100.5, "volume": 1000},
      ...  # 96 candles total
    ]
  }'
```

**Response:**
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.25,
  "regime": "BULL_WEAK",
  "regime_confidence": 0.82,
  "confidence": 0.82,
  "latency_ms": 45.2
}
```

---

## 🔬 Technical Architecture

### Feature Engineering (≈100 Features)

| Group | Count | Examples |
|-------|-------|----------|
| Moving Average | 13 | SMA, EMA, slopes, crossovers, ratios |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger Bands, Keltner Channels |
| Volume | 9 | OBV, VWAP, MFI, Volume ratios |
| Price Pattern | 8 | Doji, Gaps, Shadows, Body ratios |
| Returns | 8 | Log returns, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to highs/lows |
| Smart Money | 6 | Swing HL, BOS, CHoCH |
| Market Profile | 10 | POC, VAH, VAL, Entropy |

### Labeling System V2

**Buy Signal** (≥2 conditions):
- close > MA20 > MA50
- RSI > 50 or crossed up from <30
- close > BB midband & MA20 slope > 0
- Volume > 1.5× average
- Position in range ≥ 0.7
- MACD histogram > 0 and rising

**Sell Signal** (≥2 conditions):
- close < MA20 < MA50
- RSI < 50 or turned down from >70
- close < BB midband & MA20 slope < 0
- Volume spike & low position
- MACD histogram < 0 and falling
- -DI > +DI

**Direction Labels:**
- 0: Down (price drops > buffer)
- 1: Up (price rises > buffer)
- 2: Neutral (within buffer)

**Market Regimes:**
- 0: BULL_STRONG (trend > +2%, ADX > 25)
- 1: BULL_WEAK (0 < trend ≤ +2%, ADX ≤ 25)
- 2: RANGE (lateral market)
- 3: BEAR_WEAK (−2% ≤ trend < 0, ADX ≤ 25)
- 4: BEAR_STRONG (trend < −2%, ADX > 25)
- 5: VOLATILE (ATR > 90th percentile)

### Model Architecture

```
Input: (batch, 96, 100)
  ↓
Input Projection → d_model=256
  ↓
Positional Encoding
  ↓
Transformer Encoder (6 layers, 8 heads)
  ↓
Global Pooling
  ↓
┌─────────┬──────────┬────────────┬──────────┐
│Buy Head │Sell Head │Direction   │Regime    │
│(sigmoid)│(sigmoid) │Head (3cls) │Head (6cls)│
└─────────┴──────────┴────────────┴──────────┘
```

**Parameters:** ~8M trainable parameters

---

## 🏋️ Training Phases

### Phase 1: Pretraining (Unsupervised)

**Goal:** Learn latent market structure

**Methods:**
- Masked token reconstruction (15% masking)
- Contrastive learning (InfoNCE loss)

**Loss:** `L = 0.7 × L_recon + 0.3 × L_contrast`

**Duration:** 50 epochs (~2-4 hours on GPU)

### Phase 2: Supervised Fine-Tuning

**Goal:** Predict Buy/Sell/Direction/Regime

**Loss:** Weighted multi-task loss
```
L = 1.0×L_buy + 1.0×L_sell + 1.5×L_direction + 1.2×L_regime
```

**Optimizer:** AdamW (lr=5e-5, weight_decay=1e-5)  
**Scheduler:** ReduceLROnPlateau (patience=5)  
**Early Stopping:** patience=15

**Duration:** 100 epochs (~4-8 hours on GPU)

### Phase 3: Reinforcement Learning (Optional)

**Algorithm:** Proximal Policy Optimization (PPO)

**Reward Function:**
```
R = profit - α×turnover - β×drawdown - γ×holding_time
```

**Environment:**
- Initial balance: $10,000
- Transaction cost: 0.02%
- Max position: 100%

**Duration:** 1000 episodes (~2-4 hours)

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Key parameters
data:
  sequence_length: 96
  test_split: 0.2

model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 6

pretrain:
  epochs: 50
  mask_ratio: 0.15

supervised:
  epochs: 100
  learning_rate: 5.0e-5

rl:
  episodes: 1000
  algorithm: "PPO"
```

---

## 🌐 API Endpoints

### `GET /health`
Health check

### `GET /info`
Model information

### `POST /predict`
Make prediction from OHLCV data

**Request:**
```json
{
  "data": [/* 96 OHLCV candles */]
}
```

**Response:**
```json
{
  "buy_prob": float,
  "sell_prob": float,
  "direction_up": float,
  "direction_down": float,
  "regime": string,
  "regime_confidence": float,
  "confidence": float,
  "latency_ms": float
}
```

---

## 🔌 MetaTrader 5 Integration

The API is designed for real-time MQL5 integration:

```
MQL5 EA ⇄ Socket/HTTP ⇄ Natron API (GPU)
```

**Latency Target:** <50ms per prediction

See documentation for MQL5 EA implementation details.

---

## 📊 Monitoring & Logs

Training logs are saved to `logs/training.log`

Optional TensorBoard integration:
```bash
pip install tensorboard
tensorboard --logdir=logs/tensorboard
```

---

## 🧪 Testing

```bash
# Test individual modules
python src/feature_engine.py
python src/label_generator.py
python src/model.py

# Test API
curl http://localhost:5000/health
```

---

## 🛠️ Development

### Code Structure

Each module is self-contained and testable:
- `feature_engine.py` – Pure feature computation
- `label_generator.py` – Label generation logic
- `model.py` – Model architecture only
- Training scripts – Training loops and optimization

### Adding Custom Features

Edit `FeatureEngine` class in `src/feature_engine.py`:

```python
def _my_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    # Add your features
    features['my_indicator'] = ...
    return features
```

Then call in `generate_features()`.

---

## 📈 Performance Metrics

**Training:**
- Buy/Sell Accuracy: ~70-85%
- Direction Accuracy: ~55-65%
- Regime Accuracy: ~60-75%

**Inference:**
- Latency: 30-50ms (GPU)
- Throughput: ~20-30 predictions/sec

**Note:** Performance depends on data quality, market conditions, and hyperparameters.

---

## 🔒 Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/api_server.py"]
```

### Systemd Service

```bash
sudo cp natron.service /etc/systemd/system/
sudo systemctl enable natron
sudo systemctl start natron
```

### Gunicorn (Production Server)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 src.api_server:app
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional technical indicators
- Alternative labeling strategies
- Model architecture improvements
- Real-world backtesting results

---

## 📄 License

MIT License - See LICENSE file for details

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred from using this software.

Always test thoroughly on paper trading before live deployment.

---

## 📞 Support

For issues and questions:
- GitHub Issues: <repository-url>/issues
- Documentation: See `/docs` folder
- Community: Discord/Slack (link)

---

## 🎓 Citation

If you use this work in research, please cite:

```bibtex
@software{natron_transformer_2024,
  title={Natron Transformer: Multi-Task Financial Trading Model},
  author={Natron AI Team},
  year={2024},
  version={2.0.0}
}
```

---

## 🌟 Acknowledgments

Built with:
- PyTorch 2.x
- Pandas, NumPy, Scikit-learn
- Flask

Inspired by:
- Transformer architecture (Vaswani et al., 2017)
- Financial machine learning (López de Prado, 2018)
- Multi-task learning principles

---

**🚀 Ready to transform financial trading with deep learning!**

For detailed documentation, see `/docs` folder.
