# 🧠 Natron Transformer V2 – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for Financial Trading with Transformer Architecture

## 📋 Overview

Natron is a multi-task Transformer model that learns multiple representations of market behavior from sequences of 96 consecutive OHLCV candles. It jointly predicts:

- **Buy/Sell Classification** (binary signals)
- **Directional Prediction** (Up/Down/Neutral)
- **Market Regime Classification** (6 classes: BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)

## 🏗️ Architecture

### Model Components

1. **Transformer Encoder**: Learns temporal dependencies and market structure
   - 6 layers, 8 attention heads, 256-dimensional embeddings
   - Positional encoding for sequence understanding

2. **Multi-Task Heads**:
   - `buy_head`: Sigmoid output for buy probability
   - `sell_head`: Sigmoid output for sell probability
   - `direction_head`: Softmax(3) for Up/Down/Neutral
   - `regime_head`: Softmax(6) for market regime classification

### Training Phases

1. **Phase 1 - Pretraining**: Unsupervised learning via masked token reconstruction + contrastive learning (InfoNCE)
2. **Phase 2 - Supervised Fine-Tuning**: Multi-task prediction with balanced labels
3. **Phase 3 - Reinforcement Learning** (Optional): PPO/SAC for profit optimization

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: For TA-Lib, you may need to install system dependencies first:
# Ubuntu/Debian: sudo apt-get install ta-lib
# Or use conda: conda install -c conda-forge ta-lib
```

### Data Preparation

Place your OHLCV data in `data/data_export.csv` with columns:
- `time`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

### Training

```bash
# Run full training pipeline (Phase 1 + Phase 2)
python main.py
```

The pipeline will:
1. Load and preprocess data
2. Extract ~100 technical features
3. Generate balanced labels (Buy/Sell, Direction, Regime)
4. Create sequences of 96 candles
5. Pretrain the encoder (Phase 1)
6. Fine-tune for multi-task prediction (Phase 2)
7. Save model to `model/natron_v2.pt`

### API Server

```bash
# Start Flask API server
python src/api.py --model_path model/natron_v2.pt --port 5000
```

**API Endpoint**: `POST /predict`

**Request**:
```json
{
  "ohlcv": [
    {"time": "2024-01-01 00:00", "open": 1.1000, "high": 1.1010, "low": 1.0995, "close": 1.1005, "volume": 1000},
    ...
  ]
}
```

**Response**:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.15,
  "direction_neutral": 0.16,
  "regime": "BULL_WEAK",
  "regime_probs": {
    "BULL_STRONG": 0.12,
    "BULL_WEAK": 0.45,
    "RANGE": 0.20,
    "BEAR_WEAK": 0.10,
    "BEAR_STRONG": 0.08,
    "VOLATILE": 0.05
  },
  "confidence": 0.82
}
```

### MQL5 Socket Server

```bash
# Start socket server for MQL5 integration
python src/socket_server.py --host 127.0.0.1 --port 8888
```

The server accepts JSON messages from MQL5 EA:
```json
{
  "action": "predict",
  "ohlcv": [...]
}
```

## 📊 Feature Engineering

The model uses ~100 technical features across 11 groups:

| Group | Count | Examples |
|-------|-------|----------|
| Moving Average | 13 | MA5, MA20, MA50, EMA12/26, slopes, crossovers |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger Bands, Keltner Channels, StdDev |
| Volume | 9 | OBV, VWAP, MFI, volume ratios |
| Price Pattern | 8 | Doji, gaps, shadows, body%, position in range |
| Returns | 8 | Log returns, intraday returns, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to High/Low (20/50 periods) |
| Smart Money Concepts | 6 | Swing HL, BOS, CHoCH, Order Blocks |
| Market Profile | 10 | POC, VAH, VAL, Entropy |

## 🏷️ Label Generation (V2)

### Buy/Sell Signals

**BUY** (if ≥2 conditions true):
- `close > MA20 > MA50`
- `RSI > 50` or just crossed up from <30
- `close > BB midband` and `MA20 slope > 0`
- `volume > 1.5 × rolling20`
- `position_in_range ≥ 0.7`
- `MACD_hist > 0` and rising

**SELL** (if ≥2 conditions true):
- `close < MA20 < MA50`
- `RSI < 50` or just turned down from >70
- `close < BB midband` and `MA20 slope < 0`
- `volume > 1.5 × rolling20` and `position_in_range ≤ 0.3`
- `MACD_hist < 0` and falling
- `minus_DI > plus_DI`

### Direction Labels (3-class)

- **Up (1)**: `close[i+3] > close[i] + neutral_buffer`
- **Down (0)**: `close[i+3] < close[i] - neutral_buffer`
- **Neutral (2)**: Otherwise

### Regime Classification (6 classes)

| ID | Regime | Condition |
|----|--------|-----------|
| 0 | BULL_STRONG | trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < trend ≤ +2%, ADX ≤ 25 |
| 2 | RANGE | Lateral market (default) |
| 3 | BEAR_WEAK | -2% ≤ trend < 0, ADX ≤ 25 |
| 4 | BEAR_STRONG | trend < -2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile or volume spike |

**Label Balancing**: Automatically downsamples over-represented classes to ensure balanced distributions (~0.3-0.4 for Buy/Sell).

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- Model architecture (d_model, nhead, num_layers, etc.)
- Training parameters (batch_size, learning_rate, epochs)
- Labeling thresholds
- Data splits
- API/socket server settings

## 📁 Project Structure

```
natron_v2/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   └── data_export.csv       # Input OHLCV data
├── src/
│   ├── feature_engine.py    # Feature extraction (~100 features)
│   ├── label_generator.py   # Label generation (V2)
│   ├── sequence_creator.py  # Sequence construction
│   ├── model.py             # Transformer architecture
│   ├── pretrain.py          # Phase 1: Pretraining
│   ├── train.py             # Phase 2: Supervised training
│   ├── api.py               # Flask API server
│   └── socket_server.py     # MQL5 socket server
├── model/
│   └── natron_v2.pt         # Trained model (generated)
├── logs/                     # Training logs
├── main.py                   # Main training pipeline
└── requirements.txt          # Python dependencies
```

## 🔧 MQL5 Integration

See `mql5/NatronEA.mq5` for example MQL5 Expert Advisor code that connects to the socket server.

**Key Features**:
- Real-time tick/candle feed
- Low-latency predictions (<50ms target)
- Automatic reconnection on disconnect
- JSON message protocol

## 🎯 Performance Targets

- **Latency**: <50ms per prediction (GPU)
- **Accuracy**: Balanced across all tasks
- **Label Balance**: Buy/Sell ratio ~0.3-0.4
- **Regime Distribution**: Balanced across 6 classes

## 📝 Notes

- **GPU Recommended**: Training requires CUDA-capable GPU for reasonable training times
- **Data Requirements**: Minimum 1000+ candles recommended for stable training
- **Sequence Length**: Fixed at 96 candles (configurable in config.yaml)
- **Feature Scaling**: Features are standardized using StandardScaler

## 🐛 Troubleshooting

**Issue**: `TA-Lib not found`
- Solution: Install TA-Lib system library or use conda

**Issue**: `CUDA out of memory`
- Solution: Reduce `batch_size` in config.yaml

**Issue**: `Insufficient data`
- Solution: Ensure at least 1000+ candles in data_export.csv

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with PyTorch, Transformers, and modern deep learning best practices for financial time series.

---

**"Natron doesn't just predict Buy/Sell — it learns the grammar of the market."**