# 🧠 Natron Transformer V2 – Multi-Task Financial Trading Model

**End-to-End GPU Pipeline for Financial Market Prediction**

Natron Transformer is a state-of-the-art multi-task deep learning model designed for financial trading. It learns from sequences of 96 consecutive OHLCV candles and jointly predicts:
- **Buy/Sell signals** (binary classification)
- **Directional movement** (Up/Down/Neutral)
- **Market regime** (6 classes: Bull Strong/Weak, Bear Strong/Weak, Range, Volatile)

---

## 🎯 Features

- **~100 Technical Features**: Comprehensive feature engineering including MA, Momentum, Volatility, Volume, Price Patterns, Returns, Trend Strength, Statistical, Support/Resistance, Smart Money Concepts, and Market Profile
- **Bias-Reduced Labeling**: Institutional-quality labeling with automatic balance adjustment
- **Multi-Phase Training**: 
  - Phase 1: Unsupervised pretraining (masked token reconstruction + contrastive learning)
  - Phase 2: Supervised fine-tuning (multi-task learning)
  - Phase 3: Reinforcement Learning (optional, PPO/SAC)
- **Real-time API**: Flask REST API for predictions
- **MQL5 Integration**: Socket bridge for MetaTrader 5 Expert Advisors
- **GPU Optimized**: Full CUDA support with PyTorch 2.x

---

## 📋 Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- Ubuntu/Debian Linux (tested on GCP, Vertex AI)
- CUDA-capable GPU (recommended)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Note: If ta-lib installation fails, install system dependencies first:
# sudo apt-get install ta-lib-dev
# pip install TA-Lib
```

### 2. Prepare Data

Place your OHLCV data in CSV format at `data/data_export.csv` with columns:
- `time`: Timestamp
- `open`, `high`, `low`, `close`: Price data
- `volume`: Volume data

Example:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.0850,1.0860,1.0845,1.0855,1500
2024-01-01 00:15:00,1.0855,1.0865,1.0850,1.0860,1800
...
```

### 3. Configure

Edit `config.yaml` to adjust:
- Model architecture (d_model, nhead, num_layers, etc.)
- Training parameters (batch_size, learning_rate, epochs)
- Data paths and splits
- Labeling thresholds

### 4. Train Model

```bash
# Full training (pretraining + supervised)
python train.py --config config.yaml

# Skip pretraining (if already done)
python train.py --config config.yaml --skip-pretrain

# Pretraining only
python train.py --config config.yaml --pretrain-only
```

Training outputs:
- `checkpoints/pretrain/best_pretrain.pt`: Pretrained encoder
- `checkpoints/supervised/natron_v2.pt`: Final trained model
- `checkpoints/supervised/checkpoint.pt`: Full checkpoint with optimizer state

### 5. Run API Server

```bash
# Start Flask API server
python api_server.py --model checkpoints/supervised/natron_v2.pt --port 5000
```

Test the API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": 1234567890, "open": 1.0850, "high": 1.0860, "low": 1.0845, "close": 1.0855, "volume": 1500},
      ...
    ]
  }'
```

Response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction": "Up",
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

### 6. MQL5 Integration

```bash
# Start socket bridge
python mql5_bridge.py --model checkpoints/supervised/natron_v2.pt --port 8888
```

In MetaTrader 5:
1. Copy `mql5/NatronAI_EA.mq5` to `MQL5/Experts/`
2. Compile the EA in MetaEditor
3. Attach EA to chart
4. Configure server host/port (default: localhost:8888)
5. EA will automatically request predictions and execute trades

---

## 📁 Project Structure

```
natron_v2/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── train.py                   # Main training script
├── api_server.py              # Flask API server
├── mql5_bridge.py             # MQL5 socket bridge
├── test_components.py         # Component testing script
├── src/
│   ├── __init__.py
│   ├── feature_engine.py      # Feature generation (~100 features)
│   ├── label_generator.py     # Label generation (V2)
│   ├── sequence_creator.py    # Sequence construction
│   ├── model.py               # Transformer architecture
│   ├── training_utils.py      # Loss functions, optimizers
│   └── trainer.py             # Training classes
├── mql5/
│   └── NatronAI_EA.mq5        # MetaTrader 5 Expert Advisor
├── data/
│   └── data_export.csv        # Input OHLCV data
├── checkpoints/
│   ├── pretrain/              # Pretraining checkpoints
│   └── supervised/            # Supervised training checkpoints
└── models/                    # Final model outputs
```

---

## 🧩 Model Architecture

### Transformer Encoder
- **Input**: (batch_size, 96, 100) - 96 candles × 100 features
- **Embedding**: Linear projection to d_model (default: 256)
- **Positional Encoding**: Sinusoidal encoding
- **Encoder Layers**: 6 transformer encoder layers
  - Multi-head attention (8 heads)
  - Feedforward network (1024 dim)
  - GELU activation
  - Dropout (0.1)

### Multi-Task Heads
- **Buy Head**: Sigmoid → buy probability
- **Sell Head**: Sigmoid → sell probability
- **Direction Head**: Softmax(3) → Up/Down/Neutral
- **Regime Head**: Softmax(6) → 6 market regimes

---

## 🏷️ Label Generation (V2)

### Buy Signal (≥2 conditions)
- `close > MA20 > MA50`
- `RSI > 50` or RSI crossed up from <30
- `close > BB midband` and `MA20 slope > 0`
- `volume > 1.5 × rolling20`
- `position_in_range ≥ 0.7`
- `MACD_hist > 0` and rising

### Sell Signal (≥2 conditions)
- `close < MA20 < MA50`
- `RSI < 50` or RSI turned down from >70
- `close < BB midband` and `MA20 slope < 0`
- `volume > 1.5 × rolling20` and `position_in_range ≤ 0.3`
- `MACD_hist < 0` and falling
- `minus_DI > plus_DI`

### Direction (3-class)
- **Up (1)**: `close[i+3] > close[i] + neutral_buffer`
- **Down (0)**: `close[i+3] < close[i] - neutral_buffer`
- **Neutral (2)**: Otherwise

### Regime (6-class)
- **0**: BULL_STRONG (`trend > +2%`, `ADX > 25`)
- **1**: BULL_WEAK (`0 < trend ≤ +2%`, `ADX ≤ 25`)
- **2**: RANGE (default)
- **3**: BEAR_WEAK (`−2% ≤ trend < 0`, `ADX ≤ 25`)
- **4**: BEAR_STRONG (`trend < −2%`, `ADX > 25`)
- **5**: VOLATILE (`ATR > 90th percentile` or volume spike)

---

## ⚙️ Training Phases

### Phase 1: Pretraining
- **Objective**: Learn latent market structure
- **Methods**:
  - Masked token reconstruction (15% masking)
  - Contrastive learning (InfoNCE)
- **Loss**: `L = α × L_recon + β × L_contrastive`
- **Output**: Pretrained encoder weights

### Phase 2: Supervised Fine-Tuning
- **Objective**: Predict Buy/Sell/Direction/Regime
- **Loss**: Multi-task weighted combination
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)

### Phase 3: Reinforcement Learning (Optional)
- **Algorithm**: PPO or SAC
- **Reward**: `R = profit - α × turnover - β × drawdown`
- **Status**: Implemented but disabled by default

---

## 📊 Feature Groups

| Group | Count | Description |
|-------|-------|-------------|
| Moving Average | 13 | MA, EMA, slope, crossovers, ratios |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger, Keltner, StdDev |
| Volume | 9 | OBV, VWAP, MFI, Volume ratios |
| Price Pattern | 8 | Doji, Gaps, Shadows, Body%, Position |
| Returns | 8 | Log return, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to High/Low 20–50 |
| Smart Money Concepts | 6 | Swing HL, BOS, CHoCH |
| Market Profile | 10 | POC, VAH, VAL, Entropy |
| **Total** | **~100** | |

---

## 🔧 Configuration

Key parameters in `config.yaml`:

```yaml
data:
  sequence_length: 96        # Candles per sample
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  d_model: 256              # Model dimension
  nhead: 8                   # Attention heads
  num_layers: 6              # Encoder layers
  dim_feedforward: 1024      # FFN dimension

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs_pretrain: 50
  num_epochs_supervised: 100
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `sequence_length` (minimum: 32)
- Use gradient accumulation

### Label Imbalance
- Enable `balance_labels: true` in config.yaml
- Adjust `buy_threshold` and `sell_threshold`
- Check label statistics output during data preparation

### Model Not Converging
- Increase pretraining epochs
- Adjust learning rate
- Check data quality and feature normalization

### MQL5 Connection Issues
- Verify Python socket server is running
- Check firewall settings
- Ensure correct host/port in EA settings

---

## 📈 Performance Tips

1. **GPU Memory**: Use batch size 16-32 for GPUs with 8GB+ VRAM
2. **Data Quality**: Ensure clean OHLCV data with no gaps
3. **Feature Engineering**: All features are computed automatically
4. **Label Balance**: Monitor label distribution output
5. **Early Stopping**: Implemented via ReduceLROnPlateau scheduler

---

## 🔐 Security Notes

- **API Server**: Do not expose to public internet without authentication
- **MQL5 Bridge**: Use localhost or secure VPN for production
- **Model Files**: Keep trained models secure (proprietary IP)

---

## 📝 License

Proprietary - All rights reserved

---

## 🤝 Support

For issues, questions, or contributions:
1. Check `config.yaml` settings
2. Review training logs
3. Verify data format matches requirements
4. Test API endpoints independently

---

## 🎯 Future Enhancements

- [ ] Phase 3 RL implementation (PPO/SAC)
- [ ] Online learning / model updating
- [ ] Multi-timeframe support
- [ ] Portfolio optimization
- [ ] Risk management module
- [ ] Backtesting framework
- [ ] Web dashboard

---

**"Natron doesn't just predict Buy/Sell — it learns the grammar of the market."**
