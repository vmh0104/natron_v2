# 🧠 Natron Transformer V2 - Multi-Task Financial Trading Model

End-to-End GPU Pipeline for Financial Trading with Transformer Architecture

## 🎯 Overview

Natron Transformer V2 is a sophisticated multi-task deep learning model designed for financial trading. It learns multiple representations of market behavior from sequences of 96 consecutive OHLCV candles and outputs actionable predictions:

- **Buy/Sell Classification** - Binary signals for entry/exit
- **Directional Prediction** - 3-class (Up/Down/Neutral)
- **Market Regime Classification** - 6-class regime detection

## 🏗️ Architecture

### Model Components

1. **Feature Engine** (~100 technical features)
   - Moving Averages (13)
   - Momentum indicators (13)
   - Volatility measures (15)
   - Volume analysis (9)
   - Price patterns (8)
   - Returns (8)
   - Trend strength (6)
   - Statistical features (6)
   - Support/Resistance (4)
   - Smart Money Concepts (6)
   - Market Profile (10)

2. **Transformer Encoder**
   - 6 layers, 8 attention heads
   - d_model=256, dim_feedforward=1024
   - Positional encoding + global pooling

3. **Multi-Task Heads**
   - Buy head (sigmoid)
   - Sell head (sigmoid)
   - Direction head (softmax, 3 classes)
   - Regime head (softmax, 6 classes)

## 📋 Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- Ubuntu/Debian Linux (tested on GCP/Vertex AI)

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your OHLCV data in `data/data_export.csv` with columns:
- `time`, `open`, `high`, `low`, `close`, `volume`

### 3. Train Model

```bash
# Full training (pretraining + supervised)
python train.py --config config.yaml

# Skip pretraining (faster, for testing)
python train.py --config config.yaml --skip-pretraining
```

### 4. Run API Server

```bash
python api_server.py --model-path model/natron_v2.pt --port 5000
```

### 5. Run MQL5 Bridge Server

```bash
python mql5_bridge.py --model-path model/natron_v2.pt --port 8888
```

## 📊 Training Pipeline

### Phase 1: Pretraining (Unsupervised)
- **Masked Token Reconstruction** - Learn temporal dependencies
- **Contrastive Learning (InfoNCE)** - Learn market structure
- Output: Pretrained encoder weights

### Phase 2: Supervised Fine-Tuning
- Multi-task learning on Buy/Sell/Direction/Regime labels
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau
- Early stopping based on validation loss

### Phase 3: Reinforcement Learning (Optional)
- PPO/SAC algorithm
- Reward: profit - α×turnover - β×drawdown
- *Not implemented in this version*

## 🏷️ Label Generation

### Buy/Sell Labels (Bias-Reduced)
- **BUY**: ≥2 conditions from 6 institutional signals
- **SELL**: ≥2 conditions from 6 institutional signals
- Automatic class balancing to prevent bias

### Direction Labels (3-class)
- **Up**: Price increases >0.1% in 3 periods
- **Down**: Price decreases >0.1% in 3 periods
- **Neutral**: Otherwise

### Regime Labels (6-class)
- **BULL_STRONG**: trend >+2%, ADX >25
- **BULL_WEAK**: 0<trend≤+2%, ADX≤25
- **RANGE**: Lateral market (default)
- **BEAR_WEAK**: -2%≤trend<0, ADX≤25
- **BEAR_STRONG**: trend<-2%, ADX>25
- **VOLATILE**: ATR >90th percentile or volume spike

## 🔌 API Usage

### Flask API Endpoint

**POST** `/predict`

Request:
```json
{
  "candles": [
    {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
    ...
  ]
}
```

Response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction": "Up",
  "direction_probs": [0.15, 0.69, 0.16],
  "regime": "BULL_WEAK",
  "regime_probs": [0.1, 0.4, 0.2, 0.15, 0.1, 0.05],
  "confidence": 0.82
}
```

### MQL5 Socket Bridge

The MQL5 bridge server accepts JSON messages:

**Candle Update**:
```json
{
  "type": "candles",
  "candles": [...]
}
```

**Direct Prediction**:
```json
{
  "type": "predict",
  "candles": [...]
}
```

## 📁 Project Structure

```
.
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── train.py                 # Main training script
├── api_server.py            # Flask API server
├── mql5_bridge.py          # MQL5 socket bridge
├── src/
│   ├── __init__.py
│   ├── feature_engine.py    # Feature extraction (~100 features)
│   ├── label_generator.py  # Label generation (V2)
│   ├── sequence_creator.py  # Sequence construction
│   ├── model.py            # Transformer architecture
│   ├── pretraining.py      # Phase 1 training
│   └── supervised_training.py  # Phase 2 training
├── data/
│   └── data_export.csv      # Input OHLCV data
├── model/
│   ├── natron_v2.pt        # Trained model
│   └── feature_scaler.pkl  # Feature scaler
└── logs/                    # Training logs
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data**: sequence length, train/val/test splits
- **Model**: architecture parameters (d_model, layers, etc.)
- **Training**: batch size, learning rate, epochs
- **API**: host, port, model path
- **MQL5**: socket host/port

## 🔧 Development

### Running Tests

```bash
# Test feature extraction
python -c "from src.feature_engine import FeatureEngine; import pandas as pd; df = pd.read_csv('data/data_export.csv'); fe = FeatureEngine(); features = fe.extract_all_features(df); print(features.shape)"

# Test label generation
python -c "from src.label_generator import LabelGeneratorV2; ..."
```

### Monitoring Training

Training progress is printed to console. For production, consider:
- TensorBoard logging
- Weights & Biases integration
- Custom logging handlers

## 🚨 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `d_model` or `num_layers`
- Use gradient accumulation

### Missing Features
- Ensure data has sufficient history (at least 96 candles)
- Check feature extraction dependencies

### Model Not Loading
- Verify model path exists
- Check PyTorch version compatibility
- Ensure feature_scaler.pkl exists

## 📝 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

This is a complete end-to-end system. For extensions:
1. Add new features in `feature_engine.py`
2. Extend label generation in `label_generator.py`
3. Modify model architecture in `model.py`
4. Add new training phases as needed

## 🎓 Citation

If you use this code, please cite:

```
Natron Transformer V2 - Multi-Task Financial Trading Model
End-to-End GPU Pipeline for Financial Trading
```

---

**Built with ❤️ for algorithmic trading**