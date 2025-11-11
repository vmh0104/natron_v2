# 🧠 Natron Transformer V2 – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for Financial Trading AI

## 📋 Overview

Natron Transformer is a state-of-the-art multi-task Transformer model designed for financial trading. It learns multiple representations of market behavior and outputs actionable predictions:

- **Buy/Sell Classification**: Binary signals for entry/exit
- **Directional Prediction**: Up/Down/Neutral (3-class)
- **Market Regime Classification**: 6 regime states (Bull Strong/Weak, Bear Strong/Weak, Range, Volatile)

## 🏗️ Architecture

### Model Components

1. **Transformer Encoder**: Learns temporal dependencies from 96-candle sequences
2. **Multi-Task Heads**: Separate heads for each prediction task
3. **Feature Engine**: ~100 technical features automatically generated
4. **Label Generator V2**: Bias-reduced institutional labeling

### Training Phases

- **Phase 1: Pretraining** - Masked token reconstruction + contrastive learning (InfoNCE)
- **Phase 2: Supervised Fine-Tuning** - Multi-task learning with balanced labels
- **Phase 3: Reinforcement Learning** (Optional) - PPO/SAC for profit optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Ubuntu/Debian Linux (or compatible)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd natron_v2

# Install dependencies
pip install -r requirements.txt

# Note: If ta-lib installation fails, install system library first:
# sudo apt-get install ta-lib-dev
# pip install ta-lib
```

### Data Format

Your `data_export.csv` should have the following columns:
- `time`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

### Training

```bash
# Train the model (end-to-end pipeline)
python train.py

# The script will:
# 1. Load data from config.yaml → data.input_file
# 2. Generate ~100 technical features
# 3. Create balanced labels (Buy/Sell/Direction/Regime)
# 4. Build sequences of 96 candles
# 5. Pretrain (Phase 1)
# 6. Fine-tune (Phase 2)
# 7. Save model to models/natron_v2.pt
```

### API Server

```bash
# Start Flask API server
python api_server.py

# Or with custom paths:
MODEL_PATH=models/natron_v2.pt CONFIG_PATH=config.yaml python api_server.py
```

**API Endpoints:**

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Example Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "2024-01-01", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
      ...
    ]
  }'
```

**Example Response:**

```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.15,
  "direction_neutral": 0.16,
  "regime": "BULL_WEAK",
  "regime_id": 1,
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
# Start socket server for MetaTrader 5 integration
python mql5_socket_server.py

# Server listens on localhost:8888 by default
# Target latency: <50ms
```

## 📁 Project Structure

```
natron_v2/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── train.py                 # Main training pipeline
├── api_server.py            # Flask API server
├── mql5_socket_server.py    # Socket server for MQL5
├── feature_engine.py        # Technical feature generation
├── label_generator.py       # Label generation (V2)
├── sequence_creator.py      # Sequence dataset creation
├── model.py                 # Transformer model architecture
├── pretraining.py           # Phase 1: Pretraining
├── supervised_training.py  # Phase 2: Supervised fine-tuning
├── test_components.py       # Component testing utility
├── mql5_ea_template.mq5     # MQL5 EA template
├── models/                  # Trained models directory
├── checkpoints/             # Training checkpoints
└── logs/                    # Training logs
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data**: Input file, sequence length, train/val/test splits
- **Features**: Number of features, normalization settings
- **Labeling**: Thresholds, balancing, stochastic perturbation
- **Model**: Architecture parameters (d_model, nhead, layers, etc.)
- **Training**: Batch size, learning rate, epochs, etc.
- **API**: Host, port, debug mode
- **Socket**: MQL5 server configuration

## 🎯 Features

### Feature Engine (~100 Features)

- **Moving Averages** (13): MA, EMA, slopes, crossovers
- **Momentum** (13): RSI, ROC, CCI, Stochastic, MACD
- **Volatility** (15): ATR, Bollinger Bands, Keltner Channels
- **Volume** (9): OBV, VWAP, MFI, volume ratios
- **Price Patterns** (8): Doji, gaps, shadows, body percentage
- **Returns** (8): Log returns, intraday, cumulative
- **Trend Strength** (6): ADX, +DI, -DI, Aroon
- **Statistical** (6): Skewness, Kurtosis, Z-score, Hurst
- **Support/Resistance** (4): Distance to highs/lows
- **Smart Money Concepts** (6): Swing HL, BOS, CHoCH
- **Market Profile** (10): POC, VAH, VAL, entropy

### Label Generator V2

**Buy Signals** (≥2 conditions):
- Close > MA20 > MA50
- RSI > 50 or crossed up from <30
- Close > BB midband + MA20 slope > 0
- Volume > 1.5× rolling average
- Position in range ≥ 0.7
- MACD histogram > 0 and rising

**Sell Signals** (≥2 conditions):
- Close < MA20 < MA50
- RSI < 50 or turned down from >70
- Close < BB midband + MA20 slope < 0
- Volume spike + position ≤ 0.3
- MACD histogram < 0 and falling
- Minus_DI > Plus_DI

**Direction Labels** (3-class):
- Up: Future price > current + buffer
- Down: Future price < current - buffer
- Neutral: Otherwise

**Regime Labels** (6-class):
- BULL_STRONG: Trend > +2%, ADX > 25
- BULL_WEAK: 0 < trend ≤ +2%, ADX ≤ 25
- RANGE: Lateral market (default)
- BEAR_WEAK: -2% ≤ trend < 0, ADX ≤ 25
- BEAR_STRONG: Trend < -2%, ADX > 25
- VOLATILE: ATR > 90th percentile or volume spike

## 🔧 Development

### Running Tests

```bash
# Test individual components
python test_components.py

# Run training pipeline
python train.py

# Test API server
python api_server.py
```

### Customization

1. **Add Features**: Extend `FeatureEngine` class in `feature_engine.py`
2. **Modify Labels**: Adjust thresholds in `LabelGeneratorV2` in `label_generator.py`
3. **Change Architecture**: Edit `NatronTransformer` in `model.py`
4. **Add Training Phase**: Implement Phase 3 (RL) in new module

## 📊 Model Output

The model outputs probabilities and classifications for:

- **Buy Probability**: [0, 1] - Likelihood of buy signal
- **Sell Probability**: [0, 1] - Likelihood of sell signal
- **Direction**: [Up, Down, Neutral] - Price direction prediction
- **Regime**: [6 classes] - Current market regime
- **Confidence**: [0, 1] - Overall prediction confidence

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api_server.py"]
```

## 📝 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]

## 📧 Contact

[Your Contact Information]

---

**⚡ Natron Philosophy**: *"Natron doesn't just predict Buy/Sell — it learns the grammar of the market."*

1. **Structure Understanding** (Pretrain) - Learns temporal dependencies & market rhythm
2. **Signal Recognition** (Supervised) - Detects actionable trade setups
3. **Behavioral Adaptation** (Reinforcement) - Optimizes for real-world profit and risk
