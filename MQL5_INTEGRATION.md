# 🔌 MQL5 Integration Guide

Complete guide to integrate Natron Transformer AI with MetaTrader 5.

---

## 📋 Overview

The system consists of three components:

```
MQL5 EA (NatronAI.mq5)
    ↕️ Socket Connection
Python Socket Server (socket_server.py)
    ↕️ Direct Function Call
Natron AI Model (GPU)
```

**Latency:** 30-80ms end-to-end (MQL5 → Python → Model → MQL5)

---

## 🚀 Quick Setup

### Step 1: Train the Model

```bash
# Quick training (10-15 minutes)
python main.py --skip-pretrain --skip-rl

# Or full training (8-16 hours)
python main.py
```

### Step 2: Start Socket Server

```bash
# Start socket server on port 9999
python src/bridge/socket_server.py

# Custom host/port
python src/bridge/socket_server.py --host 0.0.0.0 --port 9999
```

You should see:
```
🚀 MQL5 Socket Server Started
============================================================
   Host: 0.0.0.0
   Port: 9999
   Waiting for MetaTrader 5 connections...
============================================================
```

### Step 3: Install MQL5 EA

1. Copy `mql5/NatronAI.mq5` to your MetaTrader 5:
   - **Windows:** `C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Experts\`
   - **Mac/Wine:** `~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/`

2. Open MetaEditor (F4 in MT5)

3. Compile `NatronAI.mq5` (F7)

4. Drag EA onto a chart

### Step 4: Configure EA

**Input Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| ServerHost | localhost | Python server address |
| ServerPort | 9999 | Python server port |
| MagicNumber | 20241111 | Unique EA identifier |
| LotSize | 0.1 | Position size |
| BuyThreshold | 0.6 | Buy signal threshold (0-1) |
| SellThreshold | 0.6 | Sell signal threshold (0-1) |
| StopLoss | 100 | Stop loss in points |
| TakeProfit | 200 | Take profit in points |
| UseTrailingStop | true | Enable trailing stop |
| TrailingStop | 50 | Trailing stop distance |
| EnableRegimeFilter | true | Filter by market regime |
| AllowedRegimes | BULL_STRONG,BULL_WEAK,RANGE | Allowed regimes |
| SequenceLength | 96 | Candles to analyze |
| UpdateInterval | 60 | Update frequency (sec) |

---

## 🎯 EA Features

### Real-Time AI Signals

The EA displays AI predictions directly on the chart:

```
🧠 NATRON AI
═══════════════
Buy:  71.00%
Sell: 24.00%
Dir↑: 69.00%
Dir↓: 25.00%
───────────────
Regime: BULL_WEAK
Confidence: 82.00%
Latency: 45.2ms
```

### Automatic Trading

- **Buy Signal:** Opens long when `buy_prob > BuyThreshold`
- **Sell Signal:** Opens short when `sell_prob > SellThreshold`
- **Close Signal:** Closes all when confidence < 0.4
- **Position Management:** Closes opposite position before opening new one

### Risk Management

1. **Stop Loss / Take Profit:** Configurable SL/TP
2. **Trailing Stop:** Dynamic stop loss adjustment
3. **Regime Filter:** Trade only in favorable market conditions
4. **Position Limit:** One position per direction
5. **Magic Number:** Separate EA instances don't interfere

### Market Regime Filter

Enable `EnableRegimeFilter` to trade only in specific regimes:

- **BULL_STRONG:** Strong uptrend (trend > 2%, ADX > 25)
- **BULL_WEAK:** Weak uptrend
- **RANGE:** Sideways market
- **BEAR_WEAK:** Weak downtrend
- **BEAR_STRONG:** Strong downtrend (trend < -2%, ADX > 25)
- **VOLATILE:** High volatility period

Example: `AllowedRegimes = "BULL_STRONG,BULL_WEAK,RANGE"`

---

## 🔧 Advanced Configuration

### Multiple Symbols

Run separate EA instances for different symbols:

1. Set different `MagicNumber` for each symbol
2. Each EA connects to the same socket server
3. Server handles multiple concurrent connections

### Remote Server

Run Python server on remote GPU machine:

**On GPU Server:**
```bash
python src/bridge/socket_server.py --host 0.0.0.0 --port 9999
```

**In MT5 EA:**
```
ServerHost = "192.168.1.100"  // Your GPU server IP
```

**Firewall:** Open port 9999 (TCP)

### High-Frequency Trading

For faster updates:

```
UpdateInterval = 15  // Update every 15 seconds
```

**Note:** Model latency is 30-80ms, but frequent updates may cause overtrading.

---

## 🧪 Testing

### Test Socket Server

```bash
# Test connection and predictions
python test_mql5_bridge.py

# Test remote server
python test_mql5_bridge.py --host 192.168.1.100 --port 9999
```

Expected output:
```
🧪 Testing MQL5 Socket Bridge
============================================================
Connecting to localhost:9999...
✅ Connected to server

📊 Test 1: Health Check
----------------------------------------
Status: healthy
Uptime: 120 seconds
Total Predictions: 45

📊 Test 3: AI Prediction
----------------------------------------
🎯 Prediction Results:
   Buy Probability:    0.7100 (71.0%)
   Sell Probability:   0.2400 (24.0%)
   ...
```

### Strategy Tester (Backtest)

⚠️ **MT5 Strategy Tester doesn't support socket connections**

For backtesting:
1. Export historical data from MT5
2. Run Python backtesting script (create custom script)
3. Analyze results before live trading

### Paper Trading

1. Open **Demo Account** in MT5
2. Run EA with real-time data
3. Monitor performance for 1-2 weeks
4. Adjust parameters based on results

---

## 📊 Monitoring

### Server Logs

Socket server logs all predictions:

```
2024-11-11 10:15:30 - INFO - 📡 New connection from 127.0.0.1
2024-11-11 10:15:31 - INFO - ✅ Prediction sent to 127.0.0.1: 
    Buy=0.71, Sell=0.24, Regime=BULL_WEAK, Latency=45.2ms
```

### EA Logs

Check MT5 Experts tab for EA logs:

```
🟢 BUY Signal - Probability: 71.0%
✅ Order executed successfully
   Ticket: 123456
   Price: 1.12345
   SL: 1.12245
   TP: 1.12545
```

### Performance Metrics

```bash
# Server statistics
python src/bridge/socket_server.py
# Press Ctrl+C to see stats

📊 Server Statistics
============================================================
   Uptime: 3600 seconds
   Total Requests: 240
   Total Predictions: 240
   Requests/sec: 0.07
============================================================
```

---

## 🔒 Security

### Network Security

1. **Localhost Only (Default):**
   ```bash
   python src/bridge/socket_server.py --host 127.0.0.1
   ```

2. **LAN Access:**
   ```bash
   python src/bridge/socket_server.py --host 0.0.0.0
   ```
   - Use firewall rules
   - Restrict to trusted IPs

3. **VPN:** For remote access, use VPN instead of exposing port

### Authentication (Optional)

Add API key authentication to `socket_server.py`:

```python
API_KEY = "your-secret-key-here"

def process_request(self, request: dict) -> dict:
    if request.get('api_key') != API_KEY:
        return {'error': 'Unauthorized'}
    # ... rest of code
```

---

## ⚡ Performance Optimization

### Reduce Latency

1. **GPU Server:** Run on machine with NVIDIA GPU
2. **Local Deployment:** Run server on same machine as MT5
3. **Model Optimization:** Use mixed precision (already enabled)
4. **Batch Size:** Already optimized for single predictions

### Increase Throughput

1. **Connection Pooling:** Keep socket connection alive (already implemented)
2. **Multiple Workers:** Server uses threading for concurrent connections
3. **Caching:** Add prediction cache for repeated queries (optional)

---

## 🐛 Troubleshooting

### EA Can't Connect to Server

**Error:** `Failed to connect: <error code>`

**Solutions:**
1. Check server is running: `python src/bridge/socket_server.py`
2. Verify host/port in EA settings
3. Check firewall allows port 9999
4. MT5 → Tools → Options → Expert Advisors → ✅ "Allow DLL imports" and "Allow socket connections"

### High Latency (>200ms)

**Causes:**
- Network latency (remote server)
- CPU-only mode (no GPU)
- Large model loading time

**Solutions:**
- Use GPU for 10-20x speedup
- Deploy server locally
- Check network connection

### No Trading Activity

**Possible Reasons:**
1. Thresholds too high (reduce `BuyThreshold`/`SellThreshold`)
2. Regime filter blocking (disable `EnableRegimeFilter`)
3. Low confidence predictions
4. Insufficient account balance

### Model Errors

**Error:** `Model not found`

**Solution:**
```bash
# Train model first
python main.py --skip-pretrain --skip-rl
```

**Error:** `CUDA out of memory`

**Solution:**
- Use CPU mode (automatic fallback)
- Or reduce batch size in config

---

## 📈 Trading Strategies

### Conservative Strategy

```
BuyThreshold = 0.7
SellThreshold = 0.7
EnableRegimeFilter = true
AllowedRegimes = "BULL_STRONG"
```

Lower frequency, higher accuracy.

### Aggressive Strategy

```
BuyThreshold = 0.5
SellThreshold = 0.5
EnableRegimeFilter = false
```

Higher frequency, more trades.

### Regime-Adaptive Strategy

```
EnableRegimeFilter = true
AllowedRegimes = "BULL_STRONG,BULL_WEAK,RANGE,BEAR_WEAK,BEAR_STRONG"
```

Trade in all regimes except volatile.

---

## 🎓 Best Practices

1. **Paper Trade First:** Test on demo account for 1-2 weeks
2. **Start Small:** Begin with minimum lot size
3. **Monitor Daily:** Check performance and logs regularly
4. **Risk Management:** Never risk more than 2% per trade
5. **Model Retraining:** Retrain model monthly with new data
6. **Diversification:** Don't rely on a single strategy
7. **Market Conditions:** Model performance varies with market regime

---

## 🔄 Model Updates

### Retrain with New Data

```bash
# Export MT5 data to CSV
# Place in data/data_export.csv

# Retrain model
python main.py

# Restart socket server (model auto-reloads)
```

### Live Trading Performance

Monitor and log:
- Win rate
- Average profit/loss
- Sharpe ratio
- Maximum drawdown

If performance degrades:
1. Collect recent market data
2. Retrain model
3. Re-test on demo account

---

## 📞 Support

### Common Issues

Check `MQL5_INTEGRATION.md` troubleshooting section.

### Logs

Useful logs for debugging:
- **Server:** Console output from `socket_server.py`
- **EA:** MT5 Experts tab
- **Training:** `logs/training.log`

---

## 🎉 You're Ready!

Your Natron EA is now connected to the AI model and ready for trading.

**Next Steps:**
1. Run EA on demo account
2. Monitor performance for 1-2 weeks
3. Optimize parameters
4. Switch to live account (with caution)

---

**⚠️ RISK DISCLAIMER:**

Trading financial instruments involves substantial risk. This EA is for educational purposes. Always:
- Test thoroughly on demo
- Use proper risk management
- Never invest more than you can afford to lose
- Past performance ≠ future results

**Happy Trading! 📈**
