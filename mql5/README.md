# MQL5 Expert Advisor - NatronAI

AI-powered trading Expert Advisor for MetaTrader 5.

## 📁 Files

- **NatronAI.mq5** - Main Expert Advisor file

## 🚀 Installation

### Windows

1. Copy `NatronAI.mq5` to:
   ```
   C:\Users\<YourName>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Experts\
   ```

2. Open MetaEditor in MT5 (press F4)

3. Open `NatronAI.mq5` and compile (press F7)

4. Drag EA from Navigator onto chart

### Mac (Wine/PlayOnMac)

1. Copy `NatronAI.mq5` to:
   ```
   ~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/
   ```

2. Follow same steps as Windows

## ⚙️ Configuration

### Required Settings in MT5

**Tools → Options → Expert Advisors:**

✅ Allow automated trading  
✅ Allow DLL imports  
✅ Allow WebRequest to listed URLs  
✅ **Allow socket connections** (IMPORTANT!)

### EA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| ServerHost | localhost | Python server IP/hostname |
| ServerPort | 9999 | Socket server port |
| MagicNumber | 20241111 | Unique identifier |
| LotSize | 0.1 | Position size |
| BuyThreshold | 0.6 | Buy signal threshold (60%) |
| SellThreshold | 0.6 | Sell signal threshold (60%) |
| StopLoss | 100 | Stop loss in points |
| TakeProfit | 200 | Take profit in points |
| UseTrailingStop | true | Enable trailing stop |
| TrailingStop | 50 | Trailing stop distance |
| EnableRegimeFilter | true | Filter by market regime |
| AllowedRegimes | BULL_STRONG,BULL_WEAK,RANGE | Allowed regimes |
| SequenceLength | 96 | Candles sent to AI |
| UpdateInterval | 60 | Update frequency (sec) |

## 🔧 Before Running

### 1. Start Python Socket Server

```bash
cd /path/to/natron-transformer
python src/bridge/socket_server.py
```

### 2. Train Model (if not done)

```bash
python main.py --skip-pretrain --skip-rl
```

### 3. Test Connection

```bash
python test_mql5_bridge.py
```

## 📊 On-Chart Display

The EA shows real-time predictions:

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

## 🎯 Trading Logic

### Entry Signals

- **BUY:** When `buy_prob > BuyThreshold` (default 60%)
- **SELL:** When `sell_prob > SellThreshold` (default 60%)

### Exit Signals

- Opposite signal (auto-closes before reversing)
- Stop loss hit
- Take profit reached
- Trailing stop triggered
- Low confidence (<40%)

### Position Management

- Only one position per direction
- Closes opposite before opening new
- Respects magic number (multi-EA support)

## 🛡️ Risk Management

### Built-in Features

✅ Stop Loss / Take Profit  
✅ Trailing Stop  
✅ Position Sizing  
✅ Regime Filtering  
✅ Confidence Threshold  
✅ Slippage Control

### Recommended Settings

**Conservative:**
```
LotSize = 0.01
BuyThreshold = 0.7
SellThreshold = 0.7
StopLoss = 50
TakeProfit = 100
```

**Moderate:**
```
LotSize = 0.1
BuyThreshold = 0.6
SellThreshold = 0.6
StopLoss = 100
TakeProfit = 200
```

**Aggressive:**
```
LotSize = 0.5
BuyThreshold = 0.5
SellThreshold = 0.5
StopLoss = 150
TakeProfit = 300
```

## 🔍 Market Regimes

The AI identifies 6 market regimes:

| Regime | Description | Typical Action |
|--------|-------------|----------------|
| BULL_STRONG | Strong uptrend | Long only |
| BULL_WEAK | Weak uptrend | Long preferred |
| RANGE | Sideways | Both directions |
| BEAR_WEAK | Weak downtrend | Short preferred |
| BEAR_STRONG | Strong downtrend | Short only |
| VOLATILE | High volatility | Avoid or reduce size |

### Regime Filter Examples

**Bull markets only:**
```
AllowedRegimes = "BULL_STRONG,BULL_WEAK"
```

**Range trading:**
```
AllowedRegimes = "RANGE"
```

**All except volatile:**
```
AllowedRegimes = "BULL_STRONG,BULL_WEAK,RANGE,BEAR_WEAK,BEAR_STRONG"
```

## 📈 Performance Tips

1. **Run on VPS:** For 24/7 operation
2. **Low Latency:** Local or nearby server
3. **Test First:** Demo account for 1-2 weeks
4. **Monitor:** Check logs and performance daily
5. **Retrain:** Update model monthly with new data
6. **Diversify:** Multiple symbols/timeframes

## 🐛 Troubleshooting

### EA won't start

Check MT5 settings:
- Tools → Options → Expert Advisors
- ✅ Enable all checkboxes including "Allow socket connections"

### Can't connect to server

1. Server running? `python src/bridge/socket_server.py`
2. Correct host/port in EA settings?
3. Firewall blocking port 9999?
4. Test: `python test_mql5_bridge.py`

### No trades

1. Lower thresholds (BuyThreshold=0.5)
2. Disable regime filter
3. Check Expert tab for messages
4. Verify sufficient account balance

### High slippage

1. Increase `Slippage` parameter
2. Trade during liquid hours
3. Use ECN broker

## 📊 Logging

Check MT5 **Experts** tab for:
- Connection status
- AI predictions
- Trade signals
- Order execution
- Errors/warnings

Example logs:
```
✅ Connected to Natron AI Server
🎯 Buy: 0.71, Sell: 0.24, Regime: BULL_WEAK
🟢 BUY Signal - Probability: 71.0%
✅ Order executed successfully - Ticket: 123456
```

## ⚠️ Important Notes

1. **Strategy Tester:** MT5 tester doesn't support sockets - backtest in Python instead
2. **Multiple Symbols:** Use different MagicNumber for each
3. **Connection:** EA reconnects automatically if disconnected
4. **Updates:** Close/reopen EA after changing settings

## 📚 Documentation

Full documentation: `../MQL5_INTEGRATION.md`

## 🎓 Support

For issues:
1. Check logs in MT5 Experts tab
2. Check server logs
3. See `MQL5_INTEGRATION.md` troubleshooting section

---

**⚠️ DISCLAIMER:**

This EA is for educational purposes. Trading involves substantial risk. Always test on demo account first. Never risk money you cannot afford to lose.

**Good luck trading! 📈**
