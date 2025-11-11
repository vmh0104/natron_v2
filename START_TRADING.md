# 🚀 Start Trading with Natron AI

Quick guide to go from installation to live trading in MetaTrader 5.

---

## ⚡ 5-Step Setup (30 minutes)

### Step 1: Install Dependencies (2 min)

```bash
git clone <repository-url>
cd natron-transformer
pip install -r requirements.txt
```

### Step 2: Train Model (10-15 min)

```bash
# Quick training (supervised only)
python main.py --skip-pretrain --skip-rl
```

**Note:** Full training takes 8-16 hours but gives better results.

### Step 3: Start Socket Server (1 min)

Open a terminal and run:

```bash
python src/bridge/socket_server.py
```

Keep this running! You should see:

```
🚀 MQL5 Socket Server Started
============================================================
   Host: 0.0.0.0
   Port: 9999
   Waiting for MetaTrader 5 connections...
============================================================
```

### Step 4: Install EA in MetaTrader 5 (5 min)

**4.1. Copy EA file:**

**Windows:**
```
Copy: mql5/NatronAI.mq5
To: C:\Users\<YourName>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Experts\
```

**Mac (Wine):**
```
Copy: mql5/NatronAI.mq5
To: ~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/
```

**4.2. Compile EA:**
1. Open MT5
2. Press F4 (MetaEditor)
3. Navigate to Experts/NatronAI.mq5
4. Press F7 (Compile)
5. Check for "0 errors, 0 warnings"

**4.3. Enable socket connections:**
1. MT5 → Tools → Options → Expert Advisors
2. ✅ Check **ALL** boxes:
   - Allow automated trading
   - Allow DLL imports
   - Allow WebRequest to listed URLs
   - **Allow socket connections** ← CRITICAL!

### Step 5: Start EA (5 min)

**5.1. Open chart:**
- Choose symbol (e.g., EURUSD)
- Set timeframe to M15 (recommended)

**5.2. Attach EA:**
- Navigator → Expert Advisors → NatronAI
- Drag onto chart

**5.3. Configure settings:**

**For Demo/Paper Trading:**
```
ServerHost: localhost
ServerPort: 9999
LotSize: 0.01 (minimum)
BuyThreshold: 0.6
SellThreshold: 0.6
EnableRegimeFilter: true
```

**5.4. Enable Auto Trading:**
- Click "AutoTrading" button in MT5 toolbar (should be green)

**5.5. Verify connection:**

Check chart - you should see:
```
🧠 NATRON AI
═══════════════
Buy:  XX.XX%
Sell: XX.XX%
...
```

Check Experts tab for:
```
✅ Connected to Natron AI Server
Ready to trade!
```

---

## 🧪 Testing Phase (1-2 weeks)

### Demo Account Testing

**DON'T GO LIVE YET!**

1. **Monitor for 1-2 weeks** on demo account
2. **Track metrics:**
   - Number of trades
   - Win rate
   - Average profit/loss
   - Maximum drawdown
   - Sharpe ratio

3. **Check logs daily:**
   - MT5 Experts tab
   - Socket server console
   - Look for errors/warnings

4. **Optimize parameters** if needed:
   - Adjust thresholds
   - Try different regimes
   - Test different symbols

---

## ⚙️ Configuration Tips

### Conservative (Low Risk)

```
LotSize: 0.01
BuyThreshold: 0.7
SellThreshold: 0.7
StopLoss: 50
TakeProfit: 100
EnableRegimeFilter: true
AllowedRegimes: "BULL_STRONG"
```

**Expected:** Low frequency, high accuracy

### Moderate (Balanced)

```
LotSize: 0.1
BuyThreshold: 0.6
SellThreshold: 0.6
StopLoss: 100
TakeProfit: 200
EnableRegimeFilter: true
AllowedRegimes: "BULL_STRONG,BULL_WEAK,RANGE"
```

**Expected:** Medium frequency, balanced risk/reward

### Aggressive (High Risk)

```
LotSize: 0.5
BuyThreshold: 0.5
SellThreshold: 0.5
StopLoss: 150
TakeProfit: 300
EnableRegimeFilter: false
```

**Expected:** High frequency, more risk

---

## 📊 What to Monitor

### Daily Checks

1. **Socket server running?**
   - Check console for errors
   - Monitor prediction logs

2. **EA connected?**
   - Check chart display
   - Check Experts tab

3. **Trades executing?**
   - Review trade history
   - Check for errors

4. **Performance metrics:**
   - Win/loss ratio
   - Profit factor
   - Drawdown

### Weekly Review

1. **Analyze performance:**
   - Compare to expectations
   - Identify patterns
   - Check regime distribution

2. **Model health:**
   - Is confidence staying high?
   - Are predictions reasonable?

3. **Parameter adjustment:**
   - Tweak thresholds if needed
   - Adjust risk settings

---

## 🔧 Common Issues & Solutions

### EA won't connect

**Symptom:** "Failed to connect to Python server"

**Solution:**
1. Check server is running: `python src/bridge/socket_server.py`
2. Verify port is correct (9999)
3. Check firewall settings
4. Verify "Allow socket connections" is enabled in MT5

### No predictions showing

**Symptom:** Chart shows EA name but no predictions

**Solution:**
1. Check Experts tab for errors
2. Verify 96+ candles available on chart
3. Check server logs for incoming requests
4. Test server: `python test_mql5_bridge.py`

### No trades executing

**Symptom:** Predictions shown but no trades

**Solution:**
1. Lower thresholds (try 0.5)
2. Disable regime filter temporarily
3. Check account balance is sufficient
4. Verify "Allow automated trading" is enabled
5. Check Experts tab for order errors

### High latency (>200ms)

**Solution:**
1. Use GPU (10-20x faster)
2. Run server locally (not remote)
3. Close other programs
4. Check network connection

---

## 📈 Going Live (After Testing)

### Pre-Live Checklist

✅ Tested on demo for 1-2 weeks  
✅ Positive profit on demo  
✅ Understanding of EA behavior  
✅ Risk management configured  
✅ Model trained on recent data  
✅ Emergency stop plan ready

### Live Trading Setup

1. **Switch to live account** in MT5

2. **Reduce position size** initially:
   ```
   LotSize: 0.01  # Start small!
   ```

3. **Tighten risk management:**
   ```
   StopLoss: 50  # Tighter stops
   ```

4. **Monitor closely** for first week

5. **Scale gradually** after success

### Risk Management Rules

1. **Never risk more than 2% per trade**
2. **Set daily loss limit** (e.g., -5%)
3. **Take regular profits**
4. **Diversify** across symbols
5. **Keep stop losses** always enabled

---

## 🔄 Maintenance

### Weekly Tasks

- Review performance metrics
- Check server logs for errors
- Verify model predictions make sense
- Backup trading logs

### Monthly Tasks

- **Retrain model** with new data:
  ```bash
  # Export recent MT5 data
  # Save to data/data_export.csv
  python main.py
  ```

- Review and adjust parameters
- Update risk management rules
- Analyze market regime changes

### Model Updates

When to retrain:
- Performance degradation
- Major market changes
- Monthly maintenance
- After 1000+ new candles

---

## 🎓 Best Practices

1. **Start Small:** Begin with minimum lot size
2. **Paper Trade:** Test thoroughly on demo first
3. **Risk Management:** Never skip stop losses
4. **Diversification:** Don't put all capital in one symbol
5. **Monitoring:** Check daily, especially first weeks
6. **Record Keeping:** Log all settings and changes
7. **Model Maintenance:** Retrain monthly
8. **Emotional Discipline:** Trust the system or stop it
9. **Emergency Plan:** Know how to stop EA quickly
10. **Continuous Learning:** Analyze what works

---

## 📞 Need Help?

### Resources

- **Full MQL5 Guide:** `MQL5_INTEGRATION.md`
- **API Documentation:** `README.md`
- **Quick Start:** `QUICKSTART.md`

### Debugging

1. **Check logs:**
   - MT5 Experts tab
   - Socket server console
   - `logs/training.log`

2. **Test components:**
   ```bash
   python test_api.py
   python test_mql5_bridge.py
   ```

3. **Verify model:**
   ```bash
   # Should exist
   ls -lh model/natron_v2.pt
   ls -lh model/scaler.pkl
   ```

---

## ⚠️ FINAL WARNING

**RISK DISCLAIMER:**

Algorithmic trading involves substantial risk of loss. This system:
- Is for educational purposes
- Does not guarantee profits
- Can and will have losing periods
- Requires proper risk management

**Before live trading:**
- Thoroughly test on demo account
- Understand all risks involved
- Never invest money you can't afford to lose
- Consider consulting a financial advisor

**The developers are NOT responsible for trading losses.**

---

## 🎉 Ready to Trade!

You now have:
✅ Trained AI model  
✅ Socket server running  
✅ EA installed in MT5  
✅ Knowledge to trade safely

**Start on demo, test thoroughly, then scale carefully.**

**Good luck and trade wisely! 📈**

---

**"The best traders are not the ones who predict the market perfectly,  
but the ones who manage risk effectively."**
