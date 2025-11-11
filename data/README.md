# Data Directory

Place your OHLCV data file here as `data_export.csv`.

## Required Format

The CSV file should have the following columns:

```
time,open,high,low,close,volume
```

### Example:

```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,100.0,101.5,99.5,101.0,1200
2024-01-01 00:15:00,101.0,102.0,100.5,101.8,1350
2024-01-01 00:30:00,101.8,102.5,101.5,102.2,1400
...
```

## Data Requirements

- **Minimum samples:** 1000+ candles (for meaningful training)
- **Recommended:** 5000+ candles
- **Time format:** Any format parseable by pandas (e.g., "2024-01-01 00:00:00")
- **Timeframe:** M15 (15-minute) or H1 (1-hour) recommended
- **Completeness:** No gaps in the time series (missing candles will be forward-filled)

## Data Sources

You can export OHLCV data from:
- MetaTrader 4/5 (via script or manual export)
- TradingView (via Pine Script or manual export)
- Cryptocurrency exchanges (Binance, Coinbase, etc.)
- Financial data APIs (Alpha Vantage, Yahoo Finance, etc.)

## Sample Data

If no data is provided, the system will automatically generate sample data for demonstration purposes.

To use real data for production trading, replace the sample data with actual market data.
