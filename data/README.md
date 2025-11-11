# Data Directory

## Place your OHLCV data here

Your input data file should be named `data_export.csv` with the following format:

```csv
time,open,high,low,close,volume
2023-01-01 00:00:00,100.0,101.5,99.5,100.5,5000
2023-01-01 01:00:00,100.5,102.0,100.0,101.8,6500
...
```

### Column Specifications

| Column | Type | Description |
|--------|------|-------------|
| time | datetime | Timestamp of candle |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | int | Trading volume |

### Notes

- Timeframe: M15, H1, or H4 recommended
- Minimum rows: ~5000 for good training
- Data should be continuous (no gaps)
- If file not found, synthetic data will be generated for testing
