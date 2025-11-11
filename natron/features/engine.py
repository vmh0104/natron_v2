"""
Feature engineering module for Natron Transformer.

Generates ~100 technical indicators derived from OHLCV data streams. The implementation
keeps dependencies minimal (NumPy/Pandas) and is optimized for vectorized operations.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from natron.config import FeatureConfig


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _roc(series: pd.Series, period: int = 12) -> pd.Series:
    return series.pct_change(periods=period)


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical_price.rolling(period).mean()
    mean_dev = (typical_price - sma).abs().rolling(period).mean()
    return (typical_price - sma) / (0.015 * mean_dev + 1e-9)


def _stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    k = (df["close"] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(3).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": hist})


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _hurst(ts: pd.Series, window: int) -> pd.Series:
    """Compute Hurst exponent via R/S analysis."""
    result = np.full(len(ts), np.nan)
    values = ts.values
    for i in range(window, len(ts)):
        segment = values[i - window : i]
        mean = segment.mean()
        dev = segment - mean
        cumulative = np.cumsum(dev)
        r = cumulative.max() - cumulative.min()
        s = np.std(segment)
        if s == 0:
            hurst_val = np.nan
        else:
            hurst_val = np.log((r + 1e-9) / (s + 1e-9)) / np.log(window)
        result[i] = hurst_val
    return pd.Series(result, index=ts.index)


def _market_profile(close: pd.Series, bins: int) -> pd.DataFrame:
    """
    Approximate market profile statistics: point of control (POC), value area high (VAH),
    value area low (VAL), entropy, and participation rate.
    """
    rolling = close.rolling(window=bins, min_periods=bins)

    poc = rolling.apply(lambda x: np.histogram(x, bins=10)[1][np.argmax(np.histogram(x, bins=10)[0])], raw=False)

    def value_area(series: pd.Series) -> tuple[float, float]:
        hist, bin_edges = np.histogram(series, bins=20)
        cumulative = np.cumsum(hist) / hist.sum()
        low_idx = np.searchsorted(cumulative, 0.15)
        high_idx = np.searchsorted(cumulative, 0.85)
        return bin_edges[low_idx], bin_edges[min(high_idx, len(bin_edges) - 1)]

    vah, val = [], []
    entropy = rolling.apply(lambda x: _entropy(np.histogram(x, bins=20)[0]), raw=False)
    participation = rolling.apply(lambda x: np.unique(np.round(x, 4)).size / len(x), raw=False)

    for window_values in rolling:
        if window_values is None or len(window_values) < bins:
            vah.append(np.nan)
            val.append(np.nan)
        else:
            low, high = value_area(window_values)
            val.append(low)
            vah.append(high)

    return pd.DataFrame(
        {
            "profile_poc": poc,
            "profile_val": pd.Series(val, index=close.index),
            "profile_vah": pd.Series(vah, index=close.index),
            "profile_entropy": entropy,
            "profile_participation": participation,
        }
    )


def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(float)
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-9)))


class FeatureEngine:
    """Generate technical features for Natron."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values("time")
        df = df.set_index("time")

        features: Dict[str, pd.Series] = {}
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Moving averages and slopes
        for window in self.config.rolling_windows:
            sma = close.rolling(window).mean()
            ema = _ema(close, window)
            slope = sma.diff()
            features[f"sma_{window}"] = sma
            features[f"ema_{window}"] = ema
            features[f"sma_slope_{window}"] = slope
            features[f"ema_slope_{window}"] = ema.diff()
            ratio = close / (sma + 1e-9)
            features[f"price_sma_ratio_{window}"] = ratio

        # Crossovers: price vs long MA, short vs long
        for short, long in zip(self.config.rolling_windows[:-1], self.config.rolling_windows[1:]):
            short_ma = close.rolling(short).mean()
            long_ma = close.rolling(long).mean()
            crossover = (short_ma > long_ma).astype(float)
            features[f"ma_crossover_{short}_{long}"] = crossover

        # Momentum indicators
        features["rsi"] = _rsi(close, period=self.config.stochastic_period)
        features["roc_5"] = _roc(close, period=5)
        features["roc_10"] = _roc(close, period=10)
        features["roc_20"] = _roc(close, period=20)
        features["cci"] = _cci(df, period=20)

        stochastic = _stochastic(df, self.config.stochastic_period)
        for col in stochastic.columns:
            features[col] = stochastic[col]

        macd = _macd(close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
        for col in macd.columns:
            features[col] = macd[col]
        features["macd_hist_slope"] = macd["macd_hist"].diff()

        # Volatility features
        atr = _atr(df, self.config.atr_window)
        features["atr"] = atr
        features["atr_pct"] = atr / close
        for window in self.config.volatility_windows:
            std = close.pct_change().rolling(window).std()
            features[f"return_std_{window}"] = std
            features[f"high_low_range_{window}"] = (high - low).rolling(window).mean()

        bb_middle = close.rolling(self.config.bb_window).mean()
        bb_std = close.rolling(self.config.bb_window).std()
        features["bb_upper"] = bb_middle + self.config.bb_std * bb_std
        features["bb_lower"] = bb_middle - self.config.bb_std * bb_std
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / bb_middle
        features["bb_percent"] = (close - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-9)

        # Keltner channel
        ema_ema = _ema(close, self.config.atr_window)
        features["keltner_upper"] = ema_ema + atr * 2
        features["keltner_lower"] = ema_ema - atr * 2
        features["keltner_width"] = (features["keltner_upper"] - features["keltner_lower"]) / ema_ema

        # Volume features
        cum_volume_price = (close * volume).cumsum()
        cum_volume = volume.cumsum()
        vwap = cum_volume_price / (cum_volume + 1e-9)
        features["vwap"] = vwap
        features["vwap_distance"] = (close - vwap) / close
        features["obv"] = np.where(close > close.shift(), volume, -volume).cumsum()

        price_range = high - low
        mfi_raw = 0.5 * (high + low) * volume
        positive_flow = np.where(close > close.shift(), mfi_raw, 0.0)
        negative_flow = np.where(close < close.shift(), mfi_raw, 0.0)
        positive_rolling = pd.Series(positive_flow, index=df.index).rolling(14).sum()
        negative_rolling = pd.Series(negative_flow, index=df.index).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_rolling / (negative_rolling + 1e-9)))
        features["mfi"] = mfi

        for window in self.config.volume_windows:
            vol_mean = volume.rolling(window).mean()
            features[f"volume_ratio_{window}"] = volume / (vol_mean + 1e-9)
            features[f"volume_zscore_{window}"] = (volume - vol_mean) / (volume.rolling(window).std() + 1e-9)

        # Price pattern features
        body = (close - open_).abs()
        candle_range = high - low
        features["doji"] = (body / (candle_range + 1e-9) < 0.1).astype(float)
        features["upper_shadow"] = (high - close) / (candle_range + 1e-9)
        features["lower_shadow"] = (open_ - low) / (candle_range + 1e-9)
        features["body_percent"] = body / (candle_range + 1e-9)
        features["gap_up"] = (open_ > close.shift()).astype(float)
        features["gap_down"] = (open_ < close.shift()).astype(float)
        features["position_in_range"] = (close - low) / (candle_range + 1e-9)

        # Returns features
        features["log_return"] = np.log(close / close.shift())
        features["return_1"] = close.pct_change()
        features["return_3"] = close.pct_change(3)
        features["return_5"] = close.pct_change(5)
        features["intraday_range"] = price_range / open_
        features["cumulative_return"] = features["log_return"].cumsum().apply(np.exp) - 1
        features["rolling_return_20"] = close.pct_change(20)
        features["drawdown"] = (close / close.cummax()) - 1

        # Trend strength
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr_rolling = tr.rolling(self.config.atr_window).mean()
        plus_dm = np.where((high - high.shift()) > (low.shift() - low), (high - high.shift()).clip(lower=0), 0.0)
        minus_dm = np.where((low.shift() - low) > (high - high.shift()), (low.shift() - low).clip(lower=0), 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(self.config.atr_window).mean() / (tr_rolling + 1e-9)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(self.config.atr_window).mean() / (tr_rolling + 1e-9)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)) * 100
        adx = dx.rolling(self.config.atr_window).mean()
        features["adx"] = adx
        features["+di"] = plus_di
        features["-di"] = minus_di

        aroon_up = (
            close.rolling(self.config.stochastic_period).apply(lambda x: len(x) - np.argmax(x[::-1]) - 1, raw=False)
            / self.config.stochastic_period
        )
        aroon_down = (
            close.rolling(self.config.stochastic_period).apply(lambda x: len(x) - np.argmin(x[::-1]) - 1, raw=False)
            / self.config.stochastic_period
        )
        features["aroon_up"] = 1 - aroon_up
        features["aroon_down"] = 1 - aroon_down

        # Statistical features
        for window in (10, 20, 50, 100):
            returns = close.pct_change().rolling(window)
            features[f"return_skew_{window}"] = returns.skew()
            features[f"return_kurt_{window}"] = returns.kurt()
            mean = close.rolling(window).mean()
            std = close.rolling(window).std()
            features[f"price_zscore_{window}"] = (close - mean) / (std + 1e-9)
        for window in self.config.hurst_windows:
            features[f"hurst_{window}"] = _hurst(close, window)

        # Support/Resistance
        for window in (20, 30, 50):
            rolling_high = high.rolling(window).max()
            rolling_low = low.rolling(window).min()
            features[f"dist_high_{window}"] = (rolling_high - close) / close
            features[f"dist_low_{window}"] = (close - rolling_low) / close

        # Smart Money Concepts approximations
        swing_high = (high.shift(1) < high) & (high.shift(-1) < high)
        swing_low = (low.shift(1) > low) & (low.shift(-1) > low)
        features["swing_high"] = swing_high.astype(float)
        features["swing_low"] = swing_low.astype(float)

        structure_break = close > close.shift(5)
        change_of_character = np.sign(close.diff()).diff().abs()
        features["bos"] = structure_break.astype(float)
        features["choch"] = (change_of_character > 0).astype(float)
        features["swing_strength"] = close.rolling(5).apply(lambda x: x[-1] - x[0], raw=False)

        # Market profile
        profile = _market_profile(close, self.config.market_profile_bins)
        for col in profile.columns:
            features[col] = profile[col]

        # Additional context
        features["price_volatility_ratio"] = features["return_std_10"] / (features["return_std_50"] + 1e-9)
        features["volume_price_correlation"] = (
            close.pct_change()
            .rolling(20)
            .corr(volume.pct_change())
        )
        features["rolling_beta"] = (
            close.pct_change()
            .rolling(20)
            .cov(close.pct_change(2).rolling(20).mean())
            / (close.pct_change().rolling(20).var() + 1e-9)
        )
        features["momentum_acceleration"] = features["roc_5"].diff()
        features["momentum_reverse"] = -features["momentum_acceleration"]

        feature_df = pd.DataFrame(features).fillna(method="bfill").fillna(method="ffill")
        feature_df = feature_df.fillna(0.0)

        # Apply mild Gaussian noise to avoid degenerate constant features
        noise = np.random.normal(loc=0.0, scale=self.config.technical_noise, size=feature_df.shape)
        feature_df += noise

        feature_df = feature_df.replace([np.inf, -np.inf], 0.0)
        feature_df = feature_df.astype(np.float32)
        feature_df.index = df.index
        return feature_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """For compatibility with scikit-like interface."""
        return self.transform(df)
