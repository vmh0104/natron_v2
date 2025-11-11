"""
Label generation module implementing bias-reduced institutional labeling strategy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from natron.config import LabelConfig


class LabelGeneratorV2:
    """Generate multi-task labels for the Natron Transformer."""

    def __init__(self, config: LabelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for buy/sell classification, direction, and regime prediction.

        Parameters
        ----------
        df:
            Original OHLCV dataframe (must contain columns: close, high, low, volume).
        features:
            DataFrame produced by FeatureEngine aligned with df index.
        """
        assert df.index.equals(features.index), "DataFrame and features must share the same index"

        labels = pd.DataFrame(index=df.index)
        labels["buy_raw"], buy_score = self._compute_buy_signals(df, features)
        labels["sell_raw"], sell_score = self._compute_sell_signals(df, features)

        # Apply balancing
        labels["buy"] = self._balance(labels["buy_raw"], buy_score, "buy")
        labels["sell"] = self._balance(labels["sell_raw"], sell_score, "sell")

        labels["direction"] = self._compute_direction(df["close"])
        labels["regime"] = self._compute_regime(df, features)

        self._print_distribution(labels)
        return labels[["buy", "sell", "direction", "regime"]]

    def _compute_buy_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = df["close"]
        volume = df["volume"]
        sma20 = features.get("sma_20", close.rolling(20).mean())
        sma50 = features.get("sma_50", close.rolling(50).mean())
        rsi = features.get("rsi", pd.Series(np.nan, index=df.index))
        bb_mid = (features["bb_upper"] + features["bb_lower"]) / 2
        macd_hist = features.get("macd_hist", pd.Series(np.nan, index=df.index))
        macd_hist_slope = features.get("macd_hist_slope", macd_hist.diff())
        ma20_slope = features.get("sma_slope_20", sma20.diff())
        vol_mean20 = volume.rolling(20).mean()
        position_in_range = features.get("position_in_range", pd.Series(np.nan, index=df.index))
        plus_conditions = []

        cond1 = (close > sma20) & (sma20 > sma50)
        cond2 = (rsi > 50) | ((rsi.diff() > 0) & (rsi.shift() < 30) & (rsi > 30))
        cond3 = (close > bb_mid) & (ma20_slope > 0)
        cond4 = volume > self.config.volume_spike_multiplier * (vol_mean20 + 1e-9)
        cond5 = position_in_range >= 0.7
        cond6 = (macd_hist > 0) & (macd_hist_slope > 0)

        conditions = [cond1, cond2, cond3, cond4, cond5, cond6]
        score = sum(cond.astype(int) for cond in conditions)
        jitter = self.rng.normal(0, self.config.stochastic_jitter, size=len(score))
        score = score + jitter
        buy = (score >= 2).astype(int)
        return buy, pd.Series(score, index=df.index)

    def _compute_sell_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = df["close"]
        volume = df["volume"]
        sma20 = features.get("sma_20", close.rolling(20).mean())
        sma50 = features.get("sma_50", close.rolling(50).mean())
        rsi = features.get("rsi", pd.Series(np.nan, index=df.index))
        bb_mid = (features["bb_upper"] + features["bb_lower"]) / 2
        ma20_slope = features.get("sma_slope_20", sma20.diff())
        macd_hist = features.get("macd_hist", pd.Series(np.nan, index=df.index))
        macd_hist_slope = features.get("macd_hist_slope", macd_hist.diff())
        minus_di = features.get("-di", pd.Series(np.nan, index=df.index))
        plus_di = features.get("+di", pd.Series(np.nan, index=df.index))
        position_in_range = features.get("position_in_range", pd.Series(np.nan, index=df.index))
        vol_mean20 = volume.rolling(20).mean()

        cond1 = (close < sma20) & (sma20 < sma50)
        cond2 = (rsi < 50) | ((rsi.diff() < 0) & (rsi.shift() > 70) & (rsi < 70))
        cond3 = (close < bb_mid) & (ma20_slope < 0)
        cond4 = (volume > self.config.volume_spike_multiplier * (vol_mean20 + 1e-9)) & (position_in_range <= 0.3)
        cond5 = (macd_hist < 0) & (macd_hist_slope < 0)
        cond6 = minus_di > plus_di

        conditions = [cond1, cond2, cond3, cond4, cond5, cond6]
        score = sum(cond.astype(int) for cond in conditions)
        jitter = self.rng.normal(0, self.config.stochastic_jitter, size=len(score))
        score = score + jitter
        sell = (score >= 2).astype(int)
        return sell, pd.Series(score, index=df.index)

    def _balance(self, series: pd.Series, score: pd.Series, label_name: str) -> pd.Series:
        """Adjust class distribution toward target using stochastic down/up sampling."""
        series = series.copy().astype(int)
        positives = series[series == 1].index
        negatives = series[series == 0].index

        frac = series.mean()
        if frac > self.config.max_class_fraction and len(positives) > 0:
            target_count = int(self.config.balance_target * len(series))
            drop_count = max(len(positives) - target_count, 0)
            if drop_count > 0:
                drop_idx = self.rng.choice(positives, size=drop_count, replace=False)
                series.loc[drop_idx] = 0
        elif frac < self.config.min_class_fraction and len(negatives) > 0:
            deficit = int(self.config.balance_target * len(series)) - len(positives)
            if deficit > 0:
                # Promote highest scores among negatives
                candidate_scores = score.loc[negatives]
                top_idx = candidate_scores.sort_values(ascending=False).head(deficit).index
                series.loc[top_idx] = 1
        return series

    def _compute_direction(self, close: pd.Series) -> pd.Series:
        future_close = close.shift(-3)
        up = future_close > (close + self.config.neutral_buffer)
        down = future_close < (close - self.config.neutral_buffer)
        direction = pd.Series(2, index=close.index)
        direction[up] = 1
        direction[down] = 0
        direction = direction.fillna(2)
        return direction.astype(int)

    def _compute_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        adx = features.get("adx", close * 0)
        atr = features.get("atr", close * 0)
        trend = close.pct_change(self.config.trend_window)
        atr_threshold = atr.rolling(100).quantile(self.config.atr_percentile).fillna(method="bfill")
        volume_spike = volume > self.config.volume_spike_multiplier * volume.rolling(20).mean()

        regime = pd.Series(2, index=df.index)  # default RANGE
        regime[(trend > 0.02) & (adx > 25)] = 0  # BULL_STRONG
        regime[(trend > 0) & (trend <= 0.02) & (adx <= 25)] = 1  # BULL_WEAK
        regime[(trend < -0.02) & (adx > 25)] = 4  # BEAR_STRONG
        regime[(trend < 0) & (trend >= -0.02) & (adx <= 25)] = 3  # BEAR_WEAK
        volatile = (atr > atr_threshold) | volume_spike
        regime[volatile] = 5

        return regime.astype(int).fillna(2)

    def _print_distribution(self, labels: pd.DataFrame) -> None:
        print("\n=== 📊 Label Distribution Summary ===")
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True, dropna=False)
            print(f"\n▶ {col.upper()} distribution:")
            print(vc.round(3))
