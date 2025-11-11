from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from natron.utils.data_utils import balance_binary_labels

logger = logging.getLogger(__name__)


@dataclass
class LabelGeneratorV2:
    neutral_buffer: float = 0.001
    buy_target_ratio: float = 0.35
    sell_target_ratio: float = 0.35
    random_state: int = 42

    def generate(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        features = features.reset_index(drop=True)

        labels = pd.DataFrame(index=df.index)
        labels["buy_raw"] = self._buy_signal(df, features)
        labels["sell_raw"] = self._sell_signal(df, features)

        labels["buy"] = balance_binary_labels(labels["buy_raw"], self.buy_target_ratio)
        labels["sell"] = balance_binary_labels(labels["sell_raw"], self.sell_target_ratio)

        labels["direction"] = self._direction_label(df["close"])
        labels["regime"] = self._regime_label(df, features)

        labels = labels.drop(columns=["buy_raw", "sell_raw"])
        self._log_distribution(labels)
        return labels

    # --- Signal Generators -------------------------------------------------
    def _buy_signal(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        conds = [
            (close > feats["ma_20"]) & (feats["ma_20"] > feats["ma_50"]),
            (feats["rsi_14"] > 50)
            | ((feats["rsi_14"] > 30) & (feats["rsi_14"].shift(1) < 30)),
            (close > feats["bb_mid"]) & (feats["ma_slope_20"] > 0),
            (volume > 1.5 * volume.rolling(20).mean()),
            self._position_in_range(df) >= 0.7,
            (feats["macd_hist"] > 0) & (feats["macd_hist"].diff() > 0),
        ]
        score = sum(cond.astype(int) for cond in conds)
        noise = np.random.default_rng(self.random_state).uniform(-0.2, 0.2, size=len(score))
        return ((score + noise) >= 2).astype(int)

    def _sell_signal(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        conds = [
            (close < feats["ma_20"]) & (feats["ma_20"] < feats["ma_50"]),
            (feats["rsi_14"] < 50)
            | ((feats["rsi_14"] < 70) & (feats["rsi_14"].shift(1) > 70)),
            (close < feats["bb_mid"]) & (feats["ma_slope_20"] < 0),
            (volume > 1.5 * volume.rolling(20).mean()) & (self._position_in_range(df) <= 0.3),
            (feats["macd_hist"] < 0) & (feats["macd_hist"].diff() < 0),
            (feats["minus_di"] > feats["plus_di"]),
        ]
        score = sum(cond.astype(int) for cond in conds)
        noise = np.random.default_rng(self.random_state + 1).uniform(-0.2, 0.2, size=len(score))
        return ((score + noise) >= 2).astype(int)

    def _direction_label(self, close: pd.Series) -> pd.Series:
        future = close.shift(-3)
        up = (future > close * (1 + self.neutral_buffer)).astype(int)
        down = (future < close * (1 - self.neutral_buffer)).astype(int) * 2  # Temporary scaling

        label = pd.Series(2, index=close.index)  # neutral by default
        label = label.where(~(future.notna()), other=2)
        label = label.where(~(up == 1), other=1)
        label = label.where(~(down == 2), other=0)
        return label.astype(int)

    def _regime_label(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend = close.pct_change(20) * 100
        adx = feats["adx_14"]
        atr = feats["atr_14"]
        volume_ratio = feats["volume_ratio_20"]

        atr_threshold = atr.quantile(0.9)
        volume_threshold = volume_ratio.quantile(0.9)

        regime = pd.Series(2, index=df.index)  # RANGE default
        regime = regime.mask((trend > 2) & (adx > 25), other=0)  # BULL_STRONG
        regime = regime.mask((trend > 0) & (trend <= 2) & (adx <= 25), other=1)  # BULL_WEAK
        regime = regime.mask((trend < 0) & (trend >= -2) & (adx <= 25), other=3)  # BEAR_WEAK
        regime = regime.mask((trend < -2) & (adx > 25), other=4)  # BEAR_STRONG
        volatile = (atr > atr_threshold) | (volume_ratio > volume_threshold)
        regime = regime.mask(volatile, other=5)  # VOLATILE
        return regime.fillna(2).astype(int)

    # --- Helpers ----------------------------------------------------------
    @staticmethod
    def _position_in_range(df: pd.DataFrame) -> pd.Series:
        rolling_high = df["high"].rolling(20).max()
        rolling_low = df["low"].rolling(20).min()
        position = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        return position.clip(0, 1)

    def _log_distribution(self, labels: pd.DataFrame) -> None:
        print("\n=== 📊 Label Distribution Summary ===")
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True)
            print(f"\n▶ {col.upper()} distribution:")
            print(vc.round(3))
            logger.info("%s distribution: %s", col, vc.round(3).to_dict())
