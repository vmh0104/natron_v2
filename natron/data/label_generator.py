"""Label generation logic for Natron multi-task outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from natron.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class LabelGeneratorConfig:
    neutral_buffer: float = 0.001
    buy_target_ratio: float = 0.35
    sell_target_ratio: float = 0.35
    random_state: int = 42
    stochastic_std: float = 0.02


class LabelGeneratorV2:
    """Generate buy/sell, direction, and regime labels."""

    def __init__(self, config: LabelGeneratorConfig | None = None) -> None:
        self.config = config or LabelGeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_state)

    def transform(self, features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute labels based on engineered features and base OHLCV data."""
        required_price_cols = {"open", "high", "low", "close", "volume"}
        if not required_price_cols.issubset(ohlcv.columns):
            raise ValueError(f"OHLCV dataframe must contain columns: {required_price_cols}")

        ohlcv_df = ohlcv.copy()
        if "time" in ohlcv_df.columns:
            ohlcv_df = ohlcv_df.sort_values("time")
            ohlcv_df = ohlcv_df.set_index("time")
        ohlcv_df = ohlcv_df.loc[features.index]

        missing_features = self._validate_required_features(features)
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")

        close = ohlcv_df["close"]
        high = ohlcv_df["high"]
        low = ohlcv_df["low"]
        volume = ohlcv_df["volume"]

        jitter = pd.Series(
            self.rng.normal(0, self.config.stochastic_std, len(features)),
            index=features.index,
        )
        volume_threshold = 1.5 * (1 + jitter.clip(-0.2, 0.2))
        high_position_threshold = (0.7 + jitter.clip(-0.1, 0.1)).clip(0.6, 0.85)
        low_position_threshold = (0.3 + jitter.clip(-0.1, 0.1)).clip(0.15, 0.4)

        buy_score, sell_score = self._compute_signal_scores(
            features,
            close=close,
            volume_ratio_threshold=volume_threshold,
            high_position_threshold=high_position_threshold,
            low_position_threshold=low_position_threshold,
        )
        buy = (buy_score >= 2).astype(int)
        sell = (sell_score >= 2).astype(int)

        buy, sell = self._resolve_signal_conflicts(buy, sell, buy_score, sell_score)
        buy = self._balance_binary_labels(buy, self.config.buy_target_ratio)
        sell = self._balance_binary_labels(sell, self.config.sell_target_ratio)

        direction = self._compute_directional_labels(close)
        regime = self._compute_regime_labels(features, close)

        labels = pd.DataFrame(
            {
                "buy": buy.astype(int),
                "sell": sell.astype(int),
                "direction": direction.astype(int),
                "regime": regime.astype(int),
            },
            index=features.index,
        )

        self._print_label_summary(labels)
        return labels

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _validate_required_features(self, features: pd.DataFrame) -> set[str]:
        required = {
            "ma_20",
            "ma_50",
            "rsi_14",
            "boll_mid_20",
            "ma_slope_20",
            "volume_ratio_20",
            "position_in_range",
            "macd_hist",
            "plus_di_14",
            "minus_di_14",
            "adx_14",
            "atr_14",
            "rolling_return_20",
        }
        return {col for col in required if col not in features.columns}

    def _compute_signal_scores(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        volume_ratio_threshold: pd.Series,
        high_position_threshold: pd.Series,
        low_position_threshold: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        ma20 = features["ma_20"]
        ma50 = features["ma_50"]
        rsi = features["rsi_14"]
        boll_mid = features["boll_mid_20"]
        ma_slope = features["ma_slope_20"]
        volume_ratio = features["volume_ratio_20"]
        position = features["position_in_range"]
        macd_hist = features["macd_hist"]
        macd_hist_diff = macd_hist.diff()
        plus_di = features["plus_di_14"]
        minus_di = features["minus_di_14"]

        # BUY logic
        buy_conditions = pd.DataFrame(
            {
                "trend_stack": (close > ma20) & (ma20 > ma50),
                "rsi_strength": (rsi > 50) | ((rsi >= 30) & (rsi.shift(1) < 30)),
                "slope_confirmation": (close > boll_mid) & (ma_slope > 0),
                "volume_expansion": volume_ratio > volume_ratio_threshold,
                "range_position": position >= high_position_threshold,
                "macd_positive": (macd_hist > 0) & (macd_hist_diff > 0),
            }
        ).fillna(False)
        buy_score = buy_conditions.sum(axis=1)

        # SELL logic
        sell_conditions = pd.DataFrame(
            {
                "trend_stack": (close < ma20) & (ma20 < ma50),
                "rsi_weakness": (rsi < 50) | ((rsi <= 70) & (rsi.shift(1) > 70)),
                "slope_confirmation": (close < boll_mid) & (ma_slope < 0),
                "volume_pressure": (volume_ratio > volume_ratio_threshold)
                & (position <= low_position_threshold),
                "macd_negative": (macd_hist < 0) & (macd_hist_diff < 0),
                "di_divergence": minus_di > plus_di,
            }
        ).fillna(False)
        sell_score = sell_conditions.sum(axis=1)

        return buy_score, sell_score

    def _resolve_signal_conflicts(
        self,
        buy: pd.Series,
        sell: pd.Series,
        buy_score: pd.Series,
        sell_score: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        conflict_mask = (buy == 1) & (sell == 1)
        if conflict_mask.any():
            buy_higher = buy_score > sell_score
            sell_higher = sell_score > buy_score
            buy.loc[conflict_mask] = buy_higher.loc[conflict_mask].astype(int)
            sell.loc[conflict_mask] = sell_higher.loc[conflict_mask].astype(int)
            tie_mask = conflict_mask & (buy_score == sell_score)
            buy.loc[tie_mask] = 0
            sell.loc[tie_mask] = 0
        return buy, sell

    def _balance_binary_labels(self, labels: pd.Series, target_ratio: float) -> pd.Series:
        if labels.sum() == 0:
            return labels
        positive_ratio = labels.mean()
        if positive_ratio <= target_ratio or positive_ratio == 0:
            return labels
        drop_prob = 1 - target_ratio / positive_ratio
        random_values = self.rng.random(len(labels))
        mask = (labels == 1) & (random_values < drop_prob)
        balanced = labels.copy()
        balanced.loc[mask] = 0
        return balanced

    def _compute_directional_labels(self, close: pd.Series) -> pd.Series:
        neutral_buffer = self.config.neutral_buffer
        future_close = close.shift(-3)
        direction = pd.Series(2, index=close.index, dtype="int64")
        delta = future_close - close
        direction.loc[delta > neutral_buffer] = 1
        direction.loc[delta < -neutral_buffer] = 0
        direction = direction.fillna(2)
        return direction

    def _compute_regime_labels(
        self,
        features: pd.DataFrame,
        close: pd.Series,
    ) -> pd.Series:
        trend_pct = features["rolling_return_20"] * 100
        adx = features["adx_14"]
        atr = features["atr_14"]
        volume_ratio = features["volume_ratio_20"]
        atr_threshold = float(atr.quantile(0.9))
        if not np.isfinite(atr_threshold) or atr_threshold <= 0:
            atr_threshold = float(atr.median())
        volume_spike = volume_ratio > 2.5

        regime = pd.Series(2, index=features.index, dtype="int64")
        regime.loc[(trend_pct > 2) & (adx > 25)] = 0  # BULL_STRONG
        regime.loc[(trend_pct > 0) & (trend_pct <= 2) & (adx <= 25)] = 1  # BULL_WEAK
        regime.loc[(trend_pct < 0) & (trend_pct >= -2) & (adx <= 25)] = 3  # BEAR_WEAK
        regime.loc[(trend_pct < -2) & (adx > 25)] = 4  # BEAR_STRONG

        volatile_mask = (atr > atr_threshold) | volume_spike
        regime.loc[volatile_mask] = 5  # VOLATILE
        regime = regime.fillna(2)
        return regime

    def _print_label_summary(self, labels: pd.DataFrame) -> None:
        print("\n=== 📊 Label Distribution Summary ===")
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True)
            print(f"\n▶ {col.upper()} distribution:")
            print(vc.round(3))


__all__ = ["LabelGeneratorV2", "LabelGeneratorConfig"]
