from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(slots=True)
class LabelConfig:
    neutral_buffer: float = 0.001
    buy_sell_balance_target: float = 0.35
    class_balance_eps: float = 0.05
    stochastic_jitter: float = 0.03
    seed: int = 42


class LabelGeneratorV2:
    def __init__(self, config: LabelConfig | None = None) -> None:
        self.cfg = config or LabelConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def generate(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        self._validate(df, features)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        buy = self._compute_buy_signals(df, features)
        sell = self._compute_sell_signals(df, features)
        direction = self._compute_direction(close)
        regime = self._compute_regime(df, features)

        labels = pd.DataFrame({
            "buy": buy.astype(int),
            "sell": sell.astype(int),
            "direction": direction.astype(int),
            "regime": regime.astype(int),
        })

        labels = self._balance_buy_sell(labels)
        self._print_label_summary(labels)
        return labels

    def _compute_buy_signals(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        ma20 = feats.get("ma_20", close.rolling(20).mean())
        ma50 = feats.get("ma_50", close.rolling(50).mean())
        ema20 = feats.get("ema_20", close.ewm(span=20, adjust=False).mean())
        rsi = feats.get("rsi_14", self._rsi(close))
        bb_mid = feats.get("bband_mid_20", close.rolling(20).mean())
        ma20_slope = feats.get("ma_slope_20", ma20.diff())
        macd_hist = feats.get("macd_hist", self._macd(close)[2])
        macd_hist_slope = feats.get("macd_hist_slope", macd_hist.diff())
        volume_ma20 = feats.get("volume_ma_20", volume.rolling(20).mean())
        pos_in_range = feats.get("position_in_range_20", self._position_in_range(df))

        jitter = 1 + self.rng.uniform(-self.cfg.stochastic_jitter, self.cfg.stochastic_jitter)

        conds = [
            close > ma20 * jitter,
            ma20 > ma50,
            (rsi > 50) | ((rsi > 30) & (rsi.shift(1) <= 30)),
            (close > bb_mid) & (ma20_slope > 0),
            volume > (1.5 * jitter * (volume_ma20 + 1e-9)),
            pos_in_range >= 0.7 * jitter,
            (macd_hist > 0) & (macd_hist_slope > 0),
        ]
        signals = sum(conds) >= 2
        return signals.astype(int)

    def _compute_sell_signals(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        ma20 = feats.get("ma_20", close.rolling(20).mean())
        ma50 = feats.get("ma_50", close.rolling(50).mean())
        rsi = feats.get("rsi_14", self._rsi(close))
        bb_mid = feats.get("bband_mid_20", close.rolling(20).mean())
        ma20_slope = feats.get("ma_slope_20", ma20.diff())
        macd_hist = feats.get("macd_hist", self._macd(close)[2])
        macd_hist_slope = feats.get("macd_hist_slope", macd_hist.diff())
        plus_di = feats.get("plus_di_14", self._plus_di(df))
        minus_di = feats.get("minus_di_14", self._minus_di(df))
        volume_ma20 = feats.get("volume_ma_20", volume.rolling(20).mean())
        pos_in_range = feats.get("position_in_range_20", self._position_in_range(df))

        jitter = 1 + self.rng.uniform(-self.cfg.stochastic_jitter, self.cfg.stochastic_jitter)

        conds = [
            close < ma20 * jitter,
            ma20 < ma50,
            (rsi < 50) | ((rsi < 70) & (rsi.shift(1) >= 70)),
            (close < bb_mid) & (ma20_slope < 0),
            (volume > (1.5 * jitter * (volume_ma20 + 1e-9))) & (pos_in_range <= 0.3 * jitter),
            (macd_hist < 0) & (macd_hist_slope < 0),
            minus_di > plus_di,
        ]
        signals = sum(conds) >= 2
        return signals.astype(int)

    def _compute_direction(self, close: pd.Series) -> pd.Series:
        future = close.shift(-3)
        diff = future - close
        up = diff > self.cfg.neutral_buffer * close
        down = diff < -self.cfg.neutral_buffer * close
        direction = pd.Series(np.where(up, 1, np.where(down, 0, 2)), index=close.index)
        return direction

    def _compute_regime(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend = close.pct_change(periods=20) * 100
        adx = feats.get("adx_14", self._adx(df))
        atr = feats.get("atr_20", self._atr(df))
        volume = df["volume"]
        volume_z = feats.get("volume_zscore_20", (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9))

        atr_threshold = atr.rolling(100).quantile(0.9).fillna(atr.quantile(0.9))
        volatile = (atr > atr_threshold) | (volume_z > 2.5)

        regime = pd.Series(2, index=close.index)  # default RANGE
        regime = regime.mask((trend > 2) & (adx > 25), 0)  # BULL_STRONG
        regime = regime.mask((trend > 0) & (trend <= 2) & (adx <= 25), 1)  # BULL_WEAK
        regime = regime.mask((trend < 0) & (trend >= -2) & (adx <= 25), 3)  # BEAR_WEAK
        regime = regime.mask((trend < -2) & (adx > 25), 4)  # BEAR_STRONG
        regime = regime.mask(volatile, 5)  # VOLATILE overrides
        return regime

    def _balance_buy_sell(self, labels: pd.DataFrame) -> pd.DataFrame:
        target = self.cfg.buy_sell_balance_target
        eps = self.cfg.class_balance_eps
        labels = labels.copy()

        for column in ["buy", "sell"]:
            positive_ratio = labels[column].mean()
            if positive_ratio > target + eps:
                drop_prob = 1 - (target / (positive_ratio + 1e-9))
                mask = self.rng.random(len(labels)) > drop_prob
                labels[column] = labels[column] * mask.astype(int)
            elif positive_ratio < target - eps:
                # Slightly relax by flipping some zeros to ones based on confidence proxies
                deficit = (target - positive_ratio)
                flip_prob = deficit
                flips = (self.rng.random(len(labels)) < flip_prob).astype(int)
                labels[column] = labels[column] | flips
        return labels

    def _print_label_summary(self, labels: pd.DataFrame) -> None:
        print("\n=== 📊 Label Distribution Summary ===")
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True, dropna=False)
            print(f"\n▶ {col.upper()} distribution:")
            print(vc.round(3))

    def _validate(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        if len(df) != len(feats):
            raise ValueError("DataFrame and features must have the same number of rows")

    # --- indicator fallbacks ---------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def _position_in_range(df: pd.DataFrame, window: int = 20) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        return (close - low.rolling(window).min()) / (
            (high.rolling(window).max() - low.rolling(window).min()) + 1e-9
        )

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr

    def _atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return self._true_range(df).rolling(window).mean()

    def _adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        tr = self._true_range(df)
        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(window).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(window).mean()
        return adx

    def _plus_di(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["high"]
        plus_dm = (high - high.shift(1)).clip(lower=0)
        atr = self._true_range(df).rolling(window).mean()
        return 100 * (plus_dm.rolling(window).mean() / (atr + 1e-9))

    def _minus_di(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        low = df["low"]
        minus_dm = (low.shift(1) - low).clip(lower=0)
        atr = self._true_range(df).rolling(window).mean()
        return 100 * (minus_dm.rolling(window).mean() / (atr + 1e-9))
