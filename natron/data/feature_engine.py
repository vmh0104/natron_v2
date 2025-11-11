from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureEngineConfig:
    rolling_windows: Sequence[int] = (5, 10, 14, 20, 30, 50, 96)
    volatility_windows: Sequence[int] = (10, 20, 30)
    momentum_windows: Sequence[int] = (5, 10, 14, 20, 30)
    volume_windows: Sequence[int] = (10, 20, 30)
    market_profile_windows: Sequence[int] = (24, 48, 96)
    dropna: bool = True
    fill_method: str | None = "bfill"
    eps: float = 1e-8
    stochastic_noise: float = 0.02
    dtype: str = "float32"


class FeatureEngine:
    """Generate a comprehensive technical feature set (~100 features).

    The implementation focuses on deterministic, vectorized pandas operations
    to keep the pipeline GPU-ready once tensors are created downstream.
    """

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        self.cfg = config or FeatureEngineConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        df = df.copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        feats: Dict[str, pd.Series] = {}

        self._add_moving_averages(close, feats)
        self._add_momentum(close, high, low, volume, feats)
        self._add_volatility(open_, high, low, close, feats)
        self._add_volume(volume, close, feats)
        self._add_price_pattern(open_, high, low, close, feats)
        self._add_returns(open_, close, feats)
        self._add_trend_strength(high, low, close, feats)
        self._add_statistical(close, feats)
        self._add_support_resistance(high, low, close, feats)
        self._add_smc(close, high, low, feats)
        self._add_market_profile(close, volume, feats)

        features = pd.DataFrame(feats)
        if self.cfg.fill_method:
            features = features.fillna(method=self.cfg.fill_method).fillna(0.0)
        elif self.cfg.dropna:
            features = features.dropna()
        features = features.astype(self.cfg.dtype)
        return features

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {"time", "open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns for feature extraction: {sorted(missing)}")

    # --- Feature group helpers -------------------------------------------------

    def _add_moving_averages(self, close: pd.Series, feats: Dict[str, pd.Series]) -> None:
        windows = self.cfg.rolling_windows
        for window in windows:
            ma = close.rolling(window).mean()
            ema = close.ewm(span=window, adjust=False).mean()
            feats[f"ma_{window}"] = ma
            feats[f"ema_{window}"] = ema
            feats[f"ma_slope_{window}"] = ma.diff()
            feats[f"ema_slope_{window}"] = ema.diff()
            feats[f"price_ma_ratio_{window}"] = close / (ma + self.cfg.eps)
            feats[f"ema_zscore_{window}"] = (ema - close.rolling(window).mean()) / (
                close.rolling(window).std() + self.cfg.eps
            )

        for fast, slow in self._pairwise(windows):
            if fast >= slow:
                continue
            feats[f"ma_diff_{fast}_{slow}"] = (
                close.rolling(fast).mean() - close.rolling(slow).mean()
            )
            feats[f"ema_ratio_{fast}_{slow}"] = (
                close.ewm(span=fast, adjust=False).mean()
                / (close.ewm(span=slow, adjust=False).mean() + self.cfg.eps)
            )

    def _add_momentum(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        for window in self.cfg.momentum_windows:
            roc = close.pct_change(periods=window)
            feats[f"roc_{window}"] = roc
            feats[f"rsi_{window}"] = self._rsi(close, window)
            feats[f"cci_{window}"] = self._cci(high, low, close, window)
            feats[f"stoch_k_{window}"] = self._stochastic_k(high, low, close, window)
            feats[f"stoch_d_{window}"] = (
                feats[f"stoch_k_{window}"]
                .rolling(3)
                .mean()
            )
            feats[f"momentum_{window}"] = close.diff(window)

        macd_fast, macd_slow, signal = 12, 26, 9
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
        feats["macd"] = macd
        feats["macd_signal"] = signal_line
        feats["macd_hist"] = macd_hist
        feats["macd_hist_slope"] = macd_hist.diff()

    def _add_volatility(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        true_range = self._true_range(high, low, close)
        for window in self.cfg.volatility_windows:
            atr = true_range.rolling(window).mean()
            feats[f"atr_{window}"] = atr
            feats[f"atr_pct_{window}"] = atr / (close + self.cfg.eps)
            std = close.rolling(window).std()
            feats[f"volatility_{window}"] = std
            feats[f"bband_mid_{window}"] = close.rolling(window).mean()
            feats[f"bband_upper_{window}"] = feats[f"bband_mid_{window}"] + 2 * std
            feats[f"bband_lower_{window}"] = feats[f"bband_mid_{window}"] - 2 * std
            feats[f"bband_width_{window}"] = (
                feats[f"bband_upper_{window}"] - feats[f"bband_lower_{window}"]
            ) / (feats[f"bband_mid_{window}"] + self.cfg.eps)
            feats[f"bband_percent_b_{window}"] = (
                (close - feats[f"bband_lower_{window}"]) /
                (feats[f"bband_upper_{window}"] - feats[f"bband_lower_{window}"] + self.cfg.eps)
            )

        ema_10 = close.ewm(span=10, adjust=False).mean()
        true_range_ema = true_range.ewm(span=10, adjust=False).mean()
        feats["keltner_upper"] = ema_10 + 2 * true_range_ema
        feats["keltner_lower"] = ema_10 - 2 * true_range_ema
        feats["keltner_width"] = (
            feats["keltner_upper"] - feats["keltner_lower"]
        ) / (ema_10 + self.cfg.eps)

    def _add_volume(self, volume: pd.Series, close: pd.Series, feats: Dict[str, pd.Series]) -> None:
        for window in self.cfg.volume_windows:
            feats[f"volume_ma_{window}"] = volume.rolling(window).mean()
            feats[f"volume_ratio_{window}"] = volume / (feats[f"volume_ma_{window}"] + self.cfg.eps)
            feats[f"volume_zscore_{window}"] = (
                (volume - feats[f"volume_ma_{window}"]) /
                (volume.rolling(window).std() + self.cfg.eps)
            )

        price_change = close.diff().fillna(0.0)
        direction = np.sign(price_change)
        feats["obv"] = (volume * direction).cumsum()
        feats["vwap"] = (close * volume).cumsum() / (volume.cumsum() + self.cfg.eps)

        # Money Flow Index
        typical_price = (close + close.shift(1) + close.shift(2)) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(price_change > 0, 0.0)
        negative_flow = raw_money_flow.where(price_change < 0, 0.0).abs()
        money_ratio = (
            positive_flow.rolling(14).sum()
            / (negative_flow.rolling(14).sum() + self.cfg.eps)
        )
        feats["mfi_14"] = 100 - (100 / (1 + money_ratio))

    def _add_price_pattern(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        body = (close - open_).abs()
        range_ = (high - low).replace(0, np.nan)
        upper_shadow = high - close.where(close >= open_, open_)
        lower_shadow = open_.where(close >= open_, close) - low

        feats["candle_body_pct"] = body / (range_ + self.cfg.eps)
        feats["upper_shadow_pct"] = upper_shadow / (range_ + self.cfg.eps)
        feats["lower_shadow_pct"] = lower_shadow / (range_ + self.cfg.eps)
        feats["candle_range"] = range_
        feats["gap_up"] = (open_ - close.shift(1)).clip(lower=0)
        feats["gap_down"] = (close.shift(1) - open_).clip(lower=0)
        feats["position_in_range_20"] = (close - low.rolling(20).min()) / (
            (high.rolling(20).max() - low.rolling(20).min()) + self.cfg.eps
        )
        feats["doji"] = (feats["candle_body_pct"] < 0.1).astype(float)
        feats["engulfing"] = (((close > open_) & (open_ <= close.shift(1)) & (close >= open_.shift(1))) | (
            (close < open_) & (open_ >= close.shift(1)) & (close <= open_.shift(1))
        )).astype(float)

    def _add_returns(self, open_: pd.Series, close: pd.Series, feats: Dict[str, pd.Series]) -> None:
        feats["log_return_1"] = np.log(close / close.shift(1).replace(0, np.nan))
        feats["log_return_5"] = np.log(close / close.shift(5).replace(0, np.nan))
        feats["intraday_return"] = np.log(close / open_.replace(0, np.nan))
        feats["cumulative_return_20"] = feats["log_return_1"].rolling(20).sum()
        feats["volatility_ratio_5_20"] = (
            close.pct_change().rolling(5).std()
            / (close.pct_change().rolling(20).std() + self.cfg.eps)
        )
        feats["drawdown_20"] = (
            close / (close.rolling(20).max() + self.cfg.eps) - 1
        )
        feats["return_skew_20"] = close.pct_change().rolling(20).skew()
        feats["return_kurt_20"] = close.pct_change().rolling(20).kurt()

    def _add_trend_strength(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        adx_window = 14
        plus_dm = ((high - high.shift(1)).clip(lower=0)).fillna(0.0)
        minus_dm = ((low.shift(1) - low).clip(lower=0)).fillna(0.0)
        tr = self._true_range(high, low, close)
        atr = tr.ewm(alpha=1 / adx_window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / adx_window, adjust=False).mean() / (atr + self.cfg.eps))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / adx_window, adjust=False).mean() / (atr + self.cfg.eps))
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + self.cfg.eps))
        adx = dx.ewm(alpha=1 / adx_window, adjust=False).mean()
        feats["adx_14"] = adx
        feats["plus_di_14"] = plus_di
        feats["minus_di_14"] = minus_di
        feats["di_diff"] = plus_di - minus_di
        feats["trend_strength"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + self.cfg.eps)
        feats["aroon_up_25"], feats["aroon_down_25"] = self._aroon(high, low, 25)

    def _add_statistical(self, close: pd.Series, feats: Dict[str, pd.Series]) -> None:
        for window in (20, 50, 96):
            rolling = close.rolling(window)
            feats[f"zscore_{window}"] = (close - rolling.mean()) / (rolling.std() + self.cfg.eps)
            feats[f"rolling_skew_{window}"] = rolling.skew()
            feats[f"rolling_kurt_{window}"] = rolling.kurt()
            feats[f"hurst_{window}"] = rolling.apply(self._hurst_exponent, raw=False)

    def _add_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        for window in (20, 30, 50):
            rolling_max = high.rolling(window).max()
            rolling_min = low.rolling(window).min()
            feats[f"dist_to_high_{window}"] = (rolling_max - close) / (rolling_max + self.cfg.eps)
            feats[f"dist_to_low_{window}"] = (close - rolling_min) / (rolling_min + self.cfg.eps)
            feats[f"support_resistance_ratio_{window}"] = (
                feats[f"dist_to_low_{window}"] / (feats[f"dist_to_high_{window}"] + self.cfg.eps)
            )

    def _add_smc(self, close: pd.Series, high: pd.Series, low: pd.Series, feats: Dict[str, pd.Series]) -> None:
        swing_high = (high.shift(1) < high) & (high.shift(-1) < high)
        swing_low = (low.shift(1) > low) & (low.shift(-1) > low)
        feats["swing_high"] = swing_high.astype(float)
        feats["swing_low"] = swing_low.astype(float)
        feats["swing_high_distance"] = (
            close - close.where(swing_high).ffill()
        ) / (close + self.cfg.eps)
        feats["swing_low_distance"] = (
            close - close.where(swing_low).ffill()
        ) / (close + self.cfg.eps)

        # Break of structure (BOS) and Change of Character (CHoCH) proxies
        hh = high.rolling(20).max()
        ll = low.rolling(20).min()
        feats["bos"] = ((close > hh.shift(1)) | (close < ll.shift(1))).astype(float)
        feats["choch"] = (
            ((close > hh.shift(1)) & (close < close.shift(1)))
            | ((close < ll.shift(1)) & (close > close.shift(1)))
        ).astype(float)

    def _add_market_profile(
        self,
        close: pd.Series,
        volume: pd.Series,
        feats: Dict[str, pd.Series],
    ) -> None:
        for window in self.cfg.market_profile_windows:
            price_window = close.rolling(window)
            vol_window = volume.rolling(window)
            mean_price = price_window.mean()
            std_price = price_window.std().fillna(0.0)
            feats[f"poc_{window}"] = price_window.apply(self._poc, raw=True)
            feats[f"vah_{window}"] = mean_price + std_price
            feats[f"val_{window}"] = mean_price - std_price
            feats[f"price_entropy_{window}"] = price_window.apply(self._price_entropy, raw=True)
            feats[f"volume_entropy_{window}"] = vol_window.apply(self._volume_entropy, raw=True)
            feats[f"volume_balance_{window}"] = (
                vol_window.apply(lambda x: x[int(len(x) / 2) :].sum() - x[: int(len(x) / 2)].sum(), raw=True)
            )
            feats[f"mean_reversion_score_{window}"] = (
                close - mean_price
            ) / (std_price + self.cfg.eps)

    # --- Static helper methods -------------------------------------------------

    @staticmethod
    def _pairwise(seq: Sequence[int]) -> Iterable[tuple[int, int]]:
        for i, a in enumerate(seq):
            for b in seq[i + 1 :]:
                yield a, b

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        tp = (high + low + close) / 3
        sma = tp.rolling(window).mean()
        mad = (tp - sma).abs().rolling(window).mean()
        return (tp - sma) / (0.015 * mad + 1e-9)

    @staticmethod
    def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        return 100 * (close - lowest_low) / ((highest_high - lowest_low) + 1e-9)

    @staticmethod
    def _aroon(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        aroon_up = high.rolling(window + 1).apply(lambda x: np.argmax(x[::-1]) / window * 100, raw=True)
        aroon_down = low.rolling(window + 1).apply(lambda x: np.argmax(x) / window * 100, raw=True)
        return aroon_up, aroon_down

    @staticmethod
    def _hurst_exponent(series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 20:
            return np.nan
        lags = range(2, min(20, len(series)))
        tau = [np.sqrt((series.diff(lag) ** 2).mean()) for lag in lags]
        if any(t <= 0 for t in tau):
            return np.nan
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        return slope * 2

    @staticmethod
    def _poc(values: np.ndarray) -> float:
        if len(values) == 0 or np.all(np.isnan(values)):
            return np.nan
        hist, bin_edges = np.histogram(values[~np.isnan(values)], bins=10)
        idx = hist.argmax()
        return float((bin_edges[idx] + bin_edges[idx + 1]) / 2)

    @staticmethod
    def _price_entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if len(clean) == 0:
            return np.nan
        hist, _ = np.histogram(clean, bins=10, density=True)
        hist = hist + 1e-9
        return float(-(hist * np.log(hist)).sum())

    @staticmethod
    def _volume_entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if len(clean) == 0:
            return np.nan
        probs = clean / (clean.sum() + 1e-9)
        probs = probs + 1e-9
        return float(-(probs * np.log(probs)).sum())
