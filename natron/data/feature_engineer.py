"""Feature engineering module producing ~100 technical indicators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from natron.utils.logging import get_logger


logger = get_logger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    def entropy(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        hist, _ = np.histogram(arr, bins=bins)
        if hist.sum() == 0:
            return np.nan
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return float(-(prob * np.log(prob)).sum())

    return series.rolling(window).apply(entropy, raw=True)


def _hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    def hurst(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        length = arr.size
        if length < 10:
            return np.nan
        lags = np.arange(2, min(20, length))
        if lags.size == 0:
            return np.nan
        tau = [np.sqrt(((arr[lag:] - arr[:-lag]) ** 2).mean()) for lag in lags]
        tau = np.array(tau)
        tau = tau[np.isfinite(tau) & (tau > 0)]
        if tau.size == 0:
            return np.nan
        poly = np.polyfit(np.log(lags[: tau.size]), np.log(tau), 1)
        return float(poly[0])

    return series.rolling(window).apply(hurst, raw=True)


@dataclass
class FeatureEngineConfig:
    window_sizes: Dict[str, int] | None = None


class FeatureEngine:
    """Generate engineered features from OHLCV data."""

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        self.config = config or FeatureEngineConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix from OHLCV dataframe."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        price_df = df.copy()
        if "time" in price_df.columns:
            price_df = price_df.sort_values("time")
            price_df = price_df.set_index("time")

        open_ = price_df["open"]
        high = price_df["high"]
        low = price_df["low"]
        close = price_df["close"]
        volume = price_df["volume"]
        typical_price = (high + low + close) / 3

        features: Dict[str, pd.Series] = {}

        # 1. Moving Average features (13)
        features.update(self._moving_average_features(close))
        # 2. Momentum features (13)
        features.update(self._momentum_features(open_, high, low, close, volume))
        # 3. Volatility features (15)
        features.update(self._volatility_features(open_, high, low, close))
        # 4. Volume features (9)
        features.update(self._volume_features(close, high, low, volume, typical_price))
        # 5. Price pattern features (10)
        features.update(self._price_pattern_features(open_, high, low, close))
        # 6. Return features (8)
        features.update(self._return_features(close, open_))
        # 7. Trend strength features (6)
        features.update(self._trend_strength_features(high, low, close))
        # 8. Statistical features (6)
        features.update(self._statistical_features(close, volume))
        # 9. Support & resistance (4)
        features.update(self._support_resistance_features(high, low, close))
        # 10. Smart money concepts (6)
        features.update(self._smart_money_features(open_, high, low, close))
        # 11. Market profile (10)
        features.update(self._market_profile_features(close, volume))

        features_df = pd.DataFrame(features, index=price_df.index)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        if features_df.shape[1] != 100:
            logger.warning("Feature count is %s but expected 100.", features_df.shape[1])

        return features_df

    # ------------------------------------------------------------------ #
    # Feature group implementations
    # ------------------------------------------------------------------ #
    def _moving_average_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma100 = close.rolling(100).mean()
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        ema72 = _ema(close, 72)

        features = {
            "ma_5": ma5,
            "ma_10": ma10,
            "ma_20": ma20,
            "ma_50": ma50,
            "ma_100": ma100,
            "ema_12": ema12,
            "ema_26": ema26,
            "ema_72": ema72,
            "ma_ratio_20": _safe_divide(close, ma20),
            "ma_ratio_50": _safe_divide(close, ma50),
            "ma_slope_20": ma20.diff(),
            "ma_slope_50": ma50.diff(),
            "ema_crossover": ema12 - ema26,
        }
        return features

    def _momentum_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, pd.Series]:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = _safe_divide(avg_gain, avg_loss)
        rsi14 = 100 - (100 / (1 + rs))

        avg_gain21 = gain.rolling(21).mean()
        avg_loss21 = loss.rolling(21).mean()
        rs21 = _safe_divide(avg_gain21, avg_loss21)
        rsi21 = 100 - (100 / (1 + rs21))

        roc5 = close.pct_change(5)
        roc10 = close.pct_change(10)

        typical_price = (high + low + close) / 3
        sma_tp14 = typical_price.rolling(14).mean()
        mad_tp14 = (typical_price - sma_tp14).abs().rolling(14).mean()
        cci14 = _safe_divide(typical_price - sma_tp14, 0.015 * mad_tp14)
        sma_tp20 = typical_price.rolling(20).mean()
        mad_tp20 = (typical_price - sma_tp20).abs().rolling(20).mean()
        cci20 = _safe_divide(typical_price - sma_tp20, 0.015 * mad_tp20)

        lowest14 = low.rolling(14).min()
        highest14 = high.rolling(14).max()
        stoch_k = _safe_divide(close - lowest14, highest14 - lowest14) * 100
        stoch_d = stoch_k.rolling(3).mean()

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _ema(macd_line, 9)
        macd_hist = macd_line - macd_signal

        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        williams_r = -100 * _safe_divide(highest_high - close, highest_high - lowest_low)

        momentum14 = close - close.shift(14)

        features = {
            "rsi_14": rsi14,
            "rsi_21": rsi21,
            "roc_5": roc5,
            "roc_10": roc10,
            "cci_14": cci14,
            "cci_20": cci20,
            "stoch_k_14": stoch_k,
            "stoch_d_14": stoch_d,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "williams_r_14": williams_r,
            "momentum_14": momentum14,
        }
        return features

    def _volatility_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr14 = _ema(true_range, 14)
        atr21 = _ema(true_range, 21)

        boll_mid = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std
        boll_width = _safe_divide(boll_upper - boll_lower, boll_mid)

        typical_price = (high + low + close) / 3
        ema_tp = _ema(typical_price, 20)
        atr = atr14
        keltner_upper = ema_tp + 2 * atr
        keltner_lower = ema_tp - 2 * atr
        keltner_range = keltner_upper - keltner_lower

        rolling_std10 = close.rolling(10).std()
        rolling_std20 = close.rolling(20).std()
        rolling_std50 = close.rolling(50).std()

        parkinson_vol = ((np.log(high / low)) ** 2).rolling(20).mean() * (1.0 / (4 * np.log(2)))
        garman = (
            0.5 * (np.log(high / low) ** 2)
            - (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)
        ).rolling(20).mean()

        features = {
            "atr_14": atr14,
            "atr_21": atr21,
            "true_range": true_range,
            "boll_mid_20": boll_mid,
            "boll_upper_20": boll_upper,
            "boll_lower_20": boll_lower,
            "boll_width_20": boll_width,
            "keltner_upper_20": keltner_upper,
            "keltner_lower_20": keltner_lower,
            "keltner_range_20": keltner_range,
            "rolling_std_10": rolling_std10,
            "rolling_std_20": rolling_std20,
            "rolling_std_50": rolling_std50,
            "parkinson_vol_20": parkinson_vol,
            "garman_klass_vol_20": garman,
        }
        return features

    def _volume_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        typical_price: pd.Series,
    ) -> Dict[str, pd.Series]:
        obv = (np.sign(close.diff().fillna(0)) * volume).fillna(0).cumsum()
        obv_slope = obv.diff(10)
        vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        volume_ema20 = _ema(volume, 20)
        volume_ratio20 = _safe_divide(volume, volume.rolling(20).mean())
        volume_std20 = volume.rolling(20).std()
        volume_zscore = _safe_divide(volume - volume.rolling(20).mean(), volume_std20)

        # Money Flow Index
        raw_money_flow = typical_price * volume
        positive_mf = raw_money_flow.where(typical_price.diff() > 0, 0.0)
        negative_mf = raw_money_flow.where(typical_price.diff() < 0, 0.0)
        mfr = _safe_divide(positive_mf.rolling(14).sum(), negative_mf.rolling(14).sum())
        mfi14 = 100 - (100 / (1 + mfr))

        pct_change_volume5 = volume.pct_change(5)
        acc_dist = (
            ((close - low) - (high - close)) / (high - low + 1e-9) * volume
        ).fillna(0).cumsum()

        features = {
            "obv": obv,
            "obv_slope_10": obv_slope,
            "vwap": vwap,
            "volume_ema_20": volume_ema20,
            "volume_ratio_20": volume_ratio20,
            "volume_zscore_20": volume_zscore,
            "mfi_14": mfi14,
            "volume_trend_5": pct_change_volume5,
            "acc_dist": acc_dist,
        }
        return features

    def _price_pattern_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        body = close - open_
        total_range = high - low
        upper_shadow = high - close.where(close >= open_, open_)
        lower_shadow = open_.where(close >= open_, close) - low
        body_pct = _safe_divide(body.abs(), total_range)
        range_ratio = _safe_divide(total_range, close)
        gap_up = open_ - close.shift()
        gap_down = close.shift() - open_
        position_in_range = _safe_divide(close - low, total_range)
        engulfing = (
            ((body > 0) & (close.shift(1) < open_.shift(1)) & (close > open_.shift(1)) & (open_ < close.shift(1)))
            | ((body < 0) & (close.shift(1) > open_.shift(1)) & (close < open_.shift(1)) & (open_ > close.shift(1)))
        ).astype(float)

        hammer = ((lower_shadow > 2 * body.abs()) & (upper_shadow < body.abs())).astype(float)

        features = {
            "candle_body": body,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "body_pct": body_pct,
            "range_ratio": range_ratio,
            "gap_up": gap_up,
            "gap_down": gap_down,
            "position_in_range": position_in_range,
            "pattern_engulfing": engulfing,
            "pattern_hammer": hammer,
        }
        return features

    def _return_features(self, close: pd.Series, open_: pd.Series) -> Dict[str, pd.Series]:
        log_return1 = np.log(_safe_divide(close, close.shift()))
        log_return5 = np.log(_safe_divide(close, close.shift(5)))
        log_return10 = np.log(_safe_divide(close, close.shift(10)))
        pct_change1 = close.pct_change()
        pct_change5 = close.pct_change(5)
        intraday = _safe_divide(close - open_, open_)
        rolling_return20 = _safe_divide(close, close.shift(20)) - 1
        cumulative_return = (1 + close.pct_change().fillna(0)).cumprod() - 1

        features = {
            "log_return_1": log_return1,
            "log_return_5": log_return5,
            "log_return_10": log_return10,
            "pct_change_1": pct_change1,
            "pct_change_5": pct_change5,
            "intraday_return": intraday,
            "rolling_return_20": rolling_return20,
            "cum_return": cumulative_return,
        }
        return features

    def _trend_strength_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        period = 14
        up_move = high.diff()
        down_move = low.shift() - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        atr = _ema(tr, period)

        plus_di = 100 * _safe_divide(_ema(plus_dm, period), atr)
        minus_di = 100 * _safe_divide(_ema(minus_dm, period), atr)
        dx = 100 * _safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
        adx = _ema(dx, period)
        adx_slope = adx.diff()

        def _aroon(series: pd.Series, window: int, mode: str) -> pd.Series:
            def calc(arr: np.ndarray) -> float:
                if np.isnan(arr).all():
                    return np.nan
                idx = np.argmax(arr) if mode == "up" else np.argmin(arr)
                return 100 * (idx + 1) / len(arr)

            return series.rolling(window).apply(calc, raw=True)

        aroon_up = _aroon(high, 25, "up")
        aroon_down = _aroon(low, 25, "down")

        features = {
            "adx_14": adx,
            "plus_di_14": plus_di,
            "minus_di_14": minus_di,
            "adx_slope": adx_slope,
            "aroon_up_25": aroon_up,
            "aroon_down_25": aroon_down,
        }
        return features

    def _statistical_features(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, pd.Series]:
        mean20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        zscore = _safe_divide(close - mean20, std20)
        skewness = close.rolling(20).skew()
        kurtosis = close.rolling(20).kurt()
        hurst = _hurst_exponent(close, 20)
        price_entropy = _rolling_entropy(close.pct_change(), 20)

        vol_entropy = _rolling_entropy(volume.pct_change(), 20)

        features = {
            "zscore_20": zscore,
            "rolling_skew_20": skewness,
            "rolling_kurt_20": kurtosis,
            "hurst_20": hurst,
            "rolling_entropy_20": price_entropy,
            "volume_entropy_20": vol_entropy,
        }
        return features

    def _support_resistance_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        high20 = high.rolling(20).max()
        low20 = low.rolling(20).min()
        high50 = high.rolling(50).max()
        low50 = low.rolling(50).min()

        features = {
            "dist_close_to_high_20": _safe_divide(high20 - close, close),
            "dist_close_to_low_20": _safe_divide(close - low20, close),
            "dist_close_to_high_50": _safe_divide(high50 - close, close),
            "dist_close_to_low_50": _safe_divide(close - low50, close),
        }
        return features

    def _smart_money_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        swing_high = ((high > high.shift(1)) & (high > high.shift(-1))).astype(float)
        swing_low = ((low < low.shift(1)) & (low < low.shift(-1))).astype(float)

        bos = ((close > high.shift(1)) & (swing_high.shift(1) > 0)).astype(float)
        choch = ((close < low.shift(1)) & (swing_low.shift(1) > 0)).astype(float)

        liquidity_sweep = ((high > high.shift(1)) & (close < open_)).astype(float)
        order_block = (
            ((open_ > close) & (high == high.rolling(10).max()))
            | ((open_ < close) & (low == low.rolling(10).min()))
        ).astype(float)

        features = {
            "swing_high_indicator": swing_high,
            "swing_low_indicator": swing_low,
            "bos_signal": bos,
            "choch_signal": choch,
            "liquidity_sweep": liquidity_sweep,
            "order_block_score": order_block,
        }
        return features

    def _market_profile_features(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, pd.Series]:
        vah = close.rolling(20).quantile(0.7)
        val = close.rolling(20).quantile(0.3)
        poc = close.rolling(20).median()
        value_area_width = vah - val
        price_vs_vah = close - vah
        price_vs_val = close - val
        time_above_poc = (close > poc).rolling(20).mean()
        balanced_profile = (
            (value_area_width / poc.replace(0, np.nan)).between(0.01, 0.05)
        ).astype(float)

        volume_profile_ratio = _safe_divide(volume, volume.rolling(20).sum())
        price_profile_ratio = _safe_divide(close, close.rolling(20).sum())

        features = {
            "vah_20": vah,
            "val_20": val,
            "poc_20": poc,
            "value_area_width_20": value_area_width,
            "price_vs_vah_20": price_vs_vah,
            "price_vs_val_20": price_vs_val,
            "time_above_poc_20": time_above_poc,
            "balanced_profile_indicator_20": balanced_profile,
            "volume_profile_ratio_20": volume_profile_ratio,
            "price_profile_ratio_20": price_profile_ratio,
        }
        return features


__all__ = ["FeatureEngine", "FeatureEngineConfig"]
