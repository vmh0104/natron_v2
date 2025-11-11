from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew


class FeatureEngine:
    """
    Generate a comprehensive feature matrix (~100 features) from OHLCV data.
    """

    def __init__(
        self,
        windows: tuple = (5, 10, 14, 20, 26, 50, 96),
        neutral_buffer: float = 0.001,
    ) -> None:
        self.windows = windows
        self.neutral_buffer = neutral_buffer

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("time").reset_index(drop=True)
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        feats = pd.DataFrame(index=df.index)

        self._moving_average_features(df, feats)
        self._momentum_features(df, feats)
        self._volatility_features(df, feats)
        self._volume_features(df, feats)
        self._price_pattern_features(df, feats)
        self._return_features(df, feats)
        self._trend_strength_features(df, feats)
        self._statistical_features(df, feats)
        self._support_resistance_features(df, feats)
        self._smart_money_features(df, feats)
        self._market_profile_features(df, feats)

        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")
        feats = feats.fillna(0.0)
        return feats

    # --- Feature Groups -------------------------------------------------
    def _moving_average_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        close = df["close"]
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            feats[f"ma_{window}"] = close.rolling(window).mean()
            feats[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            feats[f"ma_slope_{window}"] = (
                feats[f"ma_{window}"] - feats[f"ma_{window}"].shift(1)
            ) / feats[f"ma_{window}"].shift(1)

        feats["ma_ratio_20_50"] = feats["ma_20"] / feats["ma_50"]
        feats["ma_ratio_10_50"] = feats["ma_10"] / feats["ma_50"]
        feats["ma_cross_20_50"] = np.where(feats["ma_20"] > feats["ma_50"], 1, -1)
        feats["ema_cross_12_26"] = np.where(
            close.ewm(span=12, adjust=False).mean() > close.ewm(span=26, adjust=False).mean(),
            1,
            -1,
        )
        feats["price_ma_distance_20"] = (close - feats["ma_20"]) / feats["ma_20"]
        feats["price_ma_distance_50"] = (close - feats["ma_50"]) / feats["ma_50"]

    def _momentum_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        feats["rsi_14"] = 100 - (100 / (1 + rs))

        for period in [5, 10, 20]:
            feats[f"roc_{period}"] = close.pct_change(periods=period)
            feats[f"momentum_{period}"] = close.diff(periods=period)

        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        feats["cci_20"] = (tp - sma_tp) / (0.015 * mad)

        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        feats["stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        feats["stoch_d"] = feats["stoch_k"].rolling(3).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd
        feats["macd_signal"] = signal
        feats["macd_hist"] = macd - signal

        feats["awesome_oscillator"] = ((high + low) / 2).rolling(5).mean() - ((high + low) / 2).rolling(34).mean()
        feats["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low)
        feats["price_velocity"] = close.diff()
        feats["price_acceleration"] = feats["price_velocity"].diff()

    def _volatility_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        true_range = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        feats["true_range"] = true_range
        feats["atr_14"] = true_range.rolling(14).mean()

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        feats["bb_mid"] = sma20
        feats["bb_upper"] = sma20 + 2 * std20
        feats["bb_lower"] = sma20 - 2 * std20
        feats["bb_width"] = feats["bb_upper"] - feats["bb_lower"]
        feats["bb_percent_b"] = (close - feats["bb_lower"]) / feats["bb_width"]

        ema20 = close.ewm(span=20, adjust=False).mean()
        atr20 = true_range.rolling(20).mean()
        feats["keltner_upper"] = ema20 + 2 * atr20
        feats["keltner_lower"] = ema20 - 2 * atr20
        feats["keltner_width"] = feats["keltner_upper"] - feats["keltner_lower"]

        feats["stddev_20"] = std20
        feats["stddev_50"] = close.rolling(50).std()
        feats["rolling_var_20"] = close.rolling(20).var()
        feats["rolling_var_50"] = close.rolling(50).var()

        feats["donchian_high_20"] = high.rolling(20).max()
        feats["donchian_low_20"] = low.rolling(20).min()
        feats["donchian_bandwidth"] = feats["donchian_high_20"] - feats["donchian_low_20"]
        feats["volatility_ratio"] = feats["atr_14"] / feats["donchian_bandwidth"]

        feats["intraday_volatility"] = (high - low) / close
        feats["garman_klass_vol"] = (
            0.5 * np.log(high / low) ** 2
            - (2 * np.log(2) - 1) * (np.log(close / df["open"]) ** 2)
        ).rolling(20).mean()

    def _volume_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        vol = df["volume"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        feats["volume_ma_20"] = vol.rolling(20).mean()
        feats["volume_ratio_20"] = vol / (vol.rolling(20).mean())
        feats["volume_std_20"] = vol.rolling(20).std()
        feats["volume_zscore"] = (vol - feats["volume_ma_20"]) / feats["volume_std_20"]
        feats["volume_change"] = vol.pct_change()

        feats["obv"] = (np.sign(close.diff()).fillna(0) * vol).cumsum()
        tp = (high + low + close) / 3
        feats["vwap"] = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)

        money_flow = tp * vol
        positive_flow = money_flow.where(tp > tp.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(tp < tp.shift(), 0).rolling(14).sum()
        mfr = positive_flow / negative_flow.replace(0, np.nan)
        feats["mfi"] = 100 - (100 / (1 + mfr))

        feats["accum_dist"] = ((close - low) - (high - close)) / (high - low + 1e-9) * vol
        feats["chaikin_osc"] = feats["accum_dist"].ewm(span=3, adjust=False).mean() - feats["accum_dist"].ewm(span=10, adjust=False).mean()
        feats["volume_climax"] = vol / vol.rolling(96).max()

    def _price_pattern_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        open_ = df["open"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        body = (close - open_).abs()
        range_ = (high - low).replace(0, np.nan)
        upper_shadow = high - close.where(close > open_, open_)
        lower_shadow = close.where(close < open_, open_) - low

        feats["body_pct"] = body / range_
        feats["upper_shadow_pct"] = upper_shadow / range_
        feats["lower_shadow_pct"] = lower_shadow / range_
        feats["is_doji"] = (feats["body_pct"] < 0.1).astype(int)
        feats["gap_up"] = (open_ > close.shift()).astype(int)
        feats["gap_down"] = (open_ < close.shift()).astype(int)
        feats["engulfing"] = (((close > open_) & (close.shift() < open_.shift()) & (close >= open_.shift()) & (open_ <= close.shift())) | ((close < open_) & (close.shift() > open_.shift()) & (close <= open_.shift()) & (open_ >= close.shift()))).astype(int)
        feats["inside_bar"] = ((high < high.shift()) & (low > low.shift())).astype(int)

    def _return_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        close = df["close"]
        feats["log_return_1"] = np.log(close / close.shift())
        feats["log_return_5"] = np.log(close / close.shift(5))
        feats["log_return_20"] = np.log(close / close.shift(20))
        feats["intraday_return"] = np.log(df["close"] / df["open"])
        feats["cumulative_return_20"] = feats["log_return_1"].rolling(20).sum()
        feats["rolling_return_20"] = close.pct_change(20)
        feats["rolling_sharpe_20"] = feats["log_return_1"].rolling(20).mean() / (feats["log_return_1"].rolling(20).std() + 1e-9)
        feats["volatility_annualized"] = feats["log_return_1"].rolling(96).std() * np.sqrt(96)
        feats["close_open_ratio"] = close / df["open"]
        feats["candle_range"] = (df["high"] - df["low"]) / df["open"]

    def _trend_strength_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / atr

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        feats["adx_14"] = dx.rolling(14).mean()
        feats["plus_di"] = plus_di
        feats["minus_di"] = minus_di

        window = 14
        aroon_up = 100 * high.rolling(window).apply(lambda x: (np.argmax(x) + 1) / window, raw=True)
        aroon_down = 100 * low.rolling(window).apply(lambda x: (np.argmin(x) + 1) / window, raw=True)
        feats["aroon_up"] = aroon_up
        feats["aroon_down"] = aroon_down
        feats["aroon_osc"] = aroon_up - aroon_down

    def _statistical_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        close = df["close"]
        feats["rolling_mean_20"] = close.rolling(20).mean()
        feats["rolling_std_20"] = close.rolling(20).std()
        feats["rolling_skew_20"] = close.rolling(20).apply(lambda x: skew(x, bias=False), raw=True)
        feats["rolling_kurt_20"] = close.rolling(20).apply(lambda x: kurtosis(x, bias=False), raw=True)
        feats["zscore_close"] = (close - feats["rolling_mean_20"]) / (feats["rolling_std_20"] + 1e-9)

        feats["hurst_30"] = close.rolling(30).apply(self._hurst_exponent, raw=False)
        feats["price_entropy_20"] = close.rolling(20).apply(
            lambda x: entropy(np.histogram(x, bins=10)[0] + 1e-9), raw=False
        )

    def _support_resistance_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        for window in [20, 50]:
            rolling_high = high.rolling(window).max()
            rolling_low = low.rolling(window).min()
            feats[f"dist_high_{window}"] = (rolling_high - close) / (rolling_high + 1e-9)
            feats[f"dist_low_{window}"] = (close - rolling_low) / (rolling_low + 1e-9)

    def _smart_money_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        feats["swing_high"] = ((high.shift(1) < high) & (high.shift(-1) < high)).astype(int)
        feats["swing_low"] = ((low.shift(1) > low) & (low.shift(-1) > low)).astype(int)

        feats["bos_signal"] = ((close > high.shift(1)) & feats["swing_high"].shift(1).eq(1)).astype(int)
        feats["choch_signal"] = ((close < low.shift(1)) & feats["swing_low"].shift(1).eq(1)).astype(int)
        feats["liquidity_grab"] = (
            ((high > high.shift(1)) & (close < open_)) | ((low < low.shift(1)) & (close > open_))
        ).astype(int)

        orderblock_high = high.rolling(20).max()
        orderblock_low = low.rolling(20).min()
        feats["orderblock_proximity"] = np.minimum(
            abs(close - orderblock_high), abs(close - orderblock_low)
        ) / close

    def _market_profile_features(self, df: pd.DataFrame, feats: pd.DataFrame) -> None:
        close = df["close"]
        volume = df["volume"]

        window = 96

        def profile_stats(series: pd.Series, weights: pd.Series) -> tuple:
            if series.empty:
                return (np.nan,) * 8
            series_np = series.to_numpy()
            weights_np = weights.to_numpy()
            if np.allclose(weights_np.sum(), 0):
                weights_np = np.ones_like(series_np)

            hist, bin_edges = np.histogram(series_np, bins=20, weights=weights_np)
            probs = hist / (hist.sum() + 1e-9)
            poc_idx = probs.argmax() if len(probs) else 0
            poc = (bin_edges[poc_idx] + bin_edges[min(poc_idx + 1, len(bin_edges) - 1)]) / 2

            weighted_mean = np.average(series_np, weights=weights_np)
            centered = series_np - weighted_mean
            std = np.sqrt(np.average(centered**2, weights=weights_np) + 1e-9)
            skewness = np.average(centered**3, weights=weights_np) / (std**3 + 1e-9)
            kurt = np.average(centered**4, weights=weights_np) / (std**4 + 1e-9)

            vah = np.quantile(series_np, 0.7)
            val = np.quantile(series_np, 0.3)
            va_range = vah - val
            ent = entropy(probs + 1e-9)

            return poc, vah, val, va_range, ent, std, skewness, kurt

        poc_list = []
        vah_list = []
        val_list = []
        va_range_list = []
        entropy_list = []
        std_list = []
        skew_list = []
        kurt_list = []

        for i in range(len(close)):
            if i < window:
                subset = close.iloc[: i + 1]
                weights = volume.iloc[: i + 1]
            else:
                subset = close.iloc[i - window + 1 : i + 1]
                weights = volume.iloc[i - window + 1 : i + 1]
            poc, vah, val, va_range, ent, std, skewness, kurt = profile_stats(subset, weights)
            poc_list.append(poc)
            vah_list.append(vah)
            val_list.append(val)
            va_range_list.append(va_range)
            entropy_list.append(ent)
            std_list.append(std)
            skew_list.append(skewness)
            kurt_list.append(kurt)

        feats["poc_96"] = poc_list
        feats["vah_96"] = vah_list
        feats["val_96"] = val_list
        feats["va_range_96"] = va_range_list
        feats["profile_entropy_96"] = entropy_list
        feats["profile_std_96"] = std_list
        feats["profile_skew_96"] = skew_list
        feats["profile_kurt_96"] = kurt_list
        feats["poc_distance"] = (close - feats["poc_96"]) / close
        feats["va_position"] = (close - feats["val_96"]) / (feats["va_range_96"] + 1e-9)

    # --- Helpers --------------------------------------------------------
    @staticmethod
    def _hurst_exponent(ts: pd.Series) -> float:
        if len(ts) < 20:
            return np.nan
        lags = range(2, min(20, len(ts) - 1))
        tau = [np.sqrt(np.std(ts.diff(lag))) for lag in lags]
        if any(t <= 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
