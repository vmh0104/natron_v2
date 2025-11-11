"""
FeatureEngine: Generates ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngine:
    """Extracts comprehensive technical features from OHLCV data"""
    
    def __init__(self):
        self.feature_names = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV DataFrame
        Expected columns: time, open, high, low, close, volume
        """
        features_df = df.copy()
        
        # Moving Averages (13 features)
        features_df = self._add_moving_averages(features_df)
        
        # Momentum (13 features)
        features_df = self._add_momentum(features_df)
        
        # Volatility (15 features)
        features_df = self._add_volatility(features_df)
        
        # Volume (9 features)
        features_df = self._add_volume_features(features_df)
        
        # Price Patterns (8 features)
        features_df = self._add_price_patterns(features_df)
        
        # Returns (8 features)
        features_df = self._add_returns(features_df)
        
        # Trend Strength (6 features)
        features_df = self._add_trend_strength(features_df)
        
        # Statistical (6 features)
        features_df = self._add_statistical(features_df)
        
        # Support/Resistance (4 features)
        features_df = self._add_support_resistance(features_df)
        
        # Smart Money Concepts (6 features)
        features_df = self._add_smart_money(features_df)
        
        # Market Profile (10 features)
        features_df = self._add_market_profile(features_df)
        
        # Drop original OHLCV columns (keep time for reference)
        feature_cols = [c for c in features_df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        self.feature_names = feature_cols
        
        return features_df[['time'] + feature_cols]
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 MA features"""
        close = df['close']
        
        # Simple MAs
        df['MA5'] = close.rolling(5).mean()
        df['MA10'] = close.rolling(10).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()
        
        # EMAs
        df['EMA12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA26'] = close.ewm(span=26, adjust=False).mean()
        
        # MA slopes
        df['MA20_slope'] = df['MA20'].diff(5)
        df['MA50_slope'] = df['MA50'].diff(10)
        
        # MA crossovers
        df['MA5_MA20_cross'] = (df['MA5'] > df['MA20']).astype(float)
        df['MA20_MA50_cross'] = (df['MA20'] > df['MA50']).astype(float)
        df['EMA12_EMA26_cross'] = (df['EMA12'] > df['EMA26']).astype(float)
        
        # MA ratios
        df['MA20_MA50_ratio'] = df['MA20'] / (df['MA50'] + 1e-8)
        df['close_MA20_ratio'] = close / (df['MA20'] + 1e-8)
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 momentum features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_cross_up'] = ((df['RSI'] > 50) & (df['RSI'].shift(1) <= 50)).astype(float)
        df['RSI_cross_down'] = ((df['RSI'] < 50) & (df['RSI'].shift(1) >= 50)).astype(float)
        
        # ROC
        df['ROC'] = close.pct_change(10) * 100
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-8)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_rising'] = (df['MACD_hist'] > df['MACD_hist'].shift(1)).astype(float)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 volatility features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_pct'] = df['ATR'] / (close + 1e-8) * 100
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_upper'] = ma20 + 2 * std20
        df['BB_lower'] = ma20 - 2 * std20
        df['BB_mid'] = ma20
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (ma20 + 1e-8)
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        df['KC_upper'] = ema20 + 1.5 * df['ATR']
        df['KC_lower'] = ema20 - 1.5 * df['ATR']
        df['KC_width'] = (df['KC_upper'] - df['KC_lower']) / (ema20 + 1e-8)
        
        # Standard Deviation
        df['StdDev_20'] = close.rolling(20).std()
        df['StdDev_50'] = close.rolling(50).std()
        df['StdDev_ratio'] = df['StdDev_20'] / (df['StdDev_50'] + 1e-8)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 volume features"""
        close = df['close']
        volume = df['volume']
        
        # OBV
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_slope'] = df['OBV'].diff(5)
        
        # VWAP (simplified)
        typical_price = (df['high'] + df['low'] + close) / 3
        df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['close_VWAP_ratio'] = close / (df['VWAP'] + 1e-8)
        
        # MFI
        typical_price = (df['high'] + df['low'] + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-8)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        df['volume_MA20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_MA20'] + 1e-8)
        df['volume_spike'] = (volume > 1.5 * df['volume_MA20']).astype(float)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 price pattern features"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low
        
        df['body_pct'] = body / (total_range + 1e-8)
        df['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        df['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern
        df['doji'] = (body < 0.1 * total_range).astype(float)
        
        # Gaps
        df['gap_up'] = ((low > high.shift(1)) & (open_price > close.shift(1))).astype(float)
        df['gap_down'] = ((high < low.shift(1)) & (open_price < close.shift(1))).astype(float)
        
        # Position in range
        df['position_in_range'] = (close - low) / (total_range + 1e-8)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 return features"""
        close = df['close']
        
        # Log returns
        df['log_return_1'] = np.log(close / close.shift(1))
        df['log_return_3'] = np.log(close / close.shift(3))
        df['log_return_5'] = np.log(close / close.shift(5))
        
        # Simple returns
        df['return_1'] = close.pct_change(1)
        df['return_3'] = close.pct_change(3)
        
        # Intraday return
        df['intraday_return'] = (close - df['open']) / (df['open'] + 1e-8)
        
        # Cumulative return
        df['cumulative_return'] = close.pct_change().fillna(0).cumsum()
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 trend strength features (ADX, DI)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth TR and DM
        period = 14
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(period).mean()
        df['plus_DI'] = plus_di
        df['minus_DI'] = minus_di
        df['DI_diff'] = plus_di - minus_di
        
        # Aroon
        period_aroon = 14
        aroon_up = high.rolling(period_aroon).apply(lambda x: (period_aroon - x.argmax()) / period_aroon * 100)
        aroon_down = low.rolling(period_aroon).apply(lambda x: (period_aroon - x.argmin()) / period_aroon * 100)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        
        return df
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 statistical features"""
        close = df['close']
        
        # Rolling statistics
        window = 20
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        
        # Z-score
        df['z_score'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Skewness
        df['skewness'] = close.rolling(window).skew()
        
        # Kurtosis
        df['kurtosis'] = close.rolling(window).kurt()
        
        # Hurst exponent (simplified)
        returns = close.pct_change().dropna()
        if len(returns) > 0:
            lags = range(2, min(20, len(returns)))
            tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            df['hurst'] = hurst
        else:
            df['hurst'] = 0.5
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 support/resistance features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        df['dist_to_high_20'] = (high_20 - close) / (close + 1e-8)
        df['dist_to_low_20'] = (close - low_20) / (close + 1e-8)
        df['dist_to_high_50'] = (high_50 - close) / (close + 1e-8)
        df['dist_to_low_50'] = (close - low_50) / (close + 1e-8)
        
        return df
    
    def _add_smart_money(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        df['swing_high'] = (high == high.rolling(window, center=True).max()).astype(float)
        df['swing_low'] = (low == low.rolling(window, center=True).min()).astype(float)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = high.where(df['swing_high'] == 1)
        last_swing_high = swing_highs.ffill()
        df['BOS'] = (close > last_swing_high.shift(1)).astype(float)
        
        # Change of Character (CHoCH) - price breaks previous swing low
        swing_lows = low.where(df['swing_low'] == 1)
        last_swing_low = swing_lows.ffill()
        df['CHoCH'] = (close < last_swing_low.shift(1)).astype(float)
        
        # Order blocks (simplified - strong candles before reversal)
        strong_bullish = ((close > df['open']) & (df['body_pct'] > 0.7)).astype(float)
        strong_bearish = ((close < df['open']) & (df['body_pct'] > 0.7)).astype(float)
        df['order_block_bull'] = strong_bullish
        df['order_block_bear'] = strong_bearish
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Market Profile features"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price range bins
        price_range = high - low
        num_bins = 10
        bin_size = price_range / (num_bins + 1e-8)
        
        # Volume profile (simplified)
        typical_price = (high + low + close) / 2
        df['typical_price'] = typical_price
        
        # POC (Point of Control) - price level with highest volume
        window = 20
        volume_weighted_price = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
        df['POC'] = volume_weighted_price
        
        # VAH/VAL (Value Area High/Low) - simplified
        df['VAH'] = high.rolling(window).quantile(0.7)
        df['VAL'] = low.rolling(window).quantile(0.3)
        df['VA_range'] = df['VAH'] - df['VAL']
        
        # Entropy (price distribution measure)
        returns = close.pct_change().dropna()
        if len(returns) > 0:
            hist, _ = np.histogram(returns.dropna(), bins=20)
            hist = hist + 1e-8
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            df['entropy'] = entropy
        else:
            df['entropy'] = 0
        
        # Additional profile features
        df['profile_width'] = price_range / (close + 1e-8)
        df['profile_asymmetry'] = (close - (high + low) / 2) / (price_range + 1e-8)
        df['volume_concentration'] = volume / (volume.rolling(window).sum() + 1e-8)
        
        return df
    
    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names
