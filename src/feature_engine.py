"""
FeatureEngine - Generates ~100 technical features from OHLCV data
Groups: MA, Momentum, Volatility, Volume, Price Pattern, Returns, Trend, Statistical, S/R, SMC, Market Profile
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """Extracts ~100 technical features from OHLCV data"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all technical features from OHLCV DataFrame
        Expected columns: time, open, high, low, close, volume
        """
        df = df.copy()
        features_list = []
        
        # 1. Moving Averages (13 features)
        ma_features = self._moving_average_features(df)
        features_list.append(ma_features)
        
        # 2. Momentum (13 features)
        momentum_features = self._momentum_features(df)
        features_list.append(momentum_features)
        
        # 3. Volatility (15 features)
        volatility_features = self._volatility_features(df)
        features_list.append(volatility_features)
        
        # 4. Volume (9 features)
        volume_features = self._volume_features(df)
        features_list.append(volume_features)
        
        # 5. Price Pattern (8 features)
        pattern_features = self._price_pattern_features(df)
        features_list.append(pattern_features)
        
        # 6. Returns (8 features)
        returns_features = self._returns_features(df)
        features_list.append(returns_features)
        
        # 7. Trend Strength (6 features)
        trend_features = self._trend_strength_features(df)
        features_list.append(trend_features)
        
        # 8. Statistical (6 features)
        stat_features = self._statistical_features(df)
        features_list.append(stat_features)
        
        # 9. Support/Resistance (4 features)
        sr_features = self._support_resistance_features(df)
        features_list.append(sr_features)
        
        # 10. Smart Money Concepts (6 features)
        smc_features = self._smart_money_features(df)
        features_list.append(smc_features)
        
        # 11. Market Profile (10 features)
        profile_features = self._market_profile_features(df)
        features_list.append(profile_features)
        
        # Combine all features
        features_df = pd.concat(features_list, axis=1)
        
        # Fill NaN values
        features_df = features_df.fillna(method='bfill').fillna(0)
        
        return features_df
    
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 MA features"""
        features = {}
        close = df['close']
        
        # Simple MAs
        for period in [5, 10, 20, 50]:
            features[f'MA{period}'] = close.rolling(period).mean()
            features[f'EMA{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MA20/MA50 ratio and crossovers
        ma20 = features['MA20']
        ma50 = features['MA50']
        features['MA20_MA50_ratio'] = ma20 / (ma50 + 1e-8)
        features['MA20_MA50_cross'] = (ma20 > ma50).astype(float)
        
        # MA slopes
        features['MA20_slope'] = ma20.diff(5) / (ma20.shift(5) + 1e-8)
        features['MA50_slope'] = ma50.diff(10) / (ma50.shift(10) + 1e-8)
        
        # Price vs MA
        features['close_MA20_ratio'] = close / (ma20 + 1e-8)
        features['close_MA50_ratio'] = close / (ma50 + 1e-8)
        
        return pd.DataFrame(features)
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 momentum features"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # ROC
        for period in [5, 10]:
            features[f'ROC{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        features['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-8)
        features['Stoch_D'] = features['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['MACD'] = ema12 - ema26
        features['MACD_signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
        features['MACD_hist'] = features['MACD'] - features['MACD_signal']
        
        # Williams %R
        features['Williams_R'] = -100 * (high14 - close) / (high14 - low14 + 1e-8)
        
        return pd.DataFrame(features)
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 volatility features"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        for period in [14, 20]:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features[f'ATR{period}'] = tr.rolling(period).mean()
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['BB_upper'] = ma20 + 2 * std20
        features['BB_mid'] = ma20
        features['BB_lower'] = ma20 - 2 * std20
        features['BB_width'] = (features['BB_upper'] - features['BB_lower']) / (ma20 + 1e-8)
        features['BB_position'] = (close - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'] + 1e-8)
        
        # Keltner Channels
        atr = features['ATR14']
        kc_mid = close.rolling(20).mean()
        features['KC_upper'] = kc_mid + 1.5 * atr
        features['KC_lower'] = kc_mid - 1.5 * atr
        features['KC_width'] = (features['KC_upper'] - features['KC_lower']) / (kc_mid + 1e-8)
        
        # Standard Deviation
        for period in [10, 20]:
            features[f'StdDev{period}'] = close.rolling(period).std()
        
        # Volatility ratio
        features['volatility_ratio'] = features['StdDev10'] / (features['StdDev20'] + 1e-8)
        
        return pd.DataFrame(features)
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 volume features"""
        features = {}
        close = df['close']
        volume = df['volume']
        
        # OBV
        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        features['OBV'] = obv
        features['OBV_MA'] = obv.rolling(20).mean()
        
        # VWAP (simplified)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Volume ratios
        features['volume_MA20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (features['volume_MA20'] + 1e-8)
        
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-8)
        features['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume price trend
        features['VPT'] = (volume * close.pct_change()).cumsum()
        
        # On Balance Volume momentum
        features['OBV_momentum'] = obv.diff(10)
        
        return pd.DataFrame(features)
    
    def _price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 price pattern features"""
        features = {}
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low
        
        features['body_pct'] = body / (total_range + 1e-8)
        features['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        features['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern
        features['is_doji'] = (body < 0.1 * total_range).astype(float)
        
        # Gaps
        features['gap_up'] = ((low - high.shift(1)) > 0).astype(float)
        features['gap_down'] = ((high.shift(1) - low) > 0).astype(float)
        
        # Position in range
        features['position_in_range'] = (close - low) / (total_range + 1e-8)
        
        # Hammer/Shooting star (simplified)
        features['is_hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(float)
        
        return pd.DataFrame(features)
    
    def _returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 returns features"""
        features = {}
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        features['log_return_20'] = np.log(close / close.shift(20))
        
        # Intraday return
        features['intraday_return'] = (close - open_price) / (open_price + 1e-8)
        
        # Cumulative returns
        features['cumulative_return_5'] = close.pct_change(5)
        features['cumulative_return_20'] = close.pct_change(20)
        
        # Return volatility
        returns = close.pct_change()
        features['return_volatility'] = returns.rolling(20).std()
        
        # Sharpe-like ratio
        features['return_sharpe'] = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)
        
        return pd.DataFrame(features)
    
    def _trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 trend strength features (ADX, DI+, DI-, Aroon)"""
        features = {}
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
        
        # Smoothing
        period = 14
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['ADX'] = dx.rolling(period).mean()
        features['plus_DI'] = plus_di
        features['minus_DI'] = minus_di
        
        # Aroon
        period_aroon = 14
        aroon_up = high.rolling(period_aroon + 1).apply(lambda x: (period_aroon - x.argmax()) / period_aroon * 100, raw=True)
        aroon_down = low.rolling(period_aroon + 1).apply(lambda x: (period_aroon - x.argmin()) / period_aroon * 100, raw=True)
        features['Aroon_Up'] = aroon_up
        features['Aroon_Down'] = aroon_down
        features['Aroon_Oscillator'] = aroon_up - aroon_down
        
        return pd.DataFrame(features)
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 statistical features"""
        features = {}
        close = df['close']
        
        # Rolling statistics
        window = 20
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        
        # Z-score
        features['z_score'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Skewness
        features['skewness'] = close.rolling(window).skew()
        
        # Kurtosis
        features['kurtosis'] = close.rolling(window).apply(lambda x: x.kurtosis())
        
        # Hurst exponent (simplified)
        def hurst_approx(ts):
            if len(ts) < 10:
                return 0.5
            lags = range(2, min(10, len(ts)))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        features['hurst'] = close.rolling(30).apply(hurst_approx, raw=True)
        
        # Price position percentile
        features['price_percentile'] = close.rolling(window).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
        
        return pd.DataFrame(features)
    
    def _support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 support/resistance features"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        for period in [20, 50]:
            recent_high = high.rolling(period).max()
            recent_low = low.rolling(period).min()
            features[f'dist_to_high_{period}'] = (recent_high - close) / (close + 1e-8)
            features[f'dist_to_low_{period}'] = (close - recent_low) / (close + 1e-8)
        
        return pd.DataFrame(features)
    
    def _smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        features = {}
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        features['swing_high'] = (high == high.rolling(window * 2 + 1, center=True).max()).astype(float)
        features['swing_low'] = (low == low.rolling(window * 2 + 1, center=True).min()).astype(float)
        
        # Break of Structure (BOS) - simplified
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        features['BOS_up'] = ((close > rolling_high.shift(1)) & (close.shift(1) <= rolling_high.shift(1))).astype(float)
        features['BOS_down'] = ((close < rolling_low.shift(1)) & (close.shift(1) >= rolling_low.shift(1))).astype(float)
        
        # Change of Character (CHoCH) - simplified
        price_change = close.diff()
        features['CHoCH_bullish'] = ((price_change > 0) & (price_change.shift(1) <= 0)).astype(float)
        features['CHoCH_bearish'] = ((price_change < 0) & (price_change.shift(1) >= 0)).astype(float)
        
        return pd.DataFrame(features)
    
    def _market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Market Profile features (simplified)"""
        features = {}
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price levels
        typical_price = (high + low + close) / 3
        
        # POC (Point of Control) - price level with most volume
        # Simplified: use typical price weighted by volume
        features['POC'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # VAH/VAL (Value Area High/Low) - simplified
        rolling_mean = typical_price.rolling(20).mean()
        rolling_std = typical_price.rolling(20).std()
        features['VAH'] = rolling_mean + rolling_std
        features['VAL'] = rolling_mean - rolling_std
        
        # Distance to POC/VAH/VAL
        features['dist_to_POC'] = (close - features['POC']) / (close + 1e-8)
        features['dist_to_VAH'] = (close - features['VAH']) / (close + 1e-8)
        features['dist_to_VAL'] = (close - features['VAL']) / (close + 1e-8)
        
        # Volume profile entropy (simplified)
        price_bins = pd.cut(typical_price, bins=10, labels=False)
        volume_dist = price_bins.groupby(price_bins).transform(lambda x: volume / (x.sum() + 1e-8))
        features['volume_entropy'] = volume_dist.rolling(20).apply(lambda x: -np.sum(x * np.log(x + 1e-8)))
        
        # Time-based features
        features['session_high'] = high.rolling(20).max()
        features['session_low'] = low.rolling(20).min()
        features['session_range'] = features['session_high'] - features['session_low']
        
        return pd.DataFrame(features)
