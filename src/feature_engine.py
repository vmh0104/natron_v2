"""
Natron Feature Engine - Extracts ~100 Technical Indicators from OHLCV Data
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Comprehensive feature extraction for financial time series.
    Generates ~100 technical indicators across multiple categories.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all feature groups from OHLCV dataframe.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 technical features
        """
        print("🔧 Extracting features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Core price data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values
        
        # 1. Moving Averages (13 features)
        features = pd.concat([features, self._moving_averages(df)], axis=1)
        
        # 2. Momentum Indicators (13 features)
        features = pd.concat([features, self._momentum_indicators(df)], axis=1)
        
        # 3. Volatility Indicators (15 features)
        features = pd.concat([features, self._volatility_indicators(df)], axis=1)
        
        # 4. Volume Indicators (9 features)
        features = pd.concat([features, self._volume_indicators(df)], axis=1)
        
        # 5. Price Patterns (8 features)
        features = pd.concat([features, self._price_patterns(df)], axis=1)
        
        # 6. Returns (8 features)
        features = pd.concat([features, self._returns(df)], axis=1)
        
        # 7. Trend Strength (6 features)
        features = pd.concat([features, self._trend_strength(df)], axis=1)
        
        # 8. Statistical Features (6 features)
        features = pd.concat([features, self._statistical_features(df)], axis=1)
        
        # 9. Support/Resistance (4 features)
        features = pd.concat([features, self._support_resistance(df)], axis=1)
        
        # 10. Smart Money Concepts (6 features)
        features = pd.concat([features, self._smart_money_concepts(df)], axis=1)
        
        # 11. Market Profile (10 features)
        features = pd.concat([features, self._market_profile(df)], axis=1)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        print(f"✅ Extracted {len(self.feature_names)} features")
        
        return features
    
    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average indicators (13 features)"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = close.rolling(period).mean()
            
        # Exponential Moving Averages
        for period in [5, 20, 50]:
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            
        # MA slopes and ratios
        features['ma_20_slope'] = features['ma_20'].diff(5) / features['ma_20']
        features['ma_50_slope'] = features['ma_50'].diff(10) / features['ma_50']
        features['close_ma20_ratio'] = close / features['ma_20']
        features['ma20_ma50_ratio'] = features['ma_20'] / features['ma_50']
        features['golden_cross'] = (features['ma_20'] > features['ma_50']).astype(int)
        
        return features
    
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (13 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI variations
        features['rsi_7'] = self._calculate_rsi(close, 7)
        features['rsi_21'] = self._calculate_rsi(close, 21)
        
        # Rate of Change
        features['roc_5'] = ((close - close.shift(5)) / close.shift(5)) * 100
        features['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        features['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Stochastic Oscillator
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        features['macd_hist_slope'] = features['macd_hist'].diff(1)
        
        return features
    
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators (15 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_20'] = tr.rolling(20).mean()
        features['atr_pct'] = features['atr_14'] / close
        
        # Bollinger Bands
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}'] = ma + (2 * std)
            features[f'bb_lower_{period}'] = ma - (2 * std)
            features[f'bb_mid_{period}'] = ma
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / ma
            features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Keltner Channels
        kc_mid = close.ewm(span=20).mean()
        features['kc_upper'] = kc_mid + (2 * features['atr_14'])
        features['kc_lower'] = kc_mid - (2 * features['atr_14'])
        features['kc_width'] = (features['kc_upper'] - features['kc_lower']) / kc_mid
        
        # Standard Deviation
        features['std_20'] = close.rolling(20).std()
        features['std_50'] = close.rolling(50).std()
        
        # Historical Volatility
        returns = np.log(close / close.shift(1))
        features['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        return features
    
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators (9 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        features = pd.DataFrame(index=df.index)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema_20'] = obv.ewm(span=20).mean()
        
        # Volume Moving Averages
        features['volume_ma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma_20']
        
        # VWAP (Volume Weighted Average Price)
        features['vwap'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        
        # MFI (Money Flow Index)
        tp = (high + low + close) / 3
        mf = tp * volume
        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        features['mfi_14'] = 100 - (100 / (1 + mf_pos / mf_neg))
        
        # Volume trend
        features['volume_trend'] = volume.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Price-Volume Trend
        features['pvt'] = ((close.diff() / close.shift(1)) * volume).cumsum()
        
        return features
    
    def _price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern indicators (8 features)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Candle body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        candle_range = high - low
        
        features['body_pct'] = body / candle_range
        features['upper_shadow_pct'] = upper_shadow / candle_range
        features['lower_shadow_pct'] = lower_shadow / candle_range
        
        # Doji detection
        features['is_doji'] = (body / candle_range < 0.1).astype(int)
        
        # Gap detection
        features['gap_up'] = (low > high.shift(1)).astype(int)
        features['gap_down'] = (high < low.shift(1)).astype(int)
        
        # Position in range
        features['position_in_range'] = (close - low) / candle_range
        
        # Bullish/Bearish
        features['bullish_candle'] = (close > open_price).astype(int)
        
        return features
    
    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return-based features (8 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        features = pd.DataFrame(index=df.index)
        
        # Log returns
        features['log_return'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        features['log_return_10'] = np.log(close / close.shift(10))
        
        # Intraday returns
        features['intraday_return'] = (close - open_price) / open_price
        features['high_low_return'] = (high - low) / low
        
        # Cumulative returns
        features['cum_return_20'] = (close / close.shift(20)) - 1
        features['cum_return_50'] = (close / close.shift(50)) - 1
        
        # Return volatility
        features['return_vol_20'] = features['log_return'].rolling(20).std()
        
        return features
    
    def _trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength indicators (6 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        features['adx_14'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Aroon
        aroon_up = 100 * close.rolling(25).apply(lambda x: x.argmax()) / 25
        aroon_down = 100 * close.rolling(25).apply(lambda x: x.argmin()) / 25
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        
        return features
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features (6 features)"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Skewness and Kurtosis
        features['skew_20'] = close.rolling(20).skew()
        features['kurt_20'] = close.rolling(20).kurt()
        
        # Z-score
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['zscore_20'] = (close - ma_20) / std_20
        
        # Hurst Exponent (simplified)
        features['hurst_50'] = close.rolling(50).apply(self._hurst_exponent)
        
        # Quantile position
        features['quantile_20'] = close.rolling(20).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
        
        # Autocorrelation
        features['autocorr_5'] = close.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=5))
        
        return features
    
    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance levels (4 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # Distance to recent high/low
        features['dist_to_high_20'] = (high.rolling(20).max() - close) / close
        features['dist_to_low_20'] = (close - low.rolling(20).min()) / close
        features['dist_to_high_50'] = (high.rolling(50).max() - close) / close
        features['dist_to_low_50'] = (close - low.rolling(50).min()) / close
        
        return features
    
    def _smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts (6 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # Swing High/Low
        features['swing_high'] = (high > high.shift(1)) & (high > high.shift(-1))
        features['swing_low'] = (low < low.shift(1)) & (low < low.shift(-1))
        
        # Break of Structure (BOS)
        recent_high = high.rolling(20).max()
        recent_low = low.rolling(20).min()
        features['bos_bull'] = (close > recent_high.shift(1)).astype(int)
        features['bos_bear'] = (close < recent_low.shift(1)).astype(int)
        
        # Change of Character (CHoCH) - simplified
        features['choch_signal'] = features['swing_high'].astype(int) - features['swing_low'].astype(int)
        
        # Order Block proximity (simplified)
        features['near_demand_zone'] = ((close - low.rolling(20).min()) / close < 0.02).astype(int)
        
        return features
    
    def _market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile indicators (10 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        features = pd.DataFrame(index=df.index)
        
        # Value Area (simplified)
        for period in [20, 50]:
            # POC (Point of Control) - approximation
            features[f'poc_{period}'] = close.rolling(period).median()
            
            # VAH/VAL (Value Area High/Low) - approximation using percentiles
            features[f'vah_{period}'] = close.rolling(period).quantile(0.7)
            features[f'val_{period}'] = close.rolling(period).quantile(0.3)
            
            # Distance from value area
            features[f'dist_from_poc_{period}'] = (close - features[f'poc_{period}']) / close
        
        # Volume profile entropy (diversity measure)
        features['volume_entropy_20'] = volume.rolling(20).apply(self._entropy)
        
        # Price level concentration
        features['price_concentration_20'] = close.rolling(20).apply(lambda x: len(np.unique(np.round(x, 2))) / len(x))
        
        return features
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for given period"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _hurst_exponent(series):
        """Calculate Hurst exponent (simplified)"""
        if len(series) < 20:
            return 0.5
        try:
            lags = range(2, min(20, len(series) // 2))
            tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    @staticmethod
    def _entropy(series):
        """Calculate entropy of a series"""
        if len(series) == 0:
            return 0
        value_counts = pd.Series(series).value_counts(normalize=True)
        return -np.sum(value_counts * np.log(value_counts + 1e-9))


if __name__ == "__main__":
    # Test feature extraction
    print("Testing FeatureEngine...")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + np.random.rand(n),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - np.random.rand(n),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n)
    })
    
    engine = FeatureEngine()
    features = engine.extract_all_features(df)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"Shape: {features.shape}")
    print(f"\nFeature columns: {features.columns.tolist()[:10]}...")
    print(f"\nSample features:\n{features.head()}")
