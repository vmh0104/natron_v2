"""
Feature Engine - Comprehensive Technical Indicator Generator
Generates ~100 technical features from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features for financial trading.
    Target: ~100 features across 10 categories
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 technical features
        """
        print("🔧 Generating technical features...")
        
        # Make a copy to avoid modifying original
        data = df.copy()
        features = pd.DataFrame(index=data.index)
        
        # 1. Moving Average Features (13)
        ma_features = self._moving_average_features(data)
        features = pd.concat([features, ma_features], axis=1)
        
        # 2. Momentum Features (13)
        momentum_features = self._momentum_features(data)
        features = pd.concat([features, momentum_features], axis=1)
        
        # 3. Volatility Features (15)
        volatility_features = self._volatility_features(data)
        features = pd.concat([features, volatility_features], axis=1)
        
        # 4. Volume Features (9)
        volume_features = self._volume_features(data)
        features = pd.concat([features, volume_features], axis=1)
        
        # 5. Price Pattern Features (8)
        pattern_features = self._price_pattern_features(data)
        features = pd.concat([features, pattern_features], axis=1)
        
        # 6. Returns Features (8)
        returns_features = self._returns_features(data)
        features = pd.concat([features, returns_features], axis=1)
        
        # 7. Trend Strength Features (6)
        trend_features = self._trend_strength_features(data)
        features = pd.concat([features, trend_features], axis=1)
        
        # 8. Statistical Features (6)
        statistical_features = self._statistical_features(data)
        features = pd.concat([features, statistical_features], axis=1)
        
        # 9. Support/Resistance Features (4)
        sr_features = self._support_resistance_features(data)
        features = pd.concat([features, sr_features], axis=1)
        
        # 10. Smart Money Concepts (6)
        smc_features = self._smart_money_features(data)
        features = pd.concat([features, smc_features], axis=1)
        
        # 11. Market Profile Features (10)
        profile_features = self._market_profile_features(data)
        features = pd.concat([features, profile_features], axis=1)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        print(f"✅ Generated {len(self.feature_names)} features")
        
        return features
    
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average group (13 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Simple Moving Averages
        features['sma_5'] = close.rolling(5).mean()
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['sma_50'] = close.rolling(50).mean()
        
        # Exponential Moving Averages
        features['ema_5'] = close.ewm(span=5, adjust=False).mean()
        features['ema_10'] = close.ewm(span=10, adjust=False).mean()
        features['ema_20'] = close.ewm(span=20, adjust=False).mean()
        
        # MA slopes
        features['sma_20_slope'] = features['sma_20'].diff(5) / features['sma_20']
        features['ema_20_slope'] = features['ema_20'].diff(5) / features['ema_20']
        
        # MA crossovers
        features['ma_cross_5_10'] = (features['sma_5'] > features['sma_10']).astype(int)
        features['ma_cross_10_20'] = (features['sma_10'] > features['sma_20']).astype(int)
        
        # Price to MA ratio
        features['price_to_sma20'] = close / features['sma_20']
        features['price_to_sma50'] = close / features['sma_50']
        
        return features
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum group (13 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_7'] = self._calculate_rsi(close, 7)
        
        # ROC (Rate of Change)
        features['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        features['roc_20'] = ((close - close.shift(20)) / close.shift(20)) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        features['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Stochastic Oscillator
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-10))
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        features['macd_hist_slope'] = features['macd_hist'].diff()
        
        # Williams %R
        features['williams_r'] = -100 * ((high_14 - close) / (high_14 - low_14 + 1e-10))
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility group (15 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_7'] = tr.rolling(7).mean()
        features['atr_ratio'] = features['atr_7'] / (features['atr_14'] + 1e-10)
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_mid'] = sma_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        
        # Keltner Channels
        ema_20 = close.ewm(span=20, adjust=False).mean()
        features['kc_upper'] = ema_20 + (features['atr_14'] * 2)
        features['kc_lower'] = ema_20 - (features['atr_14'] * 2)
        features['kc_width'] = (features['kc_upper'] - features['kc_lower']) / ema_20
        
        # Standard Deviation
        features['std_10'] = close.rolling(10).std()
        features['std_20'] = close.rolling(20).std()
        
        # Historical Volatility
        returns = close.pct_change()
        features['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume group (9 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Volume moving averages
        features['volume_sma_10'] = volume.rolling(10).mean()
        features['volume_sma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (features['volume_sma_20'] + 1e-10)
        
        # OBV (On Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # MFI (Money Flow Index)
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        mf_volume = mf_multiplier * volume
        features['mfi_14'] = self._calculate_mfi(df, 14)
        
        # Volume price trend
        features['vpt'] = (volume * ((close - close.shift()) / close.shift())).fillna(0).cumsum()
        
        return features
    
    def _price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price Pattern group (8 features)"""
        features = pd.DataFrame(index=df.index)
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Candle body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        candle_range = high - low
        
        features['body_ratio'] = body / (candle_range + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (candle_range + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (candle_range + 1e-10)
        
        # Doji detection
        features['is_doji'] = (body < 0.1 * candle_range).astype(int)
        
        # Gap detection
        features['gap_up'] = (low > high.shift()).astype(int)
        features['gap_down'] = (high < low.shift()).astype(int)
        
        # Price position in range
        features['position_in_range'] = (close - low) / (candle_range + 1e-10)
        
        # Candle direction
        features['candle_direction'] = (close > open_price).astype(int)
        
        return features
    
    def _returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns group (8 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Simple returns
        features['return_1'] = close.pct_change(1)
        features['return_5'] = close.pct_change(5)
        features['return_10'] = close.pct_change(10)
        features['return_20'] = close.pct_change(20)
        
        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        
        # Intraday return
        features['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # Cumulative return
        features['cum_return_20'] = (close / close.shift(20)) - 1
        
        return features
    
    def _trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Strength group (6 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        adx_data = self._calculate_adx(df, 14)
        features['adx_14'] = adx_data['adx']
        features['plus_di'] = adx_data['plus_di']
        features['minus_di'] = adx_data['minus_di']
        
        # Aroon
        aroon_data = self._calculate_aroon(df, 25)
        features['aroon_up'] = aroon_data['up']
        features['aroon_down'] = aroon_data['down']
        features['aroon_oscillator'] = aroon_data['up'] - aroon_data['down']
        
        return features
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical group (6 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        
        # Skewness and Kurtosis
        features['skew_20'] = returns.rolling(20).skew()
        features['kurt_20'] = returns.rolling(20).kurt()
        
        # Z-score
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['zscore_20'] = (close - sma_20) / (std_20 + 1e-10)
        
        # Hurst Exponent (simplified)
        features['hurst_100'] = self._calculate_hurst(close, 100)
        
        # Autocorrelation
        features['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(5) if len(x) > 5 else 0, raw=False)
        
        # Mean reversion indicator
        features['mean_reversion'] = (close - sma_20) / sma_20
        
        return features
    
    def _support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance group (4 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        features['dist_to_high_20'] = (high_20 - close) / close
        features['dist_to_low_20'] = (close - low_20) / close
        features['dist_to_high_50'] = (high_50 - close) / close
        features['dist_to_low_50'] = (close - low_50) / close
        
        return features
    
    def _smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts group (6 features)"""
        features = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing highs and lows
        features['swing_high'] = (high > high.shift(1)) & (high > high.shift(-1))
        features['swing_low'] = (low < low.shift(1)) & (low < low.shift(-1))
        
        # Break of Structure (BOS)
        recent_high = high.rolling(10).max()
        recent_low = low.rolling(10).min()
        features['bos_bullish'] = (close > recent_high.shift()).astype(int)
        features['bos_bearish'] = (close < recent_low.shift()).astype(int)
        
        # Change of Character (CHoCH)
        features['choch'] = features['bos_bullish'].diff().abs()
        
        # Order block proximity (simplified)
        features['order_block_distance'] = np.minimum(
            abs(close - recent_high) / close,
            abs(close - recent_low) / close
        )
        
        return features
    
    def _market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile group (10 features)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Value Area (simplified using percentiles)
        features['vah_20'] = high.rolling(20).quantile(0.70)  # Value Area High
        features['val_20'] = low.rolling(20).quantile(0.30)   # Value Area Low
        features['poc_20'] = close.rolling(20).median()        # Point of Control
        
        features['vah_50'] = high.rolling(50).quantile(0.70)
        features['val_50'] = low.rolling(50).quantile(0.30)
        features['poc_50'] = close.rolling(50).median()
        
        # Price position relative to value area
        features['price_to_vah'] = (close - features['vah_20']) / close
        features['price_to_val'] = (close - features['val_20']) / close
        
        # Volume distribution entropy (simplified)
        features['volume_entropy'] = self._calculate_entropy(volume, 20)
        
        # Balance/Imbalance
        features['balance_indicator'] = (features['vah_20'] - features['val_20']) / close
        
        return features
    
    # Helper methods
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """Calculate ADX and directional indicators"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
    
    def _calculate_aroon(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """Calculate Aroon indicators"""
        high = df['high']
        low = df['low']
        
        aroon_up = high.rolling(period + 1).apply(
            lambda x: float(np.argmax(x)) / period * 100, raw=True
        )
        aroon_down = low.rolling(period + 1).apply(
            lambda x: float(np.argmin(x)) / period * 100, raw=True
        )
        
        return {'up': aroon_up, 'down': aroon_down}
    
    def _calculate_hurst(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Hurst Exponent (simplified rolling version)"""
        def hurst_exp(ts):
            if len(ts) < 10:
                return 0.5
            try:
                lags = range(2, min(20, len(ts) // 2))
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]
            except:
                return 0.5
        
        return series.rolling(window).apply(hurst_exp, raw=True)
    
    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate entropy of a series"""
        def entropy(x):
            if len(x) == 0:
                return 0
            hist, _ = np.histogram(x, bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        return series.rolling(window).apply(entropy, raw=True)


if __name__ == "__main__":
    # Test feature generation
    print("Testing FeatureEngine...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_samples, freq='15min'),
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + np.abs(np.random.randn(n_samples) * 0.3),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) - np.abs(np.random.randn(n_samples) * 0.3),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    engine = FeatureEngine()
    features = engine.generate_features(df)
    
    print(f"\n✅ Feature shape: {features.shape}")
    print(f"✅ Feature columns: {len(features.columns)}")
    print(f"\nFirst 5 features:\n{features.iloc[100:105, :5]}")
