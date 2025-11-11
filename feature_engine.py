"""
Natron Transformer - Feature Engineering Module
Extracts ~100 technical features from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Technical feature extraction engine
    Generates approximately 100 features across multiple categories
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features from OHLCV data
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
        
        Returns:
            features_df: DataFrame with ~100 technical features
        """
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        print("🔧 Extracting features...")
        
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
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        print(f"✅ Extracted {len(self.feature_names)} features")
        
        return features
    
    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average family (13 features)"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Simple moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = close.rolling(period).mean()
        
        # Exponential moving averages
        for period in [12, 26]:
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MA slopes
        features['sma20_slope'] = features['sma_20'].diff(5) / features['sma_20']
        features['sma50_slope'] = features['sma_50'].diff(5) / features['sma_50']
        
        # MA crossovers
        features['sma_cross_20_50'] = (features['sma_20'] > features['sma_50']).astype(int)
        
        # Price to MA ratio
        features['close_to_sma20'] = close / features['sma_20']
        features['close_to_sma50'] = close / features['sma_50']
        
        return features
    
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (13 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        features['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Stochastic Oscillator
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        features['macd_hist_slope'] = features['macd_hist'].diff()
        
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
        features['atr_pct'] = features['atr_14'] / close
        
        # Bollinger Bands
        for period in [20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma
            features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / \
                                                  (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        features['keltner_upper'] = ema20 + 2 * features['atr_14']
        features['keltner_lower'] = ema20 - 2 * features['atr_14']
        features['keltner_position'] = (close - features['keltner_lower']) / \
                                        (features['keltner_upper'] - features['keltner_lower'] + 1e-10)
        
        # Standard deviation
        for period in [10, 20, 50]:
            features[f'std_{period}'] = close.rolling(period).std() / close
        
        # Historical volatility
        log_returns = np.log(close / close.shift())
        features['hv_20'] = log_returns.rolling(20).std() * np.sqrt(252)
        
        # Volatility ratio
        features['vol_ratio'] = features['std_10'] / (features['std_50'] + 1e-10)
        
        return features
    
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators (9 features)"""
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        tp = (high + low + close) / 3
        features['vwap'] = (tp * volume).cumsum() / volume.cumsum()
        features['close_to_vwap'] = close / features['vwap']
        
        # Money Flow Index (MFI)
        raw_money_flow = tp * volume
        positive_flow = raw_money_flow.where(tp > tp.shift(), 0).rolling(14).sum()
        negative_flow = raw_money_flow.where(tp < tp.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        features['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        vol_sma20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_sma20 + 1e-10)
        features['volume_trend'] = volume.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Volume spike
        features['volume_spike'] = (volume > vol_sma20 * 1.5).astype(int)
        
        return features
    
    def _price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern features (8 features)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Candle body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        candle_range = high - low
        
        features['body_pct'] = body / (candle_range + 1e-10)
        features['upper_shadow_pct'] = upper_shadow / (candle_range + 1e-10)
        features['lower_shadow_pct'] = lower_shadow / (candle_range + 1e-10)
        
        # Doji pattern (small body)
        features['is_doji'] = (features['body_pct'] < 0.1).astype(int)
        
        # Hammer/Shooting star patterns
        features['is_hammer'] = ((features['lower_shadow_pct'] > 0.6) & 
                                  (features['upper_shadow_pct'] < 0.1)).astype(int)
        features['is_shooting_star'] = ((features['upper_shadow_pct'] > 0.6) & 
                                         (features['lower_shadow_pct'] < 0.1)).astype(int)
        
        # Price position in range
        features['position_in_range'] = (close - low) / (high - low + 1e-10)
        
        # Gap detection
        features['gap'] = (open_price - close.shift()) / close.shift()
        
        return features
    
    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return-based features (8 features)"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # Log returns
        features['log_return'] = np.log(close / close.shift())
        
        # Simple returns (multiple periods)
        for period in [1, 5, 10, 20]:
            features[f'return_{period}'] = (close - close.shift(period)) / close.shift(period)
        
        # Intraday return
        features['intraday_return'] = (close - open_price) / open_price
        
        # Cumulative return
        features['cum_return_20'] = (close / close.shift(20)) - 1
        
        return features
    
    def _trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength indicators (6 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx_14'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Aroon Indicator
        period = 25
        aroon_up = close.rolling(period).apply(lambda x: x.argmax()) / period * 100
        aroon_down = close.rolling(period).apply(lambda x: x.argmin()) / period * 100
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_osc'] = aroon_up - aroon_down
        
        return features
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features (6 features)"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Skewness and Kurtosis
        features['skew_20'] = close.rolling(20).skew()
        features['kurt_20'] = close.rolling(20).kurt()
        
        # Z-score
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        features['zscore_20'] = (close - sma) / (std + 1e-10)
        
        # Hurst exponent (simplified)
        def hurst_approx(x):
            if len(x) < 10:
                return 0.5
            lags = range(2, min(10, len(x) // 2))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        
        features['hurst_50'] = close.rolling(50).apply(hurst_approx, raw=True)
        
        # Entropy (price distribution)
        def entropy(x):
            hist, _ = np.histogram(x, bins=10)
            hist = hist / (hist.sum() + 1e-10)
            return -np.sum(hist * np.log(hist + 1e-10))
        
        features['entropy_20'] = close.rolling(20).apply(entropy, raw=True)
        
        # Fractal dimension
        features['fractal_dim'] = 1 / (features['hurst_50'] + 0.5)
        
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
        high = df['high']
        low = df['low']
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Swing highs and lows
        swing_period = 5
        features['swing_high'] = high.rolling(swing_period * 2 + 1, center=True).max()
        features['swing_low'] = low.rolling(swing_period * 2 + 1, center=True).min()
        
        # Break of Structure (BOS)
        features['bos_bull'] = (high > features['swing_high'].shift()).astype(int)
        features['bos_bear'] = (low < features['swing_low'].shift()).astype(int)
        
        # Change of Character (CHoCH)
        features['choch'] = (features['bos_bull'].diff().abs() > 0).astype(int)
        
        # Order blocks (simplified)
        features['order_block_dist'] = (close - close.rolling(20).mean()) / close
        
        return features
    
    def _market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile features (10 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        features = pd.DataFrame(index=df.index)
        
        # Point of Control (POC) - simplified
        def calc_poc(prices, volumes):
            if len(prices) == 0:
                return prices.iloc[-1] if len(prices) > 0 else 0
            hist, bins = np.histogram(prices, bins=10, weights=volumes)
            return bins[hist.argmax()]
        
        features['poc_20'] = close.rolling(20).apply(
            lambda x: calc_poc(close.iloc[max(0, len(close) - 20):len(close)], 
                               volume.iloc[max(0, len(volume) - 20):len(volume)]), 
            raw=False
        ).fillna(close)
        
        # Value Area High/Low (VAH/VAL)
        def calc_vah_val(prices, volumes):
            if len(prices) == 0:
                return prices.max(), prices.min()
            hist, bins = np.histogram(prices, bins=10, weights=volumes)
            cumsum = np.cumsum(hist)
            total = cumsum[-1]
            vah_idx = np.where(cumsum >= total * 0.7)[0][0]
            val_idx = np.where(cumsum >= total * 0.3)[0][0]
            return bins[vah_idx], bins[val_idx]
        
        vah_val = close.rolling(20).apply(
            lambda x: calc_vah_val(close.iloc[max(0, len(close) - 20):len(close)], 
                                   volume.iloc[max(0, len(volume) - 20):len(volume)])[0], 
            raw=False
        )
        features['vah_20'] = vah_val.fillna(high)
        features['val_20'] = close.rolling(20).apply(
            lambda x: calc_vah_val(close.iloc[max(0, len(close) - 20):len(close)], 
                                   volume.iloc[max(0, len(volume) - 20):len(volume)])[1], 
            raw=False
        ).fillna(low)
        
        # Distance from POC/VAH/VAL
        features['dist_from_poc'] = (close - features['poc_20']) / close
        features['dist_from_vah'] = (close - features['vah_20']) / close
        features['dist_from_val'] = (close - features['val_20']) / close
        
        # TPO count (Time Price Opportunity)
        features['tpo_balance'] = (high.rolling(20).apply(lambda x: len(np.unique(x))) / 20)
        
        # Market entropy
        features['market_entropy'] = close.rolling(20).apply(
            lambda x: -np.sum((x / x.sum()) * np.log(x / x.sum() + 1e-10)) if x.sum() > 0 else 0,
            raw=True
        )
        
        # Volume profile imbalance
        features['volume_imbalance'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-10)
        
        return features


def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV data and extract features
    
    Args:
        csv_path: Path to data_export.csv
    
    Returns:
        raw_df: Original OHLCV data
        features_df: Extracted features
    """
    print(f"📁 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_cols), \
        f"Missing required columns. Need: {required_cols}"
    
    print(f"✅ Loaded {len(df)} candles")
    
    # Extract features
    engine = FeatureEngine()
    features = engine.compute_all_features(df)
    
    return df, features


if __name__ == "__main__":
    # Test feature extraction
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data_export.csv"
    
    raw_df, features_df = load_and_prepare_data(csv_path)
    
    print(f"\n📊 Feature Summary:")
    print(f"  - Shape: {features_df.shape}")
    print(f"  - Features: {features_df.shape[1]}")
    print(f"\n  First 5 features:\n{features_df.iloc[:5, :5]}")
