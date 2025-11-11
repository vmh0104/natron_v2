"""
Natron Feature Engine - Generates ~100 Technical Indicators
Optimized for financial time series analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features from OHLCV data.
    Target: ~100 features across 11 categories.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point - generates all features.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 feature columns
        """
        features = pd.DataFrame(index=df.index)
        
        print("🔧 Generating features...")
        
        # 1. Moving Averages (13 features)
        ma_features = self._moving_average_features(df)
        features = pd.concat([features, ma_features], axis=1)
        print(f"  ✓ Moving Averages: {len(ma_features.columns)} features")
        
        # 2. Momentum Indicators (13 features)
        momentum_features = self._momentum_features(df)
        features = pd.concat([features, momentum_features], axis=1)
        print(f"  ✓ Momentum: {len(momentum_features.columns)} features")
        
        # 3. Volatility Indicators (15 features)
        volatility_features = self._volatility_features(df)
        features = pd.concat([features, volatility_features], axis=1)
        print(f"  ✓ Volatility: {len(volatility_features.columns)} features")
        
        # 4. Volume Indicators (9 features)
        volume_features = self._volume_features(df)
        features = pd.concat([features, volume_features], axis=1)
        print(f"  ✓ Volume: {len(volume_features.columns)} features")
        
        # 5. Price Patterns (8 features)
        pattern_features = self._price_pattern_features(df)
        features = pd.concat([features, pattern_features], axis=1)
        print(f"  ✓ Price Patterns: {len(pattern_features.columns)} features")
        
        # 6. Returns (8 features)
        return_features = self._return_features(df)
        features = pd.concat([features, return_features], axis=1)
        print(f"  ✓ Returns: {len(return_features.columns)} features")
        
        # 7. Trend Strength (6 features)
        trend_features = self._trend_strength_features(df)
        features = pd.concat([features, trend_features], axis=1)
        print(f"  ✓ Trend Strength: {len(trend_features.columns)} features")
        
        # 8. Statistical Features (6 features)
        stat_features = self._statistical_features(df)
        features = pd.concat([features, stat_features], axis=1)
        print(f"  ✓ Statistical: {len(stat_features.columns)} features")
        
        # 9. Support/Resistance (4 features)
        sr_features = self._support_resistance_features(df)
        features = pd.concat([features, sr_features], axis=1)
        print(f"  ✓ Support/Resistance: {len(sr_features.columns)} features")
        
        # 10. Smart Money Concepts (6 features)
        smc_features = self._smart_money_features(df)
        features = pd.concat([features, smc_features], axis=1)
        print(f"  ✓ Smart Money: {len(smc_features.columns)} features")
        
        # 11. Market Profile (10 features)
        profile_features = self._market_profile_features(df)
        features = pd.concat([features, profile_features], axis=1)
        print(f"  ✓ Market Profile: {len(profile_features.columns)} features")
        
        self.feature_names = features.columns.tolist()
        print(f"\n✅ Total features generated: {len(self.feature_names)}")
        
        return features
    
    # ==================== Moving Averages ====================
    
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Moving Average features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Simple moving averages
        features['sma_5'] = close.rolling(5).mean()
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['sma_50'] = close.rolling(50).mean()
        
        # Exponential moving averages
        features['ema_9'] = close.ewm(span=9, adjust=False).mean()
        features['ema_21'] = close.ewm(span=21, adjust=False).mean()
        
        # MA slopes
        features['sma20_slope'] = features['sma_20'].diff(5) / features['sma_20']
        features['ema21_slope'] = features['ema_21'].diff(5) / features['ema_21']
        
        # Crossovers
        features['sma10_sma20_cross'] = (features['sma_10'] - features['sma_20']) / features['sma_20']
        features['sma20_sma50_cross'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
        
        # Price relative to MAs
        features['close_sma20_ratio'] = close / features['sma_20']
        features['close_ema21_ratio'] = close / features['ema_21']
        
        # Distance from MA
        features['dist_from_sma20'] = (close - features['sma_20']) / features['sma_20']
        
        return features
    
    # ==================== Momentum ====================
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Momentum features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_7'] = self._calculate_rsi(close, 7)
        
        # Rate of Change
        features['roc_5'] = close.pct_change(5)
        features['roc_10'] = close.pct_change(10)
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        features['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Williams %R
        features['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # Momentum
        features['momentum_10'] = close - close.shift(10)
        
        return features
    
    # ==================== Volatility ====================
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Volatility features"""
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
        features['atr_ratio'] = features['atr_7'] / features['atr_14']
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = sma20 + (2 * std20)
        features['bb_middle'] = sma20
        features['bb_lower'] = sma20 - (2 * std20)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Keltner Channel
        ema20 = close.ewm(span=20, adjust=False).mean()
        features['keltner_upper'] = ema20 + (2 * features['atr_14'])
        features['keltner_lower'] = ema20 - (2 * features['atr_14'])
        features['keltner_position'] = (close - features['keltner_lower']) / (features['keltner_upper'] - features['keltner_lower'])
        
        # Standard Deviation
        features['std_10'] = close.rolling(10).std()
        features['std_20'] = close.rolling(20).std()
        
        # Historical Volatility
        returns = np.log(close / close.shift())
        features['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        return features
    
    # ==================== Volume ====================
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 Volume features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Volume ratios
        features['volume_sma20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma20']
        
        # MFI (Money Flow Index)
        mf_raw = typical_price * volume
        pos_mf = mf_raw.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        neg_mf = mf_raw.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        features['mfi_14'] = mfi
        
        # Volume delta
        features['volume_delta'] = volume.diff()
        
        # Volume trend
        features['volume_trend'] = (volume.rolling(5).mean() / volume.rolling(20).mean())
        
        return features
    
    # ==================== Price Patterns ====================
    
    def _price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Price Pattern features"""
        features = pd.DataFrame(index=df.index)
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Candle body
        features['body'] = abs(close - open_price)
        features['body_pct'] = features['body'] / open_price
        
        # Shadows
        features['upper_shadow'] = high - np.maximum(open_price, close)
        features['lower_shadow'] = np.minimum(open_price, close) - low
        
        # Doji detection
        features['is_doji'] = (features['body'] < (high - low) * 0.1).astype(float)
        
        # Gap
        features['gap'] = open_price - close.shift()
        features['gap_pct'] = features['gap'] / close.shift()
        
        # Position in range
        features['position_in_range'] = (close - low) / (high - low + 1e-8)
        
        return features
    
    # ==================== Returns ====================
    
    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Return features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        open_price = df['open']
        
        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        features['log_return_10'] = np.log(close / close.shift(10))
        
        # Intraday return
        features['intraday_return'] = (close - open_price) / open_price
        
        # Cumulative returns
        features['cum_return_20'] = (close / close.shift(20)) - 1
        features['cum_return_50'] = (close / close.shift(50)) - 1
        
        # Return volatility
        features['return_volatility'] = features['log_return_1'].rolling(20).std()
        
        # Sharpe-like ratio
        mean_return = features['log_return_1'].rolling(20).mean()
        std_return = features['log_return_1'].rolling(20).std()
        features['sharpe_ratio'] = mean_return / (std_return + 1e-8)
        
        return features
    
    # ==================== Trend Strength ====================
    
    def _trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Trend Strength features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0).where((high - high.shift()) > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0).where((low.shift() - low) > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['adx'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Aroon
        aroon_up = 100 * close.rolling(25).apply(lambda x: x.argmax()) / 25
        aroon_down = 100 * close.rolling(25).apply(lambda x: x.argmin()) / 25
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_oscillator'] = aroon_up - aroon_down
        
        return features
    
    # ==================== Statistical ====================
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Statistical features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Skewness and Kurtosis
        features['skewness_20'] = close.rolling(20).skew()
        features['kurtosis_20'] = close.rolling(20).kurt()
        
        # Z-score
        mean_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['zscore_20'] = (close - mean_20) / (std_20 + 1e-8)
        
        # Hurst Exponent (simplified)
        features['hurst_approx'] = self._calculate_hurst_approx(close)
        
        # Entropy
        features['entropy_20'] = close.rolling(20).apply(self._calculate_entropy)
        
        # Autocorrelation
        features['autocorr_5'] = close.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        return features
    
    # ==================== Support/Resistance ====================
    
    def _support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 Support/Resistance features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        features['dist_to_high20'] = (high_20 - close) / close
        features['dist_to_low20'] = (close - low_20) / close
        features['dist_to_high50'] = (high_50 - close) / close
        features['dist_to_low50'] = (close - low_50) / close
        
        return features
    
    # ==================== Smart Money Concepts ====================
    
    def _smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        features = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing highs/lows
        features['swing_high'] = self._detect_swing_high(high)
        features['swing_low'] = self._detect_swing_low(low)
        
        # Break of Structure (BOS)
        features['bos_bullish'] = self._detect_bos(close, bullish=True)
        features['bos_bearish'] = self._detect_bos(close, bullish=False)
        
        # Change of Character (CHoCH)
        features['choch'] = self._detect_choch(close)
        
        # Liquidity zones (simplified)
        features['liquidity_above'] = (high.rolling(20).max() - close) / close
        
        return features
    
    # ==================== Market Profile ====================
    
    def _market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Market Profile features"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # POC (Point of Control) - simplified
        features['poc_20'] = self._calculate_poc(close, volume, window=20)
        
        # Value Area High/Low (simplified)
        features['vah_20'] = close.rolling(20).quantile(0.7)
        features['val_20'] = close.rolling(20).quantile(0.3)
        features['dist_to_vah'] = (features['vah_20'] - close) / close
        features['dist_to_val'] = (close - features['val_20']) / close
        
        # Price distribution entropy
        features['price_entropy'] = close.rolling(20).apply(self._calculate_entropy)
        
        # Volume profile
        features['volume_at_price'] = volume.rolling(20).sum()
        
        # Range expansion/contraction
        range_20 = (high - low).rolling(20).mean()
        features['range_expansion'] = (high - low) / range_20
        
        # Balance/Imbalance
        features['price_balance'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        features['volume_imbalance'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        
        return features
    
    # ==================== Helper Methods ====================
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_hurst_approx(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Simplified Hurst exponent calculation"""
        def hurst(x):
            if len(x) < 20:
                return 0.5
            lags = range(2, min(20, len(x)//2))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            if any(t == 0 for t in tau):
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        return series.rolling(window).apply(hurst)
    
    def _calculate_entropy(self, x):
        """Shannon entropy of price distribution"""
        if len(x) < 2:
            return 0
        hist, _ = np.histogram(x, bins=10)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    def _detect_swing_high(self, high: pd.Series, window: int = 5) -> pd.Series:
        """Detect swing highs"""
        is_swing = pd.Series(0.0, index=high.index)
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                is_swing.iloc[i] = 1.0
        return is_swing
    
    def _detect_swing_low(self, low: pd.Series, window: int = 5) -> pd.Series:
        """Detect swing lows"""
        is_swing = pd.Series(0.0, index=low.index)
        for i in range(window, len(low) - window):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                is_swing.iloc[i] = 1.0
        return is_swing
    
    def _detect_bos(self, close: pd.Series, bullish: bool = True) -> pd.Series:
        """Break of Structure detection"""
        bos = pd.Series(0.0, index=close.index)
        if bullish:
            prev_high = close.rolling(10).max().shift()
            bos = (close > prev_high).astype(float)
        else:
            prev_low = close.rolling(10).min().shift()
            bos = (close < prev_low).astype(float)
        return bos
    
    def _detect_choch(self, close: pd.Series) -> pd.Series:
        """Change of Character detection (simplified)"""
        trend = (close - close.shift(10)) / close.shift(10)
        trend_change = (trend.diff().abs() > 0.01).astype(float)
        return trend_change
    
    def _calculate_poc(self, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Point of Control - price with most volume"""
        def poc(idx):
            if idx < window:
                return close.iloc[idx]
            prices = close.iloc[idx-window:idx]
            vols = volume.iloc[idx-window:idx]
            return prices.iloc[vols.argmax()]
        
        return pd.Series([poc(i) for i in range(len(close))], index=close.index)
