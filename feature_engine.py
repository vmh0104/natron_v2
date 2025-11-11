"""
FeatureEngine: Generates ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features for financial time series.
    Output: ~100 features grouped into categories.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all feature groups from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 technical features
        """
        features_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Generate feature groups
        features_df = self._add_moving_averages(features_df)
        features_df = self._add_momentum(features_df)
        features_df = self._add_volatility(features_df)
        features_df = self._add_volume_features(features_df)
        features_df = self._add_price_patterns(features_df)
        features_df = self._add_returns(features_df)
        features_df = self._add_trend_strength(features_df)
        features_df = self._add_statistical(features_df)
        features_df = self._add_support_resistance(features_df)
        features_df = self._add_smart_money_concepts(features_df)
        features_df = self._add_market_profile(features_df)
        
        # Select only feature columns (exclude original OHLCV)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return features_df[feature_cols]
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Moving Average features"""
        close = df['close']
        
        # Simple MAs
        df['MA5'] = close.rolling(5).mean()
        df['MA10'] = close.rolling(10).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()
        df['MA100'] = close.rolling(100).mean()
        
        # Exponential MAs
        df['EMA12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA26'] = close.ewm(span=26, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        
        # MA slopes
        df['MA20_slope'] = df['MA20'].diff(5) / df['MA20']
        df['MA50_slope'] = df['MA50'].diff(10) / df['MA50']
        
        # Crossovers
        df['MA5_MA20_cross'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA20_MA50_cross'] = (df['MA20'] > df['MA50']).astype(int)
        df['MA_ratio'] = df['MA20'] / df['MA50']
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Momentum features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ROC (Rate of Change)
        df['ROC10'] = close.pct_change(10) * 100
        df['ROC20'] = close.pct_change(20) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Momentum
        df['Momentum10'] = close.diff(10)
        df['Momentum20'] = close.diff(20)
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - close) / (high_14 - low_14))
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Volatility features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
        df['ATR20'] = tr.rolling(20).mean()
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_upper'] = ma20 + (2 * std20)
        df['BB_lower'] = ma20 - (2 * std20)
        df['BB_mid'] = ma20
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / ma20
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        df['KC_upper'] = ema20 + (1.5 * df['ATR20'])
        df['KC_lower'] = ema20 - (1.5 * df['ATR20'])
        df['KC_position'] = (close - df['KC_lower']) / (df['KC_upper'] - df['KC_lower'])
        
        # Standard Deviation
        df['StdDev10'] = close.rolling(10).std()
        df['StdDev20'] = close.rolling(20).std()
        df['StdDev50'] = close.rolling(50).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 Volume features"""
        close = df['close']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # VWAP approximation
        typical_price = (df['high'] + df['low'] + close) / 3
        df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # MFI (Money Flow Index)
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        df['Volume_MA20'] = volume.rolling(20).mean()
        df['Volume_ratio'] = volume / df['Volume_MA20']
        df['Volume_MA50'] = volume.rolling(50).mean()
        df['Volume_ratio50'] = volume / df['Volume_MA50']
        
        # Volume price trend
        df['VPT'] = (close.pct_change() * volume).fillna(0).cumsum()
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Price pattern features"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadow calculations
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low
        
        # Body percentage
        df['Body_pct'] = body / (total_range + 1e-8)
        
        # Doji pattern (small body)
        df['Doji'] = (df['Body_pct'] < 0.1).astype(int)
        
        # Gaps
        df['Gap_up'] = ((low - high.shift()) > 0).astype(int)
        df['Gap_down'] = ((high.shift() - low) > 0).astype(int)
        
        # Shadow ratios
        df['Upper_shadow_ratio'] = upper_shadow / (total_range + 1e-8)
        df['Lower_shadow_ratio'] = lower_shadow / (total_range + 1e-8)
        
        # Position in range
        df['Position_in_range'] = (close - low) / (total_range + 1e-8)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Return features"""
        close = df['close']
        open_price = df['open']
        
        # Log returns
        df['Log_return1'] = np.log(close / close.shift(1))
        df['Log_return5'] = np.log(close / close.shift(5))
        df['Log_return10'] = np.log(close / close.shift(10))
        df['Log_return20'] = np.log(close / close.shift(20))
        
        # Intraday return
        df['Intraday_return'] = (close - open_price) / open_price
        
        # Cumulative returns
        df['Cumulative_return5'] = close.pct_change(5)
        df['Cumulative_return10'] = close.pct_change(10)
        df['Cumulative_return20'] = close.pct_change(20)
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Trend strength features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._true_range(df)
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # Aroon
        period = 14
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        
        return df
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Statistical features"""
        close = df['close']
        
        # Rolling statistics
        rolling20 = close.rolling(20)
        rolling50 = close.rolling(50)
        
        df['Skewness20'] = rolling20.apply(lambda x: x.skew())
        df['Kurtosis20'] = rolling20.apply(lambda x: x.kurtosis())
        
        df['Skewness50'] = rolling50.apply(lambda x: x.skew())
        df['Kurtosis50'] = rolling50.apply(lambda x: x.kurtosis())
        
        # Z-score
        mean20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['Z_score20'] = (close - mean20) / (std20 + 1e-8)
        
        # Hurst exponent approximation (simplified)
        returns = close.pct_change().dropna()
        if len(returns) > 50:
            lags = range(2, min(20, len(returns) // 2))
            tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            df['Hurst'] = poly[0] * 2
        else:
            df['Hurst'] = 0.5
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 Support/Resistance features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Distance to recent highs/lows
        high20 = high.rolling(20).max()
        low20 = low.rolling(20).min()
        high50 = high.rolling(50).max()
        low50 = low.rolling(50).min()
        
        df['Dist_to_high20'] = (high20 - close) / close
        df['Dist_to_low20'] = (close - low20) / close
        df['Dist_to_high50'] = (high50 - close) / close
        df['Dist_to_low50'] = (close - low50) / close
        
        return df
    
    def _add_smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        df['Swing_High'] = (high == high.rolling(window * 2 + 1, center=True).max()).astype(int)
        df['Swing_Low'] = (low == low.rolling(window * 2 + 1, center=True).min()).astype(int)
        
        # Break of Structure (BOS) - simplified
        rolling_max = high.rolling(20).max()
        rolling_min = low.rolling(20).min()
        df['BOS_up'] = ((close > rolling_max.shift(1)) & (close.shift(1) <= rolling_max.shift(1))).astype(int)
        df['BOS_down'] = ((close < rolling_min.shift(1)) & (close.shift(1) >= rolling_min.shift(1))).astype(int)
        
        # Change of Character (CHoCH) - simplified
        df['CHoCH'] = ((df['BOS_up'] == 1) | (df['BOS_down'] == 1)).astype(int)
        
        # Order block approximation
        df['Order_block_bullish'] = ((low == low.rolling(5).min()) & (close > open_price)).astype(int)
        df['Order_block_bearish'] = ((high == high.rolling(5).max()) & (close < open_price)).astype(int)
        
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
        
        # POC (Point of Control) approximation
        typical_price = (high + low + close) / 3
        df['POC'] = typical_price.rolling(20).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean())
        
        # VAH/VAL (Value Area High/Low) approximation
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        df['VAH'] = rolling_high * 0.7 + rolling_low * 0.3
        df['VAL'] = rolling_high * 0.3 + rolling_low * 0.7
        
        # Position relative to value area
        df['Above_VAH'] = (close > df['VAH']).astype(int)
        df['Below_VAL'] = (close < df['VAL']).astype(int)
        df['In_VA'] = ((close >= df['VAL']) & (close <= df['VAH'])).astype(int)
        
        # Entropy approximation (price distribution)
        returns = close.pct_change().dropna()
        if len(returns) > 20:
            df['Entropy'] = returns.rolling(20).apply(lambda x: -np.sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-8)))
        else:
            df['Entropy'] = 0
        
        # Volume profile features
        df['Volume_price_corr'] = volume.rolling(20).corr(close)
        df['High_volume_node'] = (volume > volume.rolling(20).quantile(0.75)).astype(int)
        df['Low_volume_node'] = (volume < volume.rolling(20).quantile(0.25)).astype(int)
        
        return df
