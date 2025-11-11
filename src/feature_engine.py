"""
FeatureEngine: Generates ~100 technical features from OHLCV data
Groups: MA, Momentum, Volatility, Volume, Price Pattern, Returns, Trend, Statistical, S/R, SMC, Market Profile
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """Generates comprehensive technical features for financial time series"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all ~100 technical features
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with original columns + ~100 feature columns
        """
        features_df = df.copy()
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Fill NaN values
        features_df = features_df.ffill().bfill()
        
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
        
        # Drop original OHLCV columns (keep time for reference)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return features_df[['time'] + feature_cols]
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 MA features"""
        close = df['close']
        
        # Simple MAs
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA{period}'] = close.rolling(period).mean()
            df[f'EMA{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MA ratios and crossovers
        df['MA20_MA50_ratio'] = df['MA20'] / (df['MA50'] + 1e-8)
        df['MA5_MA20_ratio'] = df['MA5'] / (df['MA20'] + 1e-8)
        df['MA20_slope'] = df['MA20'].diff(5)
        df['MA50_slope'] = df['MA50'].diff(10)
        df['MA_cross_above'] = ((df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))).astype(float)
        df['MA_cross_below'] = ((df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))).astype(float)
        
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
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_signal'] = ((df['RSI'] > 50) & (df['RSI'].shift(1) <= 50)).astype(float)
        
        # ROC
        df['ROC_10'] = close.pct_change(10) * 100
        df['ROC_20'] = close.pct_change(20) * 100
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-8))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_change'] = df['MACD_hist'].diff()
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Volatility features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        df['ATR_20'] = tr.rolling(20).mean()
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_upper'] = ma20 + (std20 * 2)
        df['BB_mid'] = ma20
        df['BB_lower'] = ma20 - (std20 * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_mid'] + 1e-8)
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
        
        # Keltner Channels
        kc_middle = close.ewm(span=20, adjust=False).mean()
        kc_range = (high - low).rolling(20).mean()
        df['KC_upper'] = kc_middle + (kc_range * 1.5)
        df['KC_lower'] = kc_middle - (kc_range * 1.5)
        df['KC_position'] = (close - df['KC_lower']) / (df['KC_upper'] - df['KC_lower'] + 1e-8)
        
        # Standard Deviation
        df['StdDev_20'] = close.rolling(20).std()
        df['StdDev_50'] = close.rolling(50).std()
        df['Volatility_ratio'] = df['StdDev_20'] / (df['StdDev_50'] + 1e-8)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 Volume features"""
        close = df['close']
        volume = df['volume']
        
        # OBV
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_change'] = df['OBV'].diff(5)
        
        # VWAP approximation
        typical_price = (df['high'] + df['low'] + close) / 3
        df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['VWAP_distance'] = (close - df['VWAP']) / (df['VWAP'] + 1e-8)
        
        # MFI
        money_flow = typical_price * volume
        positive_flow = money_flow.where(close > close.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(close < close.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-8)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        df['Volume_MA20'] = volume.rolling(20).mean()
        df['Volume_ratio'] = volume / (df['Volume_MA20'] + 1e-8)
        df['Volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Price pattern features"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low
        
        df['Body_pct'] = body / (total_range + 1e-8)
        df['Upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        df['Lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern
        df['Doji'] = (body / (total_range + 1e-8) < 0.1).astype(float)
        
        # Gaps
        df['Gap_up'] = ((low > high.shift(1)) & (close > open_price)).astype(float)
        df['Gap_down'] = ((high < low.shift(1)) & (close < open_price)).astype(float)
        
        # Position in range
        df['Position_in_range'] = (close - low) / (total_range + 1e-8)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Return features"""
        close = df['close']
        
        # Log returns
        df['Log_return_1'] = np.log(close / close.shift(1))
        df['Log_return_5'] = np.log(close / close.shift(5))
        df['Log_return_20'] = np.log(close / close.shift(20))
        
        # Intraday return
        df['Intraday_return'] = (close - df['open']) / (df['open'] + 1e-8)
        
        # Cumulative returns
        df['Cumulative_return_5'] = close.pct_change(5)
        df['Cumulative_return_20'] = close.pct_change(20)
        
        # Return volatility
        returns = close.pct_change()
        df['Return_volatility'] = returns.rolling(20).std()
        df['Return_skew'] = returns.rolling(20).skew()
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Trend strength features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ADX approximation
        tr = self._true_range(df)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # Aroon
        period = 14
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        df['Aroon_Oscillator'] = aroon_up - aroon_down
        
        return df
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Statistical features"""
        close = df['close']
        
        # Rolling statistics
        window = 20
        df['Skewness_20'] = close.rolling(window).skew()
        df['Kurtosis_20'] = close.rolling(window).kurtosis()
        
        # Z-score
        mean_20 = close.rolling(window).mean()
        std_20 = close.rolling(window).std()
        df['Z_score'] = (close - mean_20) / (std_20 + 1e-8)
        
        # Hurst exponent approximation
        def hurst_approx(ts):
            if len(ts) < 10:
                return 0.5
            lags = range(2, min(10, len(ts)))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['Hurst'] = close.rolling(50).apply(hurst_approx, raw=True)
        
        # Price position
        df['Price_position_20'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-8)
        df['Price_position_50'] = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min() + 1e-8)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 Support/Resistance features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        df['Dist_to_high_20'] = (high_20 - close) / (close + 1e-8)
        df['Dist_to_low_20'] = (close - low_20) / (close + 1e-8)
        df['Dist_to_high_50'] = (high_50 - close) / (close + 1e-8)
        df['Dist_to_low_50'] = (close - low_50) / (close + 1e-8)
        
        return df
    
    def _add_smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        df['Swing_High'] = ((high == high.rolling(window * 2 + 1, center=True).max()) & 
                           (high.shift(-window) < high) & (high.shift(window) < high)).astype(float)
        df['Swing_Low'] = ((low == low.rolling(window * 2 + 1, center=True).min()) & 
                          (low.shift(-window) > low) & (low.shift(window) > low)).astype(float)
        
        # Break of Structure (BOS) - price breaks previous swing high/low
        swing_highs = high.where(df['Swing_High'] == 1)
        swing_lows = low.where(df['Swing_Low'] == 1)
        
        last_swing_high = swing_highs.ffill()
        last_swing_low = swing_lows.ffill()
        
        df['BOS_up'] = (close > last_swing_high.shift(1)).astype(float)
        df['BOS_down'] = (close < last_swing_low.shift(1)).astype(float)
        
        # Change of Character (CHoCH) - trend reversal signal
        df['CHoCH'] = ((df['BOS_up'] == 1) | (df['BOS_down'] == 1)).astype(float)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Market Profile features"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price range bins
        num_bins = 20
        price_range = high.max() - low.min()
        bin_size = price_range / num_bins
        
        # Volume profile approximation
        def volume_profile(row_idx):
            if row_idx < 20:
                return np.zeros(10)
            window_data = df.iloc[row_idx-20:row_idx]
            bins = np.linspace(window_data['low'].min(), window_data['high'].max(), num_bins)
            hist, _ = np.histogram(close.iloc[row_idx-20:row_idx], bins=bins, 
                                  weights=volume.iloc[row_idx-20:row_idx])
            return hist
        
        # POC (Point of Control) - price level with highest volume
        window = 20
        df['POC_distance'] = 0.0
        df['VAH'] = 0.0  # Value Area High
        df['VAL'] = 0.0  # Value Area Low
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            if len(window_data) > 0:
                bins = np.linspace(window_data['low'].min(), window_data['high'].max(), num_bins)
                hist, bin_edges = np.histogram(window_data['close'], bins=bins, 
                                             weights=window_data['volume'])
                if hist.sum() > 0:
                    poc_idx = np.argmax(hist)
                    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
                    df.loc[df.index[i], 'POC_distance'] = (close.iloc[i] - poc_price) / (close.iloc[i] + 1e-8)
                    
                    # Value Area (70% of volume)
                    cumsum = np.cumsum(np.sort(hist)[::-1])
                    value_area_vol = cumsum[cumsum <= cumsum[-1] * 0.7]
                    if len(value_area_vol) > 0:
                        val_idx = len(value_area_vol)
                        sorted_indices = np.argsort(hist)[::-1][:val_idx]
                        df.loc[df.index[i], 'VAH'] = bin_edges[sorted_indices.max() + 1]
                        df.loc[df.index[i], 'VAL'] = bin_edges[sorted_indices.min()]
        
        # Entropy (price distribution measure)
        returns = close.pct_change().dropna()
        df['Entropy'] = returns.rolling(20).apply(
            lambda x: -np.sum((x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-8))), 
            raw=False
        )
        
        # Additional profile features
        df['Profile_width'] = (df['VAH'] - df['VAL']) / (close + 1e-8)
        df['Profile_position'] = (close - df['VAL']) / (df['VAH'] - df['VAL'] + 1e-8)
        df['Volume_cluster'] = volume.rolling(10).std() / (volume.rolling(10).mean() + 1e-8)
        df['Price_efficiency'] = abs(close - df['open']) / (high - low + 1e-8)
        
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
