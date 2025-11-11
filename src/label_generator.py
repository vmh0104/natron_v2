"""
Natron Label Generator V2 - Bias-Reduced Institutional Labeling
Generates balanced labels for Buy/Sell/Direction/Regime classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """
    Advanced label generation with bias reduction and institutional logic.
    Produces balanced labels for multi-task learning.
    """
    
    def __init__(self, 
                 neutral_buffer: float = 0.001,
                 lookforward: int = 3,
                 volume_threshold: float = 1.5,
                 balance_threshold: float = 0.05,
                 adaptive_balancing: bool = True):
        """
        Args:
            neutral_buffer: Buffer zone for direction labeling (default 0.001 = 0.1%)
            lookforward: Candles to look ahead for direction (default 3)
            volume_threshold: Volume spike threshold multiplier
            balance_threshold: Max acceptable class imbalance ratio
            adaptive_balancing: Enable dynamic threshold adjustment
        """
        self.neutral_buffer = neutral_buffer
        self.lookforward = lookforward
        self.volume_threshold = volume_threshold
        self.balance_threshold = balance_threshold
        self.adaptive_balancing = adaptive_balancing
        
    def generate_labels(self, 
                       df: pd.DataFrame, 
                       features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        
        Args:
            df: Original OHLCV dataframe
            features: Feature dataframe from FeatureEngine
            
        Returns:
            DataFrame with labels [buy, sell, direction, regime]
        """
        print("\n🏷️  Generating labels (V2 - Bias-Reduced)...")
        
        labels = pd.DataFrame(index=df.index)
        
        # Generate each label type
        labels['buy'] = self._generate_buy_signals(df, features)
        labels['sell'] = self._generate_sell_signals(df, features)
        labels['direction'] = self._generate_direction_labels(df)
        labels['regime'] = self._generate_regime_labels(df, features)
        
        # Apply adaptive balancing if enabled
        if self.adaptive_balancing:
            labels = self._balance_labels(labels)
        
        # Print distribution statistics
        self._print_statistics(labels)
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        BUY signal if >= 2 conditions true:
        1. close > MA20 > MA50
        2. RSI > 50 or just crossed up from <30
        3. close > BB midband and MA20 slope > 0
        4. volume > 1.5x rolling20
        5. position_in_range >= 0.7
        6. MACD_hist > 0 and rising
        """
        close = df['close'].values
        n = len(close)
        buy_conditions = np.zeros((n, 6), dtype=int)
        
        # Condition 1: close > MA20 > MA50
        if 'ma_20' in features.columns and 'ma_50' in features.columns:
            ma_20 = features['ma_20'].values
            ma_50 = features['ma_50'].values
            buy_conditions[:, 0] = (close > ma_20) & (ma_20 > ma_50)
        
        # Condition 2: RSI > 50 or crossed up from <30
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].values
            rsi_prev = np.roll(rsi, 1)
            buy_conditions[:, 1] = (rsi > 50) | ((rsi > 30) & (rsi_prev < 30))
        
        # Condition 3: close > BB mid and MA20 slope > 0
        if 'bb_mid_20' in features.columns and 'ma_20_slope' in features.columns:
            bb_mid = features['bb_mid_20'].values
            ma_slope = features['ma_20_slope'].values
            buy_conditions[:, 2] = (close > bb_mid) & (ma_slope > 0)
        
        # Condition 4: volume > 1.5x rolling20
        if 'volume_ratio' in features.columns:
            vol_ratio = features['volume_ratio'].values
            buy_conditions[:, 3] = vol_ratio > self.volume_threshold
        
        # Condition 5: position_in_range >= 0.7
        if 'position_in_range' in features.columns:
            pos_range = features['position_in_range'].values
            buy_conditions[:, 4] = pos_range >= 0.7
        
        # Condition 6: MACD_hist > 0 and rising
        if 'macd_hist' in features.columns and 'macd_hist_slope' in features.columns:
            macd_hist = features['macd_hist'].values
            macd_slope = features['macd_hist_slope'].values
            buy_conditions[:, 5] = (macd_hist > 0) & (macd_slope > 0)
        
        # Buy if >= 2 conditions met
        buy_score = buy_conditions.sum(axis=1)
        buy_signal = (buy_score >= 2).astype(int)
        
        return pd.Series(buy_signal, index=df.index)
    
    def _generate_sell_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        SELL signal if >= 2 conditions true:
        1. close < MA20 < MA50
        2. RSI < 50 or just turned down from >70
        3. close < BB midband and MA20 slope < 0
        4. volume > 1.5x rolling20 and position_in_range <= 0.3
        5. MACD_hist < 0 and falling
        6. minus_DI > plus_DI
        """
        close = df['close'].values
        n = len(close)
        sell_conditions = np.zeros((n, 6), dtype=int)
        
        # Condition 1: close < MA20 < MA50
        if 'ma_20' in features.columns and 'ma_50' in features.columns:
            ma_20 = features['ma_20'].values
            ma_50 = features['ma_50'].values
            sell_conditions[:, 0] = (close < ma_20) & (ma_20 < ma_50)
        
        # Condition 2: RSI < 50 or turned down from >70
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].values
            rsi_prev = np.roll(rsi, 1)
            sell_conditions[:, 1] = (rsi < 50) | ((rsi < 70) & (rsi_prev > 70))
        
        # Condition 3: close < BB mid and MA20 slope < 0
        if 'bb_mid_20' in features.columns and 'ma_20_slope' in features.columns:
            bb_mid = features['bb_mid_20'].values
            ma_slope = features['ma_20_slope'].values
            sell_conditions[:, 2] = (close < bb_mid) & (ma_slope < 0)
        
        # Condition 4: high volume + low position
        if 'volume_ratio' in features.columns and 'position_in_range' in features.columns:
            vol_ratio = features['volume_ratio'].values
            pos_range = features['position_in_range'].values
            sell_conditions[:, 3] = (vol_ratio > self.volume_threshold) & (pos_range <= 0.3)
        
        # Condition 5: MACD_hist < 0 and falling
        if 'macd_hist' in features.columns and 'macd_hist_slope' in features.columns:
            macd_hist = features['macd_hist'].values
            macd_slope = features['macd_hist_slope'].values
            sell_conditions[:, 4] = (macd_hist < 0) & (macd_slope < 0)
        
        # Condition 6: minus_DI > plus_DI
        if 'minus_di' in features.columns and 'plus_di' in features.columns:
            minus_di = features['minus_di'].values
            plus_di = features['plus_di'].values
            sell_conditions[:, 5] = minus_di > plus_di
        
        # Sell if >= 2 conditions met
        sell_score = sell_conditions.sum(axis=1)
        sell_signal = (sell_score >= 2).astype(int)
        
        return pd.Series(sell_signal, index=df.index)
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        3-class balanced direction labeling:
        1 = Up: close[i+lookforward] > close[i] + neutral_buffer
        0 = Down: close[i+lookforward] < close[i] - neutral_buffer
        2 = Neutral: otherwise
        """
        close = df['close'].values
        n = len(close)
        direction = np.full(n, 2, dtype=int)  # Default to neutral
        
        for i in range(n - self.lookforward):
            current_close = close[i]
            future_close = close[i + self.lookforward]
            
            threshold = current_close * self.neutral_buffer
            
            if future_close > current_close + threshold:
                direction[i] = 1  # Up
            elif future_close < current_close - threshold:
                direction[i] = 0  # Down
            # else: remains 2 (Neutral)
        
        return pd.Series(direction, index=df.index)
    
    def _generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        6-class regime classification:
        0: BULL_STRONG (trend > +2%, ADX > 25)
        1: BULL_WEAK (0 < trend <= +2%, ADX <= 25)
        2: RANGE (lateral market - default)
        3: BEAR_WEAK (-2% <= trend < 0, ADX <= 25)
        4: BEAR_STRONG (trend < -2%, ADX > 25)
        5: VOLATILE (ATR > 90th percentile or volume spike)
        """
        close = df['close'].values
        n = len(close)
        regime = np.full(n, 2, dtype=int)  # Default to RANGE
        
        # Calculate trend (20-period return)
        trend = np.full(n, 0.0)
        for i in range(20, n):
            trend[i] = (close[i] - close[i-20]) / close[i-20]
        
        # Get ADX
        adx = features['adx_14'].values if 'adx_14' in features.columns else np.zeros(n)
        
        # Get ATR percentile
        atr = features['atr_14'].values if 'atr_14' in features.columns else np.zeros(n)
        atr_90 = np.percentile(atr[20:], 90) if len(atr) > 20 else 0
        
        # Get volume ratio
        vol_ratio = features['volume_ratio'].values if 'volume_ratio' in features.columns else np.ones(n)
        
        for i in range(20, n):
            # Check volatile condition first (highest priority)
            if atr[i] > atr_90 or vol_ratio[i] > 2.0:
                regime[i] = 5  # VOLATILE
                continue
            
            t = trend[i]
            a = adx[i]
            
            # Bull regimes
            if t > 0.02 and a > 25:
                regime[i] = 0  # BULL_STRONG
            elif t > 0 and t <= 0.02:
                regime[i] = 1  # BULL_WEAK
            
            # Bear regimes
            elif t < -0.02 and a > 25:
                regime[i] = 4  # BEAR_STRONG
            elif t < 0 and t >= -0.02:
                regime[i] = 3  # BEAR_WEAK
            
            # else: remains 2 (RANGE)
        
        return pd.Series(regime, index=df.index)
    
    def _balance_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adaptive balancing to reduce bias.
        Adds stochastic perturbation to over-represented classes.
        """
        balanced = labels.copy()
        
        # Balance Buy/Sell signals
        buy_ratio = labels['buy'].mean()
        sell_ratio = labels['sell'].mean()
        
        # If Buy is over-represented, randomly flip some to 0
        if buy_ratio > 0.4:
            excess_ratio = buy_ratio - 0.35
            buy_indices = labels[labels['buy'] == 1].index
            flip_count = int(len(buy_indices) * excess_ratio / buy_ratio)
            flip_indices = np.random.choice(buy_indices, size=flip_count, replace=False)
            balanced.loc[flip_indices, 'buy'] = 0
        
        # If Sell is over-represented, randomly flip some to 0
        if sell_ratio > 0.4:
            excess_ratio = sell_ratio - 0.35
            sell_indices = labels[labels['sell'] == 1].index
            flip_count = int(len(sell_indices) * excess_ratio / sell_ratio)
            flip_indices = np.random.choice(sell_indices, size=flip_count, replace=False)
            balanced.loc[flip_indices, 'sell'] = 0
        
        return balanced
    
    def _print_statistics(self, labels: pd.DataFrame):
        """Print comprehensive label distribution statistics"""
        print("\n" + "="*50)
        print("📊 LABEL DISTRIBUTION SUMMARY")
        print("="*50)
        
        for col in ['buy', 'sell', 'direction', 'regime']:
            if col in labels.columns:
                vc = labels[col].value_counts(normalize=True).sort_index()
                print(f"\n▶ {col.upper()} distribution:")
                
                if col == 'regime':
                    regime_names = {
                        0: "BULL_STRONG",
                        1: "BULL_WEAK",
                        2: "RANGE",
                        3: "BEAR_WEAK",
                        4: "BEAR_STRONG",
                        5: "VOLATILE"
                    }
                    for idx, val in vc.items():
                        print(f"  {int(idx)} ({regime_names.get(int(idx), 'Unknown'):12s}): {val:.3f}")
                elif col == 'direction':
                    dir_names = {0: "Down", 1: "Up", 2: "Neutral"}
                    for idx, val in vc.items():
                        print(f"  {int(idx)} ({dir_names.get(int(idx), 'Unknown'):7s}): {val:.3f}")
                else:
                    for idx, val in vc.items():
                        print(f"  {int(idx)}: {val:.3f}")
        
        # Calculate balance metrics
        print("\n" + "="*50)
        print("⚖️  BALANCE METRICS")
        print("="*50)
        
        buy_sell_ratio = labels['buy'].mean() / (labels['sell'].mean() + 1e-6)
        print(f"Buy/Sell Ratio: {buy_sell_ratio:.2f}")
        
        direction_counts = labels['direction'].value_counts(normalize=True)
        max_dir = direction_counts.max()
        min_dir = direction_counts.min()
        dir_balance = min_dir / (max_dir + 1e-6)
        print(f"Direction Balance Score: {dir_balance:.2f} (1.0 = perfect)")
        
        regime_counts = labels['regime'].value_counts(normalize=True)
        regime_entropy = -(regime_counts * np.log(regime_counts + 1e-9)).sum()
        print(f"Regime Entropy: {regime_entropy:.2f} (max: {np.log(6):.2f})")
        
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test label generation
    print("Testing LabelGeneratorV2...")
    
    # Create sample data
    from feature_engine import FeatureEngine
    
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    
    # Create synthetic price data with trend
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    base_price = 100 + trend + noise
    
    df = pd.DataFrame({
        'time': dates,
        'open': base_price + np.random.randn(n) * 0.2,
        'high': base_price + np.random.rand(n) * 1.5,
        'low': base_price - np.random.rand(n) * 1.5,
        'close': base_price,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Extract features
    engine = FeatureEngine()
    features = engine.extract_all_features(df)
    
    # Generate labels
    label_gen = LabelGeneratorV2()
    labels = label_gen.generate_labels(df, features)
    
    print(f"\n✅ Label generation complete!")
    print(f"Shape: {labels.shape}")
    print(f"\nSample labels:\n{labels.head(20)}")
