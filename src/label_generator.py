"""
LabelGeneratorV2: Bias-reduced institutional labeling for multi-task learning
Generates: Buy/Sell signals, Direction (3-class), Regime (6-class)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """Generates balanced, bias-reduced labels for trading signals"""
    
    def __init__(self, 
                 neutral_buffer: float = 0.001,
                 buy_threshold: int = 2,
                 sell_threshold: int = 2,
                 balance_labels: bool = True,
                 stochastic_perturbation: float = 0.05):
        """
        Args:
            neutral_buffer: Buffer for neutral direction classification
            buy_threshold: Minimum conditions for BUY signal
            sell_threshold: Minimum conditions for SELL signal
            balance_labels: Whether to balance label distributions
            stochastic_perturbation: Random perturbation to avoid overfitting
        """
        self.neutral_buffer = neutral_buffer
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.balance_labels = balance_labels
        self.stochastic_perturbation = stochastic_perturbation
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        
        Args:
            df: DataFrame with OHLCV + features
            
        Returns:
            DataFrame with label columns
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate individual labels
        labels['buy'] = self._generate_buy_signals(df)
        labels['sell'] = self._generate_sell_signals(df)
        labels['direction'] = self._generate_direction(df)
        labels['regime'] = self._generate_regime(df)
        
        # Balance labels if requested
        if self.balance_labels:
            labels = self._balance_labels(labels, df)
        
        # Print statistics
        self._print_label_statistics(labels)
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate BUY signals (≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close > MA20 > MA50
        if 'MA20' in df.columns and 'MA50' in df.columns:
            cond1 = (df['close'] > df['MA20']) & (df['MA20'] > df['MA50'])
            conditions.append(cond1.astype(float))
        
        # Condition 2: RSI > 50 or just crossed up from <30
        if 'RSI' in df.columns:
            rsi_above = (df['RSI'] > 50).astype(float)
            rsi_cross_up = ((df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)).astype(float)
            cond2 = (rsi_above + rsi_cross_up).clip(0, 1)
            conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20 slope > 0
        if 'BB_mid' in df.columns and 'MA20_slope' in df.columns:
            cond3 = ((df['close'] > df['BB_mid']) & (df['MA20_slope'] > 0)).astype(float)
            conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        if 'Volume_MA20' in df.columns:
            cond4 = (df['volume'] > 1.5 * df['Volume_MA20']).astype(float)
            conditions.append(cond4)
        
        # Condition 5: position_in_range ≥ 0.7
        if 'Position_in_range' in df.columns:
            cond5 = (df['Position_in_range'] >= 0.7).astype(float)
            conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and rising
        if 'MACD_hist' in df.columns:
            macd_positive = (df['MACD_hist'] > 0).astype(float)
            macd_rising = (df['MACD_hist'] > df['MACD_hist'].shift(1)).astype(float)
            cond6 = (macd_positive * macd_rising)
            conditions.append(cond6)
        
        # Count conditions met
        if conditions:
            condition_count = sum(conditions)
            buy_signal = (condition_count >= self.buy_threshold).astype(float)
        else:
            buy_signal = pd.Series(0.0, index=df.index)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.random(len(buy_signal)) < self.stochastic_perturbation
            buy_signal = buy_signal ^ noise.astype(float)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SELL signals (≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close < MA20 < MA50
        if 'MA20' in df.columns and 'MA50' in df.columns:
            cond1 = (df['close'] < df['MA20']) & (df['MA20'] < df['MA50'])
            conditions.append(cond1.astype(float))
        
        # Condition 2: RSI < 50 or just turned down from >70
        if 'RSI' in df.columns:
            rsi_below = (df['RSI'] < 50).astype(float)
            rsi_cross_down = ((df['RSI'] < 70) & (df['RSI'].shift(1) >= 70)).astype(float)
            cond2 = (rsi_below + rsi_cross_down).clip(0, 1)
            conditions.append(cond2)
        
        # Condition 3: close < BB midband and MA20 slope < 0
        if 'BB_mid' in df.columns and 'MA20_slope' in df.columns:
            cond3 = ((df['close'] < df['BB_mid']) & (df['MA20_slope'] < 0)).astype(float)
            conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        if 'Volume_MA20' in df.columns and 'Position_in_range' in df.columns:
            cond4 = ((df['volume'] > 1.5 * df['Volume_MA20']) & 
                    (df['Position_in_range'] <= 0.3)).astype(float)
            conditions.append(cond4)
        
        # Condition 5: MACD_hist < 0 and falling
        if 'MACD_hist' in df.columns:
            macd_negative = (df['MACD_hist'] < 0).astype(float)
            macd_falling = (df['MACD_hist'] < df['MACD_hist'].shift(1)).astype(float)
            cond5 = (macd_negative * macd_falling)
            conditions.append(cond5)
        
        # Condition 6: minus_DI > plus_DI
        if 'Minus_DI' in df.columns and 'Plus_DI' in df.columns:
            cond6 = (df['Minus_DI'] > df['Plus_DI']).astype(float)
            conditions.append(cond6)
        
        # Count conditions met
        if conditions:
            condition_count = sum(conditions)
            sell_signal = (condition_count >= self.sell_threshold).astype(float)
        else:
            sell_signal = pd.Series(0.0, index=df.index)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.random(len(sell_signal)) < self.stochastic_perturbation
            sell_signal = sell_signal ^ noise.astype(float)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame) -> pd.Series:
        """Generate 3-class direction labels (Up=1, Down=0, Neutral=2)"""
        close = df['close']
        direction = pd.Series(2, index=df.index, dtype=int)  # Default: Neutral
        
        # Forward-looking: compare close[i+3] with close[i]
        for i in range(len(df) - 3):
            current_price = close.iloc[i]
            future_price = close.iloc[i + 3]
            
            price_change = (future_price - current_price) / (current_price + 1e-8)
            
            if price_change > self.neutral_buffer:
                direction.iloc[i] = 1  # Up
            elif price_change < -self.neutral_buffer:
                direction.iloc[i] = 0  # Down
            else:
                direction.iloc[i] = 2  # Neutral
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate 6-class regime labels:
        0: BULL_STRONG, 1: BULL_WEAK, 2: RANGE, 3: BEAR_WEAK, 4: BEAR_STRONG, 5: VOLATILE
        """
        regime = pd.Series(2, index=df.index, dtype=int)  # Default: RANGE
        
        # Calculate trend (20-period return)
        if 'close' in df.columns:
            trend = df['close'].pct_change(20) * 100
        else:
            trend = pd.Series(0.0, index=df.index)
        
        # Get ADX
        if 'ADX' in df.columns:
            adx = df['ADX']
        else:
            adx = pd.Series(25.0, index=df.index)
        
        # Get ATR percentile for volatility
        if 'ATR_14' in df.columns:
            atr = df['ATR_14']
            atr_90th = atr.rolling(100).quantile(0.9)
        else:
            atr = pd.Series(0.0, index=df.index)
            atr_90th = pd.Series(0.0, index=df.index)
        
        # Get volume spike
        if 'Volume_ratio' in df.columns:
            volume_spike = df['Volume_ratio'] > 2.0
        else:
            volume_spike = pd.Series(False, index=df.index)
        
        # Classify regimes
        for i in range(len(df)):
            trend_val = trend.iloc[i] if not pd.isna(trend.iloc[i]) else 0.0
            adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 25.0
            atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0
            atr_threshold = atr_90th.iloc[i] if not pd.isna(atr_90th.iloc[i]) else 0.0
            
            # VOLATILE (priority check)
            if (atr_val > atr_threshold) or volume_spike.iloc[i]:
                regime.iloc[i] = 5
            # BULL_STRONG
            elif trend_val > 2.0 and adx_val > 25:
                regime.iloc[i] = 0
            # BULL_WEAK
            elif 0 < trend_val <= 2.0 and adx_val <= 25:
                regime.iloc[i] = 1
            # BEAR_STRONG
            elif trend_val < -2.0 and adx_val > 25:
                regime.iloc[i] = 4
            # BEAR_WEAK
            elif -2.0 <= trend_val < 0 and adx_val <= 25:
                regime.iloc[i] = 3
            # RANGE (default)
            else:
                regime.iloc[i] = 2
        
        return regime
    
    def _balance_labels(self, labels: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Balance label distributions to reduce bias"""
        balanced_labels = labels.copy()
        
        # Balance BUY/SELL
        buy_count = balanced_labels['buy'].sum()
        sell_count = balanced_labels['sell'].sum()
        total_signals = buy_count + sell_count
        
        if total_signals > 0:
            buy_ratio = buy_count / total_signals
            sell_ratio = sell_count / total_signals
            
            # Target ratio: ~0.3-0.4 each
            target_ratio = 0.35
            
            if buy_ratio > 0.5:
                # Downsample BUY
                buy_indices = balanced_labels[balanced_labels['buy'] == 1].index
                keep_ratio = target_ratio / buy_ratio
                keep_indices = np.random.choice(buy_indices, 
                                               size=int(len(buy_indices) * keep_ratio),
                                               replace=False)
                balanced_labels.loc[balanced_labels['buy'] == 1, 'buy'] = 0
                balanced_labels.loc[keep_indices, 'buy'] = 1
            
            if sell_ratio > 0.5:
                # Downsample SELL
                sell_indices = balanced_labels[balanced_labels['sell'] == 1].index
                keep_ratio = target_ratio / sell_ratio
                keep_indices = np.random.choice(sell_indices,
                                               size=int(len(sell_indices) * keep_ratio),
                                               replace=False)
                balanced_labels.loc[balanced_labels['sell'] == 1, 'sell'] = 0
                balanced_labels.loc[keep_indices, 'sell'] = 1
        
        # Balance Direction (ensure neutral doesn't dominate)
        direction_counts = balanced_labels['direction'].value_counts()
        if len(direction_counts) > 0:
            max_count = direction_counts.max()
            for direction_class in [0, 1, 2]:
                if direction_class in direction_counts.index:
                    count = direction_counts[direction_class]
                    if count > max_count * 1.5:  # If over-represented
                        indices = balanced_labels[balanced_labels['direction'] == direction_class].index
                        keep_ratio = (max_count * 1.2) / count
                        keep_indices = np.random.choice(indices,
                                                       size=int(len(indices) * keep_ratio),
                                                       replace=False)
                        balanced_labels.loc[balanced_labels['direction'] == direction_class, 'direction'] = 2  # Set to neutral
                        balanced_labels.loc[keep_indices, 'direction'] = direction_class
        
        return balanced_labels
    
    def _print_label_statistics(self, labels: pd.DataFrame):
        """Print label distribution statistics"""
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ["buy", "sell", "direction", "regime"]:
            if col in labels.columns:
                vc = labels[col].value_counts(normalize=True).sort_index()
                print(f"\n▶ {col.upper()} distribution:")
                for idx, val in vc.items():
                    print(f"  {idx}: {val:.3f}")
        
        # Additional statistics
        if 'buy' in labels.columns and 'sell' in labels.columns:
            buy_sell_ratio = labels['buy'].sum() / (labels['sell'].sum() + 1e-8)
            print(f"\n▶ BUY/SELL ratio: {buy_sell_ratio:.3f}")
