"""
LabelGeneratorV2: Bias-reduced institutional labeling
Generates Buy/Sell, Direction, and Regime labels with balanced distributions
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class LabelGeneratorV2:
    """Generates multi-task labels with bias reduction and balancing"""
    
    def __init__(
        self,
        buy_threshold: int = 2,
        sell_threshold: int = 2,
        neutral_buffer: float = 0.001,
        balance_classes: bool = True,
        stochastic_perturbation: float = 0.05
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.neutral_buffer = neutral_buffer
        self.balance_classes = balance_classes
        self.stochastic_perturbation = stochastic_perturbation
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Buy/Sell, Direction, and Regime labels
        
        Args:
            df: DataFrame with features (must include MA20, MA50, RSI, BB_mid, etc.)
        
        Returns:
            DataFrame with columns: buy, sell, direction, regime
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate Buy/Sell signals
        labels['buy'] = self._generate_buy_signals(df)
        labels['sell'] = self._generate_sell_signals(df)
        
        # Generate Direction labels (3-class: Up=1, Down=0, Neutral=2)
        labels['direction'] = self._generate_direction_labels(df)
        
        # Generate Regime labels (6 classes)
        labels['regime'] = self._generate_regime_labels(df)
        
        # Balance classes if requested
        if self.balance_classes:
            labels = self._balance_labels(labels, df)
        
        # Print label statistics
        self._print_label_statistics(labels)
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate BUY signals (1 if ≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close > MA20 > MA50
        cond1 = (df['close'] > df['MA20']) & (df['MA20'] > df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI > 50 or just crossed up from <30
        rsi_cross_up = (df['RSI'] > 50) | ((df['RSI'] > 30) & (df['RSI'].shift(1) <= 30))
        conditions.append(rsi_cross_up)
        
        # Condition 3: close > BB midband and MA20 slope > 0
        cond3 = (df['close'] > df['BB_mid']) & (df['MA20_slope'] > 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        cond4 = df['volume'] > 1.5 * df['volume_MA20']
        conditions.append(cond4)
        
        # Condition 5: position_in_range ≥ 0.7 (near session high)
        cond5 = df['position_in_range'] >= 0.7
        conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and rising
        cond6 = (df['MACD_hist'] > 0) & (df['MACD_hist_rising'] == 1)
        conditions.append(cond6)
        
        # Count conditions met
        condition_count = sum(conditions)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.random(len(df)) < self.stochastic_perturbation
            condition_count = condition_count + noise.astype(int)
        
        # BUY = 1 if ≥ threshold conditions met
        buy_signal = (condition_count >= self.buy_threshold).astype(float)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SELL signals (1 if ≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close < MA20 < MA50
        cond1 = (df['close'] < df['MA20']) & (df['MA20'] < df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI < 50 or just turned down from >70
        rsi_cross_down = (df['RSI'] < 50) | ((df['RSI'] < 70) & (df['RSI'].shift(1) >= 70))
        conditions.append(rsi_cross_down)
        
        # Condition 3: close < BB midband and MA20 slope < 0
        cond3 = (df['close'] < df['BB_mid']) & (df['MA20_slope'] < 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        cond4 = (df['volume'] > 1.5 * df['volume_MA20']) & (df['position_in_range'] <= 0.3)
        conditions.append(cond4)
        
        # Condition 5: MACD_hist < 0 and falling
        cond5 = (df['MACD_hist'] < 0) & (df['MACD_hist_rising'] == 0)
        conditions.append(cond5)
        
        # Condition 6: minus_DI > plus_DI
        cond6 = df['minus_DI'] > df['plus_DI']
        conditions.append(cond6)
        
        # Count conditions met
        condition_count = sum(conditions)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.random(len(df)) < self.stochastic_perturbation
            condition_count = condition_count + noise.astype(int)
        
        # SELL = 1 if ≥ threshold conditions met
        sell_signal = (condition_count >= self.sell_threshold).astype(float)
        
        return sell_signal
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Direction labels (3-class balanced)
        1 = Up (close[i+3] > close[i] + buffer)
        0 = Down (close[i+3] < close[i] - buffer)
        2 = Neutral (otherwise)
        """
        close = df['close']
        direction = pd.Series(2, index=df.index, dtype=int)  # Default: Neutral
        
        # Forward-looking (shift back by 3)
        future_close = close.shift(-3)
        current_close = close
        
        # Up: future price significantly higher
        up_mask = future_close > current_close * (1 + self.neutral_buffer)
        direction[up_mask] = 1
        
        # Down: future price significantly lower
        down_mask = future_close < current_close * (1 - self.neutral_buffer)
        direction[down_mask] = 0
        
        # Handle NaN (last 3 rows)
        direction = direction.fillna(2)
        
        return direction
    
    def _generate_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Regime labels (6 classes)
        0: BULL_STRONG (trend > +2%, ADX > 25)
        1: BULL_WEAK (0 < trend ≤ +2%, ADX ≤ 25)
        2: RANGE (lateral market - default)
        3: BEAR_WEAK (-2% ≤ trend < 0, ADX ≤ 25)
        4: BEAR_STRONG (trend < -2%, ADX > 25)
        5: VOLATILE (ATR > 90th percentile or volume spike)
        """
        regime = pd.Series(2, index=df.index, dtype=int)  # Default: RANGE
        
        # Calculate trend (20-period return)
        trend = df['return_3'] * 100  # Convert to percentage
        
        # ADX threshold
        adx_threshold = 25
        
        # ATR percentile for volatility
        atr_percentile_90 = df['ATR_pct'].quantile(0.90)
        
        # BULL_STRONG: trend > +2%, ADX > 25
        bull_strong = (trend > 2.0) & (df['ADX'] > adx_threshold)
        regime[bull_strong] = 0
        
        # BULL_WEAK: 0 < trend ≤ +2%, ADX ≤ 25
        bull_weak = (trend > 0) & (trend <= 2.0) & (df['ADX'] <= adx_threshold)
        regime[bull_weak] = 1
        
        # BEAR_WEAK: -2% ≤ trend < 0, ADX ≤ 25
        bear_weak = (trend >= -2.0) & (trend < 0) & (df['ADX'] <= adx_threshold)
        regime[bear_weak] = 3
        
        # BEAR_STRONG: trend < -2%, ADX > 25
        bear_strong = (trend < -2.0) & (df['ADX'] > adx_threshold)
        regime[bear_strong] = 4
        
        # VOLATILE: ATR > 90th percentile or volume spike
        volatile = (df['ATR_pct'] > atr_percentile_90) | (df['volume_spike'] == 1)
        regime[volatile] = 5
        
        return regime
    
    def _balance_labels(self, labels: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance label distributions by downsampling over-represented classes
        """
        balanced_labels = labels.copy()
        
        # Balance Buy/Sell
        buy_count = balanced_labels['buy'].sum()
        sell_count = balanced_labels['sell'].sum()
        total_signals = buy_count + sell_count
        
        if total_signals > 0:
            buy_ratio = buy_count / total_signals
            sell_ratio = sell_count / total_signals
            
            # Target ratio: ~0.3-0.4 each
            target_ratio = 0.35
            
            if buy_ratio > target_ratio + 0.1:
                # Downsample buys
                buy_indices = balanced_labels[balanced_labels['buy'] == 1].index
                keep_ratio = target_ratio / buy_ratio
                keep_indices = np.random.choice(
                    buy_indices,
                    size=int(len(buy_indices) * keep_ratio),
                    replace=False
                )
                balanced_labels.loc[balanced_labels['buy'] == 1, 'buy'] = 0
                balanced_labels.loc[keep_indices, 'buy'] = 1
            
            if sell_ratio > target_ratio + 0.1:
                # Downsample sells
                sell_indices = balanced_labels[balanced_labels['sell'] == 1].index
                keep_ratio = target_ratio / sell_ratio
                keep_indices = np.random.choice(
                    sell_indices,
                    size=int(len(sell_indices) * keep_ratio),
                    replace=False
                )
                balanced_labels.loc[balanced_labels['sell'] == 1, 'sell'] = 0
                balanced_labels.loc[keep_indices, 'sell'] = 1
        
        return balanced_labels
    
    def _print_label_statistics(self, labels: pd.DataFrame):
        """Print label distribution statistics"""
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True).sort_index()
            print(f"\n▶ {col.upper()} distribution:")
            for val, pct in vc.items():
                print(f"  {val}: {pct:.3f}")
        
        # Additional statistics
        print(f"\n▶ Buy/Sell Statistics:")
        print(f"  Buy signals: {labels['buy'].sum():.0f} ({labels['buy'].mean():.3f})")
        print(f"  Sell signals: {labels['sell'].sum():.0f} ({labels['sell'].mean():.3f})")
        print(f"  Buy/Sell ratio: {labels['buy'].sum() / (labels['sell'].sum() + 1e-8):.3f}")
