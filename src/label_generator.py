"""
LabelGeneratorV2 - Bias-Reduced Institutional Labeling
Generates Buy/Sell, Direction (3-class), and Regime (6-class) labels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """Generates balanced, bias-reduced labels for multi-task learning"""
    
    def __init__(self, 
                 buy_threshold: int = 2,
                 sell_threshold: int = 2,
                 neutral_buffer: float = 0.001,
                 balance_classes: bool = True,
                 stochastic_perturbation: float = 0.05):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.neutral_buffer = neutral_buffer
        self.balance_classes = balance_classes
        self.stochastic_perturbation = stochastic_perturbation
        
    def generate_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate Buy/Sell labels
        buy_labels, sell_labels = self._generate_buy_sell_labels(df, features_df)
        labels['buy'] = buy_labels
        labels['sell'] = sell_labels
        
        # Generate Direction labels (3-class)
        direction_labels = self._generate_direction_labels(df)
        labels['direction'] = direction_labels
        
        # Generate Regime labels (6-class)
        regime_labels = self._generate_regime_labels(df, features_df)
        labels['regime'] = regime_labels
        
        # Balance classes if requested
        if self.balance_classes:
            labels = self._balance_labels(labels, df)
        
        # Print label statistics
        self._print_label_statistics(labels)
        
        return labels
    
    def _generate_buy_sell_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate Buy/Sell labels based on institutional conditions"""
        close = df['close'].values
        volume = df['volume'].values
        
        # Extract required features
        ma20 = features_df.get('MA20', pd.Series(index=df.index, data=0))
        ma50 = features_df.get('MA50', pd.Series(index=df.index, data=0))
        rsi = features_df.get('RSI', pd.Series(index=df.index, data=50))
        bb_mid = features_df.get('BB_mid', close)
        ma20_slope = features_df.get('MA20_slope', pd.Series(index=df.index, data=0))
        macd_hist = features_df.get('MACD_hist', pd.Series(index=df.index, data=0))
        position_in_range = features_df.get('position_in_range', pd.Series(index=df.index, data=0.5))
        plus_DI = features_df.get('plus_DI', pd.Series(index=df.index, data=0))
        minus_DI = features_df.get('minus_DI', pd.Series(index=df.index, data=0))
        
        # Convert to numpy arrays
        ma20 = ma20.values if isinstance(ma20, pd.Series) else ma20
        ma50 = ma50.values if isinstance(ma50, pd.Series) else ma50
        rsi = rsi.values if isinstance(rsi, pd.Series) else rsi
        bb_mid = bb_mid.values if isinstance(bb_mid, pd.Series) else bb_mid
        ma20_slope = ma20_slope.values if isinstance(ma20_slope, pd.Series) else ma20_slope
        macd_hist = macd_hist.values if isinstance(macd_hist, pd.Series) else macd_hist
        position_in_range = position_in_range.values if isinstance(position_in_range, pd.Series) else position_in_range
        plus_DI = plus_DI.values if isinstance(plus_DI, pd.Series) else plus_DI
        minus_DI = minus_DI.values if isinstance(minus_DI, pd.Series) else minus_DI
        
        # Volume rolling average
        volume_ma20 = pd.Series(volume).rolling(20).mean().fillna(volume[0]).values
        
        # RSI momentum
        rsi_cross_up = (rsi > 50) | ((rsi > 30) & (pd.Series(rsi).shift(1) <= 30))
        rsi_cross_down = (rsi < 50) | ((rsi < 70) & (pd.Series(rsi).shift(1) >= 70))
        
        # MACD momentum
        macd_rising = (macd_hist > 0) & (pd.Series(macd_hist).diff() > 0)
        macd_falling = (macd_hist < 0) & (pd.Series(macd_hist).diff() < 0)
        
        # BUY conditions
        buy_conditions = np.zeros((len(df), 6), dtype=bool)
        buy_conditions[:, 0] = (close > ma20) & (ma20 > ma50)  # Trend alignment
        buy_conditions[:, 1] = rsi_cross_up  # RSI momentum
        buy_conditions[:, 2] = (close > bb_mid) & (ma20_slope > 0)  # BB + MA slope
        buy_conditions[:, 3] = volume > 1.5 * volume_ma20  # Volume confirmation
        buy_conditions[:, 4] = position_in_range >= 0.7  # Near session high
        buy_conditions[:, 5] = macd_rising  # MACD momentum
        
        # SELL conditions
        sell_conditions = np.zeros((len(df), 6), dtype=bool)
        sell_conditions[:, 0] = (close < ma20) & (ma20 < ma50)  # Trend alignment
        sell_conditions[:, 1] = rsi_cross_down  # RSI momentum
        sell_conditions[:, 2] = (close < bb_mid) & (ma20_slope < 0)  # BB + MA slope
        sell_conditions[:, 3] = (volume > 1.5 * volume_ma20) & (position_in_range <= 0.3)  # Volume + position
        sell_conditions[:, 4] = macd_falling  # MACD momentum
        sell_conditions[:, 5] = minus_DI > plus_DI  # DI divergence
        
        # Count conditions met
        buy_counts = buy_conditions.sum(axis=1)
        sell_counts = sell_conditions.sum(axis=1)
        
        # Apply thresholds with stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.normal(0, self.stochastic_perturbation, len(df))
            buy_counts = buy_counts + noise
            sell_counts = sell_counts + noise
        
        buy_labels = (buy_counts >= self.buy_threshold).astype(float)
        sell_labels = (sell_counts >= self.sell_threshold).astype(float)
        
        # Ensure mutual exclusivity (prioritize stronger signal)
        conflict_mask = (buy_labels == 1) & (sell_labels == 1)
        buy_strength = buy_counts[conflict_mask]
        sell_strength = sell_counts[conflict_mask]
        buy_labels[conflict_mask] = (buy_strength >= sell_strength).astype(float)
        sell_labels[conflict_mask] = (sell_strength > buy_strength).astype(float)
        
        return pd.Series(buy_labels, index=df.index), pd.Series(sell_labels, index=df.index)
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate 3-class direction labels: 0=Down, 1=Up, 2=Neutral"""
        close = df['close'].values
        direction = np.full(len(df), 2, dtype=int)  # Default: Neutral
        
        # Look ahead 3 periods
        future_close = np.roll(close, -3)
        
        # Calculate price change
        price_change = (future_close - close) / (close + 1e-8)
        
        # Up: price increases beyond neutral buffer
        up_mask = price_change > self.neutral_buffer
        direction[up_mask] = 1
        
        # Down: price decreases beyond neutral buffer
        down_mask = price_change < -self.neutral_buffer
        direction[down_mask] = 0
        
        # Set last 3 samples to neutral (no future data)
        direction[-3:] = 2
        
        return pd.Series(direction, index=df.index)
    
    def _generate_regime_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """Generate 6-class regime labels"""
        close = df['close'].values
        
        # Extract features
        adx = features_df.get('ADX', pd.Series(index=df.index, data=0))
        atr = features_df.get('ATR14', pd.Series(index=df.index, data=0))
        volume = df['volume'].values
        
        # Convert to arrays
        adx = adx.values if isinstance(adx, pd.Series) else adx
        atr = atr.values if isinstance(atr, pd.Series) else atr
        
        # Calculate trend (20-period return)
        trend = pd.Series(close).pct_change(20).fillna(0).values
        
        # Volume spike threshold (90th percentile)
        volume_threshold = np.percentile(volume, 90)
        atr_threshold = np.percentile(atr[atr > 0], 90) if np.any(atr > 0) else np.max(atr)
        
        # Initialize regime labels
        regime = np.full(len(df), 2, dtype=int)  # Default: RANGE
        
        # BULL_STRONG: trend > +2%, ADX > 25
        regime[(trend > 0.02) & (adx > 25)] = 0
        
        # BULL_WEAK: 0 < trend <= +2%, ADX <= 25
        regime[(trend > 0) & (trend <= 0.02) & (adx <= 25)] = 1
        
        # BEAR_WEAK: -2% <= trend < 0, ADX <= 25
        regime[(trend >= -0.02) & (trend < 0) & (adx <= 25)] = 3
        
        # BEAR_STRONG: trend < -2%, ADX > 25
        regime[(trend < -0.02) & (adx > 25)] = 4
        
        # VOLATILE: ATR > 90th percentile or volume spike
        regime[(atr > atr_threshold) | (volume > volume_threshold)] = 5
        
        return pd.Series(regime, index=df.index)
    
    def _balance_labels(self, labels: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Balance label distributions"""
        balanced_labels = labels.copy()
        
        # Balance Buy/Sell
        buy_ratio = balanced_labels['buy'].mean()
        sell_ratio = balanced_labels['sell'].mean()
        
        target_ratio = 0.35  # Target ~35% for each
        
        if buy_ratio > target_ratio * 1.2:  # Too many buys
            buy_indices = balanced_labels[balanced_labels['buy'] == 1].index
            n_to_remove = int(len(buy_indices) * (1 - target_ratio / buy_ratio))
            remove_indices = np.random.choice(buy_indices, n_to_remove, replace=False)
            balanced_labels.loc[remove_indices, 'buy'] = 0
        
        if sell_ratio > target_ratio * 1.2:  # Too many sells
            sell_indices = balanced_labels[balanced_labels['sell'] == 1].index
            n_to_remove = int(len(sell_indices) * (1 - target_ratio / sell_ratio))
            remove_indices = np.random.choice(sell_indices, n_to_remove, replace=False)
            balanced_labels.loc[remove_indices, 'sell'] = 0
        
        # Balance Direction (ensure reasonable distribution)
        direction_counts = balanced_labels['direction'].value_counts()
        min_count = direction_counts.min()
        max_count = direction_counts.max()
        
        if max_count > min_count * 2:  # Significant imbalance
            for direction_class in direction_counts.index:
                class_indices = balanced_labels[balanced_labels['direction'] == direction_class].index
                if len(class_indices) > min_count * 1.5:
                    n_to_keep = int(min_count * 1.5)
                    keep_indices = np.random.choice(class_indices, n_to_keep, replace=False)
                    balanced_labels.loc[class_indices.difference(keep_indices), 'direction'] = 2  # Set to neutral
        
        return balanced_labels
    
    def _print_label_statistics(self, labels: pd.DataFrame):
        """Print label distribution statistics"""
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ["buy", "sell", "direction", "regime"]:
            vc = labels[col].value_counts(normalize=True).sort_index()
            print(f"\n▶ {col.upper()} distribution:")
            for idx, val in vc.items():
                print(f"  {idx}: {val:.3f}")
        
        # Regime names
        regime_names = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
        
        print("\n▶ REGIME class mapping:")
        for idx, name in regime_names.items():
            count = (labels['regime'] == idx).sum()
            pct = count / len(labels) * 100
            print(f"  {idx} ({name}): {count} ({pct:.1f}%)")
