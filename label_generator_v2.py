"""
Natron Transformer - Label Generator V2
Bias-Reduced Institutional Labeling for Multi-Task Learning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """
    Advanced label generation with bias reduction
    Generates labels for: Buy, Sell, Direction, Regime
    """
    
    def __init__(self, neutral_buffer: float = 0.001, lookforward: int = 3):
        """
        Args:
            neutral_buffer: Buffer for neutral direction classification
            lookforward: Candles to look ahead for direction
        """
        self.neutral_buffer = neutral_buffer
        self.lookforward = lookforward
    
    def generate_all_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels with bias reduction
        
        Args:
            df: Original OHLCV dataframe
            features: Technical features dataframe
        
        Returns:
            labels_df: DataFrame with columns [buy, sell, direction, regime]
        """
        print("\n🏷️  Generating labels...")
        
        labels = pd.DataFrame(index=df.index)
        
        # Generate each label type
        labels['buy'] = self._generate_buy_labels(df, features)
        labels['sell'] = self._generate_sell_labels(df, features)
        labels['direction'] = self._generate_direction_labels(df)
        labels['regime'] = self._generate_regime_labels(df, features)
        
        # Apply bias reduction
        labels = self._reduce_bias(labels, df, features)
        
        # Display statistics
        self._print_label_statistics(labels)
        
        return labels
    
    def _generate_buy_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate BUY labels (1 if ≥2 conditions true)
        
        Conditions:
        1. close > MA20 > MA50
        2. RSI > 50 or just crossed up from <30
        3. close > BB midband and MA20 slope > 0
        4. volume > 1.5 × rolling20
        5. position_in_range ≥ 0.7
        6. MACD_hist > 0 and rising
        """
        close = df['close']
        volume = df['volume']
        
        # Initialize conditions
        conditions = []
        
        # Condition 1: close > MA20 > MA50
        if 'sma_20' in features.columns and 'sma_50' in features.columns:
            cond1 = (close > features['sma_20']) & (features['sma_20'] > features['sma_50'])
            conditions.append(cond1)
        
        # Condition 2: RSI > 50 or just crossed up from <30
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14']
            rsi_cross_up = (rsi > 30) & (rsi.shift() < 30)
            cond2 = (rsi > 50) | rsi_cross_up
            conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20 slope > 0
        if 'sma_20' in features.columns and 'sma20_slope' in features.columns:
            bb_mid = features['sma_20']  # BB midband is SMA20
            cond3 = (close > bb_mid) & (features['sma20_slope'] > 0)
            conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        vol_ma = volume.rolling(20).mean()
        cond4 = volume > (vol_ma * 1.5)
        conditions.append(cond4)
        
        # Condition 5: position_in_range ≥ 0.7
        if 'position_in_range' in features.columns:
            cond5 = features['position_in_range'] >= 0.7
            conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and rising
        if 'macd_hist' in features.columns:
            macd_rising = features['macd_hist'] > features['macd_hist'].shift()
            cond6 = (features['macd_hist'] > 0) & macd_rising
            conditions.append(cond6)
        
        # BUY if ≥2 conditions are true
        conditions_met = sum(conditions)
        buy_labels = (conditions_met >= 2).astype(int)
        
        return buy_labels
    
    def _generate_sell_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate SELL labels (1 if ≥2 conditions true)
        
        Conditions:
        1. close < MA20 < MA50
        2. RSI < 50 or just turned down from >70
        3. close < BB midband and MA20 slope < 0
        4. volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        5. MACD_hist < 0 and falling
        6. minus_DI > plus_DI
        """
        close = df['close']
        volume = df['volume']
        
        conditions = []
        
        # Condition 1: close < MA20 < MA50
        if 'sma_20' in features.columns and 'sma_50' in features.columns:
            cond1 = (close < features['sma_20']) & (features['sma_20'] < features['sma_50'])
            conditions.append(cond1)
        
        # Condition 2: RSI < 50 or just turned down from >70
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14']
            rsi_turn_down = (rsi < 70) & (rsi.shift() > 70)
            cond2 = (rsi < 50) | rsi_turn_down
            conditions.append(cond2)
        
        # Condition 3: close < BB midband and MA20 slope < 0
        if 'sma_20' in features.columns and 'sma20_slope' in features.columns:
            bb_mid = features['sma_20']
            cond3 = (close < bb_mid) & (features['sma20_slope'] < 0)
            conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        vol_ma = volume.rolling(20).mean()
        if 'position_in_range' in features.columns:
            cond4 = (volume > vol_ma * 1.5) & (features['position_in_range'] <= 0.3)
            conditions.append(cond4)
        
        # Condition 5: MACD_hist < 0 and falling
        if 'macd_hist' in features.columns:
            macd_falling = features['macd_hist'] < features['macd_hist'].shift()
            cond5 = (features['macd_hist'] < 0) & macd_falling
            conditions.append(cond5)
        
        # Condition 6: minus_DI > plus_DI
        if 'minus_di' in features.columns and 'plus_di' in features.columns:
            cond6 = features['minus_di'] > features['plus_di']
            conditions.append(cond6)
        
        # SELL if ≥2 conditions are true
        conditions_met = sum(conditions)
        sell_labels = (conditions_met >= 2).astype(int)
        
        return sell_labels
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate DIRECTION labels (3-class balanced)
        
        0: Down
        1: Up
        2: Neutral
        """
        close = df['close']
        
        # Look forward
        future_close = close.shift(-self.lookforward)
        price_change = future_close - close
        
        # Calculate threshold (neutral buffer)
        threshold = close * self.neutral_buffer
        
        # Classify
        direction = pd.Series(2, index=df.index)  # Default: Neutral
        direction[price_change > threshold] = 1  # Up
        direction[price_change < -threshold] = 0  # Down
        
        # Fill last lookforward candles with neutral
        direction.iloc[-self.lookforward:] = 2
        
        return direction
    
    def _generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate REGIME labels (6 classes)
        
        0: BULL_STRONG
        1: BULL_WEAK
        2: RANGE (default)
        3: BEAR_WEAK
        4: BEAR_STRONG
        5: VOLATILE
        """
        close = df['close']
        regime = pd.Series(2, index=df.index)  # Default: RANGE
        
        # Calculate trend (20-period return)
        trend = (close - close.shift(20)) / close.shift(20) * 100
        
        # Get ADX if available
        adx = features['adx_14'] if 'adx_14' in features.columns else pd.Series(20, index=df.index)
        
        # Get ATR percentile for volatility
        atr_pct = features['atr_pct'] if 'atr_pct' in features.columns else pd.Series(0, index=df.index)
        atr_threshold = atr_pct.quantile(0.9)
        
        # Get volume spike
        volume = df['volume']
        vol_ma = volume.rolling(20).mean()
        volume_spike = volume > (vol_ma * 2)
        
        # Classify regimes
        # 5: VOLATILE (highest priority)
        regime[(atr_pct > atr_threshold) | volume_spike] = 5
        
        # 0: BULL_STRONG
        regime[(trend > 2) & (adx > 25) & (regime != 5)] = 0
        
        # 1: BULL_WEAK
        regime[(trend > 0) & (trend <= 2) & (adx <= 25) & (regime != 5)] = 1
        
        # 4: BEAR_STRONG
        regime[(trend < -2) & (adx > 25) & (regime != 5)] = 4
        
        # 3: BEAR_WEAK
        regime[(trend < 0) & (trend >= -2) & (adx <= 25) & (regime != 5)] = 3
        
        return regime
    
    def _reduce_bias(self, labels: pd.DataFrame, df: pd.DataFrame, 
                     features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply bias reduction techniques
        - Balance BUY/SELL distribution
        - Add stochastic perturbation
        - Adjust thresholds adaptively
        """
        labels = labels.copy()
        
        # Calculate current buy/sell ratio
        buy_ratio = labels['buy'].mean()
        sell_ratio = labels['sell'].mean()
        
        print(f"  Initial BUY ratio: {buy_ratio:.3f}, SELL ratio: {sell_ratio:.3f}")
        
        # If buy/sell ratio is imbalanced, apply correction
        if buy_ratio > 0.5:  # Too many buys
            # Randomly flip some buys to 0
            excess_buy_indices = labels[labels['buy'] == 1].sample(
                frac=max(0, (buy_ratio - 0.4) / buy_ratio), 
                random_state=42
            ).index
            labels.loc[excess_buy_indices, 'buy'] = 0
        
        if sell_ratio > 0.5:  # Too many sells
            excess_sell_indices = labels[labels['sell'] == 1].sample(
                frac=max(0, (sell_ratio - 0.4) / sell_ratio), 
                random_state=42
            ).index
            labels.loc[excess_sell_indices, 'sell'] = 0
        
        # Add mild stochastic perturbation (flip 5% randomly)
        n_samples = len(labels)
        perturb_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        
        for idx in perturb_indices:
            if np.random.random() > 0.5:
                labels.loc[labels.index[idx], 'buy'] = 1 - labels.loc[labels.index[idx], 'buy']
            else:
                labels.loc[labels.index[idx], 'sell'] = 1 - labels.loc[labels.index[idx], 'sell']
        
        # Ensure buy and sell are mutually exclusive (mostly)
        both_true = (labels['buy'] == 1) & (labels['sell'] == 1)
        if both_true.sum() > 0:
            # Randomly choose one
            for idx in labels[both_true].index:
                if np.random.random() > 0.5:
                    labels.loc[idx, 'sell'] = 0
                else:
                    labels.loc[idx, 'buy'] = 0
        
        print(f"  After bias reduction - BUY: {labels['buy'].mean():.3f}, SELL: {labels['sell'].mean():.3f}")
        
        return labels
    
    def _print_label_statistics(self, labels: pd.DataFrame):
        """Print comprehensive label distribution statistics"""
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ['buy', 'sell', 'direction', 'regime']:
            if col in labels.columns:
                vc = labels[col].value_counts(normalize=True).sort_index()
                print(f"\n▶ {col.upper()} distribution:")
                
                if col in ['buy', 'sell']:
                    print(f"  0 (No {col}):  {vc.get(0, 0):.3f}")
                    print(f"  1 ({col.capitalize()}):     {vc.get(1, 0):.3f}")
                
                elif col == 'direction':
                    print(f"  0 (Down):     {vc.get(0, 0):.3f}")
                    print(f"  1 (Up):       {vc.get(1, 0):.3f}")
                    print(f"  2 (Neutral):  {vc.get(2, 0):.3f}")
                
                elif col == 'regime':
                    regime_names = {
                        0: 'BULL_STRONG',
                        1: 'BULL_WEAK',
                        2: 'RANGE',
                        3: 'BEAR_WEAK',
                        4: 'BEAR_STRONG',
                        5: 'VOLATILE'
                    }
                    for regime_id, regime_name in regime_names.items():
                        print(f"  {regime_id} ({regime_name:12s}): {vc.get(regime_id, 0):.3f}")
        
        print("\n" + "="*50)
    
    def get_class_weights(self, labels: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate class weights for balanced training
        
        Returns:
            Dictionary of class weights for each task
        """
        weights = {}
        
        # Buy/Sell weights (binary)
        for task in ['buy', 'sell']:
            if task in labels.columns:
                counts = labels[task].value_counts()
                total = len(labels)
                weights[task] = np.array([
                    total / (2 * counts.get(0, 1)),
                    total / (2 * counts.get(1, 1))
                ])
        
        # Direction weights (3-class)
        if 'direction' in labels.columns:
            counts = labels['direction'].value_counts()
            total = len(labels)
            weights['direction'] = np.array([
                total / (3 * counts.get(i, 1)) for i in range(3)
            ])
        
        # Regime weights (6-class)
        if 'regime' in labels.columns:
            counts = labels['regime'].value_counts()
            total = len(labels)
            weights['regime'] = np.array([
                total / (6 * counts.get(i, 1)) for i in range(6)
            ])
        
        return weights


def create_labels(df: pd.DataFrame, features: pd.DataFrame, 
                  neutral_buffer: float = 0.001, 
                  lookforward: int = 3) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function to generate labels and class weights
    
    Args:
        df: OHLCV dataframe
        features: Technical features dataframe
        neutral_buffer: Buffer for neutral direction
        lookforward: Candles ahead for direction
    
    Returns:
        labels: DataFrame with all labels
        class_weights: Dictionary of class weights
    """
    generator = LabelGeneratorV2(neutral_buffer, lookforward)
    labels = generator.generate_all_labels(df, features)
    class_weights = generator.get_class_weights(labels)
    
    return labels, class_weights


if __name__ == "__main__":
    # Test label generation
    from feature_engine import load_and_prepare_data
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data_export.csv"
    
    # Load data and features
    raw_df, features_df = load_and_prepare_data(csv_path)
    
    # Generate labels
    labels_df, weights = create_labels(raw_df, features_df)
    
    print(f"\n✅ Generated labels shape: {labels_df.shape}")
    print(f"\n📊 Class weights:")
    for task, w in weights.items():
        print(f"  {task}: {w}")
