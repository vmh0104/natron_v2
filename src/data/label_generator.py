"""
Natron Label Generator V2 - Bias-Reduced Institutional Labeling
Implements sophisticated multi-condition labeling for Buy/Sell/Direction/Regime
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """
    Generates balanced, bias-reduced labels for multi-task learning.
    
    Labels:
    - buy: Binary (0/1) based on multi-condition bullish setup
    - sell: Binary (0/1) based on multi-condition bearish setup
    - direction: 3-class (0=Down, 1=Up, 2=Neutral)
    - regime: 6-class market state classification
    """
    
    def __init__(self, config: Dict):
        self.buy_threshold = config.get('buy_threshold', 2)
        self.sell_threshold = config.get('sell_threshold', 2)
        self.neutral_buffer = config.get('neutral_buffer', 0.001)
        self.regime_trend_strong = config.get('regime_trend_strong', 0.02)
        self.regime_adx_threshold = config.get('regime_adx_threshold', 25)
        self.regime_atr_percentile = config.get('regime_atr_percentile', 90)
        self.balance_labels = config.get('balance_labels', True)
        self.downsample_heavy = config.get('downsample_heavy_classes', True)
        
    def generate_all_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point - generates all labels.
        
        Args:
            df: Original OHLCV dataframe
            features: Feature dataframe from FeatureEngine
            
        Returns:
            DataFrame with columns: [buy, sell, direction, regime]
        """
        labels = pd.DataFrame(index=df.index)
        
        print("\n🏷️  Generating labels...")
        
        # 1. Buy/Sell Labels
        labels['buy'] = self._generate_buy_labels(df, features)
        labels['sell'] = self._generate_sell_labels(df, features)
        print(f"  ✓ Buy/Sell labels generated")
        
        # 2. Direction Labels
        labels['direction'] = self._generate_direction_labels(df)
        print(f"  ✓ Direction labels generated")
        
        # 3. Regime Labels
        labels['regime'] = self._generate_regime_labels(df, features)
        print(f"  ✓ Regime labels generated")
        
        # 4. Balance labels if enabled
        if self.balance_labels:
            labels = self._balance_labels(labels)
            print(f"  ✓ Labels balanced")
        
        # 5. Print statistics
        self._print_label_statistics(labels)
        
        return labels
    
    # ==================== Buy Labels ====================
    
    def _generate_buy_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        BUY = 1 if ≥ buy_threshold conditions are true:
        1. close > MA20 > MA50
        2. RSI > 50 or just crossed up from <30
        3. close > BB midband and MA20 slope > 0
        4. volume > 1.5 × rolling20
        5. position_in_range ≥ 0.7 (near session high)
        6. MACD_hist > 0 and rising
        """
        close = df['close']
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: Bullish MA alignment
        try:
            cond1 = (close > features['sma_20']) & (features['sma_20'] > features['sma_50'])
        except KeyError:
            cond1 = close > features.get('sma_20', close)
        
        # Condition 2: RSI momentum
        try:
            rsi = features['rsi_14']
            rsi_prev = rsi.shift(1)
            cond2 = (rsi > 50) | ((rsi > 30) & (rsi_prev <= 30))
        except KeyError:
            cond2 = pd.Series(False, index=df.index)
        
        # Condition 3: Price above BB middle with positive slope
        try:
            cond3 = (close > features['bb_middle']) & (features['sma20_slope'] > 0)
        except KeyError:
            cond3 = pd.Series(False, index=df.index)
        
        # Condition 4: Volume surge
        try:
            cond4 = df['volume'] > (1.5 * features['volume_sma20'])
        except KeyError:
            cond4 = pd.Series(False, index=df.index)
        
        # Condition 5: Position in range
        try:
            cond5 = features['position_in_range'] >= 0.7
        except KeyError:
            cond5 = pd.Series(False, index=df.index)
        
        # Condition 6: MACD histogram positive and rising
        try:
            macd_hist = features['macd_hist']
            cond6 = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
        except KeyError:
            cond6 = pd.Series(False, index=df.index)
        
        # Combine conditions
        conditions['cond1'] = cond1
        conditions['cond2'] = cond2
        conditions['cond3'] = cond3
        conditions['cond4'] = cond4
        conditions['cond5'] = cond5
        conditions['cond6'] = cond6
        
        # Buy = 1 if at least buy_threshold conditions are met
        buy_score = conditions.sum(axis=1)
        buy_labels = (buy_score >= self.buy_threshold).astype(int)
        
        # Add stochastic perturbation to reduce overfitting
        if self.balance_labels:
            random_flip = np.random.random(len(buy_labels)) < 0.05
            near_threshold = (buy_score == self.buy_threshold) | (buy_score == self.buy_threshold - 1)
            buy_labels = buy_labels.where(~(random_flip & near_threshold), 1 - buy_labels)
        
        return buy_labels
    
    # ==================== Sell Labels ====================
    
    def _generate_sell_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        SELL = 1 if ≥ sell_threshold conditions are true:
        1. close < MA20 < MA50
        2. RSI < 50 or just turned down from >70
        3. close < BB midband and MA20 slope < 0
        4. volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        5. MACD_hist < 0 and falling
        6. minus_DI > plus_DI
        """
        close = df['close']
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: Bearish MA alignment
        try:
            cond1 = (close < features['sma_20']) & (features['sma_20'] < features['sma_50'])
        except KeyError:
            cond1 = close < features.get('sma_20', close)
        
        # Condition 2: RSI weakness
        try:
            rsi = features['rsi_14']
            rsi_prev = rsi.shift(1)
            cond2 = (rsi < 50) | ((rsi < 70) & (rsi_prev >= 70))
        except KeyError:
            cond2 = pd.Series(False, index=df.index)
        
        # Condition 3: Price below BB middle with negative slope
        try:
            cond3 = (close < features['bb_middle']) & (features['sma20_slope'] < 0)
        except KeyError:
            cond3 = pd.Series(False, index=df.index)
        
        # Condition 4: Volume surge with weak position
        try:
            cond4 = (df['volume'] > 1.5 * features['volume_sma20']) & (features['position_in_range'] <= 0.3)
        except KeyError:
            cond4 = pd.Series(False, index=df.index)
        
        # Condition 5: MACD histogram negative and falling
        try:
            macd_hist = features['macd_hist']
            cond5 = (macd_hist < 0) & (macd_hist < macd_hist.shift(1))
        except KeyError:
            cond5 = pd.Series(False, index=df.index)
        
        # Condition 6: Bearish directional indicator
        try:
            cond6 = features['minus_di'] > features['plus_di']
        except KeyError:
            cond6 = pd.Series(False, index=df.index)
        
        # Combine conditions
        conditions['cond1'] = cond1
        conditions['cond2'] = cond2
        conditions['cond3'] = cond3
        conditions['cond4'] = cond4
        conditions['cond5'] = cond5
        conditions['cond6'] = cond6
        
        # Sell = 1 if at least sell_threshold conditions are met
        sell_score = conditions.sum(axis=1)
        sell_labels = (sell_score >= self.sell_threshold).astype(int)
        
        # Add stochastic perturbation
        if self.balance_labels:
            random_flip = np.random.random(len(sell_labels)) < 0.05
            near_threshold = (sell_score == self.sell_threshold) | (sell_score == self.sell_threshold - 1)
            sell_labels = sell_labels.where(~(random_flip & near_threshold), 1 - sell_labels)
        
        return sell_labels
    
    # ==================== Direction Labels ====================
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Direction: 3-class balanced labeling
        0 = Down: close[i+3] < close[i] - neutral_buffer
        1 = Up: close[i+3] > close[i] + neutral_buffer
        2 = Neutral: otherwise
        """
        close = df['close']
        future_close = close.shift(-3)
        
        price_change = (future_close - close) / close
        
        direction = pd.Series(2, index=df.index)  # Default to neutral
        direction[price_change > self.neutral_buffer] = 1  # Up
        direction[price_change < -self.neutral_buffer] = 0  # Down
        
        # Fill last 3 rows with mode
        direction.iloc[-3:] = direction.mode()[0] if len(direction.mode()) > 0 else 2
        
        return direction.astype(int)
    
    # ==================== Regime Labels ====================
    
    def _generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Regime Classification (6 States):
        0: BULL_STRONG   - trend > +2%, ADX > 25
        1: BULL_WEAK     - 0 < trend ≤ +2%, ADX ≤ 25
        2: RANGE         - lateral market (default)
        3: BEAR_WEAK     - −2% ≤ trend < 0, ADX ≤ 25
        4: BEAR_STRONG   - trend < −2%, ADX > 25
        5: VOLATILE      - ATR > 90th percentile or volume spike
        """
        regime = pd.Series(2, index=df.index)  # Default to RANGE
        
        # Calculate trend (20-period price change)
        close = df['close']
        trend = (close - close.shift(20)) / close.shift(20)
        
        # Get ADX and ATR
        try:
            adx = features['adx']
        except KeyError:
            adx = pd.Series(20, index=df.index)
        
        try:
            atr = features['atr_14']
            atr_threshold = atr.quantile(self.regime_atr_percentile / 100)
        except KeyError:
            atr_threshold = float('inf')
            atr = pd.Series(0, index=df.index)
        
        # Volume spike detection
        try:
            volume_spike = df['volume'] > (2.5 * features['volume_sma20'])
        except KeyError:
            volume_spike = pd.Series(False, index=df.index)
        
        # Classify regimes
        # 5: VOLATILE
        regime[(atr > atr_threshold) | volume_spike] = 5
        
        # 0: BULL_STRONG
        regime[(trend > self.regime_trend_strong) & (adx > self.regime_adx_threshold) & (regime == 2)] = 0
        
        # 1: BULL_WEAK
        regime[(trend > 0) & (trend <= self.regime_trend_strong) & (adx <= self.regime_adx_threshold) & (regime == 2)] = 1
        
        # 4: BEAR_STRONG
        regime[(trend < -self.regime_trend_strong) & (adx > self.regime_adx_threshold) & (regime == 2)] = 4
        
        # 3: BEAR_WEAK
        regime[(trend < 0) & (trend >= -self.regime_trend_strong) & (adx <= self.regime_adx_threshold) & (regime == 2)] = 3
        
        return regime.astype(int)
    
    # ==================== Label Balancing ====================
    
    def _balance_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """
        Balance labels by downsampling over-represented classes.
        Ensures BUY/SELL ratio is reasonable (~0.3-0.4 each).
        """
        if not self.downsample_heavy:
            return labels
        
        # Balance Buy/Sell
        buy_ratio = labels['buy'].mean()
        sell_ratio = labels['sell'].mean()
        
        target_ratio = 0.35
        
        # Downsample Buy if too high
        if buy_ratio > target_ratio + 0.1:
            buy_indices = labels[labels['buy'] == 1].index
            n_keep = int(len(labels) * target_ratio)
            keep_indices = np.random.choice(buy_indices, size=min(n_keep, len(buy_indices)), replace=False)
            labels.loc[labels['buy'] == 1, 'buy'] = 0
            labels.loc[keep_indices, 'buy'] = 1
        
        # Downsample Sell if too high
        if sell_ratio > target_ratio + 0.1:
            sell_indices = labels[labels['sell'] == 1].index
            n_keep = int(len(labels) * target_ratio)
            keep_indices = np.random.choice(sell_indices, size=min(n_keep, len(sell_indices)), replace=False)
            labels.loc[labels['sell'] == 1, 'sell'] = 0
            labels.loc[keep_indices, 'sell'] = 1
        
        # Balance direction classes
        dir_counts = labels['direction'].value_counts()
        if len(dir_counts) > 0:
            min_count = int(dir_counts.min() * 1.5)  # Allow some imbalance
            for dir_class in dir_counts.index:
                if dir_counts[dir_class] > min_count * 2:
                    indices = labels[labels['direction'] == dir_class].index
                    keep_indices = np.random.choice(indices, size=min(min_count * 2, len(indices)), replace=False)
                    # Flip to neutral (class 2)
                    flip_indices = indices.difference(keep_indices)
                    labels.loc[flip_indices, 'direction'] = 2
        
        return labels
    
    # ==================== Statistics ====================
    
    def _print_label_statistics(self, labels: pd.DataFrame):
        """Print comprehensive label distribution statistics"""
        print("\n" + "="*50)
        print("📊 LABEL DISTRIBUTION SUMMARY")
        print("="*50)
        
        for col in ['buy', 'sell', 'direction', 'regime']:
            if col not in labels.columns:
                continue
                
            print(f"\n▶ {col.upper()} Distribution:")
            vc = labels[col].value_counts(normalize=True).sort_index()
            
            if col in ['buy', 'sell']:
                print(f"  Class 0 (No {col.title()}): {vc.get(0, 0):.3f}")
                print(f"  Class 1 ({col.title()}):    {vc.get(1, 0):.3f}")
            elif col == 'direction':
                print(f"  Class 0 (Down):    {vc.get(0, 0):.3f}")
                print(f"  Class 1 (Up):      {vc.get(1, 0):.3f}")
                print(f"  Class 2 (Neutral): {vc.get(2, 0):.3f}")
            elif col == 'regime':
                regime_names = {
                    0: "BULL_STRONG",
                    1: "BULL_WEAK",
                    2: "RANGE",
                    3: "BEAR_WEAK",
                    4: "BEAR_STRONG",
                    5: "VOLATILE"
                }
                for idx in sorted(vc.index):
                    print(f"  Class {idx} ({regime_names.get(idx, 'Unknown')}): {vc[idx]:.3f}")
        
        print("\n" + "="*50)
        print(f"✅ Total samples: {len(labels)}")
        print("="*50 + "\n")
