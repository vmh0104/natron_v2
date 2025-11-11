"""
Label Generator V2 - Bias-Reduced Institutional Labeling
Generates balanced labels for Buy/Sell/Direction/Regime classification
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """
    Generates bias-reduced institutional labels for multi-task learning.
    
    Outputs:
        - buy: Binary (0/1) with balanced distribution
        - sell: Binary (0/1) with balanced distribution
        - direction: 3-class (0=Down, 1=Up, 2=Neutral)
        - regime: 6-class market state classification
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configuration dictionary with labeling parameters
        """
        self.config = config or {}
        self.neutral_buffer = self.config.get('neutral_buffer', 0.001)
        self.direction_lookahead = self.config.get('direction_lookahead', 3)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.position_threshold_high = self.config.get('position_threshold_high', 0.7)
        self.position_threshold_low = self.config.get('position_threshold_low', 0.3)
        
        # Regime thresholds
        self.bull_strong_trend = self.config.get('bull_strong_trend', 0.02)
        self.bull_strong_adx = self.config.get('bull_strong_adx', 25)
        self.bear_strong_trend = self.config.get('bear_strong_trend', -0.02)
        self.bear_strong_adx = self.config.get('bear_strong_adx', 25)
        self.volatile_atr_percentile = self.config.get('volatile_atr_percentile', 90)
        
        # Balance targets
        self.target_buy_ratio = self.config.get('target_buy_ratio', 0.35)
        self.target_sell_ratio = self.config.get('target_sell_ratio', 0.35)
        
    def generate_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels from OHLCV data and features.
        
        Args:
            df: Original OHLCV DataFrame
            features: Generated features DataFrame
            
        Returns:
            DataFrame with columns: buy, sell, direction, regime
        """
        print("🏷️  Generating bias-reduced labels...")
        
        labels = pd.DataFrame(index=df.index)
        
        # 1. Generate raw signals
        labels['buy'] = self._generate_buy_signals(df, features)
        labels['sell'] = self._generate_sell_signals(df, features)
        labels['direction'] = self._generate_direction_labels(df)
        labels['regime'] = self._generate_regime_labels(df, features)
        
        # 2. Apply bias reduction and balancing
        labels = self._balance_labels(labels)
        
        # 3. Print statistics
        self._print_statistics(labels)
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate BUY signals based on multiple conditions.
        BUY = 1 if at least 2 conditions are true.
        """
        close = df['close']
        volume = df['volume']
        
        # Extract relevant features
        sma_20 = features.get('sma_20', close.rolling(20).mean())
        sma_50 = features.get('sma_50', close.rolling(50).mean())
        rsi_14 = features.get('rsi_14', self._calculate_rsi(close, 14))
        bb_mid = features.get('bb_mid', sma_20)
        volume_sma_20 = features.get('volume_sma_20', volume.rolling(20).mean())
        position_in_range = features.get('position_in_range', 0.5)
        macd_hist = features.get('macd_hist', 0)
        sma_20_slope = features.get('sma_20_slope', 0)
        
        # Condition flags
        conditions = pd.DataFrame(index=df.index)
        
        # 1. close > MA20 > MA50
        conditions['c1'] = (close > sma_20) & (sma_20 > sma_50)
        
        # 2. RSI > 50 or just crossed up from <30
        rsi_crossed_up = (rsi_14 > 30) & (rsi_14.shift(1) <= 30)
        conditions['c2'] = (rsi_14 > 50) | rsi_crossed_up
        
        # 3. close > BB midband and MA20 slope > 0
        conditions['c3'] = (close > bb_mid) & (sma_20_slope > 0)
        
        # 4. volume > 1.5 × rolling20
        conditions['c4'] = volume > (self.volume_threshold * volume_sma_20)
        
        # 5. position_in_range >= 0.7 (near session high)
        conditions['c5'] = position_in_range >= self.position_threshold_high
        
        # 6. MACD_hist > 0 and rising
        macd_rising = macd_hist > macd_hist.shift(1)
        conditions['c6'] = (macd_hist > 0) & macd_rising
        
        # Buy signal if at least 2 conditions are true
        condition_sum = conditions.sum(axis=1)
        buy_signal = (condition_sum >= 2).astype(int)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate SELL signals based on multiple conditions.
        SELL = 1 if at least 2 conditions are true.
        """
        close = df['close']
        volume = df['volume']
        
        # Extract relevant features
        sma_20 = features.get('sma_20', close.rolling(20).mean())
        sma_50 = features.get('sma_50', close.rolling(50).mean())
        rsi_14 = features.get('rsi_14', self._calculate_rsi(close, 14))
        bb_mid = features.get('bb_mid', sma_20)
        volume_sma_20 = features.get('volume_sma_20', volume.rolling(20).mean())
        position_in_range = features.get('position_in_range', 0.5)
        macd_hist = features.get('macd_hist', 0)
        sma_20_slope = features.get('sma_20_slope', 0)
        plus_di = features.get('plus_di', 50)
        minus_di = features.get('minus_di', 50)
        
        # Condition flags
        conditions = pd.DataFrame(index=df.index)
        
        # 1. close < MA20 < MA50
        conditions['c1'] = (close < sma_20) & (sma_20 < sma_50)
        
        # 2. RSI < 50 or just turned down from >70
        rsi_turned_down = (rsi_14 < 70) & (rsi_14.shift(1) >= 70)
        conditions['c2'] = (rsi_14 < 50) | rsi_turned_down
        
        # 3. close < BB midband and MA20 slope < 0
        conditions['c3'] = (close < bb_mid) & (sma_20_slope < 0)
        
        # 4. volume > 1.5 × rolling20 and position_in_range <= 0.3
        conditions['c4'] = (volume > (self.volume_threshold * volume_sma_20)) & \
                          (position_in_range <= self.position_threshold_low)
        
        # 5. MACD_hist < 0 and falling
        macd_falling = macd_hist < macd_hist.shift(1)
        conditions['c5'] = (macd_hist < 0) & macd_falling
        
        # 6. minus_DI > plus_DI
        conditions['c6'] = minus_di > plus_di
        
        # Sell signal if at least 2 conditions are true
        condition_sum = conditions.sum(axis=1)
        sell_signal = (condition_sum >= 2).astype(int)
        
        return sell_signal
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate 3-class direction labels (Up/Down/Neutral).
        
        Returns:
            0 = Down, 1 = Up, 2 = Neutral
        """
        close = df['close']
        future_close = close.shift(-self.direction_lookahead)
        
        price_change = future_close - close
        threshold = close * self.neutral_buffer
        
        # Initialize all as neutral
        direction = pd.Series(2, index=df.index)
        
        # Up if price increases beyond buffer
        direction[price_change > threshold] = 1
        
        # Down if price decreases beyond buffer
        direction[price_change < -threshold] = 0
        
        # Fill NaN at the end with neutral
        direction = direction.fillna(2).astype(int)
        
        return direction
    
    def _generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate 6-class market regime labels.
        
        Regimes:
            0: BULL_STRONG
            1: BULL_WEAK
            2: RANGE (default)
            3: BEAR_WEAK
            4: BEAR_STRONG
            5: VOLATILE
        """
        close = df['close']
        
        # Calculate trend (20-period return)
        trend = close.pct_change(20)
        
        # Extract features
        adx_14 = features.get('adx_14', 20)
        atr_14 = features.get('atr_14', close.rolling(14).std())
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # Calculate ATR threshold for volatility
        atr_threshold = atr_14.quantile(self.volatile_atr_percentile / 100)
        
        # Initialize all as RANGE
        regime = pd.Series(2, index=df.index)
        
        # VOLATILE: ATR > 90th percentile or volume spike
        is_volatile = (atr_14 > atr_threshold) | (volume_ratio > 2.0)
        regime[is_volatile] = 5
        
        # BULL_STRONG: trend > +2%, ADX > 25
        is_bull_strong = (trend > self.bull_strong_trend) & (adx_14 > self.bull_strong_adx)
        regime[is_bull_strong & ~is_volatile] = 0
        
        # BULL_WEAK: 0 < trend <= +2%, ADX <= 25
        is_bull_weak = (trend > 0) & (trend <= self.bull_strong_trend) & (adx_14 <= self.bull_strong_adx)
        regime[is_bull_weak & ~is_volatile] = 1
        
        # BEAR_STRONG: trend < -2%, ADX > 25
        is_bear_strong = (trend < self.bear_strong_trend) & (adx_14 > self.bear_strong_adx)
        regime[is_bear_strong & ~is_volatile] = 4
        
        # BEAR_WEAK: -2% <= trend < 0, ADX <= 25
        is_bear_weak = (trend < 0) & (trend >= self.bear_strong_trend) & (adx_14 <= self.bear_strong_adx)
        regime[is_bear_weak & ~is_volatile] = 3
        
        return regime.astype(int)
    
    def _balance_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """
        Apply bias reduction and label balancing.
        Downsample over-represented classes to achieve target ratios.
        """
        print("⚖️  Balancing labels...")
        
        # Balance Buy/Sell signals
        labels = self._balance_binary_label(labels, 'buy', self.target_buy_ratio)
        labels = self._balance_binary_label(labels, 'sell', self.target_sell_ratio)
        
        # Add stochastic perturbation to avoid overfitting
        labels = self._add_perturbation(labels)
        
        return labels
    
    def _balance_binary_label(self, labels: pd.DataFrame, col: str, target_ratio: float) -> pd.DataFrame:
        """Balance a binary label to target ratio"""
        current_ratio = labels[col].mean()
        
        if current_ratio > target_ratio * 1.2:  # If significantly over-represented
            # Calculate how many to flip from 1 to 0
            n_ones = labels[col].sum()
            n_total = len(labels)
            target_ones = int(target_ratio * n_total)
            n_to_flip = n_ones - target_ones
            
            if n_to_flip > 0:
                # Randomly select indices to flip
                one_indices = labels[labels[col] == 1].index
                flip_indices = np.random.choice(one_indices, size=n_to_flip, replace=False)
                labels.loc[flip_indices, col] = 0
                
        return labels
    
    def _add_perturbation(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Add mild stochastic perturbation to avoid overfitting fixed thresholds"""
        # Randomly flip a small percentage of labels
        perturbation_rate = 0.02  # 2% perturbation
        
        for col in ['buy', 'sell']:
            n_perturb = int(len(labels) * perturbation_rate)
            perturb_indices = np.random.choice(labels.index, size=n_perturb, replace=False)
            labels.loc[perturb_indices, col] = 1 - labels.loc[perturb_indices, col]
        
        return labels
    
    def _print_statistics(self, labels: pd.DataFrame):
        """Print label distribution statistics"""
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ['buy', 'sell', 'direction', 'regime']:
            vc = labels[col].value_counts(normalize=True).sort_index()
            print(f"\n▶ {col.upper()} distribution:")
            print(vc.round(3).to_string())
        
        # Additional statistics
        print(f"\n📈 Buy/Sell Statistics:")
        print(f"   Buy ratio:  {labels['buy'].mean():.3f}")
        print(f"   Sell ratio: {labels['sell'].mean():.3f}")
        print(f"   Both active: {((labels['buy'] == 1) & (labels['sell'] == 1)).sum()} samples")
        print(f"   Neither active: {((labels['buy'] == 0) & (labels['sell'] == 0)).sum()} samples")
        
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI helper"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


# Regime name mapping
REGIME_NAMES = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE"
}


def get_regime_name(regime_id: int) -> str:
    """Convert regime ID to human-readable name"""
    return REGIME_NAMES.get(regime_id, "UNKNOWN")


if __name__ == "__main__":
    # Test label generation
    print("Testing LabelGeneratorV2...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_samples, freq='15min'),
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + np.abs(np.random.randn(n_samples) * 0.3),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) - np.abs(np.random.randn(n_samples) * 0.3),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Generate features (simplified for testing)
    from feature_engine import FeatureEngine
    engine = FeatureEngine()
    features = engine.generate_features(df)
    
    # Generate labels
    generator = LabelGeneratorV2()
    labels = generator.generate_labels(df, features)
    
    print(f"\n✅ Labels shape: {labels.shape}")
    print(f"\nSample labels:\n{labels.head(20)}")
