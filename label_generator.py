"""
LabelGeneratorV2: Bias-Reduced Institutional Labeling
Generates Buy/Sell, Direction, and Regime labels with balanced distributions
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LabelGeneratorV2:
    """
    Generates multi-task labels with reduced directional bias.
    Ensures balanced label distributions for robust learning.
    """
    
    def __init__(
        self,
        neutral_buffer: float = 0.001,
        lookahead_candles: int = 3,
        buy_threshold: int = 2,
        sell_threshold: int = 2,
        balance_labels: bool = True,
        stochastic_perturbation: float = 0.05
    ):
        self.neutral_buffer = neutral_buffer
        self.lookahead_candles = lookahead_candles
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.balance_labels = balance_labels
        self.stochastic_perturbation = stochastic_perturbation
        
    def generate_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        
        Args:
            df: Original OHLCV DataFrame
            features_df: DataFrame with technical features
            
        Returns:
            DataFrame with columns: buy, sell, direction, regime
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate individual label types
        labels['buy'] = self._generate_buy_labels(df, features_df)
        labels['sell'] = self._generate_sell_labels(df, features_df)
        labels['direction'] = self._generate_direction_labels(df)
        labels['regime'] = self._generate_regime_labels(df, features_df)
        
        # Balance labels if requested
        if self.balance_labels:
            labels = self._balance_labels(labels, df)
        
        # Print label statistics
        self._print_label_statistics(labels)
        
        return labels
    
    def _generate_buy_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Generate BUY labels (1 if ≥buy_threshold conditions true)
        """
        close = df['close'].values
        volume = df['volume'].values
        
        # Get required features
        ma20 = features_df.get('MA20', pd.Series(index=df.index, data=0)).values
        ma50 = features_df.get('MA50', pd.Series(index=df.index, data=0)).values
        rsi = features_df.get('RSI', pd.Series(index=df.index, data=50)).values
        bb_mid = features_df.get('BB_mid', close).values
        ma20_slope = features_df.get('MA20_slope', pd.Series(index=df.index, data=0)).values
        volume_ma20 = features_df.get('Volume_MA20', volume).values
        position_in_range = features_df.get('Position_in_range', pd.Series(index=df.index, data=0.5)).values
        macd_hist = features_df.get('MACD_hist', pd.Series(index=df.index, data=0)).values
        
        # Calculate conditions
        conditions = np.zeros(len(df))
        
        # Condition 1: close > MA20 > MA50
        cond1 = (close > ma20) & (ma20 > ma50)
        conditions += cond1.astype(int)
        
        # Condition 2: RSI > 50 or just crossed up from <30
        rsi_prev = np.roll(rsi, 1)
        cond2 = (rsi > 50) | ((rsi > 30) & (rsi_prev <= 30))
        conditions += cond2.astype(int)
        
        # Condition 3: close > BB midband and MA20 slope > 0
        cond3 = (close > bb_mid) & (ma20_slope > 0)
        conditions += cond3.astype(int)
        
        # Condition 4: volume > 1.5 × rolling20
        cond4 = volume > (1.5 * volume_ma20)
        conditions += cond4.astype(int)
        
        # Condition 5: position_in_range ≥ 0.7
        cond5 = position_in_range >= 0.7
        conditions += cond5.astype(int)
        
        # Condition 6: MACD_hist > 0 and rising
        macd_hist_prev = np.roll(macd_hist, 1)
        cond6 = (macd_hist > 0) & (macd_hist > macd_hist_prev)
        conditions += cond6.astype(int)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.normal(0, self.stochastic_perturbation, len(conditions))
            conditions = conditions + noise
        
        # BUY = 1 if ≥buy_threshold conditions true
        buy_labels = (conditions >= self.buy_threshold).astype(int)
        
        return pd.Series(buy_labels, index=df.index)
    
    def _generate_sell_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Generate SELL labels (1 if ≥sell_threshold conditions true)
        """
        close = df['close'].values
        volume = df['volume'].values
        
        # Get required features
        ma20 = features_df.get('MA20', pd.Series(index=df.index, data=0)).values
        ma50 = features_df.get('MA50', pd.Series(index=df.index, data=0)).values
        rsi = features_df.get('RSI', pd.Series(index=df.index, data=50)).values
        bb_mid = features_df.get('BB_mid', close).values
        ma20_slope = features_df.get('MA20_slope', pd.Series(index=df.index, data=0)).values
        volume_ma20 = features_df.get('Volume_MA20', volume).values
        position_in_range = features_df.get('Position_in_range', pd.Series(index=df.index, data=0.5)).values
        macd_hist = features_df.get('MACD_hist', pd.Series(index=df.index, data=0)).values
        plus_di = features_df.get('Plus_DI', pd.Series(index=df.index, data=0)).values
        minus_di = features_df.get('Minus_DI', pd.Series(index=df.index, data=0)).values
        
        # Calculate conditions
        conditions = np.zeros(len(df))
        
        # Condition 1: close < MA20 < MA50
        cond1 = (close < ma20) & (ma20 < ma50)
        conditions += cond1.astype(int)
        
        # Condition 2: RSI < 50 or just turned down from >70
        rsi_prev = np.roll(rsi, 1)
        cond2 = (rsi < 50) | ((rsi < 70) & (rsi_prev >= 70))
        conditions += cond2.astype(int)
        
        # Condition 3: close < BB midband and MA20 slope < 0
        cond3 = (close < bb_mid) & (ma20_slope < 0)
        conditions += cond3.astype(int)
        
        # Condition 4: volume > 1.5 × rolling20 and position_in_range ≤ 0.3
        cond4 = (volume > (1.5 * volume_ma20)) & (position_in_range <= 0.3)
        conditions += cond4.astype(int)
        
        # Condition 5: MACD_hist < 0 and falling
        macd_hist_prev = np.roll(macd_hist, 1)
        cond5 = (macd_hist < 0) & (macd_hist < macd_hist_prev)
        conditions += cond5.astype(int)
        
        # Condition 6: minus_DI > plus_DI
        cond6 = minus_di > plus_di
        conditions += cond6.astype(int)
        
        # Add stochastic perturbation
        if self.stochastic_perturbation > 0:
            noise = np.random.normal(0, self.stochastic_perturbation, len(conditions))
            conditions = conditions + noise
        
        # SELL = 1 if ≥sell_threshold conditions true
        sell_labels = (conditions >= self.sell_threshold).astype(int)
        
        return pd.Series(sell_labels, index=df.index)
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Direction labels (3-class: Up=1, Down=0, Neutral=2)
        """
        close = df['close'].values
        
        # Forward-looking: compare close[i+lookahead] with close[i]
        future_close = np.roll(close, -self.lookahead_candles)
        
        # Calculate price change
        price_change = (future_close - close) / close
        
        # Generate labels
        direction = np.full(len(df), 2, dtype=int)  # Default: Neutral
        
        # Up: close[i+3] > close[i] + neutral_buffer
        direction[price_change > self.neutral_buffer] = 1
        
        # Down: close[i+3] < close[i] - neutral_buffer
        direction[price_change < -self.neutral_buffer] = 0
        
        # Set last lookahead_candles to Neutral (no future data)
        direction[-self.lookahead_candles:] = 2
        
        return pd.Series(direction, index=df.index)
    
    def _generate_regime_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Generate Regime labels (6 classes)
        0: BULL_STRONG, 1: BULL_WEAK, 2: RANGE, 3: BEAR_WEAK, 4: BEAR_STRONG, 5: VOLATILE
        """
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate trend (20-period return)
        trend = (close - np.roll(close, 20)) / np.roll(close, 20)
        
        # Get ADX
        adx = features_df.get('ADX', pd.Series(index=df.index, data=0)).values
        
        # Get ATR
        atr = features_df.get('ATR20', pd.Series(index=df.index, data=0)).values
        
        # Initialize regime labels (default: RANGE = 2)
        regime = np.full(len(df), 2, dtype=int)
        
        # BULL_STRONG: trend > +2%, ADX > 25
        regime[(trend > 0.02) & (adx > 25)] = 0
        
        # BULL_WEAK: 0 < trend ≤ +2%, ADX ≤ 25
        regime[(trend > 0) & (trend <= 0.02) & (adx <= 25)] = 1
        
        # BEAR_WEAK: −2% ≤ trend < 0, ADX ≤ 25
        regime[(trend >= -0.02) & (trend < 0) & (adx <= 25)] = 3
        
        # BEAR_STRONG: trend < −2%, ADX > 25
        regime[(trend < -0.02) & (adx > 25)] = 4
        
        # VOLATILE: ATR > 90th percentile or volume spike
        atr_threshold = np.nanpercentile(atr, 90)
        volume_threshold = np.nanpercentile(volume, 90)
        regime[(atr > atr_threshold) | (volume > volume_threshold)] = 5
        
        return pd.Series(regime, index=df.index)
    
    def _balance_labels(self, labels: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance label distributions by downsampling over-represented classes
        """
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
            
            if buy_ratio > target_ratio:
                # Downsample BUY
                buy_indices = balanced_labels[balanced_labels['buy'] == 1].index
                keep_ratio = target_ratio / buy_ratio
                keep_indices = np.random.choice(
                    buy_indices,
                    size=int(len(buy_indices) * keep_ratio),
                    replace=False
                )
                balanced_labels.loc[balanced_labels['buy'] == 1, 'buy'] = 0
                balanced_labels.loc[keep_indices, 'buy'] = 1
            
            if sell_ratio > target_ratio:
                # Downsample SELL
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
        """
        Print label distribution statistics
        """
        print("\n=== 📊 Label Distribution Summary ===")
        
        for col in ["buy", "sell", "direction", "regime"]:
            if col in labels.columns:
                vc = labels[col].value_counts(normalize=True).sort_index()
                print(f"\n▶ {col.upper()} distribution:")
                for idx, val in vc.items():
                    print(f"  {idx:>6}: {val:.3f}")
        
        print("\n" + "="*50)
