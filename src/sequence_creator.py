"""
SequenceCreator: Constructs sequences of 96 consecutive candles for training
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class SequenceCreator:
    """Creates sequences from features and labels"""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_sequences(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[np.ndarray, dict]:
        """
        Create sequences from features and labels
        
        Args:
            features_df: DataFrame with features (excluding time column)
            labels_df: DataFrame with labels (buy, sell, direction, regime)
        
        Returns:
            X: (N, sequence_length, n_features) array
            y: dict with keys 'buy', 'sell', 'direction', 'regime'
        """
        # Extract feature columns (exclude time)
        feature_cols = [c for c in features_df.columns if c != 'time']
        self.feature_names = feature_cols
        
        # Extract feature matrix
        X_raw = features_df[feature_cols].values
        
        # Handle NaN values (forward fill then backward fill)
        X_raw = pd.DataFrame(X_raw).ffill().bfill().values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Create sequences
        n_samples = len(X_scaled) - self.sequence_length + 1
        n_features = X_scaled.shape[1]
        
        X = np.zeros((n_samples, self.sequence_length, n_features))
        y = {
            'buy': np.zeros(n_samples),
            'sell': np.zeros(n_samples),
            'direction': np.zeros(n_samples, dtype=int),
            'regime': np.zeros(n_samples, dtype=int)
        }
        
        for i in range(n_samples):
            # Input: sequence of features
            X[i] = X_scaled[i:i+self.sequence_length]
            
            # Output: labels at the last timestep of the sequence
            label_idx = i + self.sequence_length - 1
            y['buy'][i] = labels_df.iloc[label_idx]['buy']
            y['sell'][i] = labels_df.iloc[label_idx]['sell']
            y['direction'][i] = labels_df.iloc[label_idx]['direction']
            y['regime'][i] = labels_df.iloc[label_idx]['regime']
        
        print(f"\n✅ Created {n_samples} sequences of length {self.sequence_length}")
        print(f"   Feature shape: {X.shape}")
        
        return X, y
    
    def transform_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Transform new features using fitted scaler"""
        feature_cols = [c for c in features_df.columns if c != 'time']
        X_raw = features_df[feature_cols].values
        X_raw = pd.DataFrame(X_raw).ffill().bfill().values
        X_scaled = self.scaler.transform(X_raw)
        return X_scaled
