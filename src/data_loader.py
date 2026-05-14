"""
Data loading and feature engineering module.
Creates 23+ features for regime and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataLoader:
    """
    Loads BTC-USD data from CSV and engineers all necessary features.
    """
    
    def __init__(self, data_path: str = config.DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV and convert to proper datetime index.
        
        Returns:
            DataFrame with datetime index and price/volume columns
        """
        df = pd.read_csv(self.data_path)
        
        # Convert first column to datetime (assuming it's the date column)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Ensure required columns exist (handle different naming)
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                # Try case-insensitive matching
                matching = [c for c in df.columns if c.lower() == col.lower()]
                if matching:
                    df[col] = df[matching[0]]
                else:
                    raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Use Close for returns (Adj Close is similar for crypto)
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        self.df = df
        print(f"✓ Loaded data: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        return self.df
    
    def create_returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create return-based features."""
        df = df.copy()
        
        # Simple returns
        df['return_1d'] = df['Close'].pct_change()
        
        # Log returns
        df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multi-period returns
        for window in config.RETURN_WINDOWS:
            if window == 1:
                continue
            df[f'return_{window}d'] = df['Close'].pct_change(window)
            df[f'log_return_{window}d'] = np.log(df['Close'] / df['Close'].shift(window))
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling volatility features."""
        df = df.copy()
        
        for window in config.VOLATILITY_WINDOWS:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std() * np.sqrt(252)
            df[f'volatility_rolling_mean_{window}d'] = df[f'volatility_{window}d'].rolling(window).mean()
        
        # Parkinson volatility (high-low range)
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                                      (np.log(df['High'] / df['Low']) ** 2).rolling(5).mean()) * np.sqrt(252)
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum (ROC) features."""
        df = df.copy()
        
        for window in config.RETURN_WINDOWS:
            df[f'momentum_{window}d'] = (df['Close'] / df['Close'].shift(window) - 1) * 100
        
        return df
    
    def create_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df = df.copy()
        
        delta = df['return_1d']
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def create_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands and position."""
        df = df.copy()
        
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        df[f'BB_mid_{window}'] = rolling_mean
        df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * num_std)
        df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * num_std)
        
        # BB position (0 = lower, 1 = upper)
        df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
        df[f'BB_position_{window}'] = df[f'BB_position_{window}'].clip(0, 1)
        
        # BB width (volatility proxy)
        df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_mid_{window}']
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df = df.copy()
        
        # Volume moving averages
        for window in config.VOLUME_MA_WINDOWS:
            df[f'volume_MA_{window}'] = df['Volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_MA_{window}']
        
        # Volume Z-score
        volume_rolling_mean = df['Volume'].rolling(21).mean()
        volume_rolling_std = df['Volume'].rolling(21).std()
        df['volume_zscore'] = (df['Volume'] - volume_rolling_mean) / volume_rolling_std
        
        # Volume price trend
        df['vwap'] = (df['Volume'] * df['Close']).rolling(5).sum() / df['Volume'].rolling(5).sum()
        df['price_vs_vwap'] = df['Close'] / df['vwap'] - 1
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features (skew, kurtosis, etc.)."""
        df = df.copy()
        
        # Rolling skewness and kurtosis of returns
        for window in [10, 21]:
            df[f'skew_{window}d'] = df['return_1d'].rolling(window).skew()
            df[f'kurtosis_{window}d'] = df['return_1d'].rolling(window).kurt()
        
        # High-low spread relative to close
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(10).mean()
        
        # Gap from previous close (overnight return)
        df['gap'] = (df['Open'] / df['Close'].shift(1) - 1)
        
        return df
    
    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run all feature engineering functions.
        
        Returns:
            DataFrame with all engineered features
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        print("Engineering features...")
        
        # Base returns
        df = self.create_returns_features(df)
        
        # Volatility
        df = self.create_volatility_features(df)
        
        # Momentum
        df = self.create_momentum_features(df)
        
        # RSI
        df = self.create_rsi(df, config.RSI_WINDOW)
        
        # Bollinger Bands
        df = self.create_bollinger_bands(df, config.BB_WINDOW, config.BB_STD)
        
        # Volume features
        df = self.create_volume_features(df)
        
        # Statistical features
        df = self.create_statistical_features(df)
        
        # Drop NaN rows (from rolling windows)
        initial_len = len(df)
        df = df.dropna()
        print(f"  - Dropped {initial_len - len(df)} rows with NaN values")
        print(f"  - Final feature count: {len(df.columns)}")
        
        return df
    
    def split_train_test(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test periods based on config dates.
        
        Returns:
            (train_df, test_df)
        """
        if df is None:
            df = self.df
        
        train_mask = df.index <= config.TRAIN_END_DATE
        test_mask = df.index >= config.TEST_START_DATE
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"✓ Train set: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
        print(f"✓ Test set: {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df
    
    def get_features_for_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select the feature columns to be used for regime and anomaly detection.
        
        Returns:
            (feature_df, feature_array)
        """
        # Primary features used in both models
        feature_cols = [
            'return_1d', 'log_return_1d',
            'volatility_5d', 'volatility_10d', 'volatility_21d',
            'momentum_5d', 'momentum_10d', 'momentum_21d',
            'RSI_14',
            'BB_position_20', 'BB_width_20',
            'volume_ratio_5', 'volume_ratio_10', 'volume_zscore',
            'hl_spread', 'gap'
        ]
        
        # Keep only columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        feature_df = df[available_cols].copy()
        
        # Handle any remaining NaN (should be minimal after dropna)
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        return feature_df, feature_df.values
    
    def run_full_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Complete data pipeline: load, engineer, split, prepare features.
        
        Returns:
            (full_df, train_df, test_df, train_features, test_features)
        """
        self.load_data()
        full_df = self.engineer_features()
        train_df, test_df = self.split_train_test(full_df)
        
        _, train_features = self.get_features_for_models(train_df)
        _, test_features = self.get_features_for_models(test_df)
        
        print(f"\n✓ Feature shapes: train={train_features.shape}, test={test_features.shape}")
        
        return full_df, train_df, test_df, train_features, test_features


# Quick test when run directly
if __name__ == "__main__":
    loader = DataLoader()
    full_df, train_df, test_df, train_features, test_features = loader.run_full_pipeline()
    print(f"\nTrain features head:\n{pd.DataFrame(train_features[:5]).describe()}")