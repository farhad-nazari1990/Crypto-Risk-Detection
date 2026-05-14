"""
Feature engineering module for crypto market data.
Generates 23+ technical and statistical features.
FIXED: Ensures proper column access on flattened data
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for cryptocurrency market data.
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Dictionary containing feature engineering parameters
        """
        self.config = config
        
    def engineer_features(self, df):
        """
        Generate all features from OHLCV data.
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # Verify required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Returns features
        df = self._add_returns(df)
        
        # Volatility features
        df = self._add_volatility(df)
        
        # Momentum features
        df = self._add_momentum(df)
        
        # Technical indicators
        df = self._add_rsi(df)
        df = self._add_bollinger_bands(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Price features
        df = self._add_atr(df)
        df = self._add_price_features(df)
        
        # Higher moments
        df = self._add_higher_moments(df)
        
        # Drop NaN rows (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Feature engineering complete. Dropped {initial_rows - len(df)} rows with NaN values.")
        logger.info(f"Final feature set shape: {df.shape}")
        
        return df
    
    def _add_returns(self, df):
        """Add return features for multiple windows."""
        for window in self.config['returns_windows']:
            df[f'return_{window}d'] = df['Close'].pct_change(window)
        return df
    
    def _add_volatility(self, df):
        """Add volatility (rolling standard deviation of returns) features."""
        daily_returns = df['Close'].pct_change()
        for window in self.config['volatility_windows']:
            df[f'volatility_{window}d'] = daily_returns.rolling(window).std()
        return df
    
    def _add_momentum(self, df):
        """Add momentum (rate of change) features."""
        for window in self.config['momentum_windows']:
            df[f'momentum_{window}d'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)
        return df
    
    def _add_rsi(self, df):
        """Add Relative Strength Index."""
        period = self.config['rsi_period']
        delta = df['Close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_bollinger_bands(self, df):
        """Add Bollinger Bands features."""
        period = self.config['bollinger_period']
        num_std = self.config['bollinger_std']
        
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Position within bands (0 = lower band, 1 = upper band)
        df['bb_position'] = (df['Close'] - lower_band) / (upper_band - lower_band)
        
        # Band width (normalized by price)
        df['bb_width'] = (upper_band - lower_band) / sma
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features."""
        ma_period = self.config['volume_ma_period']
        
        volume_ma = df['Volume'].rolling(window=ma_period).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        df['volume_change'] = df['Volume'].pct_change()
        
        return df
    
    def _add_atr(self, df):
        """Add Average True Range."""
        period = self.config['atr_period']
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        return df
    
    def _add_price_features(self, df):
        """Add basic price change features."""
        df['price_change_pct'] = df['Close'].pct_change()
        return df
    
    def _add_higher_moments(self, df):
        """Add skewness and kurtosis of returns."""
        returns = df['Close'].pct_change()
        
        df['skewness_7d'] = returns.rolling(window=7).apply(lambda x: stats.skew(x) if len(x) >= 3 else 0, raw=True)
        df['kurtosis_7d'] = returns.rolling(window=7).apply(lambda x: stats.kurtosis(x) if len(x) >= 4 else 0, raw=True)
        
