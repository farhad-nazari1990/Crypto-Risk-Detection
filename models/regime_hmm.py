"""
Hidden Markov Model for regime detection in crypto markets.
Identifies 4 market regimes: Bull Rally, Stable Growth, Consolidation, Crash/Panic.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging
import pickle

logger = logging.getLogger(__name__)


class RegimeDetectorHMM:
    """
    HMM-based regime detector for cryptocurrency markets.
    """
    
    def __init__(self, config, regime_labels):
        """
        Initialize HMM regime detector.
        
        Args:
            config: Dictionary with HMM parameters
            regime_labels: Dictionary mapping state indices to regime names
        """
        self.config = config
        self.regime_labels = regime_labels
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fit(self, features_df, feature_columns):
        """
        Fit HMM model to historical data.
        
        Args:
            features_df: DataFrame with engineered features
            feature_columns: List of column names to use for regime detection
        """
        logger.info("Fitting HMM regime detection model...")
        
        self.feature_columns = feature_columns
        X = features_df[feature_columns].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.config['n_components'],
            covariance_type=self.config['covariance_type'],
            n_iter=self.config['n_iter'],
            random_state=self.config['random_state'],
            algorithm=self.config['algorithm']
        )
        
        self.model.fit(X_scaled)
        
        logger.info("HMM model fitted successfully.")
        
        # Map states to regimes based on mean returns
        self._map_states_to_regimes(features_df)
        
    def predict(self, features_df):
        """
        Predict regimes for given features.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of regime labels
        """
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        states = self.model.predict(X_scaled)
        
        # Map states to regime labels
        regimes = [self.state_to_regime[state] for state in states]
        
        return np.array(regimes)
    
    def _map_states_to_regimes(self, features_df):
        """
        Map HMM states to interpretable regime labels based on characteristics.
        Uses mean return and volatility to assign labels.
        """
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        
        # Calculate mean return and volatility for each state
        returns = features_df['return_1d'].values
        volatility = features_df['volatility_7d'].values
        
        state_characteristics = {}
        for state in range(self.config['n_components']):
            mask = states == state
            state_characteristics[state] = {
                'mean_return': np.mean(returns[mask]),
                'mean_volatility': np.mean(volatility[mask]),
                'count': np.sum(mask)
            }
        
        # Sort states by mean return
        sorted_states = sorted(state_characteristics.items(), 
                             key=lambda x: x[1]['mean_return'], 
                             reverse=True)
        
        # Assign regime labels
        # Highest return -> Bull Rally
        # Second highest -> Stable Growth
        # Third -> Consolidation
        # Lowest (likely negative) -> Crash/Panic
        self.state_to_regime = {}
        regime_names = ["Bull Rally", "Stable Growth", "Consolidation", "Crash/Panic"]
        
        for idx, (state, chars) in enumerate(sorted_states):
            self.state_to_regime[state] = regime_names[idx]
            logger.info(f"State {state} -> {regime_names[idx]}: "
                       f"mean_return={chars['mean_return']:.4f}, "
                       f"mean_volatility={chars['mean_volatility']:.4f}, "
                       f"count={chars['count']}")
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'state_to_regime': self.state_to_regime,
                'config': self.config
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.state_to_regime = data['state_to_regime']
            self.config = data['config']
        logger.info(f"Model loaded from {filepath}")
