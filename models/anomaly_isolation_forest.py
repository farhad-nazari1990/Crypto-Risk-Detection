"""
Isolation Forest for anomaly detection in crypto markets.
Primary anomaly detection method.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import pickle

logger = logging.getLogger(__name__)


class AnomalyDetectorIsolationForest:
    """
    Isolation Forest-based anomaly detector.
    """
    
    def __init__(self, config):
        """
        Initialize Isolation Forest detector.
        
        Args:
            config: Dictionary with Isolation Forest parameters
        """
        self.config = config
        self.model = IsolationForest(
            contamination=config['contamination'],
            n_estimators=config['n_estimators'],
            max_samples=config['max_samples'],
            random_state=config['random_state'],
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fit(self, features_df, feature_columns):
        """
        Fit Isolation Forest model.
        
        Args:
            features_df: DataFrame with engineered features
            feature_columns: List of column names to use for anomaly detection
        """
        logger.info("Fitting Isolation Forest anomaly detection model...")
        
        self.feature_columns = feature_columns
        X = features_df[feature_columns].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        
        logger.info("Isolation Forest model fitted successfully.")
        
    def predict(self, features_df):
        """
        Predict anomalies.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of boolean flags (True = anomaly)
        """
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        
        # Convert to boolean (True = anomaly)
        anomalies = predictions == -1
        
        return anomalies
    
    def get_anomaly_scores(self, features_df):
        """
        Get anomaly scores (lower = more anomalous).
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of anomaly scores
        """
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
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
            self.config = data['config']
        logger.info(f"Model loaded from {filepath}")
