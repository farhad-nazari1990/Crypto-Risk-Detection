"""
Rolling MAD (Median Absolute Deviation) for anomaly detection.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectorMAD:
    """
    MAD-based anomaly detector using rolling windows.
    """
    
    def __init__(self, config):
        """
        Initialize MAD detector.
        
        Args:
            config: Dictionary with MAD parameters (window, threshold)
        """
        self.config = config
        self.window = config['window']
        self.threshold = config['threshold']
        
    def detect(self, features_df):
        """
        Detect anomalies using rolling MAD on returns.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of boolean flags (True = anomaly)
        """
        logger.info("Detecting anomalies using Rolling MAD...")
        
        returns = features_df['return_1d'].values
        
        anomalies = np.zeros(len(returns), dtype=bool)
        
        for i in range(self.window, len(returns)):
            window_data = returns[i - self.window:i]
            
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))
            
            if mad == 0:
                continue
            
            # Modified z-score
            z_score = 0.6745 * (returns[i] - median) / mad
            
            if np.abs(z_score) > self.threshold:
                anomalies[i] = True
        
        logger.info(f"MAD detected {np.sum(anomalies)} anomalies.")
        
        return anomalies
