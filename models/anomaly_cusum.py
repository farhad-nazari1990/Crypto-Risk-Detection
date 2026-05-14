"""
CUSUM (Cumulative Sum) for change point and anomaly detection.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectorCUSUM:
    """
    CUSUM-based anomaly detector for detecting shifts in mean.
    """
    
    def __init__(self, config):
        """
        Initialize CUSUM detector.
        
        Args:
            config: Dictionary with CUSUM parameters (threshold, drift)
        """
        self.config = config
        self.threshold = config['threshold']
        self.drift = config['drift']
        
    def detect(self, features_df):
        """
        Detect anomalies using CUSUM on returns.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of boolean flags (True = anomaly)
        """
        logger.info("Detecting anomalies using CUSUM...")
        
        returns = features_df['return_1d'].values
        
        # Standardize returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        standardized = (returns - mean_return) / std_return
        
        # CUSUM calculation
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        anomalies = np.zeros(len(standardized), dtype=bool)
        
        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i] - self.drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - standardized[i] - self.drift)
            
            if cusum_pos[i] > self.threshold or cusum_neg[i] > self.threshold:
                anomalies[i] = True
        
        logger.info(f"CUSUM detected {np.sum(anomalies)} anomalies.")
        
        return anomalies
