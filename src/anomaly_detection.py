"""
Anomaly Detection Module
Implements 4 coordinated techniques for market anomaly detection:
1. Isolation Forest (primary)
2. Rolling MAD (Median Absolute Deviation)
3. Bollinger Bands Z-score
4. CUSUM (Cumulative Sum)

Also includes composite score fusion for final anomaly decision.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AnomalyDetector:
    """
    Multi-method anomaly detection system for crypto markets.
    Combines 4 techniques with configurable weights.
    """
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        self.random_seed = random_seed
        
        # Models
        self.isolation_forest = None
        
        # Parameters
        self.iforest_contamination = config.IFOREST_CONTAMINATION
        self.rolling_mad_window = config.ROLLING_MAD_WINDOW
        self.rolling_mad_threshold = config.ROLLING_MAD_THRESHOLD
        self.bollinger_z_window = config.BOLLINGER_Z_WINDOW
        self.bollinger_z_threshold = config.BOLLINGER_Z_THRESHOLD
        self.cusum_threshold = config.CUSUM_THRESHOLD
        self.cusum_min_std = config.CUSUM_MIN_STD
        
        # Weights for fusion
        self.weights = config.ANOMALY_WEIGHTS
        
        # Results storage
        self.anomaly_scores = {}
        self.composite_anomaly = None
        
    def fit_isolation_forest(self, features: np.ndarray) -> IsolationForest:
        """
        Fit Isolation Forest for anomaly detection.
        
        Args:
            features: Shape (n_samples, n_features)
        
        Returns:
            Trained IsolationForest model
        """
        self.isolation_forest = IsolationForest(
            contamination=self.iforest_contamination,
            n_estimators=config.IFOREST_N_ESTIMATORS,
            random_state=self.random_seed,
            verbose=False
        )
        
        self.isolation_forest.fit(features)
        
        print(f"✓ Isolation Forest trained with contamination={self.iforest_contamination}")
        
        return self.isolation_forest
    
    def detect_isolation_forest(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Returns:
            (anomaly_labels, anomaly_scores)
            - labels: 1 for anomaly, 0 for normal
            - scores: negative is more anomalous
        """
        if self.isolation_forest is None:
            self.fit_isolation_forest(features)
        
        # Predict: -1 = anomaly, 1 = normal
        predictions = self.isolation_forest.predict(features)
        anomaly_labels = (predictions == -1).astype(int)
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.isolation_forest.score_samples(features)
        
        print(f"  - Isolation Forest anomalies: {anomaly_labels.sum()} / {len(anomaly_labels)} ({anomaly_labels.mean()*100:.1f}%)")
        
        return anomaly_labels, scores
    
    def detect_rolling_mad(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Rolling Median Absolute Deviation.
        Robust to outliers (uses median instead of mean).
        
        Args:
            series: Time series to detect anomalies in
        
        Returns:
            (anomaly_labels, z_scores)
        """
        rolling_median = series.rolling(window=self.rolling_mad_window, center=False).median()
        rolling_mad = series.rolling(window=self.rolling_mad_window, center=False).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        
        # Modified Z-score using MAD
        mad_z_scores = np.abs(series - rolling_median) / (rolling_mad * 1.4826)  # 1.4826 = scaling factor for normality
        
        anomaly_labels = (mad_z_scores > self.rolling_mad_threshold).astype(int)
        
        print(f"  - Rolling MAD anomalies: {anomaly_labels.sum()} / {len(anomaly_labels)} ({anomaly_labels.mean()*100:.1f}%)")
        
        return anomaly_labels.values, mad_z_scores.values
    
    def detect_bollinger_zscore(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Bollinger Bands Z-score.
        
        Returns:
            (anomaly_labels, z_scores)
        """
        rolling_mean = series.rolling(window=self.bollinger_z_window).mean()
        rolling_std = series.rolling(window=self.bollinger_z_window).std()
        
        z_scores = (series - rolling_mean) / rolling_std
        anomaly_labels = (np.abs(z_scores) > self.bollinger_z_threshold).astype(int)
        
        print(f"  - Bollinger Z-score anomalies: {anomaly_labels.sum()} / {len(anomaly_labels)} ({anomaly_labels.mean()*100:.1f}%)")
        
        return anomaly_labels.values, z_scores.values
    
    def detect_cusum(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using CUSUM (Cumulative Sum) algorithm.
        Best for detecting sustained shifts in mean.
        
        Returns:
            (anomaly_labels, cusum_scores)
        """
        # Standardize the series
        series_std = (series - series.mean()) / series.std()
        
        # Initialize
        n = len(series_std)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        anomaly_labels = np.zeros(n, dtype=int)
        cusum_scores = np.zeros(n)
        
        # Target drift (in std units)
        K = self.cusum_threshold
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + series_std.iloc[i] - K)
            cusum_neg[i] = max(0, cusum_neg[i-1] - series_std.iloc[i] - K)
            
            # Combined CUSUM score
            cusum_scores[i] = max(cusum_pos[i], cusum_neg[i])
            
            # Alert when threshold exceeded
            if cusum_scores[i] > self.cusum_threshold * 2:
                anomaly_labels[i] = 1
                # Reset after detection (optional - commented for multiple detections)
                # cusum_pos[i] = 0
                # cusum_neg[i] = 0
        
        print(f"  - CUSUM anomalies: {anomaly_labels.sum()} / {len(anomaly_labels)} ({anomaly_labels.mean()*100:.1f}%)")
        
        return anomaly_labels, cusum_scores
    
    def normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1] range where higher = more anomalous.
        """
        scores_clean = scores.copy()
        
        # Handle NaN/inf
        scores_clean = np.nan_to_num(scores_clean, nan=0.0, posinf=1.0, neginf=0.0)
        
        if method == 'minmax':
            min_val = scores_clean.min()
            max_val = scores_clean.max()
            if max_val - min_val > 1e-8:
                normalized = (scores_clean - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(scores_clean)
        elif method == 'percentile':
            # Use percentile for robustness
            p95 = np.percentile(scores_clean, 95)
            normalized = np.clip(scores_clean / p95, 0, 1) if p95 > 0 else scores_clean
        else:
            normalized = scores_clean
        
        return normalized
    
    def compute_composite_anomaly(self, 
                                   returns: pd.Series,
                                   volumes: pd.Series,
                                   features: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Compute composite anomaly score from all 4 methods.
        
        Args:
            returns: Daily return series
            volumes: Daily volume series
            features: Feature array for Isolation Forest (optional, uses returns if None)
        
        Returns:
            DataFrame with individual method results and composite score
        """
        print("\n" + "="*50)
        print("ANOMALY DETECTION PIPELINE")
        print("="*50)
        
        results_df = pd.DataFrame(index=returns.index)
        results_df['returns'] = returns.values
        results_df['volumes'] = volumes.values
        
        # Method 1: Isolation Forest (on returns or full features)
        if features is not None:
            iforest_labels, iforest_scores = self.detect_isolation_forest(features)
        else:
            # Use returns as feature
            iforest_labels, iforest_scores = self.detect_isolation_forest(returns.values.reshape(-1, 1))
        
        results_df['iforest_label'] = iforest_labels
        results_df['iforest_score_raw'] = iforest_scores
        results_df['iforest_score_norm'] = self.normalize_scores(-iforest_scores)  # Negative = more anomalous
        
        # Method 2: Rolling MAD on returns
        mad_labels, mad_scores = self.detect_rolling_mad(returns)
        results_df['mad_label'] = mad_labels
        results_df['mad_score_raw'] = mad_scores
        results_df['mad_score_norm'] = self.normalize_scores(mad_scores)
        
        # Method 3: Bollinger Z-score on returns
        bollinger_labels, bollinger_scores = self.detect_bollinger_zscore(returns)
        results_df['bollinger_label'] = bollinger_labels
        results_df['bollinger_score_raw'] = bollinger_scores
        results_df['bollinger_score_norm'] = self.normalize_scores(np.abs(bollinger_scores))
        
        # Method 4: CUSUM on returns
        cusum_labels, cusum_scores = self.detect_cusum(returns)
        results_df['cusum_label'] = cusum_labels
        results_df['cusum_score_raw'] = cusum_scores
        results_df['cusum_score_norm'] = self.normalize_scores(cusum_scores)
        
        # Composite weighted score
        results_df['composite_score'] = (
            results_df['iforest_score_norm'] * self.weights['isolation_forest'] +
            results_df['mad_score_norm'] * self.weights['rolling_mad'] +
            results_df['bollinger_score_norm'] * self.weights['bollinger_z'] +
            results_df['cusum_score_norm'] * self.weights['cusum']
        )
        
        # Binary anomaly decision (threshold at 0.5 composite score)
        results_df['is_anomaly'] = (results_df['composite_score'] > 0.5).astype(int)
        
        # Classify anomaly type
        results_df['anomaly_type'] = 'normal'
        
        # Volume anomaly detection (additional signal)
        volume_zscore = (volumes - volumes.rolling(21).mean()) / volumes.rolling(21).std()
        results_df['volume_anomaly'] = (np.abs(volume_zscore) > 2).astype(int)
        
        # Combine for final type classification
        conditions = [
            (results_df['is_anomaly'] == 1) & (results_df['volume_anomaly'] == 1),
            (results_df['is_anomaly'] == 1) & (results_df['bollinger_label'] == 1),
            (results_df['is_anomaly'] == 1) & (results_df['iforest_label'] == 1),
            (results_df['is_anomaly'] == 1) & (results_df['mad_label'] == 1),
            (results_df['is_anomaly'] == 1)
        ]
        choices = ['volume_spike', 'volatility_anomaly', 'structural_anomaly', 'distribution_anomaly', 'general_anomaly']
        
        results_df['anomaly_type'] = np.select(conditions, choices, default='normal')
        
        # Final summary
        anomaly_count = results_df['is_anomaly'].sum()
        print(f"\n✓ Composite Anomaly Detection Complete:")
        print(f"  - Total anomalies: {anomaly_count} / {len(results_df)} ({anomaly_count/len(results_df)*100:.1f}%)")
        print(f"  - Anomaly types: {results_df['anomaly_type'].value_counts().to_dict()}")
        
        self.composite_anomaly = results_df
        
        return results_df
    
    def get_anomaly_alerts(self, df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
        """
        Generate forward-looking anomaly alerts based on recent composite scores.
        
        Args:
            df: DataFrame with anomaly detection results
            lookback_days: Days to consider for alert generation
        
        Returns:
            DataFrame with alert signals
        """
        alerts = pd.DataFrame(index=df.index)
        alerts['composite_score'] = df['composite_score']
        alerts['is_anomaly'] = df['is_anomaly']
        alerts['anomaly_type'] = df['anomaly_type']
        
        # Rolling anomaly intensity (last N days)
        alerts['anomaly_intensity_3d'] = alerts['composite_score'].rolling(3).mean()
        alerts['anomaly_intensity_5d'] = alerts['composite_score'].rolling(5).mean()
        
        # Alert levels
        # 0 = no alert, 1 = caution, 2 = warning, 3 = severe
        alerts['alert_level'] = 0
        alerts.loc[alerts['anomaly_intensity_3d'] > 0.3, 'alert_level'] = 1
        alerts.loc[alerts['anomaly_intensity_3d'] > 0.5, 'alert_level'] = 2
        alerts.loc[alerts['anomaly_intensity_3d'] > 0.7, 'alert_level'] = 3
        
        # Recent anomaly flag (for trading decisions)
        alerts['recent_anomaly'] = (alerts['is_anomaly'].rolling(lookback_days).sum() > 0).astype(int)
        
        print(f"\n✓ Anomaly Alerts Generated:")
        print(f"  - Alert level distribution: {alerts['alert_level'].value_counts().to_dict()}")
        
        return alerts
    
    def save_models(self, path: str):
        """Save isolation forest model to disk."""
        import joblib
        
        models = {
            'isolation_forest': self.isolation_forest
        }
        
        joblib.dump(models, path)
        print(f"✓ Anomaly models saved to {path}")
    
    def load_models(self, path: str):
        """Load isolation forest model from disk."""
        import joblib
        
        models = joblib.load(path)
        self.isolation_forest = models['isolation_forest']
        print(f"✓ Anomaly models loaded from {path}")
    
    def run_full_pipeline(self, 
                         df: pd.DataFrame, 
                         features: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete anomaly detection pipeline.
        
        Args:
            df: DataFrame with returns and volume columns
            features: Feature array for Isolation Forest (optional)
        
        Returns:
            (anomaly_results_df, alerts_df)
        """
        # Compute composite anomaly
        anomaly_results = self.compute_composite_anomaly(
            returns=df['return_1d'],
            volumes=df['Volume'],
            features=features
        )
        
        # Generate alerts
        alerts = self.get_anomaly_alerts(anomaly_results)
        
        # Merge back to original dataframe
        df_merged = df.copy()
        for col in ['is_anomaly', 'composite_score', 'anomaly_type']:
            if col in anomaly_results.columns:
                df_merged[col] = anomaly_results[col]
        
        # Add alert level
        df_merged['alert_level'] = alerts['alert_level']
        df_merged['recent_anomaly'] = alerts['recent_anomaly']
        
        return df_merged, alerts


# Quick test when run directly
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    full_df, train_df, test_df, train_features, test_features = loader.run_full_pipeline()
    
    # Run anomaly detection
    detector = AnomalyDetector()
    
    # Fit Isolation Forest on training features
    detector.fit_isolation_forest(train_features)
    
    # Run full pipeline on full dataset
    full_df_with_anomaly, alerts = detector.run_full_pipeline(full_df, train_features)
    
    print(f"\n✓ Final DataFrame columns: {full_df_with_anomaly.columns.tolist()}")
    print(f"✓ Anomaly columns added: is_anomaly, composite_score, anomaly_type, alert_level, recent_anomaly")