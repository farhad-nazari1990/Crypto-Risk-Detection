"""
Regime Detection Module
Implements HMM, GMM, and Change Point Detection for market regime classification.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RegimeDetector:
    """
    Market regime detection using:
    - Hidden Markov Model (HMM) - primary
    - Gaussian Mixture Model (GMM) - baseline comparison
    - Change Point Detection (CPD) - structural break detection
    """
    
    def __init__(self, n_states: int = config.HMM_N_STATES, random_seed: int = config.RANDOM_SEED):
        self.n_states = n_states
        self.random_seed = random_seed
        
        # Models
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        
        # Results storage
        self.hmm_regimes = None
        self.gmm_regimes = None
        self.change_points = None
        
        # Feature names for interpretability
        self.feature_names = None
        
    def fit_hmm(self, features: np.ndarray) -> hmm.GaussianHMM:
        """
        Fit Hidden Markov Model on scaled features.
        
        Args:
            features: Shape (n_samples, n_features)
        
        Returns:
            Trained HMM model
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=self.random_seed,
            verbose=False
        )
        
        # Fit model
        self.hmm_model.fit(features_scaled)
        
        # Predict hidden states
        self.hmm_regimes = self.hmm_model.predict(features_scaled)
        
        # Compute transition matrix
        transition_matrix = self.hmm_model.transmat_
        
        print(f"✓ HMM trained with {self.n_states} regimes")
        print(f"  - Transition matrix shape: {transition_matrix.shape}")
        print(f"  - Regime distribution: {np.bincount(self.hmm_regimes)}")
        
        return self.hmm_model
    
    def fit_gmm(self, features: np.ndarray) -> GaussianMixture:
        """
        Fit Gaussian Mixture Model as baseline comparison.
        
        Args:
            features: Shape (n_samples, n_features)
        
        Returns:
            Trained GMM model
        """
        # Scale features
        features_scaled = self.scaler.transform(features) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(features)
        
        # Initialize GMM
        self.gmm_model = GaussianMixture(
            n_components=self.n_states,
            covariance_type='full',
            random_state=self.random_seed,
            max_iter=500
        )
        
        # Fit and predict
        self.gmm_model.fit(features_scaled)
        self.gmm_regimes = self.gmm_model.predict(features_scaled)
        
        print(f"✓ GMM trained with {self.n_states} components")
        print(f"  - Regime distribution: {np.bincount(self.gmm_regimes)}")
        
        return self.gmm_model
    
    def detect_change_points(self, prices: np.ndarray, model: str = config.CHANGE_POINT_MODEL) -> List[int]:
        """
        Detect structural change points in price series.
        
        Args:
            prices: 1D array of price data
            model: Cost function ('l1', 'l2', 'rbf')
        
        Returns:
            List of change point indices
        """
        # Reshape for ruptures (needs 2D)
        signal = prices.reshape(-1, 1)
        
        # Use PELT algorithm for efficiency
        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=config.CHANGE_POINT_PENALTY)
        
        # Remove last index (end of series)
        change_points = [cp for cp in change_points if cp < len(prices)]
        
        self.change_points = change_points
        
        print(f"✓ Change points detected: {len(change_points)} structural breaks")
        if len(change_points) <= 10:
            print(f"  - Positions: {change_points[:10]}")
        
        return change_points
    
    def compute_regime_statistics(self, df: pd.DataFrame, regime_column: str = 'hmm_regime') -> pd.DataFrame:
        """
        Compute performance statistics per regime.
        
        Args:
            df: DataFrame with returns and regime labels
            regime_column: Name of regime column
        
        Returns:
            DataFrame with per-regime metrics
        """
        if regime_column not in df.columns:
            raise ValueError(f"Column {regime_column} not found")
        
        stats = []
        
        for regime in range(self.n_states):
            regime_data = df[df[regime_column] == regime]
            
            if len(regime_data) == 0:
                continue
                
            mean_return = regime_data['return_1d'].mean()
            volatility = regime_data['return_1d'].std()
            sharpe = (mean_return / volatility) * np.sqrt(252) if volatility > 0 else 0
            max_drawdown = self._compute_max_drawdown(regime_data['Close'])
            win_rate = (regime_data['return_1d'] > 0).mean()
            
            stats.append({
                'regime': regime,
                'regime_name': config.REGIME_NAMES.get(regime, f"Regime_{regime}"),
                'days': len(regime_data),
                'pct_time': len(regime_data) / len(df) * 100,
                'mean_daily_return': mean_return * 100,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100,
                'win_rate': win_rate * 100
            })
        
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('sharpe_ratio', ascending=False)
        
        return stats_df
    
    def _compute_max_drawdown(self, prices: pd.Series) -> float:
        """Compute maximum drawdown from price series."""
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def map_regimes_to_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map numeric regimes to interpretable names based on sharpe ratio.
        
        Args:
            df: DataFrame with hmm_regime column and returns
        
        Returns:
            DataFrame with regime_name column
        """
        # Compute per-regime sharpe to order regimes
        regime_sharpe = {}
        for regime in range(self.n_states):
            regime_returns = df[df['hmm_regime'] == regime]['return_1d']
            if len(regime_returns) > 0:
                sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else -999
                regime_sharpe[regime] = sharpe
        
        # Sort regimes by sharpe ratio (highest = Bull, lowest = Crash)
        sorted_regimes = sorted(regime_sharpe.items(), key=lambda x: x[1], reverse=True)
        
        # Assign names based on rank
        regime_mapping = {}
        for rank, (regime, sharpe) in enumerate(sorted_regimes):
            if rank == 0:
                regime_mapping[regime] = 0  # Bull Rally
            elif rank == 1:
                regime_mapping[regime] = 1  # Stable Growth
            elif rank == 2:
                regime_mapping[regime] = 2  # Consolidation
            else:
                regime_mapping[regime] = 3  # Crash/Panic
        
        # Apply mapping
        df['regime_clean'] = df['hmm_regime'].map(regime_mapping)
        df['regime_name'] = df['regime_clean'].map(config.REGIME_NAMES)
        
        # Verify all regimes have names
        print(f"✓ Regime mapping: {regime_mapping}")
        
        return df
    
    def compute_model_comparison(self, df: pd.DataFrame) -> Dict:
        """
        Compare HMM vs GMM performance using clustering metrics.
        
        Returns:
            Dictionary with comparison metrics
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(df['hmm_regime'], df['gmm_regime'])
        nmi = normalized_mutual_info_score(df['hmm_regime'], df['gmm_regime'])
        
        # Compute stability (fewer transitions = more stable)
        hmm_transitions = np.sum(np.diff(df['hmm_regime']) != 0)
        gmm_transitions = np.sum(np.diff(df['gmm_regime']) != 0)
        
        # Average regime duration
        hmm_durations = self._compute_regime_durations(df['hmm_regime'].values)
        gmm_durations = self._compute_regime_durations(df['gmm_regime'].values)
        
        comparison = {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'hmm_transitions': hmm_transitions,
            'gmm_transitions': gmm_transitions,
            'hmm_avg_duration_days': np.mean(hmm_durations),
            'gmm_avg_duration_days': np.mean(gmm_durations),
            'hmm_more_stable': hmm_transitions < gmm_transitions
        }
        
        print(f"\n✓ Model Comparison:")
        print(f"  - Adjusted Rand Index: {ari:.3f}")
        print(f"  - Normalized Mutual Info: {nmi:.3f}")
        print(f"  - HMM transitions: {hmm_transitions} | GMM: {gmm_transitions}")
        print(f"  - HMM avg duration: {comparison['hmm_avg_duration_days']:.1f} days")
        
        return comparison
    
    def _compute_regime_durations(self, regimes: np.ndarray) -> List[int]:
        """Compute duration of consecutive same-regime periods."""
        durations = []
        current_regime = regimes[0]
        current_duration = 1
        
        for regime in regimes[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime
                current_duration = 1
        durations.append(current_duration)
        
        return durations
    
    def predict_future_regime(self, features: np.ndarray, n_steps: int = 5) -> np.ndarray:
        """
        Predict future regimes using HMM.
        
        Args:
            features: Current feature array (last n_samples)
            n_steps: Number of steps to predict
        
        Returns:
            Array of predicted regime indices
        """
        if self.hmm_model is None:
            raise ValueError("HMM model not fitted. Call fit_hmm first.")
        
        features_scaled = self.scaler.transform(features)
        
        # Use last observation to generate future samples
        last_state = self.hmm_model.predict(features_scaled)[-1]
        
        # Sample from transition matrix
        predictions = [last_state]
        current_state = last_state
        
        for _ in range(n_steps):
            # Sample next state from transition probabilities
            next_state = np.random.choice(
                self.n_states,
                p=self.hmm_model.transmat_[current_state]
            )
            predictions.append(next_state)
            current_state = next_state
        
        return np.array(predictions[1:])  # Exclude current
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        import joblib
        
        models = {
            'hmm': self.hmm_model,
            'gmm': self.gmm_model,
            'scaler': self.scaler
        }
        
        joblib.dump(models, path)
        print(f"✓ Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        import joblib
        
        models = joblib.load(path)
        self.hmm_model = models['hmm']
        self.gmm_model = models['gmm']
        self.scaler = models['scaler']
        print(f"✓ Models loaded from {path}")
    
    def run_full_pipeline(self, df: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
        """
        Complete regime detection pipeline.
        
        Args:
            df: DataFrame with price and return data
            features: Feature array for training
        
        Returns:
            DataFrame with added regime columns
        """
        print("\n" + "="*50)
        print("REGIME DETECTION PIPELINE")
        print("="*50)
        
        # 1. Fit HMM
        self.fit_hmm(features)
        df['hmm_regime'] = self.hmm_regimes
        
        # 2. Fit GMM for comparison
        self.fit_gmm(features)
        df['gmm_regime'] = self.gmm_regimes
        
        # 3. Map regimes to interpretable names
        df = self.map_regimes_to_names(df)
        
        # 4. Compute regime statistics
        regime_stats = self.compute_regime_statistics(df, 'hmm_regime')
        
        # 5. Detect change points on Close prices
        change_points = self.detect_change_points(df['Close'].values)
        df['is_change_point'] = 0
        for cp in change_points:
            if cp < len(df):
                df.iloc[cp, df.columns.get_loc('is_change_point')] = 1
        
        # 6. Model comparison
        comparison = self.compute_model_comparison(df)
        
        print("\n" + "-"*50)
        print("REGIME STATISTICS:")
        print("-"*50)
        print(regime_stats.to_string(index=False))
        
        return df, regime_stats, comparison


# Quick test when run directly
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    full_df, train_df, test_df, train_features, test_features = loader.run_full_pipeline()
    
    # Run regime detection
    detector = RegimeDetector()
    full_df_with_regimes, regime_stats, comparison = detector.run_full_pipeline(full_df, train_features)
    
    print(f"\n✓ Final DataFrame shape: {full_df_with_regimes.shape}")
    print(f"✓ Regime columns added: hmm_regime, gmm_regime, regime_clean, regime_name, is_change_point")