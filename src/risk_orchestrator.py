"""
Risk Decision Orchestrator
Combines regime detection and anomaly detection into unified risk signals.
Generates actionable trading decisions: STRONG_BUY, BUY, HOLD, REDUCE, EXIT.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RiskOrchestrator:
    """
    Orchestrates regime + anomaly to produce final risk signals.
    
    Signal Matrix (from config):
    - (Bull, no anomaly) → STRONG_BUY (100%)
    - (Bull, anomaly) → BUY (80%)
    - (Stable, no anomaly) → BUY (80%)
    - (Stable, anomaly) → HOLD (50%)
    - (Consolidation, no anomaly) → HOLD (50%)
    - (Consolidation, anomaly) → REDUCE (25%)
    - (Crash, no anomaly) → REDUCE (25%)
    - (Crash, anomaly) → EXIT (0%)
    """
    
    def __init__(self):
        self.signal_matrix = config.SIGNAL_MATRIX
        self.signal_to_position = config.SIGNAL_TO_POSITION
        self.signal_names = config.SIGNAL_NAMES
        self.regime_names = config.REGIME_NAMES
        
        # Results storage
        self.risk_signals = None
        self.position_sizes = None
        self.signal_history = None
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate risk signals based on regime and anomaly detection.
        
        Required columns in df:
            - 'regime_clean': int (0-3) from regime detection
            - 'is_anomaly': int (0 or 1) from anomaly detection
            - 'composite_score': float (optional, for confidence)
            - 'alert_level': int (0-3, optional)
        
        Returns:
            DataFrame with added signal columns
        """
        df_signals = df.copy()
        
        # Check required columns
        required_cols = ['regime_clean', 'is_anomaly']
        missing_cols = [col for col in required_cols if col not in df_signals.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Generate base signal from matrix
        signals = []
        confidence_scores = []
        
        for idx, row in df_signals.iterrows():
            regime = row['regime_clean']
            anomaly = row['is_anomaly']
            
            # Get signal from matrix
            key = (regime, anomaly)
            signal = self.signal_matrix.get(key, 2)  # Default to HOLD (2)
            signals.append(signal)
            
            # Calculate confidence score (0-1)
            confidence = self._calculate_confidence(row, regime, anomaly, signal)
            confidence_scores.append(confidence)
        
        df_signals['risk_signal'] = signals
        df_signals['signal_name'] = df_signals['risk_signal'].map(self.signal_names)
        df_signals['position_size'] = df_signals['risk_signal'].map(self.signal_to_position)
        df_signals['confidence'] = confidence_scores
        
        # Add additional risk metrics
        df_signals = self._add_risk_metrics(df_signals)
        
        # Store results
        self.risk_signals = df_signals
        self.position_sizes = df_signals['position_size'].values
        self.signal_history = df_signals[['regime_name', 'signal_name', 'position_size', 'confidence', 'is_anomaly']]
        
        # Print summary
        self._print_summary(df_signals)
        
        return df_signals
    
    def _calculate_confidence(self, 
                             row: pd.Series, 
                             regime: int, 
                             anomaly: int, 
                             signal: int) -> float:
        """
        Calculate confidence score for the generated signal.
        
        Factors:
        - Anomaly composite score (higher anomaly = lower confidence for bullish signals)
        - Regime stability (based on recent history if available)
        - Signal extremity (extreme signals get lower confidence)
        """
        confidence = 0.7  # Base confidence
        
        # Adjust by anomaly composite score if available
        if 'composite_score' in row and not pd.isna(row['composite_score']):
            if signal in [0, 1]:  # Bullish signals
                confidence *= (1 - row['composite_score'] * 0.5)
            elif signal in [3, 4]:  # Bearish signals
                confidence *= (0.5 + row['composite_score'] * 0.5)
            else:  # Hold signals
                confidence *= (1 - abs(row['composite_score'] - 0.5) * 0.3)
        
        # Adjust by alert level if available
        if 'alert_level' in row and not pd.isna(row['alert_level']):
            alert_penalty = row['alert_level'] * 0.1
            confidence *= (1 - alert_penalty)
        
        # Adjust by regime type
        if regime == 3:  # Crash regime
            confidence *= 0.8  # Lower confidence in crash
        
        # Clip to [0, 1]
        confidence = np.clip(confidence, 0.1, 0.95)
        
        return confidence
    
    def _add_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived risk metrics to the dataframe."""
        
        # Risk score (inverse of position size, normalized)
        df['risk_score'] = 1 - df['position_size']
        
        # Signal change detection (for monitoring transitions)
        df['signal_change'] = df['risk_signal'].diff().fillna(0).astype(int)
        df['signal_change_direction'] = df['signal_change'].apply(
            lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change')
        )
        
        # Recent signal stability (rolling std of position size)
        df['position_stability_5d'] = df['position_size'].rolling(5).std()
        
        # Drawdown warning (if price is below recent high)
        if 'Close' in df.columns:
            rolling_max = df['Close'].rolling(20).max()
            df['drawdown_from_high'] = (rolling_max - df['Close']) / rolling_max
            df['dd_warning'] = (df['drawdown_from_high'] > 0.15).astype(int)
        
        # Volume anomaly adjustment
        if 'volume_anomaly' in df.columns:
            df['volume_risk'] = df['volume_anomaly'] * 0.2
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of generated signals."""
        print("\n" + "="*50)
        print("RISK ORCHESTRATOR SUMMARY")
        print("="*50)
        
        # Signal distribution
        signal_dist = df['signal_name'].value_counts()
        print("\n📊 Signal Distribution:")
        for signal, count in signal_dist.items():
            pct = count / len(df) * 100
            print(f"  - {signal}: {count} days ({pct:.1f}%)")
        
        # Position size stats
        print(f"\n💰 Position Size Statistics:")
        print(f"  - Mean: {df['position_size'].mean():.1%}")
        print(f"  - Std: {df['position_size'].std():.1%}")
        print(f"  - Min: {df['position_size'].min():.1%}")
        print(f"  - Max: {df['position_size'].max():.1%}")
        
        # Confidence stats
        print(f"\n🎯 Confidence Score Statistics:")
        print(f"  - Mean: {df['confidence'].mean():.2f}")
        print(f"  - Std: {df['confidence'].std():.2f}")
        
        # Signal changes
        n_changes = (df['signal_change'] != 0).sum()
        print(f"\n🔄 Signal Changes:")
        print(f"  - Total changes: {n_changes}")
        print(f"  - Changes per week: {n_changes / (len(df)/7):.1f}")
        
        # Recent period summary (last 30 days if available)
        if len(df) > 30:
            recent = df.tail(30)
            print(f"\n📈 Recent Period (last 30 days):")
            print(f"  - Current signal: {recent['signal_name'].iloc[-1]}")
            print(f"  - Current position: {recent['position_size'].iloc[-1]:.0%}")
            print(f"  - Avg position last 30d: {recent['position_size'].mean():.1%}")
    
    def get_current_signal(self) -> Dict:
        """Get the most recent risk signal."""
        if self.risk_signals is None:
            return {"error": "No signals generated yet"}
        
        last_row = self.risk_signals.iloc[-1]
        
        return {
            "date": last_row.name,
            "regime": last_row.get('regime_name', 'Unknown'),
            "signal": last_row['signal_name'],
            "position_size": last_row['position_size'],
            "confidence": last_row['confidence'],
            "anomaly_detected": bool(last_row['is_anomaly']),
            "risk_score": last_row.get('risk_score', 0)
        }
    
    def get_signal_explanation(self, idx: int = -1) -> str:
        """
        Generate human-readable explanation for a signal.
        
        Args:
            idx: Index position (-1 for most recent)
        
        Returns:
            Explanation string
        """
        if self.risk_signals is None:
            return "No signals generated"
        
        row = self.risk_signals.iloc[idx]
        date = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        
        explanation = f"""
📅 Date: {date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Market Regime: {row.get('regime_name', 'Unknown')}
⚠️  Anomaly Detected: {'YES' if row['is_anomaly'] else 'NO'}
{'   └─ Type: ' + row.get('anomaly_type', 'N/A') if row['is_anomaly'] else ''}

🎯 Risk Signal: {row['signal_name']}
💰 Position Size: {row['position_size']:.0%}
🎲 Confidence: {row['confidence']:.0%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Recommended Action: 
"""
        
        signal = row['signal_name']
        if signal == "STRONG_BUY":
            explanation += "   ✅ Aggressively accumulate. Market conditions are optimal."
        elif signal == "BUY":
            explanation += "   📈 Accumulate gradually. Good risk/reward setup."
        elif signal == "HOLD":
            explanation += "   🤚 Maintain current position. Wait for clarity."
        elif signal == "REDUCE":
            explanation += "   ⚠️ Reduce exposure by 50-75%. Rising risks detected."
        else:  # EXIT
            explanation += "   🚨 Exit completely or hedge. High probability of drawdown."
        
        return explanation
    
    def export_signal_log(self, path: str):
        """Export signal history to CSV."""
        if self.risk_signals is None:
            print("No signals to export")
            return
        
        export_cols = ['regime_name', 'is_anomaly', 'anomaly_type', 'composite_score',
                       'risk_signal', 'signal_name', 'position_size', 'confidence', 
                       'risk_score', 'signal_change']
        
        available_cols = [col for col in export_cols if col in self.risk_signals.columns]
        
        self.risk_signals[available_cols].to_csv(path)
        print(f"✓ Signal log exported to {path}")
    
    def generate_case_study_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Generate the key metrics needed for the freelance case study.
        This is the core output for selling the service.
        
        Returns:
            Dictionary with case study metrics
        """
        if self.risk_signals is None:
            self.generate_signals(df)
        
        # Need close prices for backtest (will be done in backtest module)
        # Here we return the pre-backtest metrics
        
        metrics = {
            "data_period": {
                "start": df.index[0].strftime('%Y-%m-%d'),
                "end": df.index[-1].strftime('%Y-%m-%d'),
                "total_days": len(df)
            },
            "regime_detection": {
                "n_regimes": 4,
                "regime_distribution": df['regime_name'].value_counts().to_dict() if 'regime_name' in df else {},
                "n_change_points": df['is_change_point'].sum() if 'is_change_point' in df else 0
            },
            "anomaly_detection": {
                "total_anomalies": int(df['is_anomaly'].sum()) if 'is_anomaly' in df else 0,
                "anomaly_rate": float(df['is_anomaly'].mean()) if 'is_anomaly' in df else 0,
                "anomaly_types": df['anomaly_type'].value_counts().to_dict() if 'anomaly_type' in df else {}
            },
            "risk_signals": {
                "signal_distribution": self.risk_signals['signal_name'].value_counts().to_dict(),
                "avg_position_size": float(self.risk_signals['position_size'].mean()),
                "avg_confidence": float(self.risk_signals['confidence'].mean()),
                "n_signal_changes": int((self.risk_signals['signal_change'] != 0).sum())
            },
            "case_study_ready": True
        }
        
        print("\n" + "="*50)
        print("📊 CASE STUDY METRICS GENERATED")
        print("="*50)
        print(f"✓ Period: {metrics['data_period']['start']} to {metrics['data_period']['end']}")
        print(f"✓ Total anomalies detected: {metrics['anomaly_detection']['total_anomalies']}")
        print(f"✓ Avg position size: {metrics['risk_signals']['avg_position_size']:.1%}")
        print(f"✓ Signal changes: {metrics['risk_signals']['n_signal_changes']}")
        
        return metrics
    
    def run_full_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete risk orchestration pipeline.
        
        Args:
            df: DataFrame with regime and anomaly columns
        
        Returns:
            (df_with_signals, case_study_metrics)
        """
        print("\n" + "="*60)
        print("🚀 RISK ORCHESTRATOR FULL PIPELINE")
        print("="*60)
        
        # Generate signals
        df_signals = self.generate_signals(df)
        
        # Generate case study metrics
        metrics = self.generate_case_study_metrics(df_signals)
        
        # Export signal log
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        self.export_signal_log(f"{config.REPORTS_DIR}/signal_log.csv")
        
        return df_signals, metrics


# Quick test when run directly
if __name__ == "__main__":
    # Simulate data with regime and anomaly columns
    dates = pd.date_range('2026-01-14', '2026-05-14', freq='D')
    test_df = pd.DataFrame(index=dates)
    test_df['regime_clean'] = np.random.choice([0,1,2,3], size=len(dates), p=[0.2,0.3,0.3,0.2])
    test_df['is_anomaly'] = np.random.choice([0,1], size=len(dates), p=[0.9,0.1])
    test_df['composite_score'] = np.random.uniform(0, 1, size=len(dates))
    test_df['alert_level'] = np.random.choice([0,1,2,3], size=len(dates), p=[0.7,0.15,0.1,0.05])
    test_df['Close'] = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
    test_df['return_1d'] = test_df['Close'].pct_change()
    
    # Run orchestrator
    orchestrator = RiskOrchestrator()
    df_signals, metrics = orchestrator.run_full_pipeline(test_df)
    
    print(f"\n✓ Final DataFrame shape: {df_signals.shape}")
    print(f"✓ Signal columns added: risk_signal, signal_name, position_size, confidence, risk_score")
    print(f"\n📝 Most recent signal:")
    print(orchestrator.get_signal_explanation(-1))