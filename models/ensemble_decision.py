"""
Ensemble decision maker combining regime detection and anomaly detection.
Produces actionable trading signals.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EnsembleDecisionMaker:
    """
    Combines regime and anomaly signals to produce trading decisions.
    """
    
    def __init__(self, position_sizes):
        """
        Initialize decision maker.
        
        Args:
            position_sizes: Dictionary mapping decisions to position sizes
        """
        self.position_sizes = position_sizes
        
    def make_decisions(self, regimes, anomalies):
        """
        Generate trading decisions based on regime and anomaly signals.
        
        Args:
            regimes: Array of regime labels
            anomalies: Array of boolean anomaly flags
            
        Returns:
            Tuple of (decisions array, positions array)
        """
        logger.info("Generating trading decisions...")
        
        decisions = []
        positions = []
        
        for regime, anomaly in zip(regimes, anomalies):
            decision = self._decision_rule(regime, anomaly)
            position = self.position_sizes[decision]
            
            decisions.append(decision)
            positions.append(position)
        
        decisions = np.array(decisions)
        positions = np.array(positions)
        
        logger.info(f"Generated {len(decisions)} decisions.")
        logger.info(f"Decision distribution: {pd.Series(decisions).value_counts().to_dict()}")
        
        return decisions, positions
    
    def _decision_rule(self, regime, anomaly_flag):
        """
        Core decision rule logic.
        
        Args:
            regime: Current market regime
            anomaly_flag: Boolean indicating if anomaly detected
            
        Returns:
            Decision string
        """
        if regime == "Crash/Panic":
            return "EXIT"
        elif regime == "Bull Rally" and anomaly_flag:
            return "REDUCE"
        elif regime == "Bull Rally" and not anomaly_flag:
            return "HOLD 100%"
        elif regime == "Stable Growth":
            return "HOLD 80%"
        elif regime == "Consolidation":
            return "HOLD 30%"
        else:
            return "HOLD 50%"