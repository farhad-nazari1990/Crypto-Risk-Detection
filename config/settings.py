"""
Configuration settings for the Crypto Risk Decision System.
All parameters, paths, and constants are centralized here.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "saved_models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data settings
TICKER = "BTC-USD"
START_DATE = "2024-12-15"  # Extended history for feature calculation
BACKTEST_START = "2025-02-15"
BACKTEST_END = "2025-04-15"

# If future dates, use most recent 60 days
USE_RECENT_DATA = True  # Set to True to use most recent 60 days instead
RECENT_DAYS = 90  # Download 90 days to have buffer for feature calculation

# Feature engineering settings
FEATURE_CONFIG = {
    'returns_windows': [1, 3, 7, 14, 21],
    'volatility_windows': [7, 14, 21, 30],
    'momentum_windows': [7, 14, 21],
    'rsi_period': 14,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'volume_ma_period': 20,
    'atr_period': 14,
}

# HMM settings for regime detection
HMM_CONFIG = {
    'n_components': 4,  # 4 regimes
    'covariance_type': 'full',
    'n_iter': 1000,
    'random_state': 42,
    'algorithm': 'viterbi',
}

# Regime labels mapping
REGIME_LABELS = {
    0: "Bull Rally",
    1: "Stable Growth",
    2: "Consolidation",
    3: "Crash/Panic"
}

# Regime colors for visualization
REGIME_COLORS = {
    "Bull Rally": "rgba(0, 255, 0, 0.2)",
    "Stable Growth": "rgba(144, 238, 144, 0.2)",
    "Consolidation": "rgba(255, 255, 0, 0.2)",
    "Crash/Panic": "rgba(255, 0, 0, 0.2)"
}

# Anomaly detection settings
ANOMALY_CONFIG = {
    'isolation_forest': {
        'contamination': 0.05,
        'n_estimators': 100,
        'max_samples': 256,
        'random_state': 42,
    },
    'mad': {
        'window': 20,
        'threshold': 3.5,
    },
    'bollinger': {
        'window': 20,
        'num_std': 2.5,
    },
    'cusum': {
        'threshold': 3.0,
        'drift': 0.5,
    }
}

# Decision rule settings
POSITION_SIZES = {
    "EXIT": 0.0,
    "REDUCE 50%": 0.5,
    "HOLD 30%": 0.3,
    "HOLD 50%": 0.5,
    "HOLD 80%": 0.8,
    "HOLD 100%": 1.0,
}

# Backtest settings
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1% per trade

# Risk-free rate for Sharpe ratio (annualized)
RISK_FREE_RATE = 0.0

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
