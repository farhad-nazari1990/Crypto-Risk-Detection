"""
Configuration file for Crypto Risk Decision System
All hyperparameters centralized here
"""

from datetime import datetime

# ============================================
# DATA PATHS
# ============================================
DATA_PATH = "data/btc_usd_standard.csv"
OUTPUT_DIR = "outputs"
CHARTS_DIR = f"{OUTPUT_DIR}/charts"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
MODELS_DIR = f"{OUTPUT_DIR}/models"

# ============================================
# DATE RANGES (from actual data)
# ============================================
START_DATE = "2026-01-14"
END_DATE = "2026-05-14"
TRAIN_END_DATE = "2026-04-07"  # 70% of data
TEST_START_DATE = "2026-04-08"  # 30% for backtest
TEST_END_DATE = "2026-05-14"

# ============================================
# FEATURE ENGINEERING
# ============================================
RETURN_WINDOWS = [1, 5, 10, 21]  # daily, weekly, 2-week, monthly
VOLATILITY_WINDOWS = [5, 10, 21]
RSI_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2
VOLUME_MA_WINDOWS = [5, 10, 21]

# ============================================
# REGIME DETECTION - HMM
# ============================================
HMM_N_STATES = 4  # Bull, Stable, Consolidation, Crash/Panic
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 1000
HMM_RANDOM_SEED = 42

# ============================================
# CHANGE POINT DETECTION
# ============================================
CHANGE_POINT_MODEL = "l2"  # L2 cost function
CHANGE_POINT_PENALTY = 10  # Penalty coefficient

# ============================================
# ANOMALY DETECTION
# ============================================
# Isolation Forest
IFOREST_CONTAMINATION = 0.05
IFOREST_N_ESTIMATORS = 100
IFOREST_RANDOM_SEED = 42

# Rolling MAD (Median Absolute Deviation)
ROLLING_MAD_WINDOW = 20
ROLLING_MAD_THRESHOLD = 3.5

# Bollinger Z-score
BOLLINGER_Z_WINDOW = 20
BOLLINGER_Z_THRESHOLD = 2.5

# CUSUM (Cumulative Sum)
CUSUM_THRESHOLD = 1.5  # in std deviations
CUSUM_MIN_STD = 0.5

# Anomaly fusion weights (sum to 1)
ANOMALY_WEIGHTS = {
    "isolation_forest": 0.35,
    "rolling_mad": 0.20,
    "bollinger_z": 0.25,
    "cusum": 0.20
}

# ============================================
# RISK DECISION ORCHESTRATOR
# ============================================
# Mapping from (regime, anomaly_flag) to signal
# Signals: 0=STRONG_BUY, 1=BUY, 2=HOLD, 3=REDUCE, 4=EXIT
# Position sizes: 100%, 80%, 50%, 25%, 0%

SIGNAL_MATRIX = {
    # (regime_idx, anomaly_binary) -> signal_level
    # regime_idx: 0=Bull, 1=Stable, 2=Consolidation, 3=Crash
    # anomaly_binary: 0=no anomaly, 1=anomaly detected
    
    # BULL regime
    (0, 0): 0,  # STRONG_BUY
    (0, 1): 1,  # BUY (reduce slightly)
    
    # STABLE regime
    (1, 0): 1,  # BUY
    (1, 1): 2,  # HOLD
    
    # CONSOLIDATION regime
    (2, 0): 2,  # HOLD
    (2, 1): 3,  # REDUCE
    
    # CRASH regime
    (3, 0): 3,  # REDUCE
    (3, 1): 4,  # EXIT
}

SIGNAL_TO_POSITION = {
    0: 1.0,   # 100% long
    1: 0.8,   # 80% long
    2: 0.5,   # 50% long
    3: 0.25,  # 25% long
    4: 0.0    # 0% (cash)
}

SIGNAL_NAMES = {
    0: "STRONG_BUY",
    1: "BUY",
    2: "HOLD",
    3: "REDUCE",
    4: "EXIT"
}

REGIME_NAMES = {
    0: "Bull Rally",
    1: "Stable Growth",
    2: "Consolidation",
    3: "Crash/Panic"
}

# ============================================
# BACKTEST SETTINGS
# ============================================
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_BPS = 5  # 5 basis points (0.05%) per trade

# ============================================
# VISUALIZATION
# ============================================
FIGURE_DPI = 150
PLOTLY_TEMPLATE = "plotly_dark"  # or "plotly_white"

# ============================================
# REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42