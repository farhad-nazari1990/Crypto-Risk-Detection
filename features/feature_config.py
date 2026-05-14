"""
Feature configuration and feature name definitions.
"""

def get_feature_names():
    """
    Returns the list of all engineered features used for regime and anomaly detection.
    """
    features = [
        # Returns features
        'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_21d',
        
        # Volatility features
        'volatility_7d', 'volatility_14d', 'volatility_21d', 'volatility_30d',
        
        # Momentum features
        'momentum_7d', 'momentum_14d', 'momentum_21d',
        
        # Technical indicators
        'rsi_14',
        'bb_position',  # Position within Bollinger Bands
        'bb_width',     # Bollinger Band width
        
        # Volume features
        'volume_ratio',  # Current volume / MA volume
        'volume_change',
        
        # Price features
        'atr_14',  # Average True Range
        'price_change_pct',
        
        # Higher moments
        'skewness_7d',
        'kurtosis_7d',
    ]
    
    return features


def get_regime_features():
    """
    Returns features specifically used for regime detection.
    Focus on trend, volatility, and momentum.
    """
    return [
        'return_1d', 'return_7d', 'return_14d',
        'volatility_7d', 'volatility_14d', 'volatility_21d',
        'momentum_7d', 'momentum_14d',
        'rsi_14',
        'bb_position',
        'volume_ratio',
    ]


def get_anomaly_features():
    """
    Returns features specifically used for anomaly detection.
    Focus on extreme movements and microstructure.
    """
    return [
        'return_1d',
        'volatility_7d',
        'volume_ratio',
        'volume_change',
        'atr_14',
        'price_change_pct',
        'bb_position',
        'rsi_14',
    ]
