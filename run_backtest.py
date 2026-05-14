"""
Main execution script for the Crypto Risk Decision System.
FIXED: MultiIndex column handling, dynamic date splitting, data flattening
"""

import logging
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import *
from features.feature_engineering import FeatureEngineer
from features.feature_config import (
    get_regime_features,
    get_anomaly_features
)
from models.regime_hmm import RegimeDetectorHMM
from models.anomaly_isolation_forest import AnomalyDetectorIsolationForest
from models.anomaly_mad import AnomalyDetectorMAD
from models.anomaly_cusum import AnomalyDetectorCUSUM
from models.ensemble_decision import EnsembleDecisionMaker
from backtest.backtest_engine import BacktestEngine
from visualization.charts import create_case_study_chart

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


def flatten_yfinance_data(df):
    """
    Flatten MultiIndex columns from yfinance download.
    
    Args:
        df: DataFrame with MultiIndex columns from yfinance
        
    Returns:
        DataFrame with standard columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the first level (price type) and drop the ticker level
        df.columns = df.columns.get_level_values(0)
    
    # Ensure we have the expected columns
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    # Reorder columns if all are present
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df[available_cols]
    
    logger.info(f"Flattened data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def download_data():
    """
    Download BTC-USD data from Yahoo Finance.
    FIXED: Handles MultiIndex columns and returns flattened DataFrame
    """
    
    logger.info("Downloading BTC-USD data...")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DOWNLOAD_DAYS)
        
        data = yf.download(
            TICKER,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if data.empty:
            raise ValueError("No data downloaded.")
        
        # FIX: Flatten MultiIndex columns
        data = flatten_yfinance_data(data)
        
        logger.info(f"Downloaded {len(data)} rows from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Date range: {(data.index[-1] - data.index[0]).days} days")
        
        # Save raw data
        data.to_csv(DATA_DIR / "btc_usd_data.csv")
        
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def split_train_test(df, train_ratio=0.70):
    """
    Split data into training and testing sets.
    
    Args:
        df: Full DataFrame
        train_ratio: Proportion of data for training
        
    Returns:
        Tuple of (train_df, test_df, split_date)
    """
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    split_date = test_df.index[0]
    
    logger.info(f"Train period: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} days)")
    logger.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} days)")
    
    return train_df, test_df, split_date


def run_pipeline():
    """
    Execute complete risk decision system pipeline.
    FIXED: Dynamic date splitting, proper data handling
    """
    
    logger.info("=" * 60)
    logger.info("STARTING CRYPTO RISK DECISION SYSTEM")
    logger.info("=" * 60)
    
    # Step 1: Download data
    raw_data = download_data()
    
    # Step 2: Feature engineering on full dataset
    feature_engineer = FeatureEngineer(FEATURE_CONFIG)
    features_df = feature_engineer.engineer_features(raw_data)
    
    logger.info(f"Features engineered. Shape: {features_df.shape}")
    logger.info(f"Feature columns: {features_df.columns.tolist()}")
    
    # Step 3: Split into train/test
    if USE_DYNAMIC_SPLIT:
        train_features, test_features, split_date = split_train_test(
            features_df, 
            TRAIN_SPLIT_RATIO
        )
    else:
        # Use hardcoded dates (fallback)
        train_features = features_df.loc[:BACKTEST_START]
        test_features = features_df.loc[BACKTEST_START:BACKTEST_END]
        split_date = test_features.index[0]
    
    # Step 4: Regime detection - train on training data
    regime_detector = RegimeDetectorHMM(
        HMM_CONFIG,
        REGIME_LABELS
    )
    
    regime_features = get_regime_features()
    
    logger.info("Training regime detection model...")
    regime_detector.fit(train_features, regime_features)
    
    # Predict on full dataset for visualization
    regimes_full = regime_detector.predict(features_df)
    
    # Predict on test set for backtest
    regimes_test = regime_detector.predict(test_features)
    
    # Save regime predictions
    regime_predictions = pd.DataFrame({
        'Date': features_df.index,
        'regime': regimes_full
    })
    
    regime_predictions.to_csv(
        OUTPUT_DIR / "regime_predictions.csv",
        index=False
    )
    
    # Step 5: Anomaly detection - train on training data
    anomaly_features = get_anomaly_features()
    
    # Isolation Forest
    iso_detector = AnomalyDetectorIsolationForest(
        ANOMALY_CONFIG['isolation_forest']
    )
    
    logger.info("Training Isolation Forest anomaly detector...")
    iso_detector.fit(train_features, anomaly_features)
    
    iso_anomalies_full = iso_detector.predict(features_df)
    iso_anomalies_test = iso_detector.predict(test_features)
    
    # MAD anomalies
    mad_detector = AnomalyDetectorMAD(
        ANOMALY_CONFIG['mad']
    )
    
    mad_anomalies_full = mad_detector.detect(features_df)
    mad_anomalies_test = mad_detector.detect(test_features)
    
    # CUSUM anomalies
    cusum_detector = AnomalyDetectorCUSUM(
        ANOMALY_CONFIG['cusum']
    )
    
    cusum_anomalies_full = cusum_detector.detect(features_df)
    cusum_anomalies_test = cusum_detector.detect(test_features)
    
    # Ensemble anomaly decision (voting: 2 out of 3)
    combined_anomalies_full = (
        iso_anomalies_full.astype(int) +
        mad_anomalies_full.astype(int) +
        cusum_anomalies_full.astype(int)
    ) >= 2
    
    combined_anomalies_test = (
        iso_anomalies_test.astype(int) +
        mad_anomalies_test.astype(int) +
        cusum_anomalies_test.astype(int)
    ) >= 2
    
    anomaly_predictions = pd.DataFrame({
        'Date': features_df.index,
        'isolation_forest': iso_anomalies_full,
        'mad': mad_anomalies_full,
        'cusum': cusum_anomalies_full,
        'anomaly_flag': combined_anomalies_full
    })
    
    anomaly_predictions.to_csv(
        OUTPUT_DIR / "anomaly_predictions.csv",
        index=False
    )
    
    # Step 6: Generate decisions for full dataset (for visualization)
    decision_maker = EnsembleDecisionMaker(POSITION_SIZES)
    
    decisions_full, positions_full = decision_maker.make_decisions(
        regimes_full,
        combined_anomalies_full
    )
    
    decisions_test, positions_test = decision_maker.make_decisions(
        regimes_test,
        combined_anomalies_test
    )
    
    # Create combined DataFrame for full dataset
    combined_df = features_df.copy()
    combined_df['regime'] = regimes_full
    combined_df['anomaly_flag'] = combined_anomalies_full
    combined_df['decision'] = decisions_full
    combined_df['position'] = positions_full
    combined_df.reset_index(inplace=True)
    
    combined_df.to_csv(
        OUTPUT_DIR / "combined_signals.csv",
        index=False
    )
    
    # Step 7: Backtesting on test period only
    backtest_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST
    )
    
    logger.info("Running backtest on test period...")
    
    buy_hold_results = backtest_engine.run_buy_and_hold(test_features)
    
    model_results = backtest_engine.run_model_strategy(
        test_features,
        positions_test
    )
    
    comparison_table = backtest_engine.compare_strategies(
        buy_hold_results,
        model_results
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS (Test Period Only)")
    logger.info("=" * 60)
    logger.info(f"\n{comparison_table.to_string(index=False)}")
    
    comparison_table.to_csv(
        OUTPUT_DIR / "strategy_comparison.csv",
        index=False
    )
    
    # Step 8: Create chart (full dataset for context)
    chart_df = combined_df.copy()
    chart_df.set_index('Date', inplace=True)
    
    create_case_study_chart(
        chart_df,
        OUTPUT_DIR / "case_study_chart.html"
    )
    
    # Step 9: Generate golden sentence
    bh_dd = buy_hold_results['metrics']['max_drawdown_pct']
    model_dd = model_results['metrics']['max_drawdown_pct']
    
    bh_return = buy_hold_results['metrics']['total_return_pct']
    model_return = model_results['metrics']['total_return_pct']
    
    dd_reduction = bh_dd - model_dd
    
    if bh_return != 0:
        preserved_return = (model_return / bh_return * 100)
    else:
        preserved_return = 0
    
    golden_sentence = (
        f"During the backtest period ({test_features.index[0].strftime('%Y-%m-%d')} to "
        f"{test_features.index[-1].strftime('%Y-%m-%d')}), our combined regime + anomaly "
        f"system reduced maximum drawdown by {dd_reduction:.2f} percentage points "
        f"(from {bh_dd:.2f}% to {model_dd:.2f}%) while achieving "
        f"{model_return:.2f}% total return vs {bh_return:.2f}% for buy-and-hold."
    )
    
    # Step 10: Save metrics
    metrics_output = {
        'backtest_period': {
            'start': test_features.index[0].strftime('%Y-%m-%d'),
            'end': test_features.index[-1].strftime('%Y-%m-%d'),
            'days': len(test_features)
        },
        'buy_hold': buy_hold_results['metrics'],
        'model_strategy': model_results['metrics'],
        'golden_sentence': golden_sentence
    }
    
    with open(
        OUTPUT_DIR / "case_study_metrics.json",
        'w'
    ) as f:
        json.dump(metrics_output, f, indent=4)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    logger.info(f"\nGolden Sentence:\n{golden_sentence}")
    
    print("\n")
    print("=" * 60)
    print("CASE STUDY RESULTS")
    print("=" * 60)
    print(comparison_table.to_string(index=False))
    print("\n")
    print(golden_sentence)
    print("\n" + "=" * 60)
    print("OUTPUTS SAVED TO:")
    print("=" * 60)
    print(f"📊 Chart: {OUTPUT_DIR / 'case_study_chart.html'}")
    print(f"📈 Signals: {OUTPUT_DIR / 'combined_signals.csv'}")
    print(f"📉 Comparison: {OUTPUT_DIR / 'strategy_comparison.csv'}")
    print(f"📋 Metrics: {OUTPUT_DIR / 'case_study_metrics.json'}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
