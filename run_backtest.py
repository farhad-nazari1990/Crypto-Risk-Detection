"""
Main execution script for the Crypto Risk Decision System.
Runs the entire pipeline end-to-end.
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


def download_data():
    """
    Download BTC-USD data from Yahoo Finance.
    """
    
    logger.info("Downloading BTC-USD data...")
    
    try:
        if USE_RECENT_DATA:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=RECENT_DAYS)
            
            data = yf.download(
                TICKER,
                start=start_date,
                end=end_date,
                progress=False
            )
        else:
            data = yf.download(
                TICKER,
                start=START_DATE,
                end=BACKTEST_END,
                progress=False
            )
        
        if data.empty:
            raise ValueError("No data downloaded.")
        
        data.to_csv(DATA_DIR / "btc_usd_data.csv")
        
        logger.info(f"Downloaded {len(data)} rows.")
        
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def run_pipeline():
    """
    Execute complete risk decision system pipeline.
    """
    
    logger.info("=" * 60)
    logger.info("STARTING CRYPTO RISK DECISION SYSTEM")
    logger.info("=" * 60)
    
    # Step 1: Download data
    raw_data = download_data()
    
    # Step 2: Feature engineering
    feature_engineer = FeatureEngineer(FEATURE_CONFIG)
    features_df = feature_engineer.engineer_features(raw_data)
    
    # Step 3: Regime detection
    regime_detector = RegimeDetectorHMM(
        HMM_CONFIG,
        REGIME_LABELS
    )
    
    regime_features = get_regime_features()
    
    regime_detector.fit(features_df, regime_features)
    
    regimes = regime_detector.predict(features_df)
    
    # Save regime predictions
    regime_predictions = pd.DataFrame({
        'Date': features_df.index,
        'regime': regimes
    })
    
    regime_predictions.to_csv(
        OUTPUT_DIR / "regime_predictions.csv",
        index=False
    )
    
    # Step 4: Anomaly detection
    anomaly_features = get_anomaly_features()
    
    # Isolation Forest
    iso_detector = AnomalyDetectorIsolationForest(
        ANOMALY_CONFIG['isolation_forest']
    )
    
    iso_detector.fit(features_df, anomaly_features)
    
    iso_anomalies = iso_detector.predict(features_df)
    
    # MAD anomalies
    mad_detector = AnomalyDetectorMAD(
        ANOMALY_CONFIG['mad']
    )
    
    mad_anomalies = mad_detector.detect(features_df)
    
    # CUSUM anomalies
    cusum_detector = AnomalyDetectorCUSUM(
        ANOMALY_CONFIG['cusum']
    )
    
    cusum_anomalies = cusum_detector.detect(features_df)
    
    # Ensemble anomaly decision
    combined_anomalies = (
        iso_anomalies.astype(int) +
        mad_anomalies.astype(int) +
        cusum_anomalies.astype(int)
    ) >= 2
    
    anomaly_predictions = pd.DataFrame({
        'Date': features_df.index,
        'isolation_forest': iso_anomalies,
        'mad': mad_anomalies,
        'cusum': cusum_anomalies,
        'anomaly_flag': combined_anomalies
    })
    
    anomaly_predictions.to_csv(
        OUTPUT_DIR / "anomaly_predictions.csv",
        index=False
    )
    
    # Step 5: Generate decisions
    decision_maker = EnsembleDecisionMaker(POSITION_SIZES)
    
    decisions, positions = decision_maker.make_decisions(
        regimes,
        combined_anomalies
    )
    
    combined_df = features_df.copy()
    
    combined_df['regime'] = regimes
    combined_df['anomaly_flag'] = combined_anomalies
    combined_df['decision'] = decisions
    combined_df['position'] = positions
    
    combined_df.reset_index(inplace=True)
    
    combined_df.to_csv(
        OUTPUT_DIR / "combined_signals.csv",
        index=False
    )
    
    # Step 6: Backtesting
    backtest_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST
    )
    
    buy_hold_results = backtest_engine.run_buy_and_hold(combined_df)
    
    model_results = backtest_engine.run_model_strategy(
        combined_df,
        positions
    )
    
    comparison_table = backtest_engine.compare_strategies(
        buy_hold_results,
        model_results
    )
    
    comparison_table.to_csv(
        OUTPUT_DIR / "strategy_comparison.csv",
        index=False
    )
    
    # Step 7: Create chart
    chart_df = combined_df.copy()
    chart_df.set_index('Date', inplace=True)
    
    create_case_study_chart(
        chart_df,
        OUTPUT_DIR / "case_study_chart.html"
    )
    
    # Step 8: Generate golden sentence
    bh_dd = buy_hold_results['metrics']['max_drawdown_pct']
    model_dd = model_results['metrics']['max_drawdown_pct']
    
    bh_return = buy_hold_results['metrics']['total_return_pct']
    model_return = model_results['metrics']['total_return_pct']
    
    preserved_return = (
        model_return / bh_return * 100
        if bh_return != 0 else 0
    )
    
    golden_sentence = (
        f"During the backtest period, our combined regime + anomaly "
        f"system reduced maximum drawdown from {bh_dd:.2f}% "
        f"to {model_dd:.2f}% while preserving "
        f"{preserved_return:.2f}% of total return."
    )
    
    # Step 9: Save metrics
    metrics_output = {
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
    
    logger.info(f"Golden Sentence: {golden_sentence}")
    
    print("\n")
    print("=" * 60)
    print("CASE STUDY RESULTS")
    print("=" * 60)
    print(comparison_table)
    print("\n")
    print(golden_sentence)
    print("\nOutputs saved to:")
    print(f"- {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
