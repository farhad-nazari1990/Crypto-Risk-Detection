"""
Main Orchestration Script
Runs the complete Crypto Risk Decision System pipeline:
1. Load data and engineer features
2. Detect market regimes (HMM + GMM + Change Points)
3. Detect anomalies (4 methods + composite fusion)
4. Orchestrate risk signals (regime + anomaly → trading decisions)
5. Backtest strategy performance
6. Generate visualizations and case study report

This is the single entry point for the entire project.
"""

import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_loader import DataLoader
from src.regime_detection import RegimeDetector
from src.anomaly_detection import AnomalyDetector
from src.risk_orchestrator import RiskOrchestrator
from src.backtest import BacktestEngine
from src.visualize import Visualizer

import config


def setup_directories():
    """Create necessary output directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print("✓ Output directories created")


def save_metrics(metrics: dict, case_study_metrics: dict, filename: str = "case_study_metrics.json"):
    """Save metrics to JSON file for easy sharing."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "data_period": metrics.get('data_period', {}),
        "backtest_metrics": {k: v for k, v in metrics.items() if k != 'data_period'},
        "case_study_metrics": case_study_metrics
    }
    
    filepath = f"{config.REPORTS_DIR}/{filename}"
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"✓ Metrics saved to {filepath}")


def print_final_summary(df: pd.DataFrame, backtest_results: pd.DataFrame, metrics: dict):
    """Print a beautiful final summary for the console."""
    print("\n" + "="*70)
    print("🎯 CRYPTO RISK DECISION SYSTEM - EXECUTION COMPLETE")
    print("="*70)
    
    print(f"\n📅 Analysis Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"📊 Total Days Analyzed: {len(df)}")
    
    print("\n" + "-"*50)
    print("KEY METRICS FOR YOUR FREELANCE CASE STUDY")
    print("-"*50)
    
    # Regime metrics
    if 'regime_name' in df:
        regime_counts = df['regime_name'].value_counts()
        print(f"\n📈 Market Regimes Detected:")
        for regime, count in regime_counts.items():
            print(f"   • {regime}: {count} days ({count/len(df)*100:.1f}%)")
    
    # Anomaly metrics
    if 'is_anomaly' in df:
        anomaly_count = df['is_anomaly'].sum()
        print(f"\n⚠️  Anomalies Detected: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
    
    # Signal metrics
    if 'signal_name' in df:
        signal_counts = df['signal_name'].value_counts()
        print(f"\n🎯 Risk Signal Distribution:")
        for signal, count in signal_counts.items():
            print(f"   • {signal}: {count} days ({count/len(df)*100:.1f}%)")
    
    # Backtest metrics (the most important for selling)
    print(f"\n💰 BACKTEST RESULTS (The Numbers That Sell):")
    print(f"   • Strategy Return: {metrics.get('strategy_total_return', 0):.2f}%")
    print(f"   • Benchmark Return: {metrics.get('benchmark_total_return', 0):.2f}%")
    print(f"   • Excess Return: {metrics.get('excess_return', 0):.2f}%")
    print(f"\n   • Strategy Max Drawdown: {metrics.get('strategy_max_drawdown', 0):.2f}%")
    print(f"   • Benchmark Max Drawdown: {metrics.get('benchmark_max_drawdown', 0):.2f}%")
    print(f"   🔥 DRAWDOWN REDUCTION: {metrics.get('drawdown_reduction', 0):.2f}%")
    print(f"\n   • Strategy Sharpe Ratio: {metrics.get('strategy_sharpe', 0):.2f}")
    print(f"   • Benchmark Sharpe Ratio: {metrics.get('benchmark_sharpe', 0):.2f}")
    
    # Transaction costs
    print(f"\n💸 Transaction Costs: ${metrics.get('total_transaction_costs', 0):.2f}")
    
    print("\n" + "="*70)
    print("✅ NEXT STEPS FOR YOUR FREELANCE SERVICE:")
    print("="*70)
    print("1. Open outputs/charts/combined_risk_dashboard.html in your browser")
    print("2. Review outputs/reports/case_study_report.txt")
    print("3. Use the numbers above in your outreach messages")
    print("4. Show the dashboard in your first client call")
    print("\n🚀 Ready to sell your $1200 POC!")


def main():
    """
    Main execution function.
    Runs the complete pipeline end-to-end.
    """
    print("\n" + "="*70)
    print("🚀 CRYPTO RISK DECISION SYSTEM")
    print("   Quant ML Engineer - Freelance Case Study")
    print("="*70)
    
    # Step 0: Setup
    setup_directories()
    
    # Step 1: Load and prepare data
    print("\n" + "📁 STEP 1: Loading Data".center(70, "-"))
    loader = DataLoader(data_path=config.DATA_PATH)
    
    try:
        full_df, train_df, test_df, train_features, test_features = loader.run_full_pipeline()
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {config.DATA_PATH}")
        print("Please ensure 'btc_usd_standard.csv' exists in the data/ directory")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Regime Detection
    print("\n" + "🔬 STEP 2: Market Regime Detection".center(70, "-"))
    regime_detector = RegimeDetector()
    full_df, regime_stats, regime_comparison = regime_detector.run_full_pipeline(full_df, train_features)
    
    # Step 3: Anomaly Detection
    print("\n" + "⚠️ STEP 3: Anomaly Detection".center(70, "-"))
    anomaly_detector = AnomalyDetector()
    
    # Fit Isolation Forest on training data only (no leakage)
    anomaly_detector.fit_isolation_forest(train_features)
    
    # Run anomaly detection on full dataset
    full_df, anomaly_alerts = anomaly_detector.run_full_pipeline(full_df, train_features)
    
    # Step 4: Risk Orchestration
    print("\n" + "🎯 STEP 4: Risk Signal Orchestration".center(70, "-"))
    orchestrator = RiskOrchestrator()
    full_df, orchestration_metrics = orchestrator.run_full_pipeline(full_df)
    
    # Step 5: Backtest
    print("\n" + "📊 STEP 5: Backtest Engine".center(70, "-"))
    backtest_engine = BacktestEngine()
    
    # Ensure we have position_size column
    if 'position_size' not in full_df.columns:
        print("❌ position_size column missing from dataframe")
        return
    
    backtest_results, backtest_metrics = backtest_engine.run_backtest(full_df)
    
    # Step 6: Generate Case Study Metrics
    print("\n" + "📋 STEP 6: Case Study Metrics".center(70, "-"))
    case_study_metrics = backtest_engine.generate_case_study_metrics()
    
    # Step 7: Visualization
    print("\n" + "📈 STEP 7: Visualization & Reporting".center(70, "-"))
    visualizer = Visualizer()
    visualizer.run_full_visualization(
        df=full_df,
        backtest_results=backtest_results,
        metrics=backtest_metrics,
        case_study_metrics=case_study_metrics
    )
    
    # Step 8: Save all metrics
    print("\n" + "💾 STEP 8: Saving Outputs".center(70, "-"))
    
    # Add data period to metrics
    backtest_metrics['data_period'] = {
        'start': full_df.index[0].strftime('%Y-%m-%d'),
        'end': full_df.index[-1].strftime('%Y-%m-%d'),
        'total_days': len(full_df)
    }
    
    save_metrics(backtest_metrics, case_study_metrics)
    
    # Export signal log and trades
    orchestrator.export_signal_log(f"{config.REPORTS_DIR}/signal_log.csv")
    backtest_engine.export_results(f"{config.REPORTS_DIR}/backtest_results.csv")
    backtest_engine.export_trades(f"{config.REPORTS_DIR}/trades.csv")
    
    # Save trained models (optional, for future use)
    regime_detector.save_models(f"{config.MODELS_DIR}/regime_models.joblib")
    anomaly_detector.save_models(f"{config.MODELS_DIR}/anomaly_models.joblib")
    
    # Step 9: Final Summary
    print_final_summary(full_df, backtest_results, backtest_metrics)
    
    print("\n" + "="*70)
    print("🎉 PROJECT COMPLETE!")
    print("="*70)
    print(f"\n📁 Outputs saved to:")
    print(f"   • Charts: {config.CHARTS_DIR}")
    print(f"   • Reports: {config.REPORTS_DIR}")
    print(f"   • Models: {config.MODELS_DIR}")
    print("\n🔑 Key file for your freelance pitch:")
    print(f"   • Interactive Dashboard: {config.CHARTS_DIR}/combined_risk_dashboard.html")
    print(f"   • Case Study Report: {config.REPORTS_DIR}/case_study_report.txt")
    print(f"   • Metrics JSON: {config.REPORTS_DIR}/case_study_metrics.json")
    
    return full_df, backtest_results, backtest_metrics, case_study_metrics


if __name__ == "__main__":
    # Run the complete pipeline
    df, results, metrics, case_study = main()
    
    # Optional: Interactive mode - show latest signal
    print("\n" + "="*50)
    print("📱 INTERACTIVE MODE")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nType 'signal' for latest recommendation, 'explain' for details, or 'exit' to quit: ").strip().lower()
            
            if user_input == 'exit':
                print("👋 Goodbye! Ready to sell your service?")
                break
            elif user_input == 'signal':
                if 'df' in locals() and df is not None:
                    from src.risk_orchestrator import RiskOrchestrator
                    orch = RiskOrchestrator()
                    # Need to regenerate signals on the full df
                    if 'regime_clean' in df.columns and 'is_anomaly' in df.columns:
                        temp_df = orch.generate_signals(df)
                        current_signal = orch.get_current_signal()
                        print(f"\n📡 CURRENT SIGNAL:")
                        print(f"   Date: {current_signal.get('date')}")
                        print(f"   Signal: {current_signal.get('signal')}")
                        print(f"   Position: {current_signal.get('position_size', 0)*100:.0f}%")
                        print(f"   Confidence: {current_signal.get('confidence', 0)*100:.0f}%")
                    else:
                        print("❌ Signal data not available. Run main() first.")
                else:
                    print("❌ Run main() first to generate signals")
            
            elif user_input == 'explain':
                if 'df' in locals() and df is not None:
                    from src.risk_orchestrator import RiskOrchestrator
                    orch = RiskOrchestrator()
                    if 'regime_clean' in df.columns and 'is_anomaly' in df.columns:
                        temp_df = orch.generate_signals(df)
                        explanation = orch.get_signal_explanation(-1)
                        print(explanation)
                    else:
                        print("❌ Signal data not available. Run main() first.")
                else:
                    print("❌ Run main() first to generate signals")
            
            else:
                print("Commands: 'signal', 'explain', 'exit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")