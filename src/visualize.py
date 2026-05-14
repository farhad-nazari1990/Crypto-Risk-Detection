"""
Visualization Module
Creates all charts and dashboards for the case study.
Includes regime timeline, anomaly detection, combined risk dashboard, and backtest performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Visualizer:
    """
    Visualization engine for all project outputs.
    Generates static (matplotlib) and interactive (plotly) charts.
    """
    
    def __init__(self, output_dir: str = config.CHARTS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color maps for regimes
        self.regime_colors = {
            'Bull Rally': '#2ecc71',      # Green
            'Stable Growth': '#3498db',   # Blue
            'Consolidation': '#f39c12',   # Orange
            'Crash/Panic': '#e74c3c'      # Red
        }
        
        # Signal colors
        self.signal_colors = {
            'STRONG_BUY': '#27ae60',      # Dark green
            'BUY': '#2ecc71',              # Light green
            'HOLD': '#f39c12',             # Orange
            'REDUCE': '#e67e22',           # Dark orange
            'EXIT': '#e74c3c'              # Red
        }
        
    def plot_regime_timeline(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot market regimes over time with price overlay.
        
        Args:
            df: DataFrame with 'Close', 'regime_name', 'is_change_point' columns
            save: Whether to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Top subplot: Price with regime background colors
        ax1.plot(df.index, df['Close'], color='black', linewidth=1.5, label='BTC-USD Price', alpha=0.8)
        
        # Add regime background colors
        unique_regimes = df['regime_name'].unique()
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
            mask = df['regime_name'] == regime
            if mask.any():
                # Get contiguous blocks
                regime_indices = df.index[mask]
                if len(regime_indices) > 0:
                    # Find contiguous segments
                    gaps = np.where(np.diff(regime_indices) > pd.Timedelta(days=2))[0] + 1
                    segments = np.split(regime_indices, gaps)
                    
                    for seg in segments:
                        if len(seg) > 0:
                            ax1.axvspan(seg[0], seg[-1], alpha=0.3, 
                                       color=self.regime_colors.get(regime, 'gray'), 
                                       label=regime if regime not in [l.get_label() for l in ax1.patches] else "")
        
        # Add change points as vertical lines
        if 'is_change_point' in df.columns:
            change_points = df[df['is_change_point'] == 1].index
            for cp in change_points:
                ax1.axvline(cp, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_title('BTC-USD Price with Market Regimes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Regime distribution over time
        regime_numeric = df['regime_clean'].copy() if 'regime_clean' in df else pd.Series(index=df.index, data=0)
        ax2.fill_between(df.index, regime_numeric, 0, 
                         color='steelblue', alpha=0.5, step='mid')
        ax2.set_title('Regime State (0=Bull, 1=Stable, 2=Consolidation, 3=Crash)', fontsize=12)
        ax2.set_ylabel('Regime', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(-0.5, 3.5)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['Bull', 'Stable', 'Consol.', 'Crash'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/regime_timeline.png", dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/regime_timeline.png")
        
        return fig
    
    def plot_anomaly_detection(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            df: DataFrame with 'Close', 'is_anomaly', 'composite_score', 'anomaly_type'
            save: Whether to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Subplot 1: Price with anomaly markers
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], color='black', linewidth=1.5, alpha=0.7)
        
        # Mark anomalies
        anomalies = df[df['is_anomaly'] == 1]
        if len(anomalies) > 0:
            ax1.scatter(anomalies.index, anomalies['Close'], 
                       color='red', s=50, zorder=5, label='Anomaly Detected', alpha=0.8)
        
        ax1.set_title('Price with Anomaly Detection', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Composite anomaly score
        ax2 = axes[1]
        ax2.fill_between(df.index, df['composite_score'], 0, 
                         color='coral', alpha=0.5, label='Composite Score')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold')
        ax2.set_title('Anomaly Composite Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score (0-1)', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Individual method contributions (stacked)
        ax3 = axes[2]
        if all(col in df.columns for col in ['iforest_score_norm', 'mad_score_norm', 'bollinger_score_norm', 'cusum_score_norm']):
            ax3.stackplot(df.index, 
                         df['iforest_score_norm'].fillna(0),
                         df['mad_score_norm'].fillna(0),
                         df['bollinger_score_norm'].fillna(0),
                         df['cusum_score_norm'].fillna(0),
                         labels=['Isolation Forest', 'Rolling MAD', 'Bollinger Z', 'CUSUM'],
                         colors=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                         alpha=0.7)
            ax3.set_title('Anomaly Method Contributions', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Score Contribution', fontsize=12)
            ax3.legend(loc='upper left', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Method scores not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/anomaly_detection.png", dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/anomaly_detection.png")
        
        return fig
    
    def plot_combined_risk_dashboard(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create interactive Plotly dashboard combining regime, anomaly, and signals.
        This is the main dashboard for the case study.
        
        Args:
            df: DataFrame with all required columns
            save: Whether to save as HTML
        
        Returns:
            Plotly figure
        """
        # Create subplots with 4 rows
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price & Market Regimes', 
                           'Risk Signal & Position Size',
                           'Anomaly Composite Score',
                           'Signal Confidence'),
            row_heights=[0.35, 0.25, 0.2, 0.2]
        )
        
        # Row 1: Price with regime colors
        # Add candlestick or line
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'],
                      mode='lines', name='BTC Price',
                      line=dict(color='black', width=1.5)),
            row=1, col=1
        )
        
        # Add regime background using shapes (simplified - add as separate traces with fill)
        unique_regimes = df['regime_name'].unique()
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
            regime_df = df[df['regime_name'] == regime]
            if len(regime_df) > 0:
                fig.add_trace(
                    go.Scatter(x=regime_df.index, y=[df['Close'].max()] * len(regime_df),
                              fill='toself', name=f'Regime: {regime}',
                              fillcolor=self.regime_colors.get(regime, 'gray'),
                              opacity=0.3, line=dict(width=0),
                              showlegend=True),
                    row=1, col=1
                )
        
        # Row 2: Position size (as area chart)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['position_size'],
                      mode='lines', name='Position Size',
                      fill='tozeroy', line=dict(color='#2ecc71', width=2),
                      hovertemplate='Date: %{x}<br>Position: %{y:.0%}<extra></extra>'),
            row=2, col=1
        )
        
        # Add horizontal lines for reference
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="50%", row=2, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                     annotation_text="80%", row=2, col=1)
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                     annotation_text="25%", row=2, col=1)
        
        # Row 3: Anomaly composite score
        fig.add_trace(
            go.Scatter(x=df.index, y=df['composite_score'],
                      mode='lines', name='Anomaly Score',
                      fill='tozeroy', line=dict(color='#e74c3c', width=2),
                      hovertemplate='Date: %{x}<br>Anomaly Score: %{y:.2f}<extra></extra>'),
            row=3, col=1
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Threshold", row=3, col=1)
        
        # Row 4: Confidence score
        fig.add_trace(
            go.Scatter(x=df.index, y=df['confidence'],
                      mode='lines', name='Signal Confidence',
                      fill='tozeroy', line=dict(color='#3498db', width=2),
                      hovertemplate='Date: %{x}<br>Confidence: %{y:.0%}<extra></extra>'),
            row=4, col=1
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="blue", 
                     annotation_text="High Confidence", row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(text="Crypto Risk Decision Dashboard", x=0.5, font=dict(size=16)),
            height=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            template=config.PLOTLY_TEMPLATE
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Position", row=2, col=1, tickformat='.0%')
        fig.update_yaxes(title_text="Anomaly Score", row=3, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Confidence", row=4, col=1, tickformat='.0%', range=[0, 1])
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        if save:
            fig.write_html(f"{self.output_dir}/combined_risk_dashboard.html")
            print(f"✓ Saved: {self.output_dir}/combined_risk_dashboard.html")
        
        return fig
    
    def plot_backtest_performance(self, results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot backtest performance comparison.
        
        Args:
            results_df: DataFrame with 'strategy_equity', 'benchmark_equity'
            save: Whether to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Equity curves
        ax1 = axes[0, 0]
        ax1.plot(results_df.index, results_df['strategy_equity'], 
                label='Strategy', color='#2ecc71', linewidth=2)
        ax1.plot(results_df.index, results_df['benchmark_equity'], 
                label='Buy & Hold', color='#3498db', linewidth=2, alpha=0.7)
        ax1.set_title('Equity Curve Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Drawdowns
        ax2 = axes[0, 1]
        strategy_dd = self._calculate_drawdown_series(results_df['strategy_equity'])
        benchmark_dd = self._calculate_drawdown_series(results_df['benchmark_equity'])
        
        ax2.fill_between(results_df.index, strategy_dd * 100, 0, 
                        label='Strategy', color='#2ecc71', alpha=0.5)
        ax2.fill_between(results_df.index, benchmark_dd * 100, 0, 
                        label='Buy & Hold', color='#e74c3c', alpha=0.3)
        ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Daily returns distribution
        ax3 = axes[1, 0]
        ax3.hist(results_df['strategy_return'].dropna() * 100, bins=30, 
                alpha=0.5, label='Strategy', color='#2ecc71', edgecolor='black')
        ax3.hist(results_df['market_return'].dropna() * 100, bins=30, 
                alpha=0.5, label='Market', color='#3498db', edgecolor='black')
        ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Daily Return (%)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend()
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Subplot 4: Rolling Sharpe ratio
        ax4 = axes[1, 1]
        rolling_window = min(20, len(results_df) // 3)
        rolling_sharpe_strategy = results_df['strategy_return'].rolling(rolling_window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe_benchmark = results_df['market_return'].rolling(rolling_window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
        )
        
        ax4.plot(results_df.index, rolling_sharpe_strategy, 
                label='Strategy', color='#2ecc71', linewidth=2)
        ax4.plot(results_df.index, rolling_sharpe_benchmark, 
                label='Buy & Hold', color='#3498db', linewidth=2, alpha=0.7)
        ax4.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio', fontsize=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Backtest Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/backtest_performance.png", dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/backtest_performance.png")
        
        return fig
    
    def _calculate_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        cumulative = equity / equity.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def plot_signal_distribution(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot distribution of risk signals.
        
        Args:
            df: DataFrame with 'signal_name' column
            save: Whether to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        signal_counts = df['signal_name'].value_counts()
        colors = [self.signal_colors.get(s, 'gray') for s in signal_counts.index]
        bars = ax1.bar(signal_counts.index, signal_counts.values, color=colors, edgecolor='black')
        ax1.set_title('Signal Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Signal Type', fontsize=10)
        ax1.set_ylabel('Number of Days', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, signal_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(val), ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        colors = [self.signal_colors.get(s, 'gray') for s in signal_counts.index]
        wedges, texts, autotexts = ax2.pie(signal_counts.values, 
                                           labels=signal_counts.index,
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           explode=[0.05] * len(signal_counts))
        ax2.set_title('Signal Distribution (Proportion)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/signal_distribution.png", dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/signal_distribution.png")
        
        return fig
    
    def create_case_study_report(self, df: pd.DataFrame, backtest_results: pd.DataFrame, 
                                 metrics: Dict, case_study_metrics: Dict) -> str:
        """
        Generate text report for the case study (ready to copy into PDF).
        
        Returns:
            Formatted report string
        """
        report = f"""
================================================================================
                    CRYPTO RISK DECISION SYSTEM - CASE STUDY REPORT
================================================================================

📅 PERIOD: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
📊 TOTAL DAYS: {len(df)}

================================================================================
1. MARKET REGIME DETECTION
================================================================================

Detected Regimes:
"""
        
        if 'regime_name' in df:
            regime_counts = df['regime_name'].value_counts()
            for regime, count in regime_counts.items():
                pct = count / len(df) * 100
                report += f"  • {regime}: {count} days ({pct:.1f}%)\n"
        
        report += f"""
Change Points Detected: {df['is_change_point'].sum() if 'is_change_point' in df else 0} structural breaks

================================================================================
2. ANOMALY DETECTION
================================================================================

Total Anomalies: {df['is_anomaly'].sum() if 'is_anomaly' in df else 0}
Anomaly Rate: {df['is_anomaly'].mean() * 100 if 'is_anomaly' in df else 0:.1f}%

Anomaly Types:
"""
        
        if 'anomaly_type' in df:
            anomaly_types = df[df['is_anomaly'] == 1]['anomaly_type'].value_counts()
            for atype, count in anomaly_types.items():
                report += f"  • {atype}: {count} occurrences\n"
        
        report += f"""
================================================================================
3. RISK SIGNALS (DECISION OUTPUT)
================================================================================

Signal Distribution:
"""
        
        if 'signal_name' in df:
            signal_counts = df['signal_name'].value_counts()
            for signal, count in signal_counts.items():
                pct = count / len(df) * 100
                report += f"  • {signal}: {count} days ({pct:.1f}%)\n"
        
        report += f"""
Average Position Size: {df['position_size'].mean() * 100:.1f}%
Average Confidence: {df['confidence'].mean() * 100:.1f}%

================================================================================
4. BACKTEST PERFORMANCE
================================================================================

PERFORMANCE METRICS:
  • Strategy Total Return: {metrics.get('strategy_total_return', 0):.2f}%
  • Benchmark Total Return: {metrics.get('benchmark_total_return', 0):.2f}%
  • Excess Return: {metrics.get('excess_return', 0):.2f}%

RISK METRICS:
  • Strategy Max Drawdown: {metrics.get('strategy_max_drawdown', 0):.2f}%
  • Benchmark Max Drawdown: {metrics.get('benchmark_max_drawdown', 0):.2f}%
  • Drawdown Reduction: {metrics.get('drawdown_reduction', 0):.2f}% ({metrics.get('drawdown_reduction_pct', 0):.1f}% improvement)

RISK-ADJUSTED METRICS:
  • Strategy Sharpe Ratio: {metrics.get('strategy_sharpe', 0):.2f}
  • Benchmark Sharpe Ratio: {metrics.get('benchmark_sharpe', 0):.2f}
  • Strategy Volatility: {metrics.get('strategy_volatility', 0):.2f}%
  • Benchmark Volatility: {metrics.get('benchmark_volatility', 0):.2f}%

================================================================================
5. CASE STUDY SUMMARY (FOR FREELANCE SELLING)
================================================================================

🏆 KEY ACHIEVEMENT:
   {case_study_metrics.get('case_study_headline', 'N/A')}

📈 WHY THIS MATTERS FOR YOUR TRADING TEAM:
   1. Reduced downside risk without sacrificing returns
   2. Early warning system for market regime changes
   3. Data-driven position sizing (no emotional decisions)
   4. Transparent, explainable signals (not a black box)

================================================================================
                     END OF CASE STUDY REPORT
================================================================================
"""
        
        # Save report
        report_path = f"{config.REPORTS_DIR}/case_study_report.txt"
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Case study report saved to {report_path}")
        
        return report
    
    def run_full_visualization(self, df: pd.DataFrame, backtest_results: pd.DataFrame,
                              metrics: Dict, case_study_metrics: Dict):
        """
        Generate all visualizations and report.
        
        Args:
            df: Full DataFrame with all columns
            backtest_results: Results from backtest
            metrics: Backtest metrics
            case_study_metrics: Case study specific metrics
        """
        print("\n" + "="*50)
        print("VISUALIZATION PIPELINE")
        print("="*50)
        
        # Generate all charts
        print("\n📊 Generating charts...")
        self.plot_regime_timeline(df)
        self.plot_anomaly_detection(df)
        self.plot_combined_risk_dashboard(df)
        self.plot_backtest_performance(backtest_results)
        self.plot_signal_distribution(df)
        
        # Generate report
        print("\n📝 Generating case study report...")
        self.create_case_study_report(df, backtest_results, metrics, case_study_metrics)
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
        print(f"📄 Report saved to: {config.REPORTS_DIR}")


# Quick test when run directly
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2026-04-08', '2026-05-14', freq='D')
    test_df = pd.DataFrame(index=dates)
    test_df['Close'] = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
    test_df['regime_name'] = np.random.choice(['Bull Rally', 'Stable Growth', 'Consolidation', 'Crash/Panic'], len(dates))
    test_df['regime_clean'] = test_df['regime_name'].map({'Bull Rally': 0, 'Stable Growth': 1, 'Consolidation': 2, 'Crash/Panic': 3})
    test_df['is_change_point'] = np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
    test_df['is_anomaly'] = np.random.choice([0, 1], len(dates), p=[0.9, 0.1])
    test_df['composite_score'] = np.random.uniform(0, 1, len(dates))
    test_df['anomaly_type'] = np.random.choice(['normal', 'volatility_anomaly', 'volume_spike'], len(dates), p=[0.8, 0.1, 0.1])
    test_df['position_size'] = np.random.uniform(0, 1, len(dates))
    test_df['signal_name'] = np.random.choice(['STRONG_BUY', 'BUY', 'HOLD', 'REDUCE', 'EXIT'], len(dates))
    test_df['confidence'] = np.random.uniform(0.3, 0.9, len(dates))
    
    # Create backtest results
    test_backtest = pd.DataFrame(index=dates)
    test_backtest['strategy_equity'] = 10000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
    test_backtest['benchmark_equity'] = 10000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.015))
    test_backtest['strategy_return'] = test_backtest['strategy_equity'].pct_change()
    test_backtest['market_return'] = test_backtest['benchmark_equity'].pct_change()
    
    test_metrics = {
        'strategy_total_return': 15.5,
        'benchmark_total_return': 8.2,
        'excess_return': 7.3,
        'strategy_max_drawdown': -12.3,
        'benchmark_max_drawdown': -18.7,
        'drawdown_reduction': 6.4,
        'drawdown_reduction_pct': 34.2,
        'strategy_sharpe': 1.45,
        'benchmark_sharpe': 0.89,
        'strategy_volatility': 18.5,
        'benchmark_volatility': 22.3
    }
    
    test_case_study = {'case_study_headline': 'Reduced max drawdown by 6.4% while maintaining positive excess return of 7.3%'}
    
    # Run visualizer
    viz = Visualizer()
    viz.run_full_visualization(test_df, test_backtest, test_metrics, test_case_study)