"""
Visualization utilities for case study charts.
"""

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from config.settings import REGIME_COLORS


def create_case_study_chart(df, output_path):
    """
    Create interactive Plotly chart with:
    - Price line
    - Regime background colors
    - Anomaly markers
    
    Args:
        df: DataFrame with Close, regime, anomaly_flag
        output_path: Path to save HTML chart
    """
    
    fig = go.Figure()
    
    # Add regime background bands
    for regime, color in REGIME_COLORS.items():
        regime_mask = df['regime'] == regime
        
        if regime_mask.any():
            regime_dates = df.index[regime_mask]
            
            for date in regime_dates:
                fig.add_vrect(
                    x0=date,
                    x1=date,
                    fillcolor=color,
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='BTC-USD Price',
            line=dict(color='black', width=2)
        )
    )
    
    # Add anomaly markers
    anomaly_df = df[df['anomaly_flag']]
    
    fig.add_trace(
        go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df['Close'],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=10,
                symbol='x'
            )
        )
    )
    
    fig.update_layout(
        title='Crypto Risk Decision System: Regime + Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='BTC Price (USD)',
        template='plotly_white',
        height=700,
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    
    return fig


def create_equity_curve_chart(buy_hold_results, model_results):
    """
    Create equity curve comparison chart.
    """
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            y=buy_hold_results['equity_curve'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            y=model_results['equity_curve'],
            mode='lines',
            name='Model Strategy',
            line=dict(color='green', width=2)
        )
    )
    
    fig.update_layout(
        title='Strategy Performance Comparison',
        xaxis_title='Time',
        yaxis_title='Portfolio Value',
        template='plotly_white'
    )
    
    return fig
