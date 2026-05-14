"""
Streamlit dashboard for the Crypto Risk Decision System.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(
    page_title="Crypto Risk Decision System",
    layout="wide"
)

OUTPUT_DIR = Path(__file__).parent.parent / "output"


@st.cache_data
def load_data():
    """
    Load generated outputs.
    """
    combined = pd.read_csv(
        OUTPUT_DIR / "combined_signals.csv",
        parse_dates=['Date']
    )
    
    metrics = json.load(
        open(OUTPUT_DIR / "case_study_metrics.json")
    )
    
    return combined, metrics


def create_dashboard_chart(df):
    """
    Create interactive dashboard chart.
    """
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='BTC Price',
            line=dict(color='black')
        )
    )
    
    # Anomaly markers
    anomalies = df[df['anomaly_flag']]
    
    fig.add_trace(
        go.Scatter(
            x=anomalies['Date'],
            y=anomalies['Close'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        )
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white'
    )
    
    return fig


def main():
    st.title("Crypto Risk Decision System")
    
    combined_df, metrics = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        combined_df['Date'].min()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        combined_df['Date'].max()
    )
    
    filtered_df = combined_df[
        (combined_df['Date'] >= pd.Timestamp(start_date)) &
        (combined_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    # Current status
    latest = filtered_df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Regime", latest['regime'])
    
    with col2:
        st.metric("Anomaly Detected", str(latest['anomaly_flag']))
    
    with col3:
        st.metric("Decision", latest['decision'])
    
    # Main chart
    st.subheader("Market Regime + Anomaly Detection")
    
    fig = create_dashboard_chart(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    st.subheader("Backtest Performance")
    
    comparison_df = pd.DataFrame({
        'Metric': [
            'Total Return (%)',
            'Max Drawdown (%)',
            'Sharpe Ratio',
            'Winning Days (%)'
        ],
        'Buy & Hold': [
            metrics['buy_hold']['total_return_pct'],
            metrics['buy_hold']['max_drawdown_pct'],
            metrics['buy_hold']['sharpe_ratio'],
            metrics['buy_hold']['winning_days_pct']
        ],
        'Model Strategy': [
            metrics['model_strategy']['total_return_pct'],
            metrics['model_strategy']['max_drawdown_pct'],
            metrics['model_strategy']['sharpe_ratio'],
            metrics['model_strategy']['winning_days_pct']
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Golden sentence
    st.subheader("Case Study Summary")
    st.success(metrics['golden_sentence'])
    
    # Raw data
    st.subheader("Signals Data")
    st.dataframe(filtered_df.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
