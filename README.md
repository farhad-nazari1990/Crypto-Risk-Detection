# Crypto Risk Decision System

Production-grade crypto market risk intelligence platform combining:

- Hidden Markov Model (HMM) Regime Detection
- Isolation Forest Anomaly Detection
- Rolling MAD Detection
- CUSUM Change Point Detection
- Portfolio Risk Decision Engine
- Interactive Streamlit Dashboard
- Backtesting Framework

---

# Features

## Market Regimes

The system identifies 4 market regimes:

- Bull Rally
- Stable Growth
- Consolidation
- Crash/Panic

## Anomaly Detection

The system detects:

- Volatility spikes
- Volume anomalies
- Market microstructure anomalies
- Regime transitions

## Decision Signals

Outputs actionable portfolio decisions:

- EXIT
- REDUCE 50%
- HOLD 30%
- HOLD 50%
- HOLD 80%
- HOLD 100%

---

# Project Structure
```text
crypto_risk_decision_system/
├── config/
├── data/
├── features/
├── models/
├── backtest/
├── visualization/
├── dashboard/
├── output/
├── run_backtest.py
├── requirements.txt
└── README.md

---

# Installation

## Step 1: Clone or Create Project

bash
mkdir crypto_risk_decision_system
cd crypto_risk_decision_system

## Step 2: Install Dependencies

bash
pip install -r requirements.txt

---

# Run Full Pipeline

bash
python run_backtest.py

Pipeline runtime:
~1–3 minutes

---

# Expected Outputs

Generated automatically inside `output/`:

text
output/
├── regime_predictions.csv
├── anomaly_predictions.csv
├── combined_signals.csv
├── strategy_comparison.csv
├── case_study_chart.html
└── case_study_metrics.json

---

# Launch Dashboard

bash
streamlit run dashboard/streamlit_app.py

Dashboard Features:

- Interactive market regime visualization
- Anomaly markers
- Backtest comparison
- Portfolio decisions
- Date filtering
- Real-time metrics

---

# Example Golden Sentence

text
During the backtest period, our combined regime + anomaly system reduced maximum drawdown from 18.42% to 7.91% while preserving 84.55% of total return.

---

# Case Study Deliverables

This project produces:

## 1. Interactive Market Intelligence Chart

Includes:

- BTC price
- Regime background overlays
- Anomaly markers

## 2. Performance Comparison Table

Comparing:

- Buy & Hold
- Model-Based Strategy

Metrics:

- Total Return
- Max Drawdown
- Sharpe Ratio
- Winning Days
- Number of Trades

## 3. Executive Summary

One-sentence portfolio risk reduction statement.

---

# Technologies Used

- Python 3.10+
- scikit-learn
- hmmlearn
- Plotly
- Streamlit
- yfinance
- pandas
- numpy
- statsmodels

---

# Reproducibility

All random seeds are fixed.

The project is fully rerunnable and idempotent.

---

# Notes

- Data source: Yahoo Finance (`BTC-USD`)
- Default backtest period:
  - Feb 15 2025 → Apr 15 2025
- If future dates unavailable:
  - Automatically falls back to recent BTC data

---

# Author Positioning

This project is designed for:

- Quant portfolio case studies
- Client acquisition
- Risk intelligence demos
- Entry offer ($1200 POC)
- Crypto hedge fund analytics
- AI-powered trading infrastructure

`

---

# How To Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
