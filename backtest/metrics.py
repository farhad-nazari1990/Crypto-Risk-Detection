"""
Performance metrics calculation for backtesting.
FIXED: Sharpe Ratio Series comparison bug
"""

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve):
    """
    Calculate total return percentage.
    """
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown percentage.
    """
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    
    return abs(drawdown.min()) * 100


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe ratio.
    FIXED: Convert Series std() to scalar before comparison
    """
    excess_returns = returns - risk_free_rate / 252
    
    # FIX: Convert std() result to float for comparison
    returns_std = float(returns.std())
    
    if returns_std == 0 or np.isnan(returns_std):
        return 0.0
    
    sharpe = np.sqrt(252) * float(excess_returns.mean()) / returns_std
    
    return sharpe


def calculate_winning_days_percentage(returns):
    """
    Calculate percentage of positive return days.
    """
    winning_days = (returns > 0).sum()
    total_days = len(returns)
    
    if total_days == 0:
        return 0.0
    
    return (winning_days / total_days) * 100


def calculate_number_of_trades(positions):
    """
    Calculate number of position changes.
    """
    position_changes = positions.diff().fillna(0)
    trades = (position_changes != 0).sum()
    
    return int(trades)


def calculate_all_metrics(equity_curve, daily_returns, positions=None):
    """
    Calculate all performance metrics.
    """
    metrics = {
        'total_return_pct': round(calculate_total_return(equity_curve), 2),
        'max_drawdown_pct': round(calculate_max_drawdown(equity_curve), 2),
        'sharpe_ratio': round(calculate_sharpe_ratio(daily_returns), 2),
        'winning_days_pct': round(calculate_winning_days_percentage(daily_returns), 2),
    }
    
    if positions is not None:
        metrics['number_of_trades'] = calculate_number_of_trades(positions)
    
    return metrics
