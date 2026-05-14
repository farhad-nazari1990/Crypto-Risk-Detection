"""
Backtesting engine for strategy evaluation.
"""

import numpy as np
import pandas as pd
import logging

from backtest.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run_buy_and_hold(self, price_data):
        """
        Run buy-and-hold benchmark strategy.
        
        Args:
            price_data: DataFrame with Close prices
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running buy-and-hold backtest...")
        
        returns = price_data['Close'].pct_change().fillna(0)
        
        equity_curve = self.initial_capital * (1 + returns).cumprod()
        
        metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            daily_returns=returns
        )
        
        results = {
            'equity_curve': equity_curve,
            'daily_returns': returns,
            'metrics': metrics,
        }
        
        return results
    
    def run_model_strategy(self, price_data, positions):
        """
        Run model-based strategy backtest.
        
        Args:
            price_data: DataFrame with Close prices
            positions: Array of position sizes
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running model-based strategy backtest...")
        
        # Calculate asset returns
        asset_returns = price_data['Close'].pct_change().fillna(0)
        
        # Strategy returns = position * asset return
        strategy_returns = positions * asset_returns
        
        # Apply transaction costs
        position_changes = pd.Series(positions).diff().abs().fillna(0)
        transaction_costs = position_changes * self.transaction_cost
        
        strategy_returns_after_costs = strategy_returns - transaction_costs
        
        # Calculate equity curve
        equity_curve = self.initial_capital * (1 + strategy_returns_after_costs).cumprod()
        
        metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            daily_returns=strategy_returns_after_costs,
            positions=pd.Series(positions)
        )
        
        results = {
            'equity_curve': equity_curve,
            'daily_returns': strategy_returns_after_costs,
            'metrics': metrics,
            'positions': positions,
        }
        
        return results
    
    def compare_strategies(self, buy_hold_results, model_results):
        """
        Compare buy-and-hold vs model strategy.
        
        Args:
            buy_hold_results: Buy-and-hold backtest results
            model_results: Model strategy backtest results
            
        Returns:
            Comparison DataFrame
        """
        comparison = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Max Drawdown (%)',
                'Sharpe Ratio',
                'Winning Days (%)',
                'Number of Trades'
            ],
            'Buy & Hold': [
                buy_hold_results['metrics']['total_return_pct'],
                buy_hold_results['metrics']['max_drawdown_pct'],
                buy_hold_results['metrics']['sharpe_ratio'],
                buy_hold_results['metrics']['winning_days_pct'],
                1
            ],
            'Model Strategy': [
                model_results['metrics']['total_return_pct'],
                model_results['metrics']['max_drawdown_pct'],
                model_results['metrics']['sharpe_ratio'],
                model_results['metrics']['winning_days_pct'],
                model_results['metrics']['number_of_trades']
            ]
        })
        
        return comparison
