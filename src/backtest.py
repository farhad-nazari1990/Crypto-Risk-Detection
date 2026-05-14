"""
Backtest Module
Simulates trading strategy based on risk signals from orchestrator.
Compares strategy performance vs. buy-and-hold.
Generates key metrics for case study (max drawdown reduction, Sharpe improvement, etc.).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BacktestEngine:
    """
    Backtest engine for risk-based trading strategy.
    
    Strategy: Position sizing based on risk signals
    - STRONG_BUY: 100% long
    - BUY: 80% long
    - HOLD: 50% long
    - REDUCE: 25% long
    - EXIT: 0% (cash)
    """
    
    def __init__(self, 
                 initial_capital: float = config.INITIAL_CAPITAL,
                 transaction_cost_bps: float = config.TRANSACTION_COST_BPS):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10000  # Convert bps to decimal
        
        # Results storage
        self.strategy_equity = None
        self.benchmark_equity = None
        self.trades = None
        self.metrics = None
        
    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run full backtest on the signal dataframe.
        
        Required columns in df:
            - 'Close': Price data
            - 'position_size': Target position size from orchestrator (0-1)
            - 'signal_name': Signal name (for trade logging)
        
        Returns:
            (results_df, metrics_dict)
        """
        print("\n" + "="*50)
        print("BACKTEST ENGINE")
        print("="*50)
        
        df_backtest = df.copy()
        
        # Ensure required columns exist
        required_cols = ['Close', 'position_size']
        missing_cols = [col for col in required_cols if col not in df_backtest.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate daily returns
        df_backtest['market_return'] = df_backtest['Close'].pct_change()
        
        # Track position changes for transaction costs
        df_backtest['position_change'] = df_backtest['position_size'].diff().abs()
        
        # Calculate transaction costs
        df_backtest['transaction_cost'] = df_backtest['position_change'] * self.transaction_cost
        
        # Calculate strategy returns (before costs)
        df_backtest['strategy_return_raw'] = df_backtest['position_size'].shift(1) * df_backtest['market_return']
        
        # Apply transaction costs
        df_backtest['strategy_return'] = df_backtest['strategy_return_raw'] - df_backtest['transaction_cost']
        
        # Calculate equity curves
        df_backtest['strategy_equity'] = self.initial_capital * (1 + df_backtest['strategy_return']).cumprod()
        df_backtest['benchmark_equity'] = self.initial_capital * (1 + df_backtest['market_return']).cumprod()
        
        # Fill first row (no return on day 1)
        df_backtest['strategy_equity'].iloc[0] = self.initial_capital
        df_backtest['benchmark_equity'].iloc[0] = self.initial_capital
        df_backtest['strategy_return'].iloc[0] = 0
        df_backtest['transaction_cost'].iloc[0] = 0
        
        # Log trades
        self.trades = self._log_trades(df_backtest)
        
        # Store equity curves
        self.strategy_equity = df_backtest['strategy_equity']
        self.benchmark_equity = df_backtest['benchmark_equity']
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(df_backtest)
        
        # Store results
        self.results = df_backtest
        
        # Print summary
        self._print_summary()
        
        return df_backtest, self.metrics
    
    def _log_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log individual trades based on position changes.
        
        Returns:
            DataFrame with trade entries and exits
        """
        trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_size = None
        
        for i, (idx, row) in enumerate(df.iterrows()):
            current_size = row['position_size']
            current_price = row['Close']
            
            # Enter trade
            if not in_position and current_size > 0:
                in_position = True
                entry_idx = idx
                entry_price = current_price
                entry_size = current_size
            
            # Exit trade (position goes to 0 or changes significantly)
            elif in_position and (current_size == 0 or current_size != entry_size):
                exit_idx = idx
                exit_price = current_price
                
                trade_return = (exit_price - entry_price) / entry_price
                trade_duration = (exit_idx - entry_idx).days if hasattr(exit_idx, 'days') else 1
                
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': exit_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return * 100,
                    'duration_days': trade_duration,
                    'position_size': entry_size
                })
                
                # Reset for next trade
                if current_size > 0:
                    entry_idx = idx
                    entry_price = current_price
                    entry_size = current_size
                else:
                    in_position = False
        
        # Close any open position at end
        if in_position:
            exit_idx = df.index[-1]
            exit_price = df['Close'].iloc[-1]
            trade_return = (exit_price - entry_price) / entry_price
            trade_duration = (exit_idx - entry_idx).days if hasattr(exit_idx, 'days') else 1
            
            trades.append({
                'entry_date': entry_idx,
                'exit_date': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return * 100,
                'duration_days': trade_duration,
                'position_size': entry_size
            })
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        if len(trades_df) > 0:
            print(f"✓ Logged {len(trades_df)} trades")
        
        return trades_df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive backtest metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        metrics = {}
        
        # Extract equity curves
        strategy_eq = df['strategy_equity']
        benchmark_eq = df['benchmark_equity']
        strategy_returns = df['strategy_return']
        market_returns = df['market_return']
        
        # Total return
        metrics['strategy_total_return'] = (strategy_eq.iloc[-1] / self.initial_capital - 1) * 100
        metrics['benchmark_total_return'] = (benchmark_eq.iloc[-1] / self.initial_capital - 1) * 100
        metrics['excess_return'] = metrics['strategy_total_return'] - metrics['benchmark_total_return']
        
        # Annualized return
        n_days = len(df)
        n_years = n_days / 252
        metrics['strategy_annual_return'] = ((1 + metrics['strategy_total_return']/100) ** (1/n_years) - 1) * 100
        metrics['benchmark_annual_return'] = ((1 + metrics['benchmark_total_return']/100) ** (1/n_years) - 1) * 100
        
        # Volatility (annualized)
        metrics['strategy_volatility'] = strategy_returns.std() * np.sqrt(252) * 100
        metrics['benchmark_volatility'] = market_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 0% risk-free rate for crypto)
        metrics['strategy_sharpe'] = metrics['strategy_annual_return'] / metrics['strategy_volatility'] if metrics['strategy_volatility'] > 0 else 0
        metrics['benchmark_sharpe'] = metrics['benchmark_annual_return'] / metrics['benchmark_volatility'] if metrics['benchmark_volatility'] > 0 else 0
        
        # Maximum Drawdown
        metrics['strategy_max_drawdown'] = self._calculate_max_drawdown(strategy_eq) * 100
        metrics['benchmark_max_drawdown'] = self._calculate_max_drawdown(benchmark_eq) * 100
        metrics['drawdown_reduction'] = metrics['benchmark_max_drawdown'] - metrics['strategy_max_drawdown']
        metrics['drawdown_reduction_pct'] = (metrics['drawdown_reduction'] / metrics['benchmark_max_drawdown'] * 100) if metrics['benchmark_max_drawdown'] > 0 else 0
        
        # Win rate
        positive_days = (strategy_returns > 0).sum()
        total_days = len(strategy_returns[strategy_returns != 0])
        metrics['strategy_win_rate'] = (positive_days / total_days * 100) if total_days > 0 else 0
        
        # Risk-adjusted metrics
        metrics['calmar_ratio'] = metrics['strategy_annual_return'] / abs(metrics['strategy_max_drawdown']) if metrics['strategy_max_drawdown'] != 0 else 0
        
        # Transaction costs
        metrics['total_transaction_costs'] = df['transaction_cost'].sum() * self.initial_capital
        metrics['total_transaction_costs_pct'] = (metrics['total_transaction_costs'] / self.initial_capital) * 100
        
        # Trade statistics
        if self.trades is not None and len(self.trades) > 0:
            metrics['total_trades'] = len(self.trades)
            metrics['avg_trade_return'] = self.trades['return_pct'].mean()
            metrics['winning_trades'] = (self.trades['return_pct'] > 0).sum()
            metrics['losing_trades'] = (self.trades['return_pct'] <= 0).sum()
            metrics['trade_win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
            metrics['avg_winning_trade'] = self.trades[self.trades['return_pct'] > 0]['return_pct'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_losing_trade'] = self.trades[self.trades['return_pct'] <= 0]['return_pct'].mean() if metrics['losing_trades'] > 0 else 0
            metrics['profit_factor'] = abs(self.trades[self.trades['return_pct'] > 0]['return_pct'].sum() / 
                                          self.trades[self.trades['return_pct'] < 0]['return_pct'].sum()) if metrics['losing_trades'] > 0 else 999
        else:
            metrics['total_trades'] = 0
        
        # Additional metrics from signal data
        if 'signal_name' in df.columns:
            metrics['signal_distribution'] = df['signal_name'].value_counts().to_dict()
        
        if 'confidence' in df.columns:
            metrics['avg_confidence'] = df['confidence'].mean()
        
        return metrics
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        cumulative = equity / equity.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _print_summary(self):
        """Print backtest results summary."""
        print("\n" + "-"*50)
        print("📊 BACKTEST RESULTS")
        print("-"*50)
        
        print(f"\n💰 RETURNS:")
        print(f"  Strategy Total Return: {self.metrics['strategy_total_return']:.2f}%")
        print(f"  Benchmark Total Return: {self.metrics['benchmark_total_return']:.2f}%")
        print(f"  Excess Return: {self.metrics['excess_return']:.2f}%")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Strategy Max Drawdown: {self.metrics['strategy_max_drawdown']:.2f}%")
        print(f"  Benchmark Max Drawdown: {self.metrics['benchmark_max_drawdown']:.2f}%")
        print(f"  Drawdown Reduction: {self.metrics['drawdown_reduction']:.2f}% ({self.metrics['drawdown_reduction_pct']:.1f}% improvement)")
        
        print(f"\n📈 RISK-ADJUSTED:")
        print(f"  Strategy Sharpe Ratio: {self.metrics['strategy_sharpe']:.2f}")
        print(f"  Benchmark Sharpe Ratio: {self.metrics['benchmark_sharpe']:.2f}")
        print(f"  Strategy Volatility: {self.metrics['strategy_volatility']:.2f}%")
        print(f"  Benchmark Volatility: {self.metrics['benchmark_volatility']:.2f}%")
        
        print(f"\n🎯 TRADING STATS:")
        print(f"  Total Trades: {self.metrics['total_trades']}")
        if self.metrics['total_trades'] > 0:
            print(f"  Win Rate: {self.metrics['trade_win_rate']:.1f}%")
            print(f"  Profit Factor: {self.metrics['profit_factor']:.2f}")
        
        print(f"\n💸 COSTS:")
        print(f"  Total Transaction Costs: ${self.metrics['total_transaction_costs']:.2f}")
        print(f"  Cost as % of Capital: {self.metrics['total_transaction_costs_pct']:.2f}%")
    
    def generate_case_study_metrics(self) -> Dict:
        """
        Generate the specific metrics needed for the freelance case study.
        This is the key output for selling the service.
        
        Returns:
            Dictionary with case-study-ready metrics
        """
        if self.metrics is None:
            raise ValueError("Run backtest first")
        
        case_study = {
            "performance": {
                "strategy_return": self.metrics['strategy_total_return'],
                "benchmark_return": self.metrics['benchmark_total_return'],
                "excess_return": self.metrics['excess_return'],
                "annualized_strategy_return": self.metrics['strategy_annual_return']
            },
            "risk_reduction": {
                "strategy_max_drawdown": self.metrics['strategy_max_drawdown'],
                "benchmark_max_drawdown": self.metrics['benchmark_max_drawdown'],
                "drawdown_reduction_absolute": self.metrics['drawdown_reduction'],
                "drawdown_reduction_percentage": self.metrics['drawdown_reduction_pct']
            },
            "risk_adjusted": {
                "strategy_sharpe": self.metrics['strategy_sharpe'],
                "benchmark_sharpe": self.metrics['benchmark_sharpe'],
                "sharpe_improvement": self.metrics['strategy_sharpe'] - self.metrics['benchmark_sharpe'],
                "strategy_volatility": self.metrics['strategy_volatility'],
                "benchmark_volatility": self.metrics['benchmark_volatility'],
                "volatility_reduction": self.metrics['benchmark_volatility'] - self.metrics['strategy_volatility']
            },
            "trading": {
                "total_trades": self.metrics['total_trades'],
                "win_rate": self.metrics.get('trade_win_rate', 0),
                "avg_confidence": self.metrics.get('avg_confidence', 0)
            },
            "case_study_headline": f"Reduced max drawdown by {self.metrics['drawdown_reduction']:.1f}% while maintaining positive excess return of {self.metrics['excess_return']:.1f}%"
        }
        
        print("\n" + "="*50)
        print("📋 CASE STUDY METRICS READY")
        print("="*50)
        print(f"\n🏆 KEY SELLING POINT:")
        print(f"   {case_study['case_study_headline']}")
        print(f"\n📊 Detailed Metrics:")
        print(f"   • Sharpe Ratio: {case_study['risk_adjusted']['strategy_sharpe']:.2f} vs {case_study['risk_adjusted']['benchmark_sharpe']:.2f}")
        print(f"   • Max DD: {case_study['risk_reduction']['strategy_max_drawdown']:.1f}% vs {case_study['risk_reduction']['benchmark_max_drawdown']:.1f}%")
        print(f"   • Return: {case_study['performance']['strategy_return']:.1f}% vs {case_study['performance']['benchmark_return']:.1f}%")
        
        return case_study
    
    def export_results(self, path: str):
        """Export backtest results to CSV."""
        if self.results is None:
            print("No results to export")
            return
        
        export_cols = ['Close', 'position_size', 'signal_name', 'market_return', 
                       'strategy_return', 'transaction_cost', 'strategy_equity', 'benchmark_equity']
        
        available_cols = [col for col in export_cols if col in self.results.columns]
        
        self.results[available_cols].to_csv(path)
        print(f"✓ Backtest results exported to {path}")
    
    def export_trades(self, path: str):
        """Export trade log to CSV."""
        if self.trades is not None and len(self.trades) > 0:
            self.trades.to_csv(path, index=False)
            print(f"✓ Trade log exported to {path}")


# Quick test when run directly
if __name__ == "__main__":
    # Create sample data with signals
    dates = pd.date_range('2026-04-08', '2026-05-14', freq='D')
    test_df = pd.DataFrame(index=dates)
    
    # Simulate price with trend and noise
    np.random.seed(42)
    n = len(dates)
    price = 50000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    test_df['Close'] = price
    
    # Simulate position sizes based on a simple rule
    test_df['position_size'] = 0.5  # default
    test_df.loc[:10, 'position_size'] = 0.8  # first period
    test_df.loc[10:20, 'position_size'] = 0.3  # reduce
    test_df.loc[20:, 'position_size'] = 0.6  # increase again
    
    test_df['signal_name'] = test_df['position_size'].map({
        0.8: 'BUY', 0.5: 'HOLD', 0.3: 'REDUCE', 0.6: 'BUY'
    }).fillna('HOLD')
    
    # Run backtest
    backtest = BacktestEngine(initial_capital=10000)
    results, metrics = backtest.run_backtest(test_df)
    
    # Generate case study metrics
    case_study = backtest.generate_case_study_metrics()
    
    print(f"\n✓ Backtest complete")
    print(f"✓ Final strategy equity: ${results['strategy_equity'].iloc[-1]:,.2f}")
    print(f"✓ Final benchmark equity: ${results['benchmark_equity'].iloc[-1]:,.2f}")