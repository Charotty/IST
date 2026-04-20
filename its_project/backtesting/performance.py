from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from its_project.backtesting.base import BacktestResult, Trade


class PerformanceAnalyzer:
    """Performance analysis and visualization for backtest results."""
    
    def __init__(self, result: BacktestResult) -> None:
        self.result = result
        
    def calculate_detailed_metrics(self) -> Dict[str, float]:
        """Calculate detailed performance metrics."""
        equity = self.result.equity_curve
        returns = self.result.returns
        
        # Basic metrics
        total_return = (equity[-1] / equity[0]) - 1
        
        # Risk metrics
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0.0
            sortino_ratio = np.sqrt(252) * (returns.mean() / np.minimum(returns, 0).std()) if np.minimum(returns, 0).std() > 0 else 0.0
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0.0
        else:
            sharpe_ratio = sortino_ratio = var_95 = cvar_95 = 0.0
        
        # Drawdown metrics
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Find drawdown periods
        drawdown_periods = self._find_drawdown_periods(drawdown)
        avg_drawdown_duration = np.mean([p[1] - p[0] for p in drawdown_periods]) if drawdown_periods else 0
        
        # Trade metrics
        trades_with_pnl = [t for t in self.result.trades if t.pnl is not None]
        if trades_with_pnl:
            win_rate = sum(1 for t in trades_with_pnl if t.pnl > 0) / len(trades_with_pnl)
            avg_win = np.mean([t.pnl for t in trades_with_pnl if t.pnl > 0])
            avg_loss = np.mean([t.pnl for t in trades_with_pnl if t.pnl < 0])
            profit_factor = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = (total_return * 252 / len(equity)) / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': total_return * 252 / len(equity),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(self.result.trades),
            'avg_trade_duration': self._calculate_avg_trade_duration(),
        }
    
    def _find_drawdown_periods(self, drawdown: np.ndarray) -> List[tuple]:
        """Find drawdown periods (start, end indices)."""
        periods = []
        in_drawdown = False
        start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                periods.append((start, i))
        
        return periods
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in bars."""
        trades_with_pnl = [t for t in self.result.trades if t.pnl is not None]
        if len(trades_with_pnl) < 2:
            return 0.0
        
        durations = []
        for i in range(1, len(trades_with_pnl), 2):  # Assuming pairs of open/close
            if i < len(trades_with_pnl):
                duration = trades_with_pnl[i].timestamp - trades_with_pnl[i-1].timestamp
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def plot_equity_curve(self, figsize: tuple = (12, 6)) -> None:
        """Plot equity curve with drawdowns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        ax1.plot(self.result.equity_curve, label='Equity', color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown
        peak = np.maximum.accumulate(self.result.equity_curve)
        drawdown = (self.result.equity_curve - peak) / peak * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown, color='red', label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Bar')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, figsize: tuple = (12, 4)) -> None:
        """Plot returns distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(self.result.returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.result.returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, figsize: tuple = (12, 6)) -> None:
        """Plot monthly returns heatmap."""
        if len(self.result.trades) == 0:
            print("No trades to plot monthly returns")
            return
        
        # Create DataFrame with timestamps and returns
        df = pd.DataFrame({
            'timestamp': [t.timestamp for t in self.result.trades if t.pnl],
            'pnl': [t.pnl for t in self.result.trades if t.pnl]
        })
        
        # Convert to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        # Resample to monthly
        monthly_returns = df['pnl'].resample('M').sum()
        
        # Create year/month columns for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        heatmap_data = monthly_returns.to_frame('returns')
        heatmap_data['year'] = heatmap_data.index.year
        heatmap_data['month'] = heatmap_data.index.month
        
        # Pivot for heatmap
        pivot_table = heatmap_data.pivot(index='year', columns='month', values='returns')
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
        plt.ylabel('Year')
        plt.xlabel('Month')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        metrics = self.calculate_detailed_metrics()
        
        report = f"""
BACKTEST PERFORMANCE REPORT
==========================

Portfolio Metrics:
------------------
Initial Capital: ${self.result.equity_curve[0]:,.2f}
Final Capital: ${self.result.equity_curve[-1]:,.2f}
Total Return: {metrics['total_return']:.2%}
Annual Return: {metrics['annual_return']:.2%}

Risk Metrics:
------------
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2%}
Average Drawdown Duration: {metrics['avg_drawdown_duration']:.1f} bars
VaR (95%): {metrics['var_95']:.2%}
CVaR (95%): {metrics['cvar_95']:.2%}

Trading Metrics:
---------------
Total Trades: {metrics['num_trades']}
Win Rate: {metrics['win_rate']:.2%}
Average Win: ${metrics['avg_win']:.2f}
Average Loss: ${metrics['avg_loss']:.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Average Trade Duration: {metrics['avg_trade_duration']:.1f} bars

Trade Statistics:
-----------------
Total Commission: ${sum(t.commission for t in self.result.trades):,.2f}
Total Slippage: ${sum(t.slippage for t in self.result.trades):,.2f}
Average Trade Size: ${np.mean([t.size for t in self.result.trades]):.4f}

Risk-Adjusted Performance:
-------------------------
The strategy {'achieved' if metrics['sharpe_ratio'] > 1.0 else 'did not achieve'} a Sharpe ratio above 1.0.
Maximum drawdown of {metrics['max_drawdown']:.2%} is {'acceptable' if metrics['max_drawdown'] < 0.2 else 'high'}.
Win rate of {metrics['win_rate']:.2%} is {'good' if metrics['win_rate'] > 0.5 else 'needs improvement'}.

Recommendations:
----------------
"""
        # Add recommendations based on metrics
        if metrics['sharpe_ratio'] < 1.0:
            report += "- Consider improving risk-adjusted returns (Sharpe < 1.0)\n"
        if metrics['max_drawdown'] > 0.2:
            report += "- Implement tighter risk controls (drawdown > 20%)\n"
        if metrics['win_rate'] < 0.4:
            report += "- Review entry/exit criteria (win rate < 40%)\n"
        if metrics['profit_factor'] < 1.5:
            report += "- Optimize profit-taking strategy (profit factor < 1.5)\n"
        
        return report
