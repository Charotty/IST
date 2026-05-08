#!/usr/bin/env python3
"""
Cumulative PnL Tracking
======================

Advanced cumulative PnL tracking with time series analysis,
performance metrics, and visualization capabilities.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from .pnl_tracker import PnLTracker, PnLRecord, PositionSnapshot

logger = logging.getLogger(__name__)


@dataclass
class CumulativePnLSnapshot:
    """Snapshot of cumulative PnL at a point in time."""
    timestamp: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    equity_curve: List[float]
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    open_positions: int


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Efficiency metrics
    kelly_criterion: float
    gain_to_pain_ratio: float
    ulcer_index: float
    serenity_ratio: float


class CumulativePnLTracker:
    """
    Advanced cumulative PnL tracking with performance analysis.
    
    Features:
    - Real-time equity curve tracking
    - Drawdown analysis
    - Risk-adjusted performance metrics
    - Time series analysis
    - Visualization capabilities
    """
    
    def __init__(
        self,
        pnl_tracker: PnLTracker,
        lookback_days: int = 252,
        risk_free_rate: float = 0.02
    ) -> None:
        self.pnl_tracker = pnl_tracker
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        
        # Time series data
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        self.drawdown_series: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.high_watermark: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_start: Optional[datetime] = None
        self.max_drawdown_end: Optional[datetime] = None
        self.current_drawdown_start: Optional[datetime] = None
        
        # Metrics cache
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
    
    def update(self, timestamp: Optional[datetime] = None) -> CumulativePnLSnapshot:
        """
        Update cumulative PnL tracking.
        
        Args:
            timestamp: Update timestamp (default: now)
            
        Returns:
            Current PnL snapshot
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current PnL data
        total_pnl = self.pnl_tracker.get_cumulative_pnl()
        realized_pnl = sum(record.realized_pnl for record in self.pnl_tracker.pnl_records)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.pnl_tracker.positions.values())
        
        # Calculate daily PnL
        daily_pnl = self._calculate_daily_pnl(timestamp)
        
        # Update equity curve
        self._update_equity_curve(timestamp, total_pnl)
        
        # Update drawdown tracking
        self._update_drawdown(timestamp, total_pnl)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        # Create snapshot
        snapshot = CumulativePnLSnapshot(
            timestamp=timestamp,
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            equity_curve=[p[1] for p in self.equity_curve],
            max_drawdown=self.max_drawdown,
            current_drawdown=self.current_drawdown,
            sharpe_ratio=performance.sharpe_ratio,
            win_rate=performance.win_rate,
            profit_factor=performance.profit_factor,
            total_trades=len(self.pnl_tracker.pnl_records),
            open_positions=len(self.pnl_tracker.positions)
        )
        
        return snapshot
    
    def _calculate_daily_pnl(self, timestamp: datetime) -> float:
        """Calculate PnL for the current day."""
        today_str = timestamp.strftime('%Y-%m-%d')
        today_trades = [
            record for record in self.pnl_tracker.pnl_records
            if record.exit_time and record.exit_time.strftime('%Y-%m-%d') == today_str
        ]
        
        return sum(record.realized_pnl for record in today_trades)
    
    def _update_equity_curve(self, timestamp: datetime, total_pnl: float) -> None:
        """Update equity curve with new PnL data."""
        # Add to equity curve
        self.equity_curve.append((timestamp, total_pnl))
        
        # Calculate daily return
        if len(self.equity_curve) >= 2:
            prev_value = self.equity_curve[-2][1]
            if prev_value != 0:
                daily_return = (total_pnl - prev_value) / abs(prev_value)
                self.daily_returns.append((timestamp, daily_return))
        
        # Keep only recent data (memory management)
        max_points = 10000
        if len(self.equity_curve) > max_points:
            self.equity_curve = self.equity_curve[-max_points:]
        if len(self.daily_returns) > max_points:
            self.daily_returns = self.daily_returns[-max_points:]
    
    def _update_drawdown(self, timestamp: datetime, total_pnl: float) -> None:
        """Update drawdown tracking."""
        if total_pnl > self.high_watermark:
            # New high watermark
            self.high_watermark = total_pnl
            self.current_drawdown = 0.0
            self.current_drawdown_start = None
            
            # End previous drawdown if any
            if self.current_drawdown_start and self.current_drawdown > 0:
                # Check if this was the max drawdown
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.max_drawdown_end = timestamp
        else:
            # In drawdown
            if self.current_drawdown_start is None:
                self.current_drawdown_start = timestamp
            
            self.current_drawdown = (self.high_watermark - total_pnl) / self.high_watermark if self.high_watermark != 0 else 0.0
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.max_drawdown_start = self.current_drawdown_start
                self.max_drawdown_end = timestamp
        
        # Add to drawdown series
        self.drawdown_series.append((timestamp, self.current_drawdown))
        
        # Keep only recent data
        if len(self.drawdown_series) > 10000:
            self.drawdown_series = self.drawdown_series[-10000:]
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        # Check cache
        if (self._metrics_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._metrics_cache
        
        # Get basic metrics from PnL tracker
        basic_metrics = self.pnl_tracker.get_performance_metrics()
        
        # Calculate returns
        if self.daily_returns:
            returns = [r[1] for r in self.daily_returns]
            daily_return_mean = np.mean(returns)
            daily_return_std = np.std(returns)
            annualized_return = daily_return_mean * 252
            volatility = daily_return_std * np.sqrt(252)
        else:
            daily_return_mean = 0.0
            daily_return_std = 0.0
            annualized_return = 0.0
            volatility = 0.0
        
        # Calculate risk metrics
        if self.daily_returns:
            returns_array = np.array([r[1] for r in self.daily_returns])
            var_95 = np.percentile(returns_array, 5)
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        # Calculate risk-adjusted metrics
        if volatility > 0:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        if self.daily_returns:
            downside_returns = [r[1] for r in self.daily_returns if r[1] < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            else:
                sortino_ratio = float('inf')
        else:
            sortino_ratio = 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(self.max_drawdown) if self.max_drawdown > 0 else 0.0
        
        # Calculate additional metrics
        total_return = self.pnl_tracker.get_cumulative_pnl()
        
        # Kelly criterion
        winning_trades = basic_metrics.get('winning_trades', 0)
        losing_trades = basic_metrics.get('losing_trades', 0)
        avg_win = basic_metrics.get('avg_win', 0)
        avg_loss = abs(basic_metrics.get('avg_loss', 0))
        
        if losing_trades > 0 and avg_loss > 0:
            win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
            kelly_criterion = win_rate - ((1 - win_rate) * (avg_loss / avg_win)) if avg_win > 0 else 0
        else:
            kelly_criterion = 0.0
        
        # Gain to pain ratio
        if self.daily_returns:
            positive_returns = sum(r[1] for r in self.daily_returns if r[1] > 0)
            negative_returns = abs(sum(r[1] for r in self.daily_returns if r[1] < 0))
            gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        else:
            gain_to_pain_ratio = 0.0
        
        # Ulcer index
        if self.drawdown_series:
            ulcer_index = np.sqrt(np.mean([d[1]**2 for d in self.drawdown_series]))
        else:
            ulcer_index = 0.0
        
        # Serenity ratio
        if volatility > 0:
            serenity_ratio = annualized_return / (volatility * np.sqrt(1 + self.max_drawdown**2))
        else:
            serenity_ratio = 0.0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_return_mean=daily_return_mean,
            daily_return_std=daily_return_std,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration=self._calculate_drawdown_duration(),
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=0.0,  # Would need benchmark
            total_trades=basic_metrics.get('total_trades', 0),
            win_rate=basic_metrics.get('win_rate', 0),
            profit_factor=basic_metrics.get('profit_factor', 0),
            avg_win=basic_metrics.get('avg_win', 0),
            avg_loss=basic_metrics.get('avg_loss', 0),
            largest_win=basic_metrics.get('largest_win', 0),
            largest_loss=basic_metrics.get('largest_loss', 0),
            kelly_criterion=kelly_criterion,
            gain_to_pain_ratio=gain_to_pain_ratio,
            ulcer_index=ulcer_index,
            serenity_ratio=serenity_ratio
        )
        
        # Cache metrics
        self._metrics_cache = metrics
        self._cache_timestamp = datetime.now()
        
        return metrics
    
    def _calculate_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days."""
        if not self.drawdown_series:
            return 0
        
        max_duration = 0
        current_duration = 0
        in_drawdown = False
        start_time = None
        
        for timestamp, drawdown in self.drawdown_series:
            if drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                    start_time = timestamp
                current_duration = (timestamp - start_time).days
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        return max(max_duration, current_duration)
    
    def get_equity_curve_dataframe(self) -> pd.DataFrame:
        """Get equity curve as pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'pnl'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Add returns
        df['returns'] = df['pnl'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    def get_drawdown_dataframe(self) -> pd.DataFrame:
        """Get drawdown series as pandas DataFrame."""
        if not self.drawdown_series:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.drawdown_series, columns=['timestamp', 'drawdown'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> str:
        """Plot equity curve with drawdowns."""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        timestamps = [p[0] for p in self.equity_curve]
        pnl_values = [p[1] for p in self.equity_curve]
        
        ax1.plot(timestamps, pnl_values, 'b-', linewidth=2, label='Equity Curve')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('PnL')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown
        if self.drawdown_series:
            dd_timestamps = [d[0] for d in self.drawdown_series]
            dd_values = [d[1] * 100 for d in self.drawdown_series]  # Convert to percentage
            
            ax2.fill_between(dd_timestamps, dd_values, 0, color='red', alpha=0.3, label='Drawdown')
            ax2.plot(dd_timestamps, dd_values, 'r-', linewidth=1)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curve plot saved: {save_path}")
        return save_path
    
    def plot_performance_summary(self, save_path: Optional[str] = None) -> str:
        """Plot performance summary with key metrics."""
        metrics = self._calculate_performance_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Summary', fontsize=16)
        
        # Returns distribution
        if self.daily_returns:
            returns = [r[1] for r in self.daily_returns]
            axes[0, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Daily Returns Distribution')
            axes[0, 0].set_xlabel('Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Monthly returns heatmap
        if self.daily_returns:
            df = pd.DataFrame(self.daily_returns, columns=['timestamp', 'return'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            monthly_returns = df['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create heatmap data
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            heatmap_data = pd.DataFrame(index=months, columns=years)
            
            for year in years:
                for month in months:
                    try:
                        value = monthly_returns[(monthly_returns.index.year == year) & (monthly_returns.index.month == month)].iloc[0]
                        heatmap_data.loc[month, year] = value * 100  # Convert to percentage
                    except IndexError:
                        heatmap_data.loc[month, year] = 0
            
            sns.heatmap(heatmap_data.astype(float), ax=axes[0, 1], cmap='RdYlGn', center=0, 
                       annot=True, fmt='.1f', cbar_kws={'label': 'Return (%)'})
            axes[0, 1].set_title('Monthly Returns Heatmap')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Month')
        
        # Rolling Sharpe ratio
        if len(self.daily_returns) >= 30:
            df = pd.DataFrame(self.daily_returns, columns=['timestamp', 'return'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            rolling_sharpe = df['return'].rolling(window=30).mean() / df['return'].rolling(window=30).std() * np.sqrt(252)
            axes[0, 2].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[0, 2].set_title('30-Day Rolling Sharpe Ratio')
            axes[0, 2].set_ylabel('Sharpe Ratio')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Key metrics bar chart
        key_metrics = {
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Calmar': metrics.calmar_ratio,
            'Win Rate %': metrics.win_rate,
            'Profit Factor': metrics.profit_factor
        }
        
        metric_names = list(key_metrics.keys())
        metric_values = list(key_metrics.values())
        
        bars = axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Key Performance Metrics')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color bars based on value
        for bar, value in zip(bars, metric_values):
            if value >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Drawdown duration
        if self.drawdown_series:
            df = pd.DataFrame(self.drawdown_series, columns=['timestamp', 'drawdown'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate drawdown periods
            in_drawdown = False
            drawdown_periods = []
            start_time = None
            
            for timestamp, drawdown in df.itertuples():
                if drawdown > 0:
                    if not in_drawdown:
                        in_drawdown = True
                        start_time = timestamp
                else:
                    if in_drawdown:
                        duration = (timestamp - start_time).days
                        drawdown_periods.append(duration)
                        in_drawdown = False
            
            if drawdown_periods:
                axes[1, 1].hist(drawdown_periods, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Drawdown Duration Distribution')
                axes[1, 1].set_xlabel('Duration (days)')
                axes[1, 1].set_ylabel('Frequency')
        
        # Risk metrics
        risk_metrics = {
            'Max DD %': metrics.max_drawdown * 100,
            'Volatility %': metrics.volatility * 100,
            'VaR 95%': metrics.var_95 * 100,
            'CVaR 95%': metrics.cvar_95 * 100
        }
        
        risk_names = list(risk_metrics.keys())
        risk_values = list(risk_metrics.values())
        
        bars = axes[1, 2].bar(risk_names, risk_values)
        axes[1, 2].set_title('Risk Metrics')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Color bars (red for risk metrics)
        for bar in bars:
            bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance summary plot saved: {save_path}")
        return save_path
    
    def export_performance_report(self, save_path: Optional[str] = None) -> str:
        """Export comprehensive performance report."""
        metrics = self._calculate_performance_metrics()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': asdict(metrics),
            'current_positions': len(self.pnl_tracker.positions),
            'total_trades': len(self.pnl_tracker.pnl_records),
            'equity_curve_points': len(self.equity_curve),
            'max_drawdown_start': self.max_drawdown_start.isoformat() if self.max_drawdown_start else None,
            'max_drawdown_end': self.max_drawdown_end.isoformat() if self.max_drawdown_end else None,
            'current_drawdown_start': self.current_drawdown_start.isoformat() if self.current_drawdown_start else None
        }
        
        if save_path is None:
            save_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported: {save_path}")
        return save_path
