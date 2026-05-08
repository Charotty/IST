#!/usr/bin/env python3
"""
Equity Curve Tracking
===================

Advanced equity curve tracking with time series analysis,
visualization, and performance comparison capabilities.
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .pnl_tracker import PnLTracker, PnLRecord
from .data_persistence import DataPersistenceManager

logger = logging.getLogger(__name__)


@dataclass
class EquityCurvePoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity_value: float
    cumulative_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    open_positions: int
    daily_return: Optional[float] = None
    benchmark_value: Optional[float] = None
    benchmark_return: Optional[float] = None


@dataclass
class EquityCurveStatistics:
    """Statistical analysis of equity curve."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    best_day: float
    worst_day: float
    positive_days: int
    negative_days: int
    win_rate_daily: float
    average_daily_return: float
    skewness: float
    kurtosis: float


@dataclass
class EquityCurveConfig:
    """Configuration for equity curve tracking."""
    starting_capital: float = 100000.0
    include_benchmark: bool = False
    benchmark_returns: Optional[List[float]] = None
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    include_commissions: bool = True
    include_fees: bool = True
    smooth_curve: bool = False
    smoothing_window: int = 5
    time_zone: str = "UTC"


class EquityCurveTracker:
    """
    Advanced equity curve tracking with comprehensive analysis.
    
    Features:
    - Real-time equity curve updates
    - Time series statistical analysis
    - Benchmark comparison
    - Drawdown and recovery analysis
    - Performance attribution
    - Visualization and export
    """
    
    def __init__(
        self,
        pnl_tracker: PnLTracker,
        config: Optional[EquityCurveConfig] = None,
        persistence_manager: Optional[DataPersistenceManager] = None
    ) -> None:
        self.pnl_tracker = pnl_tracker
        self.config = config or EquityCurveConfig()
        self.persistence_manager = persistence_manager
        
        # Equity curve data
        self.equity_points: List[EquityCurvePoint] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = self.config.benchmark_returns or []
        
        # Performance tracking
        self.high_watermark: float = self.config.starting_capital
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_start: Optional[datetime] = None
        self.max_drawdown_end: Optional[datetime] = None
        self.current_drawdown_start: Optional[datetime] = None
        
        # Statistics cache
        self._statistics_cache: Optional[EquityCurveStatistics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        
        # Initialize with starting capital
        self._add_initial_point()
    
    def _add_initial_point(self) -> None:
        """Add initial equity curve point."""
        initial_point = EquityCurvePoint(
            timestamp=datetime.now(),
            equity_value=self.config.starting_capital,
            cumulative_pnl=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_trades=0,
            open_positions=0,
            daily_return=0.0,
            benchmark_value=self.config.starting_capital,
            benchmark_return=0.0
        )
        
        self.equity_points.append(initial_point)
        self.high_watermark = self.config.starting_capital
    
    def update_equity_curve(
        self,
        timestamp: Optional[datetime] = None,
        force_recalculate: bool = False
    ) -> EquityCurvePoint:
        """
        Update equity curve with current PnL tracker state.
        
        Args:
            timestamp: Update timestamp (default: now)
            force_recalculate: Force full recalculation
            
        Returns:
            Latest equity curve point
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current state from PnL tracker
        cumulative_pnl = self.pnl_tracker.get_cumulative_pnl()
        realized_pnl = sum(record.realized_pnl for record in self.pnl_tracker.pnl_records)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.pnl_tracker.positions.values())
        total_trades = len(self.pnl_tracker.pnl_records)
        open_positions = len(self.pnl_tracker.positions)
        
        # Calculate current equity value
        current_equity = self.config.starting_capital + cumulative_pnl
        
        # Calculate daily return
        daily_return = self._calculate_daily_return(current_equity, timestamp)
        
        # Calculate benchmark return if enabled
        benchmark_value = None
        benchmark_return = None
        if self.config.include_benchmark and self.benchmark_returns:
            benchmark_value = self._get_benchmark_value(timestamp)
            if benchmark_value:
                benchmark_return = self._calculate_benchmark_return(benchmark_value, timestamp)
        
        # Create equity point
        equity_point = EquityCurvePoint(
            timestamp=timestamp,
            equity_value=current_equity,
            cumulative_pnl=cumulative_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_trades=total_trades,
            open_positions=open_positions,
            daily_return=daily_return,
            benchmark_value=benchmark_value,
            benchmark_return=benchmark_return
        )
        
        # Add to equity curve
        if force_recalculate or not self.equity_points:
            self.equity_points = [equity_point]
        else:
            # Check if we need to add new point (avoid duplicates)
            if not self.equity_points or timestamp > self.equity_points[-1].timestamp:
                self.equity_points.append(equity_point)
            else:
                # Update last point
                self.equity_points[-1] = equity_point
        
        # Update drawdown tracking
        self._update_drawdown_analysis(equity_point)
        
        # Invalidate cache
        self._statistics_cache = None
        self._cache_timestamp = None
        
        # Save to persistence if available
        if self.persistence_manager:
            asyncio.create_task(self._save_equity_point(equity_point))
        
        logger.debug(f"Equity curve updated: {current_equity:.2f} at {timestamp}")
        return equity_point
    
    def _calculate_daily_return(
        self,
        current_equity: float,
        timestamp: datetime
    ) -> float:
        """Calculate daily return for equity point."""
        if not self.equity_points:
            return 0.0
        
        # Find previous point from same day or previous day
        previous_point = None
        for point in reversed(self.equity_points[:-1]):
            if point.timestamp.date() <= timestamp.date():
                previous_point = point
                break
        
        if previous_point and previous_point.equity_value != 0:
            daily_return = (current_equity - previous_point.equity_value) / previous_point.equity_value
            self.daily_returns.append(daily_return)
            return daily_return
        
        return 0.0
    
    def _get_benchmark_value(self, timestamp: datetime) -> Optional[float]:
        """Get benchmark value for given timestamp."""
        if not self.benchmark_returns:
            return None
        
        # Simple linear interpolation for benchmark
        # In practice, this would use actual benchmark data
        days_elapsed = (timestamp - self.equity_points[0].timestamp).days
        if days_elapsed < len(self.benchmark_returns):
            return self.benchmark_returns[days_elapsed]
        
        return self.benchmark_returns[-1] if self.benchmark_returns else None
    
    def _calculate_benchmark_return(
        self,
        benchmark_value: float,
        timestamp: datetime
    ) -> float:
        """Calculate benchmark return."""
        if not self.equity_points:
            return 0.0
        
        # Find previous benchmark value
        previous_benchmark = None
        for point in reversed(self.equity_points[:-1]):
            if point.benchmark_value is not None:
                previous_benchmark = point.benchmark_value
                break
        
        if previous_benchmark and previous_benchmark != 0:
            return (benchmark_value - previous_benchmark) / previous_benchmark
        
        return 0.0
    
    def _update_drawdown_analysis(self, equity_point: EquityCurvePoint) -> None:
        """Update drawdown analysis with new equity point."""
        current_equity = equity_point.equity_value
        
        if current_equity > self.high_watermark:
            # New high watermark
            self.high_watermark = current_equity
            self.current_drawdown = 0.0
            self.current_drawdown_start = None
            
            # End previous drawdown if any
            if self.current_drawdown_start and self.current_drawdown > 0:
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
                    self.max_drawdown_start = self.current_drawdown_start
                    self.max_drawdown_end = equity_point.timestamp
        else:
            # In drawdown
            if self.current_drawdown_start is None:
                self.current_drawdown_start = self.equity_points[0].timestamp if self.equity_points else equity_point.timestamp
            
            self.current_drawdown = (self.high_watermark - current_equity) / self.high_watermark if self.high_watermark > 0 else 0.0
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.max_drawdown_start = self.current_drawdown_start
                self.max_drawdown_end = equity_point.timestamp
    
    def calculate_statistics(self) -> EquityCurveStatistics:
        """
        Calculate comprehensive equity curve statistics.
        
        Returns:
            EquityCurveStatistics object with all metrics
        """
        # Check cache
        if (self._statistics_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._statistics_cache
        
        if not self.equity_points:
            stats = EquityCurveStatistics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, max_drawdown_duration=0,
                calmar_ratio=0.0, sortino_ratio=0.0, var_95=0.0, cvar_95=0.0,
                best_day=0.0, worst_day=0.0, positive_days=0, negative_days=0,
                win_rate_daily=0.0, average_daily_return=0.0, skewness=0.0, kurtosis=0.0
            )
            self._statistics_cache = stats
            self._cache_timestamp = datetime.now()
            return stats
        
        # Extract equity values and returns
        equity_values = [point.equity_value for point in self.equity_points]
        returns = [point.daily_return for point in self.equity_points if point.daily_return is not None]
        
        if not equity_values:
            return EquityCurveStatistics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, max_drawdown_duration=0,
                calmar_ratio=0.0, sortino_ratio=0.0, var_95=0.0, cvar_95=0.0,
                best_day=0.0, worst_day=0.0, positive_days=0, negative_days=0,
                win_rate_daily=0.0, average_daily_return=0.0, skewness=0.0, kurtosis=0.0
            )
        
        # Basic return calculations
        starting_value = self.config.starting_capital
        ending_value = equity_values[-1]
        total_return = (ending_value - starting_value) / starting_value
        
        # Time-based calculations
        time_span_days = (self.equity_points[-1].timestamp - self.equity_points[0].timestamp).days
        if time_span_days > 0:
            annualized_return = (ending_value / starting_value) ** (365.25 / time_span_days) - 1
        else:
            annualized_return = 0.0
        
        # Volatility and risk metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else float('inf')
            else:
                sortino_ratio = float('inf') if annualized_return > 0 else 0.0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean([r for r in returns if r <= var_95])
            
            # Daily statistics
            positive_days = len([r for r in returns if r > 0])
            negative_days = len([r for r in returns if r < 0])
            win_rate_daily = positive_days / len(returns) if returns else 0.0
            average_daily_return = np.mean(returns) if returns else 0.0
            
            # Distribution statistics
            skewness = float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0
            kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0
            
            # Best and worst days
            best_day = np.max(returns) if returns else 0.0
            worst_day = np.min(returns) if returns else 0.0
        else:
            volatility = sharpe_ratio = sortino_ratio = 0.0
            var_95 = cvar_95 = best_day = worst_day = 0.0
            positive_days = negative_days = win_rate_daily = average_daily_return = 0.0
            skewness = kurtosis = 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0.0
        
        # Max drawdown duration
        if self.max_drawdown_start and self.max_drawdown_end:
            max_drawdown_duration = (self.max_drawdown_end - self.max_drawdown_start).days
        else:
            max_drawdown_duration = 0
        
        stats = EquityCurveStatistics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            best_day=best_day,
            worst_day=worst_day,
            positive_days=positive_days,
            negative_days=negative_days,
            win_rate_daily=win_rate_daily,
            average_daily_return=average_daily_return,
            skewness=skewness,
            kurtosis=kurtosis
        )
        
        # Cache results
        self._statistics_cache = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as pandas DataFrame."""
        if not self.equity_points:
            return pd.DataFrame()
        
        data = []
        for point in self.equity_points:
            row = {
                'timestamp': point.timestamp,
                'equity_value': point.equity_value,
                'cumulative_pnl': point.cumulative_pnl,
                'realized_pnl': point.realized_pnl,
                'unrealized_pnl': point.unrealized_pnl,
                'total_trades': point.total_trades,
                'open_positions': point.open_positions,
                'daily_return': point.daily_return or 0.0
            }
            
            if point.benchmark_value is not None:
                row['benchmark_value'] = point.benchmark_value
                row['benchmark_return'] = point.benchmark_return or 0.0
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_returns_dataframe(self) -> pd.DataFrame:
        """Get returns as pandas DataFrame."""
        if not self.daily_returns:
            return pd.DataFrame()
        
        returns_data = []
        timestamps = []
        
        for i, point in enumerate(self.equity_points):
            if point.daily_return is not None:
                returns_data.append(point.daily_return)
                timestamps.append(point.timestamp)
        
        if returns_data:
            df = pd.DataFrame({
                'return': returns_data
            }, index=timestamps)
            df.index.name = 'timestamp'
            return df
        
        return pd.DataFrame()
    
    def get_drawdown_dataframe(self) -> pd.DataFrame:
        """Get drawdown analysis as pandas DataFrame."""
        if not self.equity_points:
            return pd.DataFrame()
        
        data = []
        for point in self.equity_points:
            # Calculate drawdown at this point
            if point.equity_value >= self.config.starting_capital:
                drawdown = 0.0
            else:
                drawdown = (self.config.starting_capital - point.equity_value) / self.config.starting_capital
            
            data.append({
                'timestamp': point.timestamp,
                'equity_value': point.equity_value,
                'drawdown': drawdown,
                'high_watermark': self.high_watermark,
                'current_drawdown': self.current_drawdown
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def create_equity_chart(
        self,
        include_benchmark: bool = False,
        include_drawdown: bool = True,
        smooth_curve: bool = False,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive equity curve chart.
        
        Args:
            include_benchmark: Include benchmark comparison
            include_drawdown: Include drawdown subplot
            smooth_curve: Apply smoothing to equity curve
            save_path: Path to save chart
            
        Returns:
            Plotly figure object
        """
        if not self.equity_points:
            return go.Figure()
        
        # Create subplots if drawdown included
        if include_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Equity Curve', 'Drawdown'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Extract data
        timestamps = [point.timestamp for point in self.equity_points]
        equity_values = [point.equity_value for point in self.equity_points]
        
        # Apply smoothing if requested
        if smooth_curve and len(equity_values) > self.config.smoothing_window:
            equity_series = pd.Series(equity_values)
            equity_values = equity_series.rolling(
                window=self.config.smoothing_window,
                center=True
            ).mean().tolist()
        
        # Plot equity curve
        equity_trace = go.Scatter(
            x=timestamps,
            y=equity_values,
            mode='lines',
            name='Equity Curve',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        )
        
        if include_drawdown:
            fig.add_trace(equity_trace, row=1, col=1)
        else:
            fig.add_trace(equity_trace)
        
        # Add benchmark if available and requested
        if include_benchmark and self.config.include_benchmark:
            benchmark_values = [point.benchmark_value for point in self.equity_points if point.benchmark_value is not None]
            if benchmark_values:
                benchmark_trace = go.Scatter(
                    x=timestamps[:len(benchmark_values)],
                    y=benchmark_values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                    hovertemplate='Date: %{x}<br>Benchmark: $%{y:,.2f}<extra></extra>'
                )
                
                if include_drawdown:
                    fig.add_trace(benchmark_trace, row=1, col=1)
                else:
                    fig.add_trace(benchmark_trace)
        
        # Add starting capital line
        starting_capital_line = go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[self.config.starting_capital, self.config.starting_capital],
            mode='lines',
            name='Starting Capital',
            line=dict(color='gray', width=1, dash='dot'),
            hovertemplate='Starting Capital: $%{y:,.2f}<extra></extra>',
            showlegend=True
        )
        
        if include_drawdown:
            fig.add_trace(starting_capital_line, row=1, col=1)
        else:
            fig.add_trace(starting_capital_line)
        
        # Add drawdown chart
        if include_drawdown:
            drawdown_values = []
            for point in self.equity_points:
                if point.equity_value >= self.config.starting_capital:
                    drawdown_values.append(0.0)
                else:
                    drawdown_values.append((self.config.starting_capital - point.equity_value) / self.config.starting_capital)
            
            drawdown_trace = go.Scatter(
                x=timestamps,
                y=drawdown_values,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>'
            )
            
            fig.add_trace(drawdown_trace, row=2, col=1)
            
            # Update layout for subplots
            fig.update_layout(
                title='Equity Curve Analysis',
                template='plotly_white',
                height=800,
                xaxis_title='Date',
                yaxis_title='Equity Value',
                yaxis2_title='Drawdown (%)'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Equity Value", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        else:
            # Update layout for single plot
            fig.update_layout(
                title='Equity Curve',
                template='plotly_white',
                height=600,
                xaxis_title='Date',
                yaxis_title='Equity Value'
            )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Equity curve chart saved to {save_path}")
        
        return fig
    
    def create_returns_distribution_chart(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create returns distribution chart."""
        if not self.daily_returns:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns Distribution', 'Returns Over Time',
                          'Daily Returns Heatmap', 'Q-Q Plot'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Returns histogram
        fig.add_trace(
            go.Histogram(
                x=self.daily_returns,
                nbinsx=50,
                name='Returns Distribution',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # Returns over time
        timestamps = [point.timestamp for point in self.equity_points if point.daily_return is not None]
        returns_with_dates = [point.daily_return for point in self.equity_points if point.daily_return is not None]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=returns_with_dates,
                mode='lines+markers',
                name='Daily Returns',
                line=dict(color='#ff7f0e', width=1),
                marker=dict(size=3, color='#ff7f0e')
            ),
            row=1, col=2
        )
        
        # Create heatmap data
        if len(self.daily_returns) >= 7:  # Need at least a week for heatmap
            returns_df = self.get_returns_dataframe()
            returns_df['weekday'] = returns_df.index.dayofweek
            returns_df['hour'] = returns_df.index.hour
            
            # Aggregate by weekday and hour (simplified)
            heatmap_data = returns_df.groupby(['weekday'])['return'].mean().reset_index()
            
            fig.add_trace(
                go.Heatmap(
                    x=heatmap_data['weekday'],
                    y=heatmap_data['return'],
                    colorscale='RdYlGn',
                    name='Returns by Weekday'
                ),
                row=2, col=1
            )
        
        # Q-Q plot
        if len(self.daily_returns) > 10:
            sorted_returns = sorted(self.daily_returns)
            theoretical_quantiles = np.linspace(0, 1, len(sorted_returns))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='#2ca02c')
                ),
                row=2, col=2
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_quantiles,
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='gray', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Returns Analysis',
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Returns distribution chart saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive performance dashboard."""
        stats = self.calculate_statistics()
        
        # Create dashboard with KPIs
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Return Metrics', 'Risk Metrics', 'Trade Statistics',
                'Current Status', 'Drawdown Analysis', 'Performance Attribution'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}]
            ]
        )
        
        # Return Metrics
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.total_return * 100,
                title={"text": "Total Return (%)"},
                gauge={'axis': {'range': [None, 50]}},
                delta={'reference': 0}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.annualized_return * 100,
                title={"text": "Annualized Return (%)"},
                gauge={'axis': {'range': [None, 100]}},
                delta={'reference': 0}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.sharpe_ratio,
                title={"text": "Sharpe Ratio"},
                gauge={'axis': {'range': [None, 3]}},
                delta={'reference': 1.0}
            ),
            row=1, col=3
        )
        
        # Risk Metrics
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.volatility * 100,
                title={"text": "Volatility (%)"},
                gauge={'axis': {'range': [None, 50]}},
                delta={'reference': 0}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.max_drawdown * 100,
                title={"text": "Max Drawdown (%)"},
                gauge={'axis': {'range': [None, 50]}},
                delta={'reference': 0}
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=stats.calmar_ratio,
                title={"text": "Calmar Ratio"},
                gauge={'axis': {'range': [None, 5]}},
                delta={'reference': 1.0}
            ),
            row=2, col=3
        )
        
        # Trade Statistics
        if self.equity_points:
            current_point = self.equity_points[-1]
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=current_point.total_trades,
                    title={"text": "Total Trades"}
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=current_point.open_positions,
                    title={"text": "Open Positions"}
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=current_point.equity_value,
                    title={"text": "Current Equity"},
                    number={'prefix': "$"}
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            title='Performance Dashboard',
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    async def _save_equity_point(self, equity_point: EquityCurvePoint) -> None:
        """Save equity point to persistence."""
        if not self.persistence_manager:
            return
        
        try:
            # Convert to dict for database storage
            equity_data = {
                'timestamp': equity_point.timestamp,
                'equity_value': equity_point.equity_value,
                'cumulative_pnl': equity_point.cumulative_pnl,
                'realized_pnl': equity_point.realized_pnl,
                'unrealized_pnl': equity_point.unrealized_pnl,
                'total_trades': equity_point.total_trades,
                'open_positions': equity_point.open_positions,
                'daily_return': equity_point.daily_return,
                'benchmark_value': equity_point.benchmark_value,
                'benchmark_return': equity_point.benchmark_return
            }
            
            # Save to database (would need to implement table for equity curves)
            # For now, just log
            logger.debug(f"Saved equity point: {equity_point.timestamp} - {equity_point.equity_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving equity point: {e}")
    
    def export_equity_curve(
        self,
        format: str = "csv",
        include_statistics: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """
        Export equity curve data.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            include_statistics: Include statistics summary
            save_path: Custom save path
            
        Returns:
            Path to exported file
        """
        if not self.equity_points:
            logger.warning("No equity curve data to export")
            return ""
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"equity_curve_{timestamp}.{format}"
        
        # Get data
        equity_df = self.get_equity_dataframe()
        returns_df = self.get_returns_dataframe()
        drawdown_df = self.get_drawdown_dataframe()
        stats = self.calculate_statistics()
        
        if format.lower() == "csv":
            # Export to CSV
            equity_df.to_csv(f"equity_curve_{save_path}")
            returns_df.to_csv(f"returns_{save_path}")
            drawdown_df.to_csv(f"drawdown_{save_path}")
            
            if include_statistics:
                with open(f"statistics_{save_path.replace('.csv', '.json')}", 'w') as f:
                    json.dump(stats.__dict__, f, indent=2, default=str)
        
        elif format.lower() == "json":
            # Export to JSON
            export_data = {
                'equity_curve': equity_df.to_dict('records'),
                'returns': returns_df.to_dict('records'),
                'drawdown': drawdown_df.to_dict('records'),
                'statistics': stats.__dict__
            }
            
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == "excel":
            # Export to Excel
            with pd.ExcelWriter(save_path) as writer:
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
                returns_df.to_excel(writer, sheet_name='Returns', index=False)
                drawdown_df.to_excel(writer, sheet_name='Drawdown', index=False)
                
                if include_statistics:
                    stats_df = pd.DataFrame([stats.__dict__])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        logger.info(f"Equity curve exported to {save_path}")
        return save_path
    
    def compare_with_benchmark(
        self,
        benchmark_returns: List[float],
        benchmark_name: str = "Benchmark"
    ) -> Dict[str, Any]:
        """
        Compare equity curve with benchmark.
        
        Args:
            benchmark_returns: List of benchmark returns
            benchmark_name: Name of benchmark
            
        Returns:
            Comparison analysis
        """
        if not self.equity_points or not benchmark_returns:
            return {}
        
        # Align benchmark with equity curve timeline
        min_length = min(len(self.daily_returns), len(benchmark_returns))
        aligned_returns = self.daily_returns[:min_length]
        aligned_benchmark = benchmark_returns[:min_length]
        
        # Calculate comparison metrics
        equity_cumulative = np.cumprod([1 + r for r in aligned_returns]) - 1
        benchmark_cumulative = np.cumprod([1 + r for r in aligned_benchmark]) - 1
        
        # Calculate tracking error
        tracking_error = np.std(np.array(equity_cumulative) - np.array(benchmark_cumulative))
        
        # Calculate information ratio
        excess_returns = np.array(aligned_returns) - np.array(aligned_benchmark)
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        
        # Calculate beta
        if np.std(aligned_benchmark) > 0:
            covariance = np.cov(np.array(aligned_returns), np.array(aligned_benchmark))[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance
        else:
            beta = 0.0
        
        # Calculate upside/downside capture
        up_markets = np.array(aligned_benchmark) > 0
        down_markets = np.array(aligned_benchmark) < 0
        
        upside_capture = np.sum(np.array(aligned_returns)[up_markets]) / np.sum(np.array(aligned_benchmark)[up_markets]) if np.sum(up_markets) > 0 else 0.0
        downside_capture = np.sum(np.array(aligned_returns)[down_markets]) / np.sum(np.array(aligned_benchmark)[down_markets]) if np.sum(down_markets) > 0 else 0.0
        
        return {
            'benchmark_name': benchmark_name,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'equity_total_return': equity_cumulative[-1] if len(equity_cumulative) > 0 else 0.0,
            'benchmark_total_return': benchmark_cumulative[-1] if len(benchmark_cumulative) > 0 else 0.0,
            'outperformance': equity_cumulative[-1] - benchmark_cumulative[-1] if len(equity_cumulative) > 0 and len(benchmark_cumulative) > 0 else 0.0
        }
    
    def reset_equity_curve(self, preserve_config: bool = True) -> None:
        """Reset equity curve data."""
        self.equity_points.clear()
        self.daily_returns.clear()
        self.high_watermark = self.config.starting_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_start = None
        self.max_drawdown_end = None
        self.current_drawdown_start = None
        
        # Reset benchmark
        self.benchmark_returns = self.config.benchmark_returns or []
        
        # Clear cache
        self._statistics_cache = None
        self._cache_timestamp = None
        
        # Add initial point
        if preserve_config:
            self._add_initial_point()
        
        logger.info("Equity curve reset")


# Convenience functions
def create_equity_curve_tracker(
    pnl_tracker: PnLTracker,
    config: Optional[EquityCurveConfig] = None,
    persistence_manager: Optional[DataPersistenceManager] = None
) -> EquityCurveTracker:
    """Create equity curve tracker with default settings."""
    return EquityCurveTracker(pnl_tracker, config, persistence_manager)


def create_equity_curve_config(
    starting_capital: float = 100000.0,
    include_benchmark: bool = False,
    benchmark_returns: Optional[List[float]] = None
) -> EquityCurveConfig:
    """Create equity curve configuration."""
    return EquityCurveConfig(
        starting_capital=starting_capital,
        include_benchmark=include_benchmark,
        benchmark_returns=benchmark_returns
    )
