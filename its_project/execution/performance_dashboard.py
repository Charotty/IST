#!/usr/bin/env python3
"""
Real-time Performance Dashboard
=============================

Interactive performance dashboard with real-time updates, alerts,
and comprehensive visualization capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import threading
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .performance_metrics import (
    AdvancedPerformanceMetrics, PerformanceDashboard,
    SharpeRatioMetrics, DrawdownMetrics, WinRateMetrics
)
from .pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for performance dashboard."""
    update_interval: int = 60  # seconds
    max_history_points: int = 1000
    chart_width: int = 1200
    chart_height: int = 800
    theme: str = "plotly_white"
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    alert_enabled: bool = True
    auto_save: bool = True
    save_interval: int = 300  # seconds


class RealTimePerformanceDashboard:
    """
    Real-time performance dashboard with interactive visualizations.
    
    Features:
    - Live performance metrics updates
    - Interactive charts with Plotly
    - Alert system with configurable thresholds
    - Historical data tracking
    - Export capabilities
    - Multi-strategy support
    """
    
    def __init__(
        self,
        pnl_tracker: PnLTracker,
        config: Optional[DashboardConfig] = None,
        benchmark_returns: Optional[List[float]] = None
    ) -> None:
        self.pnl_tracker = pnl_tracker
        self.config = config or DashboardConfig()
        
        # Performance calculator
        self.metrics_calculator = AdvancedPerformanceMetrics(
            benchmark_returns=benchmark_returns,
            confidence_level=0.95
        )
        
        # Performance dashboard
        self.performance_dashboard = PerformanceDashboard(
            metrics_calculator=self.metrics_calculator,
            alert_thresholds={
                'min_sharpe': 0.5,
                'max_drawdown': 0.15,
                'min_win_rate': 0.45,
                'min_profit_factor': 1.3
            }
        )
        
        # Data storage
        self.historical_metrics: List[Dict[str, Any]] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.returns_history: List[Tuple[datetime, float]] = []
        
        # Dashboard state
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        self.last_update: Optional[datetime] = None
        
        # Callbacks
        self.update_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Initialize data
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Initialize dashboard with existing data."""
        # Get historical data from PnL tracker
        pnl_records = self.pnl_tracker.pnl_records
        
        if pnl_records:
            # Create equity curve
            cumulative_pnl = 0.0
            for record in pnl_records:
                cumulative_pnl += record.realized_pnl
                self.equity_history.append((record.exit_time or datetime.now(), cumulative_pnl))
            
            # Calculate returns
            if len(self.equity_history) > 1:
                for i in range(1, len(self.equity_history)):
                    prev_value = self.equity_history[i-1][1]
                    curr_value = self.equity_history[i][1]
                    if prev_value != 0:
                        return_rate = (curr_value - prev_value) / abs(prev_value)
                        self.returns_history.append((self.equity_history[i][0], return_rate))
        
        # Get trade PnLs
        self.trade_pnls = [record.realized_pnl for record in pnl_records]
        
        logger.info(f"Initialized dashboard with {len(pnl_records)} historical trades")
    
    async def start(self) -> None:
        """Start the real-time dashboard."""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time performance dashboard")
        
        # Start update task
        self.update_task = asyncio.create_task(self._update_loop())
        
        # Schedule initial update
        await self._update_metrics()
    
    async def stop(self) -> None:
        """Stop the real-time dashboard."""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time performance dashboard")
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Final save
        if self.config.auto_save:
            self._save_dashboard_data()
    
    async def _update_loop(self) -> None:
        """Main update loop for the dashboard."""
        while self.is_running:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Get current data
            current_pnl = self.pnl_tracker.get_cumulative_pnl()
            current_time = datetime.now()
            
            # Update equity history
            if not self.equity_history or current_time > self.equity_history[-1][0]:
                self.equity_history.append((current_time, current_pnl))
                
                # Calculate return
                if len(self.equity_history) > 1:
                    prev_value = self.equity_history[-2][1]
                    if prev_value != 0:
                        return_rate = (current_pnl - prev_value) / abs(prev_value)
                        self.returns_history.append((current_time, return_rate))
                
                # Limit history size
                if len(self.equity_history) > self.config.max_history_points:
                    self.equity_history = self.equity_history[-self.config.max_history_points:]
                if len(self.returns_history) > self.config.max_history_points:
                    self.returns_history = self.returns_history[-self.config.max_history_points:]
            
            # Prepare data for metrics calculation
            returns = [r[1] for r in self.returns_history]
            equity_curve = [e[1] for e in self.equity_history]
            timestamps = [e[0] for e in self.equity_history]
            
            # Update performance metrics
            metrics = self.performance_dashboard.update_metrics(
                returns=returns,
                equity_curve=equity_curve,
                trade_pnls=self.trade_pnls,
                timestamps=timestamps
            )
            
            # Store in history
            self.historical_metrics.append({
                'timestamp': current_time,
                'metrics': metrics
            })
            
            # Limit history
            if len(self.historical_metrics) > self.config.max_history_points:
                self.historical_metrics = self.historical_metrics[-self.config.max_history_points:]
            
            self.last_update = current_time
            
            # Call update callbacks
            for callback in self.update_callbacks:
                try:
                    await callback(metrics)
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")
            
            # Auto-save if needed
            if (self.config.auto_save and 
                current_time.timestamp() % self.config.save_interval < self.config.update_interval):
                self._save_dashboard_data()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def create_equity_chart(self) -> go.Figure:
        """Create interactive equity curve chart."""
        if not self.equity_history:
            return go.Figure()
        
        df = pd.DataFrame(self.equity_history, columns=['timestamp', 'pnl'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(color=self.config.color_palette[0], width=2)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Drawdown
        if len(self.historical_metrics) > 0 and 'drawdown' in self.historical_metrics[-1]['metrics']:
            drawdown_data = []
            for entry in self.historical_metrics:
                if 'drawdown' in entry['metrics']:
                    drawdown_data.append({
                        'timestamp': entry['timestamp'],
                        'drawdown': entry['metrics']['drawdown'].current_drawdown * 100
                    })
            
            if drawdown_data:
                dd_df = pd.DataFrame(drawdown_data)
                fig.add_trace(
                    go.Scatter(
                        x=dd_df['timestamp'],
                        y=dd_df['drawdown'],
                        mode='lines',
                        name='Drawdown',
                        fill='tonexty',
                        line=dict(color='red', width=1),
                        fillcolor='rgba(255,0,0,0.2)'
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Real-time Performance Dashboard",
            template=self.config.theme,
            height=self.config.chart_height,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="PnL", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_metrics_chart(self) -> go.Figure:
        """Create performance metrics summary chart."""
        if not self.historical_metrics:
            return go.Figure()
        
        # Get latest metrics
        latest_metrics = self.historical_metrics[-1]['metrics']
        
        # Create subplots for different metric categories
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk-Adjusted Returns', 'Drawdown Analysis', 
                          'Win Rate Analysis', 'Trade Statistics'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Sharpe ratio indicator
        if 'sharpe_ratio' in latest_metrics:
            sharpe = latest_metrics['sharpe_ratio']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=sharpe.sharpe_ratio_annualized,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Annualized Sharpe"},
                    gauge={'axis': {'range': [None, 3]},
                          'bar': {'color': self.config.color_palette[0]},
                          'steps': [
                              {'range': [0, 1], 'color': "lightgray"},
                              {'range': [1, 2], 'color': "gray"},
                              {'range': [2, 3], 'color': "lightgreen"}
                          ],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 2}}
                ),
                row=1, col=1
            )
        
        # Max drawdown indicator
        if 'drawdown' in latest_metrics:
            dd = latest_metrics['drawdown']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=dd.max_drawdown * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Max Drawdown (%)"},
                    gauge={'axis': {'range': [None, 50]},
                          'bar': {'color': self.config.color_palette[1]},
                          'steps': [
                              {'range': [0, 10], 'color': "lightgreen"},
                              {'range': [10, 20], 'color': "yellow"},
                              {'range': [20, 50], 'color': "lightcoral"}
                          ]}
                ),
                row=1, col=2
            )
        
        # Win rate and profit factor bars
        if 'win_rate' in latest_metrics:
            wr = latest_metrics['win_rate']
            fig.add_trace(
                go.Bar(
                    x=['Win Rate', 'Profit Factor'],
                    y=[wr.win_rate * 100, wr.profit_factor],
                    name='Current',
                    marker_color=[self.config.color_palette[2], self.config.color_palette[3]]
                ),
                row=2, col=1
            )
        
        # Trade statistics
        if 'win_rate' in latest_metrics:
            wr = latest_metrics['win_rate']
            fig.add_trace(
                go.Bar(
                    x=['Total Trades', 'Winning', 'Losing'],
                    y=[wr.total_trades, wr.winning_trades, wr.losing_trades],
                    name='Trade Count',
                    marker_color=[self.config.color_palette[4], self.config.color_palette[5], 
                                self.config.color_palette[6]]
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Performance Metrics Summary",
            template=self.config.theme,
            height=self.config.chart_height,
            showlegend=False
        )
        
        return fig
    
    def create_rolling_metrics_chart(self, window: int = 50) -> go.Figure:
        """Create rolling metrics chart."""
        if len(self.historical_metrics) < window:
            return go.Figure()
        
        # Prepare data
        timestamps = []
        rolling_sharpe = []
        rolling_win_rate = []
        rolling_drawdown = []
        
        for i, entry in enumerate(self.historical_metrics):
            timestamps.append(entry['timestamp'])
            
            # Rolling Sharpe
            if 'sharpe_ratio' in entry['metrics']:
                rolling_sharpe.append(entry['metrics']['sharpe_ratio'].sharpe_ratio_annualized)
            else:
                rolling_sharpe.append(None)
            
            # Rolling win rate
            if 'win_rate' in entry['metrics']:
                rolling_win_rate.append(entry['metrics']['win_rate'].win_rate * 100)
            else:
                rolling_win_rate.append(None)
            
            # Rolling drawdown
            if 'drawdown' in entry['metrics']:
                rolling_drawdown.append(entry['metrics']['drawdown'].current_drawdown * 100)
            else:
                rolling_drawdown.append(None)
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Win Rate', 'Current Drawdown')
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_sharpe,
                mode='lines',
                name='Sharpe Ratio',
                line=dict(color=self.config.color_palette[0])
            ),
            row=1, col=1
        )
        
        # Rolling win rate
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_win_rate,
                mode='lines',
                name='Win Rate (%)',
                line=dict(color=self.config.color_palette[1])
            ),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_drawdown,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color=self.config.color_palette[2]),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Rolling Performance Metrics (Window: {window})",
            template=self.config.theme,
            height=self.config.chart_height,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        return fig
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        if not self.historical_metrics:
            return {}
        
        latest = self.historical_metrics[-1]['metrics']
        
        summary = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'total_pnl': self.pnl_tracker.get_cumulative_pnl(),
            'total_trades': len(self.pnl_tracker.pnl_records),
            'current_positions': len(self.pnl_tracker.positions),
            'risk_adjusted_returns': {
                'sharpe_ratio': latest['sharpe_ratio'].sharpe_ratio_annualized if 'sharpe_ratio' in latest else None,
                'sortino_ratio': latest['sharpe_ratio'].sharpe_ratio_sortino if 'sharpe_ratio' in latest else None,
                'information_ratio': latest['sharpe_ratio'].sharpe_ratio_information if 'sharpe_ratio' in latest else None,
                'treynor_ratio': latest['sharpe_ratio'].treynor_ratio if 'sharpe_ratio' in latest else None
            },
            'drawdown_analysis': {
                'max_drawdown': latest['drawdown'].max_drawdown if 'drawdown' in latest else None,
                'current_drawdown': latest['drawdown'].current_drawdown if 'drawdown' in latest else None,
                'max_duration': latest['drawdown'].max_drawdown_duration if 'drawdown' in latest else None,
                'ulcer_index': latest['drawdown'].ulcer_index if 'drawdown' in latest else None
            },
            'win_rate_analysis': {
                'win_rate': latest['win_rate'].win_rate if 'win_rate' in latest else None,
                'profit_factor': latest['win_rate'].profit_factor if 'win_rate' in latest else None,
                'expectancy': latest['win_rate'].expectancy if 'win_rate' in latest else None,
                'avg_win': latest['win_rate'].avg_win if 'win_rate' in latest else None,
                'avg_loss': latest['win_rate'].avg_loss if 'win_rate' in latest else None
            },
            'alerts': {
                'total_alerts': len(self.performance_dashboard.alerts_history),
                'recent_alerts': [
                    {
                        'timestamp': alert['timestamp'].isoformat(),
                        'type': alert['type'],
                        'level': alert['level'],
                        'message': alert['message']
                    }
                    for alert in self.performance_dashboard.alerts_history[-5:]
                ]
            }
        }
        
        return summary
    
    def add_update_callback(self, callback: Callable) -> None:
        """Add callback for metrics updates."""
        self.update_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def _save_dashboard_data(self) -> None:
        """Save dashboard data to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_data_{timestamp}.json"
            
            data = {
                'config': {
                    'update_interval': self.config.update_interval,
                    'max_history_points': self.config.max_history_points,
                    'theme': self.config.theme
                },
                'current_metrics': self.get_current_metrics_summary(),
                'historical_metrics': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'summary': entry['metrics']['summary'] if 'summary' in entry['metrics'] else {}
                    }
                    for entry in self.historical_metrics[-100:]  # Last 100 entries
                ],
                'equity_history': [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in self.equity_history[-1000:]  # Last 1000 points
                ],
                'alerts_history': [
                    {
                        'timestamp': alert['timestamp'].isoformat(),
                        'type': alert['type'],
                        'level': alert['level'],
                        'message': alert['message']
                    }
                    for alert in self.performance_dashboard.alerts_history[-50:]  # Last 50 alerts
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Dashboard data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving dashboard data: {e}")
    
    def export_charts(self, output_dir: str = "dashboard_charts") -> None:
        """Export all charts as HTML files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Equity chart
        equity_fig = self.create_equity_chart()
        equity_fig.write_html(f"{output_dir}/equity_chart_{timestamp}.html")
        
        # Metrics chart
        metrics_fig = self.create_metrics_chart()
        metrics_fig.write_html(f"{output_dir}/metrics_chart_{timestamp}.html")
        
        # Rolling metrics chart
        rolling_fig = self.create_rolling_metrics_chart()
        rolling_fig.write_html(f"{output_dir}/rolling_metrics_{timestamp}.html")
        
        logger.info(f"Charts exported to {output_dir}")


# Convenience functions
def create_real_time_dashboard(
    pnl_tracker: PnLTracker,
    config: Optional[DashboardConfig] = None,
    benchmark_returns: Optional[List[float]] = None
) -> RealTimePerformanceDashboard:
    """Create real-time performance dashboard with default settings."""
    return RealTimePerformanceDashboard(pnl_tracker, config, benchmark_returns)


async def run_dashboard_example():
    """Example of running the dashboard."""
    from .pnl_tracker import PnLTracker
    
    # Create PnL tracker and dashboard
    pnl_tracker = PnLTracker()
    dashboard = create_real_time_dashboard(pnl_tracker)
    
    # Add some sample data
    import random
    base_price = 50000
    for i in range(100):
        price_change = random.uniform(-0.02, 0.02)
        new_price = base_price * (1 + price_change)
        
        if i % 2 == 0:
            pnl_tracker.add_trade(f"buy_{i}", "BTC/USDT", "buy", 1.0, new_price)
        else:
            pnl_tracker.add_trade(f"sell_{i}", "BTC/USDT", "sell", 1.0, new_price)
        
        base_price = new_price
    
    # Start dashboard
    await dashboard.start()
    
    # Run for 5 minutes
    await asyncio.sleep(300)
    
    # Stop dashboard
    await dashboard.stop()
    
    # Export charts
    dashboard.export_charts()


if __name__ == "__main__":
    asyncio.run(run_dashboard_example())
