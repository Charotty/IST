#!/usr/bin/env python3
"""
Advanced Performance Metrics
==========================

Comprehensive performance metrics calculation including enhanced Sharpe ratio,
advanced drawdown analysis, and detailed win rate calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SharpeRatioMetrics:
    """Comprehensive Sharpe ratio analysis."""
    sharpe_ratio: float
    sharpe_ratio_annualized: float
    sharpe_ratio_rolling: List[float]
    sharpe_ratio_sortino: float
    sharpe_ratio_information: float
    treynor_ratio: float
    jensen_alpha: float
    beta: float
    tracking_error: float
    confidence_interval: Tuple[float, float]
    p_value: float


@dataclass
class DrawdownMetrics:
    """Advanced drawdown analysis."""
    max_drawdown: float
    max_drawdown_duration: int
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    current_drawdown: float
    current_drawdown_duration: int
    current_drawdown_start: Optional[datetime]
    average_drawdown: float
    drawdown_frequency: float
    recovery_time_avg: float
    time_under_water: float
    drawdown_volatility: float
    drawdown_distribution: Dict[str, float]
    pain_index: float
    ulcer_index: float
    martin_ratio: float


@dataclass
class WinRateMetrics:
    """Detailed win rate analysis."""
    win_rate: float
    win_rate_rolling: List[float]
    winning_trades: int
    losing_trades: int
    total_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    z_score: float
    p_value: float
    trade_distribution: Dict[str, float]
    consecutive_wins_max: int
    consecutive_losses_max: int
    avg_consecutive_wins: float
    avg_consecutive_losses: float


class AdvancedPerformanceMetrics:
    """
    Advanced performance metrics calculator with enhanced statistical analysis.
    
    Features:
    - Multiple Sharpe ratio calculation methods
    - Advanced drawdown analysis with duration tracking
    - Comprehensive win rate analysis with confidence intervals
    - Statistical significance testing
    - Rolling window analysis
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        benchmark_returns: Optional[List[float]] = None,
        confidence_level: float = 0.95
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns or []
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Statistical constants
        self.z_critical = stats.norm.ppf(1 - self.alpha / 2)
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        method: str = "simple",
        rolling_window: int = 252,
        annualization_factor: int = 252
    ) -> SharpeRatioMetrics:
        """
        Calculate comprehensive Sharpe ratio metrics.
        
        Args:
            returns: List of period returns
            method: Calculation method ('simple', 'rolling', 'sortino', 'information')
            rolling_window: Window size for rolling calculations
            annualization_factor: Factor for annualizing (252 for daily, 12 for monthly)
            
        Returns:
            SharpeRatioMetrics object with comprehensive analysis
        """
        if not returns:
            logger.warning("No returns provided for Sharpe ratio calculation")
            return SharpeRatioMetrics(0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 1.0)
        
        returns_array = np.array(returns)
        n = len(returns_array)
        
        # Basic statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # Simple Sharpe ratio
        if std_return > 0:
            sharpe_simple = (mean_return - self.risk_free_rate / annualization_factor) / std_return
        else:
            sharpe_simple = 0.0
        
        # Annualized Sharpe ratio
        sharpe_annualized = sharpe_simple * np.sqrt(annualization_factor)
        
        # Rolling Sharpe ratio
        sharpe_rolling = []
        if len(returns_array) >= rolling_window:
            for i in range(rolling_window, len(returns_array) + 1):
                window_returns = returns_array[i-rolling_window:i]
                window_mean = np.mean(window_returns)
                window_std = np.std(window_returns, ddof=1)
                
                if window_std > 0:
                    rolling_sharpe = (window_mean - self.risk_free_rate / annualization_factor) / window_std
                    sharpe_rolling.append(rolling_sharpe * np.sqrt(annualization_factor))
                else:
                    sharpe_rolling.append(0.0)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns, ddof=1)
            if downside_deviation > 0:
                sortino_ratio = (mean_return - self.risk_free_rate / annualization_factor) / downside_deviation
                sortino_ratio *= np.sqrt(annualization_factor)
            else:
                sortino_ratio = float('inf') if mean_return > 0 else 0.0
        else:
            sortino_ratio = float('inf') if mean_return > 0 else 0.0
        
        # Information ratio (vs benchmark)
        if self.benchmark_returns and len(self.benchmark_returns) == len(returns_array):
            benchmark_array = np.array(self.benchmark_returns)
            excess_returns = returns_array - benchmark_array
            
            if len(excess_returns) > 1:
                tracking_error = np.std(excess_returns, ddof=1)
                if tracking_error > 0:
                    information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(annualization_factor)
                else:
                    information_ratio = 0.0
            else:
                information_ratio = 0.0
                tracking_error = 0.0
        else:
            information_ratio = 0.0
            tracking_error = 0.0
        
        # Treynor ratio (requires beta calculation)
        if self.benchmark_returns and len(self.benchmark_returns) == len(returns_array):
            benchmark_array = np.array(self.benchmark_returns)
            if len(benchmark_array) > 1 and np.std(benchmark_array, ddof=1) > 0:
                covariance = np.cov(returns_array, benchmark_array)[0, 1]
                benchmark_variance = np.var(benchmark_array, ddof=1)
                beta = covariance / benchmark_variance
                
                if beta != 0:
                    treynor_ratio = (mean_return - self.risk_free_rate / annualization_factor) / beta * annualization_factor
                else:
                    treynor_ratio = 0.0
            else:
                beta = 0.0
                treynor_ratio = 0.0
        else:
            beta = 0.0
            treynor_ratio = 0.0
        
        # Jensen's alpha
        if self.benchmark_returns and len(self.benchmark_returns) == len(returns_array):
            benchmark_array = np.array(self.benchmark_returns)
            benchmark_mean = np.mean(benchmark_array)
            jensen_alpha = (mean_return - self.risk_free_rate / annualization_factor) - \
                          beta * (benchmark_mean - self.risk_free_rate / annualization_factor)
            jensen_alpha *= annualization_factor
        else:
            jensen_alpha = 0.0
        
        # Statistical significance
        if n > 1 and std_return > 0:
            # Standard error of Sharpe ratio
            se_sharpe = np.sqrt((1 + 0.5 * sharpe_simple**2) / n)
            
            # Confidence interval
            ci_lower = sharpe_simple - self.z_critical * se_sharpe
            ci_upper = sharpe_simple + self.z_critical * se_sharpe
            
            # P-value for testing if Sharpe > 0
            z_score = sharpe_simple / se_sharpe
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            ci_lower = ci_upper = 0.0
            z_score = 0.0
            p_value = 1.0
        
        return SharpeRatioMetrics(
            sharpe_ratio=sharpe_simple,
            sharpe_ratio_annualized=sharpe_annualized,
            sharpe_ratio_rolling=sharpe_rolling,
            sharpe_ratio_sortino=sortino_ratio,
            sharpe_ratio_information=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            beta=beta,
            tracking_error=tracking_error,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value
        )
    
    def calculate_drawdown_metrics(
        self,
        equity_curve: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> DrawdownMetrics:
        """
        Calculate advanced drawdown metrics.
        
        Args:
            equity_curve: List of equity values
            timestamps: List of corresponding timestamps
            
        Returns:
            DrawdownMetrics object with comprehensive analysis
        """
        if not equity_curve:
            logger.warning("No equity curve data provided for drawdown analysis")
            return DrawdownMetrics(0.0, 0, datetime.now(), datetime.now(), 0.0, 0, None, 
                                0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0, 0.0)
        
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(minutes=i) for i in range(len(equity_curve))]
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown series
        drawdown_series = (equity_array - running_max) / running_max
        
        # Current drawdown
        current_drawdown = drawdown_series[-1]
        current_drawdown_duration = 0
        current_drawdown_start = None
        
        # Find current drawdown start
        for i in range(len(drawdown_series) - 1, -1, -1):
            if drawdown_series[i] == 0:
                break
            current_drawdown_duration += 1
            if current_drawdown_start is None:
                current_drawdown_start = timestamps[i]
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown_series)
        max_drawdown_idx = np.argmin(drawdown_series)
        max_drawdown_start_idx = np.where(
            equity_array[:max_drawdown_idx] == np.max(equity_array[:max_drawdown_idx])
        )[0][-1] if max_drawdown_idx > 0 else 0
        
        max_drawdown_start = timestamps[max_drawdown_start_idx]
        max_drawdown_end = timestamps[max_drawdown_idx]
        max_drawdown_duration = max_drawdown_idx - max_drawdown_start_idx
        
        # Drawdown statistics
        drawdown_periods = []
        in_drawdown = False
        drawdown_start_idx = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append((drawdown_start_idx, i))
        
        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_periods.append((drawdown_start_idx, len(drawdown_series) - 1))
        
        # Calculate drawdown metrics
        if drawdown_periods:
            drawdown_magnitudes = []
            drawdown_durations = []
            recovery_times = []
            
            for start_idx, end_idx in drawdown_periods:
                magnitude = abs(np.min(drawdown_series[start_idx:end_idx + 1]))
                duration = end_idx - start_idx
                drawdown_magnitudes.append(magnitude)
                drawdown_durations.append(duration)
                
                # Recovery time (time to return to previous high)
                if end_idx < len(equity_array) - 1:
                    recovery_value = equity_array[start_idx]
                    recovery_idx = None
                    for j in range(end_idx + 1, len(equity_array)):
                        if equity_array[j] >= recovery_value:
                            recovery_idx = j
                            break
                    
                    if recovery_idx:
                        recovery_times.append(recovery_idx - end_idx)
            
            average_drawdown = np.mean(drawdown_magnitudes)
            drawdown_frequency = len(drawdown_periods) / len(equity_array)
            recovery_time_avg = np.mean(recovery_times) if recovery_times else 0
            
            # Time under water
            time_under_water = sum(drawdown_durations) / len(equity_array)
            
            # Drawdown volatility
            drawdown_volatility = np.std(drawdown_magnitudes) if len(drawdown_magnitudes) > 1 else 0
            
            # Drawdown distribution
            drawdown_distribution = {
                'min': np.min(drawdown_magnitudes),
                'max': np.max(drawdown_magnitudes),
                'mean': average_drawdown,
                'median': np.median(drawdown_magnitudes),
                'std': drawdown_volatility,
                'q25': np.percentile(drawdown_magnitudes, 25),
                'q75': np.percentile(drawdown_magnitudes, 75)
            }
        else:
            average_drawdown = 0.0
            drawdown_frequency = 0.0
            recovery_time_avg = 0.0
            time_under_water = 0.0
            drawdown_volatility = 0.0
            drawdown_distribution = {}
        
        # Pain index (average of drawdowns)
        pain_index = np.mean(np.abs(drawdown_series[drawdown_series < 0])) if np.any(drawdown_series < 0) else 0.0
        
        # Ulcer index
        ulcer_index = np.sqrt(np.mean(drawdown_series**2))
        
        # Martin ratio (return / ulcer_index)
        if len(equity_array) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            annualized_return = np.mean(returns) * 252
            martin_ratio = annualized_return / ulcer_index if ulcer_index > 0 else 0.0
        else:
            martin_ratio = 0.0
        
        return DrawdownMetrics(
            max_drawdown=abs(max_drawdown),
            max_drawdown_duration=max_drawdown_duration,
            max_drawdown_start=max_drawdown_start,
            max_drawdown_end=max_drawdown_end,
            current_drawdown=abs(current_drawdown),
            current_drawdown_duration=current_drawdown_duration,
            current_drawdown_start=current_drawdown_start,
            average_drawdown=average_drawdown,
            drawdown_frequency=drawdown_frequency,
            recovery_time_avg=recovery_time_avg,
            time_under_water=time_under_water,
            drawdown_volatility=drawdown_volatility,
            drawdown_distribution=drawdown_distribution,
            pain_index=pain_index,
            ulcer_index=ulcer_index,
            martin_ratio=martin_ratio
        )
    
    def calculate_win_rate_metrics(
        self,
        trade_pnls: List[float],
        rolling_window: int = 50
    ) -> WinRateMetrics:
        """
        Calculate comprehensive win rate metrics.
        
        Args:
            trade_pnls: List of individual trade PnL values
            rolling_window: Window size for rolling win rate calculation
            
        Returns:
            WinRateMetrics object with detailed analysis
        """
        if not trade_pnls:
            logger.warning("No trade PnL data provided for win rate analysis")
            return WinRateMetrics(0.0, [], 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                               0.0, (0.0, 0.0), 0.0, 1.0, {}, 0, 0, 0.0, 0.0)
        
        pnls_array = np.array(trade_pnls)
        total_trades = len(pnls_array)
        
        # Basic win/loss analysis
        winning_trades = np.sum(pnls_array > 0)
        losing_trades = np.sum(pnls_array < 0)
        breakeven_trades = np.sum(pnls_array == 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Win/loss statistics
        winning_pnls = pnls_array[pnls_array > 0]
        losing_pnls = pnls_array[pnls_array < 0]
        
        avg_win = np.mean(winning_pnls) if len(winning_pnls) > 0 else 0.0
        avg_loss = np.mean(losing_pnls) if len(losing_pnls) > 0 else 0.0
        largest_win = np.max(winning_pnls) if len(winning_pnls) > 0 else 0.0
        largest_loss = np.min(losing_pnls) if len(losing_pnls) > 0 else 0.0
        
        # Profit factor
        total_wins = np.sum(winning_pnls) if len(winning_pnls) > 0 else 0.0
        total_losses = abs(np.sum(losing_pnls)) if len(losing_pnls) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Rolling win rate
        win_rate_rolling = []
        if len(pnls_array) >= rolling_window:
            for i in range(rolling_window, len(pnls_array) + 1):
                window_pnls = pnls_array[i-rolling_window:i]
                window_wins = np.sum(window_pnls > 0)
                window_win_rate = window_wins / rolling_window
                win_rate_rolling.append(window_win_rate)
        
        # Statistical significance
        if total_trades > 1:
            # Standard error of win rate
            se_win_rate = np.sqrt((win_rate * (1 - win_rate)) / total_trades)
            
            # Confidence interval
            ci_lower = win_rate - self.z_critical * se_win_rate
            ci_upper = win_rate + self.z_critical * se_win_rate
            
            # Z-score and p-value for testing if win rate > 0.5
            null_hypothesis = 0.5  # 50% win rate
            z_score = (win_rate - null_hypothesis) / se_win_rate
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            se_win_rate = 0.0
            ci_lower = ci_upper = win_rate
            z_score = 0.0
            p_value = 1.0
        
        # Trade distribution
        trade_distribution = {
            'min': np.min(pnls_array),
            'max': np.max(pnls_array),
            'mean': np.mean(pnls_array),
            'median': np.median(pnls_array),
            'std': np.std(pnls_array, ddof=1),
            'skewness': stats.skew(pnls_array) if len(pnls_array) > 2 else 0.0,
            'kurtosis': stats.kurtosis(pnls_array) if len(pnls_array) > 3 else 0.0,
            'q25': np.percentile(pnls_array, 25),
            'q75': np.percentile(pnls_array, 75)
        }
        
        # Consecutive wins/losses analysis
        consecutive_wins = []
        consecutive_losses = []
        current_streak = 0
        current_type = None  # 'win' or 'loss'
        
        for pnl in pnls_array:
            if pnl > 0:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_type == 'loss':
                        consecutive_losses.append(current_streak)
                    current_streak = 1
                    current_type = 'win'
            elif pnl < 0:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_type == 'win':
                        consecutive_wins.append(current_streak)
                    current_streak = 1
                    current_type = 'loss'
            else:  # breakeven
                if current_type == 'win':
                    consecutive_wins.append(current_streak)
                elif current_type == 'loss':
                    consecutive_losses.append(current_streak)
                current_streak = 0
                current_type = None
        
        # Add final streak
        if current_type == 'win':
            consecutive_wins.append(current_streak)
        elif current_type == 'loss':
            consecutive_losses.append(current_streak)
        
        max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
        avg_consecutive_wins = np.mean(consecutive_wins) if consecutive_wins else 0.0
        avg_consecutive_losses = np.mean(consecutive_losses) if consecutive_losses else 0.0
        
        return WinRateMetrics(
            win_rate=win_rate,
            win_rate_rolling=win_rate_rolling,
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            total_trades=total_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            standard_error=se_win_rate,
            confidence_interval=(ci_lower, ci_upper),
            z_score=z_score,
            p_value=p_value,
            trade_distribution=trade_distribution,
            consecutive_wins_max=max_consecutive_wins,
            consecutive_losses_max=max_consecutive_losses,
            avg_consecutive_wins=avg_consecutive_wins,
            avg_consecutive_losses=avg_consecutive_losses
        )
    
    def calculate_comprehensive_metrics(
        self,
        returns: List[float],
        equity_curve: List[float],
        trade_pnls: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics comprehensively.
        
        Args:
            returns: List of period returns
            equity_curve: List of equity values
            trade_pnls: List of individual trade PnL values
            timestamps: List of corresponding timestamps
            
        Returns:
            Dictionary containing all performance metrics
        """
        sharpe_metrics = self.calculate_sharpe_ratio(returns)
        drawdown_metrics = self.calculate_drawdown_metrics(equity_curve, timestamps)
        win_rate_metrics = self.calculate_win_rate_metrics(trade_pnls)
        
        return {
            'sharpe_ratio': sharpe_metrics,
            'drawdown': drawdown_metrics,
            'win_rate': win_rate_metrics,
            'summary': {
                'overall_sharpe': sharpe_metrics.sharpe_ratio_annualized,
                'max_drawdown': drawdown_metrics.max_drawdown,
                'win_rate': win_rate_metrics.win_rate,
                'profit_factor': win_rate_metrics.profit_factor,
                'expectancy': win_rate_metrics.expectancy,
                'ulcer_index': drawdown_metrics.ulcer_index,
                'martin_ratio': drawdown_metrics.martin_ratio,
                'sortino_ratio': sharpe_metrics.sharpe_ratio_sortino,
                'information_ratio': sharpe_metrics.sharpe_ratio_information
            }
        }


class PerformanceDashboard:
    """
    Real-time performance dashboard with alerts and monitoring.
    """
    
    def __init__(
        self,
        metrics_calculator: AdvancedPerformanceMetrics,
        alert_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        self.metrics_calculator = metrics_calculator
        self.alert_thresholds = alert_thresholds or {
            'min_sharpe': 0.5,
            'max_drawdown': 0.2,
            'min_win_rate': 0.4,
            'min_profit_factor': 1.2
        }
        
        # Historical data
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts_history: List[Dict[str, Any]] = []
        
        # Current state
        self.current_metrics: Optional[Dict[str, Any]] = None
        self.last_update: Optional[datetime] = None
    
    def update_metrics(
        self,
        returns: List[float],
        equity_curve: List[float],
        trade_pnls: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Update performance metrics and check for alerts.
        
        Args:
            returns: List of period returns
            equity_curve: List of equity values
            trade_pnls: List of individual trade PnL values
            timestamps: List of corresponding timestamps
            
        Returns:
            Updated metrics dictionary
        """
        # Calculate new metrics
        self.current_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            returns, equity_curve, trade_pnls, timestamps
        )
        self.last_update = datetime.now()
        
        # Store in history
        self.metrics_history.append({
            'timestamp': self.last_update,
            'metrics': self.current_metrics
        })
        
        # Check for alerts
        self._check_alerts()
        
        return self.current_metrics
    
    def _check_alerts(self) -> None:
        """Check for performance alerts based on thresholds."""
        if not self.current_metrics:
            return
        
        summary = self.current_metrics['summary']
        alerts = []
        
        # Sharpe ratio alert
        if summary['overall_sharpe'] < self.alert_thresholds['min_sharpe']:
            alerts.append({
                'type': 'sharpe_ratio',
                'level': 'warning',
                'message': f"Sharpe ratio ({summary['overall_sharpe']:.2f}) below threshold ({self.alert_thresholds['min_sharpe']:.2f})",
                'timestamp': datetime.now()
            })
        
        # Drawdown alert
        if summary['max_drawdown'] > self.alert_thresholds['max_drawdown']:
            alerts.append({
                'type': 'drawdown',
                'level': 'critical',
                'message': f"Max drawdown ({summary['max_drawdown']:.1%}) exceeds threshold ({self.alert_thresholds['max_drawdown']:.1%})",
                'timestamp': datetime.now()
            })
        
        # Win rate alert
        if summary['win_rate'] < self.alert_thresholds['min_win_rate']:
            alerts.append({
                'type': 'win_rate',
                'level': 'warning',
                'message': f"Win rate ({summary['win_rate']:.1%}) below threshold ({self.alert_thresholds['min_win_rate']:.1%})",
                'timestamp': datetime.now()
            })
        
        # Profit factor alert
        if summary['profit_factor'] < self.alert_thresholds['min_profit_factor']:
            alerts.append({
                'type': 'profit_factor',
                'level': 'warning',
                'message': f"Profit factor ({summary['profit_factor']:.2f}) below threshold ({self.alert_thresholds['min_profit_factor']:.2f})",
                'timestamp': datetime.now()
            })
        
        # Store alerts
        self.alerts_history.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance Alert [{alert['level'].upper()}]: {alert['message']}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current and historical metrics."""
        if not self.current_metrics:
            return {}
        
        # Calculate trends
        if len(self.metrics_history) >= 2:
            current = self.metrics_history[-1]['metrics']['summary']
            previous = self.metrics_history[-2]['metrics']['summary']
            
            trends = {
                'sharpe_trend': current['overall_sharpe'] - previous['overall_sharpe'],
                'drawdown_trend': current['max_drawdown'] - previous['max_drawdown'],
                'win_rate_trend': current['win_rate'] - previous['win_rate'],
                'profit_factor_trend': current['profit_factor'] - previous['profit_factor']
            }
        else:
            trends = {}
        
        return {
            'current_metrics': self.current_metrics,
            'last_update': self.last_update,
            'trends': trends,
            'recent_alerts': self.alerts_history[-10:],  # Last 10 alerts
            'total_alerts': len(self.alerts_history),
            'metrics_history_count': len(self.metrics_history)
        }
    
    def export_dashboard_data(self, filename: str) -> None:
        """Export dashboard data to JSON file."""
        import json
        
        dashboard_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_metrics': self.current_metrics,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'metrics_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'summary': entry['metrics']['summary']
                }
                for entry in self.metrics_history[-100:]  # Last 100 entries
            ],
            'alerts_history': [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'type': alert['type'],
                    'level': alert['level'],
                    'message': alert['message']
                }
                for alert in self.alerts_history[-50:]  # Last 50 alerts
            ],
            'alert_thresholds': self.alert_thresholds
        }
        
        with open(filename, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Dashboard data exported to {filename}")


# Convenience functions
def create_performance_metrics(
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[List[float]] = None,
    confidence_level: float = 0.95
) -> AdvancedPerformanceMetrics:
    """Create AdvancedPerformanceMetrics with default settings."""
    return AdvancedPerformanceMetrics(risk_free_rate, benchmark_returns, confidence_level)


def create_performance_dashboard(
    metrics_calculator: AdvancedPerformanceMetrics,
    alert_thresholds: Optional[Dict[str, float]] = None
) -> PerformanceDashboard:
    """Create PerformanceDashboard with default settings."""
    return PerformanceDashboard(metrics_calculator, alert_thresholds)
