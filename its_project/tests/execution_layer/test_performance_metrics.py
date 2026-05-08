#!/usr/bin/env python3
"""
Tests for Advanced Performance Metrics
===================================

Comprehensive tests for enhanced Sharpe ratio, drawdown analysis,
and win rate calculations.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

from its_project.execution.performance_metrics import (
    AdvancedPerformanceMetrics, PerformanceDashboard,
    SharpeRatioMetrics, DrawdownMetrics, WinRateMetrics,
    create_performance_metrics, create_performance_dashboard
)
from its_project.execution.pnl_tracker import PnLTracker


class TestAdvancedPerformanceMetrics:
    """Test cases for AdvancedPerformanceMetrics class."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create AdvancedPerformanceMetrics instance for testing."""
        return AdvancedPerformanceMetrics(risk_free_rate=0.02, confidence_level=0.95)
    
    @pytest.fixture
    def sample_returns(self):
        """Sample returns data for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252).tolist()  # Daily returns for 1 year
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Sample equity curve for testing."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        equity = [100000]  # Starting capital
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        return equity
    
    @pytest.fixture
    def sample_trade_pnls(self):
        """Sample trade PnL data for testing."""
        np.random.seed(42)
        pnls = []
        for i in range(50):
            if np.random.random() > 0.4:  # 60% win rate
                pnls.append(np.random.uniform(50, 500))  # Winning trades
            else:
                pnls.append(np.random.uniform(-500, -50))  # Losing trades
        return pnls
    
    def test_init(self, metrics_calculator):
        """Test AdvancedPerformanceMetrics initialization."""
        assert metrics_calculator.risk_free_rate == 0.02
        assert metrics_calculator.confidence_level == 0.95
        assert metrics_calculator.alpha == 0.05
        assert metrics_calculator.z_critical == stats.norm.ppf(0.975)
    
    def test_calculate_sharpe_ratio_basic(self, metrics_calculator, sample_returns):
        """Test basic Sharpe ratio calculation."""
        sharpe_metrics = metrics_calculator.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe_metrics, SharpeRatioMetrics)
        assert sharpe_metrics.sharpe_ratio >= 0  # Should be positive with our seed
        assert sharpe_metrics.sharpe_ratio_annualized > sharpe_metrics.sharpe_ratio
        assert len(sharpe_metrics.sharpe_ratio_rolling) > 0
        assert sharpe_metrics.confidence_interval[0] <= sharpe_metrics.sharpe_ratio <= sharpe_metrics.confidence_interval[1]
        assert 0 <= sharpe_metrics.p_value <= 1
    
    def test_calculate_sharpe_ratio_with_benchmark(self, metrics_calculator, sample_returns):
        """Test Sharpe ratio calculation with benchmark."""
        benchmark_returns = np.random.normal(0.0005, 0.015, len(sample_returns)).tolist()
        metrics_calc_with_bench = AdvancedPerformanceMetrics(
            risk_free_rate=0.02,
            benchmark_returns=benchmark_returns
        )
        
        sharpe_metrics = metrics_calc_with_bench.calculate_sharpe_ratio(sample_returns)
        
        assert sharpe_metrics.sharpe_ratio_information != 0
        assert sharpe_metrics.beta != 0
        assert sharpe_metrics.treynor_ratio != 0
        assert sharpe_metrics.jensen_alpha != 0
        assert sharpe_metrics.tracking_error >= 0
    
    def test_calculate_sharpe_ratio_empty_data(self, metrics_calculator):
        """Test Sharpe ratio calculation with empty data."""
        sharpe_metrics = metrics_calculator.calculate_sharpe_ratio([])
        
        assert sharpe_metrics.sharpe_ratio == 0.0
        assert sharpe_metrics.sharpe_ratio_annualized == 0.0
        assert sharpe_metrics.sharpe_ratio_rolling == []
        assert sharpe_metrics.p_value == 1.0
    
    def test_calculate_drawdown_metrics(self, metrics_calculator, sample_equity_curve):
        """Test drawdown metrics calculation."""
        timestamps = [datetime.now() + timedelta(days=i) for i in range(len(sample_equity_curve))]
        
        drawdown_metrics = metrics_calculator.calculate_drawdown_metrics(
            sample_equity_curve, timestamps
        )
        
        assert isinstance(drawdown_metrics, DrawdownMetrics)
        assert drawdown_metrics.max_drawdown >= 0
        assert drawdown_metrics.current_drawdown >= 0
        assert drawdown_metrics.max_drawdown_duration >= 0
        assert drawdown_metrics.average_drawdown >= 0
        assert drawdown_metrics.ulcer_index >= 0
        assert drawdown_metrics.pain_index >= 0
        assert isinstance(drawdown_metrics.drawdown_distribution, dict)
        assert drawdown_metrics.max_drawdown_start <= drawdown_metrics.max_drawdown_end
    
    def test_calculate_drawdown_metrics_empty_data(self, metrics_calculator):
        """Test drawdown calculation with empty data."""
        drawdown_metrics = metrics_calculator.calculate_drawdown_metrics([])
        
        assert drawdown_metrics.max_drawdown == 0.0
        assert drawdown_metrics.max_drawdown_duration == 0
        assert drawdown_metrics.current_drawdown == 0.0
        assert drawdown_metrics.ulcer_index == 0.0
    
    def test_calculate_drawdown_with_no_drawdowns(self, metrics_calculator):
        """Test drawdown calculation with monotonically increasing equity."""
        increasing_equity = list(range(1000, 2000))  # Always increasing
        drawdown_metrics = metrics_calculator.calculate_drawdown_metrics(increasing_equity)
        
        assert drawdown_metrics.max_drawdown == 0.0
        assert drawdown_metrics.current_drawdown == 0.0
        assert drawdown_metrics.average_drawdown == 0.0
        assert drawdown_metrics.ulcer_index == 0.0
    
    def test_calculate_win_rate_metrics(self, metrics_calculator, sample_trade_pnls):
        """Test win rate metrics calculation."""
        win_rate_metrics = metrics_calculator.calculate_win_rate_metrics(sample_trade_pnls)
        
        assert isinstance(win_rate_metrics, WinRateMetrics)
        assert 0 <= win_rate_metrics.win_rate <= 1
        assert win_rate_metrics.winning_trades >= 0
        assert win_rate_metrics.losing_trades >= 0
        assert win_rate_metrics.total_trades == len(sample_trade_pnls)
        assert win_rate_metrics.profit_factor >= 0
        assert len(win_rate_metrics.win_rate_rolling) > 0
        assert win_rate_metrics.confidence_interval[0] <= win_rate_metrics.win_rate <= win_rate_metrics.confidence_interval[1]
        assert 0 <= win_rate_metrics.p_value <= 1
    
    def test_calculate_win_rate_metrics_all_wins(self, metrics_calculator):
        """Test win rate calculation with all winning trades."""
        winning_trades = [100, 200, 150, 300, 250]
        win_rate_metrics = metrics_calculator.calculate_win_rate_metrics(winning_trades)
        
        assert win_rate_metrics.win_rate == 1.0
        assert win_rate_metrics.winning_trades == 5
        assert win_rate_metrics.losing_trades == 0
        assert win_rate_metrics.avg_loss == 0.0
        assert win_rate_metrics.profit_factor == float('inf')
    
    def test_calculate_win_rate_metrics_all_losses(self, metrics_calculator):
        """Test win rate calculation with all losing trades."""
        losing_trades = [-100, -200, -150, -300, -250]
        win_rate_metrics = metrics_calculator.calculate_win_rate_metrics(losing_trades)
        
        assert win_rate_metrics.win_rate == 0.0
        assert win_rate_metrics.winning_trades == 0
        assert win_rate_metrics.losing_trades == 5
        assert win_rate_metrics.avg_win == 0.0
        assert win_rate_metrics.profit_factor == 0.0
    
    def test_calculate_win_rate_metrics_empty_data(self, metrics_calculator):
        """Test win rate calculation with empty data."""
        win_rate_metrics = metrics_calculator.calculate_win_rate_metrics([])
        
        assert win_rate_metrics.win_rate == 0.0
        assert win_rate_metrics.winning_trades == 0
        assert win_rate_metrics.losing_trades == 0
        assert win_rate_metrics.total_trades == 0
        assert win_rate_metrics.win_rate_rolling == []
    
    def test_calculate_comprehensive_metrics(self, metrics_calculator, sample_returns, 
                                        sample_equity_curve, sample_trade_pnls):
        """Test comprehensive metrics calculation."""
        timestamps = [datetime.now() + timedelta(days=i) for i in range(len(sample_equity_curve))]
        
        comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
            sample_returns, sample_equity_curve, sample_trade_pnls, timestamps
        )
        
        assert 'sharpe_ratio' in comprehensive_metrics
        assert 'drawdown' in comprehensive_metrics
        assert 'win_rate' in comprehensive_metrics
        assert 'summary' in comprehensive_metrics
        
        summary = comprehensive_metrics['summary']
        assert 'overall_sharpe' in summary
        assert 'max_drawdown' in summary
        assert 'win_rate' in summary
        assert 'profit_factor' in summary
    
    def test_consecutive_wins_losses_analysis(self, metrics_calculator):
        """Test consecutive wins and losses analysis."""
        trade_pnls = [100, 150, -50, -75, 200, 300, -100, 50, 75, -25]
        win_rate_metrics = metrics_calculator.calculate_win_rate_metrics(trade_pnls)
        
        assert win_rate_metrics.consecutive_wins_max >= 2  # 200, 300
        assert win_rate_metrics.consecutive_losses_max >= 2  # -50, -75
        assert win_rate_metrics.avg_consecutive_wins >= 1.0
        assert win_rate_metrics.avg_consecutive_losses >= 1.0


class TestPerformanceDashboard:
    """Test cases for PerformanceDashboard class."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create AdvancedPerformanceMetrics instance for testing."""
        return AdvancedPerformanceMetrics()
    
    @pytest.fixture
    def performance_dashboard(self, metrics_calculator):
        """Create PerformanceDashboard instance for testing."""
        return PerformanceDashboard(
            metrics_calculator=metrics_calculator,
            alert_thresholds={
                'min_sharpe': 0.5,
                'max_drawdown': 0.2,
                'min_win_rate': 0.4,
                'min_profit_factor': 1.2
            }
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for dashboard testing."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100).tolist()
        equity_curve = [100000]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        
        trade_pnls = []
        for i in range(20):
            if np.random.random() > 0.4:
                trade_pnls.append(np.random.uniform(50, 500))
            else:
                trade_pnls.append(np.random.uniform(-500, -50))
        
        return returns, equity_curve, trade_pnls
    
    def test_init(self, performance_dashboard):
        """Test PerformanceDashboard initialization."""
        assert performance_dashboard.metrics_calculator is not None
        assert 'min_sharpe' in performance_dashboard.alert_thresholds
        assert 'max_drawdown' in performance_dashboard.alert_thresholds
        assert 'min_win_rate' in performance_dashboard.alert_thresholds
        assert 'min_profit_factor' in performance_dashboard.alert_thresholds
        assert performance_dashboard.metrics_history == []
        assert performance_dashboard.alerts_history == []
    
    def test_update_metrics(self, performance_dashboard, sample_data):
        """Test metrics update functionality."""
        returns, equity_curve, trade_pnls = sample_data
        
        metrics = performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        
        assert 'sharpe_ratio' in metrics
        assert 'drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'summary' in metrics
        assert len(performance_dashboard.metrics_history) == 1
        assert performance_dashboard.last_update is not None
    
    def test_alert_generation(self, performance_dashboard):
        """Test alert generation for poor performance."""
        # Poor performing data
        returns = [-0.05] * 50  # Consistent losses
        equity_curve = [100000]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        trade_pnls = [-100] * 10  # All losing trades
        
        performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        
        # Should generate alerts for poor performance
        assert len(performance_dashboard.alerts_history) > 0
        
        # Check alert types
        alert_types = [alert['type'] for alert in performance_dashboard.alerts_history]
        assert 'drawdown' in alert_types  # High drawdown from losses
        assert 'win_rate' in alert_types  # 0% win rate
    
    def test_no_alerts_for_good_performance(self, performance_dashboard, sample_data):
        """Test no alerts for good performance."""
        returns, equity_curve, trade_pnls = sample_data
        
        # Make returns better to avoid alerts
        returns = [0.01] * len(returns)  # Consistent positive returns
        equity_curve = [100000]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        trade_pnls = [100] * len(trade_pnls)  # All winning trades
        
        performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        
        # Should not generate alerts for good performance
        assert len(performance_dashboard.alerts_history) == 0
    
    def test_metrics_summary(self, performance_dashboard, sample_data):
        """Test metrics summary functionality."""
        returns, equity_curve, trade_pnls = sample_data
        
        # Update metrics twice to test trends
        performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        
        summary = performance_dashboard.get_metrics_summary()
        
        assert 'current_metrics' in summary
        assert 'last_update' in summary
        assert 'trends' in summary
        assert 'recent_alerts' in summary
        assert 'total_alerts' in summary
        assert 'metrics_history_count' in summary
        assert summary['metrics_history_count'] == 2
    
    def test_export_dashboard_data(self, performance_dashboard, sample_data, tmp_path):
        """Test dashboard data export."""
        returns, equity_curve, trade_pnls = sample_data
        
        performance_dashboard.update_metrics(returns, equity_curve, trade_pnls)
        
        export_file = tmp_path / "dashboard_export.json"
        performance_dashboard.export_dashboard_data(str(export_file))
        
        assert export_file.exists()
        
        # Verify exported data
        import json
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        assert 'export_timestamp' in data
        assert 'current_metrics' in data
        assert 'metrics_history' in data
        assert 'alerts_history' in data
        assert 'alert_thresholds' in data


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_performance_metrics(self):
        """Test create_performance_metrics convenience function."""
        metrics_calc = create_performance_metrics(
            risk_free_rate=0.03,
            confidence_level=0.99
        )
        
        assert isinstance(metrics_calc, AdvancedPerformanceMetrics)
        assert metrics_calc.risk_free_rate == 0.03
        assert metrics_calc.confidence_level == 0.99
    
    def test_create_performance_dashboard(self):
        """Test create_performance_dashboard convenience function."""
        metrics_calc = create_performance_metrics()
        dashboard = create_performance_dashboard(
            metrics_calc,
            alert_thresholds={'min_sharpe': 1.0}
        )
        
        assert isinstance(dashboard, PerformanceDashboard)
        assert dashboard.metrics_calculator == metrics_calc
        assert dashboard.alert_thresholds['min_sharpe'] == 1.0


class TestIntegrationWithPnLTracker:
    """Integration tests with PnL tracker."""
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration with PnL tracker."""
        # Create PnL tracker and add trades
        pnl_tracker = PnLTracker(auto_save=False)
        
        # Add sample trades
        for i in range(20):
            if i % 2 == 0:
                pnl_tracker.add_trade(f"buy_{i}", "BTC/USDT", "buy", 1.0, 50000.0 + i * 100)
            else:
                pnl_tracker.add_trade(f"sell_{i}", "BTC/USDT", "sell", 1.0, 51000.0 + i * 100)
        
        # Create performance metrics calculator
        metrics_calc = create_performance_metrics()
        
        # Get data from PnL tracker
        trade_pnls = [record.realized_pnl for record in pnl_tracker.pnl_records]
        
        # Create synthetic equity curve and returns
        equity_curve = [100000]
        for pnl in trade_pnls:
            equity_curve.append(equity_curve[-1] + pnl)
        
        returns = []
        for i in range(1, len(equity_curve)):
            returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        # Calculate metrics
        comprehensive_metrics = metrics_calc.calculate_comprehensive_metrics(
            returns, equity_curve, trade_pnls
        )
        
        # Verify results
        assert 'sharpe_ratio' in comprehensive_metrics
        assert 'drawdown' in comprehensive_metrics
        assert 'win_rate' in comprehensive_metrics
        assert 'summary' in comprehensive_metrics
        
        summary = comprehensive_metrics['summary']
        assert summary['total_trades'] == len(trade_pnls)
        assert summary['win_rate'] >= 0
        assert summary['profit_factor'] >= 0
        assert isinstance(summary['overall_sharpe'], (int, float))
        assert isinstance(summary['max_drawdown'], (int, float))
    
    def test_real_time_scenario(self):
        """Test real-time scenario with incremental updates."""
        pnl_tracker = PnLTracker(auto_save=False)
        metrics_calc = create_performance_metrics()
        dashboard = create_performance_dashboard(metrics_calc)
        
        # Simulate real-time trading
        base_price = 50000
        equity_curve = [100000]
        returns = []
        trade_pnls = []
        
        for day in range(30):  # 30 days of trading
            # Simulate price movement
            price_change = np.random.normal(0, 0.02)
            new_price = base_price * (1 + price_change)
            
            # Add trade
            if day % 2 == 0:
                pnl_tracker.add_trade(f"buy_{day}", "BTC/USDT", "buy", 1.0, new_price)
            else:
                pnl_tracker.add_trade(f"sell_{day}", "BTC/USDT", "sell", 1.0, new_price)
            
            # Update equity curve
            current_pnl = pnl_tracker.get_cumulative_pnl()
            equity_curve.append(100000 + current_pnl)
            
            # Calculate return
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(daily_return)
            
            # Update dashboard
            trade_pnls = [record.realized_pnl for record in pnl_tracker.pnl_records]
            dashboard.update_metrics(returns, equity_curve, trade_pnls)
            
            base_price = new_price
        
        # Verify final state
        summary = dashboard.get_metrics_summary()
        assert summary['metrics_history_count'] == 30
        assert 'current_metrics' in summary
        assert 'trends' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
