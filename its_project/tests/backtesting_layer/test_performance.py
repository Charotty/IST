"""
Unit tests for PerformanceAnalyzer.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock seaborn and matplotlib before importing
sys.modules['seaborn'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

from its_project.backtesting.performance import PerformanceAnalyzer
from its_project.backtesting.base import BacktestResult, Trade


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""
    
    @pytest.fixture
    def sample_backtest_result(self):
        """Create sample BacktestResult."""
        # Create sample equity curve
        equity = np.array([100000, 101000, 102500, 101800, 103200, 104500])
        
        # Create sample returns
        returns = np.array([0.01, 0.01485, -0.00683, 0.01373, 0.01262])
        
        # Create sample trades
        trades = [
            Trade(
                timestamp=1000,
                symbol='BTC/USDT',
                side='buy',
                size=1.0,
                price=50000,
                commission=10,
                slippage=5,
                pnl=1000
            ),
            Trade(
                timestamp=2000,
                symbol='BTC/USDT',
                side='sell',
                size=1.0,
                price=51000,
                commission=10,
                slippage=5,
                pnl=1000
            ),
            Trade(
                timestamp=3000,
                symbol='ETH/USDT',
                side='buy',
                size=2.0,
                price=3000,
                commission=15,
                slippage=3,
                pnl=-500
            ),
            Trade(
                timestamp=4000,
                symbol='ETH/USDT',
                side='sell',
                size=2.0,
                price=2950,
                commission=15,
                slippage=3,
                pnl=-100
            ),
        ]
        
        # Create sample metrics
        metrics = {
            'total_return': 0.045,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.02,
            'win_rate': 0.75,
            'profit_factor': 2.0,
            'num_trades': 4
        }
        
        # Create sample positions DataFrame
        positions = pd.DataFrame({
            'timestamp': [1000, 2000, 3000, 4000],
            'symbol': ['BTC/USDT', 'BTC/USDT', 'ETH/USDT', 'ETH/USDT'],
            'size': [1.0, -1.0, 2.0, -2.0]
        })
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            returns=returns,
            metrics=metrics,
            positions=positions
        )
    
    @pytest.fixture
    def empty_backtest_result(self):
        """Create empty BacktestResult."""
        return BacktestResult(
            trades=[],
            equity_curve=np.array([100000]),
            returns=np.array([]),
            metrics={},
            positions=pd.DataFrame()
        )
    
    def test_initialization(self, sample_backtest_result):
        """Test PerformanceAnalyzer initialization."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        
        assert analyzer.result == sample_backtest_result
    
    def test_calculate_detailed_metrics(self, sample_backtest_result):
        """Test detailed metrics calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Check all expected keys
        expected_keys = [
            'total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'max_drawdown', 'avg_drawdown_duration',
            'var_95', 'cvar_95', 'win_rate', 'avg_win', 'avg_loss',
            'profit_factor', 'num_trades', 'avg_trade_duration'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Check values are reasonable
        assert metrics['num_trades'] == 4
        assert metrics['total_return'] >= 0
        assert metrics['max_drawdown'] >= 0
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_calculate_detailed_metrics_empty(self, empty_backtest_result):
        """Test metrics calculation with empty data."""
        analyzer = PerformanceAnalyzer(empty_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Should handle empty data gracefully
        assert metrics['num_trades'] == 0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['sortino_ratio'] == 0.0
        assert metrics['win_rate'] == 0.0
    
    def test_find_drawdown_periods(self, sample_backtest_result):
        """Test drawdown period detection."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        
        equity = sample_backtest_result.equity_curve
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        
        periods = analyzer._find_drawdown_periods(drawdown)
        
        # Should return list of tuples
        assert isinstance(periods, list)
        for period in periods:
            assert isinstance(period, tuple)
            assert len(period) == 2
            assert period[0] < period[1]
    
    def test_find_drawdown_periods_no_drawdown(self):
        """Test drawdown detection with no drawdowns."""
        result = BacktestResult(
            trades=[],
            equity_curve=np.array([100, 110, 120, 130, 140]),
            returns=np.array([0.1, 0.09, 0.08, 0.08]),
            metrics={},
            positions=pd.DataFrame()
        )
        analyzer = PerformanceAnalyzer(result)
        
        drawdown = np.array([0, 0, 0, 0, 0])
        periods = analyzer._find_drawdown_periods(drawdown)
        
        assert periods == []
    
    def test_calculate_avg_trade_duration_insufficient_trades(self):
        """Test avg trade duration with insufficient trades."""
        result = BacktestResult(
            trades=[Trade(timestamp=1000, symbol='BTC', side='buy', price=50000, size=1.0, commission=10, slippage=5, pnl=1000)],
            equity_curve=np.array([100000]),
            returns=np.array([]),
            metrics={},
            positions=pd.DataFrame()
        )
        analyzer = PerformanceAnalyzer(result)
        duration = analyzer._calculate_avg_trade_duration()
        
        assert duration == 0.0
    
    def test_plot_equity_curve(self, sample_backtest_result):
        """Test equity curve plotting."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        
        with patch('its_project.backtesting.performance.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            analyzer.plot_equity_curve(figsize=(10, 5))
            
            # Verify subplots was called
            mock_plt.subplots.assert_called_once()
    
    def test_plot_returns_distribution(self, sample_backtest_result):
        """Test returns distribution plotting."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        
        with patch('its_project.backtesting.performance.plt') as mock_plt:
            with patch('scipy.stats.probplot'):
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                
                analyzer.plot_returns_distribution(figsize=(10, 5))
                
                # Verify subplots was called
                mock_plt.subplots.assert_called_once()
    
    def test_plot_monthly_returns(self, sample_backtest_result):
        """Test monthly returns heatmap plotting."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        
        with patch('its_project.backtesting.performance.plt') as mock_plt:
            with patch('its_project.backtesting.performance.sns') as mock_sns:
                analyzer.plot_monthly_returns(figsize=(10, 5))
    
    def test_plot_monthly_returns_no_trades(self, empty_backtest_result):
        """Test monthly returns with no trades."""
        analyzer = PerformanceAnalyzer(empty_backtest_result)
        
        with patch('builtins.print') as mock_print:
            analyzer.plot_monthly_returns()
            mock_print.assert_called_with("No trades to plot monthly returns")
    
    def test_generate_report(self, sample_backtest_result):
        """Test performance report generation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        report = analyzer.generate_report()
        
        assert isinstance(report, str)
        assert "BACKTEST PERFORMANCE REPORT" in report
        assert "Portfolio Metrics" in report
        assert "Risk Metrics" in report
        assert "Trading Metrics" in report
        assert "Total Return" in report
        assert "Sharpe Ratio" in report
        assert "Win Rate" in report
    
    def test_generate_report_empty(self, empty_backtest_result):
        """Test report generation with empty data."""
        analyzer = PerformanceAnalyzer(empty_backtest_result)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.generate_report()
        
        assert isinstance(report, str)
        assert "BACKTEST PERFORMANCE REPORT" in report
    
    def test_calculate_detailed_metrics_win_rate(self, sample_backtest_result):
        """Test win rate calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # 2 winning trades (1000, 1000), 2 losing trades (-500, -100)
        # Total 4 trades with pnl
        expected_win_rate = 0.5
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01
    
    def test_calculate_detailed_metrics_profit_factor(self, sample_backtest_result):
        """Test profit factor calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Should be positive
        assert metrics['profit_factor'] > 0
    
    def test_calculate_detailed_metrics_sharpe_ratio(self, sample_backtest_result):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Sharpe should be a number
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert not np.isnan(metrics['sharpe_ratio'])
    
    def test_calculate_detailed_metrics_max_drawdown(self, sample_backtest_result):
        """Test max drawdown calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Max drawdown should be between 0 and 1
        assert 0 <= metrics['max_drawdown'] <= 1
    
    def test_calculate_detailed_metrics_var_cvar(self, sample_backtest_result):
        """Test VaR and CVaR calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # VaR and CVaR should be numbers
        assert isinstance(metrics['var_95'], (int, float))
        assert isinstance(metrics['cvar_95'], (int, float))
    
    def test_calculate_detailed_metrics_calmar_ratio(self, sample_backtest_result):
        """Test Calmar ratio calculation."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        metrics = analyzer.calculate_detailed_metrics()
        
        # Calmar should be a number
        assert isinstance(metrics['calmar_ratio'], (int, float))
    
    def test_generate_report_recommendations(self, sample_backtest_result):
        """Test report includes recommendations."""
        analyzer = PerformanceAnalyzer(sample_backtest_result)
        report = analyzer.generate_report()
        
        assert "Recommendations" in report
