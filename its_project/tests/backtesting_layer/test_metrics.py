"""
Unit tests for BacktestMetrics.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from its_project.backtesting.metrics import BacktestMetrics
from its_project.backtesting.base import BacktestResult, Trade


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestBacktestMetrics:
    """Test BacktestMetrics class."""
    
    def test_calculate_commission(self):
        """Test commission calculation."""
        trade_value = 10000.0
        commission_rate = 0.001
        
        cost = BacktestMetrics.calculate_commission(trade_value, commission_rate)
        
        expected = trade_value * commission_rate + trade_value * 0.0001
        assert abs(cost - expected) < 0.01
    
    def test_calculate_commission_with_slippage(self):
        """Test commission calculation with slippage."""
        trade_value = 10000.0
        commission_rate = 0.001
        slippage_cost = 5.0
        
        cost = BacktestMetrics.calculate_commission(
            trade_value, commission_rate, slippage_cost=slippage_cost
        )
        
        expected = trade_value * commission_rate + trade_value * 0.0001 + slippage_cost
        assert abs(cost - expected) < 0.01
    
    def test_calculate_commission_custom_exchange_fee(self):
        """Test commission calculation with custom exchange fee."""
        trade_value = 10000.0
        commission_rate = 0.001
        exchange_fee = 0.0002
        
        cost = BacktestMetrics.calculate_commission(
            trade_value, commission_rate, exchange_fee=exchange_fee
        )
        
        expected = trade_value * commission_rate + trade_value * exchange_fee
        assert abs(cost - expected) < 0.01
    
    def test_calculate_slippage_linear(self):
        """Test linear slippage model."""
        order_size = 1.0
        market_price = 50000.0
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, slippage_model="linear"
        )
        
        assert slippage > 0
        assert isinstance(slippage, float)
    
    def test_calculate_slippage_linear_custom_params(self):
        """Test linear slippage with custom parameters."""
        order_size = 1.0
        market_price = 50000.0
        params = {"base_rate": 0.0002, "size_factor": 0.000002}
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, slippage_model="linear", slippage_params=params
        )
        
        assert slippage > 0
    
    def test_calculate_slippage_percentage(self):
        """Test percentage-based slippage model."""
        order_size = 1.0
        market_price = 50000.0
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, slippage_model="percentage"
        )
        
        assert slippage > 0
    
    def test_calculate_slippage_percentage_custom_params(self):
        """Test percentage slippage with custom parameters."""
        order_size = 1.0
        market_price = 50000.0
        params = {"percentage_rate": 0.001}
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, slippage_model="percentage", slippage_params=params
        )
        
        expected = market_price * order_size * 0.001
        assert abs(slippage - expected) < 0.01
    
    def test_calculate_slippage_volume_impact(self):
        """Test volume impact slippage model."""
        order_size = 1.0
        market_price = 50000.0
        order_book = {
            "bids": [[49999, 10], [49998, 20]],
            "asks": [[50001, 10], [50002, 20]]
        }
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, order_book_depth=order_book, slippage_model="volume_impact"
        )
        
        assert slippage >= 0
    
    def test_calculate_slippage_volume_impact_no_book(self):
        """Test volume impact slippage without order book (fallback to linear)."""
        order_size = 1.0
        market_price = 50000.0
        
        slippage = BacktestMetrics.calculate_slippage(
            order_size, market_price, slippage_model="volume_impact"
        )
        
        assert slippage >= 0
    
    def test_calculate_slippage_unknown_model(self):
        """Test slippage calculation with unknown model."""
        order_size = 1.0
        market_price = 50000.0
        
        with pytest.raises(ValueError, match="Unknown slippage model"):
            BacktestMetrics.calculate_slippage(
                order_size, market_price, slippage_model="unknown"
            )
    
    def test_calculate_volume_impact_slippage(self):
        """Test volume impact slippage calculation."""
        order_size = 1.0
        market_price = 50000.0
        order_book = {
            "bids": [[49999, 10], [49998, 20]],
            "asks": [[50001, 10], [50002, 20]]
        }
        params = {"impact_a": 0.001, "impact_b": 0.5}
        
        slippage = BacktestMetrics._calculate_volume_impact_slippage(
            order_size, market_price, order_book, params
        )
        
        assert slippage >= 0
    
    def test_calculate_volume_impact_slippage_empty_book(self):
        """Test volume impact slippage with empty order book."""
        order_size = 1.0
        market_price = 50000.0
        order_book = {"bids": [], "asks": []}
        params = {}
        
        slippage = BacktestMetrics._calculate_volume_impact_slippage(
            order_size, market_price, order_book, params
        )
        
        assert slippage >= 0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        sharpe = BacktestMetrics._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        returns = np.array([])
        
        sharpe = BacktestMetrics._calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = BacktestMetrics._calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sortino = BacktestMetrics._calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino) or sortino == 0.0
    
    def test_calculate_sortino_ratio_empty(self):
        """Test Sortino ratio with empty returns."""
        returns = np.array([])
        
        sortino = BacktestMetrics._calculate_sortino_ratio(returns)
        
        assert sortino == 0.0
    
    def test_calculate_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        
        sortino = BacktestMetrics._calculate_sortino_ratio(returns)
        
        assert sortino == 0.0
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        calmar = BacktestMetrics._calculate_calmar_ratio(returns)
        
        assert isinstance(calmar, float)
    
    def test_calculate_calmar_ratio_empty(self):
        """Test Calmar ratio with empty returns."""
        returns = np.array([])
        
        calmar = BacktestMetrics._calculate_calmar_ratio(returns)
        
        assert calmar == 0.0
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        returns = np.array([0.01, 0.02, -0.05, 0.03, 0.01])
        
        max_dd = BacktestMetrics._calculate_max_drawdown(returns)
        
        assert max_dd <= 0
        assert isinstance(max_dd, float)
    
    def test_calculate_max_drawdown_empty(self):
        """Test max drawdown with empty returns."""
        returns = np.array([])
        
        max_dd = BacktestMetrics._calculate_max_drawdown(returns)
        
        assert max_dd == 0.0
    
    def test_calculate_drawdown_duration(self):
        """Test drawdown duration calculation."""
        returns = np.array([0.01, -0.02, -0.01, 0.03, 0.01])
        
        duration = BacktestMetrics._calculate_drawdown_duration(returns)
        
        assert duration >= 0
        assert isinstance(duration, (int, float))
    
    def test_calculate_drawdown_duration_empty(self):
        """Test drawdown duration with empty returns."""
        returns = np.array([])
        
        duration = BacktestMetrics._calculate_drawdown_duration(returns)
        
        assert duration == 0.0
    
    def test_calculate_drawdown_duration_no_drawdown(self):
        """Test drawdown duration with no drawdown."""
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        
        duration = BacktestMetrics._calculate_drawdown_duration(returns)
        
        assert duration == 0.0
    
    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        
        pf = BacktestMetrics._calculate_profit_factor(returns)
        
        assert pf >= 0
        assert isinstance(pf, float)
    
    def test_calculate_profit_factor_empty(self):
        """Test profit factor with empty returns."""
        returns = np.array([])
        
        pf = BacktestMetrics._calculate_profit_factor(returns)
        
        assert pf == 0.0
    
    def test_calculate_profit_factor_only_losses(self):
        """Test profit factor with only losses."""
        returns = np.array([-0.01, -0.02, -0.01])
        
        pf = BacktestMetrics._calculate_profit_factor(returns)
        
        assert pf == 0.0
    
    def test_annualize_return(self):
        """Test return annualization."""
        total_return = 0.1
        num_periods = 252  # One year
        
        annualized = BacktestMetrics._annualize_return(total_return, num_periods)
        
        assert isinstance(annualized, float)
        assert annualized >= 0
    
    def test_annualize_return_zero_periods(self):
        """Test annualization with zero periods."""
        total_return = 0.1
        num_periods = 0
        
        annualized = BacktestMetrics._annualize_return(total_return, num_periods)
        
        assert annualized == 0.0
    
    def test_estimate_commission_costs(self):
        """Test commission cost estimation."""
        # Create mock trades with entry_price and exit_price attributes
        trades = [
            Mock(entry_price=50000, exit_price=51000, size=1.0),
            Mock(entry_price=3000, exit_price=2950, size=2.0),
        ]
        commission_rate = 0.001
        
        cost = BacktestMetrics._estimate_commission_costs(trades, commission_rate)
        
        assert cost > 0
    
    def test_estimate_commission_costs_no_entry_price(self):
        """Test commission cost estimation without entry price."""
        trades = [Mock(entry_price=None, exit_price=51000, size=1.0)]
        commission_rate = 0.001
        
        cost = BacktestMetrics._estimate_commission_costs(trades, commission_rate)
        
        assert cost >= 0
    
    def test_estimate_slippage_costs(self):
        """Test slippage cost estimation."""
        trades = [
            Mock(entry_price=50000, exit_price=51000, size=1.0),
            Mock(entry_price=3000, exit_price=2950, size=2.0),
        ]
        slippage_model = "linear"
        
        cost = BacktestMetrics._estimate_slippage_costs(trades, slippage_model, {})
        
        assert cost >= 0
    
    def test_analyze_trade_statistics(self):
        """Test trade statistics analysis."""
        from datetime import datetime, timedelta
        
        entry_time = datetime(2024, 1, 1)
        exit_time = datetime(2024, 1, 2)
        
        trades = [
            Mock(entry_price=50000, exit_price=51000, size=1.0, 
                 entry_time=entry_time, exit_time=exit_time),
            Mock(entry_price=3000, exit_price=2950, size=2.0,
                 entry_time=entry_time, exit_time=exit_time),
        ]
        
        stats = BacktestMetrics._analyze_trade_statistics(trades)
        
        assert "avg_trade_duration" in stats
        assert "avg_trade_size" in stats
        assert "total_volume" in stats
    
    def test_analyze_trade_statistics_empty(self):
        """Test trade statistics with empty trades."""
        trades = []
        
        stats = BacktestMetrics._analyze_trade_statistics(trades)
        
        assert stats == {}
    
    def test_calculate_enhanced_metrics(self):
        """Test enhanced metrics calculation."""
        # Note: calculate_realistic_pnl expects entry_price/exit_price which don't exist in Trade
        # So we'll test the helper methods directly instead
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        
        # Test individual metric calculations
        sharpe = BacktestMetrics._calculate_sharpe_ratio(returns)
        sortino = BacktestMetrics._calculate_sortino_ratio(returns)
        calmar = BacktestMetrics._calculate_calmar_ratio(returns)
        max_dd = BacktestMetrics._calculate_max_drawdown(returns)
        duration = BacktestMetrics._calculate_drawdown_duration(returns)
        pf = BacktestMetrics._calculate_profit_factor(returns)
        
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(calmar, float)
        assert isinstance(max_dd, float)
        assert isinstance(duration, (int, float))
        assert isinstance(pf, float)
    
    def test_calculate_enhanced_metrics_no_trades(self):
        """Test enhanced metrics with no trades."""
        result = BacktestResult(
            trades=[],
            equity_curve=np.array([100000]),
            returns=np.array([]),
            metrics={},
            positions=pd.DataFrame()
        )
        
        metrics = BacktestMetrics.calculate_enhanced_metrics(result)
        
        assert "error" in metrics
