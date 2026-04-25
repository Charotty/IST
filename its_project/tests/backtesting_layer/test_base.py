"""
Unit tests for BaseBacktester and related classes.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.backtesting.base import Trade, BacktestResult, BaseBacktester


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a Trade."""
        trade = Trade(
            timestamp=1704067200000,
            symbol="BTC/USDT",
            side="buy",
            price=42000.0,
            size=0.1,
            commission=4.2,
            slippage=2.1
        )
        
        assert trade.timestamp == 1704067200000
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.price == 42000.0
        assert trade.size == 0.1
        assert trade.commission == 4.2
        assert trade.slippage == 2.1
        assert trade.pnl is None
    
    def test_trade_with_pnl(self):
        """Test creating a Trade with PnL."""
        trade = Trade(
            timestamp=1704067200000,
            symbol="BTC/USDT",
            side="sell",
            price=42500.0,
            size=0.1,
            commission=4.25,
            slippage=2.125,
            pnl=450.0
        )
        
        assert trade.pnl == 450.0


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_backtest_result_creation(self):
        """Test creating a BacktestResult."""
        trades = []
        equity_curve = np.array([10000, 10100, 10200])
        returns = np.array([0.0, 0.01, 0.01])
        metrics = {
            'total_return': 0.02,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.01,
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'num_trades': 10
        }
        positions = pd.DataFrame()
        
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            positions=positions
        )
        
        assert len(result.trades) == 0
        assert len(result.equity_curve) == 3
        assert result.metrics['total_return'] == 0.02
    
    def test_backtest_result_summary(self):
        """Test BacktestResult summary method."""
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 2.1,
            'max_drawdown': -0.05,
            'win_rate': 0.65,
            'profit_factor': 2.3,
            'num_trades': 25
        }
        
        result = BacktestResult(
            trades=[],
            equity_curve=np.array([]),
            returns=np.array([]),
            metrics=metrics,
            positions=pd.DataFrame()
        )
        
        summary = result.summary()
        
        assert "15.00%" in summary
        assert "2.10" in summary
        assert "-5.00%" in summary
        assert "65.00%" in summary
        assert "2.30" in summary
        assert "25" in summary


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestBaseBacktester:
    """Test BaseBacktester abstract class."""
    
    def test_initialization(self):
        """Test BaseBacktester initialization."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([10000]),
                    returns=np.array([0.0]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        config = {
            'commission_rate': 0.002,
            'slippage_rate': 0.001,
            'initial_capital': 50000
        }
        
        backtester = ConcreteBacktester(config)
        
        assert backtester.config == config
        assert backtester.commission_rate == 0.002
        assert backtester.slippage_rate == 0.001
        assert backtester.initial_capital == 50000
        assert backtester.current_capital == 50000
        assert len(backtester.trades) == 0
        assert backtester.equity_history == [50000]
        assert backtester.current_position is None
    
    def test_default_config(self):
        """Test default configuration values."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({})
        
        assert backtester.commission_rate == 0.001
        assert backtester.slippage_rate == 0.0005
        assert backtester.initial_capital == 10000
    
    def test_calculate_commission(self):
        """Test commission calculation."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({'commission_rate': 0.001})
        
        commission = backtester.calculate_commission(42000.0, 0.1)
        
        assert commission == 42000.0 * 0.1 * 0.001  # 4.2
    
    def test_calculate_slippage_buy(self):
        """Test slippage calculation for buy."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({'slippage_rate': 0.0005})
        
        slippage = backtester.calculate_slippage(42000.0, 0.1, 'buy')
        
        assert slippage == 42000.0 * 0.0005  # Positive for buy
    
    def test_calculate_slippage_sell(self):
        """Test slippage calculation for sell."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({'slippage_rate': 0.0005})
        
        slippage = backtester.calculate_slippage(42000.0, 0.1, 'sell')
        
        assert slippage == -42000.0 * 0.0005  # Negative for sell
    
    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'initial_capital': 10000
        })
        
        trade = backtester.execute_trade(
            timestamp=1704067200000,
            symbol="BTC/USDT",
            side="buy",
            price=42000.0,
            size=0.1
        )
        
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.price > 42000.0  # Price increased due to slippage
        assert trade.commission > 0
        assert trade.slippage > 0
        assert backtester.current_capital < 10000  # Capital decreased
        assert len(backtester.trades) == 1
        assert len(backtester.equity_history) == 2
    
    def test_execute_trade_sell(self):
        """Test executing a sell trade."""
        
        class ConcreteBacktester(BaseBacktester):
            def run(self, data, strategy):
                return BacktestResult(
                    trades=[],
                    equity_curve=np.array([]),
                    returns=np.array([]),
                    metrics={},
                    positions=pd.DataFrame()
                )
        
        backtester = ConcreteBacktester({
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'initial_capital': 10000
        })
        
        trade = backtester.execute_trade(
            timestamp=1704067200000,
            symbol="BTC/USDT",
            side="sell",
            price=42000.0,
            size=0.1
        )
        
        assert trade.side == "sell"
        assert trade.price < 42000.0  # Price decreased due to slippage
        assert backtester.current_capital > 10000  # Capital increased
