"""
Unit tests for SimpleBacktester.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, Mock
from its_project.backtesting.simple import SimpleBacktester
from its_project.backtesting.base import BacktestResult, Trade
from its_project.decision.decision import Signal, Action, Decision
from its_project.models.base import BaseModel


# Mock model for testing
class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, predictions=None, confidences=None):
        self.predictions = predictions or [2, 1, 0]  # buy, hold, sell
        self.confidences = confidences or [0.8, 0.5, 0.7]
        self.call_count = 0
        self._fitted = False
    
    def fit(self, X, y):
        self._fitted = True
        return self
    
    def predict(self, features):
        self.call_count += 1
        idx = (self.call_count - 1) % len(self.predictions)
        return np.array([self.predictions[idx]])
    
    def predict_proba(self, features):
        # Return dummy probabilities
        return np.array([[0.3, 0.4, 0.3]])
    
    def get_confidence(self, features):
        idx = (self.call_count - 1) % len(self.confidences)
        return np.array([self.confidences[idx]])


# Mock decision maker for testing
class MockDecisionMaker:
    """Mock decision maker for testing."""
    
    def __init__(self, decisions=None):
        self.decisions = decisions or [
            Decision(action=Action.BUY, symbol="BTCUSDT", size=1.0, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test"),
            Decision(action=Action.HOLD, symbol="BTCUSDT", size=0.0, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test"),
            Decision(action=Action.SELL, symbol="BTCUSDT", size=1.0, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test"),
        ]
        self.call_count = 0
    
    def decide(self, signal, market_state):
        self.call_count += 1
        idx = (self.call_count - 1) % len(self.decisions)
        return self.decisions[idx]


@pytest.mark.unit
@pytest.mark.backtesting_layer
class TestSimpleBacktester:
    """Test SimpleBacktester functionality."""
    
    def test_initialization(self):
        """Test SimpleBacktester initialization."""
        config = {
            'initial_capital': 10000.0,
            'commission': 0.001,
            'slippage': 0.0001,
            'min_window': 50
        }
        
        backtester = SimpleBacktester(config)
        
        assert backtester.config == config
        assert backtester.initial_capital == 10000.0
    
    def test_initialization_defaults(self):
        """Test initialization with default config."""
        backtester = SimpleBacktester({})
        
        assert backtester.initial_capital == 10000  # default from base
    
    def test_extract_features(self):
        """Test feature extraction."""
        backtester = SimpleBacktester({})
        
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        features = backtester._extract_features(data)
        
        assert features.shape == (3, 5)
        assert np.array_equal(features, data[['open', 'high', 'low', 'close', 'volume']].values)
    
    def test_prediction_to_action_buy(self):
        """Test converting prediction to BUY action."""
        backtester = SimpleBacktester({})
        
        action = backtester._prediction_to_action(2)
        
        assert action == Action.BUY
    
    def test_prediction_to_action_hold(self):
        """Test converting prediction to HOLD action."""
        backtester = SimpleBacktester({})
        
        action = backtester._prediction_to_action(1)
        
        assert action == Action.HOLD
    
    def test_prediction_to_action_sell(self):
        """Test converting prediction to SELL action."""
        backtester = SimpleBacktester({})
        
        action = backtester._prediction_to_action(0)
        
        assert action == Action.SELL
    
    def test_prediction_to_action_invalid(self):
        """Test converting invalid prediction defaults to HOLD."""
        backtester = SimpleBacktester({})
        
        action = backtester._prediction_to_action(99)
        
        assert action == Action.HOLD
    
    def test_open_position(self):
        """Test opening a position."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        current_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        
        backtester._open_position('buy', 0.5, current_bar)
        
        assert backtester.current_position is not None
        assert backtester.current_position['side'] == 'buy'
        assert backtester.current_position['size'] == 0.5
        # Entry price includes slippage
        assert backtester.current_position['entry_price'] > 42000.0
        assert len(backtester.trades) == 1
    
    def test_close_position_long(self):
        """Test closing a long position."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        # Open a long position first
        entry_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._open_position('buy', 0.5, entry_bar)
        
        # Close the position at higher price
        close_bar = pd.Series({
            'timestamp': 1704067300000,
            'close': 43000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._close_position(close_bar)
        
        assert backtester.current_position is None
        assert len(backtester.trades) == 2
        
        # Check PnL calculation for long (includes commission and slippage)
        # Opening trade PnL is set to -pnl (negative of closing PnL)
        assert backtester.trades[0].pnl is not None
        assert backtester.trades[1].pnl is not None
    
    def test_close_position_short(self):
        """Test closing a short position."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        # Open a short position first
        entry_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._open_position('sell', 0.5, entry_bar)
        
        # Close the position at lower price (profit for short)
        close_bar = pd.Series({
            'timestamp': 1704067300000,
            'close': 41000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._close_position(close_bar)
        
        assert backtester.current_position is None
        assert len(backtester.trades) == 2
        
        # Check PnL calculation for short (includes commission and slippage)
        # Opening trade PnL is set to -pnl (negative of closing PnL)
        assert backtester.trades[0].pnl is not None
        assert backtester.trades[1].pnl is not None
    
    def test_close_position_no_position(self):
        """Test closing when no position exists."""
        backtester = SimpleBacktester({})
        
        current_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0
        })
        
        # Should not raise error
        backtester._close_position(current_bar)
        assert backtester.current_position is None
    
    def test_execute_decision_buy(self):
        """Test executing BUY decision."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        decision = Decision(action=Action.BUY, symbol="BTCUSDT", size=0.5, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test")
        current_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        
        backtester._execute_decision(decision, current_bar)
        
        assert backtester.current_position is not None
        assert backtester.current_position['side'] == 'buy'
    
    def test_execute_decision_sell(self):
        """Test executing SELL decision."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        decision = Decision(action=Action.SELL, symbol="BTCUSDT", size=0.5, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test")
        current_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        
        backtester._execute_decision(decision, current_bar)
        
        assert backtester.current_position is not None
        assert backtester.current_position['side'] == 'sell'
    
    def test_execute_decision_hold(self):
        """Test executing HOLD decision does nothing."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        decision = Decision(action=Action.HOLD, symbol="BTCUSDT", size=0.0, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test")
        current_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0
        })
        
        backtester._execute_decision(decision, current_bar)
        
        assert backtester.current_position is None
        assert len(backtester.trades) == 0
    
    def test_execute_decision_close_long(self):
        """Test executing SELL closes long position."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        # Open long position
        entry_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._open_position('buy', 0.5, entry_bar)
        
        # Close with SELL decision
        # Note: after closing, a new short position will be opened because action is SELL
        decision = Decision(action=Action.SELL, symbol="BTCUSDT", size=0.5, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test")
        close_bar = pd.Series({
            'timestamp': 1704067300000,
            'close': 43000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._execute_decision(decision, close_bar)
        
        # Position is closed and then a new short position is opened
        assert backtester.current_position is not None
        assert backtester.current_position['side'] == 'sell'
        assert len(backtester.trades) == 2  # open long, close long (and open short in same trade)
    
    def test_execute_decision_close_short(self):
        """Test executing BUY closes short position."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        # Open short position
        entry_bar = pd.Series({
            'timestamp': 1704067200000,
            'close': 42000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._open_position('sell', 0.5, entry_bar)
        
        # Close with BUY decision
        # Note: after closing, a new long position will be opened because action is BUY
        decision = Decision(action=Action.BUY, symbol="BTCUSDT", size=0.5, price=None, stop_loss=None, take_profit=None, timestamp=0, reason="test")
        close_bar = pd.Series({
            'timestamp': 1704067300000,
            'close': 41000.0,
            'symbol': 'BTCUSDT'
        })
        backtester._execute_decision(decision, close_bar)
        
        # Position is closed and then a new long position is opened
        assert backtester.current_position is not None
        assert backtester.current_position['side'] == 'buy'
        assert len(backtester.trades) == 2  # open short, close short (and open long in same trade)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        # Set up some trades
        backtester.trades = [
            Trade(timestamp=1, symbol="BTCUSDT", side="buy", price=100, size=1, commission=1, slippage=0, pnl=50),
            Trade(timestamp=2, symbol="BTCUSDT", side="sell", price=110, size=1, commission=1, slippage=0, pnl=-50),
            Trade(timestamp=3, symbol="BTCUSDT", side="buy", price=100, size=1, commission=1, slippage=0, pnl=100),
        ]
        backtester.equity_history = [10000, 10050, 10000, 10100]
        
        equity_curve = np.array(backtester.equity_history)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = backtester._calculate_metrics(equity_curve, returns)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'num_trades' in metrics
        assert metrics['num_trades'] == 3
    
    def test_calculate_metrics_no_trades(self):
        """Test metrics calculation with no trades."""
        backtester = SimpleBacktester({'initial_capital': 10000.0})
        
        backtester.trades = []
        backtester.equity_history = [10000]
        
        equity_curve = np.array(backtester.equity_history)
        returns = np.array([])
        
        metrics = backtester._calculate_metrics(equity_curve, returns)
        
        assert metrics['win_rate'] == 0.0
        assert metrics['num_trades'] == 0
        assert metrics['profit_factor'] == float('inf')
    
    def test_create_positions_df(self):
        """Test creating positions DataFrame."""
        backtester = SimpleBacktester({})
        
        backtester.trades = [
            Trade(timestamp=1, symbol="BTCUSDT", side="buy", price=100, size=1, commission=1, slippage=0, pnl=50),
            Trade(timestamp=2, symbol="BTCUSDT", side="sell", price=110, size=1, commission=1, slippage=0, pnl=-50),
        ]
        
        positions_df = backtester._create_positions_df()
        
        assert len(positions_df) == 2
        assert positions_df.iloc[0]['side'] == 'buy'
        assert positions_df.iloc[1]['side'] == 'sell'
    
    def test_run_unsorted_data_raises_error(self):
        """Test that unsorted data raises ValueError."""
        backtester = SimpleBacktester({})
        
        # Create unsorted data
        data = pd.DataFrame({
            'timestamp': [3, 1, 2],
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        model = MockModel()
        decision_maker = MockDecisionMaker()
        
        with pytest.raises(ValueError, match="Data MUST be sorted by time"):
            backtester.run(data, model, decision_maker)
    
    def test_run_sorted_data(self):
        """Test running backtest with sorted data."""
        backtester = SimpleBacktester({'initial_capital': 10000.0, 'min_window': 2})
        
        # Create sorted data
        data = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'open': [1, 2, 3, 4, 5],
            'high': [2, 3, 4, 5, 6],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5],
            'volume': [100, 200, 300, 400, 500]
        })
        
        model = MockModel()
        decision_maker = MockDecisionMaker()
        
        result = backtester.run(data, model, decision_maker)
        
        assert isinstance(result, BacktestResult)
        assert result.trades is not None
        assert result.equity_curve is not None
        assert result.metrics is not None
