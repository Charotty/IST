"""
Unit tests for PortfolioManager and Position.
"""
import pytest
from its_project.decision.portfolio import Position, PortfolioManager


@pytest.mark.unit
@pytest.mark.decision_layer
class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a Position."""
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.side == "long"
        assert position.size == 0.5
        assert position.entry_price == 42000.0
        assert position.current_price == 42500.0
        assert position.unrealized_pnl == 250.0
    
    def test_position_short(self):
        """Test creating a short position."""
        position = Position(
            symbol="BTC/USDT",
            side="short",
            size=0.3,
            entry_price=43000.0,
            current_price=42500.0,
            unrealized_pnl=150.0,
            timestamp=1704067200000
        )
        
        assert position.side == "short"
        assert position.unrealized_pnl == 150.0


@pytest.mark.unit
@pytest.mark.decision_layer
class TestPortfolioManager:
    """Test PortfolioManager functionality."""
    
    def test_initialization(self):
        """Test PortfolioManager initialization."""
        config = {'max_positions': 10}
        manager = PortfolioManager(config)
        
        assert manager.config == config
        assert manager.positions == {}
        assert manager.max_positions == 10
    
    def test_default_config(self):
        """Test default configuration values."""
        manager = PortfolioManager({})
        
        assert manager.max_positions == 5
    
    def test_can_open_position_true(self):
        """Test can_open_position when position can be opened."""
        manager = PortfolioManager({'max_positions': 5})
        
        result = manager.can_open_position("BTC/USDT")
        
        assert result is True
    
    def test_can_open_position_symbol_exists(self):
        """Test can_open_position when symbol already exists."""
        manager = PortfolioManager({'max_positions': 5})
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        manager.add_position(position)
        
        result = manager.can_open_position("BTC/USDT")
        
        assert result is False
    
    def test_can_open_position_max_reached(self):
        """Test can_open_position when max positions reached."""
        manager = PortfolioManager({'max_positions': 2})
        
        # Add 2 positions
        for i in range(2):
            position = Position(
                symbol=f"SYMBOL{i}/USDT",
                side="long",
                size=0.5,
                entry_price=42000.0,
                current_price=42500.0,
                unrealized_pnl=250.0,
                timestamp=1704067200000
            )
            manager.add_position(position)
        
        result = manager.can_open_position("BTC/USDT")
        
        assert result is False
    
    def test_add_position(self):
        """Test adding a position."""
        manager = PortfolioManager({})
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        
        manager.add_position(position)
        
        assert "BTC/USDT" in manager.positions
        assert manager.positions["BTC/USDT"] == position
    
    def test_remove_position(self):
        """Test removing a position."""
        manager = PortfolioManager({})
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        manager.add_position(position)
        
        removed = manager.remove_position("BTC/USDT")
        
        assert removed == position
        assert "BTC/USDT" not in manager.positions
    
    def test_remove_nonexistent_position(self):
        """Test removing a non-existent position."""
        manager = PortfolioManager({})
        
        removed = manager.remove_position("BTC/USDT")
        
        assert removed is None
    
    def test_get_position(self):
        """Test getting a position."""
        manager = PortfolioManager({})
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        manager.add_position(position)
        
        retrieved = manager.get_position("BTC/USDT")
        
        assert retrieved == position
    
    def test_get_nonexistent_position(self):
        """Test getting a non-existent position."""
        manager = PortfolioManager({})
        
        retrieved = manager.get_position("BTC/USDT")
        
        assert retrieved is None
    
    def test_update_prices(self):
        """Test updating prices for all positions."""
        manager = PortfolioManager({})
        position1 = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42000.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        position2 = Position(
            symbol="ETH/USDT",
            side="long",
            size=1.0,
            entry_price=2500.0,
            current_price=2500.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        manager.add_position(position1)
        manager.add_position(position2)
        
        prices = {"BTC/USDT": 42500.0, "ETH/USDT": 2550.0}
        manager.update_prices(prices)
        
        assert manager.positions["BTC/USDT"].current_price == 42500.0
        assert manager.positions["ETH/USDT"].current_price == 2550.0
        assert manager.positions["BTC/USDT"].unrealized_pnl == 250.0  # 0.5 * (42500 - 42000)
        assert manager.positions["ETH/USDT"].unrealized_pnl == 50.0  # 1.0 * (2550 - 2500)
    
    def test_get_total_pnl(self):
        """Test getting total PnL."""
        manager = PortfolioManager({})
        position1 = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        position2 = Position(
            symbol="ETH/USDT",
            side="long",
            size=1.0,
            entry_price=2500.0,
            current_price=2550.0,
            unrealized_pnl=50.0,
            timestamp=1704067200000
        )
        manager.add_position(position1)
        manager.add_position(position2)
        
        total_pnl = manager.get_total_pnl()
        
        assert total_pnl == 300.0
    
    def test_get_total_pnl_empty(self):
        """Test getting total PnL with no positions."""
        manager = PortfolioManager({})
        
        total_pnl = manager.get_total_pnl()
        
        assert total_pnl == 0.0
    
    def test_get_exposure(self):
        """Test getting exposure per position."""
        manager = PortfolioManager({})
        position1 = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        position2 = Position(
            symbol="ETH/USDT",
            side="long",
            size=1.0,
            entry_price=2500.0,
            current_price=2550.0,
            unrealized_pnl=50.0,
            timestamp=1704067200000
        )
        manager.add_position(position1)
        manager.add_position(position2)
        
        exposure = manager.get_exposure()
        
        assert exposure["BTC/USDT"] == 21250.0  # 0.5 * 42500
        assert exposure["ETH/USDT"] == 2550.0  # 1.0 * 2550
    
    def test_calculate_pnl_long(self):
        """Test PnL calculation for long position."""
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        
        pnl = PortfolioManager._calculate_pnl(position)
        
        assert pnl == 250.0  # 0.5 * (42500 - 42000)
    
    def test_calculate_pnl_short(self):
        """Test PnL calculation for short position."""
        position = Position(
            symbol="BTC/USDT",
            side="short",
            size=0.5,
            entry_price=43000.0,
            current_price=42500.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        
        pnl = PortfolioManager._calculate_pnl(position)
        
        assert pnl == 250.0  # 0.5 * (43000 - 42500)
    
    def test_calculate_pnl_long_loss(self):
        """Test PnL calculation for long position with loss."""
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42500.0,
            current_price=42000.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        
        pnl = PortfolioManager._calculate_pnl(position)
        
        assert pnl == -250.0  # 0.5 * (42000 - 42500)
