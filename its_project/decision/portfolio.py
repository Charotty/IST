from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from its_project.decision.risk import Position
except ImportError:
    from .risk import Position


@dataclass
class Position:
    """Current position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: int


class PortfolioManager:
    """Portfolio management."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.max_positions = config.get("max_positions", 5)

    def can_open_position(self, symbol: str) -> bool:
        """Check if new position can be opened."""
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.max_positions:
            return False
        return True

    def add_position(self, position: Position) -> None:
        """Add position."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove position."""
        return self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position."""
        return self.positions.get(symbol)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update prices and PnL for all positions."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.unrealized_pnl = self._calculate_pnl(position)

    def get_total_pnl(self) -> float:
        """Total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_exposure(self) -> Dict[str, float]:
        """Exposure per position."""
        return {
            symbol: pos.size * pos.current_price
            for symbol, pos in self.positions.items()
        }

    @staticmethod
    def _calculate_pnl(position: Position) -> float:
        """Calculate PnL for a position."""
        if position.side == "long":
            return position.size * (position.current_price - position.entry_price)
        else:  # short
            return position.size * (position.entry_price - position.current_price)
