from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from its_project.decision.decision import Decision
except ImportError:
    from .decision import Decision


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


class RiskManager:
    """Risk management."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_position_size = config.get("max_position_size", 0.1)
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.05)
        self.max_drawdown = config.get("max_drawdown", 0.15)
        self.max_correlation = config.get("max_correlation", 0.7)
        self.current_drawdown = 0.0
        self.peak_equity = 0.0

    def check_decision(
        self,
        decision: Decision,
        current_positions: List[Position],
        account_balance: float,
    ) -> Tuple[bool, str]:
        """Check decision against risk limits."""
        if decision.size > self.max_position_size:
            return False, f"Size {decision.size} > max {self.max_position_size}"
        total_risk = self._calculate_portfolio_risk(current_positions, decision, account_balance)
        if total_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {total_risk:.2%} > max {self.max_portfolio_risk:.2%}"
        if self.current_drawdown > self.max_drawdown:
            return False, f"Drawdown {self.current_drawdown:.2%} > max {self.max_drawdown:.2%}"
        if self._high_correlation(decision, current_positions):
            return False, "High correlation with existing positions"
        return True, "OK"

    def _calculate_portfolio_risk(
        self,
        positions: List[Position],
        new_decision: Decision,
        account_balance: float,
    ) -> float:
        """Calculate total portfolio risk."""
        total_risk = 0.0
        for pos in positions:
            position_value = pos.size * pos.current_price
            risk_per_position = position_value * 0.02  # 2% risk per position
            total_risk += risk_per_position
        if new_decision.stop_loss:
            potential_loss = abs(
                new_decision.size * (new_decision.stop_loss - (new_decision.price or 0))
            )
            total_risk += potential_loss
        return total_risk / account_balance

    def _high_correlation(
        self,
        decision: Decision,
        positions: List[Position],
    ) -> bool:
        """Simple correlation check (same symbol)."""
        for pos in positions:
            if pos.symbol == decision.symbol:
                return True
        return False

    def update_drawdown(self, current_equity: float) -> None:
        """Update current drawdown."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
