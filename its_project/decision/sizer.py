from __future__ import annotations

from typing import Dict, Any

try:
    from its_project.decision.decision import Signal
except ImportError:
    from .decision import Signal


class PositionSizer:
    """Position sizing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.method = config.get("method", "fixed")
        self.config = config

    def calculate_size(
        self,
        signal: Signal,
        account_balance: float,
        risk_per_trade: float = 0.02,
    ) -> float:
        """Calculate position size."""
        if self.method == "fixed":
            return self._fixed_size()
        elif self.method == "fixed_fraction":
            return self._fixed_fraction(account_balance)
        elif self.method == "kelly":
            return self._kelly_criterion(signal, account_balance)
        elif self.method == "risk_based":
            return self._risk_based(signal, account_balance, risk_per_trade)
        else:
            raise ValueError(f"Unknown sizing method: {self.method}")

    def _fixed_size(self) -> float:
        """Fixed size."""
        return self.config.get("size", 0.01)

    def _fixed_fraction(self, balance: float) -> float:
        """Fixed fraction of balance."""
        fraction = self.config.get("fraction", 0.02)
        return balance * fraction

    def _kelly_criterion(self, signal: Signal, balance: float) -> float:
        """Kelly Criterion sizing."""
        win_prob = signal.confidence
        win_loss_ratio = self.config.get("win_loss_ratio", 2.0)
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fraction = max(0, kelly_fraction)
        conservative_fraction = kelly_fraction * self.config.get("kelly_fraction", 0.25)
        return balance * conservative_fraction

    def _risk_based(
        self,
        signal: Signal,
        balance: float,
        risk_per_trade: float,
    ) -> float:
        """Risk-based sizing."""
        entry_price = signal.metadata.get("price", 100)
        stop_distance_pct = self.config.get("stop_loss_pct", 0.02)
        stop_distance = entry_price * stop_distance_pct
        risk_amount = balance * risk_per_trade
        return risk_amount / stop_distance
