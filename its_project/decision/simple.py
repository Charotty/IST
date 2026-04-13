from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from its_project.decision.decision import BaseDecisionMaker, Signal, Decision, Action


class SimpleDecisionMaker(BaseDecisionMaker):
    """Simple decision maker with fixed confidence threshold."""

    def decide(
        self,
        signal: Signal,
        market_state: Dict[str, Any],
    ) -> Optional[Decision]:
        """Make decision."""
        if not self.validate_signal(signal):
            return None
        if signal.confidence < self.confidence_threshold:
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Confidence {signal.confidence:.2f} < threshold {self.confidence_threshold}",
            )
        position_size = self._calculate_position_size(signal, market_state)
        stop_loss, take_profit = self._calculate_risk_levels(signal, market_state)
        return Decision(
            action=signal.action,
            symbol=signal.symbol,
            size=position_size,
            price=None,  # Market order
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.timestamp,
            reason=f"Confidence: {signal.confidence:.2f}",
        )

    def _calculate_position_size(
        self,
        signal: Signal,
        market_state: Dict[str, Any],
    ) -> float:
        """Calculate position size."""
        base_size = self.config.get("position_size", 0.01)
        return base_size * signal.confidence

    def _calculate_risk_levels(
        self,
        signal: Signal,
        market_state: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit."""
        current_price = market_state.get("price", 0)
        if current_price == 0:
            return None, None
        stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
        take_profit_pct = self.config.get("take_profit_pct", 0.04)
        if signal.action == Action.BUY:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif signal.action == Action.SELL:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            return None, None
        return stop_loss, take_profit
