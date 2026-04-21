from __future__ import annotations

from typing import Optional

try:
    from its_project.decision.decision import Signal, Decision, Action
    from its_project.decision.simple import SimpleDecisionMaker
    from its_project.decision.risk import RiskManager
    from its_project.decision.sizer import PositionSizer
    from its_project.decision.portfolio import PortfolioManager
except ImportError:
    from .decision import Signal, Decision, Action
    from .simple import SimpleDecisionMaker
    from .risk import RiskManager
    from .sizer import PositionSizer
    from .portfolio import PortfolioManager


class TradingDecisionEngine:
    """Full trading decision engine."""

    def __init__(self, config: dict) -> None:
        self.decision_maker = SimpleDecisionMaker(config["decision"])
        self.risk_manager = RiskManager(config["risk"])
        self.position_sizer = PositionSizer(config["sizing"])
        self.portfolio = PortfolioManager(config["portfolio"])

    def process_signal(
        self,
        signal: Signal,
        market_state: dict,
        account_balance: float,
    ) -> Optional[Decision]:
        """
        Process signal and make decision.

        Pipeline: Signal → Decision → Risk Check → Sizing → Final Decision
        """
        # Step 1: Decision Maker primary decision
        decision = self.decision_maker.decide(signal, market_state)
        if decision is None or decision.action == Action.HOLD:
            return None

        # Step 2: Portfolio check (can open position?)
        if not self.portfolio.can_open_position(signal.symbol):
            return None

        # Step 3: Position sizing
        position_size = self.position_sizer.calculate_size(
            signal, account_balance
        )
        decision.size = position_size

        # Step 4: Risk Manager validation
        allowed, reason = self.risk_manager.check_decision(
            decision,
            list(self.portfolio.positions.values()),
            account_balance,
        )
        if not allowed:
            print(f"Decision rejected: {reason}")
            return None

        # Step 5: Final decision
        return decision
