from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from its_project.decision.decision import Decision, Position

logger = logging.getLogger(__name__)


class RiskManager:
    """Enhanced risk manager with comprehensive risk controls."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Position-level risk limits
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% of portfolio
        self.max_position_value = config.get("max_position_value", 10000.0)  # Max USD value
        self.max_leverage = config.get("max_leverage", 1.0)  # No leverage by default
        
        # Portfolio-level risk limits
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.05)  # 5% of portfolio
        self.max_total_exposure = config.get("max_total_exposure", 0.5)  # 50% max exposure
        self.max_correlation = config.get("max_correlation", 0.7)  # Max correlation between positions
        
        # Drawdown controls
        self.max_drawdown = config.get("max_drawdown", 0.15)  # 15% max drawdown
        self.daily_loss_limit = config.get("daily_loss_limit", 0.02)  # 2% daily loss limit
        self.consecutive_loss_limit = config.get("consecutive_loss_limit", 3)  # Max consecutive losing days
        
        # Stop loss and take profit
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.02)  # 2%
        self.default_take_profit_pct = config.get("default_take_profit_pct", 0.04)  # 4%
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.01)  # 1%
        self.atr_multiplier = config.get("atr_multiplier", 2.0)  # ATR-based stops
        
        # Risk metrics tracking
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.position_history: List[Dict[str, Any]] = []
        
        # Volatility-based sizing
        self.volatility_window = config.get("volatility_window", 20)
        self.volatility_target = config.get("volatility_target", 0.15)  # 15% annual vol
        self.position_risk_pct = config.get("position_risk_pct", 0.02)  # 2% risk per position
    
    def check_decision(
        self,
        decision: Decision,
        positions: List[Position],
        account_balance: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str]:
        """
        Check if decision passes all risk controls.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Update peak equity
        if account_balance > self.peak_equity:
            self.peak_equity = account_balance
        
        # Calculate current drawdown
        self.current_drawdown = (self.peak_equity - account_balance) / self.peak_equity
        
        # Check 1: Maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}"
        
        # Check 2: Daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * self.peak_equity:
            return False, f"Daily loss limit exceeded: {abs(self.daily_pnl):.2f}"
        
        # Check 3: Consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check 4: Position size limits
        if not self._check_position_size(decision, account_balance):
            return False, f"Position size exceeds limits: {decision.size}"
        
        # Check 5: Portfolio risk limits
        if not self._check_portfolio_risk(decision, positions, account_balance):
            return False, "Portfolio risk exceeds limits"
        
        # Check 6: Correlation limits
        if not self._check_correlation(decision, positions):
            return False, "Position correlation too high"
        
        # Check 7: Market-specific risks
        if market_data and not self._check_market_conditions(decision, market_data):
            return False, "Unfavorable market conditions"
        
        return True, "Decision approved"
    
    def _check_position_size(self, decision: Decision, account_balance: float) -> bool:
        """Check if position size is within limits."""
        position_value = decision.size * (decision.price or 0)
        
        # Check percentage of portfolio
        if decision.size > self.max_position_size:
            return False
        
        # Check absolute value
        if position_value > self.max_position_value:
            return False
        
        # Check leverage (if short position)
        if decision.action.value in ["sell", "close_long"] and abs(decision.size) > self.max_leverage:
            return False
        
        return True
    
    def _check_portfolio_risk(self, decision: Decision, positions: List[Position], account_balance: float) -> bool:
        """Check portfolio-level risk limits."""
        # Calculate total exposure
        total_exposure = sum(abs(pos.size) for pos in positions)
        if decision.action.value not in ["hold", "close_long", "close_short"]:
            total_exposure += abs(decision.size)
        
        if total_exposure > self.max_total_exposure:
            return False
        
        # Calculate portfolio risk
        portfolio_risk = total_exposure / account_balance
        if portfolio_risk > self.max_portfolio_risk:
            return False
        
        return True
    
    def _check_correlation(self, decision: Decision, positions: List[Position]) -> bool:
        """Check position correlation limits."""
        # Simplified correlation check
        # In practice, you'd calculate actual correlation based on historical returns
        same_asset_positions = [pos for pos in positions if pos.symbol == decision.symbol]
        
        if len(same_asset_positions) > 0:
            # Already have position in this asset
            return False
        
        return True
    
    def _check_market_conditions(self, decision: Decision, market_data: Dict[str, Any]) -> bool:
        """Check market-specific risk conditions."""
        # Check volatility
        if "volatility" in market_data:
            volatility = market_data["volatility"]
            if volatility > 0.05:  # 5% volatility is very high
                logger.warning(f"High volatility detected: {volatility:.2%}")
                # Could reduce position size or reject
        
        # Check spread
        if "spread" in market_data:
            spread = market_data["spread"]
            max_spread = 0.001  # 0.1%
            if spread > max_spread:
                return False
        
        # Check volume
        if "volume" in market_data:
            volume = market_data["volume"]
            min_volume = 1000000  # $1M minimum volume
            if volume < min_volume:
                return False
        
        return True
    
    def calculate_position_size(
        self,
        signal_strength: float,
        volatility: Optional[float] = None,
        account_balance: float = 10000.0,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            signal_strength: Signal confidence (0-1)
            volatility: Market volatility
            account_balance: Available capital
            atr: Average True Range for stop loss calculation
            
        Returns:
            Position size as percentage of portfolio
        """
        # Base position size
        base_size = self.max_position_size * signal_strength
        
        # Volatility adjustment
        if volatility is not None:
            # Reduce size in high volatility
            vol_adjustment = min(1.0, self.volatility_target / (volatility * np.sqrt(252)))
            base_size *= vol_adjustment
        
        # ATR-based position sizing
        if atr is not None and account_balance > 0:
            # Risk 2% of capital per trade
            risk_amount = account_balance * self.position_risk_pct
            atr_size = risk_amount / (atr * self.atr_multiplier)
            base_size = min(base_size, atr_size / account_balance)
        
        # Apply maximum limits
        final_size = min(base_size, self.max_position_size)
        
        return max(0.0, final_size)
    
    def set_stop_loss_take_profit(
        self,
        decision: Decision,
        entry_price: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Returns:
            (stop_loss_price, take_profit_price)
        """
        # ATR-based stops
        if atr is not None:
            stop_distance = atr * self.atr_multiplier
            stop_loss = entry_price - stop_distance if decision.action.value == "buy" else entry_price + stop_distance
            take_profit = entry_price + (stop_distance * 2) if decision.action.value == "buy" else entry_price - (stop_distance * 2)
        else:
            # Percentage-based stops
            stop_loss = entry_price * (1 - self.default_stop_loss_pct) if decision.action.value == "buy" else entry_price * (1 + self.default_stop_loss_pct)
            take_profit = entry_price * (1 + self.default_take_profit_pct) if decision.action.value == "buy" else entry_price * (1 - self.default_take_profit_pct)
        
        # Volatility adjustment
        if volatility is not None:
            vol_factor = min(2.0, max(0.5, volatility / 0.02))  # Normalize around 2%
            stop_distance = abs(entry_price - stop_loss) * vol_factor
            stop_loss = entry_price - stop_distance if decision.action.value == "buy" else entry_price + stop_distance
        
        return stop_loss, take_profit
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L and consecutive loss counter."""
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (called at start of new day)."""
        self.daily_pnl = 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return {
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "max_position_size": self.max_position_size,
            "max_portfolio_risk": self.max_portfolio_risk,
            "max_drawdown_limit": self.max_drawdown,
            "risk_utilization": self.current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0
        }
    
    def should_reduce_risk(self) -> bool:
        """Check if risk should be reduced due to recent performance."""
        # Reduce risk if drawdown is high
        if self.current_drawdown > self.max_drawdown * 0.8:
            return True
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit - 1:
            return True
        
        # Reduce risk if daily losses are high
        if self.daily_pnl < -self.daily_loss_limit * self.peak_equity * 0.8:
            return True
        
        return False
    
    def get_risk_reduction_factor(self) -> float:
        """Get factor by which to reduce position sizes."""
        if self.current_drawdown > 0.1:  # 10% drawdown
            return 0.5
        elif self.consecutive_losses >= 2:
            return 0.7
        elif self.daily_pnl < -self.daily_loss_limit * self.peak_equity * 0.5:
            return 0.6
        else:
            return 1.0
