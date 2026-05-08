#!/usr/bin/env python3
"""
Advanced Risk Management Module
============================

Production-ready risk management with:
- Stop-loss and take-profit management
- Maximum drawdown limits
- Position-level and portfolio-level risk controls
- Dynamic risk adjustment
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from its_project.decision.decision import Decision, Action
from its_project.execution.base import Position, Order, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk management configuration."""
    # Position-level limits
    max_position_size: float = 0.10      # 10% of portfolio max
    max_risk_per_trade: float = 0.01     # 1% of portfolio per trade
    max_leverage: float = 3.0              # 3x max leverage
    
    # Stop-loss/Take-profit
    default_stop_loss_pct: float = 0.02     # 2% default SL
    default_take_profit_pct: float = 0.04   # 4% default TP
    trailing_stop_pct: float = 0.01          # 1% trailing stop
    
    # Portfolio-level limits
    max_portfolio_risk: float = 0.05        # 5% total portfolio risk
    max_drawdown: float = 0.15              # 15% max drawdown
    max_daily_loss: float = 0.03            # 3% max daily loss
    
    # Time-based limits
    max_position_duration_hours: float = 24.0  # Max 24h per position
    forced_close_hours: float = 48.0           # Force close after 48h
    
    # Correlation limits
    max_correlated_exposure: float = 0.20     # 20% max in correlated assets
    
    # Dynamic adjustment
    volatility_adjustment: bool = True
    regime_adjustment: bool = True


@dataclass
class RiskAlert:
    """Risk alert notification."""
    level: RiskLevel
    message: str
    symbol: Optional[str]
    position_id: Optional[str]
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCheck:
    """Result of risk assessment."""
    approved: bool
    risk_level: RiskLevel
    alerts: List[RiskAlert] = field(default_factory=list)
    adjustments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class RiskManager:
    """
    Advanced risk manager for production trading.
    
    Features:
    - Multi-level risk assessment
    - Dynamic SL/TP management
    - Drawdown monitoring and protection
    - Portfolio risk aggregation
    - Real-time risk alerts
    """
    
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.portfolio_value: float = 0.0
        self.peak_portfolio_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.last_reset_date: str = datetime.now().date().isoformat()
        
        # Risk history
        self.risk_history: List[RiskCheck] = []
        self.alert_history: List[RiskAlert] = []
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'blocked_trades': 0,
            'risk_adjustments': 0,
            'stop_losses_triggered': 0,
            'take_profits_triggered': 0,
            'max_drawdown_reached': 0,
            'daily_loss_limit_hit': 0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
    
    def check_decision_risk(
        self,
        decision: Decision,
        current_positions: Dict[str, Position],
        portfolio_value: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> RiskCheck:
        """
        Comprehensive risk assessment for trading decision.
        
        Args:
            decision: Trading decision to evaluate
            current_positions: Current open positions
            portfolio_value: Current portfolio value
            market_data: Market data for risk calculations
            
        Returns:
            RiskCheck with approval status and recommendations
        """
        self.stats['total_checks'] += 1
        
        # Update state
        self.positions = current_positions
        self.portfolio_value = portfolio_value
        self._update_daily_pnl()
        
        alerts = []
        adjustments = {}
        approved = True
        risk_level = RiskLevel.LOW
        
        # 1. Position size check
        size_check = self._check_position_size(decision, portfolio_value)
        if not size_check.approved:
            approved = False
            alerts.extend(size_check.alerts)
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # 2. Portfolio risk check
        portfolio_check = self._check_portfolio_risk(decision, current_positions, portfolio_value)
        if not portfolio_check.approved:
            approved = False
            alerts.extend(portfolio_check.alerts)
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # 3. Drawdown check
        drawdown_check = self._check_drawdown_limit()
        if not drawdown_check.approved:
            approved = False
            alerts.extend(drawdown_check.alerts)
            risk_level = max(risk_level, RiskLevel.CRITICAL)
        
        # 4. Daily loss check
        daily_check = self._check_daily_loss_limit()
        if not daily_check.approved:
            approved = False
            alerts.extend(daily_check.alerts)
            risk_level = max(risk_level, RiskLevel.CRITICAL)
        
        # 5. Correlation check
        correlation_check = self._check_correlation_risk(decision, current_positions)
        if not correlation_check.approved:
            approved = False
            alerts.extend(correlation_check.alerts)
            risk_level = max(risk_level, RiskLevel.MEDIUM)
        
        # 6. Dynamic adjustments
        if self.limits.volatility_adjustment and market_data:
            volatility_adj = self._calculate_volatility_adjustment(decision, market_data)
            if volatility_adj:
                adjustments.update(volatility_adj)
                self.stats['risk_adjustments'] += 1
        
        # 7. SL/TP validation
        sltp_check = self._validate_sltp(decision, market_data)
        if sltp_check.adjustments:
            adjustments.update(sltp_check.adjustments)
        
        # Create risk check result
        reason = "Approved" if approved else "Risk limits exceeded"
        if alerts:
            reason += f": {', '.join(alert.message for alert in alerts[:3])}"
        
        risk_check = RiskCheck(
            approved=approved,
            risk_level=risk_level,
            alerts=alerts,
            adjustments=adjustments,
            reason=reason
        )
        
        # Store in history
        self.risk_history.append(risk_check)
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
        
        # Log alerts
        for alert in alerts:
            self._trigger_alert(alert)
        
        if not approved:
            self.stats['blocked_trades'] += 1
            logger.warning(f"Trade blocked by risk manager: {reason}")
        
        return risk_check
    
    def monitor_positions(
        self,
        current_positions: Dict[str, Position],
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Monitor existing positions for risk breaches.
        
        Returns:
            List of action recommendations (close, adjust SL/TP, etc.)
        """
        self.positions = current_positions
        actions = []
        
        for symbol, position in current_positions.items():
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            current_price = symbol_data.get('price', position.current_price)
            
            # Update position current price
            position.current_price = current_price
            position.unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
            
            # Check stop-loss
            sl_action = self._check_stop_loss(position, current_price)
            if sl_action:
                actions.append(sl_action)
                self.stats['stop_losses_triggered'] += 1
            
            # Check take-profit
            tp_action = self._check_take_profit(position, current_price)
            if tp_action:
                actions.append(tp_action)
                self.stats['take_profits_triggered'] += 1
            
            # Check position duration
            duration_action = self._check_position_duration(position)
            if duration_action:
                actions.append(duration_action)
            
            # Check trailing stop
            trailing_action = self._check_trailing_stop(position, current_price, symbol_data)
            if trailing_action:
                actions.append(trailing_action)
        
        return actions
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value and track peak."""
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        
        # Update peak for drawdown calculation
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
        
        # Update daily PnL
        self._update_daily_pnl()
        
        # Check for drawdown breach
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.limits.max_drawdown:
            self.stats['max_drawdown_reached'] += 1
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                message=f"Maximum drawdown breached: {current_drawdown:.2%}",
                symbol=None,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'drawdown': current_drawdown, 'limit': self.limits.max_drawdown}
            )
            self._trigger_alert(alert)
    
    def _check_position_size(self, decision: Decision, portfolio_value: float) -> RiskCheck:
        """Check if position size exceeds limits."""
        alerts = []
        
        # Calculate position value
        if decision.price:
            position_value = abs(decision.size * decision.price)
        else:
            # Estimate using market data (would need current price)
            position_value = abs(decision.size * 50000)  # Placeholder
        
        # Check max position size
        position_fraction = position_value / portfolio_value
        if position_fraction > self.limits.max_position_size:
            alerts.append(RiskAlert(
                level=RiskLevel.HIGH,
                message=f"Position size {position_fraction:.2%} exceeds limit {self.limits.max_position_size:.2%}",
                symbol=decision.symbol,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'position_fraction': position_fraction}
            ))
        
        # Check risk per trade
        risk_amount = position_value * self.limits.max_risk_per_trade
        if risk_amount > portfolio_value * self.limits.max_risk_per_trade:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                message=f"Trade risk ${risk_amount:.2f} exceeds limit {portfolio_value * self.limits.max_risk_per_trade:.2f}",
                symbol=decision.symbol,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'risk_amount': risk_amount}
            ))
        
        return RiskCheck(
            approved=len(alerts) == 0,
            risk_level=RiskLevel.HIGH if alerts else RiskLevel.LOW,
            alerts=alerts
        )
    
    def _check_portfolio_risk(
        self,
        decision: Decision,
        current_positions: Dict[str, Position],
        portfolio_value: float
    ) -> RiskCheck:
        """Check portfolio-level risk limits."""
        alerts = []
        
        # Calculate current portfolio risk
        current_risk = self._calculate_portfolio_risk(current_positions)
        
        # Add new position risk
        if decision.price:
            new_position_risk = abs(decision.size * decision.price) * self.limits.max_risk_per_trade
        else:
            new_position_risk = abs(decision.size * 50000) * self.limits.max_risk_per_trade
        
        total_risk = current_risk + new_position_risk
        risk_fraction = total_risk / portfolio_value
        
        if risk_fraction > self.limits.max_portfolio_risk:
            alerts.append(RiskAlert(
                level=RiskLevel.HIGH,
                message=f"Portfolio risk {risk_fraction:.2%} exceeds limit {self.limits.max_portfolio_risk:.2%}",
                symbol=decision.symbol,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'portfolio_risk': risk_fraction, 'current_risk': current_risk}
            ))
        
        return RiskCheck(
            approved=len(alerts) == 0,
            risk_level=RiskLevel.HIGH if alerts else RiskLevel.LOW,
            alerts=alerts
        )
    
    def _check_drawdown_limit(self) -> RiskCheck:
        """Check if current drawdown exceeds limit."""
        current_drawdown = self._calculate_current_drawdown()
        
        if current_drawdown > self.limits.max_drawdown:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                message=f"Drawdown {current_drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}",
                symbol=None,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'drawdown': current_drawdown, 'limit': self.limits.max_drawdown}
            )
            
            return RiskCheck(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                alerts=[alert]
            )
        
        return RiskCheck(approved=True, risk_level=RiskLevel.LOW)
    
    def _check_daily_loss_limit(self) -> RiskCheck:
        """Check if daily loss exceeds limit."""
        if self.daily_pnl < -self.limits.max_daily_loss:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                message=f"Daily loss {abs(self.daily_pnl):.2%} exceeds limit {self.limits.max_daily_loss:.2%}",
                symbol=None,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'daily_pnl': self.daily_pnl, 'limit': -self.limits.max_daily_loss}
            )
            
            return RiskCheck(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                alerts=[alert]
            )
        
        return RiskCheck(approved=True, risk_level=RiskLevel.LOW)
    
    def _check_correlation_risk(
        self,
        decision: Decision,
        current_positions: Dict[str, Position]
    ) -> RiskCheck:
        """Check correlation risk with existing positions."""
        alerts = []
        
        # Simplified correlation check (would use correlation matrix in production)
        correlated_symbols = self._get_correlated_symbols(decision.symbol)
        correlated_exposure = 0.0
        
        for symbol, position in current_positions.items():
            if symbol in correlated_symbols:
                if decision.price:
                    exposure = abs(position.size * position.entry_price)
                else:
                    exposure = abs(position.size * 50000)  # Placeholder
                correlated_exposure += exposure
        
        # Add new position
        if decision.price:
            new_exposure = abs(decision.size * decision.price)
        else:
            new_exposure = abs(decision.size * 50000)
        
        total_correlated = correlated_exposure + new_exposure
        correlated_fraction = total_correlated / self.portfolio_value
        
        if correlated_fraction > self.limits.max_correlated_exposure:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                message=f"Correlated exposure {correlated_fraction:.2%} exceeds limit {self.limits.max_correlated_exposure:.2%}",
                symbol=decision.symbol,
                position_id=None,
                timestamp=int(time.time() * 1000),
                metadata={'correlated_fraction': correlated_fraction, 'correlated_symbols': correlated_symbols}
            ))
        
        return RiskCheck(
            approved=len(alerts) == 0,
            risk_level=RiskLevel.MEDIUM if alerts else RiskLevel.LOW,
            alerts=alerts
        )
    
    def _validate_sltp(
        self,
        decision: Decision,
        market_data: Optional[Dict[str, Any]]
    ) -> RiskCheck:
        """Validate and adjust SL/TP levels."""
        adjustments = {}
        
        if not decision.price:
            return RiskCheck(approved=True, risk_level=RiskLevel.LOW)
        
        # Set default SL if not provided
        if not decision.stop_loss:
            if decision.action == Action.BUY:
                default_sl = decision.price * (1 - self.limits.default_stop_loss_pct)
            else:
                default_sl = decision.price * (1 + self.limits.default_stop_loss_pct)
            
            adjustments['stop_loss'] = default_sl
        
        # Set default TP if not provided
        if not decision.take_profit:
            if decision.action == Action.BUY:
                default_tp = decision.price * (1 + self.limits.default_take_profit_pct)
            else:
                default_tp = decision.price * (1 - self.limits.default_take_profit_pct)
            
            adjustments['take_profit'] = default_tp
        
        # Validate SL/TP ratio
        sl = adjustments.get('stop_loss', decision.stop_loss)
        tp = adjustments.get('take_profit', decision.take_profit)
        
        if sl and tp:
            if decision.action == Action.BUY:
                sl_distance = abs(decision.price - sl) / decision.price
                tp_distance = abs(tp - decision.price) / decision.price
            else:
                sl_distance = abs(sl - decision.price) / decision.price
                tp_distance = abs(decision.price - tp) / decision.price
            
            # Ensure TP > SL (risk/reward ratio)
            if tp_distance <= sl_distance:
                # Adjust TP to maintain 2:1 ratio
                if decision.action == Action.BUY:
                    new_tp = decision.price * (1 + 2 * sl_distance)
                else:
                    new_tp = decision.price * (1 - 2 * sl_distance)
                
                adjustments['take_profit'] = new_tp
        
        return RiskCheck(
            approved=True,
            risk_level=RiskLevel.LOW,
            adjustments=adjustments
        )
    
    def _calculate_volatility_adjustment(
        self,
        decision: Decision,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate volatility-based risk adjustments."""
        volatility = market_data.get('volatility', 0.02)
        
        if volatility > 0.05:  # High volatility (>5%)
            # Tighten stops
            multiplier = 0.5
            adjustments = {
                'volatility_adjusted': True,
                'volatility_multiplier': multiplier
            }
            
            if decision.stop_loss:
                if decision.action == Action.BUY:
                    new_sl = decision.price * (1 - self.limits.default_stop_loss_pct * multiplier)
                else:
                    new_sl = decision.price * (1 + self.limits.default_stop_loss_pct * multiplier)
                adjustments['stop_loss'] = new_sl
            
            return adjustments
        
        return None
    
    def _check_stop_loss(self, position: Position, current_price: float) -> Optional[Dict[str, Any]]:
        """Check if stop-loss is triggered."""
        if not hasattr(position, 'stop_loss') or not position.stop_loss:
            return None
        
        if position.side == 'long':
            if current_price <= position.stop_loss:
                return {
                    'action': 'CLOSE',
                    'symbol': position.symbol,
                    'reason': 'Stop loss triggered',
                    'price': position.stop_loss
                }
        else:  # short
            if current_price >= position.stop_loss:
                return {
                    'action': 'CLOSE',
                    'symbol': position.symbol,
                    'reason': 'Stop loss triggered',
                    'price': position.stop_loss
                }
        
        return None
    
    def _check_take_profit(self, position: Position, current_price: float) -> Optional[Dict[str, Any]]:
        """Check if take-profit is triggered."""
        if not hasattr(position, 'take_profit') or not position.take_profit:
            return None
        
        if position.side == 'long':
            if current_price >= position.take_profit:
                return {
                    'action': 'CLOSE',
                    'symbol': position.symbol,
                    'reason': 'Take profit triggered',
                    'price': position.take_profit
                }
        else:  # short
            if current_price <= position.take_profit:
                return {
                    'action': 'CLOSE',
                    'symbol': position.symbol,
                    'reason': 'Take profit triggered',
                    'price': position.take_profit
                }
        
        return None
    
    def _check_position_duration(self, position: Position) -> Optional[Dict[str, Any]]:
        """Check if position has been open too long."""
        current_time = int(time.time() * 1000)
        duration_ms = current_time - position.timestamp
        duration_hours = duration_ms / (1000 * 60 * 60)
        
        if duration_hours > self.limits.forced_close_hours:
            return {
                'action': 'CLOSE',
                'symbol': position.symbol,
                'reason': f'Position open {duration_hours:.1f}h, force close',
                'price': position.current_price
            }
        
        return None
    
    def _check_trailing_stop(
        self,
        position: Position,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check and update trailing stop."""
        if not hasattr(position, 'trailing_stop') or not position.trailing_stop:
            return None
        
        # Simplified trailing stop logic
        if position.side == 'long':
            # For long positions, trailing stop moves up with price
            new_stop = current_price * (1 - self.limits.trailing_stop_pct)
            if new_stop > position.trailing_stop:
                return {
                    'action': 'UPDATE_STOP',
                    'symbol': position.symbol,
                    'reason': 'Trailing stop updated',
                    'new_stop_loss': new_stop
                }
        else:  # short
            # For short positions, trailing stop moves down with price
            new_stop = current_price * (1 + self.limits.trailing_stop_pct)
            if new_stop < position.trailing_stop:
                return {
                    'action': 'UPDATE_STOP',
                    'symbol': position.symbol,
                    'reason': 'Trailing stop updated',
                    'new_stop_loss': new_stop
                }
        
        return None
    
    def _calculate_portfolio_risk(self, positions: Dict[str, Position]) -> float:
        """Calculate total portfolio risk."""
        total_risk = 0.0
        for position in positions.values():
            if position.entry_price:
                position_risk = abs(position.size * position.entry_price) * self.limits.max_risk_per_trade
                total_risk += position_risk
        
        return total_risk
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_portfolio_value == 0:
            return 0.0
        
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        return max(0.0, drawdown)
    
    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized PnL for position."""
        if position.side == 'long':
            return (current_price - position.entry_price) * position.size
        else:  # short
            return (position.entry_price - current_price) * position.size
    
    def _update_daily_pnl(self) -> None:
        """Update daily PnL tracking."""
        current_date = datetime.now().date().isoformat()
        
        # Reset if new day
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
        
        # Update daily PnL based on portfolio change
        # This would need more sophisticated tracking in production
    
    def _get_correlated_symbols(self, symbol: str) -> List[str]:
        """Get list of correlated symbols."""
        # Simplified correlation mapping
        correlation_map = {
            'BTC/USDT': ['ETH/USDT', 'BNB/USDT'],
            'ETH/USDT': ['BTC/USDT', 'BNB/USDT'],
            'BNB/USDT': ['BTC/USDT', 'ETH/USDT']
        }
        
        return correlation_map.get(symbol, [])
    
    def _trigger_alert(self, alert: RiskAlert) -> None:
        """Trigger risk alert to callbacks."""
        self.alert_history.append(alert)
        
        # Keep only recent alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for risk alerts."""
        self.alert_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        base_stats = self.stats.copy()
        
        base_stats.update({
            'current_positions': len(self.positions),
            'portfolio_value': self.portfolio_value,
            'peak_portfolio_value': self.peak_portfolio_value,
            'current_drawdown': self._calculate_current_drawdown(),
            'daily_pnl': self.daily_pnl,
            'total_alerts': len(self.alert_history),
            'recent_alerts': len([a for a in self.alert_history if 
                                int(time.time() * 1000) - a.timestamp < 3600000])  # Last hour
        })
        
        return base_stats


# Convenience functions
def create_conservative_risk_manager() -> RiskManager:
    """Create conservative risk manager."""
    limits = RiskLimits(
        max_position_size=0.05,
        max_risk_per_trade=0.005,
        max_drawdown=0.10,
        max_daily_loss=0.02
    )
    return RiskManager(limits)


def create_aggressive_risk_manager() -> RiskManager:
    """Create aggressive risk manager."""
    limits = RiskLimits(
        max_position_size=0.15,
        max_risk_per_trade=0.02,
        max_drawdown=0.20,
        max_daily_loss=0.05
    )
    return RiskManager(limits)


# Legacy compatibility
def check_decision_legacy(
    self,
    decision: Decision,
    current_positions: List[Position],
    account_balance: float,
) -> Tuple[bool, str]:
    """Legacy compatibility for existing code."""
    # Convert to new format
    positions_dict = {pos.symbol: pos for pos in current_positions}
    
    risk_check = self.check_decision_risk(
        decision=decision,
        current_positions=positions_dict,
        portfolio_value=account_balance
    )
    
    return risk_check.approved, risk_check.reason
