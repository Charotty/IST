#!/usr/bin/env python3
"""
Position Sizing Module
====================

Production-ready position sizing with multiple strategies:
- Fixed fraction sizing
- Kelly criterion
- Volatility-based sizing
- Risk-adjusted sizing
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from its_project.decision.decision import Decision, Action
from its_project.execution.base import Position

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTION = "fixed_fraction"
    FIXED_PERCENTAGE = "fixed_percentage"  # Fixed 1-2% sizing
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    method: SizingMethod
    base_fraction: float = 0.02  # 2% base position size
    max_fraction: float = 0.10    # 10% maximum position size
    min_fraction: float = 0.001   # 0.1% minimum position size
    
    # Fixed percentage sizing parameters (1-2%)
    fixed_percentage_min: float = 0.01   # 1% minimum fixed percentage
    fixed_percentage_max: float = 0.02   # 2% maximum fixed percentage
    fixed_percentage_default: float = 0.015  # 1.5% default fixed percentage
    
    # Kelly-specific parameters
    kelly_fraction: float = 0.25   # Fraction of full Kelly (conservative)
    win_rate_threshold: float = 0.55  # Minimum win rate for Kelly
    
    # Volatility-specific parameters
    volatility_window: int = 20
    volatility_target: float = 0.02   # 2% daily volatility target
    volatility_scaling: bool = True
    
    # Risk parameters
    max_risk_per_trade: float = 0.01   # 1% max risk per trade
    max_portfolio_risk: float = 0.05    # 5% max portfolio risk
    
    # Adaptive parameters
    adaptive_window: int = 50
    adaptive_rebalance_freq: int = 10


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    size: float
    size_fraction: float  # As fraction of portfolio
    risk_amount: float    # Amount at risk
    method_used: str
    confidence_adjusted: bool
    metadata: Dict[str, Any]


class PositionSizer:
    """
    Advanced position sizer with multiple strategies.
    
    Features:
    - Multiple sizing methods
    - Risk limits enforcement
    - Confidence-based adjustment
    - Volatility scaling
    - Portfolio-level risk management
    """
    
    def __init__(self, config: SizingConfig) -> None:
        self.config = config
        
        # History for adaptive sizing
        self.trade_history: Dict[str, list] = {}
        self.portfolio_history: list = []
        
        # Performance tracking
        self.stats = {
            'total_sizing_decisions': 0,
            'risk_limited_trades': 0,
            'confidence_adjustments': 0,
            'method_usage': {method.value: 0 for method in SizingMethod}
        }
    
    def calculate_position_size(
        self,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position],
        market_data: Optional[Dict[str, Any]] = None
    ) -> SizingResult:
        """
        Calculate optimal position size for a decision.
        
        Args:
            decision: Trading decision
            portfolio_value: Current portfolio value
            current_positions: Current open positions
            market_data: Market data for calculations
            
        Returns:
            SizingResult with calculated size and metadata
        """
        self.stats['total_sizing_decisions'] += 1
        
        # Get base size using configured method
        base_result = self._calculate_base_size(
            decision, portfolio_value, current_positions, market_data
        )
        
        # Apply risk limits
        risk_adjusted_result = self._apply_risk_limits(
            base_result, decision, portfolio_value, current_positions
        )
        
        # Apply confidence adjustment
        final_result = self._apply_confidence_adjustment(
            risk_adjusted_result, decision
        )
        
        # Update statistics
        self._update_stats(final_result)
        
        return final_result
    
    def _calculate_base_size(
        self,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position],
        market_data: Optional[Dict[str, Any]]
    ) -> SizingResult:
        """Calculate base position size using configured method."""
        
        if self.config.method == SizingMethod.FIXED_FRACTION:
            return self._fixed_fraction_sizing(decision, portfolio_value)
        
        elif self.config.method == SizingMethod.FIXED_PERCENTAGE:
            return self._fixed_percentage_sizing(decision, portfolio_value)
        
        elif self.config.method == SizingMethod.KELLY:
            return self._kelly_sizing(decision, portfolio_value, current_positions)
        
        elif self.config.method == SizingMethod.VOLATILITY:
            return self._volatility_sizing(decision, portfolio_value, market_data)
        
        elif self.config.method == SizingMethod.RISK_PARITY:
            return self._risk_parity_sizing(decision, portfolio_value, current_positions)
        
        elif self.config.method == SizingMethod.ADAPTIVE:
            return self._adaptive_sizing(decision, portfolio_value, current_positions)
        
        else:
            # Fallback to fixed fraction
            return self._fixed_fraction_sizing(decision, portfolio_value)
    
    def _fixed_fraction_sizing(
        self,
        decision: Decision,
        portfolio_value: float
    ) -> SizingResult:
        """Fixed fraction position sizing."""
        size_fraction = self.config.base_fraction
        
        # Adjust for existing position
        current_position = self._get_current_position_size(decision.symbol, decision.action)
        if current_position != 0:
            # Reduce size if we're adding to existing position
            size_fraction *= 0.5
        
        size_value = portfolio_value * size_fraction
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            # Estimate size using market data (would need current price)
            size_units = size_value / 50000  # Placeholder for BTC/USDT
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=size_fraction,
            risk_amount=risk_amount,
            method_used=SizingMethod.FIXED_FRACTION.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'current_position': current_position
            }
        )
    
    def _kelly_sizing(
        self,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position]
    ) -> SizingResult:
        """Kelly criterion position sizing."""
        
        # Get historical performance for this symbol
        symbol_stats = self._get_symbol_performance(decision.symbol)
        
        if not symbol_stats or symbol_stats['win_rate'] < self.config.win_rate_threshold:
            # Fallback to fixed fraction if insufficient data
            logger.warning(f"Insufficient data for Kelly sizing on {decision.symbol}, using fixed fraction")
            return self._fixed_fraction_sizing(decision, portfolio_value)
        
        win_rate = symbol_stats['win_rate']
        avg_win = symbol_stats['avg_win']
        avg_loss = symbol_stats['avg_loss']
        
        # Calculate Kelly fraction
        if avg_loss == 0:
            kelly_fraction = self.config.max_fraction
        else:
            win_loss_ratio = avg_win / abs(avg_loss)
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply conservative Kelly fraction
            kelly_fraction *= self.config.kelly_fraction
        
        # Clamp to limits
        kelly_fraction = max(self.config.min_fraction, 
                           min(kelly_fraction, self.config.max_fraction))
        
        size_value = portfolio_value * kelly_fraction
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            size_units = size_value / 50000  # Placeholder
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=kelly_fraction,
            risk_amount=risk_amount,
            method_used=SizingMethod.KELLY.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'win_loss_ratio': win_loss_ratio if avg_loss != 0 else None
            }
        )
    
    def _fixed_percentage_sizing(
        self,
        decision: Decision,
        portfolio_value: float
    ) -> SizingResult:
        """
        Fixed percentage position sizing (1-2%).
        
        Args:
            decision: Trading decision
            portfolio_value: Current portfolio value
            
        Returns:
            SizingResult with calculated size
        """
        # Get fixed percentage from config (default 1.5%)
        fixed_percentage = self.config.fixed_percentage_default
        
        # Adjust for existing position
        current_position = self._get_current_position_size(decision.symbol, decision.action)
        if current_position != 0:
            # Reduce size if we're adding to existing position
            fixed_percentage *= 0.5
        
        # Clamp to configured limits
        fixed_percentage = max(self.config.fixed_percentage_min,
                              min(fixed_percentage, self.config.fixed_percentage_max))
        
        size_value = portfolio_value * fixed_percentage
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            # Estimate size using market data
            size_units = size_value / 50000  # Placeholder for BTC/USDT
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=fixed_percentage,
            risk_amount=risk_amount,
            method_used=SizingMethod.FIXED_PERCENTAGE.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'fixed_percentage': fixed_percentage,
                'portfolio_value': portfolio_value,
                'percentage_range': f"{self.config.fixed_percentage_min:.1%}-{self.config.fixed_percentage_max:.1%}"
            }
        )
    
    def _volatility_sizing(
        self,
        decision: Decision,
        portfolio_value: float,
        market_data: Optional[Dict[str, Any]]
    ) -> SizingResult:
        """Volatility-based position sizing."""
        
        if not market_data:
            logger.warning("No market data for volatility sizing, using fixed fraction")
            return self._fixed_fraction_sizing(decision, portfolio_value)
        
        # Get volatility from market data
        volatility = market_data.get('volatility', 0.02)  # Default 2%
        
        if self.config.volatility_scaling:
            # Target constant volatility
            size_fraction = self.config.volatility_target / volatility
            size_fraction *= self.config.base_fraction
        else:
            size_fraction = self.config.base_fraction
        
        # Clamp to limits
        size_fraction = max(self.config.min_fraction,
                           min(size_fraction, self.config.max_fraction))
        
        size_value = portfolio_value * size_fraction
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            size_units = size_value / market_data.get('price', 50000)
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=size_fraction,
            risk_amount=risk_amount,
            method_used=SizingMethod.VOLATILITY.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'volatility': volatility,
                'volatility_adjusted': self.config.volatility_scaling
            }
        )
    
    def _risk_parity_sizing(
        self,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position]
    ) -> SizingResult:
        """Risk parity position sizing."""
        
        # Calculate current portfolio risk
        current_risk = self._calculate_portfolio_risk(current_positions)
        
        # Target equal risk contribution
        target_risk_per_position = self.config.max_portfolio_risk / 10  # Assume 10 positions max
        
        if current_risk >= self.config.max_portfolio_risk:
            # Portfolio at max risk, minimum size
            size_fraction = self.config.min_fraction
        else:
            # Size based on remaining risk budget
            remaining_risk = self.config.max_portfolio_risk - current_risk
            size_fraction = min(remaining_risk, self.config.base_fraction)
        
        size_fraction = max(self.config.min_fraction,
                           min(size_fraction, self.config.max_fraction))
        
        size_value = portfolio_value * size_fraction
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            size_units = size_value / 50000  # Placeholder
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=size_fraction,
            risk_amount=risk_amount,
            method_used=SizingMethod.RISK_PARITY.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'current_portfolio_risk': current_risk,
                'remaining_risk': remaining_risk if current_risk < self.config.max_portfolio_risk else 0
            }
        )
    
    def _adaptive_sizing(
        self,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position]
    ) -> SizingResult:
        """Adaptive position sizing based on recent performance."""
        
        # Get recent performance
        recent_stats = self._get_recent_performance(decision.symbol, self.config.adaptive_window)
        
        if not recent_stats:
            return self._fixed_fraction_sizing(decision, portfolio_value)
        
        # Adjust base fraction based on recent performance
        performance_multiplier = 1.0
        
        if recent_stats['win_rate'] > 0.6:
            performance_multiplier = 1.2  # Increase size
        elif recent_stats['win_rate'] < 0.4:
            performance_multiplier = 0.8  # Decrease size
        
        if recent_stats['recent_drawdown'] > 0.05:  # 5% drawdown
            performance_multiplier *= 0.5  # Halve size
        
        adjusted_fraction = self.config.base_fraction * performance_multiplier
        
        # Clamp to limits
        adjusted_fraction = max(self.config.min_fraction,
                              min(adjusted_fraction, self.config.max_fraction))
        
        size_value = portfolio_value * adjusted_fraction
        
        # Convert to units
        if decision.price:
            size_units = size_value / decision.price
        else:
            size_units = size_value / 50000  # Placeholder
        
        risk_amount = size_value * self.config.max_risk_per_trade
        
        return SizingResult(
            size=size_units,
            size_fraction=adjusted_fraction,
            risk_amount=risk_amount,
            method_used=SizingMethod.ADAPTIVE.value,
            confidence_adjusted=False,
            metadata={
                'size_value': size_value,
                'performance_multiplier': performance_multiplier,
                'recent_win_rate': recent_stats['win_rate'],
                'recent_drawdown': recent_stats['recent_drawdown']
            }
        )
    
    def _apply_risk_limits(
        self,
        base_result: SizingResult,
        decision: Decision,
        portfolio_value: float,
        current_positions: Dict[str, Position]
    ) -> SizingResult:
        """Apply risk limits to position size."""
        
        # Check individual trade risk limit
        max_risk_amount = portfolio_value * self.config.max_risk_per_trade
        
        if base_result.risk_amount > max_risk_amount:
            # Reduce size to meet risk limit
            size_multiplier = max_risk_amount / base_result.risk_amount
            new_size = base_result.size * size_multiplier
            new_fraction = base_result.size_fraction * size_multiplier
            
            self.stats['risk_limited_trades'] += 1
            
            return SizingResult(
                size=new_size,
                size_fraction=new_fraction,
                risk_amount=max_risk_amount,
                method_used=base_result.method_used,
                confidence_adjusted=base_result.confidence_adjusted,
                metadata={
                    **base_result.metadata,
                    'risk_limited': True,
                    'original_size': base_result.size,
                    'risk_limit_reason': 'individual_trade_risk'
                }
            )
        
        # Check portfolio-level risk
        portfolio_risk = self._calculate_portfolio_risk_with_new_position(
            current_positions, decision, base_result.size
        )
        
        if portfolio_risk > self.config.max_portfolio_risk * portfolio_value:
            # Reduce size to meet portfolio risk limit
            risk_multiplier = (self.config.max_portfolio_risk * portfolio_value) / portfolio_risk
            new_size = base_result.size * risk_multiplier
            new_fraction = base_result.size_fraction * risk_multiplier
            new_risk = base_result.risk_amount * risk_multiplier
            
            self.stats['risk_limited_trades'] += 1
            
            return SizingResult(
                size=new_size,
                size_fraction=new_fraction,
                risk_amount=new_risk,
                method_used=base_result.method_used,
                confidence_adjusted=base_result.confidence_adjusted,
                metadata={
                    **base_result.metadata,
                    'risk_limited': True,
                    'original_size': base_result.size,
                    'risk_limit_reason': 'portfolio_risk'
                }
            )
        
        return base_result
    
    def _apply_confidence_adjustment(
        self,
        result: SizingResult,
        decision: Decision
    ) -> SizingResult:
        """Adjust position size based on decision confidence."""
        
        # This would need confidence from the decision/signal
        # For now, we'll assume a confidence field in decision metadata
        confidence = decision.metadata.get('confidence', 1.0)
        
        if confidence < 0.5:
            # Low confidence, reduce size
            confidence_multiplier = 0.5
        elif confidence < 0.7:
            # Medium confidence, slight reduction
            confidence_multiplier = 0.8
        else:
            # High confidence, no adjustment
            confidence_multiplier = 1.0
        
        if confidence_multiplier < 1.0:
            new_size = result.size * confidence_multiplier
            new_fraction = result.size_fraction * confidence_multiplier
            new_risk = result.risk_amount * confidence_multiplier
            
            self.stats['confidence_adjustments'] += 1
            
            return SizingResult(
                size=new_size,
                size_fraction=new_fraction,
                risk_amount=new_risk,
                method_used=result.method_used,
                confidence_adjusted=True,
                metadata={
                    **result.metadata,
                    'confidence_adjusted': True,
                    'confidence': confidence,
                    'confidence_multiplier': confidence_multiplier,
                    'original_size': result.size
                }
            )
        
        return result
    
    def _get_current_position_size(self, symbol: str, action: Action) -> float:
        """Get current position size for symbol."""
        # This would integrate with portfolio manager
        # For now, return 0 (no existing position)
        return 0.0
    
    def _get_symbol_performance(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get historical performance statistics for symbol."""
        if symbol not in self.trade_history:
            return None
        
        trades = self.trade_history[symbol]
        if len(trades) < 10:  # Need minimum trades
            return None
        
        returns = [trade['return'] for trade in trades]
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        return {
            'win_rate': len(wins) / len(returns),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_trades': len(returns)
        }
    
    def _get_recent_performance(self, symbol: str, window: int) -> Dict[str, float]:
        """Get recent performance statistics."""
        if symbol not in self.trade_history:
            return {'win_rate': 0.5, 'recent_drawdown': 0.0}
        
        trades = self.trade_history[symbol][-window:]
        if len(trades) < 5:
            return {'win_rate': 0.5, 'recent_drawdown': 0.0}
        
        returns = [trade['return'] for trade in trades]
        wins = [r for r in returns if r > 0]
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            'win_rate': len(wins) / len(returns),
            'recent_drawdown': max_drawdown
        }
    
    def _calculate_portfolio_risk(self, positions: Dict[str, Position]) -> float:
        """Calculate current portfolio risk."""
        total_risk = 0.0
        for position in positions.values():
            # Simplified risk calculation (would use correlation matrix in production)
            position_risk = abs(position.size * position.entry_price) * 0.02  # 2% risk per position
            total_risk += position_risk
        
        return total_risk
    
    def _calculate_portfolio_risk_with_new_position(
        self,
        current_positions: Dict[str, Position],
        decision: Decision,
        new_size: float
    ) -> float:
        """Calculate portfolio risk including new position."""
        
        # Start with current risk
        current_risk = self._calculate_portfolio_risk(current_positions)
        
        # Add risk for new position
        if decision.price:
            new_position_risk = abs(new_size * decision.price) * 0.02  # 2% risk assumption
        else:
            new_position_risk = abs(new_size * 50000) * 0.02  # Placeholder
        
        return current_risk + new_position_risk
    
    def _update_stats(self, result: SizingResult) -> None:
        """Update sizing statistics."""
        self.stats['method_usage'][result.method_used] += 1
    
    def update_trade_result(self, symbol: str, return_pct: float) -> None:
        """Update trade history with new result."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        self.trade_history[symbol].append({
            'return': return_pct,
            'timestamp': None  # Would add actual timestamp
        })
        
        # Keep only recent trades
        if len(self.trade_history[symbol]) > 1000:
            self.trade_history[symbol] = self.trade_history[symbol][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sizing statistics."""
        return self.stats.copy()


# Convenience functions
def create_fixed_fraction_sizer(fraction: float = 0.02) -> PositionSizer:
    """Create fixed fraction position sizer."""
    config = SizingConfig(
        method=SizingMethod.FIXED_FRACTION,
        base_fraction=fraction
    )
    return PositionSizer(config)


def create_kelly_sizer(
    kelly_fraction: float = 0.25,
    win_rate_threshold: float = 0.55
) -> PositionSizer:
    """Create Kelly criterion position sizer."""
    config = SizingConfig(
        method=SizingMethod.KELLY,
        kelly_fraction=kelly_fraction,
        win_rate_threshold=win_rate_threshold
    )
    return PositionSizer(config)


def create_volatility_sizer(
    base_fraction: float = 0.02,
    volatility_target: float = 0.02
) -> PositionSizer:
    """Create volatility-based position sizer."""
    config = SizingConfig(
        method=SizingMethod.VOLATILITY,
        base_fraction=base_fraction,
        volatility_target=volatility_target
    )
    return PositionSizer(config)


if __name__ == "__main__":
    # Test position sizer
    logging.basicConfig(level=logging.INFO)
    
    config = SizingConfig(
        method=SizingMethod.FIXED_FRACTION,
        base_fraction=0.02,
        max_fraction=0.05
    )
    
    sizer = PositionSizer(config)
    
    # Test decision
    decision = Decision(
        action=Action.BUY,
        symbol="BTC/USDT",
        size=0.0,  # Will be calculated
        price=42000.0,
        stop_loss=41000.0,
        take_profit=44000.0,
        timestamp=int(time.time() * 1000),
        reason="Test signal"
    )
    
    # Calculate position size
    result = sizer.calculate_position_size(
        decision=decision,
        portfolio_value=100000.0,
        current_positions={},
        market_data={'volatility': 0.03}
    )
    
    print(f"Position sizing result:")
    print(f"  Size: {result.size:.6f}")
    print(f"  Fraction: {result.size_fraction:.2%}")
    print(f"  Risk amount: ${result.risk_amount:.2f}")
    print(f"  Method: {result.method_used}")
    print(f"  Metadata: {result.metadata}")
    
    print(f"\nSizing stats: {sizer.get_stats()}")
