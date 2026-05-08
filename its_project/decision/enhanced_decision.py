"""
Enhanced Decision Maker with ΔP_hat sign logic and confidence filtering.

Implements:
- action = sign(ΔP_hat)
- Confidence thresholding: p_hat < threshold → HOLD
- Minimum movement filtering: |ΔP_hat| < min_move → HOLD
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

from its_project.decision.decision import BaseDecisionMaker, Signal, Decision, Action

logger = logging.getLogger(__name__)


class EnhancedDecisionMaker(BaseDecisionMaker):
    """
    Enhanced decision maker with ΔP_hat sign logic and confidence filtering.
    
    Implements the following logic:
    1. action = sign(ΔP_hat) - sign of predicted price change
    2. Confidence filtering: if p_hat < threshold → HOLD
    3. Minimum movement filtering: if |ΔP_hat| < min_move → HOLD
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Additional configuration parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.min_price_move = config.get("min_price_move", 0.0005)  # 0.05% minimum move
        self.use_price_sign_logic = config.get("use_price_sign_logic", True)
        self.fallback_to_hold = config.get("fallback_to_hold", True)
        
        # Cooldown parameters
        self.cooldown_enabled = config.get("cooldown_enabled", True)
        self.cooldown_period = config.get("cooldown_period", 300)  # 5 minutes in seconds
        self.last_signal_time = {}
        
        # Volatility filter parameters
        self.volatility_filter_enabled = config.get("volatility_filter_enabled", True)
        self.volatility_window = config.get("volatility_window", 20)
        self.max_volatility_threshold = config.get("max_volatility_threshold", 0.05)  # 5% max volatility
        self.volatility_history = []
        self.max_volatility_history = 100
        
        logger.info(f"EnhancedDecisionMaker initialized:")
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Minimum price move: {self.min_price_move}")
        logger.info(f"  Use price sign logic: {self.use_price_sign_logic}")
        logger.info(f"  Cooldown enabled: {self.cooldown_enabled}, period: {self.cooldown_period}s")
        logger.info(f"  Volatility filter enabled: {self.volatility_filter_enabled}")
        logger.info(f"  Max volatility threshold: {self.max_volatility_threshold}")

    def decide(
        self,
        signal: Signal,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Decision]:
        """
        Make enhanced decision based on ΔP_hat and confidence.
        
        Args:
            signal: Signal from model with ΔP_hat and p_hat
            market_state: Current market state
            
        Returns:
            Enhanced Decision or None
        """
        if not self.validate_signal(signal):
            logger.warning(f"Signal validation failed: confidence={signal.confidence}")
            return None
        
        # Extract ΔP_hat and p_hat from signal metadata
        delta_p_hat = signal.metadata.get("delta_p_hat", 0.0)
        p_hat = signal.confidence
        current_time = signal.timestamp
        
        # Step 1: Cooldown check
        if self.cooldown_enabled and self._is_in_cooldown(signal.symbol, current_time):
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Cooldown active for {signal.symbol}"
            )
        
        # Step 2: Volatility filter
        if self.volatility_filter_enabled and self._is_volatility_too_high(market_state):
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Volatility too high: {self._get_current_volatility(market_state):.4f} > {self.max_volatility_threshold}"
            )
        
        # Step 3: Confidence filtering
        if p_hat < self.confidence_threshold:
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Low confidence: {p_hat:.3f} < {self.confidence_threshold}"
            )
        
        # Step 4: Minimum movement filtering
        if abs(delta_p_hat) < self.min_price_move:
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Insufficient price move: |ΔP_hat|={abs(delta_p_hat):.6f} < {self.min_price_move}"
            )
        
        # Step 5: Action determination using sign(ΔP_hat)
        if self.use_price_sign_logic:
            action = self._delta_p_to_action(delta_p_hat)
            reason = f"ΔP_hat sign logic: ΔP_hat={delta_p_hat:.6f} → {action.value}"
        else:
            # Fallback to original signal Action
            action = signal.action
            reason = f"Original signal action: {action.value} (confidence: {p_hat:.3f})"
        
        # Step 6: Update cooldown
        if self.cooldown_enabled:
            self._update_cooldown(signal.symbol, current_time, action)
        
        # Calculate position size and risk levels
        position_size = self._calculate_position_size(signal, market_state, delta_p_hat, p_hat)
        stop_loss, take_profit = self._calculate_risk_levels(
            signal, market_state, delta_p_hat, action
        )
        
        return Decision(
            action=action,
            symbol=signal.symbol,
            size=position_size,
            price=None,  # Market order
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.timestamp,
            reason=reason
        )

    def _delta_p_to_action(self, delta_p_hat: float) -> Action:
        """
        Convert ΔP_hat to action using sign logic.
        
        Args:
            delta_p_hat: Predicted price change
            
        Returns:
            Action based on sign of ΔP_hat
        """
        if delta_p_hat > 0:
            return Action.BUY
        elif delta_p_hat < 0:
            return Action.SELL
        else:
            return Action.HOLD

    def _calculate_position_size(
        self,
        signal: Signal,
        market_state: Optional[Dict[str, Any]],
        delta_p_hat: float,
        p_hat: float
    ) -> float:
        """
        Calculate position size based on confidence and price move magnitude.
        
        Args:
            signal: Original signal
            market_state: Market state
            delta_p_hat: Predicted price change
            p_hat: Confidence
            
        Returns:
            Position size
        """
        base_size = self.config.get("position_size", 0.01)
        
        # Size scaling factors
        confidence_factor = p_hat  # Higher confidence = larger position
        magnitude_factor = min(abs(delta_p_hat) / 0.001, 2.0)  # Scale by move magnitude
        
        # Combined size calculation
        position_size = base_size * confidence_factor * magnitude_factor
        
        # Apply maximum size limit
        max_size = self.config.get("max_position_size", 0.1)
        position_size = min(position_size, max_size)
        
        logger.debug(f"Position size calculation: base={base_size}, conf={p_hat:.3f}, "
                    f"mag={abs(delta_p_hat):.6f}, final={position_size:.6f}")
        
        return position_size

    def _calculate_risk_levels(
        self,
        signal: Signal,
        market_state: Optional[Dict[str, Any]],
        delta_p_hat: float,
        action: Action
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit based on ΔP_hat.
        
        Args:
            signal: Original signal
            market_state: Market state
            delta_p_hat: Predicted price change
            action: Determined action
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        current_price = market_state.get("price", 0) if market_state else 0
        if current_price == 0:
            return None, None
        
        # Risk percentages
        stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
        take_profit_pct = self.config.get("take_profit_pct", 0.04)
        
        # Dynamic risk adjustment based on ΔP_hat magnitude
        risk_adjustment = min(abs(delta_p_hat) / 0.002, 1.5)  # Adjust for move size
        
        adjusted_stop_pct = stop_loss_pct * risk_adjustment
        adjusted_take_pct = take_profit_pct * risk_adjustment
        
        if action == Action.BUY:
            stop_loss = current_price * (1 - adjusted_stop_pct)
            take_profit = current_price * (1 + adjusted_take_pct)
        elif action == Action.SELL:
            stop_loss = current_price * (1 + adjusted_stop_pct)
            take_profit = current_price * (1 - adjusted_take_pct)
        else:
            return None, None
        
        logger.debug(f"Risk levels: price={current_price}, action={action.value}, "
                    f"ΔP_hat={delta_p_hat:.6f}, SL={adjusted_stop_pct:.3f}, TP={adjusted_take_pct:.3f}")
        
        return stop_loss, take_profit

    def get_decision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about decision making process.
        
        Returns:
            Dictionary with decision statistics
        """
        return {
            "confidence_threshold": self.confidence_threshold,
            "min_price_move": self.min_price_move,
            "use_price_sign_logic": self.use_price_sign_logic,
            "fallback_to_hold": self.fallback_to_hold,
            "config": self.config
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update decision maker configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update derived parameters
        if "confidence_threshold" in new_config:
            self.confidence_threshold = new_config["confidence_threshold"]
        if "min_price_move" in new_config:
            self.min_price_move = new_config["min_price_move"]
        if "use_price_sign_logic" in new_config:
            self.use_price_sign_logic = new_config["use_price_sign_logic"]
        if "fallback_to_hold" in new_config:
            self.fallback_to_hold = new_config["fallback_to_hold"]
        
        # Update cooldown parameters
        if "cooldown_enabled" in new_config:
            self.cooldown_enabled = new_config["cooldown_enabled"]
        if "cooldown_period" in new_config:
            self.cooldown_period = new_config["cooldown_period"]
        
        # Update volatility filter parameters
        if "volatility_filter_enabled" in new_config:
            self.volatility_filter_enabled = new_config["volatility_filter_enabled"]
        if "volatility_window" in new_config:
            self.volatility_window = new_config["volatility_window"]
        if "max_volatility_threshold" in new_config:
            self.max_volatility_threshold = new_config["max_volatility_threshold"]
        
        logger.info(f"EnhancedDecisionMaker config updated: {new_config}")
    
    def _is_in_cooldown(self, symbol: str, current_time: int) -> bool:
        """
        Check if symbol is in cooldown period.
        
        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            
        Returns:
            True if in cooldown period
        """
        if not self.cooldown_enabled:
            return False
        
        if symbol not in self.last_signal_time:
            return False
        
        time_since_last = current_time - self.last_signal_time[symbol]
        return time_since_last < self.cooldown_period
    
    def _update_cooldown(self, symbol: str, current_time: int, action: Action) -> None:
        """
        Update cooldown tracking for symbol.
        
        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            action: Action taken
        """
        if not self.cooldown_enabled:
            return
        
        # Only update cooldown for non-HOLD actions
        if action != Action.HOLD:
            self.last_signal_time[symbol] = current_time
            logger.debug(f"Updated cooldown for {symbol}: {current_time}")
    
    def _is_volatility_too_high(self, market_state: Optional[Dict[str, Any]]) -> bool:
        """
        Check if market volatility is too high for trading.
        
        Args:
            market_state: Market state with price data
            
        Returns:
            True if volatility is too high
        """
        if not self.volatility_filter_enabled:
            return False
        
        if not market_state or "price" not in market_state:
            return False
        
        # Calculate recent volatility from price history
        price_history = market_state.get("price_history", [])
        if len(price_history) < self.volatility_window:
            return False
        
        recent_prices = price_history[-self.volatility_window:]
        
        # Calculate returns and volatility
        returns = []
        for i in range(1, len(recent_prices)):
            returns.append((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1])
        
        if len(returns) < 2:
            return False
        
        volatility = np.std(returns)
        
        # Update volatility history
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.max_volatility_history:
            self.volatility_history = self.volatility_history[-self.max_volatility_history:]
        
        current_volatility = volatility
        logger.debug(f"Current volatility: {current_volatility:.6f}, threshold: {self.max_volatility_threshold}")
        
        return current_volatility > self.max_volatility_threshold
    
    def _get_current_volatility(self, market_state: Optional[Dict[str, Any]]) -> float:
        """
        Get current market volatility.
        
        Args:
            market_state: Market state
            
        Returns:
            Current volatility
        """
        if not self.volatility_history:
            return 0.0
        
        return self.volatility_history[-1] if self.volatility_history else 0.0


class ConfidenceBasedDecisionMaker(BaseDecisionMaker):
    """
    Alternative decision maker focused purely on confidence filtering.
    
    Uses traditional action from signal but applies enhanced confidence filtering.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Enhanced confidence parameters
        self.strict_confidence = config.get("strict_confidence", False)
        self.confidence_decay = config.get("confidence_decay", 0.1)
        self.min_confidence_for_trade = config.get("min_confidence_for_trade", 0.8)
        
        # Historical confidence tracking
        self.confidence_history = []
        self.max_history_length = config.get("confidence_history_length", 100)
        
        logger.info(f"ConfidenceBasedDecisionMaker initialized:")
        logger.info(f"  Strict confidence: {self.strict_confidence}")
        logger.info(f"  Confidence decay: {self.confidence_decay}")
        logger.info(f"  Min confidence for trade: {self.min_confidence_for_trade}")

    def decide(
        self,
        signal: Signal,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Decision]:
        """
        Make decision with enhanced confidence filtering.
        """
        if not self.validate_signal(signal):
            return None
        
        p_hat = signal.confidence
        
        # Update confidence history
        self.confidence_history.append(p_hat)
        if len(self.confidence_history) > self.max_history_length:
            self.confidence_history = self.confidence_history[-self.max_history_length:]
        
        # Calculate adjusted confidence
        adjusted_confidence = self._calculate_adjusted_confidence(p_hat)
        
        # Enhanced confidence filtering
        if self.strict_confidence:
            # Strict mode: require higher confidence
            effective_threshold = max(
                self.confidence_threshold,
                self.min_confidence_for_trade
            )
        else:
            # Normal mode: use base threshold
            effective_threshold = self.confidence_threshold
        
        if adjusted_confidence < effective_threshold:
            return Decision(
                action=Action.HOLD,
                symbol=signal.symbol,
                size=0.0,
                price=None,
                stop_loss=None,
                take_profit=None,
                timestamp=signal.timestamp,
                reason=f"Low adjusted confidence: {adjusted_confidence:.3f} < {effective_threshold}"
            )
        
        # Use original signal action with confidence-based sizing
        position_size = self._calculate_confidence_based_size(adjusted_confidence, signal)
        stop_loss, take_profit = self._calculate_standard_risk_levels(signal, market_state)
        
        return Decision(
            action=signal.action,
            symbol=signal.symbol,
            size=position_size,
            price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.timestamp,
            reason=f"Confidence-based: {adjusted_confidence:.3f} → {signal.action.value}"
        )

    def _calculate_adjusted_confidence(self, current_confidence: float) -> float:
        """
        Calculate confidence adjusted by historical performance.
        """
        if len(self.confidence_history) < 10:
            return current_confidence
        
        # Calculate recent confidence trend
        recent_confidence = self.confidence_history[-10:]
        avg_recent = np.mean(recent_confidence)
        
        # Apply decay factor based on trend
        if current_confidence < avg_recent:
            # Confidence is declining, apply decay
            adjusted = current_confidence * (1 - self.confidence_decay)
        else:
            # Confidence is stable or improving, no decay
            adjusted = current_confidence
        
        return max(adjusted, 0.1)  # Minimum confidence floor

    def _calculate_confidence_based_size(
        self,
        adjusted_confidence: float,
        signal: Signal
    ) -> float:
        """
        Calculate position size based on adjusted confidence.
        """
        base_size = self.config.get("position_size", 0.01)
        
        # Non-linear sizing: confidence^1.5 for aggressive scaling
        confidence_power = 1.5
        size_multiplier = adjusted_confidence ** confidence_power
        
        position_size = base_size * size_multiplier
        
        # Apply limits
        max_size = self.config.get("max_position_size", 0.1)
        min_size = self.config.get("min_position_size", 0.001)
        
        return np.clip(position_size, min_size, max_size)

    def _calculate_standard_risk_levels(
        self,
        signal: Signal,
        market_state: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate standard risk levels (stop loss, take profit).
        """
        current_price = market_state.get("price", 0) if market_state else 0
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

    def get_confidence_stats(self) -> Dict[str, Any]:
        """
        Get confidence-related statistics.
        """
        if not self.confidence_history:
            return {"message": "No confidence history available"}
        
        return {
            "current_confidence": self.confidence_history[-1] if self.confidence_history else 0,
            "avg_confidence": np.mean(self.confidence_history),
            "min_confidence": np.min(self.confidence_history),
            "max_confidence": np.max(self.confidence_history),
            "confidence_trend": "improving" if len(self.confidence_history) > 1 and 
                              self.confidence_history[-1] > self.confidence_history[-2] else "declining",
            "history_length": len(self.confidence_history),
            "strict_mode": self.strict_confidence,
            "decay_factor": self.confidence_decay
        }
