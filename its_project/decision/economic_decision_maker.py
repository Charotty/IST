from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from its_project.decision.decision import BaseDecisionMaker, Signal, Decision, Action
except ImportError:
    from .decision import BaseDecisionMaker, Signal, Decision, Action


@dataclass
class MarketState:
    """Enhanced market state with economic indicators."""
    price: float
    volatility: float
    trend: float
    volume_ratio: float
    momentum: float
    mean_reversion_signal: float
    liquidity_score: float
    market_efficiency: float
    timestamp: datetime
    capital: float
    in_position: bool
    position_side: Optional[str] = None


@dataclass
class RiskMetrics:
    """Risk metrics for decision making."""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_position_size: float
    stop_loss_distance: float
    take_profit_distance: float
    risk_reward_ratio: float


class EconomicDecisionMaker(BaseDecisionMaker):
    """
    Economic decision maker with sophisticated risk management.
    
    Features:
    - Dynamic position sizing based on volatility
    - Risk-adjusted decision thresholds
    - Market regime detection
    - Portfolio-level risk management
    - Economic utility optimization
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Decision parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.min_edge_threshold = config.get("min_edge_threshold", 0.02)  # 2% minimum expected edge
        self.max_position_size = config.get("max_position_size", 0.2)  # 20% max position
        
        # Risk management
        self.use_dynamic_sizing = config.get("use_dynamic_sizing", True)
        self.volatility_target = config.get("volatility_target", 0.15)  # 15% annual volatility
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        
        # Market regime detection
        self.use_regime_detection = config.get("use_regime_detection", True)
        self.regime_window = config.get("regime_window", 50)
        
        # Portfolio management
        self.max_correlation = config.get("max_correlation", 0.7)
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.15)
        
        # Economic optimization
        self.use_utility_optimization = config.get("use_utility_optimization", True)
        self.risk_aversion = config.get("risk_aversion", 2.0)
        
        # State tracking
        self.current_regime = "normal"
        self.portfolio_value = config.get("initial_capital", 10000.0)
        self.peak_portfolio_value = self.portfolio_value
        self.current_drawdown = 0.0
        
    def decide(self, signal: Signal, market_state: Dict[str, Any]) -> Optional[Decision]:
        """
        Make economic decision with risk management.
        
        Args:
            signal: Trading signal with prediction and confidence
            market_state: Current market state information
            
        Returns:
            Trading decision or None if no action
        """
        # Validate signal
        if not self.validate_signal(signal):
            return None
        
        # Create enhanced market state
        enhanced_state = self._create_enhanced_market_state(market_state, signal)
        
        # Check for risk limits
        if self._check_risk_limits(enhanced_state):
            return self._create_risk_reduction_decision(enhanced_state, signal)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(enhanced_state)
        
        # Apply regime-based adjustments
        adjusted_signal = self._adjust_for_regime(signal, enhanced_state)
        
        # Calculate expected returns and utility
        expected_returns = self._calculate_expected_returns(adjusted_signal, enhanced_state)
        utility = self._calculate_utility(expected_returns, risk_metrics)
        
        # Make decision based on economic criteria
        if utility > 0 and adjusted_signal.confidence >= self.confidence_threshold:
            return self._create_economic_decision(adjusted_signal, enhanced_state, risk_metrics)
        
        return None
    
    def _create_enhanced_market_state(self, market_state: Dict[str, Any], 
                                     signal: Signal) -> MarketState:
        """Create enhanced market state with additional indicators."""
        return MarketState(
            price=market_state.get("price", 0),
            volatility=market_state.get("volatility", 0.02),
            trend=market_state.get("trend", 0),
            volume_ratio=market_state.get("volume_ratio", 1.0),
            momentum=market_state.get("momentum", 0),
            mean_reversion_signal=market_state.get("mean_reversion_signal", 0),
            liquidity_score=market_state.get("liquidity_score", 1.0),
            market_efficiency=market_state.get("market_efficiency", 0.5),
            timestamp=datetime.fromtimestamp(signal.timestamp / 1000),
            capital=market_state.get("capital", self.portfolio_value),
            in_position=market_state.get("in_position", False),
            position_side=market_state.get("position_side")
        )
    
    def _check_risk_limits(self, state: MarketState) -> bool:
        """Check if risk limits are breached."""
        # Check maximum drawdown
        if self.current_drawdown > self.max_drawdown_limit:
            return True
        
        # Check volatility limits
        if state.volatility > self.volatility_target * 2:
            return True
        
        # Check liquidity
        if state.liquidity_score < 0.3:
            return True
        
        return False
    
    def _calculate_risk_metrics(self, state: MarketState) -> RiskMetrics:
        """Calculate risk metrics for position sizing."""
        # Value at Risk calculations
        var_95 = state.volatility * 1.645  # 95% VaR
        var_99 = state.volatility * 2.326  # 99% VaR
        
        # Expected Shortfall (average loss beyond VaR)
        expected_shortfall = var_99 * 1.1
        
        # Dynamic position sizing based on volatility
        if self.use_dynamic_sizing:
            vol_adjusted_size = self.volatility_target / (state.volatility + 1e-10)
            max_position_size = min(self.max_position_size, vol_adjusted_size)
        else:
            max_position_size = self.max_position_size
        
        # Stop loss and take profit based on volatility
        stop_loss_distance = state.volatility * 2  # 2x volatility stop loss
        take_profit_distance = state.volatility * 3  # 3x volatility take profit
        
        # Risk-reward ratio
        risk_reward_ratio = take_profit_distance / stop_loss_distance
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_position_size=max_position_size,
            stop_loss_distance=stop_loss_distance,
            take_profit_distance=take_profit_distance,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _adjust_for_regime(self, signal: Signal, state: MarketState) -> Signal:
        """Adjust signal based on market regime."""
        if not self.use_regime_detection:
            return signal
        
        # Detect market regime
        regime = self._detect_market_regime(state)
        self.current_regime = regime
        
        # Adjust confidence based on regime
        adjusted_confidence = signal.confidence
        
        if regime == "high_volatility":
            adjusted_confidence *= 0.7  # Reduce confidence in high volatility
        elif regime == "trending":
            adjusted_confidence *= 1.2  # Increase confidence in trending markets
        elif regime == "mean_reverting":
            adjusted_confidence *= 1.1  # Slightly increase in mean reverting
        
        # Clamp confidence
        adjusted_confidence = np.clip(adjusted_confidence, 0, 1)
        
        # Create adjusted signal
        return Signal(
            action=signal.action,
            confidence=adjusted_confidence,
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            metadata={**signal.metadata, "regime": regime}
        )
    
    def _detect_market_regime(self, state: MarketState) -> str:
        """Detect current market regime."""
        # Simplified regime detection
        if state.volatility > 0.05:  # High volatility threshold
            return "high_volatility"
        elif abs(state.trend) > 0.02:  # Strong trend threshold
            return "trending"
        elif abs(state.mean_reversion_signal) > 0.5:
            return "mean_reverting"
        else:
            return "normal"
    
    def _calculate_expected_returns(self, signal: Signal, state: MarketState) -> Dict[str, float]:
        """Calculate expected returns for each action."""
        # Base expected returns (simplified)
        base_returns = {
            "buy": 0.001,  # 0.1% expected return for buy
            "sell": 0.001,  # 0.1% expected return for sell
            "hold": 0.0     # 0% for hold
        }
        
        # Adjust based on market conditions
        if state.trend > 0.01:  # Uptrend
            base_returns["buy"] *= 1.5
            base_returns["sell"] *= 0.5
        elif state.trend < -0.01:  # Downtrend
            base_returns["buy"] *= 0.5
            base_returns["sell"] *= 1.5
        
        # Adjust based on momentum
        if state.momentum > 0.01:  # Positive momentum
            base_returns["buy"] *= 1.3
        elif state.momentum < -0.01:  # Negative momentum
            base_returns["sell"] *= 1.3
        
        # Adjust based on mean reversion
        if state.mean_reversion_signal > 0.5:  # Overbought
            base_returns["sell"] *= 1.2
            base_returns["buy"] *= 0.8
        elif state.mean_reversion_signal < -0.5:  # Oversold
            base_returns["buy"] *= 1.2
            base_returns["sell"] *= 0.8
        
        # Apply confidence weighting
        action_map = {0: "sell", 1: "hold", 2: "buy"}
        predicted_action = action_map.get(signal.action, "hold")
        
        expected_returns = {}
        for action, base_ret in base_returns.items():
            if action == predicted_action:
                expected_returns[action] = base_ret * signal.confidence
            else:
                expected_returns[action] = base_ret * (1 - signal.confidence)
        
        return expected_returns
    
    def _calculate_utility(self, expected_returns: Dict[str, float], 
                         risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Calculate economic utility for each action."""
        utility = {}
        
        for action, expected_return in expected_returns.items():
            # Mean-variance utility: U = E[R] - 0.5 * λ * Var[R]
            variance = risk_metrics.var_95 ** 2  # Simplified variance estimate
            action_utility = expected_return - 0.5 * self.risk_aversion * variance
            
            utility[action] = action_utility
        
        return utility
    
    def _create_economic_decision(self, signal: Signal, state: MarketState,
                               risk_metrics: RiskMetrics) -> Decision:
        """Create economic decision with proper risk management."""
        # Calculate position size
        position_size = self._calculate_position_size(signal, state, risk_metrics)
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(signal, state, risk_metrics)
        
        # Determine action
        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action = Action(action_map.get(signal.action, "hold"))
        
        # Create decision
        return Decision(
            action=action,
            symbol=signal.symbol,
            size=position_size,
            price=None,  # Market order
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.timestamp,
            reason=f"Economic decision: confidence={signal.confidence:.2f}, regime={self.current_regime}, utility={self._get_best_utility(signal, state, risk_metrics):.4f}"
        )
    
    def _calculate_position_size(self, signal: Signal, state: MarketState,
                                risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size."""
        # Base position size
        base_size = self.config.get("position_size", 0.01)
        
        # Adjust for confidence
        confidence_adjusted_size = base_size * signal.confidence
        
        # Adjust for volatility
        if self.use_dynamic_sizing:
            vol_adjusted_size = confidence_adjusted_size * risk_metrics.max_position_size
        else:
            vol_adjusted_size = confidence_adjusted_size
        
        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier()
        final_size = vol_adjusted_size * regime_multiplier
        
        # Ensure within limits
        final_size = min(final_size, risk_metrics.max_position_size)
        final_size = max(final_size, 0.001)  # Minimum position size
        
        return final_size
    
    def _calculate_risk_levels(self, signal: Signal, state: MarketState,
                             risk_metrics: RiskMetrics) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        current_price = state.price
        
        if signal.action == 2:  # BUY
            stop_loss = current_price * (1 - risk_metrics.stop_loss_distance)
            take_profit = current_price * (1 + risk_metrics.take_profit_distance)
        elif signal.action == 0:  # SELL
            stop_loss = current_price * (1 + risk_metrics.stop_loss_distance)
            take_profit = current_price * (1 - risk_metrics.take_profit_distance)
        else:  # HOLD
            return None, None
        
        return stop_loss, take_profit
    
    def _get_regime_multiplier(self) -> float:
        """Get position size multiplier based on regime."""
        multipliers = {
            "normal": 1.0,
            "high_volatility": 0.5,
            "trending": 1.2,
            "mean_reverting": 1.1
        }
        return multipliers.get(self.current_regime, 1.0)
    
    def _get_best_utility(self, signal: Signal, state: MarketState,
                         risk_metrics: RiskMetrics) -> float:
        """Get utility for the predicted action."""
        expected_returns = self._calculate_expected_returns(signal, state)
        utility = self._calculate_utility(expected_returns, risk_metrics)
        
        action_map = {0: "sell", 1: "hold", 2: "buy"}
        predicted_action = action_map.get(signal.action, "hold")
        
        return utility.get(predicted_action, 0.0)
    
    def _create_risk_reduction_decision(self, state: MarketState, 
                                     signal: Signal) -> Decision:
        """Create decision to reduce risk when limits are breached."""
        return Decision(
            action=Action.HOLD,
            symbol=signal.symbol,
            size=0.0,
            price=None,
            stop_loss=None,
            take_profit=None,
            timestamp=signal.timestamp,
            reason=f"Risk reduction: drawdown={self.current_drawdown:.2%}, volatility={state.volatility:.2%}"
        )
    
    def update_portfolio_state(self, portfolio_value: float, peak_value: float) -> None:
        """Update portfolio state for risk management."""
        self.portfolio_value = portfolio_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, peak_value)
        self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision maker statistics."""
        return {
            "current_regime": self.current_regime,
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "current_drawdown": self.current_drawdown,
            "confidence_threshold": self.confidence_threshold,
            "max_position_size": self.max_position_size,
            "risk_aversion": self.risk_aversion
        }
