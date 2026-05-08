#!/usr/bin/env python3
"""
Realistic Trading Costs Module
===========================

Production-ready cost modeling for accurate backtesting:
- Exchange fees (maker/taker)
- Slippage modeling
- Latency impact
- Market impact
- Network fees
"""

from __future__ import annotations

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import math

from its_project.execution.base import Order, OrderType, OrderStatus
from its_project.decision.decision import Decision, Action

logger = logging.getLogger(__name__)


class CostModel(Enum):
    """Cost calculation models."""
    FIXED = "fixed"
    TIERED = "tiered"
    VOLUME_BASED = "volume_based"
    DYNAMIC = "dynamic"


@dataclass
class ExchangeFeeSchedule:
    """Exchange fee schedule."""
    maker_fee: float = 0.0002      # 0.02% maker fee
    taker_fee: float = 0.0004      # 0.04% taker fee
    volume_tiers: List[Tuple[float, float]] = field(default_factory=list)  # (volume, fee_rate)
    vip_discounts: Dict[str, float] = field(default_factory=dict)
    minimum_fee: float = 0.0
    fee_currency: str = "USDT"


@dataclass
class SlippageModel:
    """Slippage modeling parameters."""
    base_slippage: float = 0.0005    # 0.05% base slippage
    volatility_multiplier: float = 2.0
    volume_impact_factor: float = 0.0001
    order_size_impact: float = 0.00001
    spread_impact: bool = True
    time_decay_factor: float = 0.1


@dataclass
class LatencyModel:
    """Latency modeling parameters."""
    base_latency_ms: float = 50.0      # 50ms base latency
    exchange_latency_ms: float = 20.0    # 20ms exchange processing
    network_latency_ms: float = 30.0     # 30ms network latency
    queue_delay_factor: float = 0.1
    peak_hour_multiplier: float = 1.5
    weekend_multiplier: float = 2.0


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a trade."""
    total_cost: float
    exchange_fee: float
    slippage_cost: float
    latency_cost: float
    network_fee: float
    market_impact: float
    financing_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealisticCostCalculator:
    """
    Advanced cost calculator for realistic backtesting.
    
    Features:
    - Multi-tier exchange fees
    - Dynamic slippage modeling
    - Latency impact simulation
    - Market impact calculation
    - Time-based cost variations
    """
    
    def __init__(
        self,
        exchange_schedule: Optional[ExchangeFeeSchedule] = None,
        slippage_model: Optional[SlippageModel] = None,
        latency_model: Optional[LatencyModel] = None,
        cost_model: CostModel = CostModel.DYNAMIC
    ) -> None:
        self.exchange_schedule = exchange_schedule or ExchangeFeeSchedule()
        self.slippage_model = slippage_model or SlippageModel()
        self.latency_model = latency_model or LatencyModel()
        self.cost_model = cost_model
        
        # Historical data for dynamic modeling
        self.volume_history: Dict[str, List[float]] = {}
        self.volatility_history: Dict[str, List[float]] = {}
        self.spread_history: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'total_costs': 0.0,
            'avg_cost_per_trade': 0.0,
            'max_slippage': 0.0,
            'avg_slippage': 0.0,
            'latency_rejections': 0
        }
    
    def calculate_trade_costs(
        self,
        order: Order,
        market_data: Dict[str, Any],
        execution_price: Optional[float] = None,
        execution_time: Optional[int] = None
    ) -> CostBreakdown:
        """
        Calculate comprehensive costs for a trade.
        
        Args:
            order: Order details
            market_data: Current market conditions
            execution_price: Actual execution price (if different from order)
            execution_time: Execution timestamp
            
        Returns:
            CostBreakdown with detailed cost analysis
        """
        self.stats['total_trades'] += 1
        
        # 1. Exchange fees
        exchange_fee = self._calculate_exchange_fee(order, market_data)
        
        # 2. Slippage costs
        slippage_cost = self._calculate_slippage_cost(order, market_data, execution_price)
        
        # 3. Latency costs
        latency_cost = self._calculate_latency_cost(order, market_data, execution_time)
        
        # 4. Network fees
        network_fee = self._calculate_network_fee(order, market_data)
        
        # 5. Market impact
        market_impact = self._calculate_market_impact(order, market_data)
        
        # 6. Financing costs (for positions held overnight)
        financing_cost = self._calculate_financing_cost(order, market_data)
        
        # Total cost
        total_cost = (
            exchange_fee + slippage_cost + latency_cost +
            network_fee + market_impact + financing_cost
        )
        
        # Update statistics
        self.stats['total_costs'] += total_cost
        self.stats['avg_cost_per_trade'] = self.stats['total_costs'] / self.stats['total_trades']
        
        if abs(slippage_cost) > self.stats['max_slippage']:
            self.stats['max_slippage'] = abs(slippage_cost)
        
        # Update slippage average
        self.stats['avg_slippage'] = (
            (self.stats['avg_slippage'] * (self.stats['total_trades'] - 1) + abs(slippage_cost)) /
            self.stats['total_trades']
        )
        
        return CostBreakdown(
            total_cost=total_cost,
            exchange_fee=exchange_fee,
            slippage_cost=slippage_cost,
            latency_cost=latency_cost,
            network_fee=network_fee,
            market_impact=market_impact,
            financing_cost=financing_cost,
            metadata={
                'order_id': order.order_id,
                'symbol': order.symbol,
                'order_type': order.order_type.value,
                'execution_price': execution_price,
                'market_volume': market_data.get('volume', 0),
                'market_volatility': market_data.get('volatility', 0),
                'spread': market_data.get('spread', 0)
            }
        )
    
    def _calculate_exchange_fee(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate exchange trading fees."""
        
        if self.cost_model == CostModel.FIXED:
            return self._fixed_exchange_fee(order)
        
        elif self.cost_model == CostModel.TIERED:
            return self._tiered_exchange_fee(order, market_data)
        
        elif self.cost_model == CostModel.VOLUME_BASED:
            return self._volume_based_exchange_fee(order, market_data)
        
        elif self.cost_model == CostModel.DYNAMIC:
            return self._dynamic_exchange_fee(order, market_data)
        
        else:
            return self._fixed_exchange_fee(order)
    
    def _fixed_exchange_fee(self, order: Order) -> float:
        """Fixed fee calculation."""
        order_value = abs(order.size * order.price)
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            fee_rate = self.exchange_schedule.maker_fee
        else:
            fee_rate = self.exchange_schedule.taker_fee
        
        fee = order_value * fee_rate
        return max(fee, self.exchange_schedule.minimum_fee)
    
    def _tiered_exchange_fee(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Tiered fee calculation based on volume."""
        order_value = abs(order.size * order.price)
        
        # Get user's trading volume (simplified)
        user_volume = market_data.get('user_30d_volume', 0)
        
        # Find applicable tier
        fee_rate = self.exchange_schedule.taker_fee  # Default
        for volume_threshold, tier_fee in self.exchange_schedule.volume_tiers:
            if user_volume >= volume_threshold:
                fee_rate = tier_fee
        
        # Apply maker/taker distinction
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            fee_rate = min(fee_rate, self.exchange_schedule.maker_fee)
        
        fee = order_value * fee_rate
        return max(fee, self.exchange_schedule.minimum_fee)
    
    def _volume_based_exchange_fee(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Volume-based fee calculation."""
        order_value = abs(order.size * order.price)
        
        # Market volume impact
        market_volume = market_data.get('volume', 1_000_000)
        volume_ratio = abs(order.size * order.price) / market_volume
        
        # Adjust fee based on volume ratio
        base_fee_rate = self.exchange_schedule.taker_fee
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            base_fee_rate = self.exchange_schedule.maker_fee
        
        # Higher fees for large orders relative to market
        if volume_ratio > 0.01:  # > 1% of market volume
            fee_multiplier = 1.0 + (volume_ratio - 0.01) * 10
        else:
            fee_multiplier = 1.0
        
        fee_rate = base_fee_rate * fee_multiplier
        fee = order_value * fee_rate
        return max(fee, self.exchange_schedule.minimum_fee)
    
    def _dynamic_exchange_fee(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Dynamic fee calculation based on market conditions."""
        order_value = abs(order.size * order.price)
        
        # Base fee
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            base_fee_rate = self.exchange_schedule.maker_fee
        else:
            base_fee_rate = self.exchange_schedule.taker_fee
        
        # Market condition adjustments
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.0001)
        
        # Higher fees during high volatility
        volatility_multiplier = 1.0 + min(volatility * 10, 0.5)
        
        # Higher fees for wide spreads
        spread_multiplier = 1.0 + min(spread * 100, 0.3)
        
        # Time-based adjustments
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 16:  # Trading hours
            time_multiplier = 1.0
        else:  # Off-hours
            time_multiplier = 1.2
        
        fee_rate = base_fee_rate * volatility_multiplier * spread_multiplier * time_multiplier
        fee = order_value * fee_rate
        return max(fee, self.exchange_schedule.minimum_fee)
    
    def _calculate_slippage_cost(
        self,
        order: Order,
        market_data: Dict[str, Any],
        execution_price: Optional[float] = None
    ) -> float:
        """Calculate slippage costs."""
        
        if not execution_price:
            return 0.0
        
        # Base slippage
        base_slippage = self.slippage_model.base_slippage
        
        # Volatility impact
        volatility = market_data.get('volatility', 0.02)
        volatility_slippage = base_slippage * volatility * self.slippage_model.volatility_multiplier
        
        # Volume impact
        market_volume = market_data.get('volume', 1_000_000)
        order_value = abs(order.size * order.price)
        volume_impact = (order_value / market_volume) * self.slippage_model.volume_impact_factor
        
        # Order size impact
        size_impact = abs(order.size) * self.slippage_model.order_size_impact
        
        # Spread impact
        spread_impact = 0.0
        if self.slippage_model.spread_impact:
            spread = market_data.get('spread', 0.0001)
            spread_impact = spread * 0.5  # Half the spread on average
        
        # Total slippage rate
        total_slippage_rate = (
            base_slippage + volatility_slippage + volume_impact + size_impact + spread_impact
        )
        
        # Calculate slippage cost
        if order.side == 'buy':
            slippage_cost = (execution_price - order.price) * order.size
        else:  # sell
            slippage_cost = (order.price - execution_price) * order.size
        
        # Apply slippage rate if no actual execution price difference
        if abs(slippage_cost) < 0.001:  # Minimal difference
            order_value = abs(order.size * order.price)
            slippage_cost = order_value * total_slippage_rate * (1 if order.side == 'buy' else -1)
        
        return slippage_cost
    
    def _calculate_latency_cost(
        self,
        order: Order,
        market_data: Dict[str, Any],
        execution_time: Optional[int] = None
    ) -> float:
        """Calculate latency-related costs."""
        
        # Base latency
        base_latency = (
            self.latency_model.base_latency_ms +
            self.latency_model.exchange_latency_ms +
            self.latency_model.network_latency_ms
        )
        
        # Queue delay
        queue_delay = base_latency * self.latency_model.queue_delay_factor
        
        # Time-based multipliers
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 16:  # Peak hours
            time_multiplier = self.latency_model.peak_hour_multiplier
        else:
            time_multiplier = 1.0
        
        # Weekend multiplier
        import datetime
        if datetime.datetime.now().weekday() >= 5:  # Weekend
            time_multiplier *= self.latency_model.weekend_multiplier
        
        total_latency_ms = (base_latency + queue_delay) * time_multiplier
        
        # Convert to opportunity cost
        # Higher latency means worse prices, especially in volatile markets
        volatility = market_data.get('volatility', 0.02)
        order_value = abs(order.size * order.price)
        
        # Opportunity cost = latency * volatility * order_value
        latency_seconds = total_latency_ms / 1000.0
        opportunity_cost = order_value * volatility * math.sqrt(latency_seconds / 86400)  # Daily vol
        
        # Check for rejection due to latency
        if total_latency_ms > 1000:  # > 1 second
            self.stats['latency_rejections'] += 1
            # Higher cost for very high latency
            opportunity_cost *= 2.0
        
        return opportunity_cost
    
    def _calculate_network_fee(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate network/withdrawal fees."""
        
        # Simplified network fee calculation
        # In reality, this would depend on blockchain network
        
        symbol = order.symbol
        if 'BTC' in symbol:
            network_fee = 0.0001  # BTC network fee
        elif 'ETH' in symbol:
            network_fee = 0.005   # ETH network fee (gas)
        else:
            network_fee = 0.1      # USDT network fee
        
        # Scale with order size
        order_value = abs(order.size * order.price)
        if order_value > 10000:  # Large orders
            network_fee *= 2.0
        
        return network_fee
    
    def _calculate_market_impact(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate market impact of large orders."""
        
        # Market impact is proportional to order size relative to market depth
        order_value = abs(order.size * order.price)
        
        # Get market depth (simplified)
        bid_volume = market_data.get('bid_volume', 0)
        ask_volume = market_data.get('ask_volume', 0)
        total_depth = bid_volume + ask_volume
        
        if total_depth == 0:
            return 0.0
        
        # Impact factor
        depth_ratio = order_value / total_depth
        impact_rate = depth_ratio * 0.001  # 0.1% impact per depth ratio
        
        # Higher impact for aggressive orders
        if order.order_type == OrderType.MARKET:
            impact_rate *= 2.0
        
        # Calculate impact cost
        impact_cost = order_value * impact_rate
        
        return impact_cost
    
    def _calculate_financing_cost(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate financing costs for leveraged positions."""
        
        # Simplified financing cost calculation
        # In reality, this would depend on exchange funding rates
        
        if not hasattr(order, 'leverage') or order.leverage <= 1:
            return 0.0
        
        order_value = abs(order.size * order.price)
        
        # Daily funding rate (simplified)
        funding_rate = market_data.get('funding_rate', 0.0001)  # 0.01% daily
        
        # Assume position held for 1 day on average
        financing_cost = order_value * (order.leverage - 1) * funding_rate
        
        return financing_cost
    
    def update_market_data(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> None:
        """Update historical market data for dynamic modeling."""
        
        # Update volume history
        volume = market_data.get('volume', 0)
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > 1000:
            self.volume_history[symbol] = self.volume_history[symbol][-1000:]
        
        # Update volatility history
        volatility = market_data.get('volatility', 0)
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        self.volatility_history[symbol].append(volatility)
        if len(self.volatility_history[symbol]) > 1000:
            self.volatility_history[symbol] = self.volatility_history[symbol][-1000:]
        
        # Update spread history
        spread = market_data.get('spread', 0)
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []
        self.spread_history[symbol].append(spread)
        if len(self.spread_history[symbol]) > 1000:
            self.spread_history[symbol] = self.spread_history[symbol][-1000:]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost statistics."""
        return {
            'total_trades': self.stats['total_trades'],
            'total_costs': self.stats['total_costs'],
            'avg_cost_per_trade': self.stats['avg_cost_per_trade'],
            'max_slippage': self.stats['max_slippage'],
            'avg_slippage': self.stats['avg_slippage'],
            'latency_rejections': self.stats['latency_rejections'],
            'cost_breakdown': {
                'exchange_fees': self.stats.get('total_exchange_fees', 0),
                'slippage_costs': self.stats.get('total_slippage_costs', 0),
                'latency_costs': self.stats.get('total_latency_costs', 0),
                'network_fees': self.stats.get('total_network_fees', 0),
                'market_impact': self.stats.get('total_market_impact', 0),
                'financing_costs': self.stats.get('total_financing_costs', 0)
            },
            'cost_as_percentage': self._calculate_cost_percentage(),
            'model_parameters': {
                'exchange_schedule': self.exchange_schedule,
                'slippage_model': self.slippage_model,
                'latency_model': self.latency_model,
                'cost_model': self.cost_model.value
            }
        }
    
    def _calculate_cost_percentage(self) -> float:
        """Calculate total costs as percentage of traded value."""
        if self.stats['total_trades'] == 0:
            return 0.0
        
        # This would need total traded value tracking
        # Simplified calculation
        avg_trade_value = 10000  # Placeholder
        total_traded_value = self.stats['total_trades'] * avg_trade_value
        
        return (self.stats['total_costs'] / total_traded_value) * 100 if total_traded_value > 0 else 0.0


# Convenience functions
def create_binance_cost_calculator() -> RealisticCostCalculator:
    """Create cost calculator for Binance."""
    exchange_schedule = ExchangeFeeSchedule(
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0004,  # 0.04%
        volume_tiers=[
            (0, 0.0004),
            (50000, 0.00035),
            (250000, 0.0003),
            (1000000, 0.00025),
            (5000000, 0.0002)
        ]
    )
    
    slippage_model = SlippageModel(
        base_slippage=0.0003,
        volatility_multiplier=1.5
    )
    
    latency_model = LatencyModel(
        base_latency_ms=30.0,
        exchange_latency_ms=15.0
    )
    
    return RealisticCostCalculator(
        exchange_schedule=exchange_schedule,
        slippage_model=slippage_model,
        latency_model=latency_model,
        cost_model=CostModel.DYNAMIC
    )


def create_conservative_cost_calculator() -> RealisticCostCalculator:
    """Create conservative cost calculator (higher costs)."""
    exchange_schedule = ExchangeFeeSchedule(
        maker_fee=0.0005,  # 0.05%
        taker_fee=0.0010,  # 0.10%
        minimum_fee=1.0
    )
    
    slippage_model = SlippageModel(
        base_slippage=0.0010,  # 0.10%
        volatility_multiplier=3.0,
        volume_impact_factor=0.0002
    )
    
    latency_model = LatencyModel(
        base_latency_ms=100.0,
        exchange_latency_ms=50.0,
        network_latency_ms=100.0
    )
    
    return RealisticCostCalculator(
        exchange_schedule=exchange_schedule,
        slippage_model=slippage_model,
        latency_model=latency_model,
        cost_model=CostModel.DYNAMIC
    )


if __name__ == "__main__":
    # Test cost calculator
    logging.basicConfig(level=logging.INFO)
    
    calculator = create_binance_cost_calculator()
    
    # Test order
    order = Order(
        order_id="test_001",
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side="buy",
        size=0.1,
        price=42000.0,
        status=OrderStatus.FILLED,
        timestamp=int(time.time() * 1000)
    )
    
    # Market data
    market_data = {
        'volume': 1000000,
        'volatility': 0.03,
        'spread': 0.0002,
        'bid_volume': 500000,
        'ask_volume': 500000,
        'funding_rate': 0.0001
    }
    
    # Calculate costs
    costs = calculator.calculate_trade_costs(
        order=order,
        market_data=market_data,
        execution_price=42005.0  # 5 USD slippage
    )
    
    print(f"Cost breakdown for {order.symbol}:")
    print(f"  Total cost: ${costs.total_cost:.4f}")
    print(f"  Exchange fee: ${costs.exchange_fee:.4f}")
    print(f"  Slippage cost: ${costs.slippage_cost:.4f}")
    print(f"  Latency cost: ${costs.latency_cost:.4f}")
    print(f"  Network fee: ${costs.network_fee:.4f}")
    print(f"  Market impact: ${costs.market_impact:.4f}")
    print(f"  Financing cost: ${costs.financing_cost:.4f}")
    
    print(f"\nCost summary: {calculator.get_cost_summary()}")
