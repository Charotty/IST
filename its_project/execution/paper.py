from __future__ import annotations

import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from its_project.execution.base import BaseExecutor, Order, OrderType, OrderStatus, Position
except ImportError:
    from .base import BaseExecutor, Order, OrderType, OrderStatus, Position


@dataclass
class BalanceSnapshot:
    """Snapshot of balance state."""
    timestamp: datetime
    balances: Dict[str, float]
    total_equity_usdt: float
    unrealized_pnl: float
    realized_pnl: float
    total_commissions: float
    total_slippage: float


@dataclass
class CommissionConfig:
    """Commission configuration."""
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    fee_currency: str = "quote"  # 'quote' or 'base'
    min_commission: float = 0.0


@dataclass
class SlippageConfig:
    """Slippage configuration."""
    enabled: bool = True
    base_slippage: float = 0.0005  # 0.05%
    max_slippage: float = 0.01  # 1%
    volatility_based: bool = True
    volatility_multiplier: float = 2.0


@dataclass
class Position:
    """Realistic position tracking."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float  # Position size in base currency
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    trades_count: int = 0
    avg_entry_price: float = 0.0
    total_cost: float = 0.0


@dataclass
class Trade:
    """Trade execution record."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    order_id: str
    position_side: Optional[str] = None  # 'long', 'short', or None for closing
    pnl: Optional[float] = None  # Realized PnL for closing trades


@dataclass
class EquitySnapshot:
    """Equity snapshot with detailed breakdown."""
    timestamp: datetime
    total_equity: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_commissions: float
    total_slippage: float
    net_equity: float  # After all costs


class PaperTradingExecutor(BaseExecutor):
    """
    Paper trading simulation without real money.
    
    Enhanced with:
    - Real balance tracking with detailed snapshots
    - Commission calculation and deduction
    - Slippage simulation for realistic execution
    - PnL tracking and equity curve
    - Transaction cost analysis
    
    Used for:
    - Real-time strategy testing
    - Exchange integration verification
    - Execution logic debugging
    - Performance analysis with realistic costs
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Virtual balance with detailed tracking
        self.balances = {
            "USDT": config.get("initial_balance", 10000.0),
            "BTC": 0.0,
            "ETH": 0.0,
        }
        
        # Balance history for equity curve
        self.balance_history: List[BalanceSnapshot] = []
        self._record_balance_snapshot()
        
        # Order counter
        self.order_counter = 0
        
        # Virtual positions
        self.positions: List[Position] = []
        
        # Simulation latency
        self.latency = config.get("latency_ms", 50) / 1000  # seconds
        
        # Current prices (simplified - should come from market data)
        self.current_prices = {
            "BTC/USDT": 42000.0,
            "ETH/USDT": 2500.0,
        }
        
        # Commission configuration
        commission_config = config.get("commission", {})
        self.commission_config = CommissionConfig(
            maker_fee=commission_config.get("maker_fee", 0.001),
            taker_fee=commission_config.get("taker_fee", 0.001),
            fee_currency=commission_config.get("fee_currency", "quote"),
            min_commission=commission_config.get("min_commission", 0.0)
        )
        
        # Slippage configuration
        slippage_config = config.get("slippage", {})
        self.slippage_config = SlippageConfig(
            enabled=slippage_config.get("enabled", True),
            base_slippage=slippage_config.get("base_slippage", 0.0005),
            max_slippage=slippage_config.get("max_slippage", 0.01),
            volatility_based=slippage_config.get("volatility_based", True),
            volatility_multiplier=slippage_config.get("volatility_multiplier", 2.0)
        )
        
        # Transaction cost tracking
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.commission_history: List[Dict[str, Any]] = []
        self.slippage_history: List[Dict[str, Any]] = []
        
        # PnL tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Realistic position tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
        # Equity tracking
        self.equity_history: List[EquitySnapshot] = []
        self._record_equity_snapshot()

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """Create virtual order with commission and slippage."""
        
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Generate ID
        self.order_counter += 1
        order_id = f"paper_{self.order_counter}"
        
        # Get current price
        current_price = await self._fetch_current_price(symbol)
        
        # Determine execution price with slippage
        if order_type == OrderType.MARKET:
            execution_price = self._apply_slippage(current_price, side, symbol)
        else:
            execution_price = price or current_price
        
        # Calculate commission
        commission = self._calculate_commission(amount, execution_price, order_type, side, symbol)
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=execution_price,
            status=OrderStatus.PENDING,
            filled=0.0,
            remaining=amount,
            timestamp=int(time.time() * 1000),
            info={
                "commission": commission,
                "slippage": abs(execution_price - current_price) if order_type == OrderType.MARKET else 0.0
            },
        )
        
        # Immediately fill market orders
        if order_type == OrderType.MARKET:
            await self._fill_order(order, execution_price, commission)
        
        self.active_orders[order_id] = order
        
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel virtual order."""
        await asyncio.sleep(self.latency)
        
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            if order.status in {OrderStatus.PENDING, OrderStatus.OPEN}:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def fetch_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        await asyncio.sleep(self.latency)
        
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Return dummy order if not found
        return Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side="buy",
            amount=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            filled=0.0,
            remaining=0.0,
            timestamp=int(time.time() * 1000),
            info={"error": "Order not found"},
        )

    async def fetch_balance(self) -> Dict[str, float]:
        """Get virtual balance."""
        await asyncio.sleep(self.latency)
        return self.balances.copy()

    async def fetch_positions(self) -> List[Position]:
        """Get virtual positions."""
        await asyncio.sleep(self.latency)
        return self.positions.copy()

    def is_live(self) -> bool:
        """This is paper trading."""
        return False

    async def _fetch_current_price(self, symbol: str) -> float:
        """Get current price (simplified)."""
        # In real implementation, this would come from market data
        return self.current_prices.get(symbol, 0.0)

    async def _fill_order(self, order: Order, price: float, commission: float) -> None:
        """Fill order with commission deduction and position tracking."""
        base, quote = order.symbol.split("/")
        slippage = abs(price - self.current_prices.get(order.symbol, price))
        
        if order.side == "buy":
            # Buy base currency with quote
            cost = order.amount * price
            total_cost = cost + commission
            
            if self.balances.get(quote, 0) < total_cost:
                order.status = OrderStatus.REJECTED
                return
            
            self.balances[quote] -= total_cost
            self.balances[base] = self.balances.get(base, 0) + order.amount
            
        else:  # sell
            # Sell base currency for quote
            if self.balances.get(base, 0) < order.amount:
                order.status = OrderStatus.REJECTED
                return
            
            proceeds = order.amount * price
            net_proceeds = proceeds - commission
            
            self.balances[base] -= order.amount
            self.balances[quote] = self.balances.get(quote, 0) + net_proceeds
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled = order.amount
        order.remaining = 0.0
        
        # Record commission
        self.total_commissions += commission
        self.commission_history.append({
            "timestamp": datetime.now(),
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "amount": order.amount,
            "price": price,
            "commission": commission,
            "commission_rate": commission / (order.amount * price) if order.amount * price > 0 else 0
        })
        
        # Create trade record
        trade = await self._create_trade_record(order, price, commission, slippage)
        
        # Update positions with realistic logic
        await self._update_positions_realistic(trade)
        
        # Record balance and equity snapshots
        self._record_balance_snapshot()
        self._record_equity_snapshot()

    async def _create_trade_record(self, order: Order, price: float, commission: float, slippage: float) -> Trade:
        """Create detailed trade record."""
        self.trade_counter += 1
        trade_id = f"trade_{self.trade_counter}"
        
        trade = Trade(
            id=trade_id,
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=price,
            commission=commission,
            slippage=slippage,
            timestamp=datetime.now(),
            order_id=order.id
        )
        
        self.trades.append(trade)
        return trade
    
    async def _update_positions_realistic(self, trade: Trade) -> None:
        """Update positions with realistic trading logic."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # No existing position, create new one
            position_side = "long" if trade.side == "buy" else "short"
            position = Position(
                symbol=symbol,
                side=position_side,
                size=trade.amount,
                entry_price=trade.price,
                current_price=trade.price,
                commission_paid=trade.commission,
                trades_count=1,
                avg_entry_price=trade.price,
                total_cost=trade.amount * trade.price + trade.commission
            )
            
            self.positions[symbol] = position
            trade.position_side = position_side
            
        else:
            # Existing position, update it
            position = self.positions[symbol]
            current_price = self.current_prices.get(symbol, trade.price)
            
            # Calculate PnL for position modification
            if (position.side == "long" and trade.side == "sell") or \
               (position.side == "short" and trade.side == "buy"):
                # Closing or reducing position
                pnl_amount = self._calculate_position_pnl(position, trade.amount, trade.price)
                trade.pnl = pnl_amount
                
                # Update realized PnL
                self.realized_pnl += pnl_amount
                position.realized_pnl += pnl_amount
                
                # Reduce position size
                if trade.amount >= position.size:
                    # Close position completely
                    self.realized_pnl += position.unrealized_pnl
                    position.realized_pnl += position.unrealized_pnl
                    del self.positions[symbol]
                    trade.position_side = None
                else:
                    # Reduce position size
                    position.size -= trade.amount
                    position.trades_count += 1
                    position.commission_paid += trade.commission
                    position.total_cost += trade.amount * trade.price + trade.commission
                    # Recalculate average entry price
                    position.avg_entry_price = position.total_cost / position.size
                    trade.position_side = position.side
            else:
                # Adding to position
                if position.side == "long" and trade.side == "buy":
                    # Adding to long position
                    new_total_cost = position.total_cost + trade.amount * trade.price + trade.commission
                    new_total_size = position.size + trade.amount
                    position.avg_entry_price = new_total_cost / new_total_size
                    position.size = new_total_size
                    position.total_cost = new_total_cost
                    position.commission_paid += trade.commission
                    position.trades_count += 1
                    trade.position_side = "long"
                    
                elif position.side == "short" and trade.side == "sell":
                    # Adding to short position
                    new_total_cost = position.total_cost + trade.amount * trade.price + trade.commission
                    new_total_size = position.size + trade.amount
                    position.avg_entry_price = new_total_cost / new_total_size
                    position.size = new_total_size
                    position.total_cost = new_total_cost
                    position.commission_paid += trade.commission
                    position.trades_count += 1
                    trade.position_side = "short"
        
        # Update unrealized PnL for all positions
        self._update_unrealized_pnl()
    
    def _calculate_position_pnl(self, position: Position, close_amount: float, close_price: float) -> float:
        """Calculate PnL for position closing."""
        if position.side == "long":
            # Long position: (close_price - entry_price) * amount
            return (close_price - position.avg_entry_price) * close_amount
        else:
            # Short position: (entry_price - close_price) * amount
            return (position.avg_entry_price - close_price) * close_amount
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL for all positions."""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position.current_price)
            position.current_price = current_price
            
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.avg_entry_price) * position.size
            else:
                position.unrealized_pnl = (position.avg_entry_price - current_price) * position.size
            
            total_unrealized += position.unrealized_pnl
        
        self.unrealized_pnl = total_unrealized
    
    def _record_equity_snapshot(self) -> None:
        """Record detailed equity snapshot."""
        cash_balance = self.balances.get("USDT", 0.0)
        
        # Calculate positions value
        positions_value = 0.0
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position.current_price)
            positions_value += position.size * current_price
        
        total_equity = cash_balance + positions_value
        net_equity = total_equity - self.total_commissions - self.total_slippage
        
        snapshot = EquitySnapshot(
            timestamp=datetime.now(),
            total_equity=total_equity,
            cash_balance=cash_balance,
            positions_value=positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_commissions=self.total_commissions,
            total_slippage=self.total_slippage,
            net_equity=net_equity
        )
        
        self.equity_history.append(snapshot)
    
    def _calculate_commission(
        self,
        amount: float,
        price: float,
        order_type: OrderType,
        side: str,
        symbol: str
    ) -> float:
        """Calculate commission for order."""
        # Determine fee rate based on order type
        if order_type == OrderType.LIMIT:
            fee_rate = self.commission_config.maker_fee
        else:
            fee_rate = self.commission_config.taker_fee
        
        # Calculate commission amount
        commission = amount * price * fee_rate
        
        # Apply minimum commission
        commission = max(commission, self.commission_config.min_commission)
        
        return commission
    
    def _apply_slippage(self, price: float, side: str, symbol: str) -> float:
        """Apply slippage to execution price."""
        if not self.slippage_config.enabled:
            return price
        
        # Calculate slippage percentage
        slippage_pct = self.slippage_config.base_slippage
        
        # Add randomness for realistic simulation
        slippage_pct *= random.uniform(0.5, 1.5)
        
        # Apply volatility multiplier if enabled
        if self.slippage_config.volatility_based:
            # In real implementation, this would use actual volatility
            slippage_pct *= self.slippage_config.volatility_multiplier
        
        # Cap at maximum slippage
        slippage_pct = min(slippage_pct, self.slippage_config.max_slippage)
        
        # Apply slippage based on side
        if side == "buy":
            # Buy orders pay more (worse price)
            execution_price = price * (1 + slippage_pct)
        else:
            # Sell orders receive less (worse price)
            execution_price = price * (1 - slippage_pct)
        
        # Record slippage
        slippage_amount = abs(execution_price - price)
        self.total_slippage += slippage_amount
        self.slippage_history.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": side,
            "original_price": price,
            "execution_price": execution_price,
            "slippage_amount": slippage_amount,
            "slippage_pct": slippage_pct
        })
        
        return execution_price
    
    def _record_balance_snapshot(self) -> None:
        """Record current balance state."""
        total_equity = self._calculate_total_equity()
        
        snapshot = BalanceSnapshot(
            timestamp=datetime.now(),
            balances=self.balances.copy(),
            total_equity_usdt=total_equity,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_commissions=self.total_commissions,
            total_slippage=self.total_slippage
        )
        
        self.balance_history.append(snapshot)
    
    def _calculate_total_equity(self) -> float:
        """Calculate total equity in USDT."""
        total_equity = 0.0
        
        for currency, amount in self.balances.items():
            if currency == "USDT":
                total_equity += amount
            else:
                # Convert to USDT using current price
                symbol = f"{currency}/USDT"
                price = self.current_prices.get(symbol, 0.0)
                total_equity += amount * price
        
        return total_equity
    
    def get_balance_history(self) -> List[BalanceSnapshot]:
        """Get balance history for equity curve."""
        return self.balance_history.copy()
    
    def get_commission_history(self) -> List[Dict[str, Any]]:
        """Get commission history."""
        return self.commission_history.copy()
    
    def get_slippage_history(self) -> List[Dict[str, Any]]:
        """Get slippage history."""
        return self.slippage_history.copy()
    
    def get_transaction_costs(self) -> Dict[str, Any]:
        """Get total transaction costs."""
        return {
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "total_costs": self.total_commissions + self.total_slippage,
            "commission_count": len(self.commission_history),
            "slippage_count": len(self.slippage_history),
            "avg_commission": self.total_commissions / len(self.commission_history) if self.commission_history else 0.0,
            "avg_slippage": self.total_slippage / len(self.slippage_history) if self.slippage_history else 0.0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.balance_history:
            return {}
        
        initial_balance = self.balance_history[0].total_equity_usdt
        current_balance = self.balance_history[-1].total_equity_usdt
        total_return = (current_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0
        
        # Calculate position statistics
        total_positions = len(self.positions)
        long_positions = sum(1 for p in self.positions.values() if p.side == "long")
        short_positions = sum(1 for p in self.positions.values() if p.side == "short")
        
        # Calculate trade statistics
        winning_trades = sum(1 for t in self.trades if t.pnl and t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl and t.pnl < 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0.0
        
        return {
            "initial_balance": initial_balance,
            "current_balance": current_balance,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "net_return": (current_balance - initial_balance - self.total_commissions - self.total_slippage) / initial_balance if initial_balance > 0 else 0.0,
            "net_return_pct": ((current_balance - initial_balance - self.total_commissions - self.total_slippage) / initial_balance * 100) if initial_balance > 0 else 0.0,
            "trade_count": len(self.trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_positions": total_positions,
            "long_positions": long_positions,
            "short_positions": short_positions,
            "current_balances": self.balances.copy()
        }
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_trades(self) -> List[Trade]:
        """Get trade history."""
        return self.trades.copy()
    
    def get_equity_history(self) -> List[EquitySnapshot]:
        """Get equity history for analysis."""
        return self.equity_history.copy()
    
    def get_position_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed position summary."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = self.current_prices.get(symbol, position.current_price)
        
        return {
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "avg_entry_price": position.avg_entry_price,
            "current_price": current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "total_pnl": position.unrealized_pnl + position.realized_pnl,
            "pnl_pct": ((position.unrealized_pnl + position.realized_pnl) / position.total_cost * 100) if position.total_cost > 0 else 0.0,
            "commission_paid": position.commission_paid,
            "trades_count": position.trades_count,
            "timestamp": position.timestamp
        }
    
    def close_position(self, symbol: str, amount: Optional[float] = None) -> Optional[Trade]:
        """Close position (simulated)."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        close_amount = amount or position.size
        current_price = self.current_prices.get(symbol, position.current_price)
        
        # Determine closing side
        close_side = "sell" if position.side == "long" else "buy"
        
        # Calculate commission
        commission = self._calculate_commission(close_amount, current_price, OrderType.MARKET, close_side, symbol)
        
        # Create closing trade
        self.trade_counter += 1
        trade = Trade(
            id=f"trade_{self.trade_counter}",
            symbol=symbol,
            side=close_side,
            amount=close_amount,
            price=current_price,
            commission=commission,
            slippage=0.0,
            timestamp=datetime.now(),
            order_id="close_position",
            position_side=None,
            pnl=self._calculate_position_pnl(position, close_amount, current_price)
        )
        
        # Update balances
        base, quote = symbol.split("/")
        if position.side == "long":
            # Close long position
            proceeds = close_amount * current_price - commission
            self.balances[base] = self.balances.get(base, 0) - close_amount
            self.balances[quote] = self.balances.get(quote, 0) + proceeds
        else:
            # Close short position
            proceeds = close_amount * current_price - commission
            self.balances[base] = self.balances.get(base, 0) + close_amount
            self.balances[quote] = self.balances.get(quote, 0) + proceeds
        
        # Update PnL
        self.realized_pnl += trade.pnl
        self.realized_pnl += position.unrealized_pnl
        
        # Remove or reduce position
        if close_amount >= position.size:
            del self.positions[symbol]
        else:
            position.size -= close_amount
            position.trades_count += 1
        
        # Record trade and update snapshots
        self.trades.append(trade)
        self._record_balance_snapshot()
        self._record_equity_snapshot()
        
        return trade
    
    def update_prices(self, new_prices: Dict[str, float]) -> None:
        """Update current prices and recalculate unrealized PnL."""
        self.current_prices.update(new_prices)
        self._update_unrealized_pnl()
        self._record_equity_snapshot()
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get detailed trade statistics."""
        if not self.trades:
            return {}
        
        profitable_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades_list = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        avg_win = sum(t.pnl for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0.0
        avg_loss = sum(t.pnl for t in losing_trades_list) / len(losing_trades_list) if losing_trades_list else 0.0
        
        return {
            "total_trades": len(self.trades),
            "profitable_trades": len(profitable_trades),
            "losing_trades": len(losing_trades_list),
            "win_rate": len(profitable_trades) / len(self.trades) if self.trades else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            "total_volume": sum(t.amount * t.price for t in self.trades),
            "avg_trade_size": sum(t.amount for t in self.trades) / len(self.trades) if self.trades else 0.0
        }
