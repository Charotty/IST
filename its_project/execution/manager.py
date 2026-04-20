from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional, List
import time

from its_project.execution.base import BaseExecutor, Order, OrderType, OrderStatus


class OrderManager:
    """
    Order lifecycle management with SL/TP support.
    
    Handles:
    - Order creation and tracking
    - Stop loss and take profit orders
    - Order status updates
    - Error handling and retries
    """

    def __init__(self, executor: BaseExecutor, config: Dict[str, Any]) -> None:
        self.executor = executor
        self.config = config
        
        # Active orders tracking
        self.active_orders: Dict[str, Order] = {}
        self.sl_tp_orders: Dict[str, List[str]] = {}  # main_order_id -> [sl_order_id, tp_order_id]
        
        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start order monitoring."""
        self.monitor_task = asyncio.create_task(self._monitor_orders())

    async def stop(self) -> None:
        """Stop order monitoring."""
        self.stop_event.set()
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def execute_decision(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """
        Execute trading decision with SL/TP.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Position size
            order_type: 'market' or 'limit'
            price: Order price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Main order object
        """
        
        # Create main order
        main_order = await self.executor.create_order(
            symbol=symbol,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            side=side,
            amount=amount,
            price=price,
        )
        
        self.active_orders[main_order.id] = main_order
        
        # Create SL/TP orders if specified
        sl_tp_ids = []
        
        if stop_loss is not None:
            sl_order = await self._create_stop_loss_order(
                symbol=symbol,
                side=side,
                amount=amount,
                stop_loss=stop_loss,
            )
            sl_tp_ids.append(sl_order.id)
        
        if take_profit is not None:
            tp_order = await self._create_take_profit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                take_profit=take_profit,
            )
            sl_tp_ids.append(tp_order.id)
        
        # Store SL/TP relationship
        if sl_tp_ids:
            self.sl_tp_orders[main_order.id] = sl_tp_ids
        
        return main_order

    async def _create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: float,
    ) -> Order:
        """Create stop loss order."""
        # Stop loss side is opposite to main order
        sl_side = "sell" if side == "buy" else "buy"
        
        sl_order = await self.executor.create_order(
            symbol=symbol,
            order_type=OrderType.STOP_LOSS,
            side=sl_side,
            amount=amount,
            price=stop_loss,
        )
        
        self.active_orders[sl_order.id] = sl_order
        return sl_order

    async def _create_take_profit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        take_profit: float,
    ) -> Order:
        """Create take profit order."""
        # Take profit side is opposite to main order
        tp_side = "sell" if side == "buy" else "buy"
        
        tp_order = await self.executor.create_order(
            symbol=symbol,
            order_type=OrderType.TAKE_PROFIT,
            side=tp_side,
            amount=amount,
            price=take_profit,
        )
        
        self.active_orders[tp_order.id] = tp_order
        return tp_order

    async def _monitor_orders(self) -> None:
        """Monitor order status and handle updates."""
        while not self.stop_event.is_set():
            try:
                # Update status of all active orders
                for order_id in list(self.active_orders.keys()):
                    order = await self.executor.fetch_order_status(
                        order_id, self.active_orders[order_id].symbol
                    )
                    self.active_orders[order_id] = order
                    
                    # Handle filled orders
                    if order.status == OrderStatus.FILLED:
                        await self._handle_filled_order(order)
                    
                    # Handle cancelled/rejected orders
                    elif order.status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                        await self._handle_closed_order(order)
                
                # Sleep before next check
                await asyncio.sleep(self.config.get("monitor_interval", 1.0))
                
            except Exception as e:
                # Log error and continue
                print(f"Order monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _handle_filled_order(self, order: Order) -> None:
        """Handle filled order logic."""
        # If main order is filled, keep SL/TP active
        # If SL/TP is filled, cancel other orders
        if order.id in self.sl_tp_orders:
            # This is main order, SL/TP already created
            pass
        else:
            # This might be SL/TP order, find and cancel related orders
            await self._cancel_related_orders(order.id)

    async def _handle_closed_order(self, order: Order) -> None:
        """Handle closed order (cancelled/rejected)."""
        # Remove from tracking
        self.active_orders.pop(order.id, None)
        
        # Cancel related SL/TP if main order cancelled
        if order.id in self.sl_tp_orders:
            related_ids = self.sl_tp_orders.pop(order.id, [])
            for related_id in related_ids:
                try:
                    await self.executor.cancel_order(related_id, order.symbol)
                    self.active_orders.pop(related_id, None)
                except Exception:
                    pass

    async def _cancel_related_orders(self, filled_order_id: str) -> None:
        """Cancel orders related to filled SL/TP."""
        # Find main order that has this SL/TP
        for main_id, sl_tp_ids in self.sl_tp_orders.items():
            if filled_order_id in sl_tp_ids:
                # Cancel other SL/TP orders
                for order_id in sl_tp_ids:
                    if order_id != filled_order_id:
                        try:
                            order = self.active_orders.get(order_id)
                            if order:
                                await self.executor.cancel_order(order_id, order.symbol)
                                self.active_orders.pop(order_id, None)
                        except Exception:
                            pass
                # Remove main order tracking
                self.sl_tp_orders.pop(main_id, None)
                break

    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())

    async def get_balance(self) -> Dict[str, float]:
        """Get current balance."""
        return await self.executor.fetch_balance()

    async def get_positions(self) -> List:
        """Get current positions."""
        return await self.executor.fetch_positions()
