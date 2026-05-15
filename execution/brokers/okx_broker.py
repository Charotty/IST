"""
OKX broker for live trading.

Implements live trading execution on OKX exchange using ccxt library.
"""

import ccxt
from typing import Dict, Any, Optional
import time

from .base_broker import (
    BaseBroker, Order, OrderType, OrderSide, OrderStatus,
    Position, AccountInfo
)


class OKXBroker(BaseBroker):
    """
    OKX broker for live trading.
    
    Uses ccxt library to interact with OKX exchange API.
    Supports sandbox mode for testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OKX broker.
        
        Args:
            config: Broker configuration with api_key, secret, passphrase
        """
        super().__init__(config)
        
        # OKX credentials
        self.api_key = config.get('api_key')
        self.secret = config.get('secret')
        self.passphrase = config.get('passphrase')
        
        # Initialize ccxt exchange
        self.exchange = ccxt.okx({
            'apiKey': self.api_key,
            'secret': self.secret,
            'password': self.passphrase,
            'sandbox': self.sandbox,
            'enableRateLimit': True,
        })
        
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to OKX exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by fetching balance
            self.exchange.fetch_balance()
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to OKX: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from OKX exchange.
        
        Returns:
            True (always successful)
        """
        self.connected = False
        return True
    
    def place_order(self, order: Order) -> Order:
        """
        Place an order on OKX.
        
        Args:
            order: Order to place
            
        Returns:
            Order with updated status and order_id
        """
        if not self.connected:
            order.status = OrderStatus.REJECTED
            order.error_message = "Not connected to exchange"
            return order
        
        if not self.validate_order(order):
            order.status = OrderStatus.REJECTED
            order.error_message = "Invalid order"
            return order
        
        try:
            # Map order types to ccxt
            ccxt_order_type = self._map_order_type(order.order_type)
            ccxt_side = self._map_order_side(order.order_side)
            
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'type': ccxt_order_type,
                'side': ccxt_side,
                'amount': order.quantity,
            }
            
            # Add price for limit orders
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
                order_params['price'] = order.price
            
            # Add stop price for stop orders
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price:
                order_params['stopPrice'] = order.stop_price
            
            # Place order via ccxt
            ccxt_order = self.exchange.create_order(**order_params)
            
            # Update order with exchange response
            order.order_id = ccxt_order['id']
            order.status = self._map_order_status(ccxt_order['status'])
            order.timestamp = ccxt_order.get('timestamp', time.time())
            
            if order.status == OrderStatus.FILLED:
                order.filled_quantity = ccxt_order.get('filled', 0)
                order.filled_price = ccxt_order.get('price', order.price)
                order.fees = self._extract_fees(ccxt_order)
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on OKX.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status from OKX.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        if not self.connected:
            return OrderStatus.REJECTED
        
        try:
            ccxt_order = self.exchange.fetch_order(order_id)
            return self._map_order_status(ccxt_order['status'])
        except Exception as e:
            print(f"Failed to fetch order status: {e}")
            return OrderStatus.REJECTED
    
    def get_account_info(self) -> AccountInfo:
        """
        Get account information from OKX.
        
        Returns:
            AccountInfo with balance and positions
        """
        if not self.connected:
            return AccountInfo(balance=0.0, available_balance=0.0, positions={})
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Get total balance
            total_balance = balance.get('USDT', {}).get('total', 0.0)
            available_balance = balance.get('USDT', {}).get('free', 0.0)
            
            # Get positions
            positions = {}
            positions_data = self.exchange.fetch_positions()
            
            for pos in positions_data:
                if float(pos.get('contracts', 0)) > 0:
                    symbol = pos['symbol']
                    quantity = float(pos['contracts'])
                    entry_price = float(pos.get('entryPrice', 0))
                    current_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                    
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        current_price=current_price,
                        unrealized_pnl=unrealized_pnl
                    )
            
            return AccountInfo(
                balance=total_balance,
                available_balance=available_balance,
                positions=positions
            )
            
        except Exception as e:
            print(f"Failed to fetch account info: {e}")
            return AccountInfo(balance=0.0, available_balance=0.0, positions={})
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol from OKX.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        if not self.connected:
            return None
        
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    return Position(
                        symbol=symbol,
                        quantity=float(pos['contracts']),
                        entry_price=float(pos.get('entryPrice', 0)),
                        current_price=float(pos.get('markPrice', 0)),
                        unrealized_pnl=float(pos.get('unrealizedPnl', 0))
                    )
            
            return None
            
        except Exception as e:
            print(f"Failed to fetch position: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol from OKX.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        if not self.connected:
            return 0.0
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except Exception as e:
            print(f"Failed to fetch price: {e}")
            return 0.0
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map order type to ccxt format."""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit'
        }
        return mapping.get(order_type, 'market')
    
    def _map_order_side(self, order_side: OrderSide) -> str:
        """Map order side to ccxt format."""
        mapping = {
            OrderSide.BUY: 'buy',
            OrderSide.SELL: 'sell'
        }
        return mapping.get(order_side, 'buy')
    
    def _map_order_status(self, ccxt_status: str) -> OrderStatus:
        """Map ccxt order status to internal format."""
        mapping = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        return mapping.get(ccxt_status.lower(), OrderStatus.PENDING)
    
    def _extract_fees(self, ccxt_order: Dict[str, Any]) -> float:
        """Extract total fees from ccxt order."""
        fees = ccxt_order.get('fee', {})
        if isinstance(fees, dict):
            return float(fees.get('cost', 0))
        return 0.0
