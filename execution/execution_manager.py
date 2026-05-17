"""
Execution manager for ITS.

Main orchestrator for trade execution, integrating decision signals, risk management,
and broker operations.
"""

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

from .brokers.base_broker import BaseBroker, Order, OrderType, OrderSide
from .brokers.paper_broker import PaperBroker
from .brokers.okx_broker import OKXBroker
from .orders.order_manager import OrderManager, OrderRequest


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    mode: str = "paper"  # paper | live
    exchange: str = "okx"
    sandbox: bool = True
    default_order_type: OrderType = OrderType.MARKET
    max_order_size: Optional[float] = None
    commission_rate: float = 0.0006  # 0.06%
    slippage_rate: float = 0.0002  # 0.02%
    initial_balance: float = 10000.0
    # Live mode credentials (required for live trading)
    api_key: Optional[str] = None
    secret: Optional[str] = None
    passphrase: Optional[str] = None


class ExecutionManager:
    """
    Main execution manager for trading.
    
    Orchestrates the execution flow:
    - Receives signals from decision layer
    - Applies risk management (position sizing)
    - Submits orders via broker
    - Monitors execution
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize execution manager.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize broker based on mode
        if config.mode == "paper":
            broker_config = {
                'initial_balance': config.initial_balance,
                'commission_rate': config.commission_rate,
                'slippage_rate': config.slippage_rate,
                'sandbox': config.sandbox
            }
            self.broker = PaperBroker(broker_config)
        elif config.mode == "live":
            if config.exchange == "okx":
                if not config.api_key or not config.secret or not config.passphrase:
                    raise ValueError("Live mode requires api_key, secret, and passphrase in ExecutionConfig")
                broker_config = {
                    'api_key': config.api_key,
                    'secret': config.secret,
                    'passphrase': config.passphrase,
                    'sandbox': config.sandbox,
                    'commission_rate': config.commission_rate,
                    'slippage_rate': config.slippage_rate
                }
                self.broker = OKXBroker(broker_config)
            else:
                raise ValueError(f"Unsupported exchange: {config.exchange}")
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")
        
        # Initialize order manager
        self.order_manager = OrderManager(self.broker)
        
        # Callbacks for signal processing
        self.on_signal_callback: Optional[Callable] = None
        self.on_fill_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
    
    def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            success = self.broker.connect()
            if success:
                self.logger.info(f"Connected to broker in {self.config.mode} mode")
            else:
                self.logger.error("Failed to connect to broker")
            return success
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            success = self.broker.disconnect()
            if success:
                self.logger.info("Disconnected from broker")
            return success
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            return False
    
    def execute_signal(
        self,
        signal: int,
        symbol: str,
        position_size: float,
        current_price: float,
        risk_multiplier: float = 1.0
    ) -> Optional[Order]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            symbol: Trading symbol
            position_size: Base position size from risk management
            current_price: Current market price
            risk_multiplier: Risk multiplier from RL layer
            
        Returns:
            Executed order or None if no order placed
        """
        if signal == 0:
            self.logger.info("Signal is 0, no action taken")
            return None
        
        # Apply risk multiplier
        final_position_size = position_size * risk_multiplier
        
        # Check max order size
        if self.config.max_order_size and final_position_size > self.config.max_order_size:
            final_position_size = self.config.max_order_size
            self.logger.warning(f"Position size capped at {self.config.max_order_size}")
        
        # Determine order side
        if signal == 1:
            side = OrderSide.BUY
        elif signal == -1:
            side = OrderSide.SELL
        else:
            self.logger.warning(f"Invalid signal: {signal}")
            return None
        
        # Update paper broker price if in paper mode
        if isinstance(self.broker, PaperBroker):
            self.broker.update_price(symbol, current_price)
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.config.default_order_type,
            quantity=final_position_size,
            price=current_price if self.config.default_order_type == OrderType.LIMIT else None
        )
        
        # Submit order
        try:
            order = self.order_manager.submit_order(order_request)
            
            self.logger.info(
                f"Order submitted: {order.order_id}, {side.value} {final_position_size} {symbol}"
            )
            
            # Trigger callback
            if self.on_signal_callback:
                self.on_signal_callback(signal, order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            
            if self.on_error_callback:
                self.on_error_callback(e, order_request)
            
            return None
    
    def close_position(self, symbol: str, current_price: float) -> Optional[Order]:
        """
        Close position for a symbol.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Order to close position or None if no position
        """
        position = self.broker.get_position(symbol)
        
        if not position or position.quantity == 0:
            self.logger.info(f"No position to close for {symbol}")
            return None
        
        # Determine opposite side
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        
        # Update paper broker price if in paper mode
        if isinstance(self.broker, PaperBroker):
            self.broker.update_price(symbol, current_price)
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.config.default_order_type,
            quantity=abs(position.quantity),
            price=current_price if self.config.default_order_type == OrderType.LIMIT else None
        )
        
        # Submit order
        try:
            order = self.order_manager.submit_order(order_request)
            self.logger.info(f"Position close order submitted: {order.order_id}")
            return order
        except Exception as e:
            self.logger.error(f"Position close failed: {e}")
            return None
    
    def get_account_info(self):
        """
        Get current account information.
        
        Returns:
            AccountInfo from broker
        """
        return self.broker.get_account_info()
    
    def get_position(self, symbol: str):
        """
        Get current position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position from broker
        """
        return self.broker.get_position(symbol)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Number of orders cancelled
        """
        return self.order_manager.cancel_all_orders(symbol)
    
    def emergency_stop(self) -> int:
        """
        Emergency stop: cancel all orders and close all positions.
        
        Returns:
            Number of orders cancelled
        """
        self.logger.warning("EMERGENCY STOP triggered")
        
        # Cancel all orders
        cancelled = self.cancel_all_orders()
        
        # Close all positions (would need current prices for each symbol)
        # This is a simplified version - in production would fetch current prices
        account_info = self.get_account_info()
        
        for symbol in account_info.positions:
            self.logger.warning(f"Would close position for {symbol}")
            # In production: self.close_position(symbol, current_price)
        
        return cancelled
    
    def set_callbacks(
        self,
        on_signal: Optional[Callable] = None,
        on_fill: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Set execution callbacks.
        
        Args:
            on_signal: Callback when signal is executed
            on_fill: Callback when order is filled
            on_error: Callback when error occurs
        """
        self.on_signal_callback = on_signal
        self.on_fill_callback = on_fill
        self.on_error_callback = on_error
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        order_stats = self.order_manager.get_order_statistics()
        account_info = self.get_account_info()
        
        return {
            'order_statistics': order_stats,
            'account_balance': account_info.balance,
            'available_balance': account_info.available_balance,
            'open_positions': len(account_info.positions)
        }
