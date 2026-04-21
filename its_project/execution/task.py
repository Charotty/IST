#!/usr/bin/env python3
"""
Execution Task for Pipeline Integration
=====================================

This task integrates the execution layer with the main pipeline.
It consumes trading decisions and executes them through the appropriate executor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from its_project.execution.base import BaseExecutor, OrderType, OrderStatus
    from its_project.execution.manager import OrderManager
    from its_project.execution.paper import PaperTradingExecutor
    from its_project.execution.live import LiveExecutor
    from its_project.decision.decision import Decision, Action
except ImportError:
    from .base import BaseExecutor, OrderType, OrderStatus
    from .manager import OrderManager
    from .paper import PaperTradingExecutor
    from .live import LiveExecutor
    try:
        from decision.decision import Decision, Action
    except ImportError:
        from its_project.decision.decision import Decision, Action

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order_id: Optional[str]
    error: Optional[str]
    execution_time: int
    decision: Decision


class ExecutionTask:
    """
    Async task for executing trading decisions.
    
    This task:
    - Consumes decisions from the decision queue
    - Executes orders through the appropriate executor
    - Manages order lifecycle and monitoring
    - Handles errors and retries
    - Provides execution feedback
    """
    
    def __init__(
        self,
        decision_queue: asyncio.Queue,
        execution_result_queue: asyncio.Queue,
        config: Dict[str, Any]
    ) -> None:
        self.decision_queue = decision_queue
        self.execution_result_queue = execution_result_queue
        self.config = config
        
        # Execution mode
        self.mode = ExecutionMode(config.get("execution_mode", "paper"))
        
        # Initialize executor and manager
        self.executor = self._create_executor()
        self.order_manager = OrderManager(self.executor, config.get("order_manager", {}))
        
        # Task state
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "decisions_processed": 0,
            "orders_executed": 0,
            "orders_failed": 0,
            "total_pnl": 0.0,
            "start_time": None,
        }
    
    def _create_executor(self) -> BaseExecutor:
        """Create appropriate executor based on mode."""
        if self.mode == ExecutionMode.PAPER:
            logger.info("Initializing Paper Trading Executor")
            return PaperTradingExecutor(self.config.get("paper_executor", {}))
        elif self.mode == ExecutionMode.LIVE:
            logger.warning("Initializing LIVE Trading Executor - USE WITH CAUTION!")
            return LiveExecutor(self.config.get("live_executor", {}))
        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")
    
    async def start(self) -> None:
        """Start execution task."""
        if self.running:
            logger.warning("Execution task already running")
            return
        
        logger.info(f"Starting Execution Task in {self.mode.value} mode")
        self.running = True
        self.stats["start_time"] = asyncio.get_event_loop().time()
        
        # Start order manager
        await self.order_manager.start()
        
        # Start main execution loop
        self.task = asyncio.create_task(self._execution_loop())
        
        logger.info("Execution Task started successfully")
    
    async def stop(self) -> None:
        """Stop execution task."""
        if not self.running:
            return
        
        logger.info("Stopping Execution Task")
        self.running = False
        self.stop_event.set()
        
        # Stop order manager
        await self.order_manager.stop()
        
        # Cancel main task
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info("Execution Task stopped")
    
    async def _execution_loop(self) -> None:
        """Main execution loop."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get decision with timeout
                decision = await asyncio.wait_for(
                    self.decision_queue.get(),
                    timeout=1.0
                )
                
                # Process decision
                result = await self._process_decision(decision)
                
                # Send result back
                await self.execution_result_queue.put(result)
                
                # Update statistics
                self.stats["decisions_processed"] += 1
                if result.success:
                    self.stats["orders_executed"] += 1
                else:
                    self.stats["orders_failed"] += 1
                
            except asyncio.TimeoutError:
                # No decisions to process, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                # Continue processing other decisions
    
    async def _process_decision(self, decision: Decision) -> ExecutionResult:
        """Process a single trading decision."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Processing decision: {decision.action} {decision.size} {decision.symbol}")
            
            # Skip HOLD actions
            if decision.action in [Action.HOLD]:
                logger.info("Skipping HOLD action")
                return ExecutionResult(
                    success=True,
                    order_id=None,
                    error=None,
                    execution_time=int((asyncio.get_event_loop().time() - start_time) * 1000),
                    decision=decision
                )
            
            # Convert action to order side and type
            order_side, order_type = self._action_to_order_params(decision.action)
            
            # Execute order through order manager
            order_id = await self.order_manager.execute_decision(
                symbol=decision.symbol,
                side=order_side,
                amount=decision.size,
                order_type="market",  # Default to market orders
                price=decision.price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit
            )
            
            if order_id:
                logger.info(f"Order executed successfully: {order_id}")
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    error=None,
                    execution_time=int((asyncio.get_event_loop().time() - start_time) * 1000),
                    decision=decision
                )
            else:
                logger.error("Order execution failed")
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    error="Order execution failed",
                    execution_time=int((asyncio.get_event_loop().time() - start_time) * 1000),
                    decision=decision
                )
        
        except Exception as e:
            logger.error(f"Error processing decision: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=str(e),
                execution_time=int((asyncio.get_event_loop().time() - start_time) * 1000),
                decision=decision
            )
    
    def _action_to_order_params(self, action: Action) -> tuple[str, OrderType]:
        """Convert decision action to order parameters."""
        if action in [Action.BUY]:
            return "buy", OrderType.MARKET
        elif action in [Action.SELL]:
            return "sell", OrderType.MARKET
        elif action in [Action.CLOSE_LONG]:
            return "sell", OrderType.MARKET
        elif action in [Action.CLOSE_SHORT]:
            return "buy", OrderType.MARKET
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.stats.copy()
        
        if stats["start_time"]:
            stats["uptime_seconds"] = asyncio.get_event_loop().time() - stats["start_time"]
            stats["decisions_per_minute"] = (
                stats["decisions_processed"] / max(stats["uptime_seconds"] / 60, 1)
            )
        
        # Add executor-specific stats
        if hasattr(self.executor, 'balances'):
            stats["balances"] = self.executor.balances
        
        if hasattr(self.executor, 'active_orders'):
            stats["active_orders"] = len(self.executor.active_orders)
        
        return stats
    
    async def get_positions(self) -> list:
        """Get current positions."""
        return await self.executor.fetch_positions()
    
    async def get_balance(self) -> Dict[str, float]:
        """Get current balance."""
        return await self.executor.fetch_balance()
    
    def is_live(self) -> bool:
        """Check if this is live trading."""
        return self.executor.is_live()


# Factory function for easy instantiation
def create_execution_task(
    decision_queue: asyncio.Queue,
    execution_result_queue: asyncio.Queue,
    config: Dict[str, Any]
) -> ExecutionTask:
    """Create execution task with proper configuration."""
    
    # Validate configuration
    required_keys = ["execution_mode"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Safety checks for live trading
    if config["execution_mode"] == "live":
        logger.warning("LIVE TRADING MODE ENABLED!")
        logger.warning("Ensure you have:")
        logger.warning("1. Tested extensively in paper mode")
        logger.warning("2. Proper risk management settings")
        logger.warning("3. Sufficient capital")
        logger.warning("4. Monitoring and alerting")
        
        # Add safety defaults for live trading
        live_config = config.get("live_executor", {})
        live_config.setdefault("sandbox", True)  # Default to testnet
        live_config.setdefault("max_order_size", 0.1)
        live_config.setdefault("daily_loss_limit", 50.0)
        config["live_executor"] = live_config
    
    return ExecutionTask(decision_queue, execution_result_queue, config)
