from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import signal

from its_project.common.types import MarketData
from its_project.data_layer.base import BaseDataSource
from its_project.features.pipeline import FeaturePipeline
from its_project.models.base import BaseModel
from its_project.decision.engine import TradingDecisionEngine
from its_project.execution.base import BaseExecutor
from its_project.decision.decision import Decision

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for pipeline orchestrator."""
    data_source: BaseDataSource
    feature_pipeline: FeaturePipeline
    model: BaseModel
    decision_engine: TradingDecisionEngine
    executor: BaseExecutor
    symbols: list[str]
    loop_interval_ms: int = 1000
    max_retries: int = 3
    enable_paper_trading: bool = True


@dataclass
class PipelineMetrics:
    """Metrics for pipeline monitoring."""
    processed_ticks: int = 0
    signals_generated: int = 0
    decisions_executed: int = 0
    errors: int = 0
    last_tick_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    uptime_seconds: float = 0.0


class PipelineOrchestrator:
    """
    Centralized pipeline orchestrator connecting all layers.
    
    Pipeline flow:
    Data Source → Feature Pipeline → Model → Decision Engine → Executor
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._running = False
        self._stop_event = asyncio.Event()
        self._metrics = PipelineMetrics()
        self._start_time: Optional[datetime] = None
        self._callbacks: Dict[str, list[Callable]] = {
            "on_tick": [],
            "on_signal": [],
            "on_decision": [],
            "on_error": [],
        }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for pipeline events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    async def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline orchestrator")
        
        try:
            await self.config.data_source.connect()
            logger.info("Data source connected")
            
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown all pipeline components gracefully."""
        logger.info("Shutting down pipeline orchestrator")
        self._running = False
        self._stop_event.set()
        
        try:
            await self.config.data_source.disconnect()
            logger.info("Data source disconnected")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def _process_tick(self, market_data: MarketData) -> Optional[Decision]:
        """
        Process a single market data tick through the pipeline.
        
        Returns:
            Decision if signal generated, None otherwise
        """
        try:
            self._metrics.processed_ticks += 1
            self._metrics.last_tick_time = datetime.now()
            
            await self._emit("on_tick", market_data)
            
            # Convert to DataFrame for feature pipeline
            import pandas as pd
            df = pd.DataFrame([{
                "timestamp": market_data.timestamp_ms,
                "symbol": market_data.symbol,
                **market_data.data
            }])
            
            # Feature extraction
            features = self.config.feature_pipeline.transform(df)
            
            # Model prediction
            prediction = self.config.model.predict(features)
            proba = self.config.model.predict_proba(features) if hasattr(self.config.model, "predict_proba") else None
            
            # Signal generation (through decision engine)
            from its_project.decision.decision import Signal
            signal = Signal(
                symbol=market_data.symbol,
                action=prediction[0] if prediction is not None else "HOLD",
                confidence=float(proba[0][prediction[0]]) if proba is not None else 0.0,
                timestamp=market_data.timestamp_ms,
                metadata={"prediction": prediction, "proba": proba}
            )
            
            self._metrics.signals_generated += 1
            self._metrics.last_signal_time = datetime.now()
            await self._emit("on_signal", signal)
            
            # Decision making
            market_state = {"price": market_data.data.get("last", 0.0)}
            balance = await self.config.executor.fetch_balance()
            account_balance = balance.get("USDT", 0.0)
            
            decision = self.config.decision_engine.process_signal(
                signal, market_state, account_balance
            )
            
            if decision:
                self._metrics.decisions_executed += 1
                await self._emit("on_decision", decision)
            
            return decision
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error processing tick: {e}")
            await self._emit("on_error", e)
            return None

    async def _execute_decision(self, decision: Decision) -> None:
        """Execute a trading decision."""
        try:
            from its_project.execution.base import OrderType
            
            order = await self.config.executor.create_order(
                symbol=decision.symbol,
                order_type=OrderType.MARKET,
                side=decision.action.lower(),
                amount=decision.size,
            )
            logger.info(f"Order executed: {order.id} {decision.action} {decision.size} {decision.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            await self._emit("on_error", e)

    async def _main_loop(self) -> None:
        """Main pipeline loop."""
        logger.info("Starting main pipeline loop")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Fetch latest data
                for symbol in self.config.symbols:
                    try:
                        market_data = await self.config.data_source.fetch(symbol)
                        if market_data:
                            decision = await self._process_tick(market_data)
                            if decision and self.config.enable_paper_trading:
                                await self._execute_decision(decision)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        self._metrics.errors += 1
                
                # Update uptime
                if self._start_time:
                    self._metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
                
                # Wait for next interval
                await asyncio.sleep(self.config.loop_interval_ms / 1000.0)
                
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(1)  # Brief pause on error

    async def start(self) -> None:
        """Start the pipeline orchestrator."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting pipeline orchestrator")
        self._running = True
        self._start_time = datetime.now()
        self._stop_event.clear()
        
        await self.initialize()
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Start main loop
        await self._main_loop()

    async def stop(self) -> None:
        """Stop the pipeline orchestrator."""
        logger.info("Stopping pipeline orchestrator")
        await self.shutdown()

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self._metrics

    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "uptime_seconds": self._metrics.uptime_seconds,
            "processed_ticks": self._metrics.processed_ticks,
            "signals_generated": self._metrics.signals_generated,
            "decisions_executed": self._metrics.decisions_executed,
            "errors": self._metrics.errors,
            "last_tick_time": self._metrics.last_tick_time.isoformat() if self._metrics.last_tick_time else None,
            "last_signal_time": self._metrics.last_signal_time.isoformat() if self._metrics.last_signal_time else None,
            "symbols": self.config.symbols,
        }
