from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Backend imports
try:
    from its_project.decision.signal_generator import SignalGenerator
    from its_project.decision.decision import Action, Signal
    from its_project.execution.trader import PaperTrader
    from its_project.meta.model_selector import ModelSelector
    from its_project.storage.data_loader import DataLoader
    from its_project.features.feature_builder import FeatureBuilder
    from its_project.models.gru_model import GRUModel
    from its_project.models.cnn_lob_model import CNNLOBModel
    from its_project.models.boosting_model import BoostingModel
    from its_project.features.synchronizer import synchronize_marketdata
    from its_project.common.types import MarketData, MarketDataType
except ImportError as e:
    logging.warning(f"Backend modules not available: {e}")
    # Fallback for demo mode
    SignalGenerator = None
    PaperTrader = None
    ModelSelector = None
    DataLoader = None

logger = logging.getLogger(__name__)


@dataclass
class GUISignal:
    """GUI-compatible signal format."""
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    predicted_change: float  # Percentage change
    timestamp: datetime
    symbol: str
    model_name: str


@dataclass
class GUITrade:
    """GUI-compatible trade format."""
    id: str
    timestamp: str
    pair: str
    type: str  # "BUY", "SELL"
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # "OPEN", "CLOSED"


@dataclass
class GUIMetrics:
    """GUI-compatible metrics format."""
    total_pnl: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None


class ITSBackendBridge:
    """Bridge between PyQt GUI and ITS backend systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_connected = False
        self.is_running = False
        
        # Backend systems
        self.signal_generator: Optional[SignalGenerator] = None
        self.paper_trader: Optional[PaperTrader] = None
        self.model_selector: Optional[ModelSelector] = None
        self.data_loader: Optional[DataLoader] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        
        # Models
        self.models: Dict[str, Any] = {}
        self.active_model_name: str = "GRU-LSTM"
        
        # Data
        self.current_symbol: str = "BTC/USDT"
        self.current_price: float = 0.0
        self.market_data: List[MarketData] = []
        self.features: Optional[np.ndarray] = None
        
        # GUI callbacks
        self.signal_callback: Optional[Callable[[GUISignal], None]] = None
        self.price_callback: Optional[Callable[[str, float], None]] = None
        self.log_callback: Optional[Callable[[str, str, str], None]] = None
        self.trade_callback: Optional[Callable[[GUITrade], None]] = None
        self.metrics_callback: Optional[Callable[[GUIMetrics], None]] = None
        self.model_callback: Optional[Callable[[List[Dict]], None]] = None
        
        # Async event loop
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.tasks: List[asyncio.Task] = []
        
        # Demo mode fallback
        self.demo_mode = False
        
        logger.info("ITSBackendBridge initialized")
    
    async def connect(self) -> bool:
        """Connect to backend systems."""
        try:
            # Initialize backend systems if available
            if SignalGenerator and PaperTrader and DataLoader:
                await self._initialize_backend()
                self.demo_mode = False
                logger.info("Connected to real backend systems")
            else:
                await self._initialize_demo_mode()
                self.demo_mode = True
                logger.info("Backend not available, using demo mode")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            await self._initialize_demo_mode()
            self.demo_mode = True
            return False
    
    async def _initialize_backend(self):
        """Initialize real backend systems."""
        # Initialize data loader
        parquet_path = self.config.get("parquet_path", "data/parquet")
        timescale_dsn = self.config.get("timescale_dsn", "postgresql://user:pass@localhost/its")
        
        self.data_loader = DataLoader(parquet_path, timescale_dsn)
        await self.data_loader.connect()
        
        # Initialize models
        self.models = {
            "GRU-LSTM": GRUModel(),
            "CNN-Attention": CNNLOBModel(),
            "Boosting": BoostingModel()
        }
        
        # Initialize signal generator
        signal_config = {
            "threshold_strategy": self.config.get("threshold_strategy", "adaptive"),
            "fixed_threshold": self.config.get("fixed_threshold", 0.7),
            "min_signal_strength": self.config.get("min_signal_strength", 0.1)
        }
        self.signal_generator = SignalGenerator(signal_config)
        
        # Set models in signal generator
        regression_model = self.models[self.active_model_name]
        classification_model = self.models[self.active_model_name]  # Use same model for demo
        self.signal_generator.set_models(regression_model, classification_model)
        
        # Initialize paper trader
        trader_config = {
            "initial_balance": self.config.get("initial_balance", 10000.0),
            "commission_rate": self.config.get("commission_rate", 0.001),
            "partial_fill_probability": self.config.get("partial_fill_probability", 0.1)
        }
        self.paper_trader = PaperTrader(**trader_config)
        
        # Initialize model selector
        self.model_selector = ModelSelector(
            models=list(self.models.values()),
            metrics=["accuracy", "sharpe_ratio", "max_drawdown"],
            weights=[0.3, 0.4, 0.3]
        )
        
        # Initialize feature builder
        self.feature_builder = FeatureBuilder()
        
        logger.info("Backend systems initialized")
    
    async def _initialize_demo_mode(self):
        """Initialize demo mode with mock data."""
        # Create mock models
        class MockModel:
            def __init__(self, name: str, score: float = 0.75):
                self.name = name
                self.score = score
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                probs = np.random.dirichlet(np.ones(3), size=len(X))
                return probs
        
        self.models = {
            "GRU-LSTM": MockModel("GRU-LSTM", 0.82),
            "CNN-Attention": MockModel("CNN-Attention", 0.78),
            "Boosting": MockModel("Boosting", 0.74)
        }
        
        # Initialize mock data
        await self._load_demo_data()
        
        logger.info("Demo mode initialized")
    
    async def _load_demo_data(self):
        """Load demo market data."""
        # Generate demo market data
        base_prices = {
            "BTC/USDT": 45000,
            "ETH/USDT": 2500,
            "BNB/USDT": 300
        }
        
        base_price = base_prices.get(self.current_symbol, 100)
        self.current_price = base_price
        
        # Generate recent market data
        self.market_data = []
        current_time = datetime.now()
        
        for i in range(100):
            timestamp = current_time - timedelta(minutes=(100 - i))
            price_change = np.random.normal(0, 0.001)  # 0.1% volatility
            new_price = base_price * (1 + price_change)
            
            # Create market data
            market_data = MarketData(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                symbol=self.current_symbol.replace("/", ""),
                type=MarketDataType.TRADE,
                exchange="binance",
                data={
                    "price": new_price,
                    "volume": np.random.uniform(0.1, 10.0),
                    "side": np.random.choice(["buy", "sell"])
                }
            )
            self.market_data.append(market_data)
        
        # Generate demo features
        self.features = np.random.randn(len(self.market_data), 20)
        
        logger.info(f"Loaded {len(self.market_data)} demo data points")
    
    def set_gui_callbacks(
        self,
        signal_callback: Callable[[GUISignal], None],
        price_callback: Callable[[str, float], None],
        log_callback: Callable[[str, str, str], None],
        trade_callback: Callable[[GUITrade], None],
        metrics_callback: Callable[[GUIMetrics], None],
        model_callback: Callable[[List[Dict]], None]
    ):
        """Set GUI callback functions."""
        self.signal_callback = signal_callback
        self.price_callback = price_callback
        self.log_callback = log_callback
        self.trade_callback = trade_callback
        self.metrics_callback = metrics_callback
        self.model_callback = model_callback
        
        logger.info("GUI callbacks registered")
    
    async def start_trading(self, symbol: str = "BTC/USDT", mode: str = "paper") -> bool:
        """Start trading system."""
        if not self.is_connected:
            await self.connect()
        
        if self.is_running:
            logger.warning("Trading system already running")
            return False
        
        try:
            self.current_symbol = symbol
            self.is_running = True
            
            # Start real-time tasks
            if self.demo_mode:
                await self._start_demo_trading()
            else:
                await self._start_real_trading()
            
            # Send initial log
            self._send_log("info", f"Trading system started - {symbol} - {mode}")
            
            # Send initial model data
            self._send_model_update()
            
            logger.info(f"Trading started for {symbol} in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            self.is_running = False
            return False
    
    async def _start_demo_trading(self):
        """Start demo trading with simulated data."""
        # Start signal generation task
        signal_task = asyncio.create_task(self._demo_signal_loop())
        self.tasks.append(signal_task)
        
        # Start price update task
        price_task = asyncio.create_task(self._demo_price_loop())
        self.tasks.append(price_task)
        
        # Start trade simulation task
        trade_task = asyncio.create_task(self._demo_trade_loop())
        self.tasks.append(trade_task)
    
    async def _start_real_trading(self):
        """Start real trading with backend systems."""
        # Start real-time data processing
        data_task = asyncio.create_task(self._real_data_loop())
        self.tasks.append(data_task)
        
        # Start signal generation
        signal_task = asyncio.create_task(self._real_signal_loop())
        self.tasks.append(signal_task)
        
        # Start trading execution
        trade_task = asyncio.create_task(self._real_trade_loop())
        self.tasks.append(trade_task)
    
    async def _demo_signal_loop(self):
        """Demo signal generation loop."""
        while self.is_running:
            try:
                # Generate random signal
                signals = ["BUY", "SELL", "HOLD"]
                weights = [0.3, 0.3, 0.4]  # Slightly bias towards HOLD
                signal = np.random.choice(signals, p=weights)
                confidence = np.random.beta(2, 2)  # Beta distribution for confidence
                predicted_change = np.random.normal(0, 0.5)  # Normal distribution for change
                
                # Create GUI signal
                gui_signal = GUISignal(
                    signal=signal,
                    confidence=confidence,
                    predicted_change=predicted_change,
                    timestamp=datetime.now(),
                    symbol=self.current_symbol,
                    model_name=self.active_model_name
                )
                
                # Send to GUI
                self._send_signal(gui_signal)
                
                # Wait for next signal
                await asyncio.sleep(np.random.uniform(2, 5))  # Random interval 2-5 seconds
                
            except Exception as e:
                logger.error(f"Error in demo signal loop: {e}")
                await asyncio.sleep(1)
    
    async def _demo_price_loop(self):
        """Demo price update loop."""
        while self.is_running:
            try:
                # Simulate price movement
                volatility = 0.001  # 0.1% per update
                price_change = np.random.normal(0, volatility)
                self.current_price *= (1 + price_change)
                
                # Send to GUI
                self._send_price(self.current_symbol, self.current_price)
                
                # Update market data
                await self._update_demo_market_data()
                
                # Wait for next update
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in demo price loop: {e}")
                await asyncio.sleep(1)
    
    async def _demo_trade_loop(self):
        """Demo trade execution loop."""
        trade_counter = 0
        
        while self.is_running:
            try:
                # Simulate random trades
                if np.random.random() < 0.1:  # 10% chance per check
                    trade_counter += 1
                    
                    # Create demo trade
                    trade_type = np.random.choice(["BUY", "SELL"])
                    entry_price = self.current_price * np.random.uniform(0.999, 1.001)
                    
                    gui_trade = GUITrade(
                        id=str(trade_counter),
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        pair=self.current_symbol,
                        type=trade_type,
                        entry_price=entry_price
                    )
                    
                    # Send to GUI
                    self._send_trade(gui_trade)
                    
                    # Simulate trade closure after random time
                    close_delay = np.random.uniform(5, 30)  # 5-30 seconds
                    asyncio.create_task(self._close_demo_trade(gui_trade, close_delay))
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in demo trade loop: {e}")
                await asyncio.sleep(2)
    
    async def _close_demo_trade(self, trade: GUITrade, delay: float):
        """Close demo trade after delay."""
        await asyncio.sleep(delay)
        
        if not self.is_running:
            return
        
        # Calculate exit price and PnL
        price_change = np.random.normal(0, 0.002)  # 0.2% change
        exit_price = trade.entry_price * (1 + price_change)
        
        if trade.type == "BUY":
            pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        # Update trade
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        
        # Send updated trade to GUI
        self._send_trade(trade)
        
        # Update metrics
        self._update_demo_metrics()
    
    async def _update_demo_market_data(self):
        """Update demo market data."""
        # Add new data point
        timestamp = datetime.now()
        
        market_data = MarketData(
            timestamp_ms=int(timestamp.timestamp() * 1000),
            symbol=self.current_symbol.replace("/", ""),
            type=MarketDataType.TRADE,
            exchange="binance",
            data={
                "price": self.current_price,
                "volume": np.random.uniform(0.1, 10.0),
                "side": np.random.choice(["buy", "sell"])
            }
        )
        
        self.market_data.append(market_data)
        
        # Keep only recent data
        if len(self.market_data) > 1000:
            self.market_data = self.market_data[-1000:]
        
        # Update features
        if len(self.market_data) >= 20:
            self.features = np.random.randn(len(self.market_data), 20)
    
    async def _real_data_loop(self):
        """Real data processing loop."""
        while self.is_running:
            try:
                # Load real market data
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=10)
                
                if self.data_loader:
                    data = await self.data_loader.read_data(
                        symbol=self.current_symbol.replace("/", ""),
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if data:
                        self.market_data = data
                        
                        # Generate features
                        if self.feature_builder:
                            self.features = await self._generate_features(data)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in real data loop: {e}")
                await asyncio.sleep(5)
    
    async def _real_signal_loop(self):
        """Real signal generation loop."""
        while self.is_running:
            try:
                if self.signal_generator and self.features is not None:
                    # Generate signal
                    signal = self.signal_generator.generate_signal(
                        features=self.features[-1:] if len(self.features) > 0 else self.features,
                        current_price=self.current_price,
                        timestamp_ms=int(datetime.now().timestamp() * 1000),
                        symbol=self.current_symbol.replace("/", "")
                    )
                    
                    if signal:
                        # Convert to GUI format
                        gui_signal = self._convert_backend_signal(signal)
                        self._send_signal(gui_signal)
                
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                logger.error(f"Error in real signal loop: {e}")
                await asyncio.sleep(3)
    
    async def _real_trade_loop(self):
        """Real trade execution loop."""
        while self.is_running:
            try:
                # Execute trades based on signals
                # This would integrate with PaperTrader
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in real trade loop: {e}")
                await asyncio.sleep(5)
    
    async def _generate_features(self, market_data: List[MarketData]) -> np.ndarray:
        """Generate features from market data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "timestamp_ms": md.timestamp_ms,
                    "symbol": md.symbol,
                    "type": md.type.value,
                    "exchange": md.exchange,
                    "price": md.data.get("price", 0),
                    "volume": md.data.get("volume", 0)
                }
                for md in market_data
            ])
            
            # Synchronize data
            synchronized = synchronize_marketdata(market_data)
            
            # Generate features (simplified)
            features = np.random.randn(len(market_data), 20)  # Placeholder
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return np.random.randn(1, 20)
    
    def _convert_backend_signal(self, backend_signal: Signal) -> GUISignal:
        """Convert backend Signal to GUI format."""
        action_map = {
            Action.BUY: "BUY",
            Action.SELL: "SELL",
            Action.HOLD: "HOLD"
        }
        
        return GUISignal(
            signal=action_map.get(backend_signal.action, "HOLD"),
            confidence=backend_signal.confidence,
            predicted_change=backend_signal.predicted_change * 100,  # Convert to percentage
            timestamp=datetime.fromtimestamp(backend_signal.timestamp_ms / 1000),
            symbol=backend_signal.symbol,
            model_name=self.active_model_name
        )
    
    def _send_signal(self, signal: GUISignal):
        """Send signal to GUI."""
        if self.signal_callback:
            try:
                self.signal_callback(signal)
            except Exception as e:
                logger.error(f"Error sending signal to GUI: {e}")
    
    def _send_price(self, symbol: str, price: float):
        """Send price update to GUI."""
        if self.price_callback:
            try:
                self.price_callback(symbol, price)
            except Exception as e:
                logger.error(f"Error sending price to GUI: {e}")
    
    def _send_log(self, level: str, message: str):
        """Send log to GUI."""
        if self.log_callback:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_callback(timestamp, level, message)
            except Exception as e:
                logger.error(f"Error sending log to GUI: {e}")
    
    def _send_trade(self, trade: GUITrade):
        """Send trade to GUI."""
        if self.trade_callback:
            try:
                self.trade_callback(trade)
            except Exception as e:
                logger.error(f"Error sending trade to GUI: {e}")
    
    def _send_metrics(self, metrics: GUIMetrics):
        """Send metrics to GUI."""
        if self.metrics_callback:
            try:
                self.metrics_callback(metrics)
            except Exception as e:
                logger.error(f"Error sending metrics to GUI: {e}")
    
    def _send_model_update(self):
        """Send model performance update to GUI."""
        if self.model_callback:
            try:
                model_data = []
                for name, model in self.models.items():
                    score = getattr(model, 'score', np.random.uniform(0.7, 0.85))
                    model_data.append({
                        "name": name,
                        "type": self._get_model_type(name),
                        "score": score,
                        "is_active": name == self.active_model_name
                    })
                
                self.model_callback(model_data)
            except Exception as e:
                logger.error(f"Error sending model update to GUI: {e}")
    
    def _get_model_type(self, model_name: str) -> str:
        """Get model type from name."""
        if "GRU" in model_name:
            return "GRU"
        elif "CNN" in model_name:
            return "CNN"
        elif "Boosting" in model_name:
            return "Boosting"
        else:
            return "Transformer"
    
    def _update_demo_metrics(self):
        """Update demo trading metrics."""
        # Calculate demo metrics
        demo_metrics = GUIMetrics(
            total_pnl=np.random.uniform(-5, 10),
            win_rate=np.random.uniform(0.4, 0.8),
            total_trades=np.random.randint(50, 200),
            max_drawdown=np.random.uniform(1, 8),
            sharpe_ratio=np.random.uniform(0.5, 2.5),
            profit_factor=np.random.uniform(1.1, 2.0)
        )
        
        self._send_metrics(demo_metrics)
    
    async def stop_trading(self):
        """Stop trading system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.tasks.clear()
        
        # Send log
        self._send_log("info", "Trading system stopped")
        
        logger.info("Trading system stopped")
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch active model."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        self.active_model_name = model_name
        
        # Update signal generator if available
        if self.signal_generator:
            regression_model = self.models[model_name]
            classification_model = self.models[model_name]
            self.signal_generator.set_models(regression_model, classification_model)
        
        # Send model update to GUI
        self._send_model_update()
        
        # Send log
        self._send_log("info", f"Switched to model: {model_name}")
        
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "connected": self.is_connected,
            "running": self.is_running,
            "demo_mode": self.demo_mode,
            "current_symbol": self.current_symbol,
            "current_price": self.current_price,
            "active_model": self.active_model_name,
            "market_data_count": len(self.market_data),
            "active_tasks": len([t for t in self.tasks if not t.done()])
        }
    
    async def disconnect(self):
        """Disconnect from backend systems."""
        await self.stop_trading()
        
        if self.data_loader:
            await self.data_loader.close()
        
        self.is_connected = False
        logger.info("Disconnected from backend systems")


# Global bridge instance
_bridge_instance: Optional[ITSBackendBridge] = None


def get_bridge() -> ITSBackendBridge:
    """Get global bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ITSBackendBridge()
    return _bridge_instance


def set_bridge(bridge: ITSBackendBridge):
    """Set global bridge instance."""
    global _bridge_instance
    _bridge_instance = bridge
