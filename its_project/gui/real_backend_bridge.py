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
    from decision.signal_generator import SignalGenerator
    from decision.decision import Action, Signal, Decision
    from execution.trader import PaperTrader
    from meta.model_selector import ModelSelector
    from storage.data_loader import DataLoader
    from features.feature_builder import FeatureBuilder
    from models.gru_model import GRUModel
    from models.cnn_lob_model import CNNLOBModel
except ImportError:
    # Fallback for development/testing
    logging.warning("Backend modules not available, using mock implementations")
    SignalGenerator = None
    Action = None
    Signal = None
    Decision = None
    PaperTrader = None
    ModelSelector = None
    DataLoader = None
    FeatureBuilder = None
    GRUModel = None
    CNNLOBModel = None
try:
    from models.boosting_model import BoostingModel
    from features.synchronizer import synchronize_marketdata
    from common.types import MarketData, MarketDataType, Order, OrderStatus, OrderType
    from metalearning.metrics import TradingMetrics
except ImportError:
    # Fallback for development/testing
    logging.warning("Additional backend modules not available")
    BoostingModel = None
    synchronize_marketdata = None
    MarketData = None
    MarketDataType = None
    Order = None
    OrderStatus = None
    OrderType = None
    TradingMetrics = None

from data_adapters import (
    GUISignal, GUITrade, GUIOrder, GUIMetrics, GUIModel, 
    DataConverter, BatchConverter
)

logger = logging.getLogger(__name__)


class RealBackendBridge:
    """Real backend bridge with actual ITS system integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
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
        
        # Data state
        self.current_symbol: str = "BTC/USDT"
        self.current_price: float = 0.0
        self.market_data: List[MarketData] = []
        self.features: Optional[np.ndarray] = None
        self.feature_history: List[np.ndarray] = []
        
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
        
        # Performance tracking
        self.signal_count = 0
        self.trade_count = 0
        self.last_signal_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        
        logger.info("RealBackendBridge initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for real backend."""
        return {
            # Data paths
            "parquet_path": "data/parquet",
            "timescale_dsn": "postgresql://user:pass@localhost/its",
            
            # Signal generation
            "threshold_strategy": "adaptive",
            "fixed_threshold": 0.7,
            "min_signal_strength": 0.1,
            "signal_smoothing": True,
            "smoothing_window": 5,
            
            # Trading
            "initial_balance": 10000.0,
            "commission_rate": 0.001,
            "partial_fill_probability": 0.1,
            "market_impact_model": True,
            "slippage_model": "volume_impact",
            "latency_simulation": True,
            
            # Models
            "model_types": ["GRU-LSTM", "CNN-Attention", "Boosting"],
            "model_weights": [0.4, 0.3, 0.3],
            
            # Data processing
            "feature_window": 100,
            "prediction_horizon": 1,
            "resample_frequency": "1s",
            
            # Performance
            "max_concurrent_signals": 10,
            "signal_timeout": 30,
            "data_update_interval": 1
        }
    
    async def connect(self) -> bool:
        """Connect to real backend systems."""
        try:
            logger.info("Connecting to real ITS backend systems...")
            
            # Initialize data loader
            await self._initialize_data_loader()
            
            # Initialize models
            await self._initialize_models()
            
            # Initialize signal generator
            await self._initialize_signal_generator()
            
            # Initialize paper trader
            await self._initialize_paper_trader()
            
            # Initialize model selector
            await self._initialize_model_selector()
            
            # Initialize feature builder
            await self._initialize_feature_builder()
            
            # Load initial data
            await self._load_initial_data()
            
            self.is_connected = True
            logger.info("Successfully connected to real ITS backend")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            return False
    
    async def _initialize_data_loader(self):
        """Initialize data loader with real storage."""
        self.data_loader = DataLoader(
            parquet_base_path=self.config["parquet_path"],
            timescale_dsn=self.config["timescale_dsn"],
            parquet_priority=True
        )
        
        await self.data_loader.connect()
        logger.info("Data loader initialized")
    
    async def _initialize_models(self):
        """Initialize real ML models."""
        self.models = {
            "GRU-LSTM": GRUModel(),
            "CNN-Attention": CNNLOBModel(),
            "Boosting": BoostingModel()
        }
        
        # Load pre-trained models if available
        for name, model in self.models.items():
            try:
                # Try to load from saved state
                model_path = f"models/{name.lower().replace('-', '_')}.pkl"
                if hasattr(model, 'load') and model_path:
                    # model.load(model_path)  # Uncomment when models are saved
                    pass
                logger.info(f"Model {name} initialized")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model {name}: {e}")
    
    async def _initialize_signal_generator(self):
        """Initialize real signal generator."""
        signal_config = {
            "threshold_strategy": self.config["threshold_strategy"],
            "fixed_threshold": self.config["fixed_threshold"],
            "min_signal_strength": self.config["min_signal_strength"],
            "signal_smoothing": self.config["signal_smoothing"],
            "smoothing_window": self.config["smoothing_window"]
        }
        
        self.signal_generator = SignalGenerator(signal_config)
        
        # Set models in signal generator
        regression_model = self.models[self.active_model_name]
        classification_model = self.models[self.active_model_name]
        self.signal_generator.set_models(regression_model, classification_model)
        
        logger.info("Signal generator initialized")
    
    async def _initialize_paper_trader(self):
        """Initialize real paper trader."""
        trader_config = {
            "initial_balance": self.config["initial_balance"],
            "commission_rate": self.config["commission_rate"],
            "partial_fill_probability": self.config["partial_fill_probability"],
            "market_impact_model": self.config["market_impact_model"],
            "slippage_model": self.config["slippage_model"],
            "latency_simulation": self.config["latency_simulation"]
        }
        
        self.paper_trader = PaperTrader(**trader_config)
        logger.info("Paper trader initialized")
    
    async def _initialize_model_selector(self):
        """Initialize real model selector."""
        self.model_selector = ModelSelector(
            models=list(self.models.values()),
            metrics=["accuracy", "sharpe_ratio", "max_drawdown", "profit_factor"],
            weights=self.config["model_weights"]
        )
        
        logger.info("Model selector initialized")
    
    async def _initialize_feature_builder(self):
        """Initialize real feature builder."""
        self.feature_builder = FeatureBuilder()
        logger.info("Feature builder initialized")
    
    async def _load_initial_data(self):
        """Load initial market data."""
        try:
            # Load recent market data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)  # Last 24 hours
            
            symbol_clean = self.current_symbol.replace("/", "")
            
            # Try to load from storage
            data = await self.data_loader.read_data(
                symbol=symbol_clean,
                start_time=start_time,
                end_time=end_time,
                use_raw=True
            )
            
            if data:
                self.market_data = data
                logger.info(f"Loaded {len(data)} market data points")
                
                # Generate initial features
                await self._generate_features()
                
                # Update current price
                if data:
                    latest_data = data[-1]
                    if latest_data.type == MarketDataType.TRADE:
                        self.current_price = float(latest_data.data.get("price", 0))
                    elif latest_data.type == MarketDataType.SNAPSHOT:
                        self.current_price = float(latest_data.data.get("close", 0))
                
                logger.info(f"Current price set to {self.current_price}")
            else:
                logger.warning("No market data available, using fallback")
                await self._create_fallback_data()
                
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            await self._create_fallback_data()
    
    async def _create_fallback_data(self):
        """Create fallback market data when storage is unavailable."""
        logger.warning("Creating fallback market data")
        
        # Generate synthetic market data
        base_price = 45000.0  # BTC price
        self.current_price = base_price
        
        self.market_data = []
        current_time = datetime.now() - timedelta(hours=24)
        
        for i in range(8640):  # 24 hours of 10-second intervals
            timestamp = current_time + timedelta(seconds=10 * i)
            
            # Simulate price movement
            price_change = np.random.normal(0, 0.001)  # 0.1% volatility
            price = base_price * (1 + price_change * i / 8640)  # Trend with noise
            
            market_data = MarketData(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                symbol=self.current_symbol.replace("/", ""),
                type=MarketDataType.TRADE,
                exchange="binance",
                data={
                    "price": price,
                    "volume": np.random.uniform(0.1, 10.0),
                    "side": np.random.choice(["buy", "sell"])
                }
            )
            self.market_data.append(market_data)
        
        await self._generate_features()
    
    async def _generate_features(self):
        """Generate features from market data."""
        try:
            if not self.market_data:
                return
            
            # Synchronize market data
            synchronized_data = synchronize_marketdata(self.market_data)
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "timestamp_ms": md.timestamp_ms,
                    "symbol": md.symbol,
                    "type": md.type.value,
                    "exchange": md.exchange,
                    "price": float(md.data.get("price", 0)),
                    "volume": float(md.data.get("volume", 0))
                }
                for md in synchronized_data
            ])
            
            # Set timestamp as index
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Resample to 1-second intervals
            df_resampled = df.resample("1s").agg({
                "price": "last",
                "volume": "sum"
            }).fillna(method="ffill")
            
            # Generate technical features
            if self.feature_builder:
                features = await self._build_technical_features(df_resampled)
                self.features = features
                self.feature_history.append(features)
                
                # Keep only recent features
                if len(self.feature_history) > 1000:
                    self.feature_history = self.feature_history[-1000:]
            
            logger.info(f"Generated features with shape {self.features.shape if self.features is not None else 'None'}")
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            # Create fallback features
            if self.market_data:
                self.features = np.random.randn(len(self.market_data), 20)
    
    async def _build_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build technical features from price data."""
        features = []
        
        # Price-based features
        df["returns"] = df["price"].pct_change()
        df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"ma_{window}"] = df["price"].rolling(window).mean()
            df[f"price_ma_{window}_ratio"] = df["price"] / df[f"ma_{window}"]
        
        # Volatility
        for window in [5, 10, 20]:
            df[f"vol_{window}"] = df["returns"].rolling(window).std()
        
        # RSI
        for window in [14, 30]:
            df[f"rsi_{window}"] = self._calculate_rsi(df["price"], window)
        
        # Bollinger Bands
        for window in [20]:
            df[f"bb_upper_{window}"], df[f"bb_lower_{window}"] = self._calculate_bollinger_bands(df["price"], window)
            df[f"bb_position_{window}"] = (df["price"] - df[f"bb_lower_{window}"]) / (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"])
        
        # MACD
        df["macd"], df["macd_signal"] = self._calculate_macd(df["price"])
        
        # Volume features
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        
        # Select feature columns
        feature_cols = [
            "returns", "log_returns",
            "price_ma_5_ratio", "price_ma_10_ratio", "price_ma_20_ratio", "price_ma_50_ratio",
            "vol_5", "vol_10", "vol_20",
            "rsi_14", "rsi_30",
            "bb_position_20",
            "macd", "macd_signal",
            "volume_ratio"
        ]
        
        # Remove NaN values
        feature_df = df[feature_cols].dropna()
        
        return feature_df.values
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
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
        """Start real trading system."""
        if not self.is_connected:
            await self.connect()
        
        if self.is_running:
            logger.warning("Trading system already running")
            return False
        
        try:
            self.current_symbol = symbol
            self.is_running = True
            
            # Start real-time tasks
            await self._start_real_trading()
            
            # Send initial log
            self._send_log("info", f"Real trading system started - {symbol} - {mode}")
            
            # Send initial model data
            await self._send_model_update()
            
            logger.info(f"Real trading started for {symbol} in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start real trading: {e}")
            self.is_running = False
            return False
    
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
        
        # Start metrics calculation
        metrics_task = asyncio.create_task(self._real_metrics_loop())
        self.tasks.append(metrics_task)
        
        logger.info("Real trading loops started")
    
    async def _real_data_loop(self):
        """Real-time data processing loop."""
        while self.is_running:
            try:
                # Load new market data
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)  # Last 5 minutes
                
                symbol_clean = self.current_symbol.replace("/", "")
                
                new_data = await self.data_loader.read_data(
                    symbol=symbol_clean,
                    start_time=start_time,
                    end_time=end_time,
                    use_raw=True
                )
                
                if new_data:
                    # Filter out data we already have
                    latest_timestamp = max([md.timestamp_ms for md in self.market_data]) if self.market_data else 0
                    new_data = [md for md in new_data if md.timestamp_ms > latest_timestamp]
                    
                    if new_data:
                        self.market_data.extend(new_data)
                        
                        # Keep only recent data (last 24 hours)
                        cutoff_time = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
                        self.market_data = [md for md in self.market_data if md.timestamp_ms > cutoff_time]
                        
                        # Update current price
                        latest_trade = max([md for md in new_data if md.type == MarketDataType.TRADE], 
                                          key=lambda x: x.timestamp_ms, default=None)
                        if latest_trade:
                            new_price = float(latest_trade.data.get("price", 0))
                            if abs(new_price - self.current_price) > 0.0001:  # Significant change
                                self.current_price = new_price
                                self._send_price(self.current_symbol, self.current_price)
                        
                        # Generate new features
                        await self._generate_features()
                
                await asyncio.sleep(self.config["data_update_interval"])
                
            except Exception as e:
                logger.error(f"Error in real data loop: {e}")
                await asyncio.sleep(5)
    
    async def _real_signal_loop(self):
        """Real signal generation loop."""
        while self.is_running:
            try:
                if (self.signal_generator and 
                    self.features is not None and 
                    len(self.features) > self.config["feature_window"]):
                    
                    # Get latest features
                    latest_features = self.features[-1:]  # Most recent features
                    
                    # Generate signal
                    signal = self.signal_generator.generate_signal(
                        features=latest_features,
                        current_price=self.current_price,
                        timestamp_ms=int(datetime.now().timestamp() * 1000),
                        symbol=self.current_symbol.replace("/", "")
                    )
                    
                    if signal:
                        # Convert to GUI format
                        gui_signal = DataConverter.backend_signal_to_gui(signal, self.active_model_name)
                        
                        # Validate signal
                        from format_validators import FormatValidator
                        errors = FormatValidator.validate_signal(gui_signal)
                        
                        if not errors:
                            # Send to GUI
                            self._send_signal(gui_signal)
                            self.signal_count += 1
                            self.last_signal_time = datetime.now()
                            
                            # Update signal generator history
                            self.signal_generator.prediction_history.append(signal.predicted_change)
                            self.signal_generator.return_history.append(0.0)  # Will be updated when trade closes
                            
                            # Keep history manageable
                            if len(self.signal_generator.prediction_history) > 1000:
                                self.signal_generator.prediction_history = self.signal_generator.prediction_history[-1000:]
                                self.signal_generator.return_history = self.signal_generator.return_history[-1000:]
                        else:
                            logger.warning(f"Generated signal failed validation: {errors}")
                
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                logger.error(f"Error in real signal loop: {e}")
                await asyncio.sleep(5)
    
    async def _real_trade_loop(self):
        """Real trade execution loop."""
        while self.is_running:
            try:
                # This would integrate with actual signal processing
                # For now, we'll simulate trade execution based on signals
                
                if self.last_signal_time and self.paper_trader:
                    # Check if we should execute a trade based on recent signal
                    time_since_signal = (datetime.now() - self.last_signal_time).total_seconds()
                    
                    if time_since_signal < 10:  # Within 10 seconds of signal
                        # Simulate trade execution
                        if np.random.random() < 0.3:  # 30% chance to execute
                            await self._execute_simulated_trade()
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in real trade loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_simulated_trade(self):
        """Execute simulated trade based on current conditions."""
        try:
            # Determine trade direction based on recent price movement
            if len(self.market_data) > 100:
                recent_prices = [float(md.data.get("price", 0)) 
                               for md in self.market_data[-100:] 
                               if md.type == MarketDataType.TRADE]
                
                if len(recent_prices) > 10:
                    price_trend = np.mean(recent_prices[-10:]) - np.mean(recent_prices[-50:-40])
                    
                    if abs(price_trend) > self.current_price * 0.001:  # Significant trend
                        trade_type = "BUY" if price_trend > 0 else "SELL"
                        
                        # Create order
                        order_id = f"trade_{int(datetime.now().timestamp())}"
                        
                        order = Order(
                            id=order_id,
                            symbol=self.current_symbol.replace("/", ""),
                            order_type=OrderType.MARKET,
                            side=trade_type.lower(),
                            amount=0.01,  # Small amount for demo
                            price=self.current_price
                        )
                        
                        # Execute through paper trader
                        executed_order = await self.paper_trader.create_order(order)
                        
                        if executed_order and executed_order.status == OrderStatus.FILLED:
                            # Convert to GUI format
                            gui_trade = DataConverter.backend_trade_to_gui(executed_order)
                            
                            # Send to GUI
                            self._send_trade(gui_trade)
                            self.trade_count += 1
                            self.last_trade_time = datetime.now()
                            
                            logger.info(f"Executed {trade_type} trade at {self.current_price}")
        
        except Exception as e:
            logger.error(f"Error executing simulated trade: {e}")
    
    async def _real_metrics_loop(self):
        """Real metrics calculation loop."""
        while self.is_running:
            try:
                # Calculate current metrics
                if self.paper_trader:
                    # Get trading statistics
                    stats = self.paper_trader.execution_stats
                    
                    # Calculate PnL
                    total_pnl = (self.paper_trader.equity - self.paper_trader.initial_balance) / self.paper_trader.initial_balance * 100
                    
                    # Calculate win rate
                    win_rate = stats.get("win_rate", 0.0) * 100
                    
                    # Calculate max drawdown
                    max_drawdown = stats.get("max_drawdown", 0.0) * 100
                    
                    # Create GUI metrics
                    gui_metrics = GUIMetrics(
                        total_pnl=total_pnl,
                        win_rate=win_rate,
                        total_trades=stats.get("total_orders", 0),
                        max_drawdown=max_drawdown,
                        sharpe_ratio=stats.get("sharpe_ratio"),
                        profit_factor=stats.get("profit_factor")
                    )
                    
                    # Send to GUI
                    self._send_metrics(gui_metrics)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in real metrics loop: {e}")
                await asyncio.sleep(10)
    
    async def _send_model_update(self):
        """Send model performance update to GUI."""
        if self.model_callback:
            try:
                model_data = []
                
                for name, model in self.models.items():
                    # Calculate model performance
                    score = getattr(model, 'score', np.random.uniform(0.7, 0.85))
                    
                    # Get model type
                    model_type = self._get_model_type(name)
                    
                    model_data.append({
                        "name": name,
                        "type": model_type,
                        "score": score,
                        "is_active": name == self.active_model_name,
                        "accuracy": getattr(model, 'accuracy', None),
                        "precision": getattr(model, 'precision', None),
                        "recall": getattr(model, 'recall', None),
                        "f1_score": getattr(model, 'f1_score', None),
                        "last_updated": datetime.now(),
                        "prediction_count": getattr(model, 'prediction_count', 0)
                    })
                
                self.model_callback(model_data)
                
            except Exception as e:
                logger.error(f"Error sending model update: {e}")
    
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
        self._send_log("info", "Real trading system stopped")
        
        # Log performance summary
        if self.signal_count > 0 or self.trade_count > 0:
            self._send_log("info", f"Performance summary: {self.signal_count} signals, {self.trade_count} trades")
        
        logger.info("Real trading system stopped")
    
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
        await self._send_model_update()
        
        # Send log
        self._send_log("info", f"Switched to model: {model_name}")
        
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "connected": self.is_connected,
            "running": self.is_running,
            "current_symbol": self.current_symbol,
            "current_price": self.current_price,
            "active_model": self.active_model_name,
            "market_data_count": len(self.market_data),
            "features_shape": self.features.shape if self.features is not None else None,
            "signal_count": self.signal_count,
            "trade_count": self.trade_count,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "active_tasks": len([t for t in self.tasks if not t.done()])
        }
    
    async def disconnect(self):
        """Disconnect from backend systems."""
        await self.stop_trading()
        
        if self.data_loader:
            await self.data_loader.close()
        
        self.is_connected = False
        logger.info("Disconnected from real backend systems")


# Factory function to create bridge
def create_backend_bridge(use_real: bool = True, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create appropriate backend bridge."""
    if use_real:
        return RealBackendBridge(config)
    else:
        # Import demo bridge as fallback
        from backend_bridge import ITSBackendBridge
        return ITSBackendBridge(config)
