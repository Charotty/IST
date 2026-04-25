"""
Backend Service for OKX WS + CCXT Integration
=============================================

In-process backend service with HTTP (commands) + WebSocket (events).
Runs on localhost:5050.

This service is designed to run inside the GUI process but with explicit
interfaces to enable future migration to separate process.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

from common.backend_contract import (
    # Events
    PriceUpdateEvent,
    OrderbookUpdateEvent,
    CandleUpdateEvent,
    SignalUpdateEvent,
    PaperTradeUpdateEvent,
    MetricsUpdateEvent,
    LogEvent,
    SystemStatusEvent,
    # Commands
    ConnectCommand,
    DisconnectCommand,
    StartIngestionCommand,
    StopIngestionCommand,
    StartTradingCommand,
    StopTradingCommand,
    SetModeCommand,
    UpdateSubscriptionsCommand,
    PaperResetCommand,
    # Responses
    CommandResponse,
    HealthResponse,
    StatusResponse,
    # Enums
    ConnectionStatus,
    TradingMode,
    ChannelType,
    # Utils
    serialize_event,
    validate_symbol_gui_format,
    validate_channel,
    validate_trading_mode,
)
from dataclasses import asdict

from .okx_ws_client import OKXWSClient
from .okx_rest_client import OKXRESTClient
from .paper_trading import PaperTradingEngine, PaperPortfolio

logger = logging.getLogger(__name__)


class BackendServiceState:
    """Backend service state."""
    
    def __init__(self):
        # Connection status
        self.connected_okx_ws = ConnectionStatus.DISCONNECTED
        self.connected_okx_rest = ConnectionStatus.DISCONNECTED
        
        # Running state
        self.running_ingestion = False
        self.running_trading = False
        self.trading_mode = TradingMode.PAPER
        
        # Subscriptions
        self.active_symbols: List[str] = []
        self.active_channels: List[str] = []
        self.orderbook_depth = 20
        self.candle_timeframes: List[str] = []
        
        # Queue sizes (simulated for now)
        self.queues: Dict[str, int] = {
            "price_updates": 0,
            "orderbook_updates": 0,
            "candle_updates": 0,
            "signals": 0,
            "trades": 0,
        }
        
        # Heartbeat
        self.last_heartbeat = int(time.time() * 1000)
        self.start_time = time.time()
        
        # Paper trading state
        self.paper_balance = 10000.0
        self.paper_positions: Dict[str, float] = {}
        self.paper_trades: List[Dict] = []
        
        # Historical data storage
        self.historical_data: Optional[Dict] = None
        
        # Features data storage
        self.features_data: Optional[Dict] = None
        
        # Trained model storage
        self.trained_model: Optional[Dict] = None
        
        # Deployment config storage
        self.deployment_config: Optional[Dict] = None
        
        # OKX clients
        self.okx_ws_client = OKXWSClient()
        self.okx_rest_client = OKXRESTClient()
        
        # Paper trading engine
        self.paper_portfolio = PaperPortfolio()
        self.paper_engine = PaperTradingEngine(self.paper_portfolio)
        


class BackendService:
    """Backend service with HTTP + WebSocket interfaces."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5050):
        self.host = host
        self.port = port
        self.state = BackendServiceState()
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'its-backend-secret-key'
        self.app.config['DEBUG'] = False
        
        # SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Background thread for heartbeat
        self.heartbeat_thread = None
        self.heartbeat_running = False
        
        # Wire up WS client callbacks
        self.state.okx_ws_client.on_price(self._on_price_update)
        self.state.okx_ws_client.on_orderbook(self._on_orderbook_update)
        self.state.okx_ws_client.on_candle(self._on_candle_update)
        
        # Wire up paper trading callbacks
        self.state.paper_engine.on_trade(self._on_paper_trade)
        
        # Setup routes
        self._setup_http_routes()
        self._setup_ws_events()
        
        logger.info(f"BackendService initialized on {host}:{port}")
    
    def _setup_http_routes(self):
        """Setup HTTP command routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            uptime = int(time.time() - self.state.start_time)
            
            response = HealthResponse(
                healthy=True,
                version="1.0.0",
                uptime_seconds=uptime,
                components={
                    "okx_ws": self.state.connected_okx_ws.value,
                    "okx_rest": self.state.connected_okx_rest.value,
                    "ingestion": "running" if self.state.running_ingestion else "stopped",
                    "trading": "running" if self.state.running_trading else "stopped",
                },
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get detailed status."""
            response = StatusResponse(
                connected_okx_ws=self.state.connected_okx_ws,
                connected_okx_rest=self.state.connected_okx_rest,
                running_ingestion=self.state.running_ingestion,
                running_trading=self.state.running_trading,
                trading_mode=self.state.trading_mode,
                active_symbols=self.state.active_symbols.copy(),
                active_channels=self.state.active_channels.copy(),
                queues=self.state.queues.copy(),
                last_heartbeat=self.state.last_heartbeat,
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/')
        def serve_index():
            """Serve the web interface index page."""
            try:
                return send_from_directory('../web_interface', 'index.html')
            except Exception as e:
                logger.error(f"Error serving index: {e}")
                return jsonify({'error': 'Web interface not found'}), 404
        
        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """Serve static files for web interface."""
            try:
                return send_from_directory('../web_interface', filename)
            except Exception as e:
                logger.error(f"Error serving static file {filename}: {e}")
                return jsonify({'error': 'File not found'}), 404
        
        @self.app.route('/api/connect', methods=['POST'])
        def connect():
            """Connect to backend."""
            self.state.connected_okx_ws = ConnectionStatus.CONNECTING
            self.state.connected_okx_rest = ConnectionStatus.CONNECTING
            
            self._emit_log("info", "Connecting to OKX WebSocket...", "system")
            self._emit_system_status()
            
            # Connect to OKX WebSocket
            import asyncio
            
            def connect_task():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def do_connect():
                    try:
                        success = await self.state.okx_ws_client.connect()
                        if success:
                            self.state.connected_okx_ws = ConnectionStatus.CONNECTED
                            self._emit_log("info", "Connected to OKX WebSocket", "okx_ws")
                            self._emit_system_status()
                        else:
                            self.state.connected_okx_ws = ConnectionStatus.ERROR
                            self._emit_log("error", "Failed to connect to OKX WebSocket", "okx_ws")
                            self._emit_system_status()
                    except Exception as e:
                        logger.error(f"OKX WS connection error: {e}")
                        self.state.connected_okx_ws = ConnectionStatus.ERROR
                        self._emit_log("error", f"OKX WS connection error: {e}", "okx_ws")
                        self._emit_system_status()
                
                loop.run_until_complete(do_connect())
            
            connect_thread = threading.Thread(target=connect_task, daemon=True)
            connect_thread.start()
            
            # For now, assume REST is connected (public endpoints don't need auth)
            self.state.connected_okx_rest = ConnectionStatus.CONNECTED
            
            response = CommandResponse(
                success=True,
                message="Connection initiated",
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/disconnect', methods=['POST'])
        def disconnect():
            """Disconnect from backend."""
            # Stop ingestion if running
            if self.state.running_ingestion:
                self._stop_ingestion()
            
            # Stop trading if running
            if self.state.running_trading:
                self._stop_trading()
            
            self.state.connected_okx_ws = ConnectionStatus.DISCONNECTED
            self.state.connected_okx_rest = ConnectionStatus.DISCONNECTED
            self.state.active_symbols.clear()
            self.state.active_channels.clear()
            
            self._emit_log("info", "Disconnected from OKX", "system")
            
            response = CommandResponse(
                success=True,
                message="Disconnected from OKX",
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/ingestion/start', methods=['POST'])
        def start_ingestion():
            """Start data ingestion."""
            data = request.get_json() or {}
            
            symbols = data.get('symbols', [])
            channels = data.get('channels', [])
            depth = data.get('depth', 20)
            candle_timeframes = data.get('candle_timeframes', [])
            
            # Validate
            if not symbols:
                return jsonify(CommandResponse(
                    success=False,
                    message="No symbols provided",
                    ts=int(time.time() * 1000)
                ))
            
            for symbol in symbols:
                if not validate_symbol_gui_format(symbol):
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Invalid symbol format: {symbol}",
                        ts=int(time.time() * 1000)
                    ))
            
            for channel in channels:
                if not validate_channel(channel):
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Invalid channel: {channel}",
                        ts=int(time.time() * 1000)
                    ))
            
            # Update state
            self.state.active_symbols = symbols
            self.state.active_channels = channels
            self.state.orderbook_depth = depth
            self.state.candle_timeframes = candle_timeframes
            
            # Start ingestion
            self._start_ingestion()
            
            # Subscribe to channels via WS client (async)
            import asyncio
            
            def subscribe_task():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def do_subscriptions():
                    try:
                        if 'tickers' in channels:
                            await self.state.okx_ws_client.subscribe_tickers(symbols)
                        if 'orderbook_l2' in channels:
                            await self.state.okx_ws_client.subscribe_orderbook(symbols, depth)
                        if 'candles' in channels and candle_timeframes:
                            await self.state.okx_ws_client.subscribe_candles(symbols, candle_timeframes)
                    except Exception as e:
                        logger.error(f"Subscription error: {e}")
                
                loop.run_until_complete(do_subscriptions())
            
            sub_thread = threading.Thread(target=subscribe_task, daemon=True)
            sub_thread.start()
            
            self._emit_log("info", f"Started ingestion for {len(symbols)} symbols", "ingestion")
            
            response = CommandResponse(
                success=True,
                message=f"Ingestion started for {len(symbols)} symbols",
                data={
                    "symbols": symbols,
                    "channels": channels,
                    "depth": depth,
                    "candle_timeframes": candle_timeframes
                },
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/ingestion/stop', methods=['POST'])
        def stop_ingestion():
            """Stop data ingestion."""
            self._stop_ingestion()
            
            self._emit_log("info", "Stopped ingestion", "ingestion")
            
            response = CommandResponse(
                success=True,
                message="Ingestion stopped",
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/data/load', methods=['POST'])
        def load_historical_data():
            """Load historical data from OKX REST API."""
            import asyncio
            from datetime import datetime
            
            data = request.get_json() or {}
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '1h')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            data_sources = data.get('data_sources', ['candles'])
            
            # Validate
            if not symbol:
                return jsonify(CommandResponse(
                    success=False,
                    message="Symbol is required",
                    ts=int(time.time() * 1000)
                ))
            
            if not start_date or not end_date:
                return jsonify(CommandResponse(
                    success=False,
                    message="Start date and end date are required",
                    ts=int(time.time() * 1000)
                ))
            
            try:
                # Convert dates to timestamps
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            except ValueError as e:
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Invalid date format: {e}",
                    ts=int(time.time() * 1000)
                ))
            
            self._emit_log("info", f"Loading historical data for {symbol} ({timeframe}) from {start_date} to {end_date}", "data")
            
            # Load data in thread
            def load_data_task():
                try:
                    # Connect to OKX REST if not connected
                    if not self.state.okx_rest_client.connected:
                        self.state.okx_rest_client.connect()
                    
                    # Load candles
                    candles_data = []
                    if 'candles' in data_sources:
                        current_ts = start_ts
                        while current_ts < end_ts:
                            candles = self.state.okx_rest_client.get_historical_ohlcv(
                                symbol, timeframe, since=current_ts, limit=1000
                            )
                            if not candles:
                                break
                            candles_data.extend(candles)
                            current_ts = candles[-1][0] + 1  # Move to next candle
                            logger.info(f"Loaded {len(candles)} candles, total: {len(candles_data)}")
                    
                    # Store data in state
                    self.state.historical_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_date': start_date,
                        'end_date': end_date,
                        'candles': candles_data,
                        'data_sources': data_sources
                    }
                    
                    self._emit_log("info", f"Loaded {len(candles_data)} candles for {symbol}", "data")
                    
                    return {
                        'success': True,
                        'candle_count': len(candles_data),
                        'data_sources': data_sources
                    }
                except Exception as e:
                    logger.error(f"Error loading historical data: {e}")
                    self._emit_log("error", f"Error loading historical data: {e}", "data")
                    return {
                        'success': False,
                        'error': str(e)
                    }
            
            # Run in thread
            import threading
            load_thread = threading.Thread(target=load_data_task, daemon=True)
            load_thread.start()
            
            # Wait for completion (simplified for now)
            load_thread.join(timeout=60)
            
            if not self.state.historical_data:
                return jsonify(CommandResponse(
                    success=False,
                    message="Failed to load historical data",
                    ts=int(time.time() * 1000)
                ))
            
            response = CommandResponse(
                success=True,
                message=f"Loaded {len(self.state.historical_data.get('candles', []))} candles",
                data={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'candle_count': len(self.state.historical_data.get('candles', [])),
                    'data_sources': data_sources
                },
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/features/calculate', methods=['POST'])
        def calculate_features():
            """Calculate features from loaded historical data."""
            import pandas as pd
            import numpy as np
            
            data = request.get_json() or {}
            technical_indicators = data.get('technical_indicators', [])
            orderbook_features = data.get('orderbook_features', [])
            
            # Check if data is loaded
            if not self.state.historical_data:
                return jsonify(CommandResponse(
                    success=False,
                    message="No historical data loaded. Please load data first.",
                    ts=int(time.time() * 1000)
                ))
            
            candles = self.state.historical_data.get('candles', [])
            if not candles:
                return jsonify(CommandResponse(
                    success=False,
                    message="No candles in loaded data",
                    ts=int(time.time() * 1000)
                ))
            
            self._emit_log("info", f"Calculating features: {technical_indicators}, {orderbook_features}", "features")
            
            try:
                # Convert candles to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate technical indicators
                features = {}
                
                if 'sma' in technical_indicators:
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    features['sma'] = ['sma_20', 'sma_50']
                
                if 'ema' in technical_indicators:
                    df['ema_12'] = df['close'].ewm(span=12).mean()
                    df['ema_26'] = df['close'].ewm(span=26).mean()
                    features['ema'] = ['ema_12', 'ema_26']
                
                if 'rsi' in technical_indicators:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    features['rsi'] = ['rsi']
                
                if 'macd' in technical_indicators:
                    df['ema_12'] = df['close'].ewm(span=12).mean()
                    df['ema_26'] = df['close'].ewm(span=26).mean()
                    df['macd'] = df['ema_12'] - df['ema_26']
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    features['macd'] = ['macd', 'macd_signal']
                
                if 'bollinger' in technical_indicators:
                    df['bb_middle'] = df['close'].rolling(window=20).mean()
                    df['bb_std'] = df['close'].rolling(window=20).std()
                    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
                    features['bollinger'] = ['bb_upper', 'bb_middle', 'bb_lower']
                
                # Store features in state
                self.state.features_data = {
                    'df': df,
                    'features': features,
                    'technical_indicators': technical_indicators,
                    'orderbook_features': orderbook_features
                }
                
                self._emit_log("info", f"Calculated {len(features)} feature groups", "features")
                
                response = CommandResponse(
                    success=True,
                    message=f"Calculated {len(features)} feature groups",
                    data={
                        'feature_groups': list(features.keys()),
                        'total_features': sum(len(v) for v in features.values())
                    },
                    ts=int(time.time() * 1000)
                )
                
                return jsonify(asdict(response))
            except Exception as e:
                logger.error(f"Error calculating features: {e}")
                self._emit_log("error", f"Error calculating features: {e}", "features")
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Error calculating features: {e}",
                    ts=int(time.time() * 1000)
                ))
        
        @self.app.route('/api/models/train', methods=['POST'])
        def train_model():
            """Train a model on prepared features."""
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            data = request.get_json() or {}
            model_type = data.get('model_type', 'lstm')
            sequence_length = data.get('sequence_length', 60)
            hidden_layers = data.get('hidden_layers', 2)
            dropout = data.get('dropout', 0.2)
            learning_rate = data.get('learning_rate', 0.001)
            epochs = data.get('epochs', 50)
            batch_size = data.get('batch_size', 32)
            train_test_split_ratio = data.get('train_test_split', 0.8)
            
            # Check if features are calculated
            if not self.state.features_data:
                return jsonify(CommandResponse(
                    success=False,
                    message="No features calculated. Please calculate features first.",
                    ts=int(time.time() * 1000)
                ))
            
            df = self.state.features_data.get('df')
            if df is None or df.empty:
                return jsonify(CommandResponse(
                    success=False,
                    message="No feature data available",
                    ts=int(time.time() * 1000)
                ))
            
            self._emit_log("info", f"Training {model_type} model with {epochs} epochs", "training")
            
            try:
                # Prepare data for training
                # Drop NaN values (from rolling windows)
                df_clean = df.dropna()
                
                if len(df_clean) < sequence_length:
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Not enough data: {len(df_clean)} rows, need at least {sequence_length}",
                        ts=int(time.time() * 1000)
                    ))
                
                # Select feature columns
                feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                if not feature_cols:
                    return jsonify(CommandResponse(
                        success=False,
                        message="No feature columns found",
                        ts=int(time.time() * 1000)
                    ))
                
                X = df_clean[feature_cols].values
                y = (df_clean['close'].shift(-1) > df_clean['close']).astype(int).values[:-1]  # Binary classification: price up or down
                X = X[:-1]  # Align with y
                
                # Import real models
                from its_project.models.lstm import LSTMModel
                from its_project.models.gru_model import GRUModel
                from its_project.models.transformer import TransformerModel
                
                # Prepare sequences for neural networks
                def create_sequences(X, y, seq_len):
                    X_seq, y_seq = [], []
                    for i in range(len(X) - seq_len):
                        X_seq.append(X[i:i+seq_len])
                        y_seq.append(y[i+seq_len])
                    return np.array(X_seq), np.array(y_seq)
                
                X_seq, y_seq = create_sequences(X, y, sequence_length)
                
                if len(X_seq) < 100:
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Not enough sequences after windowing: {len(X_seq)}, need at least 100",
                        ts=int(time.time() * 1000)
                    ))
                
                # Train/test split
                split_idx = int(len(X_seq) * train_test_split_ratio)
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                
                # Select model based on type
                if model_type == 'lstm':
                    model = LSTMModel({
                        'input_size': X_train.shape[2],
                        'hidden_size': hidden_layers * 64,
                        'num_layers': hidden_layers,
                        'num_classes': 2,
                        'dropout': dropout,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size
                    })
                elif model_type == 'gru':
                    model = GRUModel({
                        'input_size': X_train.shape[2],
                        'hidden_size': hidden_layers * 64,
                        'num_layers': hidden_layers,
                        'num_classes': 2,
                        'dropout': dropout,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size
                    })
                elif model_type == 'transformer':
                    model = TransformerModel({
                        'input_size': X_train.shape[2],
                        'd_model': hidden_layers * 64,
                        'nhead': 8,
                        'num_layers': hidden_layers,
                        'num_classes': 2,
                        'dropout': dropout,
                        'max_seq_len': sequence_length,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size
                    })
                elif model_type == 'ensemble':
                    # Use LSTM as base for ensemble (simplified)
                    model = LSTMModel({
                        'input_size': X_train.shape[2],
                        'hidden_size': hidden_layers * 64,
                        'num_layers': hidden_layers,
                        'num_classes': 2,
                        'dropout': dropout,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size
                    })
                else:
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Unknown model type: {model_type}",
                        ts=int(time.time() * 1000)
                    ))
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                
                # Store model in state
                self.state.trained_model = {
                    'model': model,
                    'model_type': model_type,
                    'feature_cols': feature_cols,
                    'sequence_length': sequence_length,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'trained_at': int(time.time() * 1000)
                }
                
                self._emit_log("info", f"Model trained: accuracy={accuracy:.3f}, f1={f1:.3f}", "training")
                
                response = CommandResponse(
                    success=True,
                    message=f"Model trained successfully",
                    data={
                        'model_type': model_type,
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    },
                    ts=int(time.time() * 1000)
                )
                
                return jsonify(asdict(response))
            except Exception as e:
                logger.error(f"Error training model: {e}")
                self._emit_log("error", f"Error training model: {e}", "training")
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Error training model: {e}",
                    ts=int(time.time() * 1000)
                ))
        
        @self.app.route('/api/backtest/run', methods=['POST'])
        def run_backtest():
            """Run backtesting on trained model."""
            import pandas as pd
            import numpy as np
            
            data = request.get_json() or {}
            entry_threshold = data.get('entry_threshold', 0.7)
            exit_threshold = data.get('exit_threshold', 0.5)
            stop_loss = data.get('stop_loss', 2.0)
            take_profit = data.get('take_profit', 4.0)
            position_size = data.get('position_size', 10.0)
            
            # Check if model is trained
            if not self.state.trained_model:
                return jsonify(CommandResponse(
                    success=False,
                    message="No trained model. Please train a model first.",
                    ts=int(time.time() * 1000)
                ))
            
            if not self.state.features_data:
                return jsonify(CommandResponse(
                    success=False,
                    message="No feature data available",
                    ts=int(time.time() * 1000)
                ))
            
            self._emit_log("info", f"Running backtest with entry={entry_threshold}, exit={exit_threshold}", "backtest")
            
            try:
                model = self.state.trained_model['model']
                feature_cols = self.state.trained_model['feature_cols']
                sequence_length = self.state.trained_model.get('sequence_length', 60)
                df = self.state.features_data['df'].copy()
                
                # Prepare data
                df_clean = df.dropna()
                X = df_clean[feature_cols].values
                prices = df_clean['close'].values
                
                # Check if model is sequence-based (neural network) or sklearn
                model_type = self.state.trained_model['model_type']
                is_sequence_model = model_type in ['lstm', 'gru', 'transformer', 'ensemble']
                
                if is_sequence_model:
                    # Prepare sequences for neural networks
                    def create_sequences(X, seq_len):
                        X_seq = []
                        for i in range(len(X) - seq_len):
                            X_seq.append(X[i:i+seq_len])
                        return np.array(X_seq)
                    
                    X_seq = create_sequences(X, sequence_length)
                    
                    # Get predictions from sequence model
                    predictions_proba = model.predict_proba(X_seq)
                    predictions = predictions_proba[:, 1]  # Probability of price going up
                    
                    # Align predictions with prices (shift by sequence_length)
                    predictions = np.concatenate([np.full(sequence_length, 0.5), predictions])
                else:
                    # sklearn model - direct predictions
                    predictions = model.predict_proba(X)[:, 1]
                
                # Simulate trading
                balance = 10000.0
                position = 0.0
                trades = []
                equity_curve = [balance]
                
                for i in range(len(predictions)):
                    current_price = prices[i]
                    signal = predictions[i]
                    
                    # Entry logic
                    if position == 0 and signal >= entry_threshold:
                        position = (balance * position_size / 100) / current_price
                        balance -= position * current_price
                        entry_price = current_price
                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'amount': position,
                            'timestamp': i
                        })
                    
                    # Exit logic
                    elif position > 0:
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        
                        if signal <= exit_threshold or pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                            balance += position * current_price
                            trades.append({
                                'type': 'sell',
                                'price': current_price,
                                'amount': position,
                                'pnl_pct': pnl_pct,
                                'timestamp': i
                            })
                            position = 0
                    
                    # Update equity
                    if position > 0:
                        current_equity = balance + position * current_price
                    else:
                        current_equity = balance
                    equity_curve.append(current_equity)
                
                # Calculate metrics
                final_balance = balance if position == 0 else balance + position * prices[-1]
                total_return = (final_balance - 10000) / 10000 * 100
                
                winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
                losing_trades = [t for t in trades if t.get('pnl_pct', 0) < 0]
                win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
                
                # Calculate Sharpe ratio (simplified)
                returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak * 100
                max_drawdown = np.max(drawdown)
                
                # Calculate profit factor
                total_profit = sum(t.get('pnl_pct', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('pnl_pct', 0) for t in losing_trades))
                profit_factor = total_profit / total_loss if total_loss > 0 else 0
                
                self._emit_log("info", f"Backtest completed: return={total_return:.2f}%, trades={len(trades)}", "backtest")
                
                response = CommandResponse(
                    success=True,
                    message="Backtest completed",
                    data={
                        'total_return': round(total_return, 2),
                        'sharpe_ratio': round(sharpe_ratio, 2),
                        'max_drawdown': round(max_drawdown, 2),
                        'win_rate': round(win_rate, 2),
                        'profit_factor': round(profit_factor, 2),
                        'trade_count': len(trades)
                    },
                    ts=int(time.time() * 1000)
                )
                
                return jsonify(asdict(response))
            except Exception as e:
                logger.error(f"Error running backtest: {e}")
                self._emit_log("error", f"Error running backtest: {e}", "backtest")
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Error running backtest: {e}",
                    ts=int(time.time() * 1000)
                ))
        
        @self.app.route('/api/deployment/deploy', methods=['POST'])
        def deploy_model():
            """Deploy trained model for real-time trading."""
            data = request.get_json() or {}
            trading_mode = data.get('trading_mode', 'paper')
            position_size = data.get('position_size', 10.0)
            max_daily_loss = data.get('max_daily_loss', 5.0)
            
            # Check if model is trained
            if not self.state.trained_model:
                return jsonify(CommandResponse(
                    success=False,
                    message="No trained model. Please train a model first.",
                    ts=int(time.time() * 1000)
                ))
            
            self._emit_log("info", f"Deploying model in {trading_mode} mode", "deployment")
            
            try:
                # Update state
                self.state.trading_mode = TradingMode(trading_mode)
                self.state.running_trading = True
                
                # Store deployment config
                self.state.deployment_config = {
                    'trading_mode': trading_mode,
                    'position_size': position_size,
                    'max_daily_loss': max_daily_loss,
                    'deployed_at': int(time.time() * 1000)
                }
                
                # Start WebSocket connection for real-time data
                import threading
                
                def start_ws_task():
                    try:
                        if not self.state.okx_ws_client.connected:
                            self.state.okx_ws_client.connect()
                            # Subscribe to symbol
                            symbol = self.state.historical_data.get('symbol', 'BTC/USDT')
                            self.state.okx_ws_client.subscribe_ticker(symbol)
                            self.state.okx_ws_client.subscribe_orderbook(symbol, 20)
                    except Exception as e:
                        logger.error(f"Error starting WebSocket: {e}")
                
                ws_thread = threading.Thread(target=start_ws_task, daemon=True)
                ws_thread.start()
                
                self._emit_log("info", f"Model deployed in {trading_mode} mode", "deployment")
                
                response = CommandResponse(
                    success=True,
                    message=f"Model deployed in {trading_mode} mode",
                    data={
                        'trading_mode': trading_mode,
                        'position_size': position_size,
                        'max_daily_loss': max_daily_loss
                    },
                    ts=int(time.time() * 1000)
                )
                
                return jsonify(asdict(response))
            except Exception as e:
                logger.error(f"Error deploying model: {e}")
                self._emit_log("error", f"Error deploying model: {e}", "deployment")
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Error deploying model: {e}",
                    ts=int(time.time() * 1000)
                ))
        
        @self.app.route('/api/deployment/stop', methods=['POST'])
        def stop_deployment():
            """Stop deployed model."""
            self._emit_log("info", "Stopping deployment", "deployment")
            
            try:
                # Stop WebSocket
                import threading
                
                def stop_ws_task():
                    try:
                        if self.state.okx_ws_client.connected:
                            self.state.okx_ws_client.disconnect()
                    except Exception as e:
                        logger.error(f"Error stopping WebSocket: {e}")
                
                ws_thread = threading.Thread(target=stop_ws_task, daemon=True)
                ws_thread.start()
                ws_thread.join(timeout=5)
                
                # Update state
                self.state.running_trading = False
                self.state.deployment_config = None
                
                self._emit_log("info", "Deployment stopped", "deployment")
                
                response = CommandResponse(
                    success=True,
                    message="Deployment stopped",
                    ts=int(time.time() * 1000)
                )
                
                return jsonify(asdict(response))
            except Exception as e:
                logger.error(f"Error stopping deployment: {e}")
                self._emit_log("error", f"Error stopping deployment: {e}", "deployment")
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Error stopping deployment: {e}",
                    ts=int(time.time() * 1000)
                ))
        
        @self.app.route('/api/trading/start', methods=['POST'])
        def start_trading():
            """Start trading."""
            data = request.get_json() or {}
            mode_str = data.get('mode', 'paper')
            
            if not validate_trading_mode(mode_str):
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Invalid trading mode: {mode_str}",
                    ts=int(time.time() * 1000)
                ))
            
            mode = TradingMode(mode_str)
            
            # Update state
            self.state.trading_mode = mode
            self._start_trading()
            
            self._emit_log("info", f"Started trading in {mode.value} mode", "trading")
            
            response = CommandResponse(
                success=True,
                message=f"Trading started in {mode.value} mode",
                data={"mode": mode.value},
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/trading/stop', methods=['POST'])
        def stop_trading():
            """Stop trading."""
            self._stop_trading()
            
            self._emit_log("info", "Stopped trading", "trading")
            
            response = CommandResponse(
                success=True,
                message="Trading stopped",
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/mode', methods=['POST'])
        def set_mode():
            """Set trading mode."""
            data = request.get_json() or {}
            mode_str = data.get('mode')
            
            if not validate_trading_mode(mode_str):
                return jsonify(CommandResponse(
                    success=False,
                    message=f"Invalid trading mode: {mode_str}",
                    ts=int(time.time() * 1000)
                ))
            
            mode = TradingMode(mode_str)
            self.state.trading_mode = mode
            
            self._emit_log("info", f"Trading mode set to {mode.value}", "trading")
            
            response = CommandResponse(
                success=True,
                message=f"Trading mode set to {mode.value}",
                data={"mode": mode.value},
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/subscriptions', methods=['POST'])
        def update_subscriptions():
            """Update active subscriptions."""
            data = request.get_json() or {}
            
            symbols = data.get('symbols', [])
            channels = data.get('channels', [])
            depth = data.get('depth', 20)
            candle_timeframes = data.get('candle_timeframes', [])
            
            # Validate
            for symbol in symbols:
                if not validate_symbol_gui_format(symbol):
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Invalid symbol format: {symbol}",
                        ts=int(time.time() * 1000)
                    ))
            
            for channel in channels:
                if not validate_channel(channel):
                    return jsonify(CommandResponse(
                        success=False,
                        message=f"Invalid channel: {channel}",
                        ts=int(time.time() * 1000)
                    ))
            
            # Update state
            self.state.active_symbols = symbols
            self.state.active_channels = channels
            self.state.orderbook_depth = depth
            self.state.candle_timeframes = candle_timeframes
            
            self._emit_log("info", f"Updated subscriptions: {len(symbols)} symbols, {len(channels)} channels", "ingestion")
            
            response = CommandResponse(
                success=True,
                message="Subscriptions updated",
                data={
                    "symbols": symbols,
                    "channels": channels,
                    "depth": depth,
                    "candle_timeframes": candle_timeframes
                },
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
        
        @self.app.route('/api/paper/reset', methods=['POST'])
        def paper_reset():
            """Reset paper trading portfolio."""
            self.state.paper_balance = 10000.0
            self.state.paper_positions.clear()
            self.state.paper_trades.clear()
            
            self._emit_log("info", "Paper trading portfolio reset", "trading")
            
            response = CommandResponse(
                success=True,
                message="Paper trading portfolio reset",
                ts=int(time.time() * 1000)
            )
            
            return jsonify(asdict(response))
    
    def _setup_ws_events(self):
        """Setup WebSocket events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'status': 'connected', 'message': 'Connected to backend service'})
            
            # Send initial status
            self._emit_system_status()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription request."""
            channels = data.get('channels', [])
            logger.info(f"Client {request.sid} subscribed to: {channels}")
            emit('subscribed', {'channels': channels})
    
    def _start_ingestion(self):
        """Start ingestion."""
        if not self.state.connected_okx_ws:
            logger.warning("Cannot start ingestion: OKX WS not connected")
            return
        
        self.state.running_ingestion = True
        
        # Start WS client message loop in background thread
        import asyncio
        
        def run_ws_client():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run():
                try:
                    # Start message loop
                    await self.state.okx_ws_client._message_loop()
                except Exception as e:
                    logger.error(f"WS client error: {e}")
            
            loop.run_until_complete(run())
        
        ws_thread = threading.Thread(target=run_ws_client, daemon=True)
        ws_thread.start()
        
        # Subscribe to configured channels
        if self.state.active_symbols and self.state.active_channels:
            # This will be done via the WS client's async methods
            # For now, we'll handle this in the connect flow
            pass
    
    def _stop_ingestion(self):
        """Stop ingestion."""
        self.state.running_ingestion = False
        
        # Stop WS client
        import asyncio
        
        def disconnect_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def do_disconnect():
                try:
                    await self.state.okx_ws_client.disconnect()
                except Exception as e:
                    logger.error(f"WS disconnect error: {e}")
            
            loop.run_until_complete(do_disconnect())
        
        disconnect_thread = threading.Thread(target=disconnect_task, daemon=True)
        disconnect_thread.start()
    
    def _start_trading(self):
        """Start trading."""
        self.state.running_trading = True
        # TODO: Integrate with paper trading engine
    
    def _stop_trading(self):
        """Stop trading."""
        self.state.running_trading = False
        # TODO: Stop paper trading engine
    
    def _on_price_update(self, event: PriceUpdateEvent):
        """Handle price update from WS client."""
        self.state.queues['price_updates'] += 1
        self.socketio.emit('price_update', serialize_event(event))
    
    def _on_orderbook_update(self, event: OrderbookUpdateEvent):
        """Handle orderbook update from WS client."""
        self.state.queues['orderbook_updates'] += 1
        self.socketio.emit('orderbook_update', serialize_event(event))
    
    def _on_candle_update(self, event: CandleUpdateEvent):
        """Handle candle update from WS client."""
        self.state.queues['candle_updates'] += 1
        self.socketio.emit('candle_update', serialize_event(event))
    
    def _on_paper_trade(self, event: PaperTradeUpdateEvent):
        """Handle paper trade event."""
        self.state.queues['trades'] += 1
        self.socketio.emit('paper_trade_update', serialize_event(event))
        
        # Emit updated metrics
        metrics = self.state.paper_engine.get_metrics()
        self.socketio.emit('metrics_update', serialize_event(metrics))
    
    def _emit_log(self, level: str, message: str, component: str):
        """Emit log event to all connected clients."""
        event = LogEvent(
            level=level,
            message=message,
            component=component,
            ts=int(time.time() * 1000)
        )
        self.socketio.emit('log_event', serialize_event(event))
    
    def _emit_system_status(self):
        """Emit system status to all connected clients."""
        event = SystemStatusEvent(
            connected_okx_ws=self.state.connected_okx_ws,
            connected_okx_rest=self.state.connected_okx_rest,
            running_ingestion=self.state.running_ingestion,
            running_trading=self.state.running_trading,
            queues=self.state.queues.copy(),
            last_heartbeat=self.state.last_heartbeat,
            ts=int(time.time() * 1000),
            active_symbols=self.state.active_symbols.copy(),
            active_channels=self.state.active_channels.copy()
        )
        self.socketio.emit('system_status', serialize_event(event))
    
    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self.heartbeat_running:
            try:
                self.state.last_heartbeat = int(time.time() * 1000)
                self._emit_system_status()
                time.sleep(5)  # Heartbeat every 5 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5)
    
    def run(self, block: bool = True):
        """Run the backend service."""
        # Start heartbeat thread
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        logger.info(f"Starting backend service on {self.host}:{self.port}")
        
        if block:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        else:
            # Run in background
            server_thread = threading.Thread(
                target=lambda: self.socketio.run(self.app, host=self.host, port=self.port, debug=False),
                daemon=True
            )
            server_thread.start()
            return server_thread
    
    def stop(self):
        """Stop the backend service."""
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2)
        
        # Stop ingestion and trading
        if self.state.running_ingestion:
            self._stop_ingestion()
        if self.state.running_trading:
            self._stop_trading()
        
        logger.info("Backend service stopped")


# Global service instance
_service_instance: Optional[BackendService] = None


def get_service() -> BackendService:
    """Get global backend service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BackendService()
    return _service_instance


def create_service(host: str = "127.0.0.1", port: int = 5050) -> BackendService:
    """Create a new backend service instance."""
    global _service_instance
    _service_instance = BackendService(host=host, port=port)
    return _service_instance
