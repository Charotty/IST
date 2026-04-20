from __future__ import annotations

import sys
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QFrame, QTabWidget,
    QTextEdit, QTableWidget, QTableWidgetItem, QProgressBar,
    QComboBox, QGroupBox, QScrollArea, QGridLayout, QMessageBox,
    QStatusBar, QMenuBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QEvent
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QPen, QPixmap, QAction

from components.sidebar import Sidebar
from components.main_chart import MainChart
from components.signal_panel import SignalPanel
from components.model_panel import ModelPanel
from components.bottom_panel import BottomPanel
from styles.dark_theme import DarkTheme
from real_backend_bridge import RealBackendBridge, create_backend_bridge
from data_adapters import GUISignal, GUITrade, GUIMetrics
from format_validators import FormatValidator, ValidationError
from async_integration import (
    AsyncEventLoopManager, AsyncSignalBridge, AsyncDataStream,
    AsyncBatchProcessor, AsyncCache, get_async_manager,
    get_signal_bridge, get_data_stream, get_batch_processor,
    get_cache, async_slot, async_cached
)

logger = logging.getLogger(__name__)


class AsyncMainWindow(QMainWindow):
    """Async-enabled ITS trading application with comprehensive PyQt integration."""
    
    # Standard signals
    signal_updated = pyqtSignal(str, float, float)  # signal, confidence, predicted_change
    price_updated = pyqtSignal(str, float)  # symbol, price
    log_added = pyqtSignal(str, str, str)  # timestamp, level, message
    trade_executed = pyqtSignal(dict)  # trade data
    metrics_updated = pyqtSignal(dict)  # metrics data
    models_updated = pyqtSignal(list)  # model data
    
    # Async-specific signals
    async_operation_started = pyqtSignal(str)  # operation_name
    async_operation_completed = pyqtSignal(str, object)  # operation_name, result
    async_operation_failed = pyqtSignal(str, str)  # operation_name, error
    async_operation_progress = pyqtSignal(str, float)  # operation_name, progress
    
    def __init__(self, use_real_backend: bool = True):
        super().__init__()
        
        self.use_real_backend = use_real_backend
        
        # Apply dark theme
        self.setStyleSheet(DarkTheme.get_stylesheet())
        
        # Window properties
        self.setWindowTitle("Intelligent Trading System - Async")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # System state
        self.system_status = "STOPPED"
        self.selected_pair = "BTC/USDT"
        self.trading_mode = "paper"
        self.backend_connected = False
        
        # Async system components
        self.async_manager: Optional[AsyncEventLoopManager] = None
        self.signal_bridge: Optional[AsyncSignalBridge] = None
        self.data_stream: Optional[AsyncDataStream] = None
        self.batch_processor: Optional[AsyncBatchProcessor] = None
        self.cache: Optional[AsyncCache] = None
        
        # Backend bridge
        self.backend_bridge = create_backend_bridge(use_real_backend)
        
        # Performance tracking
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        self.operation_count = 0
        
        # Initialize UI
        self.init_ui()
        
        # Setup async system
        self.setup_async_system()
        
        # Setup timers
        self.setup_timers()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize backend connection
        self.initialize_backend()
        
        logger.info("AsyncMainWindow initialized")
    
    def init_ui(self):
        """Initialize the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create sidebar
        self.sidebar = Sidebar()
        self.sidebar.pair_selected.connect(self.on_pair_selected)
        self.sidebar.start_clicked.connect(self.on_start_clicked)
        self.sidebar.stop_clicked.connect(self.on_stop_clicked)
        self.sidebar.mode_changed.connect(self.on_mode_changed)
        main_splitter.addWidget(self.sidebar)
        
        # Create right panel container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Create status bar
        self.create_status_bar(right_layout)
        
        # Create top splitter for chart and side panels
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_layout.addWidget(top_splitter, 1)
        
        # Create main chart
        self.main_chart = MainChart()
        top_splitter.addWidget(self.main_chart)
        
        # Create right side panels container
        right_panels_container = QWidget()
        right_panels_layout = QVBoxLayout(right_panels_container)
        right_panels_layout.setContentsMargins(0, 0, 0, 0)
        right_panels_layout.setSpacing(0)
        
        # Create signal panel
        self.signal_panel = SignalPanel()
        right_panels_layout.addWidget(self.signal_panel)
        
        # Create model panel
        self.model_panel = ModelPanel()
        self.model_panel.model_selected.connect(self.on_model_selected)
        right_panels_layout.addWidget(self.model_panel)
        
        top_splitter.addWidget(right_panels_container)
        
        # Set splitter sizes (70% chart, 30% side panels)
        top_splitter.setSizes([980, 420])
        
        # Create bottom panel
        self.bottom_panel = BottomPanel()
        right_layout.addWidget(self.bottom_panel, 0)
        
        # Set bottom panel height
        self.bottom_panel.setFixedHeight(220)
        
        # Add right container to main splitter
        main_splitter.addWidget(right_container)
        
        # Set main splitter sizes (200px sidebar, rest content)
        main_splitter.setSizes([200, 1200])
        
        # Make splitter handles invisible
        for splitter in [main_splitter, top_splitter]:
            splitter.setHandleWidth(1)
            splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #1F2933;
                }
            """)
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Export actions
        export_trades_action = QAction("Export Trades", self)
        export_trades_action.triggered.connect(self.export_trades)
        file_menu.addAction(export_trades_action)
        
        export_signals_action = QAction("Export Signals", self)
        export_signals_action.triggered.connect(self.export_signals)
        file_menu.addAction(export_signals_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Clear cache action
        clear_cache_action = QAction("Clear Cache", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)
        
        # Refresh data action
        refresh_action = QAction("Refresh Data", self)
        refresh_action.triggered.connect(self.refresh_data)
        tools_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self, layout):
        """Create enhanced status bar with async status."""
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
                border-bottom: 1px solid #1F2933;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(16, 8, 16, 8)
        
        # Backend status indicator
        self.backend_status_label = QLabel("Backend: Disconnected")
        self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
        
        # Async status indicator
        self.async_status_label = QLabel("Async: Ready")
        self.async_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
        
        # Mode indicator
        self.mode_label = QLabel(f"Mode: {'Real' if self.use_real_backend else 'Demo'}")
        self.mode_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        # Performance info
        self.performance_label = QLabel("Signals: 0 | Trades: 0 | Errors: 0 | Ops: 0")
        self.performance_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        # Cache info
        self.cache_label = QLabel("Cache: 0 items")
        self.cache_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        status_layout.addWidget(self.backend_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.async_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.mode_label)
        status_layout.addStretch()
        status_layout.addWidget(self.performance_label)
        status_layout.addStretch()
        status_layout.addWidget(self.cache_label)
        
        layout.addWidget(status_frame)
    
    def setup_async_system(self):
        """Setup async system components."""
        try:
            # Initialize async manager
            self.async_manager = get_async_manager()
            
            # Initialize async components
            self.signal_bridge = get_signal_bridge()
            self.data_stream = get_data_stream()
            self.batch_processor = get_batch_processor()
            self.cache = get_cache(ttl=300.0)  # 5 minutes TTL
            
            # Connect async signals
            self.signal_bridge.signal_result.connect(self.on_async_signal_result)
            self.signal_bridge.signal_error.connect(self.on_async_signal_error)
            self.data_stream.data_received.connect(self.on_stream_data_received)
            self.data_stream.stream_error.connect(self.on_stream_error)
            self.batch_processor.batch_completed.connect(self.on_batch_completed)
            self.batch_processor.batch_error.connect(self.on_batch_error)
            self.batch_processor.batch_progress.connect(self.on_batch_progress)
            
            # Connect operation signals
            self.async_operation_started.connect(self.on_async_operation_started)
            self.async_operation_completed.connect(self.on_async_operation_completed)
            self.async_operation_failed.connect(self.on_async_operation_failed)
            self.async_operation_progress.connect(self.on_async_operation_progress)
            
            logger.info("Async system setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup async system: {e}")
            self.add_log("error", f"Async system setup failed: {e}")
    
    def setup_timers(self):
        """Setup timers for GUI operations."""
        # Backend status check timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_backend_status)
        self.status_timer.start(5000)  # Check every 5 seconds
        
        # Performance update timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_display)
        self.performance_timer.start(2000)  # Update every 2 seconds
        
        # Cache status update timer
        self.cache_timer = QTimer()
        self.cache_timer.timeout.connect(self.update_cache_display)
        self.cache_timer.start(10000)  # Update every 10 seconds
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.signal_updated.connect(self.on_signal_updated)
        self.price_updated.connect(self.on_price_updated)
        self.log_added.connect(self.on_log_added)
        self.trade_executed.connect(self.on_trade_executed)
        self.metrics_updated.connect(self.on_metrics_updated)
        self.models_updated.connect(self.on_models_updated)
    
    def initialize_backend(self):
        """Initialize backend connection asynchronously."""
        if not self.async_manager:
            self.add_log("error", "Async system not available")
            return
        
        # Submit async backend connection
        task_id = self.async_manager.submit_task(
            self.backend_bridge.connect(),
            callback=self.on_backend_connected,
            error_callback=self.on_backend_connection_error
        )
        
        self.add_log("info", f"Connecting to {'real' if self.use_real_backend else 'demo'} backend...")
        self.async_operation_started.emit("backend_connect")
    
    @async_slot()
    async def on_backend_connected(self, connected: bool):
        """Handle backend connection completion."""
        self.backend_connected = connected
        
        if connected:
            self.backend_status_label.setText("Backend: Connected")
            self.backend_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
            self.add_log("info", f"Backend connected successfully ({'real' if self.use_real_backend else 'demo'} mode)")
            
            # Setup data streams
            await self.setup_data_streams()
            
            # Get initial status
            status = self.backend_bridge.get_system_status()
            self.add_log("info", f"System ready - {status.get('current_symbol', 'Unknown')}")
        else:
            self.backend_status_label.setText("Backend: Failed")
            self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
            self.add_log("error", "Backend connection failed")
        
        self.async_operation_completed.emit("backend_connect", connected)
    
    def on_backend_connection_error(self, error: str):
        """Handle backend connection error."""
        self.backend_connected = False
        self.backend_status_label.setText("Backend: Error")
        self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
        self.add_log("error", f"Backend connection error: {error}")
        self.increment_error_count()
        self.async_operation_failed.emit("backend_connect", error)
    
    async def setup_data_streams(self):
        """Setup async data streams."""
        if not self.data_stream or not self.backend_bridge:
            return
        
        try:
            # Create price data stream
            await self.data_stream.create_stream("price_data", max_size=1000)
            
            # Create signal data stream
            await self.data_stream.create_stream("signal_data", max_size=500)
            
            # Create trade data stream
            await self.data_stream.create_stream("trade_data", max_size=1000)
            
            # Start stream processors
            self.data_stream.start_stream_processor(
                "price_data",
                self.process_price_data,
                error_handler=self.handle_stream_error
            )
            
            self.data_stream.start_stream_processor(
                "signal_data",
                self.process_signal_data,
                error_handler=self.handle_stream_error
            )
            
            self.data_stream.start_stream_processor(
                "trade_data",
                self.process_trade_data,
                error_handler=self.handle_stream_error
            )
            
            logger.info("Data streams setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup data streams: {e}")
            self.add_log("error", f"Data streams setup failed: {e}")
    
    def process_price_data(self, data: Any) -> Optional[Any]:
        """Process price data from stream."""
        try:
            if isinstance(data, dict) and "symbol" in data and "price" in data:
                self.price_updated.emit(data["symbol"], data["price"])
                return data
        except Exception as e:
            logger.error(f"Error processing price data: {e}")
        return None
    
    def process_signal_data(self, data: Any) -> Optional[Any]:
        """Process signal data from stream."""
        try:
            if isinstance(data, GUISignal):
                # Validate signal
                errors = FormatValidator.validate_signal(data)
                if not errors:
                    self.signal_updated.emit(data.signal, data.confidence, data.predicted_change)
                    return data
                else:
                    logger.warning(f"Invalid signal in stream: {errors[0].message}")
        except Exception as e:
            logger.error(f"Error processing signal data: {e}")
        return None
    
    def process_trade_data(self, data: Any) -> Optional[Any]:
        """Process trade data from stream."""
        try:
            if isinstance(data, GUITrade):
                # Validate trade
                errors = FormatValidator.validate_trade(data)
                if not errors:
                    trade_dict = {
                        "id": data.id,
                        "timestamp": data.timestamp,
                        "pair": data.pair,
                        "type": data.type,
                        "entry_price": data.entry_price,
                        "exit_price": data.exit_price,
                        "pnl": data.pnl
                    }
                    self.trade_executed.emit(trade_dict)
                    return trade_dict
                else:
                    logger.warning(f"Invalid trade in stream: {errors[0].message}")
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
        return None
    
    def handle_stream_error(self, error: Exception):
        """Handle stream processing errors."""
        logger.error(f"Stream processing error: {error}")
        self.add_log("error", f"Stream error: {error}")
        self.increment_error_count()
    
    # Async signal handlers
    def on_async_signal_result(self, signal_name: str, result: Any):
        """Handle async signal result."""
        logger.debug(f"Async signal {signal_name} completed with result: {type(result)}")
    
    def on_async_signal_error(self, signal_name: str, error: str):
        """Handle async signal error."""
        logger.error(f"Async signal {signal_name} failed: {error}")
        self.add_log("error", f"Async operation {signal_name} failed: {error}")
        self.increment_error_count()
    
    def on_stream_data_received(self, stream_id: str, data: Any):
        """Handle stream data received."""
        logger.debug(f"Stream {stream_id} received data: {type(data)}")
    
    def on_stream_error(self, stream_id: str, error: str):
        """Handle stream error."""
        logger.error(f"Stream {stream_id} error: {error}")
        self.add_log("error", f"Stream {stream_id} error: {error}")
        self.increment_error_count()
    
    def on_batch_completed(self, batch_id: str, results: List[Any]):
        """Handle batch completion."""
        logger.debug(f"Batch {batch_id} completed with {len(results)} results")
        self.add_log("info", f"Batch {batch_id} processed {len(results)} items")
    
    def on_batch_error(self, batch_id: str, error: str):
        """Handle batch error."""
        logger.error(f"Batch {batch_id} error: {error}")
        self.add_log("error", f"Batch {batch_id} failed: {error}")
        self.increment_error_count()
    
    def on_batch_progress(self, batch_id: str, completed: int, total: int):
        """Handle batch progress."""
        progress = (completed / total) * 100 if total > 0 else 0
        logger.debug(f"Batch {batch_id} progress: {progress:.1f}%")
    
    def on_async_operation_started(self, operation_name: str):
        """Handle async operation start."""
        self.operation_count += 1
        logger.debug(f"Async operation {operation_name} started")
    
    def on_async_operation_completed(self, operation_name: str, result: Any):
        """Handle async operation completion."""
        logger.debug(f"Async operation {operation_name} completed")
    
    def on_async_operation_failed(self, operation_name: str, error: str):
        """Handle async operation failure."""
        logger.error(f"Async operation {operation_name} failed: {error}")
        self.increment_error_count()
    
    def on_async_operation_progress(self, operation_name: str, progress: float):
        """Handle async operation progress."""
        logger.debug(f"Async operation {operation_name} progress: {progress:.1f}%")
    
    # GUI event handlers
    def on_pair_selected(self, pair: str):
        """Handle pair selection from sidebar."""
        self.selected_pair = pair
        self.main_chart.set_pair(pair)
        self.add_log("info", f"Selected trading pair: {pair}")
        
        # Update backend if running
        if self.system_status == "RUNNING" and self.backend_connected:
            self.restart_trading_with_new_pair(pair)
    
    def restart_trading_with_new_pair(self, pair: str):
        """Restart trading with new pair asynchronously."""
        if not self.async_manager:
            return
        
        # Stop current trading
        task_id = self.async_manager.submit_task(
            self.backend_bridge.stop_trading(),
            callback=lambda: self.start_trading_with_pair(pair),
            error_callback=lambda e: self.add_log("error", f"Failed to restart trading: {e}")
        )
        
        self.async_operation_started.emit("restart_trading")
    
    def start_trading_with_pair(self, pair: str):
        """Start trading with new pair."""
        if not self.async_manager:
            return
        
        task_id = self.async_manager.submit_task(
            self.backend_bridge.start_trading(pair, self.trading_mode),
            callback=self.on_trading_restarted,
            error_callback=lambda e: self.on_trading_restart_error(e)
        )
    
    def on_trading_restarted(self, started: bool):
        """Handle trading restart completion."""
        if started:
            self.add_log("info", f"Trading restarted with {self.selected_pair}")
            self.async_operation_completed.emit("restart_trading", started)
        else:
            self.add_log("error", "Failed to restart trading")
            self.async_operation_failed.emit("restart_trading", "Failed to start")
    
    def on_trading_restart_error(self, error: str):
        """Handle trading restart error."""
        self.add_log("error", f"Trading restart error: {error}")
        self.async_operation_failed.emit("restart_trading", error)
    
    def on_start_clicked(self):
        """Handle start button click."""
        if self.system_status == "STOPPED":
            if not self.backend_connected:
                self.show_error_message("Backend not connected", "Please wait for backend to connect or check connection.")
                return
            
            self.system_status = "RUNNING"
            self.sidebar.set_system_status(self.system_status)
            
            # Start backend trading asynchronously
            if self.async_manager:
                task_id = self.async_manager.submit_task(
                    self.backend_bridge.start_trading(self.selected_pair, self.trading_mode),
                    callback=self.on_trading_started,
                    error_callback=lambda e: self.on_trading_start_error(e)
                )
                self.async_operation_started.emit("start_trading")
            
            self.add_log("info", "Starting trading system...")
    
    def on_trading_started(self, started: bool):
        """Handle trading start completion."""
        if started:
            self.add_log("info", f"Trading system started - {self.selected_pair}")
            self.add_log("info", f"Mode: {self.trading_mode.upper()}")
            self.async_operation_completed.emit("start_trading", started)
        else:
            self.add_log("error", "Failed to start trading")
            self.system_status = "ERROR"
            self.sidebar.set_system_status(self.system_status)
            self.async_operation_failed.emit("start_trading", "Failed to start")
    
    def on_trading_start_error(self, error: str):
        """Handle trading start error."""
        self.add_log("error", f"Trading start error: {error}")
        self.system_status = "ERROR"
        self.sidebar.set_system_status(self.system_status)
        self.async_operation_failed.emit("start_trading", error)
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.system_status == "RUNNING":
            self.system_status = "STOPPED"
            self.sidebar.set_system_status(self.system_status)
            
            # Stop backend trading asynchronously
            if self.async_manager:
                task_id = self.async_manager.submit_task(
                    self.backend_bridge.stop_trading(),
                    callback=self.on_trading_stopped,
                    error_callback=lambda e: self.on_trading_stop_error(e)
                )
                self.async_operation_started.emit("stop_trading")
            
            self.add_log("info", "Stopping trading system...")
    
    def on_trading_stopped(self, result):
        """Handle trading stop completion."""
        self.add_log("info", "Trading system stopped")
        self.async_operation_completed.emit("stop_trading", result)
    
    def on_trading_stop_error(self, error: str):
        """Handle trading stop error."""
        self.add_log("error", f"Trading stop error: {error}")
        self.async_operation_failed.emit("stop_trading", error)
    
    def on_mode_changed(self, mode: str):
        """Handle trading mode change."""
        self.trading_mode = mode
        self.add_log("info", f"Trading mode changed to: {mode.upper()}")
        
        # Restart trading if running
        if self.system_status == "RUNNING":
            self.restart_trading_with_mode(mode)
    
    def restart_trading_with_mode(self, mode: str):
        """Restart trading with new mode."""
        if not self.async_manager:
            return
        
        task_id = self.async_manager.submit_task(
            self.backend_bridge.stop_trading(),
            callback=lambda: self.start_trading_with_mode(mode),
            error_callback=lambda e: self.add_log("error", f"Failed to restart trading: {e}")
        )
    
    def start_trading_with_mode(self, mode: str):
        """Start trading with new mode."""
        if not self.async_manager:
            return
        
        task_id = self.async_manager.submit_task(
            self.backend_bridge.start_trading(self.selected_pair, mode),
            callback=self.on_trading_restarted,
            error_callback=lambda e: self.on_trading_restart_error(e)
        )
    
    def on_model_selected(self, model_name: str):
        """Handle model selection from model panel."""
        if self.backend_connected and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.backend_bridge.switch_model(model_name),
                callback=self.on_model_switched,
                error_callback=lambda e: self.on_model_switch_error(e)
            )
            self.async_operation_started.emit("switch_model")
    
    def on_model_switched(self, switched: bool):
        """Handle model switch completion."""
        if switched:
            status = self.backend_bridge.get_system_status()
            active_model = status.get("active_model", "Unknown")
            self.add_log("info", f"Switched to model: {active_model}")
            self.async_operation_completed.emit("switch_model", switched)
        else:
            self.add_log("error", "Failed to switch model")
            self.async_operation_failed.emit("switch_model", "Failed to switch")
    
    def on_model_switch_error(self, error: str):
        """Handle model switch error."""
        self.add_log("error", f"Model switch error: {error}")
        self.async_operation_failed.emit("switch_model", error)
    
    # Backend callback handlers
    def on_backend_signal(self, signal: GUISignal):
        """Handle signal from backend."""
        # Validate signal
        errors = FormatValidator.validate_signal(signal)
        if errors:
            self.add_log("warning", f"Invalid signal received: {errors[0].message}")
            return
        
        # Send to stream if available
        if self.data_stream:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.data_stream.send_data("signal_data", signal),
                    self.async_manager.loop
                )
            except:
                pass
        
        self.signal_updated.emit(signal.signal, signal.confidence, signal.predicted_change)
    
    def on_backend_price(self, symbol: str, price: float):
        """Handle price update from backend."""
        # Send to stream if available
        if self.data_stream:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.data_stream.send_data("price_data", {"symbol": symbol, "price": price}),
                    self.async_manager.loop
                )
            except:
                pass
        
        self.price_updated.emit(symbol, price)
    
    def on_backend_log(self, timestamp: str, level: str, message: str):
        """Handle log from backend."""
        self.log_added.emit(timestamp, level, message)
    
    def on_backend_trade(self, trade: GUITrade):
        """Handle trade from backend."""
        # Validate trade
        errors = FormatValidator.validate_trade(trade)
        if errors:
            self.add_log("warning", f"Invalid trade received: {errors[0].message}")
            return
        
        # Send to stream if available
        if self.data_stream:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.data_stream.send_data("trade_data", trade),
                    self.async_manager.loop
                )
            except:
                pass
        
        trade_dict = {
            "id": trade.id,
            "timestamp": trade.timestamp,
            "pair": trade.pair,
            "type": trade.type,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl
        }
        self.trade_executed.emit(trade_dict)
    
    def on_backend_metrics(self, metrics: GUIMetrics):
        """Handle metrics from backend."""
        # Validate metrics
        errors = FormatValidator.validate_metrics(metrics)
        if errors:
            self.add_log("warning", f"Invalid metrics received: {errors[0].message}")
            return
        
        metrics_dict = {
            "totalPnL": metrics.total_pnl,
            "winRate": metrics.win_rate,
            "totalTrades": metrics.total_trades,
            "maxDrawdown": metrics.max_drawdown,
            "sharpeRatio": metrics.sharpe_ratio,
            "profitFactor": metrics.profit_factor
        }
        self.metrics_updated.emit(metrics_dict)
    
    def on_backend_models(self, model_data: List[Dict]):
        """Handle model data from backend."""
        self.models_updated.emit(model_data)
    
    # GUI signal handlers
    def on_signal_updated(self, signal: str, confidence: float, predicted_change: float):
        """Handle signal update."""
        self.signal_panel.update_signal(signal, confidence, predicted_change)
        self.main_chart.add_signal(signal)
    
    def on_price_updated(self, symbol: str, price: float):
        """Handle price update."""
        self.main_chart.update_price(price)
    
    def on_log_added(self, timestamp: str, level: str, message: str):
        """Handle log addition."""
        self.bottom_panel.add_log(timestamp, level, message)
    
    def on_trade_executed(self, trade_data: dict):
        """Handle trade execution."""
        self.bottom_panel.add_trade(trade_data)
    
    def on_metrics_updated(self, metrics: dict):
        """Handle metrics update."""
        self.bottom_panel.update_metrics(metrics)
    
    def on_models_updated(self, model_data: List[Dict]):
        """Handle models update."""
        self.model_panel.set_models(model_data)
    
    def check_backend_status(self):
        """Periodically check backend status."""
        if self.backend_connected and self.backend_bridge.is_connected:
            try:
                status = self.backend_bridge.get_system_status()
                
                # Update performance display
                signal_count = status.get("signal_count", 0)
                trade_count = status.get("trade_count", 0)
                
                # Check for unexpected stops
                if self.system_status == "RUNNING" and not status.get("running", False):
                    self.system_status = "ERROR"
                    self.sidebar.set_system_status(self.system_status)
                    self.add_log("error", "Backend stopped unexpectedly")
                    self.increment_error_count()
                
                # Update backend status indicator
                if status.get("connected", False):
                    self.backend_status_label.setText("Backend: Connected")
                    self.backend_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
                else:
                    self.backend_status_label.setText("Backend: Disconnected")
                    self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
                
            except Exception as e:
                self.add_log("error", f"Backend status check failed: {e}")
                self.increment_error_count()
    
    def update_performance_display(self):
        """Update performance display."""
        if self.backend_connected:
            try:
                status = self.backend_bridge.get_system_status()
                
                signal_count = status.get("signal_count", 0)
                trade_count = status.get("trade_count", 0)
                
                self.performance_label.setText(
                    f"Signals: {signal_count} | Trades: {trade_count} | Errors: {self.error_count} | Ops: {self.operation_count}"
                )
            except:
                pass
    
    def update_cache_display(self):
        """Update cache display."""
        if self.cache and self.async_manager:
            try:
                cache_size = len(self.cache.cache)
                self.cache_label.setText(f"Cache: {cache_size} items")
            except:
                pass
    
    def increment_error_count(self):
        """Increment error counter."""
        self.error_count += 1
        self.last_error_time = datetime.now()
    
    def add_log(self, level: str, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_added.emit(timestamp, level, message)
    
    def show_error_message(self, title: str, message: str):
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
    
    # Menu actions
    def export_trades(self):
        """Export trades to CSV."""
        if not self.batch_processor:
            self.add_log("error", "Batch processor not available")
            return
        
        # Get trades from bottom panel
        trades = self.bottom_panel.trades_widget.get_all_trades()
        
        if not trades:
            self.add_log("info", "No trades to export")
            return
        
        # Process export asynchronously
        self.batch_processor.process_batch(
            "export_trades",
            trades,
            self.process_trade_for_export,
            batch_size=50
        )
        
        self.add_log("info", f"Exporting {len(trades)} trades...")
    
    def process_trade_for_export(self, trade: dict) -> Optional[dict]:
        """Process trade for export."""
        return trade  # Return trade as-is for CSV export
    
    def export_signals(self):
        """Export signals to CSV."""
        self.add_log("info", "Signal export not implemented yet")
    
    def clear_cache(self):
        """Clear async cache."""
        if self.cache and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.cache.clear(),
                callback=lambda: self.add_log("info", "Cache cleared"),
                error_callback=lambda e: self.add_log("error", f"Failed to clear cache: {e}")
            )
    
    def refresh_data(self):
        """Refresh all data."""
        if self.backend_connected and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.backend_bridge._load_initial_data(),
                callback=lambda: self.add_log("info", "Data refreshed"),
                error_callback=lambda e: self.add_log("error", f"Failed to refresh data: {e}")
            )
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ITS",
            "Intelligent Trading System\n"
            "Version 1.0.0\n\n"
            "Async-enabled trading application with real backend integration.\n\n"
            "Features:\n"
            "Real-time signal generation\n"
            "Paper trading execution\n"
            "Advanced ML models\n"
            "Comprehensive analytics"
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop backend
        if self.backend_bridge.is_running and self.async_manager:
            task_id = self.async_manager.submit_task(self.backend_bridge.stop_trading)
            # Wait briefly for completion
            threading.Event().wait(1.0)
        
        # Disconnect backend
        if self.backend_bridge.is_connected and self.async_manager:
            task_id = self.async_manager.submit_task(self.backend_bridge.disconnect)
            # Wait briefly for completion
            threading.Event().wait(1.0)
        
        # Stop streams
        if self.data_stream:
            for stream_id in ["price_data", "signal_data", "trade_data"]:
                self.data_stream.stop_stream(stream_id)
        
        # Add final log
        self.add_log("info", "Application shutting down")
        
        # Log final performance summary
        if self.error_count > 0:
            self.add_log("warning", f"Session completed with {self.error_count} errors")
        else:
            self.add_log("info", "Session completed successfully")
        
        # Accept close event
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Intelligent Trading System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ITS")
    
    # Check command line arguments for backend mode
    use_real_backend = "--demo" not in sys.argv
    
    # Create and show main window
    window = AsyncMainWindow(use_real_backend)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
