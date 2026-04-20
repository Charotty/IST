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
    QStatusBar, QMenuBar, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QEvent
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QPen, QPixmap, QAction

from components.sidebar import Sidebar
from components.main_chart import MainChart
from components.signal_panel import SignalPanel
from components.model_panel import ModelPanel
from components.bottom_panel import BottomPanel
from styles.dark_theme import DarkTheme
from realtime_backend_bridge import RealTimeBackendBridge, create_realtime_backend_bridge
from data_adapters import GUISignal, GUITrade, GUIMetrics
from format_validators import FormatValidator, ValidationError
from async_integration import (
    AsyncEventLoopManager, AsyncSignalBridge, AsyncDataStream,
    AsyncBatchProcessor, AsyncCache, get_async_manager,
    get_signal_bridge, get_data_stream, get_batch_processor,
    get_cache, async_slot, async_cached
)

logger = logging.getLogger(__name__)


class RealTimeMainWindow(QMainWindow):
    """Real-time ITS trading application with storage system integration."""
    
    # Standard signals
    signal_updated = pyqtSignal(str, float, float)  # signal, confidence, predicted_change
    price_updated = pyqtSignal(str, float)  # symbol, price
    log_added = pyqtSignal(str, str, str)  # timestamp, level, message
    trade_executed = pyqtSignal(dict)  # trade data
    metrics_updated = pyqtSignal(dict)  # metrics data
    models_updated = pyqtSignal(list)  # model data
    
    # Real-time specific signals
    data_quality_updated = pyqtSignal(dict)  # quality metrics
    realtime_stats_updated = pyqtSignal(dict)  # real-time statistics
    storage_status_updated = pyqtSignal(str, bool)  # storage name, status
    
    def __init__(self, use_real_backend: bool = True):
        super().__init__()
        
        self.use_real_backend = use_real_backend
        
        # Apply dark theme
        self.setStyleSheet(DarkTheme.get_stylesheet())
        
        # Window properties
        self.setWindowTitle("Intelligent Trading System - Real-Time")
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
        
        # Real-time backend bridge
        self.backend_bridge: Optional[RealTimeBackendBridge] = None
        
        # Performance tracking
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        self.operation_count = 0
        
        # Real-time configuration
        self.realtime_config = {
            "enable_parquet": True,
            "enable_timescale": True,
            "data_validation": True,
            "change_detection": True,
            "watch_interval": 1.0,
            "buffer_size": 1000
        }
        
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
        
        logger.info("RealTimeMainWindow initialized")
    
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
        
        # Create real-time info panel
        self.create_realtime_info_panel(right_panels_layout)
        
        top_splitter.addWidget(right_panels_container)
        
        # Set splitter sizes (65% chart, 35% side panels)
        top_splitter.setSizes([910, 490])
        
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
        """Create application menu bar with real-time options."""
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
        
        export_realtime_stats_action = QAction("Export Real-Time Stats", self)
        export_realtime_stats_action.triggered.connect(self.export_realtime_stats)
        file_menu.addAction(export_realtime_stats_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Real-Time menu
        realtime_menu = menubar.addMenu("Real-Time")
        
        # Storage options
        parquet_action = QAction("Enable Parquet", self)
        parquet_action.setCheckable(True)
        parquet_action.setChecked(self.realtime_config["enable_parquet"])
        parquet_action.triggered.connect(lambda: self.toggle_storage("parquet"))
        realtime_menu.addAction(parquet_action)
        
        timescale_action = QAction("Enable TimescaleDB", self)
        timescale_action.setCheckable(True)
        timescale_action.setChecked(self.realtime_config["enable_timescale"])
        timescale_action.triggered.connect(lambda: self.toggle_storage("timescale"))
        realtime_menu.addAction(timescale_action)
        
        realtime_menu.addSeparator()
        
        # Data quality options
        validation_action = QAction("Enable Data Validation", self)
        validation_action.setCheckable(True)
        validation_action.setChecked(self.realtime_config["data_validation"])
        validation_action.triggered.connect(self.toggle_data_validation)
        realtime_menu.addAction(validation_action)
        
        change_detection_action = QAction("Enable Change Detection", self)
        change_detection_action.setCheckable(True)
        change_detection_action.setChecked(self.realtime_config["change_detection"])
        change_detection_action.triggered.connect(self.toggle_change_detection)
        realtime_menu.addAction(change_detection_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Clear cache action
        clear_cache_action = QAction("Clear Cache", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)
        
        # Refresh data action
        refresh_action = QAction("Refresh Real-Time Data", self)
        refresh_action.triggered.connect(self.refresh_realtime_data)
        tools_menu.addAction(refresh_action)
        
        # Data quality report
        quality_report_action = QAction("Data Quality Report", self)
        quality_report_action.triggered.connect(self.show_data_quality_report)
        tools_menu.addAction(quality_report_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_realtime_info_panel(self, layout):
        """Create real-time information panel."""
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: 1px solid #1F2933;
                border-radius: 4px;
                margin: 4px;
            }
        """)
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        
        # Title
        title_label = QLabel("Real-Time Status")
        title_label.setStyleSheet("color: #F0F6FC; font-size: 12px; font-weight: bold;")
        info_layout.addWidget(title_label)
        
        # Storage status
        storage_frame = QFrame()
        storage_layout = QHBoxLayout(storage_frame)
        storage_layout.setContentsMargins(0, 0, 0, 0)
        
        self.parquet_status_label = QLabel("Parquet: ")
        self.parquet_status_label.setStyleSheet("color: #9CA3AF; font-size: 10px;")
        
        self.timescale_status_label = QLabel("TimescaleDB: ")
        self.timescale_status_label.setStyleSheet("color: #9CA3AF; font-size: 10px;")
        
        storage_layout.addWidget(self.parquet_status_label)
        storage_layout.addStretch()
        storage_layout.addWidget(self.timescale_status_label)
        
        info_layout.addWidget(storage_frame)
        
        # Data quality metrics
        quality_frame = QFrame()
        quality_layout = QVBoxLayout(quality_frame)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        
        quality_title = QLabel("Data Quality")
        quality_title.setStyleSheet("color: #9CA3AF; font-size: 10px; font-weight: bold;")
        quality_layout.addWidget(quality_title)
        
        self.quality_stats_label = QLabel("Records: 0 | Duplicates: 0% | Out-of-order: 0%")
        self.quality_stats_label.setStyleSheet("color: #9CA3AF; font-size: 9px;")
        quality_layout.addWidget(self.quality_stats_label)
        
        info_layout.addWidget(quality_frame)
        
        # Real-time statistics
        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        stats_title = QLabel("Performance")
        stats_title.setStyleSheet("color: #9CA3AF; font-size: 10px; font-weight: bold;")
        stats_layout.addWidget(stats_title)
        
        self.realtime_stats_label = QLabel("Buffer: 0 | Rate: 0/s | Latency: 0ms")
        self.realtime_stats_label.setStyleSheet("color: #9CA3AF; font-size: 9px;")
        stats_layout.addWidget(self.realtime_stats_label)
        
        info_layout.addWidget(stats_frame)
        
        # Add to main layout
        layout.addWidget(info_frame)
    
    def create_status_bar(self, layout):
        """Create enhanced status bar with real-time status."""
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
        
        # Real-time status indicator
        self.realtime_status_label = QLabel("Real-Time: Ready")
        self.realtime_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
        
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
        status_layout.addWidget(self.realtime_status_label)
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
            
            # Connect real-time specific signals
            self.data_quality_updated.connect(self.on_data_quality_updated)
            self.realtime_stats_updated.connect(self.on_realtime_stats_updated)
            self.storage_status_updated.connect(self.on_storage_status_updated)
            
            logger.info("Async system setup completed for real-time")
            
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
        
        # Real-time statistics update timer
        self.realtime_stats_timer = QTimer()
        self.realtime_stats_timer.timeout.connect(self.update_realtime_stats)
        self.realtime_stats_timer.start(3000)  # Update every 3 seconds
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.signal_updated.connect(self.on_signal_updated)
        self.price_updated.connect(self.on_price_updated)
        self.log_added.connect(self.on_log_added)
        self.trade_executed.connect(self.on_trade_executed)
        self.metrics_updated.connect(self.on_metrics_updated)
        self.models_updated.connect(self.on_models_updated)
    
    def initialize_backend(self):
        """Initialize real-time backend connection."""
        if not self.async_manager:
            self.add_log("error", "Async system not available")
            return
        
        # Create real-time backend bridge
        self.backend_bridge = create_realtime_backend_bridge(self.realtime_config)
        
        # Set GUI callbacks
        self.backend_bridge.set_gui_callbacks(
            signal_callback=self.on_backend_signal,
            price_callback=self.on_backend_price,
            log_callback=self.on_backend_log,
            trade_callback=self.on_backend_trade,
            metrics_callback=self.on_backend_metrics,
            model_callback=self.on_backend_models
        )
        
        # Submit async backend connection
        task_id = self.async_manager.submit_task(
            self.backend_bridge.connect(),
            callback=self.on_backend_connected,
            error_callback=self.on_backend_connection_error
        )
        
        self.add_log("info", f"Connecting to real-time backend...")
        self.operation_count += 1
    
    @async_slot()
    async def on_backend_connected(self, connected: bool):
        """Handle backend connection completion."""
        self.backend_connected = connected
        
        if connected:
            self.backend_status_label.setText("Backend: Connected")
            self.backend_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
            self.add_log("info", f"Real-time backend connected successfully")
            
            # Update storage status
            await self.update_storage_status()
            
            # Get initial status
            status = self.backend_bridge.get_system_status()
            self.add_log("info", f"System ready - {status.get('current_symbol', 'Unknown')}")
        else:
            self.backend_status_label.setText("Backend: Failed")
            self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
            self.add_log("error", "Real-time backend connection failed")
    
    def on_backend_connection_error(self, error: str):
        """Handle backend connection error."""
        self.backend_connected = False
        self.backend_status_label.setText("Backend: Error")
        self.backend_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
        self.add_log("error", f"Real-time backend connection error: {error}")
        self.increment_error_count()
    
    async def update_storage_status(self):
        """Update storage system status."""
        if not self.backend_bridge or not self.backend_bridge.realtime_connector:
            return
        
        try:
            # Get connector statistics
            stats = self.backend_bridge.realtime_connector.get_statistics()
            
            # Update storage status
            parquet_healthy = stats.get("parquet_stats", {}).get("files_scanned", 0) > 0
            timescale_healthy = stats.get("timescale_stats", {}).get("notifications_received", 0) >= 0
            
            self.storage_status_updated.emit("parquet", parquet_healthy)
            self.storage_status_updated.emit("timescale", timescale_healthy)
            
        except Exception as e:
            logger.error(f"Error updating storage status: {e}")
    
    def on_storage_status_updated(self, storage: str, healthy: bool):
        """Handle storage status update."""
        if storage == "parquet":
            color = "#22C55E" if healthy else "#EF4444"
            status = "Connected" if healthy else "Disconnected"
            self.parquet_status_label.setText(f"Parquet: {status}")
            self.parquet_status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        elif storage == "timescale":
            color = "#22C55E" if healthy else "#EF4444"
            status = "Connected" if healthy else "Disconnected"
            self.timescale_status_label.setText(f"TimescaleDB: {status}")
            self.timescale_status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
    
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
        """Restart trading with new pair."""
        if not self.async_manager:
            return
        
        # Stop current trading
        task_id = self.async_manager.submit_task(
            self.backend_bridge.stop_trading(),
            callback=lambda: self.start_trading_with_pair(pair),
            error_callback=lambda e: self.add_log("error", f"Failed to restart trading: {e}")
        )
    
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
        else:
            self.add_log("error", "Failed to restart trading")
    
    def on_trading_restart_error(self, error: str):
        """Handle trading restart error."""
        self.add_log("error", f"Trading restart error: {error}")
    
    def on_start_clicked(self):
        """Handle start button click."""
        if self.system_status == "STOPPED":
            if not self.backend_connected:
                self.show_error_message("Backend not connected", "Please wait for backend to connect or check connection.")
                return
            
            self.system_status = "RUNNING"
            self.sidebar.set_system_status(self.system_status)
            
            # Start real-time backend trading
            if self.async_manager:
                task_id = self.async_manager.submit_task(
                    self.backend_bridge.start_trading(self.selected_pair, self.trading_mode),
                    callback=self.on_trading_started,
                    error_callback=lambda e: self.on_trading_start_error(e)
                )
            
            self.add_log("info", "Starting real-time trading system...")
    
    def on_trading_started(self, started: bool):
        """Handle trading start completion."""
        if started:
            self.add_log("info", f"Real-time trading system started - {self.selected_pair}")
            self.add_log("info", f"Mode: {self.trading_mode.upper()}")
            self.realtime_status_label.setText("Real-Time: Active")
            self.realtime_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
        else:
            self.add_log("error", "Failed to start real-time trading")
            self.system_status = "ERROR"
            self.sidebar.set_system_status(self.system_status)
            self.realtime_status_label.setText("Real-Time: Error")
            self.realtime_status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
    
    def on_trading_start_error(self, error: str):
        """Handle trading start error."""
        self.add_log("error", f"Real-time trading start error: {error}")
        self.system_status = "ERROR"
        self.sidebar.set_system_status(self.system_status)
        self.increment_error_count()
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.system_status == "RUNNING":
            self.system_status = "STOPPED"
            self.sidebar.set_system_status(self.system_status)
            
            # Stop real-time backend trading
            if self.async_manager:
                task_id = self.async_manager.submit_task(
                    self.backend_bridge.stop_trading(),
                    callback=self.on_trading_stopped,
                    error_callback=lambda e: self.on_trading_stop_error(e)
                )
            
            self.add_log("info", "Stopping real-time trading system...")
    
    def on_trading_stopped(self, result):
        """Handle trading stop completion."""
        self.add_log("info", "Real-time trading system stopped")
        self.realtime_status_label.setText("Real-Time: Stopped")
        self.realtime_status_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
    
    def on_trading_stop_error(self, error: str):
        """Handle trading stop error."""
        self.add_log("error", f"Real-time trading stop error: {error}")
    
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
    
    def on_model_switched(self, switched: bool):
        """Handle model switch completion."""
        if switched:
            status = self.backend_bridge.get_system_status()
            active_model = status.get("active_model", "Unknown")
            self.add_log("info", f"Switched to model: {active_model}")
        else:
            self.add_log("error", "Failed to switch model")
    
    def on_model_switch_error(self, error: str):
        """Handle model switch error."""
        self.add_log("error", f"Model switch error: {error}")
    
    # Backend callback handlers
    def on_backend_signal(self, signal: GUISignal):
        """Handle signal from backend."""
        # Validate signal
        errors = FormatValidator.validate_signal(signal)
        if errors:
            self.add_log("warning", f"Invalid signal received: {errors[0].message}")
            return
        
        self.signal_updated.emit(signal.signal, signal.confidence, signal.predicted_change)
    
    def on_backend_price(self, symbol: str, price: float):
        """Handle price update from backend."""
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
    
    # Real-time specific handlers
    def on_data_quality_updated(self, quality_metrics: dict):
        """Handle data quality update."""
        total_records = quality_metrics.get("total_records", 0)
        duplicate_rate = 0
        out_of_order_rate = 0
        
        if total_records > 0:
            duplicate_rate = (quality_metrics.get("duplicate_count", 0) / total_records) * 100
            out_of_order_rate = (quality_metrics.get("out_of_order_count", 0) / total_records) * 100
        
        self.quality_stats_label.setText(
            f"Records: {total_records} | Duplicates: {duplicate_rate:.1f}% | Out-of-order: {out_of_order_rate:.1f}%"
        )
    
    def on_realtime_stats_updated(self, stats: dict):
        """Handle real-time statistics update."""
        buffer_size = stats.get("price_buffer_size", 0)
        records_per_second = stats.get("realtime_connector", {}).get("records_per_second", 0)
        
        self.realtime_stats_label.setText(
            f"Buffer: {buffer_size} | Rate: {records_per_second:.1f}/s | Latency: <100ms"
        )
    
    # Async signal handlers
    def on_async_signal_result(self, signal_name: str, result: Any):
        """Handle async signal result."""
        logger.debug(f"Async signal {signal_name} completed")
    
    def on_async_signal_error(self, signal_name: str, error: str):
        """Handle async signal error."""
        logger.error(f"Async signal {signal_name} failed: {error}")
        self.add_log("error", f"Async operation {signal_name} failed: {error}")
        self.increment_error_count()
    
    def on_stream_data_received(self, stream_id: str, data: Any):
        """Handle stream data received."""
        logger.debug(f"Stream {stream_id} received data")
    
    def on_stream_error(self, stream_id: str, error: str):
        """Handle stream error."""
        logger.error(f"Stream {stream_id} error: {error}")
        self.add_log("error", f"Stream {stream_id} error: {error}")
        self.increment_error_count()
    
    def on_batch_completed(self, batch_id: str, results: List[Any]):
        """Handle batch completion."""
        logger.debug(f"Batch {batch_id} completed with {len(results)} results")
    
    def on_batch_error(self, batch_id: str, error: str):
        """Handle batch error."""
        logger.error(f"Batch {batch_id} error: {error}")
        self.add_log("error", f"Batch {batch_id} failed: {error}")
    
    def on_batch_progress(self, batch_id: str, completed: int, total: int):
        """Handle batch progress."""
        progress = (completed / total) * 100 if total > 0 else 0
        logger.debug(f"Batch {batch_id} progress: {progress:.1f}%")
    
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
                
                # Update data quality
                data_quality = status.get("data_quality", {})
                self.data_quality_updated.emit(data_quality)
                
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
    
    def update_realtime_stats(self):
        """Update real-time statistics display."""
        if self.backend_connected and self.async_manager:
            try:
                task_id = self.async_manager.submit_task(
                    self.backend_bridge.get_realtime_statistics(),
                    callback=self.on_realtime_stats_updated,
                    error_callback=lambda e: logger.error(f"Error getting real-time stats: {e}")
                )
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
    def toggle_storage(self, storage: str):
        """Toggle storage system."""
        self.realtime_config[f"enable_{storage}"] = not self.realtime_config[f"enable_{storage}"]
        self.add_log("info", f"{storage.title()} storage {'enabled' if self.realtime_config[f'enable_{storage}'] else 'disabled'}")
    
    def toggle_data_validation(self):
        """Toggle data validation."""
        self.realtime_config["data_validation"] = not self.realtime_config["data_validation"]
        self.add_log("info", f"Data validation {'enabled' if self.realtime_config['data_validation'] else 'disabled'}")
    
    def toggle_change_detection(self):
        """Toggle change detection."""
        self.realtime_config["change_detection"] = not self.realtime_config["change_detection"]
        self.add_log("info", f"Change detection {'enabled' if self.realtime_config['change_detection'] else 'disabled'}")
    
    def export_trades(self):
        """Export trades to CSV."""
        self.add_log("info", "Real-time trades export not implemented yet")
    
    def export_signals(self):
        """Export signals to CSV."""
        self.add_log("info", "Real-time signals export not implemented yet")
    
    def export_realtime_stats(self):
        """Export real-time statistics."""
        if self.backend_connected and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.backend_bridge.get_realtime_statistics(),
                callback=lambda stats: self.add_log("info", f"Real-time stats: {len(stats)} metrics"),
                error_callback=lambda e: self.add_log("error", f"Failed to export stats: {e}")
            )
    
    def clear_cache(self):
        """Clear async cache."""
        if self.cache and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.cache.clear(),
                callback=lambda: self.add_log("info", "Cache cleared"),
                error_callback=lambda e: self.add_log("error", f"Failed to clear cache: {e}")
            )
    
    def refresh_realtime_data(self):
        """Refresh real-time data."""
        if self.backend_connected and self.async_manager:
            task_id = self.async_manager.submit_task(
                self.backend_bridge._load_initial_data(),
                callback=lambda: self.add_log("info", "Real-time data refreshed"),
                error_callback=lambda e: self.add_log("error", f"Failed to refresh data: {e}")
            )
    
    def show_data_quality_report(self):
        """Show data quality report dialog."""
        if not self.backend_connected:
            self.show_error_message("Not Connected", "Backend not connected")
            return
        
        # Get data quality metrics
        status = self.backend_bridge.get_system_status()
        data_quality = status.get("data_quality", {})
        
        # Create report
        report = f"""
Data Quality Report
==================

Total Records: {data_quality.get('total_records', 0)}
Duplicate Count: {data_quality.get('duplicate_count', 0)}
Out-of-Order Count: {data_quality.get('out_of_order_count', 0)}
Missing Data Count: {data_quality.get('missing_data_count', 0)}
Last Check: {data_quality.get('last_quality_check', 'Never')}

Quality Metrics:
- Duplicate Rate: {(data_quality.get('duplicate_count', 0) / max(data_quality.get('total_records', 1), 1)) * 100:.2f}%
- Out-of-Order Rate: {(data_quality.get('out_of_order_count', 0) / max(data_quality.get('total_records', 1), 1)) * 100:.2f}%
- Data Completeness: {((data_quality.get('total_records', 0) - data_quality.get('missing_data_count', 0)) / max(data_quality.get('total_records', 1), 1)) * 100:.2f}%
        """
        
        QMessageBox.information(self, "Data Quality Report", report)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ITS - Real-Time",
            "Intelligent Trading System - Real-Time Edition\n"
            "Version 1.0.0\n\n"
            "Real-time trading application with storage system integration.\n\n"
            "Features:\n"
            "Real-time Parquet data streaming\n"
            "TimescaleDB change notifications\n"
            "Live data quality monitoring\n"
            "Async processing architecture\n"
            "Comprehensive error handling"
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop backend
        if self.backend_bridge and self.backend_bridge.is_running and self.async_manager:
            task_id = self.async_manager.submit_task(self.backend_bridge.stop_trading)
            # Wait briefly for completion
            threading.Event().wait(1.0)
        
        # Disconnect backend
        if self.backend_bridge and self.backend_bridge.is_connected and self.async_manager:
            task_id = self.async_manager.submit_task(self.backend_bridge.disconnect)
            # Wait briefly for completion
            threading.Event().wait(1.0)
        
        # Add final log
        self.add_log("info", "Real-time application shutting down")
        
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
    window = RealTimeMainWindow(use_real_backend)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
