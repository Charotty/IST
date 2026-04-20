from __future__ import annotations

import sys
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QFrame, QTabWidget,
    QTextEdit, QTableWidget, QTableWidgetItem, QProgressBar,
    QComboBox, QGroupBox, QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QPen, QPixmap

from components.sidebar import Sidebar
from components.main_chart import MainChart
from components.signal_panel import SignalPanel
from components.model_panel import ModelPanel
from components.bottom_panel import BottomPanel
from styles.dark_theme import DarkTheme
from backend_bridge import ITSBackendBridge, GUISignal, GUITrade, GUIMetrics


class AsyncThread(QThread):
    """Thread for running async operations in PyQt."""
    
    def __init__(self, coro, *args, **kwargs):
        super().__init__()
        self.coro = coro
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
    
    def run(self):
        """Run the coroutine in new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.result = loop.run_until_complete(self.coro(*self.args, **self.kwargs))
        except Exception as e:
            self.error = e
        finally:
            loop.close()


class IntegratedMainWindow(QMainWindow):
    """Integrated ITS trading application with backend connection."""
    
    # Signals for real-time updates
    signal_updated = pyqtSignal(str, float, float)  # signal, confidence, predicted_change
    price_updated = pyqtSignal(str, float)  # symbol, price
    log_added = pyqtSignal(str, str, str)  # timestamp, level, message
    trade_executed = pyqtSignal(dict)  # trade data
    metrics_updated = pyqtSignal(dict)  # metrics data
    models_updated = pyqtSignal(list)  # model data
    
    def __init__(self):
        super().__init__()
        
        # Apply dark theme
        self.setStyleSheet(DarkTheme.get_stylesheet())
        
        # Window properties
        self.setWindowTitle("Intelligent Trading System - Integrated")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # System state
        self.system_status = "STOPPED"
        self.selected_pair = "BTC/USDT"
        self.trading_mode = "paper"
        
        # Backend bridge
        self.backend_bridge = ITSBackendBridge()
        self.async_thread = None
        
        # Initialize UI
        self.init_ui()
        
        # Setup timers for GUI updates
        self.setup_timers()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize backend connection
        self.initialize_backend()
    
    def init_ui(self):
        """Initialize the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
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
    
    def setup_timers(self):
        """Setup timers for GUI operations."""
        # Backend status check timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_backend_status)
        self.status_timer.start(5000)  # Check every 5 seconds
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.signal_updated.connect(self.on_signal_updated)
        self.price_updated.connect(self.on_price_updated)
        self.log_added.connect(self.on_log_added)
        self.trade_executed.connect(self.on_trade_executed)
        self.metrics_updated.connect(self.on_metrics_updated)
        self.models_updated.connect(self.on_models_updated)
    
    def initialize_backend(self):
        """Initialize backend connection."""
        # Set GUI callbacks for backend bridge
        self.backend_bridge.set_gui_callbacks(
            signal_callback=self.on_backend_signal,
            price_callback=self.on_backend_price,
            log_callback=self.on_backend_log,
            trade_callback=self.on_backend_trade,
            metrics_callback=self.on_backend_metrics,
            model_callback=self.on_backend_models
        )
        
        # Start backend connection in thread
        self.async_thread = AsyncThread(self.backend_bridge.connect)
        self.async_thread.finished.connect(self.on_backend_connected)
        self.async_thread.start()
        
        self.add_log("info", "Connecting to backend systems...")
    
    def on_backend_connected(self):
        """Handle backend connection completion."""
        if self.async_thread.error:
            self.add_log("error", f"Failed to connect to backend: {self.async_thread.error}")
        else:
            connected = self.async_thread.result
            if connected:
                self.add_log("info", "Backend connected successfully")
                status = self.backend_bridge.get_system_status()
                self.add_log("info", f"Mode: {'Demo' if status['demo_mode'] else 'Production'}")
            else:
                self.add_log("warning", "Backend connected in demo mode")
    
    def check_backend_status(self):
        """Periodically check backend status."""
        if self.backend_bridge.is_connected:
            status = self.backend_bridge.get_system_status()
            
            # Update status indicator if needed
            if self.system_status == "RUNNING" and not status["running"]:
                # Backend stopped unexpectedly
                self.system_status = "ERROR"
                self.sidebar.set_system_status(self.system_status)
                self.add_log("error", "Backend stopped unexpectedly")
    
    def on_pair_selected(self, pair: str):
        """Handle pair selection from sidebar."""
        self.selected_pair = pair
        self.main_chart.set_pair(pair)
        self.add_log("info", f"Selected trading pair: {pair}")
        
        # Update backend if running
        if self.backend_bridge.is_running:
            # Restart trading with new pair
            self.async_thread = AsyncThread(self.backend_bridge.stop_trading)
            self.async_thread.finished.connect(lambda: self.restart_with_new_pair(pair))
            self.async_thread.start()
    
    def restart_with_new_pair(self, pair: str):
        """Restart trading with new pair."""
        self.async_thread = AsyncThread(
            self.backend_bridge.start_trading, 
            pair, 
            self.trading_mode
        )
        self.async_thread.start()
    
    def on_start_clicked(self):
        """Handle start button click."""
        if self.system_status == "STOPPED":
            self.system_status = "RUNNING"
            self.sidebar.set_system_status(self.system_status)
            
            # Start backend trading
            self.async_thread = AsyncThread(
                self.backend_bridge.start_trading,
                self.selected_pair,
                self.trading_mode
            )
            self.async_thread.finished.connect(self.on_trading_started)
            self.async_thread.start()
            
            self.add_log("info", "Starting trading system...")
    
    def on_trading_started(self):
        """Handle trading start completion."""
        if self.async_thread.error:
            self.add_log("error", f"Failed to start trading: {self.async_thread.error}")
            self.system_status = "ERROR"
            self.sidebar.set_system_status(self.system_status)
        else:
            started = self.async_thread.result
            if started:
                self.add_log("info", f"Trading system started - {self.selected_pair}")
                self.add_log("info", f"Mode: {self.trading_mode.upper()}")
            else:
                self.add_log("error", "Failed to start trading")
                self.system_status = "ERROR"
                self.sidebar.set_system_status(self.system_status)
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.system_status == "RUNNING":
            self.system_status = "STOPPED"
            self.sidebar.set_system_status(self.system_status)
            
            # Stop backend trading
            self.async_thread = AsyncThread(self.backend_bridge.stop_trading)
            self.async_thread.finished.connect(self.on_trading_stopped)
            self.async_thread.start()
            
            self.add_log("info", "Stopping trading system...")
    
    def on_trading_stopped(self):
        """Handle trading stop completion."""
        if self.async_thread.error:
            self.add_log("error", f"Error stopping trading: {self.async_thread.error}")
        else:
            self.add_log("info", "Trading system stopped")
    
    def on_mode_changed(self, mode: str):
        """Handle trading mode change."""
        self.trading_mode = mode
        self.add_log("info", f"Trading mode changed to: {mode.upper()}")
        
        # Restart trading if running
        if self.system_status == "RUNNING":
            self.async_thread = AsyncThread(self.backend_bridge.stop_trading)
            self.async_thread.finished.connect(lambda: self.restart_with_mode(mode))
            self.async_thread.start()
    
    def restart_with_mode(self, mode: str):
        """Restart trading with new mode."""
        self.async_thread = AsyncThread(
            self.backend_bridge.start_trading,
            self.selected_pair,
            mode
        )
        self.async_thread.start()
    
    def on_model_selected(self, model_name: str):
        """Handle model selection from model panel."""
        if self.backend_bridge.is_connected:
            self.async_thread = AsyncThread(self.backend_bridge.switch_model, model_name)
            self.async_thread.start()
    
    # Backend callback handlers
    def on_backend_signal(self, signal: GUISignal):
        """Handle signal from backend."""
        self.signal_updated.emit(signal.signal, signal.confidence, signal.predicted_change)
    
    def on_backend_price(self, symbol: str, price: float):
        """Handle price update from backend."""
        self.price_updated.emit(symbol, price)
    
    def on_backend_log(self, timestamp: str, level: str, message: str):
        """Handle log from backend."""
        self.log_added.emit(timestamp, level, message)
    
    def on_backend_trade(self, trade: GUITrade):
        """Handle trade from backend."""
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
        
        # Add signal to chart
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
    
    def add_log(self, level: str, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_added.emit(timestamp, level, message)
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop backend
        if self.backend_bridge.is_running:
            self.async_thread = AsyncThread(self.backend_bridge.stop_trading)
            self.async_thread.wait()
        
        # Disconnect backend
        if self.backend_bridge.is_connected:
            self.async_thread = AsyncThread(self.backend_bridge.disconnect)
            self.async_thread.wait()
        
        # Add final log
        self.add_log("info", "Application shutting down")
        
        # Accept close event
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Intelligent Trading System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ITS")
    
    # Create and show main window
    window = IntegratedMainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
