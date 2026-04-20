from __future__ import annotations

import sys
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


class MainWindow(QMainWindow):
    """Main ITS trading application window with dark theme."""
    
    # Signals for real-time updates
    signal_updated = pyqtSignal(str, float, float)  # signal, confidence, predicted_change
    price_updated = pyqtSignal(str, float)  # symbol, price
    log_added = pyqtSignal(str, str, str)  # timestamp, level, message
    trade_executed = pyqtSignal(dict)  # trade data
    
    def __init__(self):
        super().__init__()
        
        # Apply dark theme
        self.setStyleSheet(DarkTheme.get_stylesheet())
        
        # Window properties
        self.setWindowTitle("Intelligent Trading System")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # System state
        self.system_status = "STOPPED"
        self.selected_pair = "BTC/USDT"
        self.trading_mode = "paper"  # paper or live
        
        # Initialize UI
        self.init_ui()
        
        # Setup timers for real-time simulation
        self.setup_timers()
        
        # Connect signals
        self.connect_signals()
    
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
        """Setup timers for real-time data simulation."""
        # Timer for signal updates
        self.signal_timer = QTimer()
        self.signal_timer.timeout.connect(self.simulate_signal_update)
        
        # Timer for price updates
        self.price_timer = QTimer()
        self.price_timer.timeout.connect(self.simulate_price_update)
        
        # Timer for log updates
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.simulate_log_update)
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.signal_updated.connect(self.on_signal_updated)
        self.price_updated.connect(self.on_price_updated)
        self.log_added.connect(self.on_log_added)
        self.trade_executed.connect(self.on_trade_executed)
    
    def on_pair_selected(self, pair: str):
        """Handle pair selection from sidebar."""
        self.selected_pair = pair
        self.main_chart.set_pair(pair)
        self.add_log("info", f"Selected trading pair: {pair}")
    
    def on_start_clicked(self):
        """Handle start button click."""
        if self.system_status == "STOPPED":
            self.system_status = "RUNNING"
            self.sidebar.set_system_status(self.system_status)
            
            # Start simulation timers
            self.signal_timer.start(3000)  # Update every 3 seconds
            self.price_timer.start(1000)   # Update every 1 second
            self.log_timer.start(2000)     # Update every 2 seconds
            
            self.add_log("info", "Trading system started")
            self.add_log("info", f"Mode: {self.trading_mode.upper()}")
            self.add_log("info", f"Pair: {self.selected_pair}")
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.system_status == "RUNNING":
            self.system_status = "STOPPED"
            self.sidebar.set_system_status(self.system_status)
            
            # Stop simulation timers
            self.signal_timer.stop()
            self.price_timer.stop()
            self.log_timer.stop()
            
            self.add_log("info", "Trading system stopped")
    
    def on_mode_changed(self, mode: str):
        """Handle trading mode change."""
        self.trading_mode = mode
        self.add_log("info", f"Trading mode changed to: {mode.upper()}")
    
    def on_signal_updated(self, signal: str, confidence: float, predicted_change: float):
        """Handle signal update."""
        self.signal_panel.update_signal(signal, confidence, predicted_change)
        
        # Add signal to chart
        self.main_chart.add_signal(signal)
        
        # Add log
        self.add_log("info", f"Signal: {signal}, Confidence: {confidence:.1%}, Change: {predicted_change:+.3f}%")
    
    def on_price_updated(self, symbol: str, price: float):
        """Handle price update."""
        self.main_chart.update_price(price)
    
    def on_log_added(self, timestamp: str, level: str, message: str):
        """Handle log addition."""
        self.bottom_panel.add_log(timestamp, level, message)
    
    def on_trade_executed(self, trade_data: dict):
        """Handle trade execution."""
        self.bottom_panel.add_trade(trade_data)
    
    def simulate_signal_update(self):
        """Simulate signal updates for demo purposes."""
        import random
        
        signals = ["BUY", "SELL", "HOLD"]
        signal = random.choice(signals)
        confidence = random.uniform(0.4, 0.9)
        predicted_change = random.uniform(-2.0, 2.0)
        
        self.signal_updated.emit(signal, confidence, predicted_change)
    
    def simulate_price_update(self):
        """Simulate price updates for demo purposes."""
        import random
        
        # Get base price for selected pair
        base_prices = {
            "BTC/USDT": 45000,
            "ETH/USDT": 2500,
            "BNB/USDT": 300,
            "SOL/USDT": 100,
            "XRP/USDT": 0.5,
            "ADA/USDT": 0.3
        }
        
        base_price = base_prices.get(self.selected_pair, 100)
        price_change = random.uniform(-0.02, 0.02)  # ±2% change
        new_price = base_price * (1 + price_change)
        
        self.price_updated.emit(self.selected_pair, new_price)
    
    def simulate_log_update(self):
        """Simulate log updates for demo purposes."""
        import random
        
        log_messages = [
            ("info", "Market data received"),
            ("info", "Model prediction updated"),
            ("info", "Risk check passed"),
            ("warning", "High volatility detected"),
            ("info", "Order book updated"),
            ("info", "Connection stable"),
        ]
        
        level, message = random.choice(log_messages)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.log_added.emit(timestamp, level, message)
    
    def add_log(self, level: str, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_added.emit(timestamp, level, message)
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop all timers
        self.signal_timer.stop()
        self.price_timer.stop()
        self.log_timer.stop()
        
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
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
