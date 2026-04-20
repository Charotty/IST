from __future__ import annotations

import datetime
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor


class LogEntry:
    """Log entry data structure."""
    
    def __init__(self, timestamp: str, level: str, message: str):
        self.timestamp = timestamp
        self.level = level
        self.message = message


class TradeEntry:
    """Trade entry data structure."""
    
    def __init__(self, trade_id: str, timestamp: str, pair: str, trade_type: str,
                 entry_price: float, exit_price: Optional[float] = None, pnl: Optional[float] = None):
        self.id = trade_id
        self.timestamp = timestamp
        self.pair = pair
        self.type = trade_type
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl = pnl


class LogWidget(QWidget):
    """Widget for displaying log entries."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.max_logs = 100
    
    def init_ui(self):
        """Initialize log widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #161B22;
                border: none;
                color: #C9D1D9;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        
        layout.addWidget(self.log_display)
    
    def add_log(self, timestamp: str, level: str, message: str):
        """Add log entry."""
        # Get color based on level
        colors = {
            "info": "#C9D1D9",
            "warning": "#F59E0B",
            "error": "#EF4444"
        }
        color = colors.get(level, "#C9D1D9")
        
        # Format log entry
        log_entry = f'<span style="color: #9CA3AF;">{timestamp}</span> ' \
                   f'<span style="color: {color};">[{level.upper()}]</span> ' \
                   f'<span style="color: #C9D1D9;">{message}</span>'
        
        # Add to display
        self.log_display.append(log_entry)
        
        # Limit number of logs
        if self.log_display.document().blockCount() > self.max_logs:
            # Remove oldest lines
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
        
        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear all logs."""
        self.log_display.clear()


class TradesWidget(QWidget):
    """Widget for displaying trade history."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize trades widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels([
            "Time", "Pair", "Type", "Entry", "Exit", "PnL"
        ])
        
        # Style table
        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: #161B22;
                border: none;
                gridline-color: #1F2933;
                selection-background-color: #3B82F6;
                selection-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #1F2933;
                color: #C9D1D9;
            }
            QTableWidget::item:selected {
                background-color: #3B82F6;
                color: white;
            }
            QHeaderView::section {
                background-color: #1F2933;
                border: none;
                border-bottom: 1px solid #1F2933;
                padding: 8px;
                color: #9CA3AF;
                font-weight: 500;
            }
        """)
        
        # Set column widths
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        self.trades_table.setColumnWidth(0, 80)   # Time
        self.trades_table.setColumnWidth(1, 80)   # Pair
        self.trades_table.setColumnWidth(2, 60)   # Type
        
        layout.addWidget(self.trades_table)
    
    def add_trade(self, trade_data: Dict):
        """Add trade entry."""
        row_position = self.trades_table.rowCount()
        self.trades_table.insertRow(row_position)
        
        # Time
        time_item = QTableWidgetItem(trade_data["timestamp"])
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.trades_table.setItem(row_position, 0, time_item)
        
        # Pair
        pair_item = QTableWidgetItem(trade_data["pair"])
        pair_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.trades_table.setItem(row_position, 1, pair_item)
        
        # Type
        type_item = QTableWidgetItem(trade_data["type"])
        type_color = "#22C55E" if trade_data["type"] == "BUY" else "#EF4444"
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.trades_table.setItem(row_position, 2, type_item)
        # Set color via table widget
        self.trades_table.item(row_position, 2).setForeground(QColor(type_color))
        
        # Entry price
        entry_price = trade_data["entry_price"]
        entry_item = QTableWidgetItem(f"${entry_price:.2f}")
        entry_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.trades_table.setItem(row_position, 3, entry_item)
        
        # Exit price
        exit_price = trade_data.get("exit_price")
        exit_item = QTableWidgetItem(f"${exit_price:.2f}" if exit_price else "-")
        exit_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.trades_table.setItem(row_position, 4, exit_item)
        
        # PnL
        pnl = trade_data.get("pnl")
        if pnl is not None:
            pnl_text = f"{pnl:+.2f}%"
            pnl_color = "#22C55E" if pnl >= 0 else "#EF4444"
            pnl_item = QTableWidgetItem(pnl_text)
            pnl_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.trades_table.setItem(row_position, 5, pnl_item)
            # Set color via table widget
            self.trades_table.item(row_position, 5).setForeground(QColor(pnl_color))
        else:
            pnl_item = QTableWidgetItem("-")
            pnl_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.trades_table.setItem(row_position, 5, pnl_item)
            # Set gray color for empty PnL
            self.trades_table.item(row_position, 5).setForeground(QColor("#9CA3AF"))
        
        # Scroll to bottom
        self.trades_table.scrollToBottom()
    
    def clear_trades(self):
        """Clear all trades."""
        self.trades_table.setRowCount(0)


class MetricsWidget(QWidget):
    """Widget for displaying trading metrics."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize metrics widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Metrics grid
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(16)
        
        # Create metric cards
        self.total_pnl_card = self.create_metric_card("Total PnL", "+0.00%", "#22C55E")
        self.win_rate_card = self.create_metric_card("Win Rate", "0.0%", "#C9D1D9")
        self.total_trades_card = self.create_metric_card("Total Trades", "0", "#C9D1D9")
        self.max_drawdown_card = self.create_metric_card("Max Drawdown", "-0.00%", "#EF4444")
        
        # Add cards to grid
        metrics_layout.addWidget(self.total_pnl_card, 0, 0)
        metrics_layout.addWidget(self.win_rate_card, 0, 1)
        metrics_layout.addWidget(self.total_trades_card, 1, 0)
        metrics_layout.addWidget(self.max_drawdown_card, 1, 1)
        
        layout.addLayout(metrics_layout)
        layout.addStretch()
    
    def create_metric_card(self, title: str, value: str, color: str) -> QFrame:
        """Create a metric card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1F2933;
                border: none;
                border-radius: 8px;
            }
        """)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(8)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        card_layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 600;
            color: {color};
        """)
        card_layout.addWidget(value_label)
        
        # Store reference for updates
        card.value_label = value_label
        
        return card
    
    def update_metrics(self, metrics: Dict):
        """Update all metrics."""
        # Total PnL
        total_pnl = metrics.get("totalPnL", 0)
        pnl_text = f"{total_pnl:+.2f}%"
        pnl_color = "#22C55E" if total_pnl >= 0 else "#EF4444"
        self.total_pnl_card.value_label.setText(pnl_text)
        self.total_pnl_card.value_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 600;
            color: {pnl_color};
        """)
        
        # Win Rate
        win_rate = metrics.get("winRate", 0)
        self.win_rate_card.value_label.setText(f"{win_rate:.1f}%")
        
        # Total Trades
        total_trades = metrics.get("totalTrades", 0)
        self.total_trades_card.value_label.setText(str(total_trades))
        
        # Max Drawdown
        max_drawdown = metrics.get("maxDrawdown", 0)
        self.max_drawdown_card.value_label.setText(f"-{max_drawdown:.2f}%")


class BottomPanel(QWidget):
    """Bottom panel with tabs for logs, trades, and metrics."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize bottom panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1F2933;
                background-color: #161B22;
            }
            QTabBar::tab {
                background-color: #1F2933;
                border: none;
                padding: 8px 16px;
                margin-right: 2px;
                color: #C9D1D9;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: #161B22;
                color: #C9D1D9;
                border-bottom: 2px solid #3B82F6;
            }
            QTabBar::tab:hover {
                background-color: #374151;
            }
        """)
        
        # Create tab widgets
        self.log_widget = LogWidget()
        self.trades_widget = TradesWidget()
        self.metrics_widget = MetricsWidget()
        
        # Add tabs
        self.tab_widget.addTab(self.log_widget, "Logs")
        self.tab_widget.addTab(self.trades_widget, "Trades")
        self.tab_widget.addTab(self.metrics_widget, "Metrics")
        
        layout.addWidget(self.tab_widget)
        
        # Initialize with demo data
        self.init_demo_data()
    
    def init_demo_data(self):
        """Initialize with demo data."""
        # Add demo logs
        demo_logs = [
            ("14:23:45", "info", "System initialized"),
            ("14:23:46", "info", "Connected to Binance API"),
            ("14:23:47", "info", "Models loaded successfully"),
            ("14:23:48", "warning", "High volatility detected"),
            ("14:23:49", "info", "Risk check passed"),
        ]
        
        for timestamp, level, message in demo_logs:
            self.log_widget.add_log(timestamp, level, message)
        
        # Add demo trades
        demo_trades = [
            {
                "id": "1",
                "timestamp": "14:15:32",
                "pair": "BTC/USDT",
                "type": "BUY",
                "entry_price": 45234.50,
                "exit_price": 45567.20,
                "pnl": 0.73
            },
            {
                "id": "2",
                "timestamp": "14:18:12",
                "pair": "ETH/USDT",
                "type": "SELL",
                "entry_price": 2534.80,
                "exit_price": 2498.30,
                "pnl": -1.44
            },
            {
                "id": "3",
                "timestamp": "14:22:05",
                "pair": "BTC/USDT",
                "type": "BUY",
                "entry_price": 45456.00,
                "exit_price": None,
                "pnl": None
            }
        ]
        
        for trade in demo_trades:
            self.trades_widget.add_trade(trade)
        
        # Set demo metrics
        demo_metrics = {
            "totalPnL": 2.34,
            "winRate": 67.5,
            "totalTrades": 156,
            "maxDrawdown": 4.23
        }
        
        self.metrics_widget.update_metrics(demo_metrics)
    
    def add_log(self, timestamp: str, level: str, message: str):
        """Add log entry."""
        self.log_widget.add_log(timestamp, level, message)
    
    def add_trade(self, trade_data: Dict):
        """Add trade entry."""
        self.trades_widget.add_trade(trade_data)
    
    def update_metrics(self, metrics: Dict):
        """Update trading metrics."""
        self.metrics_widget.update_metrics(metrics)
    
    def clear_logs(self):
        """Clear all logs."""
        self.log_widget.clear_logs()
    
    def clear_trades(self):
        """Clear all trades."""
        self.trades_widget.clear_trades()
    
    def get_current_tab(self) -> int:
        """Get current tab index."""
        return self.tab_widget.currentIndex()
    
    def set_tab(self, index: int):
        """Set current tab by index."""
        self.tab_widget.setCurrentIndex(index)
