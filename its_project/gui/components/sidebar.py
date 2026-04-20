from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QButtonGroup, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPainter, QColor


class StatusIndicator(QWidget):
    """Custom status indicator widget."""
    
    def __init__(self, status: str = "STOPPED"):
        super().__init__()
        self.status = status
        self.setFixedSize(8, 8)
    
    def set_status(self, status: str):
        """Update status and repaint."""
        self.status = status
        self.update()
    
    def paintEvent(self, event):
        """Paint status indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set color based on status
        colors = {
            "RUNNING": QColor("#22C55E"),
            "STOPPED": QColor("#9CA3AF"),
            "ERROR": QColor("#EF4444")
        }
        
        color = colors.get(self.status, QColor("#9CA3AF"))
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Draw circle
        painter.drawEllipse(0, 0, self.width(), self.height())


class TradingPairButton(QPushButton):
    """Custom button for trading pairs."""
    
    def __init__(self, pair: str, is_selected: bool = False):
        super().__init__(pair)
        self.pair = pair
        self.is_selected = is_selected
        self.setCheckable(True)
        self.setChecked(is_selected)
        self.setFixedHeight(32)
    
    def paintEvent(self, event):
        """Custom paint for selected state."""
        from PyQt6.QtGui import QPainter, QColor, QFontMetrics
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        if self.isChecked():
            painter.fillRect(self.rect(), QColor("#3B82F6"))
            text_color = QColor("#FFFFFF")
        else:
            painter.fillRect(self.rect(), QColor("transparent"))
            text_color = QColor("#C9D1D9")
        
        # Text
        painter.setPen(text_color)
        font = self.font()
        font.setPointSize(11)
        painter.setFont(font)
        
        fm = QFontMetrics(font)
        text_rect = self.rect().adjusted(12, 0, -12, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self.pair)


class ModeToggle(QWidget):
    """Custom toggle for paper/live mode."""
    
    mode_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.current_mode = "paper"
        self.init_ui()
    
    def init_ui(self):
        """Initialize mode toggle UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        
        # Container widget with background
        self.container = QWidget()
        self.container.setFixedHeight(28)
        self.container.setStyleSheet("""
            QWidget {
                background-color: #0D1117;
                border-radius: 4px;
            }
        """)
        
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Paper button
        self.paper_btn = QPushButton("Paper")
        self.paper_btn.setCheckable(True)
        self.paper_btn.setChecked(True)
        self.paper_btn.clicked.connect(lambda: self.set_mode("paper"))
        
        # Live button
        self.live_btn = QPushButton("Live")
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(lambda: self.set_mode("live"))
        
        # Style buttons
        button_style = """
            QPushButton {
                border: none;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 500;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #3B82F6;
                color: white;
            }
            QPushButton:!checked {
                background-color: transparent;
                color: #9CA3AF;
            }
        """
        
        self.paper_btn.setStyleSheet(button_style)
        self.live_btn.setStyleSheet(button_style)
        
        container_layout.addWidget(self.paper_btn)
        container_layout.addWidget(self.live_btn)
        
        layout.addWidget(self.container)
    
    def set_mode(self, mode: str):
        """Set current mode."""
        if mode != self.current_mode:
            self.current_mode = mode
            self.paper_btn.setChecked(mode == "paper")
            self.live_btn.setChecked(mode == "live")
            self.mode_changed.emit(mode)


class Sidebar(QWidget):
    """Sidebar with system controls and trading pairs."""
    
    # Signals
    pair_selected = pyqtSignal(str)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize sidebar UI."""
        self.setFixedWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Status section
        self.create_status_section(layout)
        
        # Control buttons section
        self.create_control_section(layout)
        
        # Mode toggle section
        self.create_mode_section(layout)
        
        # Trading pairs section
        self.create_pairs_section(layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
    
    def create_status_section(self, layout):
        """Create status indicator section."""
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
                border-bottom: 1px solid #1F2933;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(16, 16, 16, 16)
        
        # Status indicator
        self.status_indicator = StatusIndicator("STOPPED")
        
        # Status label
        self.status_label = QLabel("STOPPED")
        self.status_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addWidget(status_frame)
    
    def create_control_section(self, layout):
        """Create control buttons section."""
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
                border-bottom: 1px solid #1F2933;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(8)
        
        # Start button
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_clicked)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #22C55E;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #16A34A;
            }
            QPushButton:disabled {
                background-color: #1F2933;
                color: #6B7280;
            }
        """)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:disabled {
                background-color: #1F2933;
                color: #6B7280;
            }
        """)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        layout.addWidget(control_frame)
    
    def create_mode_section(self, layout):
        """Create mode toggle section."""
        mode_frame = QFrame()
        mode_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
                border-bottom: 1px solid #1F2933;
            }
        """)
        
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(16, 16, 16, 16)
        mode_layout.setSpacing(8)
        
        # Mode label
        mode_label = QLabel("Mode")
        mode_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        # Mode toggle
        self.mode_toggle = ModeToggle()
        self.mode_toggle.mode_changed.connect(self.mode_changed)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_toggle)
        
        layout.addWidget(mode_frame)
    
    def create_pairs_section(self, layout):
        """Create trading pairs section."""
        pairs_frame = QFrame()
        pairs_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
            }
        """)
        
        pairs_layout = QVBoxLayout(pairs_frame)
        pairs_layout.setContentsMargins(16, 16, 16, 16)
        pairs_layout.setSpacing(8)
        
        # Pairs label
        pairs_label = QLabel("Trading Pairs")
        pairs_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        # Scroll area for pairs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1F2933;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #374151;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        
        # Pairs container
        pairs_container = QWidget()
        pairs_container_layout = QVBoxLayout(pairs_container)
        pairs_container_layout.setContentsMargins(0, 0, 0, 0)
        pairs_container_layout.setSpacing(2)
        
        # Trading pairs
        trading_pairs = [
            "BTC/USDT",
            "ETH/USDT", 
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT"
        ]
        
        self.pair_buttons = []
        for i, pair in enumerate(trading_pairs):
            is_selected = (i == 0)  # Select first pair by default
            btn = TradingPairButton(pair, is_selected)
            btn.clicked.connect(lambda checked, p=pair: self.on_pair_clicked(p))
            self.pair_buttons.append(btn)
            pairs_container_layout.addWidget(btn)
        
        scroll_area.setWidget(pairs_container)
        
        pairs_layout.addWidget(pairs_label)
        pairs_layout.addWidget(scroll_area)
        
        layout.addWidget(pairs_frame)
    
    def on_pair_clicked(self, pair: str):
        """Handle pair selection."""
        # Update button states
        for btn in self.pair_buttons:
            btn.setChecked(btn.pair == pair)
        
        # Emit signal
        self.pair_selected.emit(pair)
    
    def set_system_status(self, status: str):
        """Update system status."""
        self.status_indicator.set_status(status)
        self.status_label.setText(status)
        
        # Update button states
        if status == "RUNNING":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
        elif status == "STOPPED":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        elif status == "ERROR":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setStyleSheet("color: #EF4444; font-size: 11px;")
    
    def get_selected_pair(self) -> str:
        """Get currently selected trading pair."""
        for btn in self.pair_buttons:
            if btn.isChecked():
                return btn.pair
        return "BTC/USDT"  # Default
    
    def get_mode(self) -> str:
        """Get current trading mode."""
        return self.mode_toggle.current_mode
