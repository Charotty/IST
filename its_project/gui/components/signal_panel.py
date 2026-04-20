from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPainterPath, QPen, QBrush


class SignalIndicator(QWidget):
    """Custom signal indicator with icon."""
    
    def __init__(self):
        super().__init__()
        self.signal = "HOLD"
        self.setFixedSize(120, 60)
    
    def set_signal(self, signal: str):
        """Update signal and repaint."""
        self.signal = signal
        self.update()
    
    def paintEvent(self, event):
        """Paint signal indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#1F2933"))
        
        # Signal color
        colors = {
            "BUY": QColor("#22C55E"),
            "SELL": QColor("#EF4444"),
            "HOLD": QColor("#9CA3AF")
        }
        
        color = colors.get(self.signal, QColor("#9CA3AF"))
        
        # Draw signal icon (arrow)
        painter.setPen(QPen(color, 3))
        painter.setBrush(QBrush(color))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        if self.signal == "BUY":
            # Up arrow
            path = QPainterPath()
            path.moveTo(center_x, center_y + 15)
            path.lineTo(center_x - 10, center_y)
            path.lineTo(center_x - 5, center_y)
            path.lineTo(center_x - 5, center_y - 10)
            path.lineTo(center_x + 5, center_y - 10)
            path.lineTo(center_x + 5, center_y)
            path.lineTo(center_x + 10, center_y)
            path.lineTo(center_x, center_y + 15)
            painter.drawPath(path)
        elif self.signal == "SELL":
            # Down arrow
            path = QPainterPath()
            path.moveTo(center_x, center_y - 15)
            path.lineTo(center_x - 10, center_y)
            path.lineTo(center_x - 5, center_y)
            path.lineTo(center_x - 5, center_y + 10)
            path.lineTo(center_x + 5, center_y + 10)
            path.lineTo(center_x + 5, center_y)
            path.lineTo(center_x + 10, center_y)
            path.lineTo(center_x, center_y - 15)
            painter.drawPath(path)
        else:
            # Horizontal line for HOLD
            painter.drawLine(center_x - 15, center_y, center_x + 15, center_y)
        
        # Signal text
        painter.setPen(color)
        font = QFont()
        font.setPointSize(12)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        
        text_rect = self.rect().adjusted(0, self.height() - 20, 0, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.signal)


class ConfidenceBar(QProgressBar):
    """Custom confidence progress bar."""
    
    def __init__(self):
        super().__init__()
        self.setFixedHeight(8)
        self.setRange(0, 100)
        
    def paintEvent(self, event):
        """Paint custom confidence bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#1F2933"))
        
        # Get value
        value = self.value()
        if value <= 0:
            return
        
        # Determine color based on confidence
        if value > 70:
            color = QColor("#22C55E")
        elif value > 40:
            color = QColor("#F59E0B")
        else:
            color = QColor("#9CA3AF")
        
        # Draw progress
        progress_width = (self.width() * value) / 100
        progress_rect = self.rect().adjusted(0, 0, int(self.width() - progress_width), 0)
        
        painter.fillRect(progress_rect, color)


class SignalPanel(QWidget):
    """Signal panel with current signal, confidence, and predictions."""
    
    def __init__(self):
        super().__init__()
        self.signal = "HOLD"
        self.confidence = 0.65
        self.predicted_change = 0.0
        self.current_model = "GRU-LSTM"
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize signal panel UI."""
        self.setFixedWidth(280)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Current signal section
        self.create_signal_section(layout)
        
        # Metrics section
        self.create_metrics_section(layout)
        
        # Risk warning (initially hidden)
        self.risk_warning = self.create_risk_warning()
        layout.addWidget(self.risk_warning)
        self.risk_warning.hide()
    
    def create_signal_section(self, layout):
        """Create current signal display."""
        signal_frame = QFrame()
        signal_frame.setStyleSheet("""
            QFrame {
                background-color: #1F2933;
                border: none;
                border-radius: 8px;
            }
        """)
        
        signal_layout = QVBoxLayout(signal_frame)
        signal_layout.setContentsMargins(16, 16, 16, 16)
        signal_layout.setSpacing(12)
        
        # Title
        title_label = QLabel("Current Signal")
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        signal_layout.addWidget(title_label)
        
        # Signal display
        signal_display = QHBoxLayout()
        
        self.signal_indicator = SignalIndicator()
        
        self.signal_label = QLabel(self.signal)
        self.signal_label.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
        """)
        
        signal_display.addWidget(self.signal_indicator)
        signal_display.addWidget(self.signal_label)
        signal_display.addStretch()
        
        signal_layout.addLayout(signal_display)
        layout.addWidget(signal_frame)
    
    def create_metrics_section(self, layout):
        """Create metrics display section."""
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: none;
                border-bottom: 1px solid #1F2933;
            }
        """)
        
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(24)
        
        # Predicted change
        self.create_predicted_change_section(metrics_layout)
        
        # Confidence
        self.create_confidence_section(metrics_layout)
        
        # Active model
        self.create_model_section(metrics_layout)
        
        layout.addWidget(metrics_frame)
    
    def create_predicted_change_section(self, layout):
        """Create predicted change display."""
        change_layout = QVBoxLayout()
        change_layout.setSpacing(8)
        
        # Title
        title_label = QLabel("Predicted Change (P_t)")
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        change_layout.addWidget(title_label)
        
        # Value
        self.change_label = QLabel("+0.000%")
        self.change_label.setStyleSheet("""
            font-size: 20px;
            font-weight: 500;
        """)
        change_layout.addWidget(self.change_label)
        
        layout.addLayout(change_layout)
    
    def create_confidence_section(self, layout):
        """Create confidence display."""
        confidence_layout = QVBoxLayout()
        confidence_layout.setSpacing(8)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Confidence (p_t)")
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        self.confidence_value_label = QLabel("65.0%")
        self.confidence_value_label.setStyleSheet("color: #C9D1D9; font-size: 14px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.confidence_value_label)
        
        confidence_layout.addLayout(header_layout)
        
        # Progress bar
        self.confidence_bar = ConfidenceBar()
        confidence_layout.addWidget(self.confidence_bar)
        
        layout.addLayout(confidence_layout)
    
    def create_model_section(self, layout):
        """Create active model display."""
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        
        # Title
        title_label = QLabel("Active Model")
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        model_layout.addWidget(title_label)
        
        # Model name
        self.model_label = QLabel(self.current_model)
        self.model_label.setStyleSheet("""
            background-color: #3B82F6;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        """)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_layout.addWidget(self.model_label)
        
        layout.addLayout(model_layout)
    
    def create_risk_warning(self) -> QFrame:
        """Create risk warning widget."""
        warning_frame = QFrame()
        warning_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(245, 158, 11, 0.1);
                border: none;
                border-left: 3px solid #F59E0B;
                border-radius: 4px;
            }
        """)
        
        warning_layout = QVBoxLayout(warning_frame)
        warning_layout.setContentsMargins(12, 8, 12, 8)
        
        warning_label = QLabel("Low confidence signal. Consider manual review.")
        warning_label.setStyleSheet("color: #F59E0B; font-size: 11px;")
        warning_label.setWordWrap(True)
        
        warning_layout.addWidget(warning_label)
        
        return warning_frame
    
    def update_signal(self, signal: str, confidence: float, predicted_change: float):
        """Update signal display."""
        self.signal = signal
        self.confidence = confidence
        self.predicted_change = predicted_change
        
        # Update signal indicator
        self.signal_indicator.set_signal(signal)
        
        # Update signal label
        self.signal_label.setText(signal)
        
        # Update signal label color
        colors = {
            "BUY": "#22C55E",
            "SELL": "#EF4444",
            "HOLD": "#9CA3AF"
        }
        color = colors.get(signal, "#9CA3AF")
        self.signal_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {color};
        """)
        
        # Update predicted change
        change_text = f"{predicted_change:+.3f}%"
        self.change_label.setText(change_text)
        
        change_color = "#22C55E" if predicted_change >= 0 else "#EF4444"
        self.change_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 500;
            color: {change_color};
        """)
        
        # Update confidence
        confidence_percent = confidence * 100
        self.confidence_value_label.setText(f"{confidence_percent:.1f}%")
        self.confidence_bar.setValue(int(confidence_percent))
        
        # Update confidence bar color
        self.confidence_bar.update()
        
        # Show/hide risk warning
        if confidence < 0.5:
            self.risk_warning.show()
        else:
            self.risk_warning.hide()
    
    def set_current_model(self, model_name: str):
        """Update current model display."""
        self.current_model = model_name
        self.model_label.setText(model_name)
    
    def get_current_signal(self) -> str:
        """Get current signal."""
        return self.signal
    
    def get_confidence(self) -> float:
        """Get current confidence."""
        return self.confidence
    
    def get_predicted_change(self) -> float:
        """Get predicted change."""
        return self.predicted_change
