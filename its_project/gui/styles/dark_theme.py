"""Dark theme styling for ITS trading application."""

from typing import Dict


class DarkTheme:
    """Dark theme color palette and styles for ITS application."""
    
    # Color palette (GitHub-inspired dark theme)
    COLORS = {
        # Background colors
        "background": "#0D1117",
        "surface": "#161B22",
        "panel": "#1F2933",
        
        # Text colors
        "text_primary": "#C9D1D9",
        "text_secondary": "#9CA3AF",
        "text_muted": "#6B7280",
        
        # Accent colors
        "blue": "#3B82F6",
        "green": "#22C55E",
        "red": "#EF4444",
        "yellow": "#F59E0B",
        "orange": "#F97316",
        
        # Border colors
        "border": "#1F2933",
        "border_light": "#374151",
        
        # Status colors
        "status_running": "#22C55E",
        "status_stopped": "#9CA3AF",
        "status_error": "#EF4444",
        
        # Chart colors
        "chart_price": "#22C55E",
        "chart_prediction": "#3B82F6",
        "chart_grid": "#1F2933",
        "chart_signal_buy": "#22C55E",
        "chart_signal_sell": "#EF4444",
    }
    
    @classmethod
    def get_stylesheet(cls) -> str:
        """Generate complete stylesheet for dark theme."""
        return f"""
        /* QMainWindow and QWidget styles */
        QMainWindow {{
            background-color: {cls.COLORS["background"]};
            color: {cls.COLORS["text_primary"]};
        }}
        
        QWidget {{
            background-color: {cls.COLORS["background"]};
            color: {cls.COLORS["text_primary"]};
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            font-size: 12px;
        }}
        
        /* QPushButton styles */
        QPushButton {{
            background-color: {cls.COLORS["blue"]};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 12px;
        }}
        
        QPushButton:hover {{
            background-color: #2563EB;
        }}
        
        QPushButton:pressed {{
            background-color: #1D4ED8;
        }}
        
        QPushButton:disabled {{
            background-color: {cls.COLORS["panel"]};
            color: {cls.COLORS["text_muted"]};
        }}
        
        QPushButton.danger {{
            background-color: {cls.COLORS["red"]};
        }}
        
        QPushButton.danger:hover {{
            background-color: #DC2626;
        }}
        
        QPushButton.success {{
            background-color: {cls.COLORS["green"]};
        }}
        
        QPushButton.success:hover {{
            background-color: #16A34A;
        }}
        
        /* QLabel styles */
        QLabel {{
            color: {cls.COLORS["text_primary"]};
            font-size: 12px;
        }}
        
        QLabel.heading {{
            font-size: 18px;
            font-weight: 600;
            color: {cls.COLORS["text_primary"]};
        }}
        
        QLabel.subheading {{
            font-size: 14px;
            font-weight: 500;
            color: {cls.COLORS["text_primary"]};
        }}
        
        QLabel.caption {{
            font-size: 11px;
            color: {cls.COLORS["text_secondary"]};
        }}
        
        QLabel.muted {{
            color: {cls.COLORS["text_muted"]};
        }}
        
        QLabel.price {{
            font-size: 24px;
            font-weight: 600;
            color: {cls.COLORS["text_primary"]};
        }}
        
        QLabel.signal-buy {{
            color: {cls.COLORS["green"]};
            font-weight: 600;
        }}
        
        QLabel.signal-sell {{
            color: {cls.COLORS["red"]};
            font-weight: 600;
        }}
        
        QLabel.signal-hold {{
            color: {cls.COLORS["text_secondary"]};
            font-weight: 600;
        }}
        
        /* QFrame styles */
        QFrame {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 8px;
        }}
        
        QFrame.panel {{
            background-color: {cls.COLORS["panel"]};
            border: 1px solid {cls.COLORS["border"]};
        }}
        
        QFrame.separator {{
            background-color: {cls.COLORS["border"]};
            border: none;
            max-height: 1px;
        }}
        
        /* QGroupBox styles */
        QGroupBox {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 8px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: 500;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            color: {cls.COLORS["text_secondary"]};
            font-size: 11px;
        }}
        
        /* QProgressBar styles */
        QProgressBar {{
            background-color: {cls.COLORS["panel"]};
            border: none;
            border-radius: 4px;
            height: 8px;
        }}
        
        QProgressBar::chunk {{
            background-color: {cls.COLORS["blue"]};
            border-radius: 4px;
        }}
        
        QProgressBar.high {{
            background-color: {cls.COLORS["green"]};
        }}
        
        QProgressBar.medium {{
            background-color: {cls.COLORS["yellow"]};
        }}
        
        QProgressBar.low {{
            background-color: {cls.COLORS["text_muted"]};
        }}
        
        /* QComboBox styles */
        QComboBox {{
            background-color: {cls.COLORS["panel"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 6px;
            padding: 6px 12px;
            color: {cls.COLORS["text_primary"]};
        }}
        
        QComboBox:hover {{
            border-color: {cls.COLORS["border_light"]};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid {cls.COLORS["text_secondary"]};
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            selection-background-color: {cls.COLORS["blue"]};
            selection-color: white;
        }}
        
        /* QTabWidget styles */
        QTabWidget::pane {{
            border: 1px solid {cls.COLORS["border"]};
            background-color: {cls.COLORS["surface"]};
        }}
        
        QTabBar::tab {{
            background-color: {cls.COLORS["panel"]};
            border: none;
            padding: 8px 16px;
            margin-right: 2px;
            color: {cls.COLORS["text_secondary"]};
        }}
        
        QTabBar::tab:selected {{
            background-color: {cls.COLORS["surface"]};
            color: {cls.COLORS["text_primary"]};
            border-bottom: 2px solid {cls.COLORS["blue"]};
        }}
        
        QTabBar::tab:hover {{
            background-color: {cls.COLORS["border_light"]};
        }}
        
        /* QTableWidget styles */
        QTableWidget {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            gridline-color: {cls.COLORS["border"]};
            selection-background-color: {cls.COLORS["blue"]};
            selection-color: white;
        }}
        
        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {cls.COLORS["border"]};
        }}
        
        QTableWidget::item:selected {{
            background-color: {cls.COLORS["blue"]};
            color: white;
        }}
        
        QHeaderView::section {{
            background-color: {cls.COLORS["panel"]};
            border: none;
            border-bottom: 1px solid {cls.COLORS["border"]};
            padding: 8px;
            color: {cls.COLORS["text_secondary"]};
            font-weight: 500;
        }}
        
        /* QTextEdit styles */
        QTextEdit {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            color: {cls.COLORS["text_primary"]};
            padding: 8px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
        }}
        
        /* QScrollArea styles */
        QScrollArea {{
            background-color: {cls.COLORS["surface"]};
            border: none;
        }}
        
        QScrollBar:vertical {{
            background-color: {cls.COLORS["panel"]};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {cls.COLORS["border_light"]};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {cls.COLORS["text_muted"]};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        /* Status indicator styles */
        .status-indicator {{
            width: 8px;
            height: 8px;
            border-radius: 4px;
        }}
        
        .status-running {{
            background-color: {cls.COLORS["status_running"]};
        }}
        
        .status-stopped {{
            background-color: {cls.COLORS["status_stopped"]};
        }}
        
        .status-error {{
            background-color: {cls.COLORS["status_error"]};
        }}
        
        /* Custom widget styles */
        .signal-panel {{
            background-color: {cls.COLORS["surface"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 8px;
            padding: 16px;
        }}
        
        .model-card {{
            background-color: {cls.COLORS["panel"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 6px;
            padding: 12px;
            margin: 4px 0;
        }}
        
        .model-card.active {{
            border-color: {cls.COLORS["blue"]};
            background-color: rgba(59, 130, 246, 0.1);
        }}
        
        .metric-card {{
            background-color: {cls.COLORS["panel"]};
            border: 1px solid {cls.COLORS["border"]};
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }}
        
        .log-entry {{
            padding: 4px 8px;
            border-bottom: 1px solid {cls.COLORS["border"]};
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
        }}
        
        .log-entry.info {{
            color: {cls.COLORS["text_primary"]};
        }}
        
        .log-entry.warning {{
            color: {cls.COLORS["yellow"]};
        }}
        
        .log-entry.error {{
            color: {cls.COLORS["red"]};
        }}
        
        .trade-row {{
            padding: 8px;
            border-bottom: 1px solid {cls.COLORS["border"]};
        }}
        
        .trade-buy {{
            color: {cls.COLORS["green"]};
        }}
        
        .trade-sell {{
            color: {cls.COLORS["red"]};
        }}
        
        .trade-profit {{
            color: {cls.COLORS["green"]};
        }}
        
        .trade-loss {{
            color: {cls.COLORS["red"]};
        }}
        """
