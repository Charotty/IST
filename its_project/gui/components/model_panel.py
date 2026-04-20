from __future__ import annotations

from typing import List, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont


class ModelCard(QFrame):
    """Card widget for displaying model information."""
    
    def __init__(self, model_data: Dict):
        super().__init__()
        self.model_data = model_data
        self.init_ui()
    
    def init_ui(self):
        """Initialize model card UI."""
        self.setFixedHeight(80)
        
        # Style based on active status
        if self.model_data.get("is_active", False):
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(59, 130, 246, 0.1);
                    border: 1px solid #3B82F6;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #1F2933;
                    border: 1px solid transparent;
                    border-radius: 6px;
                }
            """)
        
        # Check if layout already exists to avoid warnings
        if self.layout() is None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)
        else:
            layout = self.layout()
        
        # Clear existing widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Header with icon and name
        header_layout = QHBoxLayout()
        
        # Model icon (simplified text representation)
        icon_label = self.get_model_icon()
        icon_label.setFixedSize(16, 16)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Model name
        name_label = QLabel(self.model_data["name"])
        name_color = "#3B82F6" if self.model_data.get("is_active", False) else "#C9D1D9"
        name_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 500;
            color: {name_color};
        """)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(name_label)
        
        # Active badge
        if self.model_data.get("is_active", False):
            active_badge = QLabel("ACTIVE")
            active_badge.setStyleSheet("""
                background-color: #3B82F6;
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: 500;
            """)
            header_layout.addWidget(active_badge)
        
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Score display
        score_layout = QHBoxLayout()
        
        score_label = QLabel("Score")
        score_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        
        score_value = QLabel(f"{self.model_data['score'] * 100:.1f}%")
        score_color = self.get_score_color(self.model_data["score"])
        score_value.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 500;
            color: {score_color};
        """)
        
        score_layout.addWidget(score_label)
        score_layout.addStretch()
        score_layout.addWidget(score_value)
        
        layout.addLayout(score_layout)
    
    def get_model_icon(self) -> QLabel:
        """Get model icon based on type."""
        icons = {
            "GRU": "G",
            "Transformer": "T", 
            "CNN": "C"
        }
        
        icon_text = icons.get(self.model_data.get("type", "GRU"), "M")
        
        icon_label = QLabel(icon_text)
        icon_color = "#3B82F6" if self.model_data.get("is_active", False) else "#9CA3AF"
        icon_label.setStyleSheet(f"""
            background-color: {icon_color};
            color: white;
            border-radius: 8px;
            font-size: 10px;
            font-weight: 600;
        """)
        
        return icon_label
    
    def get_score_color(self, score: float) -> str:
        """Get color based on score value."""
        if score > 0.7:
            return "#22C55E"
        elif score > 0.5:
            return "#F59E0B"
        else:
            return "#EF4444"
    
    def update_active_status(self, is_active: bool):
        """Update active status and restyle."""
        self.model_data["is_active"] = is_active
        self.init_ui()


class ModelPanel(QWidget):
    """Panel displaying model performance information."""
    
    model_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.models = []
        self.model_cards = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize model panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)
        
        # Title
        title_label = QLabel("Model Performance")
        title_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        layout.addWidget(title_label)
        
        # Add spacing
        layout.addSpacing(16)
        
        # Scroll area for models
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
        
        # Models container
        self.models_container = QWidget()
        self.models_layout = QVBoxLayout(self.models_container)
        self.models_layout.setContentsMargins(0, 0, 0, 0)
        self.models_layout.setSpacing(12)
        
        scroll_area.setWidget(self.models_container)
        layout.addWidget(scroll_area)
        
        # Initialize with default models
        self.set_default_models()
    
    def set_default_models(self):
        """Set default model data."""
        default_models = [
            {
                "name": "GRU-LSTM",
                "type": "GRU",
                "score": 0.82,
                "is_active": True
            },
            {
                "name": "Transformer",
                "type": "Transformer", 
                "score": 0.78,
                "is_active": False
            },
            {
                "name": "CNN-Attention",
                "type": "CNN",
                "score": 0.74,
                "is_active": False
            }
        ]
        
        self.set_models(default_models)
    
    def set_models(self, models: List[Dict]):
        """Set models data and update display."""
        self.models = models
        
        # Clear existing cards
        for card in self.model_cards:
            card.deleteLater()
        self.model_cards.clear()
        
        # Create new cards
        for model_data in models:
            card = ModelCard(model_data)
            # Make card clickable
            card.mousePressEvent = lambda event, m=model_data: self.on_model_clicked(m)
            self.models_layout.addWidget(card)
            self.model_cards.append(card)
    
    def on_model_clicked(self, model_data: Dict):
        """Handle model card click."""
        # Update active status
        for i, model in enumerate(self.models):
            model["is_active"] = (model["name"] == model_data["name"])
            self.model_cards[i].update_active_status(model["is_active"])
        
        # Emit signal
        self.model_selected.emit(model_data["name"])
    
    def update_model_score(self, model_name: str, score: float):
        """Update score for specific model."""
        for model in self.models:
            if model["name"] == model_name:
                model["score"] = score
                break
        
        # Refresh display
        self.set_models(self.models)
    
    def get_active_model(self) -> Dict:
        """Get currently active model."""
        for model in self.models:
            if model.get("is_active", False):
                return model
        return {}
    
    def get_model_names(self) -> List[str]:
        """Get list of model names."""
        return [model["name"] for model in self.models]
    
    def add_model(self, model_data: Dict):
        """Add new model to the panel."""
        self.models.append(model_data)
        
        card = ModelCard(model_data)
        card.mousePressEvent = lambda event, m=model_data: self.on_model_clicked(m)
        self.models_layout.addWidget(card)
        self.model_cards.append(card)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove model by name."""
        for i, model in enumerate(self.models):
            if model["name"] == model_name:
                self.models.pop(i)
                self.model_cards[i].deleteLater()
                self.model_cards.pop(i)
                return True
        return False
    
    def refresh(self):
        """Refresh the model display."""
        self.set_models(self.models)
