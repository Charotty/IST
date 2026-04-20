from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF, QSize
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


class ChartDataPoint:
    """Single data point for chart."""
    
    def __init__(self, timestamp: datetime, open_price: float, high: float, 
                 low: float, close: float, prediction: Optional[float] = None, 
                 signal: Optional[str] = None):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.prediction = prediction
        self.signal = signal


class MainChart(QWidget):
    """Main trading chart with price and prediction lines."""
    
    def __init__(self):
        super().__init__()
        self.pair = "BTC/USDT"
        self.chart_data: List[ChartDataPoint] = []
        self.current_price = 0.0
        self.price_change = 0.0
        
        # Chart settings
        self.padding = 40
        self.grid_color = QColor("#1F2933")
        self.price_color = QColor("#22C55E")
        self.prediction_color = QColor("#3B82F6")
        self.text_color = QColor("#C9D1D9")
        self.signal_buy_color = QColor("#22C55E")
        self.signal_sell_color = QColor("#EF4444")
        
        # Initialize with demo data
        self.generate_demo_data()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart_data)
    
    def generate_demo_data(self):
        """Generate demo chart data."""
        self.chart_data.clear()
        
        # Base price for selected pair
        base_prices = {
            "BTC/USDT": 45000,
            "ETH/USDT": 2500,
            "BNB/USDT": 300,
            "SOL/USDT": 100,
            "XRP/USDT": 0.5,
            "ADA/USDT": 0.3
        }
        
        base_price = base_prices.get(self.pair, 100)
        current_price = base_price
        
        # Generate 50 data points (roughly 50 minutes)
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=(49 - i))
            
            # Random walk with volatility
            volatility = base_price * 0.02
            price_change = (random.random() - 0.5) * volatility
            
            open_price = current_price
            close_price = open_price + price_change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.5
            low_price = min(open_price, close_price) - random.random() * volatility * 0.5
            
            # Add prediction
            prediction = close_price + (random.random() - 0.5) * volatility * 0.3
            
            # Add signals at specific points
            signal = None
            if i == 30:
                signal = "BUY"
            elif i == 45:
                signal = "SELL"
            
            data_point = ChartDataPoint(
                timestamp, open_price, high_price, low_price, close_price, prediction, signal
            )
            self.chart_data.append(data_point)
            
            current_price = close_price
        
        # Update current price and change
        if self.chart_data:
            first_price = self.chart_data[0].close
            last_price = self.chart_data[-1].close
            self.current_price = last_price
            self.price_change = ((last_price - first_price) / first_price) * 100
    
    def update_chart_data(self):
        """Update chart with new data point."""
        if not self.chart_data:
            return
        
        # Remove oldest data point
        if len(self.chart_data) > 50:
            self.chart_data.pop(0)
        
        # Add new data point
        last_point = self.chart_data[-1]
        volatility = self.current_price * 0.02
        
        new_timestamp = datetime.now()
        new_open = last_point.close
        new_close = new_open + (random.random() - 0.5) * volatility
        new_high = max(new_open, new_close) + random.random() * volatility * 0.5
        new_low = min(new_open, new_close) - random.random() * volatility * 0.5
        new_prediction = new_close + (random.random() - 0.5) * volatility * 0.3
        
        new_point = ChartDataPoint(
            new_timestamp, new_open, new_high, new_low, new_close, new_prediction
        )
        
        self.chart_data.append(new_point)
        self.current_price = new_close
        
        # Update price change
        if len(self.chart_data) > 1:
            first_price = self.chart_data[0].close
            self.price_change = ((new_close - first_price) / first_price) * 100
        
        self.update()
    
    def set_pair(self, pair: str):
        """Set trading pair and regenerate data."""
        self.pair = pair
        self.generate_demo_data()
        self.update()
    
    def update_price(self, price: float):
        """Update current price."""
        if self.chart_data:
            self.current_price = price
            self.chart_data[-1].close = price
            
            # Update price change
            first_price = self.chart_data[0].close
            self.price_change = ((price - first_price) / first_price) * 100
            
            self.update()
    
    def add_signal(self, signal: str):
        """Add signal to latest data point."""
        if self.chart_data:
            self.chart_data[-1].signal = signal
            self.update()
    
    def paintEvent(self, event):
        """Paint the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#0D1117"))
        
        if not self.chart_data:
            return
        
        # Calculate chart area
        chart_rect = QRectF(
            self.padding,
            self.padding,
            self.width() - 2 * self.padding,
            self.height() - 2 * self.padding - 60  # Space for header
        )
        
        # Draw header
        self.draw_header(painter)
        
        # Draw grid
        self.draw_grid(painter, chart_rect)
        
        # Draw price line
        self.draw_price_line(painter, chart_rect)
        
        # Draw prediction line
        self.draw_prediction_line(painter, chart_rect)
        
        # Draw signals
        self.draw_signals(painter, chart_rect)
        
        # Draw axes
        self.draw_axes(painter, chart_rect)
    
    def draw_header(self, painter: QPainter):
        """Draw chart header with pair and price info."""
        header_y = 20
        
        # Pair name
        font = QFont()
        font.setPointSize(16)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(self.text_color)
        painter.drawText(self.padding, header_y, self.pair)
        
        # Current price
        font.setPointSize(20)
        painter.setFont(font)
        price_text = f"${self.current_price:.2f}"
        price_x = self.padding + 150
        painter.drawText(price_x, header_y, price_text)
        
        # Price change
        change_color = self.price_color if self.price_change >= 0 else QColor("#EF4444")
        painter.setPen(change_color)
        font.setPointSize(14)
        painter.setFont(font)
        change_text = f"{self.price_change:+.2f}%"
        change_x = price_x + 120
        painter.drawText(change_x, header_y, change_text)
    
    def draw_grid(self, painter: QPainter, chart_rect: QRectF):
        """Draw background grid."""
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.SolidLine))
        
        # Horizontal lines
        for i in range(5):
            y = chart_rect.top() + (chart_rect.height() / 4) * i
            painter.drawLine(QPointF(chart_rect.left(), y), QPointF(chart_rect.right(), y))
        
        # Vertical lines
        for i in range(5):
            x = chart_rect.left() + (chart_rect.width() / 4) * i
            painter.drawLine(QPointF(x, chart_rect.top()), QPointF(x, chart_rect.bottom()))
    
    def draw_price_line(self, painter: QPainter, chart_rect: QRectF):
        """Draw price line."""
        if len(self.chart_data) < 2:
            return
        
        # Get price range
        prices = [point.close for point in self.chart_data]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        if price_range == 0:
            price_range = 1
        
        # Create path
        path = QPainterPath()
        
        for i, point in enumerate(self.chart_data):
            x = chart_rect.left() + (chart_rect.width() / (len(self.chart_data) - 1)) * i
            y = chart_rect.bottom() - ((point.close - min_price) / price_range) * chart_rect.height()
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        # Draw line
        painter.setPen(QPen(self.price_color, 2, Qt.PenStyle.SolidLine))
        painter.drawPath(path)
    
    def draw_prediction_line(self, painter: QPainter, chart_rect: QRectF):
        """Draw prediction line."""
        if len(self.chart_data) < 2:
            return
        
        # Filter points with predictions
        pred_points = [p for p in self.chart_data if p.prediction is not None]
        if len(pred_points) < 2:
            return
        
        # Get prediction range
        predictions = [p.prediction for p in pred_points]
        min_pred = min(predictions)
        max_pred = max(predictions)
        pred_range = max_pred - min_pred
        if pred_range == 0:
            pred_range = 1
        
        # Create path
        path = QPainterPath()
        first_point = True
        
        for i, point in enumerate(self.chart_data):
            if point.prediction is None:
                continue
            
            x = chart_rect.left() + (chart_rect.width() / (len(self.chart_data) - 1)) * i
            y = chart_rect.bottom() - ((point.prediction - min_pred) / pred_range) * chart_rect.height()
            
            if first_point:
                path.moveTo(x, y)
                first_point = False
            else:
                path.lineTo(x, y)
        
        # Draw dashed line
        painter.setPen(QPen(self.prediction_color, 2, Qt.PenStyle.DashLine))
        painter.drawPath(path)
    
    def draw_signals(self, painter: QPainter, chart_rect: QRectF):
        """Draw buy/sell signals."""
        if len(self.chart_data) < 2:
            return
        
        # Get price range for positioning
        prices = [point.close for point in self.chart_data]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        if price_range == 0:
            price_range = 1
        
        for i, point in enumerate(self.chart_data):
            if point.signal is None:
                continue
            
            x = chart_rect.left() + (chart_rect.width() / (len(self.chart_data) - 1)) * i
            
            # Position signal at high/low of candle
            if point.signal == "BUY":
                y = chart_rect.bottom() - ((point.low - min_price) / price_range) * chart_rect.height()
                color = self.signal_buy_color
            elif point.signal == "SELL":
                y = chart_rect.bottom() - ((point.high - min_price) / price_range) * chart_rect.height()
                color = self.signal_sell_color
            else:
                continue
            
            # Draw signal circle
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawEllipse(QPointF(x, y), 6, 6)
    
    def draw_axes(self, painter: QPainter, chart_rect: QRectF):
        """Draw axes with labels."""
        painter.setPen(self.text_color)
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        
        # Y-axis labels (price)
        if self.chart_data:
            prices = [point.close for point in self.chart_data]
            min_price = min(prices)
            max_price = max(prices)
            
            for i in range(5):
                price = min_price + (max_price - min_price) * (i / 4)
                y = chart_rect.bottom() - (chart_rect.height() / 4) * i
                price_text = f"${price:.2f}"
                
                # Right align
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(price_text)
                painter.drawText(
                    QPointF(chart_rect.right() - text_width - 5, y + fm.height() / 4),
                    price_text
                )
        
        # X-axis labels (time)
        if len(self.chart_data) >= 2:
            for i in range(0, len(self.chart_data), 10):  # Show every 10th label
                if i >= len(self.chart_data):
                    break
                
                x = chart_rect.left() + (chart_rect.width() / (len(self.chart_data) - 1)) * i
                time_text = self.chart_data[i].timestamp.strftime("%H:%M")
                
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(time_text)
                painter.drawText(
                    QPointF(x - text_width / 2, chart_rect.bottom() + 20),
                    time_text
                )
    
    def start_real_time_updates(self):
        """Start real-time chart updates."""
        self.update_timer.start(1000)  # Update every second
    
    def stop_real_time_updates(self):
        """Stop real-time chart updates."""
        self.update_timer.stop()
    
    def sizeHint(self) -> 'QSize':
        """Return recommended size."""
        return QSize(800, 400)
