"""Свечной график для pyqtgraph."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPicture


class CandlestickItem(pg.GraphicsObject):
    """OHLC свечи; ``data`` — список dict с ключами open/high/low/close."""

    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data
        self.picture = QPicture()
        self.generate_picture()

    def generate_picture(self) -> None:
        painter = QPainter(self.picture)
        w = 0.35
        for i, d in enumerate(self.data):
            o, h, l, c = d["open"], d["high"], d["low"], d["close"]
            color = "#26a69a" if c >= o else "#ef5350"
            painter.setPen(pg.mkPen(color))
            painter.setBrush(pg.mkBrush(color))
            painter.drawLine(QPointF(i, l), QPointF(i, h))
            body = abs(c - o) or (h - l) * 0.01 or 1e-6
            painter.drawRect(QRectF(i - w, min(o, c), w * 2, body))
        painter.end()

    def paint(self, painter, *args) -> None:
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())
