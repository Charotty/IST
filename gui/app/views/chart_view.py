"""График — свечи OHLCV с OKX."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import ChartBar, ChartPayload
from gui.app.widgets.candlestick import CandlestickItem
from gui.app.workers import OkxChartWorker

try:
    import pyqtgraph as pg

    _HAS_PG = True
except ImportError:
    _HAS_PG = False


class ChartView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._worker: Optional[OkxChartWorker] = None
        self._payload: Optional[ChartPayload] = None

        layout = QVBoxLayout(self)
        if not _HAS_PG:
            layout.addWidget(
                QLabel("Установите pyqtgraph: pip install -r requirements-gui.txt")
            )
            return

        hint = QLabel(
            "Котировки загружаются с OKX в реальном времени (публичный API). "
            "Сигналы модели — на вкладках «Обзор» и «Исполнение» после обучения bundle."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #000000; padding: 4px 0;")
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Свечей:"))
        self._limit = QComboBox()
        self._limit.addItems(["100", "200", "300"])
        self._limit.setCurrentText("200")
        row.addWidget(self._limit)
        self._btn_load = QPushButton("Загрузить с OKX")
        self._btn_load.clicked.connect(self.refresh)
        row.addWidget(self._btn_load)
        row.addStretch()
        layout.addLayout(row)

        self._plot = pg.PlotWidget(title="Свечной график (OKX)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("left", "Цена")
        self._plot.setLabel("bottom", "Бар")
        layout.addWidget(self._plot, stretch=1)

        self._status = QLabel("")
        layout.addWidget(self._status)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if not _HAS_PG:
            return
        if self._worker and self._worker.isRunning():
            return
        limit = int(self._limit.currentText())
        self._status.setText(f"Загрузка {self._symbol} {self._timeframe} с OKX…")
        self._plot.clear()
        self._worker = OkxChartWorker(
            self._symbol, self._timeframe, limit=limit, api=self._api
        )
        self._worker.finished.connect(self._on_payload)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_fail(self, msg: str) -> None:
        self._payload = None
        self._status.setText(f"Ошибка OKX: {msg}")

    def _on_payload(self, payload: ChartPayload) -> None:
        self._payload = payload
        self._redraw()

    def _redraw(self) -> None:
        if not _HAS_PG or self._payload is None:
            return
        bars = self._payload.bars
        self._plot.clear()
        if not bars:
            self._status.setText("Нет данных")
            return

        data = [{"open": b.open, "high": b.high, "low": b.low, "close": b.close} for b in bars]
        item = CandlestickItem(data)
        self._plot.addItem(item)

        closes = np.array([b.close for b in bars], dtype=float)
        xs = np.arange(len(closes))
        self._plot.plot(xs, closes, pen=pg.mkPen("#1565c0", width=1, style=Qt.PenStyle.DotLine))

        vols = [b.volume or 0 for b in bars]
        self._status.setText(
            f"{len(bars)} свечей · {bars[0].t[:16]} … {bars[-1].t[:16]} · "
            f"close {closes[-1]:,.2f} · vol {vols[-1]:,.0f}"
        )
