"""Per-fold OOS returns and cumulative equity (from journal)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.api.types import FoldEquityPoint

try:
    import pyqtgraph as pg

    _HAS_PG = True
except ImportError:
    _HAS_PG = False


class FoldEquityChart(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PG:
            layout.addWidget(QLabel("Для графика equity нужен pyqtgraph"))
            self._plot = None
            return
        self._plot = pg.PlotWidget(title="WFO — доходность по фолдам")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Фолд")
        self._plot.setLabel("left", "Доходность %")
        layout.addWidget(self._plot)

    def set_curve(self, points: List[FoldEquityPoint]) -> None:
        if not _HAS_PG or self._plot is None:
            return
        self._plot.clear()
        if not points:
            self._plot.setTitle("В этом прогоне нет метрик по фолдам")
            return
        xs = np.array([p.fold for p in points], dtype=float)
        rets = np.array([p.oos_return_pct for p in points], dtype=float)
        cum = np.array([p.cumulative_pct for p in points], dtype=float)
        bar = pg.BarGraphItem(x=xs, height=rets, width=0.35, brush="#90caf9")
        self._plot.addItem(bar)
        self._plot.plot(xs, cum, pen=pg.mkPen("#ef6c00", width=2), symbol="o", name="cumulative")
        self._plot.setTitle(
            f"{len(points)} фолдов · итоговая кумулятивная {cum[-1]:.2f}%"
        )
