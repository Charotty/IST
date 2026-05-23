"""Вкладка «График» — ChartView + Решение / Bundle (режим — отдельная вкладка)."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from gui.api import IstGuiClient
from gui.app.views.chart_view import ChartView
from gui.app.views.models_view import ModelsView
from gui.app.views.overview_view import OverviewView


class ChartHubView(QWidget):
    """Обёртка: график на весь split; снизу только Решение и Bundle."""

    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Vertical)
        self.chart = ChartView(self._api)
        split.addWidget(self.chart)

        self._aux = QTabWidget()
        self._aux.setDocumentMode(True)
        self.overview = OverviewView(self._api)
        self.models = ModelsView(self._api)
        self._aux.addTab(self.overview, "Решение")
        self._aux.addTab(self.models, "Bundle")
        self._aux.setMaximumHeight(320)
        self._aux.currentChanged.connect(self._on_aux_tab)
        split.addWidget(self._aux)

        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 2)
        split.setSizes([580, 240])
        layout.addWidget(split)

    def _on_aux_tab(self, index: int) -> None:
        if index == 0:
            self.overview.refresh()
        elif index == 1:
            self.models.refresh(validate_schema=False)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self.chart.set_context(symbol, timeframe)
        self.overview.set_context(symbol, timeframe)
        self.models.set_context(symbol, timeframe)

    def refresh(self) -> None:
        self.chart.refresh()
        idx = self._aux.currentIndex()
        if idx == 0:
            self.overview.refresh()
        elif idx == 1:
            self.models.refresh(validate_schema=False)
