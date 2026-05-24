"""Вкладка «Задачи» — Pipeline + Журнал + Paper (обёртка без изменения дочерних view)."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from gui.api import IstGuiClient
from gui.app.views.backtests_view import BacktestsView
from gui.app.views.execution_view import ExecutionView
from gui.app.views.jobs_view import JobsView


class JobsHubView(QWidget):
    """Подвкладки: Pipeline (JobsView), Журнал (BacktestsView), Paper (ExecutionView)."""

    pipeline_finished = pyqtSignal()

    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self.pipeline = JobsView(self._api)
        self.journal = BacktestsView(self._api)
        self.paper = ExecutionView(self._api)

        self._tabs.addTab(self.pipeline, "Pipeline")
        self._tabs.addTab(self.journal, "Журнал WFO")
        self._tabs.addTab(self.paper, "Практика")

        self.pipeline.pipeline_finished.connect(self.pipeline_finished.emit)

        layout.addWidget(self._tabs)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self.pipeline.set_context(symbol, timeframe)
        self.journal.set_context(symbol, timeframe)
        self.paper.set_context(symbol, timeframe)

    def refresh(self) -> None:
        current = self._tabs.currentWidget()
        if current is self.pipeline and hasattr(self.pipeline, "refresh"):
            self.pipeline.refresh()
        elif current is self.journal and hasattr(self.journal, "refresh"):
            self.journal.refresh()
        elif current is self.paper and hasattr(self.paper, "refresh"):
            self.paper.refresh()

    def refresh_journal(self) -> None:
        self.journal.refresh()
