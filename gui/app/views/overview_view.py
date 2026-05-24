"""Overview tab — карточка решения на последнем баре."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from gui.api import IstGuiClient
from gui.api.types import ExplainSnapshot
from gui.app.widgets.explain_card import ExplainCard
from gui.app.workers import ExplainWorker


class OverviewView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._worker: Optional[ExplainWorker] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)

        self._explain = ExplainCard()
        layout.addWidget(self._explain)
        layout.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._explain.set_loading()
        self._worker = ExplainWorker(
            self._symbol, self._timeframe, window=256, api=self._api
        )
        self._worker.finished.connect(self._on_explain)
        self._worker.failed.connect(self._explain.set_error)
        self._worker.start()

    def _on_explain(self, snap: ExplainSnapshot) -> None:
        self._explain.set_snapshot(snap)
