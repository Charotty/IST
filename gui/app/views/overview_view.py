"""Overview tab — explain card + decision steps."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

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

        layout = QVBoxLayout(self)
        self._explain = ExplainCard()
        layout.addWidget(self._explain)

        layout.addWidget(QLabel("Шаги decision pipeline"))
        self._steps = QListWidget()
        self._steps.setMaximumHeight(160)
        layout.addWidget(self._steps)
        layout.addStretch()

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._explain.set_loading()
        self._steps.clear()
        self._worker = ExplainWorker(
            self._symbol, self._timeframe, window=256, api=self._api
        )
        self._worker.finished.connect(self._on_explain)
        self._worker.failed.connect(self._explain.set_error)
        self._worker.start()

    def _on_explain(self, snap: ExplainSnapshot) -> None:
        self._explain.set_snapshot(snap)
        self._fill_steps(snap)

    def _fill_steps(self, snap: ExplainSnapshot) -> None:
        self._steps.clear()
        cfg = snap.config or {}
        thr = float(cfg.get("direction_threshold", 0.52))
        margin = float(cfg.get("min_signal_margin", 0.0))

        items = [
            (
                f"Ensemble P(up) = {snap.meta_probability:.4f}",
                abs(snap.meta_probability - 0.5) >= (thr - 0.5),
            ),
            (
                f"Direction: {snap.direction} (signal {snap.signal})",
                snap.signal != 0,
            ),
            (
                "Фильтр meta (интегрированный порог)",
                snap.why_blocked is None or snap.signal != 0,
            ),
            (
                f"Мёртвая зона (min_signal_margin={margin})",
                margin <= 0 or abs(snap.meta_probability - 0.5) >= margin,
            ),
            (
                f"Режим сделок: {cfg.get('trade_mode', 'both')}",
                True,
            ),
            (
                f"Position size fraction: {snap.position_size_frac:.4f}",
                snap.position_size_frac > 0,
            ),
        ]
        for text, ok in items:
            mark = "✓" if ok else "✗"
            item = QListWidgetItem(f"{mark}  {text}")
            if not ok:
                item.setForeground(item.foreground().color().darker(150))
            self._steps.addItem(item)
