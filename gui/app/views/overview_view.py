"""Overview tab — explain card + decision steps."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)

        self._explain = ExplainCard()
        layout.addWidget(self._explain)

        steps_title = QLabel(
            "Шаги decision pipeline — проверки на последнем баре (✓ прошло / ✗ нет)"
        )
        steps_title.setWordWrap(True)
        steps_title.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(steps_title)
        self._steps = QListWidget()
        self._steps.setMinimumHeight(120)
        layout.addWidget(self._steps)
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

        p = snap.meta_probability
        items = [
            (
                f"P(up) ансамбля = {p:.4f} (порог направления thr={thr})",
                abs(p - 0.5) >= (thr - 0.5),
            ),
            (
                f"Итог: {snap.direction.upper()} (сигнал {snap.signal})",
                snap.signal != 0,
            ),
            (
                "Нет блокировки risk/decision (см. поле «Блокировка»)",
                snap.why_blocked is None,
            ),
            (
                f"Вне мёртвой зоны: |P−0.5| ≥ margin ({margin})",
                margin <= 0 or abs(p - 0.5) >= margin,
            ),
            (
                f"trade_mode={cfg.get('trade_mode', 'both')} допускает сторону",
                True,
            ),
            (
                f"Доля позиции > 0: {snap.position_size_frac:.4f}",
                snap.position_size_frac > 0,
            ),
        ]
        for text, ok in items:
            mark = "✓" if ok else "✗"
            item = QListWidgetItem(f"{mark}  {text}")
            if not ok:
                item.setForeground(item.foreground().color().darker(150))
            self._steps.addItem(item)
