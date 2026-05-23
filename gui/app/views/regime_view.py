"""Вкладка «Режим» — история regime (API: inference.regime_series)."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import RegimeBar
from gui.app.widgets.help_label import help_label
from gui.app.workers import RegimeSeriesWorker


class RegimeView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._worker: Optional[RegimeSeriesWorker] = None
        self._rows: List[RegimeBar] = []

        layout = QVBoxLayout(self)
        layout.addWidget(
            help_label(
                "История режима рынка (trend / range) из <code>orchestration.introspect.regime_history</code>. "
                "На графике IST те же данные отображаются как фоновые полосы.",
                background="#fff8e1",
                border="#ffcc80",
            )
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Шаг (разрежение):"))
        self._step = QSpinBox()
        self._step.setRange(1, 168)
        self._step.setValue(1)
        row.addWidget(self._step)
        row.addWidget(QLabel("Макс. баров:"))
        self._max_display = QSpinBox()
        self._max_display.setRange(50, 5000)
        self._max_display.setValue(500)
        row.addWidget(self._max_display)
        self._btn_load = QPushButton("Загрузить")
        self._btn_load.clicked.connect(self.refresh)
        row.addWidget(self._btn_load)
        row.addStretch()
        layout.addLayout(row)

        self._stats = QLabel("—")
        self._stats.setWordWrap(True)
        layout.addWidget(self._stats)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Время", "Режим", "int", "Close"])
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table, stretch=1)

        self._export = QPlainTextEdit()
        self._export.setReadOnly(True)
        self._export.setMaximumHeight(100)
        self._export.setPlaceholderText("JSON (последние строки)…")
        layout.addWidget(self._export)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._stats.setText("Загрузка…")
        self._table.setRowCount(0)
        self._worker = RegimeSeriesWorker(
            self._symbol,
            self._timeframe,
            step=self._step.value(),
            api=self._api,
        )
        self._worker.finished.connect(self._on_data)
        self._worker.failed.connect(lambda m: self._stats.setText(f"Ошибка: {m}"))
        self._worker.start()

    def _on_data(self, rows: List[RegimeBar]) -> None:
        self._rows = rows
        if not rows:
            self._stats.setText("Нет данных (нужен OHLCV parquet).")
            return

        n_trend = sum(1 for r in rows if r.regime_int == 1)
        n_range = len(rows) - n_trend
        pct_t = 100.0 * n_trend / len(rows)
        self._stats.setText(
            f"Всего {len(rows)} точек (step={self._step.value()}): "
            f"trend {n_trend} ({pct_t:.1f}%), range {n_range} ({100 - pct_t:.1f}%) · "
            f"{rows[0].t[:16]} … {rows[-1].t[:16]}"
        )

        tail = rows[-self._max_display.value() :]
        self._table.setRowCount(len(tail))
        for i, r in enumerate(tail):
            self._table.setItem(i, 0, QTableWidgetItem(r.t[:19]))
            self._table.setItem(i, 1, QTableWidgetItem(r.regime))
            self._table.setItem(i, 2, QTableWidgetItem(str(r.regime_int)))
            close = "" if r.close is None else f"{r.close:,.2f}"
            self._table.setItem(i, 3, QTableWidgetItem(close))
        self._table.resizeColumnsToContents()

        import json

        sample = [
            {"t": r.t, "regime": r.regime, "regime_int": r.regime_int, "close": r.close}
            for r in tail[-20:]
        ]
        self._export.setPlainText(json.dumps(sample, indent=2, ensure_ascii=False))
