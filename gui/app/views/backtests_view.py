"""Журнал WFO / бэктестов."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import BacktestRunSummary, FoldEquityPoint
from gui.app.widgets.fold_equity_chart import FoldEquityChart
from gui.app.widgets.help_label import help_label
from gui.app.workers import BacktestListWorker, FoldEquityWorker


class BacktestsView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol: Optional[str] = None
        self._timeframe: Optional[str] = None
        self._worker: Optional[BacktestListWorker] = None
        self._equity_worker: Optional[FoldEquityWorker] = None
        self._runs: List[BacktestRunSummary] = []

        layout = QVBoxLayout(self)

        help_text = help_label(
            "<b>Что здесь показывается</b> — журнал экспериментов "
            "<code>docs/backtest_journal/</code> (walk-forward и отчёты CLI).<br>"
            "• <b>ID</b> — уникальный прогон; клик → JSON с метриками по фолдам.<br>"
            "• <b>UTC</b> — время записи в журнал.<br>"
            "• <b>этап</b> — тип прогона (wfo_integrated, orchestration_tune, report_real…).<br>"
            "• <b>метка</b> — имя профиля/эксперимента из конфига.<br>"
            "• <b>accept</b> — прошёл ли критерии acceptance (мин. Sharpe, DD, WFE…).<br>"
            "• <b>target</b> — целевые KPI (строже acceptance).<br>"
            "• <b>Sharpe</b> — средний OOS Sharpe по фолдам из summary.<br>"
            "Нижний график — накопленная доходность по фолдам WFO (из % return каждого фолда).",
            background="#f3e5f5",
            border="#ce93d8",
        )
        layout.addWidget(help_text)

        split = QSplitter(Qt.Orientation.Vertical)

        top = QSplitter(Qt.Orientation.Horizontal)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["ID", "UTC", "этап", "метка", "accept", "target", "Sharpe"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.cellClicked.connect(self._on_select)
        top.addWidget(self._table)

        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("Выберите строку — откроется полный снимок прогона…")
        top.addWidget(self._detail)
        top.setSizes([500, 400])
        split.addWidget(top)

        self._equity = FoldEquityChart()
        split.addWidget(self._equity)
        split.setSizes([420, 220])

        layout.addWidget(split, stretch=1)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._table.setRowCount(0)
        self._detail.clear()
        self._equity.set_curve([])
        self._worker = BacktestListWorker(
            symbol=self._symbol,
            timeframe=self._timeframe,
            limit=80,
            api=self._api,
        )
        self._worker.finished.connect(self._on_runs)
        self._worker.failed.connect(lambda m: self._detail.setPlainText(f"Ошибка: {m}"))
        self._worker.start()

    def _on_runs(self, runs: List[BacktestRunSummary]) -> None:
        self._runs = runs
        self._table.setRowCount(len(runs))
        for row, r in enumerate(runs):
            s = r.summary or {}
            cells = [
                r.run_id[:18],
                (r.timestamp_utc or "")[:19],
                r.stage or "",
                (r.label or "")[:20],
                "PASS" if r.acceptance_passed else "FAIL",
                "PASS" if r.target_passed else "FAIL",
                f"{float(s.get('mean_sharpe', 0)):.3f}",
            ]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()

    def _on_select(self, row: int, _col: int) -> None:
        if row < 0 or row >= len(self._runs):
            return
        run_id = self._runs[row].run_id
        try:
            data = self._api.backtests.get_run(run_id)
            import json

            self._detail.setPlainText(json.dumps(data, indent=2, default=str, ensure_ascii=False))
        except Exception as e:
            self._detail.setPlainText(str(e))
            return

        if self._equity_worker and self._equity_worker.isRunning():
            return
        self._equity_worker = FoldEquityWorker(run_id, api=self._api)
        self._equity_worker.finished.connect(self._on_equity)
        self._equity_worker.failed.connect(lambda _: self._equity.set_curve([]))
        self._equity_worker.start()

    def _on_equity(self, points: List[FoldEquityPoint]) -> None:
        self._equity.set_curve(points)
