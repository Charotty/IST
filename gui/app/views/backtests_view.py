"""Журнал WFO / бэктестов — фильтры, сводка, equity."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import BacktestRunSummary, FoldEquityPoint
from gui.app.backtest_run_format import stage_display
from gui.app.formatters import format_mean_sharpe, format_pass_fail
from gui.app.widgets.fold_equity_chart import FoldEquityChart
from gui.app.widgets.help_label import help_label
from gui.app.widgets.run_snapshot_panel import RunSnapshotPanel
from gui.app.workers import BacktestListWorker, FoldEquityWorker

_PASS_BG = QColor("#e8f5e9")
_FAIL_BG = QColor("#ffebee")
_NEUTRAL_BG = QColor("#ffffff")


def _summary_float(summary: dict, key: str) -> float:
    try:
        return float(summary.get(key, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


class BacktestsView(QWidget):
    """Journal browser with filters and structured summary."""

    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol: Optional[str] = None
        self._timeframe: Optional[str] = None
        self._worker: Optional[BacktestListWorker] = None
        self._equity_worker: Optional[FoldEquityWorker] = None
        self._all_runs: List[BacktestRunSummary] = []
        self._runs: List[BacktestRunSummary] = []
        self._pending_select_id: Optional[str] = None

        layout = QVBoxLayout(self)

        layout.addWidget(
            help_label(
                "<b>Журнал</b> <code>docs/backtest_journal/</code> — все прогоны WFO и CLI.<br>"
                "Фильтруйте по этапу, сортируйте по Sharpe, открывайте <b>Сводку</b> с критериями acceptance/target. "
                "Новый прогон — кнопка <b>Отчёт WFO</b> на вкладке «Задачи → Pipeline».",
                background="#f3e5f5",
                border="#ce93d8",
            )
        )

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Обновить")
        self._btn_best = QPushButton("Лучший PASS")
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_best.clicked.connect(self._select_best_acceptance)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_best)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("Этап:"))
        self._stage = QComboBox()
        self._stage.setMinimumWidth(160)
        self._stage.currentIndexChanged.connect(self._apply_view_filters)
        filt.addWidget(self._stage)

        self._only_pass = QCheckBox("Только acceptance PASS")
        self._only_pass.toggled.connect(self._apply_view_filters)
        filt.addWidget(self._only_pass)

        filt.addWidget(QLabel("Сортировка:"))
        self._sort = QComboBox()
        self._sort.addItems(
            ["UTC ↓ (новые)", "Sharpe ↓", "UTC ↑ (старые)", "PF ↓"]
        )
        self._sort.currentIndexChanged.connect(self._apply_view_filters)
        filt.addWidget(self._sort)

        filt.addWidget(QLabel("Строк:"))
        self._limit = QSpinBox()
        self._limit.setRange(20, 500)
        self._limit.setValue(150)
        self._limit.setSingleStep(25)
        self._limit.valueChanged.connect(self._apply_view_filters)
        filt.addWidget(self._limit)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #666;")
        filt.addWidget(self._count_label)
        filt.addStretch()
        layout.addLayout(filt)

        split = QSplitter(Qt.Orientation.Vertical)

        top = QSplitter(Qt.Orientation.Horizontal)
        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            [
                "ID",
                "UTC",
                "этап",
                "accept",
                "target",
                "Sharpe",
                "PF",
                "WFE",
                "folds",
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background: #fafafa; gridline-color: #e0e0e0; }"
            "QTableWidget::item:selected { background: #bbdefb; color: #111; }"
        )
        self._table.cellClicked.connect(self._on_select)
        top.addWidget(self._table)

        self._detail = RunSnapshotPanel()
        top.addWidget(self._detail)
        top.setSizes([520, 380])
        split.addWidget(top)

        self._equity = FoldEquityChart()
        split.addWidget(self._equity)
        split.setSizes([440, 200])

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
            limit=None,
            api=self._api,
        )
        self._worker.finished.connect(self._on_runs_loaded)
        self._worker.failed.connect(lambda m: self._detail.set_error(f"Ошибка: {m}"))
        self._worker.start()

    def _on_runs_loaded(self, runs: List[BacktestRunSummary]) -> None:
        from datetime import datetime, timezone

        self._all_runs = runs
        self._rebuild_stage_combo()
        self._apply_view_filters()
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        sym = self._journal_symbol_key()
        self._count_label.setText(
            f"Показано {len(self._runs)} · в журнале для пары: {len(self._all_runs)} · "
            f"обновлено {ts} UTC · {sym}"
        )

    def _rebuild_stage_combo(self) -> None:
        stages = sorted(
            {
                stage_display(r.label, r.stage)
                for r in self._all_runs
                if stage_display(r.label, r.stage) != "—"
            }
        )
        prev = self._stage.currentText()
        self._stage.blockSignals(True)
        self._stage.clear()
        self._stage.addItem("(все этапы)")
        for s in stages:
            self._stage.addItem(s)
        idx = self._stage.findText(prev)
        self._stage.setCurrentIndex(idx if idx >= 0 else 0)
        self._stage.blockSignals(False)

    def _apply_view_filters(self) -> None:
        runs = list(self._all_runs)
        stage_sel = self._stage.currentText()
        if stage_sel and stage_sel != "(все этапы)":
            runs = [
                r
                for r in runs
                if stage_display(r.label, r.stage) == stage_sel
            ]
        if self._only_pass.isChecked():
            runs = [r for r in runs if r.acceptance_passed is True]

        sort_mode = self._sort.currentIndex()
        if sort_mode == 0:
            runs.sort(key=lambda r: r.timestamp_utc or "", reverse=True)
        elif sort_mode == 1:
            runs.sort(
                key=lambda r: _summary_float(r.summary or {}, "mean_sharpe"),
                reverse=True,
            )
        elif sort_mode == 2:
            runs.sort(key=lambda r: r.timestamp_utc or "")
        else:
            runs.sort(
                key=lambda r: _summary_float(r.summary or {}, "mean_profit_factor"),
                reverse=True,
            )

        cap = self._limit.value()
        self._runs = runs[:cap]
        self._fill_table()

    def _fill_table(self) -> None:
        self._table.setRowCount(len(self._runs))
        for row, r in enumerate(self._runs):
            s = r.summary or {}
            cells = [
                r.run_id[:20],
                (r.timestamp_utc or "")[:19],
                stage_display(r.label, r.stage)[:24],
                format_pass_fail(r.acceptance_passed),
                format_pass_fail(r.target_passed),
                format_mean_sharpe(s),
                self._fmt_metric(s, "mean_profit_factor"),
                self._fmt_metric(s, "mean_wfe"),
                str(s.get("n_folds", "—")),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col in (3, 4):
                    if text == "PASS":
                        item.setBackground(_PASS_BG)
                        item.setForeground(QColor("#1b5e20"))
                    elif text == "FAIL":
                        item.setBackground(_FAIL_BG)
                        item.setForeground(QColor("#b71c1c"))
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        if self._pending_select_id:
            for row, r in enumerate(self._runs):
                if r.run_id == self._pending_select_id:
                    self._table.selectRow(row)
                    self._on_select(row, 0)
                    self._pending_select_id = None
                    break

    @staticmethod
    def _fmt_metric(summary: dict, key: str) -> str:
        raw = summary.get(key)
        if raw is None:
            return "—"
        try:
            return f"{float(raw):.3f}"
        except (TypeError, ValueError):
            return "—"

    def _on_select(self, row: int, _col: int) -> None:
        if row < 0 or row >= len(self._runs):
            return
        run_id = self._runs[row].run_id
        try:
            data = self._api.backtests.get_run(run_id)
            self._detail.set_run(data)
        except Exception as e:
            self._detail.set_error(str(e))
            return

        if self._equity_worker and self._equity_worker.isRunning():
            return
        self._equity_worker = FoldEquityWorker(run_id, api=self._api)
        self._equity_worker.finished.connect(self._on_equity)
        self._equity_worker.failed.connect(lambda _: self._equity.set_curve([]))
        self._equity_worker.start()

    def _on_equity(self, points: List[FoldEquityPoint]) -> None:
        self._equity.set_curve(points)

    def _normalized_symbol(self) -> Optional[str]:
        if not self._symbol:
            return None
        return self._api.symbols.normalize_symbol(self._symbol)

    def _journal_symbol_key(self) -> str:
        if not self._symbol:
            return "все пары"
        sym = self._api.symbols.normalize_symbol(self._symbol)
        tf = self._timeframe or "1h"
        return f"{sym} {tf}"

    def _select_best_acceptance(self) -> None:
        try:
            best = self._api.backtests.best_acceptance(
                symbol=self._normalized_symbol(),
                timeframe=self._timeframe,
            )
        except Exception as e:
            self._detail.set_error(f"Ошибка: {e}")
            return
        if best is None:
            self._detail.set_error(
                "Нет прогона с acceptance_passed=true для текущей пары в журнале."
            )
            return
        for row, r in enumerate(self._runs):
            if r.run_id == best.run_id:
                self._table.selectRow(row)
                self._on_select(row, 0)
                return
        self._pending_select_id = best.run_id
        self._only_pass.setChecked(False)
        self.refresh()
