"""Вкладка исполнения — paper trading и сверка с Backtester."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.execution_api import PaperExecutionSession
from gui.api.types import ExecutionAccountSnapshot, ExecutionStepResult, OrderRecord
from gui.app.widgets.help_label import help_label
from gui.app.workers import PaperCompareWorker, PaperConnectWorker, PaperStepWorker

try:
    import numpy as np
    import pyqtgraph as pg

    _HAS_PG = True
except ImportError:
    _HAS_PG = False


class ExecutionView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._api = api or IstGuiClient()
        self._session: Optional[PaperExecutionSession] = None
        self._connect_worker: Optional[PaperConnectWorker] = None
        self._step_worker: Optional[PaperStepWorker] = None
        self._compare_worker: Optional[PaperCompareWorker] = None
        self._last_step: Optional[ExecutionStepResult] = None
        self._orders: List[OrderRecord] = []
        self._balance_history: List[float] = []

        layout = QVBoxLayout(self)

        about = help_label(
            "<b>Исполнение (paper)</b> — симуляция сделок на последнем баре: "
            "inference → сигнал → ордер в виртуальном брокере. "
            "Не путать с графиком OKX: здесь проверяется цепочка «решение → позиция → комиссии». "
            "Сверка с Backtester показывает расхождение тайминга (close vs shift(1)).",
            background="#e8f5e9",
            border="#a5d6a7",
        )
        layout.addWidget(about)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Режим:"))
        self._mode = QComboBox()
        self._mode.addItems(["Paper (симуляция)", "Live (нужны ключи OKX)"])
        self._mode.setCurrentIndex(0)
        self._mode.setEnabled(False)
        mode_row.addWidget(self._mode)
        self._live_hint = QLabel("")
        self._live_hint.setStyleSheet("color: #666;")
        mode_row.addWidget(self._live_hint, stretch=1)
        layout.addLayout(mode_row)

        conn = QHBoxLayout()
        self._btn_connect = QPushButton("Подключить paper")
        self._btn_disconnect = QPushButton("Отключить")
        self._btn_disconnect.setEnabled(False)
        self._btn_reset = QPushButton("Сброс портфеля")
        self._btn_emergency = QPushButton("Аварийная остановка")
        self._btn_emergency.setEnabled(False)
        conn.addWidget(self._btn_connect)
        conn.addWidget(self._btn_disconnect)
        conn.addWidget(self._btn_reset)
        conn.addWidget(self._btn_emergency)
        conn.addStretch()
        layout.addLayout(conn)

        acct_box = QGroupBox("Счёт")
        acct_form = QFormLayout(acct_box)
        self._balance = QLabel("—")
        self._pnl = QLabel("—")
        self._fees = QLabel("—")
        self._status_conn = QLabel("Не подключено")
        acct_form.addRow("Статус:", self._status_conn)
        acct_form.addRow("Баланс:", self._balance)
        acct_form.addRow("PnL к началу:", self._pnl)
        acct_form.addRow("Комиссии:", self._fees)
        layout.addWidget(acct_box)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(QLabel("Позиции"))
        self._positions = QTableWidget(0, 4)
        self._positions.setHorizontalHeaderLabels(
            ["Символ", "Кол-во", "Вход", "uPnL"]
        )
        self._positions.horizontalHeader().setStretchLastSection(True)
        left_l.addWidget(self._positions)

        left_l.addWidget(QLabel("Ордера"))
        self._orders_table = QTableWidget(0, 6)
        self._orders_table.setHorizontalHeaderLabels(
            ["ID", "Сторона", "Статус", "Кол-во", "Цена", "Комиссия"]
        )
        self._orders_table.horizontalHeader().setStretchLastSection(True)
        left_l.addWidget(self._orders_table)
        if _HAS_PG:
            self._bal_plot = pg.PlotWidget(title="Баланс по шагам")
            self._bal_plot.setMaximumHeight(160)
            self._bal_plot.showGrid(x=True, y=True, alpha=0.3)
            left_l.addWidget(self._bal_plot)
        else:
            self._bal_plot = None
        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        step_box = QGroupBox("Инференс + исполнение (последний бар)")
        step_l = QVBoxLayout(step_box)
        row = QHBoxLayout()
        self._btn_step = QPushButton("Выполнить шаг")
        self._btn_step.setEnabled(False)
        self._auto = QComboBox()
        self._auto.addItems(["Вручную", "Авто каждые 30 с", "Авто каждые 60 с"])
        self._window = QSpinBox()
        self._window.setRange(64, 512)
        self._window.setValue(256)
        row.addWidget(self._btn_step)
        row.addWidget(QLabel("Окно:"))
        row.addWidget(self._window)
        row.addWidget(self._auto)
        step_l.addLayout(row)
        self._step_log = QPlainTextEdit()
        self._step_log.setReadOnly(True)
        step_l.addWidget(self._step_log)

        compare_box = QGroupBox("Сверка с Backtester (тот же бар)")
        compare_l = QVBoxLayout(compare_box)
        self._compare_text = QPlainTextEdit()
        self._compare_text.setReadOnly(True)
        self._compare_text.setPlaceholderText(
            "После шага появится сравнение paper и векторного бэктеста…"
        )
        self._compare_text.setMaximumHeight(220)
        compare_l.addWidget(self._compare_text)
        step_l.addWidget(compare_box)

        right_l.addWidget(step_box)
        split.addWidget(right)
        split.setSizes([480, 400])
        layout.addWidget(split, stretch=1)

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._run_step)

        self._btn_connect.clicked.connect(self._connect_paper)
        self._btn_disconnect.clicked.connect(self._disconnect)
        self._btn_reset.clicked.connect(self._reset_portfolio)
        self._btn_emergency.clicked.connect(self._emergency_stop)
        self._btn_step.clicked.connect(self._run_step)
        self._auto.currentIndexChanged.connect(self._on_auto_changed)

        self._update_live_hint()

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        self._update_live_hint()
        if self._session and self._session.connected:
            self._refresh_account()

    def _update_live_hint(self) -> None:
        if self._api.execution.live_available():
            self._live_hint.setText("Ключи OKX в env найдены — live в следующих версиях")
        else:
            self._live_hint.setText("Для live: OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE")

    def _connect_paper(self) -> None:
        if self._connect_worker and self._connect_worker.isRunning():
            return
        self._btn_connect.setEnabled(False)
        self._connect_worker = PaperConnectWorker(api=self._api)
        self._connect_worker.finished.connect(self._on_connected)
        self._connect_worker.failed.connect(self._on_connect_failed)
        self._connect_worker.start()

    def _on_connected(self, session: PaperExecutionSession) -> None:
        self._session = session
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)
        self._btn_step.setEnabled(True)
        self._btn_emergency.setEnabled(True)
        self._btn_reset.setEnabled(True)
        self._status_conn.setText("Подключено (paper)")
        self._status_conn.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self._append_log("Paper-сессия подключена.")
        self._balance_history = [float(session.config.initial_balance)]
        self._refresh_account()

    def _on_connect_failed(self, msg: str) -> None:
        self._btn_connect.setEnabled(True)
        self._append_log(f"Ошибка подключения: {msg}")

    def _disconnect(self) -> None:
        self._auto_timer.stop()
        if self._session:
            self._session.disconnect()
            self._session = None
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._btn_step.setEnabled(False)
        self._btn_emergency.setEnabled(False)
        self._status_conn.setText("Не подключено")
        self._status_conn.setStyleSheet("color: #c62828;")
        self._append_log("Отключено.")
        self._compare_text.clear()

    def _reset_portfolio(self) -> None:
        if self._session:
            self._session.reset()
            self._balance_history.clear()
            self._plot_balance()
            self._append_log("Портфель сброшен.")
            self._refresh_account()

    def _emergency_stop(self) -> None:
        if not self._session:
            return
        n = self._session.emergency_stop()
        self._auto_timer.stop()
        self._append_log(f"Аварийная остановка — отменено ордеров: {n}.")
        self._refresh_account()

    def _on_auto_changed(self, index: int) -> None:
        self._auto_timer.stop()
        if index == 1:
            self._auto_timer.start(30_000)
        elif index == 2:
            self._auto_timer.start(60_000)

    def _run_step(self) -> None:
        if not self._session or not self._session.connected:
            self._append_log("Сначала подключите paper-сессию.")
            return
        if self._step_worker and self._step_worker.isRunning():
            return
        self._btn_step.setEnabled(False)
        self._step_worker = PaperStepWorker(
            self._session,
            self._symbol,
            self._timeframe,
            window=self._window.value(),
        )
        self._step_worker.finished.connect(self._on_step)
        self._step_worker.failed.connect(self._on_step_failed)
        self._step_worker.start()

    def _on_step(self, result: ExecutionStepResult) -> None:
        self._btn_step.setEnabled(True)
        self._last_step = result
        line = (
            f"[{result.as_of}] {result.direction} сигнал={result.signal} "
            f"p={result.meta_probability:.4f} qty={result.quantity:.6f} @ {result.price:.2f} — "
            f"{result.message}"
        )
        self._append_log(line)
        self._refresh_account()
        self._run_compare(result)

    def _run_compare(self, step: ExecutionStepResult) -> None:
        if self._compare_worker and self._compare_worker.isRunning():
            return
        self._compare_text.setPlainText("Сверка с Backtester…")
        self._compare_worker = PaperCompareWorker(
            step, self._symbol, self._timeframe, api=self._api
        )
        self._compare_worker.finished.connect(self._on_compare)
        self._compare_worker.failed.connect(self._on_compare_failed)
        self._compare_worker.start()

    def _on_compare(self, cmp_result) -> None:
        self._compare_text.setPlainText(cmp_result.to_russian_text())

    def _on_compare_failed(self, msg: str) -> None:
        self._compare_text.setPlainText(f"Ошибка сверки: {msg}")

    def _on_step_failed(self, msg: str) -> None:
        self._btn_step.setEnabled(True)
        self._append_log(f"Ошибка шага: {msg}")

    def _refresh_account(self) -> None:
        if not self._session:
            return
        snap = self._session.account_snapshot()
        self._apply_account(snap)
        self._orders = self._session.order_history(limit=40)
        self._fill_orders_table(self._orders)

    def _apply_account(self, snap: ExecutionAccountSnapshot) -> None:
        self._balance.setText(f"{snap.balance:,.2f} USDT")
        pnl = snap.balance - snap.initial_balance
        self._pnl.setText(f"{pnl:+,.2f} USDT ({100 * pnl / snap.initial_balance:+.2f}%)")
        self._fees.setText(f"{snap.total_fees:.4f}")
        bal = float(snap.balance)
        if not self._balance_history or abs(self._balance_history[-1] - bal) > 1e-6:
            self._balance_history.append(bal)
        self._plot_balance()

        self._positions.setRowCount(len(snap.positions))
        for row, p in enumerate(snap.positions):
            self._positions.setItem(row, 0, QTableWidgetItem(p.symbol))
            self._positions.setItem(row, 1, QTableWidgetItem(f"{p.quantity:.6f}"))
            self._positions.setItem(row, 2, QTableWidgetItem(f"{p.entry_price:.2f}"))
            self._positions.setItem(row, 3, QTableWidgetItem(f"{p.unrealized_pnl:+.2f}"))

    def _fill_orders_table(self, orders: List[OrderRecord]) -> None:
        self._orders_table.setRowCount(len(orders))
        for row, o in enumerate(orders):
            cells = [
                o.order_id[:14],
                o.side,
                o.status,
                f"{o.quantity:.6f}",
                f"{o.filled_price:.2f}" if o.filled_price else "—",
                f"{o.fees:.4f}",
            ]
            for col, text in enumerate(cells):
                self._orders_table.setItem(row, col, QTableWidgetItem(text))

    def _plot_balance(self) -> None:
        if not _HAS_PG or self._bal_plot is None or not self._balance_history:
            if _HAS_PG and self._bal_plot is not None:
                self._bal_plot.clear()
            return
        self._bal_plot.clear()
        ys = np.array(self._balance_history, dtype=float)
        xs = np.arange(len(ys))
        self._bal_plot.plot(xs, ys, pen=pg.mkPen("#2e7d32", width=2), symbol="o", symbolSize=6)

    def _append_log(self, text: str) -> None:
        self._step_log.appendPlainText(text)
