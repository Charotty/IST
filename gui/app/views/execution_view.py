"""Вкладка «Практика» — replay, сигналы на графике, калибровка прогноза."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.app.widgets.help_label import help_label
from gui.app.workers import (
    PaperCalibrationWorker,
    PaperReplayWorker,
    PaperSignalsWorker,
)

try:
    import numpy as np
    import pyqtgraph as pg

    _HAS_PG = True
except ImportError:
    _HAS_PG = False


class ExecutionView(QWidget):
    """Практическая проверка модели (frozen bundle, без OKX API)."""

    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._api = api or IstGuiClient()
        self._replay_worker: Optional[PaperReplayWorker] = None
        self._signals_worker: Optional[PaperSignalsWorker] = None
        self._cal_worker: Optional[PaperCalibrationWorker] = None
        self._last_decisions = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            help_label(
                "<b>Практика</b> — проверка работы модели без биржи: прогон по последним N барам "
                "на <b>замороженном bundle</b> (без переобучения). "
                "Replay сравнивает paper-брокер и Backtester; график сигналов показывает, "
                "когда система входит в рынок; калибровка — средняя доходность следующего бара "
                "по корзинам meta_probability (канон shift(1)).",
                background="#e8f5e9",
                border="#a5d6a7",
            )
        )

        params = QHBoxLayout()
        params.addWidget(QLabel("Баров:"))
        self._n_bars = QSpinBox()
        self._n_bars.setRange(50, 2000)
        self._n_bars.setValue(300)
        params.addWidget(self._n_bars)
        params.addWidget(QLabel("Окно inference:"))
        self._window = QSpinBox()
        self._window.setRange(64, 512)
        self._window.setValue(256)
        params.addWidget(self._window)
        params.addWidget(QLabel("Старт USDT:"))
        self._balance = QSpinBox()
        self._balance.setRange(1000, 1_000_000)
        self._balance.setSingleStep(1000)
        self._balance.setValue(10_000)
        params.addWidget(self._balance)
        self._bundle_lbl = QLabel("Bundle: —")
        self._bundle_lbl.setStyleSheet("color: #555;")
        params.addWidget(self._bundle_lbl, stretch=1)
        layout.addLayout(params)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_replay_tab(), "1. Практический прогон")
        self._tabs.addTab(self._build_signals_tab(), "2. Сигналы и цена")
        self._tabs.addTab(self._build_calibration_tab(), "3. Калибровка")
        layout.addWidget(self._tabs, stretch=1)

    def _build_replay_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self._btn_replay = QPushButton("Запустить replay")
        self._btn_replay.clicked.connect(self._run_replay)
        row.addWidget(self._btn_replay)
        self._replay_status = QLabel("")
        row.addWidget(self._replay_status, stretch=1)
        v.addLayout(row)

        self._replay_summary = QLabel("—")
        self._replay_summary.setWordWrap(True)
        v.addWidget(self._replay_summary)

        if _HAS_PG:
            self._replay_plot = pg.PlotWidget(title="Equity: Paper vs Backtester")
            self._replay_plot.addLegend(offset=(10, 10))
            self._replay_plot.showGrid(x=True, y=True, alpha=0.3)
            self._replay_plot.setLabel("left", "USDT")
            self._replay_plot.setMinimumHeight(220)
            v.addWidget(self._replay_plot)
        else:
            self._replay_plot = None
            v.addWidget(QLabel("Установите pyqtgraph для графиков."))

        v.addWidget(QLabel("Сделки paper"))
        self._trades_table = QTableWidget(0, 6)
        self._trades_table.setHorizontalHeaderLabels(
            ["Время", "Сторона", "Кол-во", "Цена", "Комиссия", "Баланс"]
        )
        self._trades_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self._trades_table)
        return w

    def _build_signals_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self._btn_signals = QPushButton("Построить график")
        self._btn_signals.clicked.connect(self._run_signals)
        row.addWidget(self._btn_signals)
        self._signals_status = QLabel("")
        row.addWidget(self._signals_status, stretch=1)
        v.addLayout(row)

        if _HAS_PG:
            self._sig_plot = pg.PlotWidget(title="Close и сигналы")
            self._sig_plot.showGrid(x=True, y=True, alpha=0.3)
            self._sig_plot.setLabel("left", "Цена")
            self._sig_plot.setMinimumHeight(280)
            v.addWidget(self._sig_plot, stretch=2)
            self._p_plot = pg.PlotWidget(title="Meta probability")
            self._p_plot.showGrid(x=True, y=True, alpha=0.3)
            self._p_plot.setLabel("left", "p")
            self._p_plot.setMaximumHeight(140)
            v.addWidget(self._p_plot)
        else:
            self._sig_plot = None
            self._p_plot = None
        self._signals_legend = QLabel(
            "● long  ● short  |  фон: trend (светло-зелёный) / range (светло-серый)"
        )
        self._signals_legend.setStyleSheet("color: #666;")
        v.addWidget(self._signals_legend)
        return w

    def _build_calibration_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Корзин:"))
        self._n_buckets = QSpinBox()
        self._n_buckets.setRange(4, 16)
        self._n_buckets.setValue(8)
        row.addWidget(self._n_buckets)
        self._btn_cal = QPushButton("Рассчитать калибровку")
        self._btn_cal.clicked.connect(self._run_calibration)
        row.addWidget(self._btn_cal)
        self._cal_status = QLabel("")
        row.addWidget(self._cal_status, stretch=1)
        v.addLayout(row)
        self._cal_summary = QLabel("—")
        self._cal_summary.setWordWrap(True)
        v.addWidget(self._cal_summary)
        if _HAS_PG:
            self._cal_plot = pg.PlotWidget(title="Средняя net-доходность след. бара (%) по p[t-1]")
            self._cal_plot.showGrid(x=True, y=True, alpha=0.3)
            self._cal_plot.setMinimumHeight(260)
            v.addWidget(self._cal_plot, stretch=1)
        else:
            self._cal_plot = None
        return w

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._bundle_lbl.setText("Bundle: (запустите расчёт)")

    def refresh(self) -> None:
        pass

    def _guard_demo(self) -> bool:
        if getattr(self._api, "demo", False):
            self._replay_status.setText("Демо-режим: нужны реальные bundle и parquet.")
            return True
        return False

    def _run_replay(self) -> None:
        if self._guard_demo():
            return
        if self._replay_worker and self._replay_worker.isRunning():
            return
        self._btn_replay.setEnabled(False)
        self._replay_status.setText("Считаю replay (inference + paper + backtester)…")
        self._replay_worker = PaperReplayWorker(
            self._symbol,
            self._timeframe,
            n_bars=self._n_bars.value(),
            window=self._window.value(),
            initial_balance=float(self._balance.value()),
            api=self._api,
        )
        self._replay_worker.finished.connect(self._on_replay_done)
        self._replay_worker.failed.connect(self._on_replay_fail)
        self._replay_worker.start()

    def _on_replay_done(self, result) -> None:
        self._btn_replay.setEnabled(True)
        self._replay_status.setText("Готово")
        self._bundle_lbl.setText(f"Bundle: …{result.bundle_dir[-40:]}")
        self._last_decisions = result.decisions
        self._replay_summary.setText(
            f"<b>{result.n_bars}</b> баров · сигналов ≠0: <b>{result.n_nonzero_signals}</b> · "
            f"сделок paper: <b>{result.n_trades}</b><br>"
            f"Доходность (equity MTM): paper <b>{result.total_return_paper_pct:+.2f}%</b> · "
            f"Backtester shift(1): <b>{result.total_return_bt_pct:+.2f}%</b><br>"
            f"Капитал: {result.initial_balance:,.0f} → {result.final_balance_paper:,.2f} USDT "
            f"(cash+позиция; не только cash после buy)"
        )
        self._fill_trades(result.trades)
        self._plot_replay_equity(result.equity_paper, result.equity_backtester)

    def _on_replay_fail(self, msg: str) -> None:
        self._btn_replay.setEnabled(True)
        self._replay_status.setText(f"Ошибка: {msg}")

    def _fill_trades(self, trades) -> None:
        self._trades_table.setRowCount(len(trades))
        for row, t in enumerate(trades):
            cells = [
                t.as_of[:19],
                t.side,
                f"{t.quantity:.6f}",
                f"{t.price:.2f}",
                f"{t.fees:.4f}",
                f"{t.balance_after:,.2f}",
            ]
            for col, text in enumerate(cells):
                self._trades_table.setItem(row, col, QTableWidgetItem(text))

    def _plot_replay_equity(self, paper: List[float], bt: List[float]) -> None:
        if not _HAS_PG or self._replay_plot is None:
            return
        self._replay_plot.clear()
        n = min(len(paper), len(bt))
        if n < 1:
            return
        xs = np.arange(n)
        self._replay_plot.plot(
            xs,
            np.array(paper[:n]),
            pen=pg.mkPen("#2e7d32", width=2),
            name="Paper",
        )
        self._replay_plot.plot(
            xs,
            np.array(bt[:n]),
            pen=pg.mkPen("#1565c0", width=2),
            name="Backtester",
        )

    def _run_signals(self) -> None:
        if self._guard_demo():
            return
        if self._signals_worker and self._signals_worker.isRunning():
            return
        self._btn_signals.setEnabled(False)
        self._signals_status.setText("Inference по барам…")
        self._signals_worker = PaperSignalsWorker(
            self._symbol,
            self._timeframe,
            n_bars=self._n_bars.value(),
            window=self._window.value(),
            api=self._api,
        )
        self._signals_worker.finished.connect(self._on_signals_done)
        self._signals_worker.failed.connect(self._on_signals_fail)
        self._signals_worker.start()

    def _on_signals_done(self, payload: Tuple[str, list]) -> None:
        self._btn_signals.setEnabled(True)
        bundle_dir, decisions = payload
        self._signals_status.setText(f"Готово · {len(decisions)} баров")
        self._bundle_lbl.setText(f"Bundle: …{bundle_dir[-40:]}")
        self._last_decisions = decisions
        self._plot_signals(decisions)

    def _on_signals_fail(self, msg: str) -> None:
        self._btn_signals.setEnabled(True)
        self._signals_status.setText(f"Ошибка: {msg}")

    def _add_regime_bands(self, decisions) -> None:
        if not decisions or self._sig_plot is None:
            return
        start = 0
        cur = decisions[0].regime
        for i in range(1, len(decisions)):
            if decisions[i].regime != cur:
                self._add_one_regime_band(start, i - 1, cur)
                start = i
                cur = decisions[i].regime
        self._add_one_regime_band(start, len(decisions) - 1, cur)

    def _add_one_regime_band(self, i0: int, i1: int, regime: str) -> None:
        color = QColor(200, 230, 200, 50) if regime == "trend" else QColor(210, 210, 210, 40)
        region = pg.LinearRegionItem(
            values=(i0 - 0.5, i1 + 0.5),
            movable=False,
            brush=color,
        )
        region.setZValue(-10)
        self._sig_plot.addItem(region)

    def _plot_signals(self, decisions) -> None:
        if not _HAS_PG or self._sig_plot is None:
            return
        n = len(decisions)
        xs = np.arange(n)
        closes = np.array([d.close for d in decisions], dtype=float)
        ps = np.array([d.meta_probability for d in decisions], dtype=float)

        self._sig_plot.clear()
        self._sig_plot.plot(xs, closes, pen=pg.mkPen("#333", width=1.5))

        self._add_regime_bands(decisions)

        long_x, long_y, short_x, short_y, flat_x, flat_y = [], [], [], [], [], []
        for i, d in enumerate(decisions):
            if d.signal > 0:
                long_x.append(i)
                long_y.append(d.close)
            elif d.signal < 0:
                short_x.append(i)
                short_y.append(d.close)
            else:
                flat_x.append(i)
                flat_y.append(d.close)

        if long_x:
            self._sig_plot.plot(
                long_x,
                long_y,
                pen=None,
                symbol="t",
                symbolBrush="#2e7d32",
                symbolSize=12,
            )
        if short_x:
            self._sig_plot.plot(
                short_x,
                short_y,
                pen=None,
                symbol="t1",
                symbolBrush="#c62828",
                symbolSize=12,
            )
        if flat_x:
            self._sig_plot.plot(
                flat_x,
                flat_y,
                pen=None,
                symbol="o",
                symbolBrush="#9e9e9e",
                symbolSize=5,
            )

        if self._p_plot is not None:
            self._p_plot.clear()
            self._p_plot.plot(xs, ps, pen=pg.mkPen("#6a1b9a", width=1))
            self._p_plot.addLine(y=0.5, pen=pg.mkPen("#999", style=Qt.PenStyle.DashLine))
            thr = 0.56
            self._p_plot.addLine(y=thr, pen=pg.mkPen("#2e7d32", style=Qt.PenStyle.DotLine))
            self._p_plot.addLine(y=1 - thr, pen=pg.mkPen("#c62828", style=Qt.PenStyle.DotLine))

    def _run_calibration(self) -> None:
        if self._guard_demo():
            return
        if self._cal_worker and self._cal_worker.isRunning():
            return
        self._btn_cal.setEnabled(False)
        self._cal_status.setText("Считаю…")
        self._cal_worker = PaperCalibrationWorker(
            self._symbol,
            self._timeframe,
            n_bars=max(self._n_bars.value(), 200),
            window=self._window.value(),
            n_buckets=self._n_buckets.value(),
            api=self._api,
        )
        self._cal_worker.finished.connect(self._on_cal_done)
        self._cal_worker.failed.connect(self._on_cal_fail)
        self._cal_worker.start()

    def _on_cal_done(self, result) -> None:
        self._btn_cal.setEnabled(True)
        self._cal_status.setText("Готово")
        self._bundle_lbl.setText(f"Bundle: …{result.bundle_dir[-40:]}")
        total = sum(b.count for b in result.buckets)
        self._cal_summary.setText(
            f"Пар (p[t-1] → net[t]) по <b>{result.n_bars}</b> барам · "
            f"<b>{total}</b> точек в корзинах · канон Backtester shift(1)"
        )
        self._plot_calibration(result.buckets)

    def _on_cal_fail(self, msg: str) -> None:
        self._btn_cal.setEnabled(True)
        self._cal_status.setText(f"Ошибка: {msg}")

    def _plot_calibration(self, buckets) -> None:
        if not _HAS_PG or self._cal_plot is None:
            return
        self._cal_plot.clear()
        xs = np.arange(len(buckets))
        heights = np.array([b.mean_forward_net_pct for b in buckets], dtype=float)
        counts = [b.count for b in buckets]
        bg = pg.BarGraphItem(x=xs, height=heights, width=0.7, brush="#1565c0")
        self._cal_plot.addItem(bg)
        for i, (b, h) in enumerate(zip(buckets, heights)):
            if b.count > 0:
                self._cal_plot.plot([i], [h], pen=None, symbol="o", symbolSize=8)
        labels = [f"{b.label}\n(n={b.count})" for b in buckets]
        ax = self._cal_plot.getAxis("bottom")
        ax.setTicks([[(i, labels[i].replace("\n", " ")) for i in range(len(labels))]])
