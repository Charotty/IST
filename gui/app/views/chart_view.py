"""График — OKX (история + live-хвост) или IST (chart_payload + regime + сигналы)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import ChartBar, ChartPayload
from gui.app.chart_bar_merge import LiveBarChange, LiveMergeResult, merge_chart_bars
from gui.app.widgets.chart_layers import ChartPlotController
from gui.app.widgets.help_label import help_label
from gui.app.workers import ChartPayloadWorker, OkxChartWorker

try:
    import pyqtgraph as pg

    _HAS_PG = True
except ImportError:
    _HAS_PG = False

# OKX: всего баров (пагинация); IST: окно отображения
_OKX_BAR_OPTIONS = ["300", "900", "1500", "3000"]
_IST_BAR_OPTIONS = ["200", "500", "800", "1200"]

_OKX_POLL_MS = {"15m": 20_000, "1h": 30_000, "4h": 90_000, "1d": 180_000}
_IST_POLL_MS = {"15m": 60_000, "1h": 120_000, "4h": 300_000, "1d": 600_000}


def _poll_interval_ms(timeframe: str, *, okx: bool) -> int:
    table = _OKX_POLL_MS if okx else _IST_POLL_MS
    return table.get(timeframe, 30_000 if okx else 120_000)


class ChartView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._worker: Optional[OkxChartWorker] = None
        self._local_worker: Optional[ChartPayloadWorker] = None
        self._payload: Optional[ChartPayload] = None
        self._in_flight = False
        self._last_updated: Optional[datetime] = None
        self._okx_history_ready = False
        self._live_tick = False
        self._plot = ChartPlotController()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(lambda: self._reload(full=True))

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        # OKX: частый опрос хвоста + тикер (формирующаяся свеча), как series.update()
        self._forming_timer = QTimer(self)
        self._forming_timer.setInterval(3000)
        self._forming_timer.timeout.connect(self._on_forming_tick)

        layout = QVBoxLayout(self)
        if not _HAS_PG:
            layout.addWidget(
                QLabel("Установите pyqtgraph: pip install -r requirements-gui.txt")
            )
            return

        layout.addWidget(
            help_label(
                "<b>OKX</b> — загружается <b>история</b> (прокрутка влево колёсиком/мышью); "
                "в <b>Live</b> обновляется только последняя свеча или добавляется новая.<br>"
                "<b>IST</b> — локальные данные + regime + сигналы на хвосте окна inference.",
                background="#e8eaf6",
                border="#9fa8da",
            )
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Источник:"))
        self._source = QComboBox()
        self._source.addItems(["OKX (биржа)", "IST (локально)"])
        row.addWidget(self._source)

        row.addWidget(QLabel("Свечей:"))
        self._limit = QComboBox()
        row.addWidget(self._limit)

        row.addWidget(QLabel("Шаг regime:"))
        self._regime_step = QSpinBox()
        self._regime_step.setRange(1, 48)
        self._regime_step.setValue(1)
        row.addWidget(self._regime_step)

        self._signals_cb = QCheckBox("Сигналы (bundle)")
        self._signals_cb.setChecked(True)
        row.addWidget(self._signals_cb)

        row.addWidget(QLabel("Окно inference:"))
        self._window = QSpinBox()
        self._window.setRange(64, 1024)
        self._window.setValue(384)
        row.addWidget(self._window)

        self._live_cb = QCheckBox("Live")
        self._live_cb.setChecked(True)
        self._live_cb.setToolTip(
            "OKX: подтягивает хвост без сброса масштаба; IST: обновление с сохранением zoom"
        )
        row.addWidget(self._live_cb)

        self._btn_refresh = QPushButton("Сейчас")
        self._btn_refresh.setToolTip("Перезагрузить историю (F5)")
        self._btn_refresh.clicked.connect(lambda: self._reload(full=True))
        row.addWidget(self._btn_refresh)
        row.addStretch()
        layout.addLayout(row)

        self._price_plot = pg.PlotWidget(title="Цена")
        self._price_plot.showGrid(x=True, y=True, alpha=0.3)
        self._price_plot.setMouseEnabled(x=True, y=True)
        self._meta_plot = pg.PlotWidget(title="Meta P(up)")
        self._meta_plot.showGrid(x=True, y=True, alpha=0.3)
        self._meta_plot.setMaximumHeight(140)
        self._meta_plot.setVisible(False)
        self._plot.attach(self._price_plot, self._meta_plot)

        layout.addWidget(self._price_plot, stretch=3)
        layout.addWidget(self._meta_plot, stretch=1)

        self._status = QLabel(
            "Прокрутка: колёсико / перетаскивание. Live добавляет свечи справа без перезагрузки."
        )
        layout.addWidget(self._status)

        self._source.currentIndexChanged.connect(self._on_source_changed)
        self._limit.currentTextChanged.connect(self._schedule_reload)
        self._regime_step.valueChanged.connect(self._schedule_reload)
        self._window.valueChanged.connect(self._schedule_reload)
        self._signals_cb.toggled.connect(self._on_signals_toggled)
        self._live_cb.toggled.connect(self._sync_poll_timer)

        self._populate_limit_combo(okx_mode=True)

    def _is_okx_mode(self) -> bool:
        return self._source.currentIndex() == 0

    def _populate_limit_combo(self, *, okx_mode: bool) -> None:
        current = self._limit.currentText()
        options = _OKX_BAR_OPTIONS if okx_mode else _IST_BAR_OPTIONS
        self._limit.blockSignals(True)
        self._limit.clear()
        self._limit.addItems(options)
        if current in options:
            self._limit.setCurrentText(current)
        else:
            self._limit.setCurrentText(options[1] if okx_mode else "500")
        self._limit.blockSignals(False)

    def _on_source_changed(self) -> None:
        okx = self._is_okx_mode()
        self._populate_limit_combo(okx_mode=okx)
        local = not okx
        self._regime_step.setEnabled(local)
        self._signals_cb.setEnabled(local)
        self._window.setEnabled(local)
        self._on_signals_toggled()
        self._reset_series()
        self._sync_poll_timer()
        self._schedule_reload()

    def _reset_series(self) -> None:
        self._okx_history_ready = False
        self._payload = None
        self._plot.clear()

    def _on_signals_toggled(self) -> None:
        local = self._source.currentIndex() == 1
        self._meta_plot.setVisible(
            local and self._signals_cb.isChecked() and self._signals_cb.isEnabled()
        )
        if local:
            self._schedule_reload()

    def _schedule_reload(self) -> None:
        if not _HAS_PG:
            return
        self._debounce.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if _HAS_PG:
            self._sync_poll_timer()
            if self._payload is None and not self._in_flight:
                self._reload(full=True)
            elif self._live_cb.isChecked():
                self._sync_poll_timer()

    def hideEvent(self, event) -> None:
        self._poll_timer.stop()
        self._forming_timer.stop()
        super().hideEvent(event)

    def _sync_poll_timer(self) -> None:
        self._poll_timer.stop()
        self._forming_timer.stop()
        if not _HAS_PG or not self._live_cb.isChecked() or not self.isVisible():
            return
        if self._is_okx_mode():
            self._forming_timer.start()
        else:
            ms = _poll_interval_ms(self._timeframe, okx=False)
            self._poll_timer.setInterval(ms)
            self._poll_timer.start()

    def _on_forming_tick(self) -> None:
        if not self.isVisible() or not self._live_cb.isChecked() or not self._is_okx_mode():
            return
        if not self._okx_history_ready:
            self._reload(full=True)
            return
        self._reload(full=False)

    def _on_poll_tick(self) -> None:
        if not self.isVisible() or not self._live_cb.isChecked() or self._is_okx_mode():
            return
        self._reload(full=False)

    def set_context(self, symbol: str, timeframe: str) -> None:
        prev = (self._symbol, self._timeframe)
        self._symbol = symbol
        self._timeframe = timeframe
        if prev != (symbol, timeframe):
            self._reset_series()
            self._sync_poll_timer()

    def _reload(self, *, full: bool) -> None:
        if not _HAS_PG:
            return
        if self._in_flight:
            return
        if (self._worker and self._worker.isRunning()) or (
            self._local_worker and self._local_worker.isRunning()
        ):
            return

        self._live_tick = not full
        self._in_flight = True
        limit = int(self._limit.currentText())

        if self._is_okx_mode():
            self._meta_plot.setVisible(False)
            if full or not self._okx_history_ready:
                self._status.setText(
                    f"Загрузка истории OKX {self._symbol} ({limit} баров)…"
                )
                self._worker = OkxChartWorker(
                    self._symbol,
                    self._timeframe,
                    limit=limit,
                    mode="history",
                    api=self._api,
                )
            else:
                self._worker = OkxChartWorker(
                    self._symbol,
                    self._timeframe,
                    mode="live",
                    api=self._api,
                )
            self._worker.finished.connect(self._on_okx_payload)
            self._worker.failed.connect(self._on_fail)
            self._worker.start()
        else:
            self._on_signals_toggled()
            if full or self._payload is None:
                self._status.setText(f"Загрузка IST {self._symbol}…")
            self._local_worker = ChartPayloadWorker(
                self._symbol,
                self._timeframe,
                max_bars=limit,
                window=self._window.value(),
                regime_step=self._regime_step.value(),
                include_signals=self._signals_cb.isChecked(),
                api=self._api,
            )
            self._local_worker.finished.connect(self._on_local_payload)
            self._local_worker.failed.connect(self._on_fail)
            self._local_worker.start()

    def refresh(self, *, force: bool = True) -> None:
        """API для MainWindow: ``force`` = полная перезагрузка истории."""
        if force:
            self._reset_series()
        self._reload(full=force)

    def _live_suffix(self) -> str:
        if not self._live_cb.isChecked():
            return ""
        if self._is_okx_mode():
            sec = self._forming_timer.interval() // 1000
            return f" · live {sec}с (тикер+свеча)"
        sec = self._poll_timer.interval() // 1000
        return f" · live {sec}с"

    def _updated_suffix(self) -> str:
        if self._last_updated is None:
            return ""
        t = self._last_updated.astimezone().strftime("%H:%M:%S")
        return f" · обновлено {t}"

    def _on_fail(self, msg: str) -> None:
        was_live = self._live_tick
        self._in_flight = False
        self._live_tick = False
        if not was_live:
            self._payload = None
        self._status.setText(f"Ошибка: {msg}{self._live_suffix()}")

    def _merge_okx_tail(self, tail: List[ChartBar]) -> LiveMergeResult:
        base = list(self._payload.bars) if self._payload else []
        return merge_chart_bars(base, tail)

    def _on_okx_payload(self, payload: ChartPayload) -> None:
        self._in_flight = False
        live = self._live_tick
        self._live_tick = False
        self._last_updated = datetime.now(timezone.utc)

        if live and self._okx_history_ready and self._payload:
            merged = self._merge_okx_tail(payload.bars)
            bars = merged.bars
            self._plot.update_okx_live(
                bars,
                title=f"OKX · {len(bars)} свечей",
                change=merged.change,
            )
            self._payload = ChartPayload(bars=bars, regime_segments=[])
            if merged.change == LiveBarChange.NEW_BAR:
                hint = " · новая свеча"
            elif merged.change == LiveBarChange.FORMING:
                hint = " · формируется"
            else:
                hint = ""
            self._set_okx_status(bars, hint=hint)
            return

        bars = payload.bars
        self._payload = ChartPayload(bars=bars, regime_segments=[])
        self._okx_history_ready = True
        self._plot.full_render(
            self._payload,
            title=f"OKX · {len(bars)} свечей (история)",
            focus_recent=True,
        )
        self._set_okx_status(bars, hint=" · история")

    def _set_okx_status(self, bars: List[ChartBar], *, hint: str = "") -> None:
        if not bars:
            self._status.setText(f"OKX: нет данных{self._live_suffix()}")
            return
        self._status.setText(
            f"OKX: {len(bars)} свечей{hint}"
            f"{self._live_suffix()}{self._updated_suffix()} · "
            f"прокрутка влево — раньше · "
            f"{bars[0].t[:16]} … {bars[-1].t[:16]} · close {bars[-1].close:,.2f}"
        )

    def _on_local_payload(self, payload: ChartPayload) -> None:
        self._in_flight = False
        live = self._live_tick
        self._live_tick = False
        self._last_updated = datetime.now(timezone.utc)
        preserve = live and self._payload is not None
        self._payload = payload
        meta = self._signals_cb.isChecked()
        n_seg = len(payload.regime_segments)
        n_sig = sum(1 for b in payload.bars if b.signal)
        self._plot.full_render_from_bars(
            payload.bars,
            regime_segments=payload.regime_segments,
            meta_plot=meta,
            title=f"IST · {len(payload.bars)} баров · regime сегм. {n_seg}",
            preserve_view=preserve,
        )

        if payload.bars:
            b = payload.bars
            win = self._window.value()
            self._status.setText(
                f"IST: {len(b)} баров · сигналов {n_sig} на хвосте {win}"
                f"{self._live_suffix()}{self._updated_suffix()} · "
                f"regime step={self._regime_step.value()} · "
                f"{b[0].t[:16]} … {b[-1].t[:16]}"
            )
        else:
            self._status.setText(
                "Нет локальных данных — нужен OHLCV parquet и/или features"
                f"{self._live_suffix()}"
            )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F5:
            self.refresh(force=True)
            event.accept()
            return
        super().keyPressEvent(event)
