"""IST main window — tab shell + symbol toolbar."""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import SymbolEntry
from gui.app import i18n_ru as ru
from gui.app.views.backtests_view import BacktestsView
from gui.app.views.chart_view import ChartView
from gui.app.views.execution_view import ExecutionView
from gui.app.views.jobs_view import JobsView
from gui.app.views.models_view import ModelsView
from gui.app.views.overview_view import OverviewView
from gui.app.views.settings_view import SettingsView
from gui.app.widgets.symbol_toolbar import SymbolToolbar
from gui.app.workers import LoadOkxMarketsWorker, ResolveSymbolWorker


class MainWindow(QMainWindow):
    def __init__(self, api: Optional[IstGuiClient] = None):
        super().__init__()
        self._api = api or IstGuiClient()
        self._settings = QSettings("IST", "Desktop")
        self.setWindowTitle(ru.WINDOW_TITLE)
        self.resize(1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        from PyQt6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(central)

        self._toolbar = SymbolToolbar()
        layout.addWidget(self._toolbar)

        self._tabs = QTabWidget()
        self._views: Dict[str, QWidget] = {
            ru.TAB_OVERVIEW: OverviewView(self._api),
            ru.TAB_CHART: ChartView(self._api),
            ru.TAB_MODELS: ModelsView(self._api),
            ru.TAB_BACKTESTS: BacktestsView(self._api),
            ru.TAB_EXECUTION: ExecutionView(self._api),
            ru.TAB_JOBS: JobsView(self._api),
            ru.TAB_CONFIG: SettingsView(self._api),
        }
        for name, view in self._views.items():
            self._tabs.addTab(view, name)
            if name == ru.TAB_JOBS and hasattr(view, "pipeline_finished"):
                view.pipeline_finished.connect(self._on_pipeline_finished)
        layout.addWidget(self._tabs, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Загрузка списка пар OKX…")

        self._toolbar.symbol_changed.connect(self._on_symbol_changed)
        self._tabs.currentChanged.connect(self._refresh_current_tab)

        interval = int(self._settings.value("refresh_interval_sec", 0) or 0)
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._refresh_current_tab)
        if interval > 0:
            self._auto_timer.start(interval * 1000)

        self._resolve_worker: Optional[ResolveSymbolWorker] = None
        self._load_markets()

    def _load_markets(self) -> None:
        self._markets_worker = LoadOkxMarketsWorker(self._api)
        self._markets_worker.finished.connect(self._on_markets)
        self._markets_worker.failed.connect(self._on_markets_failed)
        self._markets_worker.start()

    def _on_markets(self, symbols: list) -> None:
        self._toolbar.set_markets(symbols)
        self.statusBar().showMessage(f"Пар OKX: {len(symbols)}")
        self._resolve_pipeline_status()

    def _on_markets_failed(self, msg: str) -> None:
        from gui.api.okx_api import _FALLBACK_SYMBOLS

        self._toolbar.set_markets(list(_FALLBACK_SYMBOLS))
        self.statusBar().showMessage(f"OKX недоступен, базовый список ({msg[:60]})")
        self._resolve_pipeline_status()

    def _resolve_pipeline_status(self) -> None:
        if self._resolve_worker and self._resolve_worker.isRunning():
            return
        sym = self._toolbar.current_symbol()
        tf = self._toolbar.current_timeframe()
        self._resolve_worker = ResolveSymbolWorker(sym, tf, api=self._api)
        self._resolve_worker.finished.connect(self._on_resolved)
        self._resolve_worker.failed.connect(lambda _: self._toolbar.set_pipeline_status(None))
        self._resolve_worker.start()

    def _on_resolved(self, entry: SymbolEntry) -> None:
        if not self._api.demo:
            self._toolbar.set_pipeline_status(entry)
        else:
            entry.has_bundle = True
            entry.latest_bundle_run_id = entry.latest_bundle_run_id or "demo_bundle"
            self._toolbar.set_pipeline_status(entry)

    def _on_pipeline_finished(self) -> None:
        self._resolve_pipeline_status()
        self._refresh_current_tab()

    def _on_symbol_changed(self, symbol: str, timeframe: str) -> None:
        for view in self._views.values():
            if hasattr(view, "set_context"):
                view.set_context(symbol, timeframe)
        self._resolve_pipeline_status()
        self._refresh_current_tab()

    def _refresh_current_tab(self) -> None:
        view = self._views.get(self._tabs.tabText(self._tabs.currentIndex()))
        if view and hasattr(view, "refresh"):
            sym = self._toolbar.current_symbol()
            tf = self._toolbar.current_timeframe()
            self.statusBar().showMessage(
                f"{self._tabs.tabText(self._tabs.currentIndex())} — {sym} {tf}"
            )
            view.refresh()
