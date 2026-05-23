"""IST main window — 3 вкладки: График, Задачи, Конфигурация."""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget

from gui.api import IstGuiClient
from gui.api.types import SymbolEntry
from gui.app import i18n_ru as ru
from gui.app.views.chart_hub_view import ChartHubView
from gui.app.views.jobs_hub_view import JobsHubView
from gui.app.views.regime_view import RegimeView
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
        layout = QVBoxLayout(central)

        self._toolbar = SymbolToolbar()
        layout.addWidget(self._toolbar)

        self._chart_hub = ChartHubView(self._api)
        self._regime = RegimeView(self._api)
        self._jobs_hub = JobsHubView(self._api)
        self._config = SettingsView(self._api)

        self._tabs = QTabWidget()
        self._views: Dict[str, QWidget] = {
            ru.TAB_CHART: self._chart_hub,
            ru.TAB_REGIME: self._regime,
            ru.TAB_JOBS: self._jobs_hub,
            ru.TAB_CONFIG: self._config,
        }
        for name, view in self._views.items():
            self._tabs.addTab(view, name)

        self._jobs_hub.pipeline_finished.connect(self._on_pipeline_finished)
        self._config.config_saved.connect(self._on_config_saved)

        layout.addWidget(self._tabs, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Загрузка списка пар OKX…")

        self._toolbar.symbol_changed.connect(self._on_symbol_changed)
        self._tabs.currentChanged.connect(self._on_tab_changed)

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
        self._jobs_hub.refresh_journal()
        self._refresh_current_tab()

    def _on_config_saved(self) -> None:
        self._resolve_pipeline_status()
        self._chart_hub.overview.refresh()
        self._refresh_current_tab()

    def _on_symbol_changed(self, symbol: str, timeframe: str) -> None:
        for view in self._views.values():
            if hasattr(view, "set_context"):
                view.set_context(symbol, timeframe)
        self._resolve_pipeline_status()
        self._refresh_current_tab()

    def _on_tab_changed(self, _index: int) -> None:
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
