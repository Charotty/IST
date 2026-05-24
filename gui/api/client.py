"""
Единая точка входа API для PyQt6 GUI.

Не дублирует логику pipeline — только обёртки над ``orchestration``,
``backtesting``, ``orchestration.introspect``.

Пример::

    from gui.api import IstGuiClient

    api = IstGuiClient()
    symbols = api.symbols.list_symbols()
    card = api.inference.explain("BTC/USDT", "1h")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui.api.backtests_api import BacktestsApi
from gui.api.bundles_api import BundlesApi
from gui.api.cli_api import CliApi
from gui.api.demo_api import (
    DemoBacktestsApi,
    DemoBundlesApi,
    DemoCliApi,
    DemoExecutionApi,
    DemoInferenceApi,
    DemoJobsApi,
    DemoReconcileApi,
    DemoSymbolsApi,
)
from gui.api.execution_api import ExecutionApi
from gui.api.reconcile_api import ReconcileApi
from gui.api.paper_evidence_api import PaperEvidenceApi
from gui.api.inference_api import InferenceApi
from gui.api.jobs_api import JobsApi
from gui.api.config_api import ConfigApi
from gui.api.symbols_api import SymbolsApi


class IstGuiClient:
    """
    Facade for desktop UI workers and widgets.

    Threading: методы синхронные; в PyQt6 вызывайте из ``QThread`` /
    ``QRunnable`` для тяжёлых вызовов (``inference.chart_bars``,
    ``inference.explain`` с DL).
    """

    def __init__(
        self,
        *,
        demo: bool = False,
        repo_root: Optional[Path] = None,
        journal_root: Optional[Path] = None,
    ):
        self.demo = demo
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.config = ConfigApi()
        if demo:
            self.symbols = DemoSymbolsApi()
            self.bundles = DemoBundlesApi()
            self.inference = DemoInferenceApi()
            self.backtests = DemoBacktestsApi()
            self.jobs = DemoJobsApi()
            self.cli = DemoCliApi()
            self.execution = DemoExecutionApi()
            self.reconcile = DemoReconcileApi()
            self.paper_evidence = PaperEvidenceApi()
        else:
            self.symbols = SymbolsApi()
            self.bundles = BundlesApi()
            self.inference = InferenceApi()
            self.backtests = BacktestsApi(journal_root=journal_root)
            self.jobs = JobsApi()
            self.cli = CliApi()
            self.execution = ExecutionApi()
            self.reconcile = ReconcileApi()
            self.paper_evidence = PaperEvidenceApi()
