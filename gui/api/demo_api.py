"""API-заглушки для демо-режима GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from gui.api import demo_data
from gui.api.backtests_api import BacktestsApi
from gui.api.bundles_api import BundlesApi
from gui.api.cli_api import CliApi
from gui.api.execution_api import ExecutionApi
from gui.api.inference_api import InferenceApi
from gui.api.jobs_api import JobsApi
from gui.api.reconcile_api import ReconcileApi
from gui.api.symbols_api import SymbolsApi
from gui.api.types import (
    BacktestRunSummary,
    BundleInfo,
    ChartPayload,
    DataHealth,
    ExplainSnapshot,
    FoldEquityPoint,
    JobEvent,
    PipelineStepStatus,
    SymbolEntry,
)
from gui.api.demo_data import DemoPaperSession


class DemoSymbolsApi(SymbolsApi):
    def list_symbols(self) -> List[SymbolEntry]:
        return demo_data.demo_symbols()

    def resolve(self, symbol: str, timeframe: str = "1h") -> SymbolEntry:
        norm = symbol.replace("-", "/").upper()
        if "/" not in norm and "-" in norm:
            norm = norm.replace("-", "/", 1)
        for e in demo_data.demo_symbols():
            es = e.symbol.replace("-", "/")
            if es == norm and e.timeframe == timeframe:
                return e
        base = demo_data.demo_symbols()[0]
        return SymbolEntry(
            slug=f"{norm.replace('/', '-')}_{timeframe}",
            symbol=norm if "/" in norm else base.symbol,
            timeframe=timeframe,
            has_bundle=True,
            latest_bundle_run_id="20260517T120000Z_demo01",
        )

    def merged_config(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        return demo_data.demo_merged_config(symbol, timeframe)

    def data_health(self, symbol: str, timeframe: str = "1h") -> DataHealth:
        return demo_data.demo_data_health(symbol, timeframe)


class DemoInferenceApi(InferenceApi):
    def explain(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        window: int = 256,
    ) -> ExplainSnapshot:
        return demo_data.demo_explain(symbol, timeframe)

    def chart_payload(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        max_bars: int = 1200,
        window: int = 384,
        regime_step: int = 1,
        include_signals: bool = True,
    ) -> ChartPayload:
        return demo_data.demo_chart_payload(symbol, timeframe, max_bars=max_bars)


class DemoBundlesApi(BundlesApi):
    def bundle_info(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        bundle_dir: Optional[Path] = None,
        validate_schema: bool = True,
    ) -> BundleInfo:
        return demo_data.demo_bundle_info(symbol, timeframe)

    def latest_bundle_dir(self, symbol: str, timeframe: str = "1h") -> Optional[Path]:
        return demo_data._DEMO_ROOT


class DemoBacktestsApi(BacktestsApi):
    def __init__(self) -> None:
        pass

    @property
    def journal_root(self) -> Path:
        return Path("docs/backtest_journal")

    def list_runs(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        stage: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BacktestRunSummary]:
        rows = demo_data.demo_backtest_runs(symbol=symbol, timeframe=timeframe)
        if stage:
            rows = [r for r in rows if r.stage == stage]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return demo_data.demo_run_detail(run_id)

    def fold_equity_curve(self, run: Dict[str, Any]) -> List[FoldEquityPoint]:
        return demo_data.demo_fold_equity(run)


class DemoJobsApi(JobsApi):
    def pipeline_status(self, symbol: str, timeframe: str = "1h") -> List[PipelineStepStatus]:
        return demo_data.demo_pipeline_status(symbol, timeframe)

    def list_active_tasks(self, symbol: str, timeframe: str = "1h") -> List[str]:
        return ["demo_prepare_20260517"]

    def read_task_log(
        self,
        symbol: str,
        timeframe: str,
        task_id: str,
        *,
        tail: int = 200,
    ) -> List[JobEvent]:
        events = demo_data.demo_task_events(task_id)
        if tail > 0:
            events = events[-tail:]
        return events


class DemoExecutionApi(ExecutionApi):
    def create_paper_session(
        self,
        *,
        initial_balance: float = 10_000.0,
        max_position_fraction: float = 0.35,
    ) -> DemoPaperSession:
        session = DemoPaperSession(initial_balance=initial_balance)
        session.max_position_fraction = max_position_fraction
        return session

    def live_available(self) -> bool:
        return False


class DemoReconcileApi(ReconcileApi):
    def compare_paper_step(self, step, symbol: str, timeframe: str = "1h", **kwargs):
        return demo_data.demo_paper_compare(step)


class DemoCliApi(CliApi):
    """CLI в демо не запускается — команды строятся, но JobsView их блокирует."""
