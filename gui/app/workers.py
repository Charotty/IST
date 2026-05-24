"""Background workers — keep orchestration/ML off the UI thread."""

from __future__ import annotations

from typing import Any, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from gui.api import IstGuiClient
from gui.api.execution_api import PaperExecutionSession
from gui.api.types import (
    BacktestRunSummary,
    BundleInfo,
    ChartPayload,
    DataHealth,
    ExplainSnapshot,
    ExecutionAccountSnapshot,
    ExecutionStepResult,
    FoldEquityPoint,
    PipelineStepStatus,
    RegimeBar,
    SymbolEntry,
)


class _BaseWorker(QThread):
    failed = pyqtSignal(str)

    def __init__(self, api: Optional[IstGuiClient] = None):
        super().__init__()
        self._api = api or IstGuiClient()


class LoadSymbolsWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def run(self) -> None:
        try:
            self.finished.emit(self._api.symbols.list_symbols())
        except Exception as e:
            self.failed.emit(str(e))


class LoadOkxMarketsWorker(_BaseWorker):
    """Список USDT spot пар с OKX."""

    finished = pyqtSignal(list)

    def run(self) -> None:
        from gui.api.okx_api import list_usdt_spot_symbols

        try:
            self.finished.emit(list_usdt_spot_symbols())
        except Exception as e:
            self.failed.emit(str(e))


class ResolveSymbolWorker(_BaseWorker):
    """Локальный статус parquet / bundle для пары из тулбара."""

    finished = pyqtSignal(object)

    def __init__(self, symbol: str, timeframe: str, api: Optional[IstGuiClient] = None):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe

    def run(self) -> None:
        try:
            entry: SymbolEntry = self._api.symbols.resolve(self._symbol, self._tf)
            self.finished.emit(entry)
        except Exception as e:
            self.failed.emit(str(e))


class OkxChartWorker(_BaseWorker):
    """OKX OHLCV: ``history`` (пагинация) или ``live`` (хвост + тикер)."""

    finished = pyqtSignal(object)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 300,
        mode: str = "history",
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._limit = limit
        self._mode = mode

    def run(self) -> None:
        from gui.api import okx_api
        from gui.api.types import ChartPayload

        try:
            if self._mode in ("tail", "live"):
                bars = okx_api.fetch_okx_live_tail(
                    self._symbol, self._tf, ohlcv_limit=3, use_ticker=True
                )
            else:
                bars = okx_api.fetch_ohlcv_history(
                    self._symbol, self._tf, max_bars=self._limit
                )
            self.finished.emit(ChartPayload(bars=bars, regime_segments=[]))
        except Exception as e:
            if getattr(self._api, "demo", False):
                from gui.api import demo_data

                self.finished.emit(
                    demo_data.demo_chart_payload(
                        self._symbol, self._tf, max_bars=self._limit
                    )
                )
            else:
                self.failed.emit(str(e))


class ExplainWorker(_BaseWorker):
    finished = pyqtSignal(object)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        window: int = 256,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._window = window

    def run(self) -> None:
        try:
            snap: ExplainSnapshot = self._api.inference.explain(
                self._symbol, self._tf, window=self._window
            )
            self.finished.emit(snap)
        except Exception as e:
            self.failed.emit(str(e))


class BundleInfoWorker(_BaseWorker):
    finished = pyqtSignal(object)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        validate_schema: bool = False,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._validate_schema = validate_schema

    def run(self) -> None:
        try:
            info: BundleInfo = self._api.bundles.bundle_info(
                self._symbol,
                self._tf,
                validate_schema=self._validate_schema,
            )
            self.finished.emit(info)
        except Exception as e:
            self.failed.emit(str(e))


class PipelineStatusWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def __init__(self, symbol: str, timeframe: str, api: Optional[IstGuiClient] = None):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe

    def run(self) -> None:
        try:
            steps: List[PipelineStepStatus] = self._api.jobs.pipeline_status(
                self._symbol, self._tf
            )
            health: DataHealth = self._api.symbols.data_health(self._symbol, self._tf)
            self.finished.emit([steps, health])
        except Exception as e:
            self.failed.emit(str(e))


class ChartPayloadWorker(_BaseWorker):
    finished = pyqtSignal(object)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        max_bars: int = 1200,
        window: int = 384,
        regime_step: int = 1,
        include_signals: bool = True,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._max_bars = max_bars
        self._window = window
        self._step = regime_step
        self._include_signals = include_signals

    def run(self) -> None:
        try:
            payload: ChartPayload = self._api.inference.chart_payload(
                self._symbol,
                self._tf,
                max_bars=self._max_bars,
                window=self._window,
                regime_step=self._step,
                include_signals=self._include_signals,
            )
            self.finished.emit(payload)
        except Exception as e:
            self.failed.emit(str(e))


class RegimeSeriesWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        step: int = 1,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._step = step

    def run(self) -> None:
        try:
            rows = self._api.inference.regime_series(
                self._symbol, self._tf, step=self._step
            )
            self.finished.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


class ManifestWorker(_BaseWorker):
    finished = pyqtSignal(object)

    def __init__(self, symbol: str, timeframe: str, api: Optional[IstGuiClient] = None):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe

    def run(self) -> None:
        from orchestration.symbol_pipeline import read_symbol_manifest
        from orchestration.symbols import paths_for

        try:
            sp = paths_for(self._symbol, self._tf)
            man = read_symbol_manifest(sp)
            self.finished.emit(man or {})
        except Exception as e:
            self.failed.emit(str(e))


class SymbolsListWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def run(self) -> None:
        try:
            self.finished.emit(self._api.symbols.list_symbols())
        except Exception as e:
            self.failed.emit(str(e))


class CliProcessWorker(_BaseWorker):
    """Run ``python -m orchestration …`` in a background thread."""

    line = pyqtSignal(str)
    finished_code = pyqtSignal(int)

    def __init__(self, argv: List[str], cwd: Optional[str] = None):
        super().__init__(api=None)
        self._argv = list(argv)
        self._cwd = cwd
        self._proc = None
        self._emitted_done = False

    def _emit_done(self, code: int) -> None:
        if self._emitted_done:
            return
        self._emitted_done = True
        self.finished_code.emit(int(code))

    def run(self) -> None:
        import subprocess

        code = -1
        try:
            self._proc = subprocess.Popen(
                self._argv,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self.isInterruptionRequested():
                    break
                self.line.emit(line.rstrip("\n\r"))
            code = self._proc.wait()
        except Exception as e:
            self.line.emit(f"ERROR: {e}")
            code = -1
        finally:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
            self._proc = None
            self._emit_done(code)

    def stop_process(self) -> None:
        """Остановить subprocess без убийства QThread (кнопка «Стоп»)."""
        self.requestInterruption()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


class TaskLogWorker(_BaseWorker):
    finished = pyqtSignal(str)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        task_id: str,
        *,
        tail: int = 150,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._task_id = task_id
        self._tail = tail

    def run(self) -> None:
        try:
            if not self._task_id or self._task_id.startswith("("):
                self.finished.emit("")
                return
            events = self._api.jobs.read_task_log(
                self._symbol, self._tf, self._task_id, tail=self._tail
            )
            self.finished.emit("\n".join(str(e.raw) for e in events))
        except Exception as e:
            self.finished.emit(f"Ошибка чтения лога: {e}")


class FoldEquityWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def __init__(self, run_id: str, api: Optional[IstGuiClient] = None):
        super().__init__(api)
        self._run_id = run_id

    def run(self) -> None:
        try:
            run = self._api.backtests.get_run(self._run_id)
            curve: List[FoldEquityPoint] = self._api.backtests.fold_equity_curve(run)
            self.finished.emit(curve)
        except Exception as e:
            self.failed.emit(str(e))


class BacktestListWorker(_BaseWorker):
    finished = pyqtSignal(list)

    def __init__(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = 100,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__(api)
        self._symbol = symbol
        self._tf = timeframe
        self._limit = limit

    def run(self) -> None:
        try:
            norm_sym = None
            if self._symbol:
                norm_sym = self._api.symbols.normalize_symbol(self._symbol)
            kwargs: dict = {"symbol": norm_sym, "timeframe": self._tf}
            if self._limit is not None:
                kwargs["limit"] = self._limit
            rows: List[BacktestRunSummary] = self._api.backtests.list_runs(
                **kwargs
            )
            self.finished.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


class PaperConnectWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        max_position_fraction: float = 0.35,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__()
        self._api = api or IstGuiClient()
        self._balance = initial_balance
        self._max_frac = max_position_fraction

    def run(self) -> None:
        try:
            session = self._api.execution.create_paper_session(
                initial_balance=self._balance,
                max_position_fraction=self._max_frac,
            )
            if not session.connect():
                raise RuntimeError("Paper broker connect failed")
            self.finished.emit(session)
        except Exception as e:
            self.failed.emit(str(e))


class PaperStepWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        session: PaperExecutionSession,
        symbol: str,
        timeframe: str,
        *,
        window: int = 256,
    ):
        super().__init__()
        self._session = session
        self._symbol = symbol
        self._tf = timeframe
        self._window = window

    def run(self) -> None:
        try:
            step: ExecutionStepResult = self._session.run_step(
                self._symbol, self._tf, window=self._window
            )
            self.finished.emit(step)
        except Exception as e:
            self.failed.emit(str(e))


class PaperCompareWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        step: ExecutionStepResult,
        symbol: str,
        timeframe: str,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__()
        self._api = api or IstGuiClient()
        self._step = step
        self._symbol = symbol
        self._tf = timeframe

    def run(self) -> None:
        try:
            cmp_result = self._api.reconcile.compare_paper_step(
                self._step, self._symbol, self._tf
            )
            self.finished.emit(cmp_result)
        except Exception as e:
            self.failed.emit(str(e))


class PaperAccountWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, session: PaperExecutionSession):
        super().__init__()
        self._session = session

    def run(self) -> None:
        snap: ExecutionAccountSnapshot = self._session.account_snapshot()
        self.finished.emit(snap)


class PaperReplayWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        n_bars: int = 300,
        window: int = 256,
        initial_balance: float = 10_000.0,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__()
        self._api = api or IstGuiClient()
        self._symbol = symbol
        self._tf = timeframe
        self._n_bars = n_bars
        self._window = window
        self._balance = initial_balance

    def run(self) -> None:
        try:
            result = self._api.paper_evidence.run_replay(
                self._symbol,
                self._tf,
                n_bars=self._n_bars,
                window=self._window,
                initial_balance=self._balance,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class PaperSignalsWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        n_bars: int = 300,
        window: int = 256,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__()
        self._api = api or IstGuiClient()
        self._symbol = symbol
        self._tf = timeframe
        self._n_bars = n_bars
        self._window = window

    def run(self) -> None:
        try:
            bundle_dir, decisions = self._api.paper_evidence.collect_decisions(
                self._symbol,
                self._tf,
                n_bars=self._n_bars,
                window=self._window,
            )
            self.finished.emit((bundle_dir, decisions))
        except Exception as e:
            self.failed.emit(str(e))


class PaperCalibrationWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        n_bars: int = 500,
        window: int = 256,
        n_buckets: int = 8,
        api: Optional[IstGuiClient] = None,
    ):
        super().__init__()
        self._api = api or IstGuiClient()
        self._symbol = symbol
        self._tf = timeframe
        self._n_bars = n_bars
        self._window = window
        self._n_buckets = n_buckets

    def run(self) -> None:
        try:
            result = self._api.paper_evidence.calibration(
                self._symbol,
                self._tf,
                n_bars=self._n_bars,
                window=self._window,
                n_buckets=self._n_buckets,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
