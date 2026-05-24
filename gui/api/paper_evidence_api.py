"""Paper replay, сигналы и калибровка для вкладки «Практика»."""

from __future__ import annotations

from orchestration.paper_evidence import (
    BarDecision,
    CalibrationResult,
    PaperReplayResult,
    build_calibration,
    collect_bar_decisions,
    run_paper_replay,
)


class PaperEvidenceApi:
    def collect_decisions(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        n_bars: int = 300,
        window: int = 256,
    ) -> tuple[str, list[BarDecision]]:
        return collect_bar_decisions(symbol, timeframe, n_bars=n_bars, window=window)

    def run_replay(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        n_bars: int = 300,
        window: int = 256,
        initial_balance: float = 10_000.0,
        max_position_fraction: float = 0.35,
    ) -> PaperReplayResult:
        return run_paper_replay(
            symbol,
            timeframe,
            n_bars=n_bars,
            window=window,
            initial_balance=initial_balance,
            max_position_fraction=max_position_fraction,
        )

    def calibration(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        n_bars: int = 500,
        window: int = 256,
        n_buckets: int = 8,
    ) -> CalibrationResult:
        return build_calibration(
            symbol,
            timeframe,
            n_bars=n_bars,
            window=window,
            n_buckets=n_buckets,
        )
