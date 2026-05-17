"""Backtest journal access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtesting.results_journal import BacktestResultsJournal

from gui.api.types import BacktestRunSummary, FoldEquityPoint


class BacktestsApi:
    def __init__(self, journal_root: Optional[Path] = None):
        self._journal = BacktestResultsJournal(journal_root or "docs/backtest_journal")

    @property
    def journal_root(self) -> Path:
        return self._journal.root

    def list_runs(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        stage: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BacktestRunSummary]:
        rows = self._journal.list_runs(symbol=symbol, timeframe=timeframe, stage=stage)
        if limit is not None:
            rows = rows[-limit:]
        return [
            BacktestRunSummary(
                run_id=r["run_id"],
                timestamp_utc=r.get("timestamp_utc", ""),
                label=r.get("label", ""),
                symbol=r.get("symbol"),
                timeframe=r.get("timeframe"),
                stage=r.get("stage"),
                acceptance_passed=r.get("acceptance_passed"),
                target_passed=r.get("target_passed"),
                summary=dict(r.get("summary") or {}),
            )
            for r in reversed(rows)
        ]

    def get_run(self, run_id: str) -> Dict[str, Any]:
        path = self._journal.runs_dir / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run snapshot not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def best_acceptance(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Optional[BacktestRunSummary]:
        row = self._journal.best_acceptance_run(
            symbol=symbol, timeframe=timeframe, stage=stage
        )
        if row is None:
            return None
        return BacktestRunSummary(
            run_id=row["run_id"],
            timestamp_utc=row.get("timestamp_utc", ""),
            label=row.get("label", ""),
            symbol=row.get("symbol"),
            timeframe=row.get("timeframe"),
            stage=row.get("stage"),
            acceptance_passed=row.get("acceptance_passed"),
            target_passed=row.get("target_passed"),
            summary=dict(row.get("summary") or {}),
        )

    def fold_equity_curve(self, run: Dict[str, Any]) -> List[FoldEquityPoint]:
        """
        Approximate equity from per-fold OOS ``Total Return (%)`` (compounded).
        Journal snapshots do not store bar-level equity.
        """
        folds = run.get("fold_metrics") or []
        if not isinstance(folds, list):
            return []
        cum = 1.0
        out: List[FoldEquityPoint] = []
        for i, fold in enumerate(folds):
            if not isinstance(fold, dict):
                continue
            ret = float(fold.get("Total Return (%)", 0.0) or 0.0)
            cum *= 1.0 + ret / 100.0
            fold_no = int(fold.get("Fold", i + 1))
            out.append(
                FoldEquityPoint(
                    fold=fold_no,
                    oos_return_pct=ret,
                    cumulative_pct=(cum - 1.0) * 100.0,
                )
            )
        return out
