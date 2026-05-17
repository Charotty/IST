"""
Отдельный журнал прогонов бэктеста / WFO: JSONL + снимки + INDEX.md.

Не путать с ``docs/TEST_RESULTS_LOG.md`` (pytest/CI). Здесь только метрики стратегий.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class BacktestResultsJournal:
    """Append-only журнал в ``docs/backtest_journal/``."""

    def __init__(self, root: str | Path = "docs/backtest_journal"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.jsonl_path = self.root / "runs.jsonl"
        self.index_path = self.root / "INDEX.md"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def append_run(
        self,
        report: Dict[str, Any],
        *,
        label: str = "",
        params: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> str:
        """
        Сохраняет полный отчёт и строку в JSONL. Возвращает ``run_id``.
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        acceptance = report.get("criteria", {}).get("acceptance", {})
        target = report.get("criteria", {}).get("target", {})
        summary = report.get("summary") or {}

        params = dict(params or {})
        profile = report.get("profile") or params.get("profile")
        if profile and "profile" not in params:
            params["profile"] = profile

        entry = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "label": label,
            "notes": notes,
            "profile": profile,
            "params": params,
            "symbol": report.get("symbol"),
            "timeframe": report.get("timeframe"),
            "stage": report.get("stage"),
            "parquet": report.get("parquet"),
            "feature_rows": report.get("feature_rows"),
            "max_rows": report.get("max_rows"),
            "train_span": report.get("train_span"),
            "holdout_span": report.get("holdout_span"),
            "acceptance_passed": acceptance.get("passed"),
            "target_passed": target.get("passed"),
            "summary": summary,
            "acceptance_checks": acceptance.get("checks"),
            "target_checks": target.get("checks"),
        }

        snapshot = {
            **entry,
            "fold_metrics": _folds_to_records(report.get("fold_metrics")),
            "criteria": report.get("criteria"),
        }
        snap_path = self.runs_dir / f"{run_id}.json"
        snap_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        self.regenerate_index()
        return run_id

    def regenerate_index(self) -> None:
        """Пересобирает INDEX.md из runs.jsonl."""
        rows: List[Dict[str, Any]] = []
        if self.jsonl_path.is_file():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        lines = [
            "# Журнал результатов бэктеста (IST)",
            "",
            "Отдельный от `TEST_RESULTS_LOG.md`: только WFO/бэктест и метрики на **реальных** OHLCV.",
            "",
            f"Обновлено: {_utc_now_iso()} · записей: **{len(rows)}**",
            "",
            "| run_id | UTC | symbol/tf | stage | label | folds | acc | tgt | mean Sharpe | mean PF | mean WFE | mean ret % |",
            "|--------|-----|-----------|-------|-------|-------|-----|-----|-------------|---------|----------|------------|",
        ]
        for r in reversed(rows[-200:]):
            s = r.get("summary") or {}
            sym = r.get("symbol") or ""
            tf = r.get("timeframe") or ""
            sym_tf = f"{sym}/{tf}" if sym or tf else ""
            lines.append(
                "| {rid} | {ts} | {symtf} | {st} | {lbl} | {nf} | {acc} | {tgt} | {sh:.3f} | {pf:.3f} | {wfe:.3f} | {ret:.2f} |".format(
                    rid=r.get("run_id", "")[:20],
                    ts=(r.get("timestamp_utc") or "")[:19],
                    symtf=sym_tf[:18],
                    st=(r.get("stage") or "")[:10],
                    lbl=(r.get("label") or "")[:24],
                    nf=s.get("n_folds", ""),
                    acc="PASS" if r.get("acceptance_passed") else "FAIL",
                    tgt="PASS" if r.get("target_passed") else "FAIL",
                    sh=float(s.get("mean_sharpe") or 0),
                    pf=float(s.get("mean_profit_factor") or 0),
                    wfe=float(s.get("mean_wfe") or 0),
                    ret=float(s.get("mean_total_return_pct") or 0),
                )
            )
        lines.extend(
            [
                "",
                "Полные снимки: `runs/<run_id>.json` · поток: `runs.jsonl`.",
                "",
            ]
        )
        self.index_path.write_text("\n".join(lines), encoding="utf-8")

    def list_runs(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.jsonl_path.is_file():
            return []
        rows = [
            json.loads(ln)
            for ln in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if symbol:
            rows = [r for r in rows if (r.get("symbol") or "") == symbol]
        if timeframe:
            rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
        if stage:
            rows = [r for r in rows if (r.get("stage") or "") == stage]
        return rows

    def best_acceptance_run(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        passed = [
            r for r in self.list_runs(symbol=symbol, timeframe=timeframe, stage=stage)
            if r.get("acceptance_passed")
        ]
        if not passed:
            return None
        return max(
            passed,
            key=lambda r: (r.get("summary") or {}).get("mean_sharpe", float("-inf")),
        )


def _folds_to_records(folds: Any) -> List[Dict[str, Any]]:
    if folds is None:
        return []
    if isinstance(folds, pd.DataFrame):
        return folds.to_dict(orient="records")
    if isinstance(folds, list):
        return folds
    return []
