#!/usr/bin/env python3
"""Сравнение двух JSON отчётов report-real (WFE и acceptance)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path, help="JSON эталон (variant B)")
    p.add_argument("experiment", type=Path, help="JSON эксперимент")
    args = p.parse_args()
    a, b = load(args.baseline), load(args.experiment)
    sa, sb = a.get("summary") or {}, b.get("summary") or {}

    print(f"Baseline:    {args.baseline.name}")
    print(f"Experiment:  {args.experiment.name}\n")
    for key in (
        "n_folds",
        "mean_sharpe",
        "mean_profit_factor",
        "mean_wfe",
        "mean_wfe_sharpe",
        "mean_total_return_pct",
        "worst_max_drawdown_pct",
        "mean_recovery_factor",
        "total_trade_events",
        "folds_positive_return",
    ):
        va, vb = sa.get(key), sb.get(key)
        if va is None and vb is None:
            continue
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"  (Δ {vb - va:+.4f})" if key.startswith("mean") or key == "n_folds" else ""
        print(f"  {key}: {va} → {vb}{delta}")

    print(f"\n  acceptance_passed: {a.get('acceptance_passed')} → {b.get('acceptance_passed')}")
    for label, data in (("baseline", a), ("experiment", b)):
        for c in data.get("acceptance_checks") or []:
            if c.get("name") == "min_wfe":
                print(f"  [{label}] min_wfe: {c.get('detail')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
