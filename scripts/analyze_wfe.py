#!/usr/bin/env python3
"""Разбор WFE по JSON отчёта report-real (folds + acceptance_checks)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(folds: pd.DataFrame) -> None:
    if folds.empty:
        print("Нет фолдов в отчёте.")
        return

    wfe = pd.to_numeric(folds["Walk-Forward Efficiency"], errors="coerce")
    sh_oos = pd.to_numeric(folds["Sharpe Ratio"], errors="coerce")
    sh_is = pd.to_numeric(folds.get("IS_Sharpe Ratio"), errors="coerce")
    ret_oos = pd.to_numeric(folds["Total Return (%)"], errors="coerce")
    ret_is = pd.to_numeric(folds["IS_Total Return (%)"], errors="coerce")

    wfe_sh = sh_oos / sh_is.replace(0, np.nan)

    print("=== Walk-Forward Efficiency (IST) ===\n")
    print(f"Фолдов: {len(folds)}")
    print(f"mean WFE (return ratio):     {wfe.mean():.4f}")
    print(f"median WFE:                  {wfe.median():.4f}")
    print(f"фолдов WFE >= 0.5:           {(wfe >= 0.5).sum()}")
    print(f"фолдов WFE < 0:              {(wfe < 0).sum()}")
    print()
    print(f"mean OOS Sharpe:             {sh_oos.mean():.4f}")
    print(f"mean IS Sharpe:              {sh_is.mean():.4f}")
    print(f"mean WFE (Sharpe ratio):     {wfe_sh.mean():.4f}")
    print(f"median WFE (Sharpe ratio):   {wfe_sh.median():.4f}")
    print(f"corr(WFE, OOS Sharpe):       {wfe.corr(sh_oos):.4f}")
    print()

    acc = folds.copy()
    acc["WFE"] = wfe
    acc["WFE_sharpe"] = wfe_sh
    cols = ["Fold", "WFE", "WFE_sharpe", "Sharpe Ratio", "IS_Sharpe Ratio", "Total Return (%)", "IS_Total Return (%)"]
    cols = [c for c in cols if c in acc.columns or c in ("WFE", "WFE_sharpe")]
    show = [c for c in cols if c in acc.columns]
    print("--- Худшие 5 фолдов по WFE ---")
    print(acc.sort_values("WFE").head(5)[show].to_string(index=False))
    print()
    print("--- Лучшие 5 фолдов по WFE ---")
    print(acc.sort_values("WFE", ascending=False).head(5)[show].to_string(index=False))
    print()

    flat = (ret_oos.abs() < 0.01) & (wfe == 0)
    if flat.any():
        print(f"Фолды с ~0% OOS return и WFE=0: {list(acc.loc[flat, 'Fold'].astype(int))}")
        print("(часто мало сделок / фильтр волатильности)\n")

    print("--- Интерпретация ---")
    if wfe.mean() < 0.5 and sh_oos.mean() > 0.5:
        print(
            "Низкий mean WFE при высоком OOS Sharpe: типично завышенный IS-бэктест на train-окне.\n"
            "См. docs/WFE_ANALYSIS.md"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(REPO / "docs" / "thesis_btc_4model_acceptance.json"),
        help="JSON из report-real --json-out",
    )
    args = parser.parse_args()
    path = Path(args.json_path)
    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1
    data = load_report(path)
    folds = pd.DataFrame(data.get("folds") or [])
    summary = data.get("summary") or {}
    checks = data.get("acceptance_checks") or []

    print(f"Отчёт: {path}\n")
    if summary:
        print(
            f"summary: mean_sharpe={summary.get('mean_sharpe')}, "
            f"mean_wfe={summary.get('mean_wfe')}, "
            f"acceptance_passed={data.get('acceptance_passed')}\n"
        )
    for c in checks:
        if c.get("name") == "min_wfe":
            print(f"acceptance min_wfe: {c.get('detail')}\n")
            break

    analyze(folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
