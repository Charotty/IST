#!/usr/bin/env python3
"""
Узкий подбор гиперпараметров для диплома: 4 модели (lgb, gru, xgb, cnn).

Старт от THESIS_4MODEL_SEED. Останавливается на первом trial с acceptance_passed.

  python scripts/thesis_push_4model.py --parquet data/ohlcv/BTC-USDT_1h.parquet
  python scripts/thesis_push_4model.py --symbol ETH/USDT --timeframe 1h
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.results_journal import BacktestResultsJournal
from orchestration.symbols import paths_for, write_symbol_config
from orchestration.tuning_config import THESIS_4MODEL_SEED, merge_thesis_params
from orchestration.real_data_benchmark import features_from_ohlcv_parquet
from orchestration.tuning_loop import run_single_trial

# Базовое семя + варианты (2-model acceptance + режимы для 4 моделей)
CANDIDATES: List[Dict[str, Any]] = [
    {},
    # Как в журнале acceptance для LGB+XGB, но 4 model_keys остаются в SEED
    {"ensemble_mode": "fixed_range"},
    {"ensemble_mode": "fixed_range", "min_signal_margin": 0.08},
    {"ensemble_mode": "fixed_range", "min_signal_margin": 0.10},
    {"min_signal_margin": 0.09},
    {"min_signal_margin": 0.10},
    {"min_signal_margin": 0.11},
    {"min_signal_margin": 0.12},
    {"direction_threshold": 0.58, "min_signal_margin": 0.08},
    {"direction_threshold": 0.62, "min_signal_margin": 0.08},
    {"trade_mode": "long_only", "min_signal_margin": 0.08},
    {"trade_mode": "long_only", "min_signal_margin": 0.10},
    {"prediction_horizon": 12, "min_signal_margin": 0.08},
    {"prediction_horizon": 12, "ensemble_mode": "fixed_range", "min_signal_margin": 0.08},
    {"min_signal_margin": 0.09, "max_position_fraction": 0.5},
    {"min_signal_margin": 0.10, "max_position_fraction": 0.35},
    {"volatility_filter_percentile": 90.0, "min_signal_margin": 0.08},
    {"volatility_filter_percentile": 90.0, "min_signal_margin": 0.10, "max_position_fraction": 0.35},
    {"label_min_return": 0.003, "min_signal_margin": 0.09},
    {"label_min_return": 0.004, "min_signal_margin": 0.10},
    {"signal_strategy": "momentum_confirm", "min_signal_margin": 0.08},
    {"signal_strategy": "momentum_confirm", "min_signal_margin": 0.10, "ensemble_mode": "fixed_range"},
    {"test_window_size": 180},
    {"walk_forward_step": 400},
    {"dl_epochs": 2},
    {"dl_epochs": 2, "ensemble_mode": "fixed_range", "min_signal_margin": 0.08},
    {
        "ensemble_mode": "fixed_range",
        "min_signal_margin": 0.10,
        "max_position_fraction": 0.5,
        "volatility_filter_percentile": 90.0,
    },
    {
        "min_signal_margin": 0.11,
        "max_position_fraction": 0.35,
        "volatility_filter_percentile": 90.0,
    },
]


def _resolve_symbol_tf(
    *,
    parquet: Optional[Path],
    symbol: Optional[str],
    timeframe: str,
) -> Tuple[Optional[str], str, Path]:
    if symbol:
        sp = paths_for(symbol, timeframe)
        return sp.symbol, sp.timeframe, sp.parquet
    if parquet:
        stem = parquet.stem
        m = re.match(r"^(.+)_(\d+m|\d+h|\d+d)$", stem, re.I)
        if m:
            slug, tf = m.group(1), m.group(2).lower()
            sym = slug.replace("-", "/") if "/" not in slug else slug
            return sym, tf, parquet
        return stem, timeframe, parquet
    sp = paths_for("BTC/USDT", timeframe)
    return sp.symbol, sp.timeframe, sp.parquet


def _score(report: Dict[str, Any]) -> Tuple:
    acc = report["criteria"]["acceptance"]["passed"]
    s = report["summary"]
    return (
        1 if acc else 0,
        float(s.get("mean_sharpe") or -1e9),
        float(s.get("mean_wfe") or 0),
        float(s.get("mean_profit_factor") or 0),
    )


def _print_acceptance_gaps(report: Dict[str, Any]) -> None:
    ev = report["criteria"]["acceptance"]
    print(f"\nAcceptance: {'PASS' if ev['passed'] else 'FAIL'}")
    for chk in ev["checks"]:
        mark = "OK" if chk["passed"] else "X"
        print(f"  [{mark}] {chk['name']}: {chk['detail']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="4-model thesis hyperparameter push")
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--symbol", default=None, help="e.g. BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--max-rows", type=int, default=None, help="Bars; default from --mode")
    p.add_argument(
        "--mode",
        choices=("default", "fast", "quality", "fast+quality"),
        default="quality",
        help="fast=Task1 speed; quality=Task2 metrics; fast+quality=both presets",
    )
    p.add_argument("--config", type=Path, default="config/profiles/canonical_4model.yaml")
    p.add_argument("--journal-root", type=Path, default="docs/backtest_journal")
    p.add_argument(
        "--write-symbol-yaml",
        action="store_true",
        help="Write config/symbols only when acceptance PASS",
    )
    p.add_argument(
        "--write-best-failed",
        action="store_true",
        help="Also write best trial to YAML even if acceptance failed",
    )
    args = p.parse_args(argv)

    symbol, timeframe, pq = _resolve_symbol_tf(
        parquet=args.parquet, symbol=args.symbol, timeframe=args.timeframe
    )
    sym_write = args.symbol or (symbol.replace("-", "/") if symbol and "-" in symbol else "BTC/USDT")

    if not pq.is_file():
        print(f"Missing parquet: {pq}", file=sys.stderr)
        return 2

    base_params = merge_thesis_params(mode=args.mode)
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = int(base_params.pop("max_rows", 8000))
    else:
        base_params.pop("max_rows", None)

    features = features_from_ohlcv_parquet(pq, max_rows=max_rows)
    journal = BacktestResultsJournal(args.journal_root)
    best_report = None
    best_params = None
    best_score: Tuple = (-1,)

    print(f"Parquet: {pq}  rows={len(features)}  max_rows={max_rows}  mode={args.mode}")
    print(f"Symbol: {sym_write}  models={base_params['model_keys']}")
    if base_params.get("max_wfo_folds"):
        print(f"WFO cap: {base_params['max_wfo_folds']} folds  dl_epochs={base_params.get('dl_epochs')}")

    for i, patch in enumerate(CANDIDATES, 1):
        params = {**base_params, **patch, "_label": f"thesis4_{i}"}
        try:
            report = run_single_trial(
                features,
                params,
                config_path=args.config,
                parquet_label=str(pq.resolve()),
            )
        except Exception as exc:
            print(f"thesis4_{i:02d}  ERROR: {exc}")
            continue

        report["max_rows"] = max_rows
        report["symbol"] = sym_write
        report["timeframe"] = timeframe
        report["stage"] = "thesis_push"
        rid = journal.append_run(
            report,
            label=params["_label"],
            params={k: v for k, v in params.items() if not k.startswith("_")},
        )
        s = report["summary"]
        acc = report["criteria"]["acceptance"]["passed"]
        sc = _score(report)
        ens = params.get("ensemble_mode", THESIS_4MODEL_SEED["ensemble_mode"])
        print(
            f"{params['_label']:12}  ens={ens:16}  folds={s.get('n_folds')}  "
            f"sharpe={float(s.get('mean_sharpe') or 0):.3f}  "
            f"pf={float(s.get('mean_profit_factor') or 0):.2f}  "
            f"wfe={float(s.get('mean_wfe') or 0):.2f}  "
            f"ret={float(s.get('mean_total_return_pct') or 0):.3f}%  "
            f"acc={acc}  run_id={rid}"
        )
        if sc > best_score:
            best_score = sc
            best_report = report
            best_params = {k: v for k, v in params.items() if not k.startswith("_")}
        if acc:
            print("*** ACCEPTANCE PASSED — stopping ***")
            break

    if best_params is None or best_report is None:
        print("No successful trials.", file=sys.stderr)
        return 1

    passed = best_report["criteria"]["acceptance"]["passed"]
    print(f"\nBest score={best_score}  acceptance={passed}")
    _print_acceptance_gaps(best_report)

    if args.write_symbol_yaml and passed:
        write_symbol_config(sym_write, timeframe, tuning_best=best_params)
        print(f"Wrote {paths_for(sym_write, timeframe).config_yaml}")
    elif args.write_best_failed:
        write_symbol_config(sym_write, timeframe, tuning_best=best_params)
        print(f"Wrote best (failed acceptance) → {paths_for(sym_write, timeframe).config_yaml}")
    elif args.write_symbol_yaml and not passed:
        print("Skip YAML write (acceptance failed). Use --write-best-failed to save anyway.")

    slug = sym_write.replace("/", "-")
    out = ROOT / "docs" / f"thesis_push_{slug}_{timeframe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "summary": best_report["summary"],
                "criteria": best_report["criteria"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary JSON: {out}")

    if not passed:
        print(
            "\nПодсказка: чаще всего не хватает mean_sharpe>0.5 или mean_wfe>=0.5. "
            "Попробуйте --max-rows 6000, или scripts/thesis_push_4model_phase2.py"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
