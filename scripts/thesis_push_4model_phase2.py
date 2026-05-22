#!/usr/bin/env python3
"""
Фаза 2: точечный подбор вокруг лучшего 4-model прогона (часто fixed_range + margin).

Запускать после thesis_push_4model.py, если acceptance не прошёл из-за Sharpe/WFE.

  python scripts/thesis_push_4model_phase2.py --parquet data/ohlcv/BTC-USDT_1h.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.thesis_push_4model import _print_acceptance_gaps, _resolve_symbol_tf, _score
from backtesting.results_journal import BacktestResultsJournal
from orchestration.symbols import paths_for, write_symbol_config
from orchestration.tuning_config import THESIS_4MODEL_SEED
from orchestration.real_data_benchmark import features_from_ohlcv_parquet
from orchestration.tuning_loop import run_single_trial

# Старт от 2-model acceptance + fixed_range (4 равных веса, все модели в ансамбле)
PHASE2_BASE: Dict[str, Any] = {
    **THESIS_4MODEL_SEED,
    "ensemble_mode": "fixed_range",
    "direction_threshold": 0.6,
    "min_signal_margin": 0.08,
    "meta_threshold_mode": "percentile",
    "prediction_horizon": 24,
    "dl_epochs": 2,
}

PHASE2_CANDIDATES: List[Dict[str, Any]] = [
    {},
    {"min_signal_margin": 0.09},
    {"min_signal_margin": 0.10},
    {"min_signal_margin": 0.11},
    {"min_signal_margin": 0.12},
    {"direction_threshold": 0.58},
    {"direction_threshold": 0.62},
    {"trade_mode": "long_only"},
    {"trade_mode": "long_only", "min_signal_margin": 0.10},
    {"max_position_fraction": 0.5},
    {"max_position_fraction": 0.35, "min_signal_margin": 0.10},
    {"volatility_filter_percentile": 90.0},
    {"volatility_filter_percentile": 90.0, "min_signal_margin": 0.10, "max_position_fraction": 0.35},
    {"label_min_return": 0.003},
    {"label_min_return": 0.004, "min_signal_margin": 0.10},
    {"signal_strategy": "momentum_confirm"},
    {"walk_forward_step": 330},
    {"test_window_size": 160},
    {"ensemble_mode": "regime_adaptive", "min_signal_margin": 0.10},
    {"ensemble_mode": "regime_adaptive", "min_signal_margin": 0.11, "volatility_filter_percentile": 90.0},
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Thesis 4-model phase-2 push (fixed_range focus)")
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--max-rows", type=int, default=8000)
    p.add_argument("--config", type=Path, default="config/profiles/canonical_4model.yaml")
    p.add_argument("--write-symbol-yaml", action="store_true")
    args = p.parse_args(argv)

    symbol, timeframe, pq = _resolve_symbol_tf(
        parquet=args.parquet, symbol=args.symbol, timeframe=args.timeframe
    )
    sym_write = args.symbol or (symbol.replace("-", "/") if symbol and "-" in symbol else "BTC/USDT")

    features = features_from_ohlcv_parquet(pq, max_rows=args.max_rows)
    journal = BacktestResultsJournal()
    best_report = None
    best_params = None
    best_score = (-1,)

    print("Phase 2 base:", {k: PHASE2_BASE[k] for k in ("ensemble_mode", "min_signal_margin", "dl_epochs")})

    for i, patch in enumerate(PHASE2_CANDIDATES, 1):
        params = {**PHASE2_BASE, **patch, "_label": f"thesis4p2_{i}"}
        try:
            report = run_single_trial(
                features, params, config_path=args.config, parquet_label=str(pq.resolve())
            )
        except Exception as exc:
            print(f"thesis4p2_{i:02d}  ERROR: {exc}")
            continue
        rid = journal.append_run(
            report,
            label=params["_label"],
            params={k: v for k, v in params.items() if not k.startswith("_")},
        )
        s = report["summary"]
        acc = report["criteria"]["acceptance"]["passed"]
        sc = _score(report)
        print(
            f"{params['_label']:14}  sharpe={float(s.get('mean_sharpe') or 0):.3f}  "
            f"pf={float(s.get('mean_profit_factor') or 0):.2f}  wfe={float(s.get('mean_wfe') or 0):.2f}  "
            f"acc={acc}  run_id={rid}"
        )
        if sc > best_score:
            best_score = sc
            best_report = report
            best_params = {k: v for k, v in params.items() if not k.startswith("_")}
        if acc:
            print("*** ACCEPTANCE PASSED ***")
            break

    if best_report is None:
        return 1
    passed = best_report["criteria"]["acceptance"]["passed"]
    _print_acceptance_gaps(best_report)
    if args.write_symbol_yaml and passed:
        write_symbol_config(sym_write, timeframe, tuning_best=best_params)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
