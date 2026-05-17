"""Узкий подбор вокруг refine_19 (7/8 acceptance) — только Sharpe > 0.5."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.results_journal import BacktestResultsJournal
from orchestration.benchmark_runner import load_tuning_best_params
from orchestration.real_data_benchmark import default_real_parquet, features_from_ohlcv_parquet
from orchestration.tuning_loop import run_single_trial

BASE = {
    **load_tuning_best_params("config.yaml"),
    "use_risk_bridge": False,
    "regime": "momentum",
}

CANDIDATES = [
    {},
    {"min_signal_margin": 0.09},
    {"min_signal_margin": 0.10},
    {"min_signal_margin": 0.11},
    {"min_signal_margin": 0.12},
    {"max_position_fraction": 0.5},
    {"max_position_fraction": 0.35},
    {"max_position_fraction": 0.25},
    {"volatility_filter_percentile": 90.0},
    {"min_signal_margin": 0.09, "max_position_fraction": 0.5},
    {"min_signal_margin": 0.10, "max_position_fraction": 0.35},
    {"min_signal_margin": 0.09, "volatility_filter_percentile": 90.0},
    {"label_min_return": 0.003},
    {"label_min_return": 0.004, "min_signal_margin": 0.09},
    {"label_min_return": 0.005, "min_signal_margin": 0.10},
    {"signal_strategy": "momentum_confirm", "min_signal_margin": 0.08},
    {"signal_strategy": "momentum_confirm", "min_signal_margin": 0.10},
    {"test_window_size": 150},
    {"test_window_size": 170},
    {"direction_threshold": 0.58, "min_signal_margin": 0.09},
    {"direction_threshold": 0.62, "min_signal_margin": 0.09},
    {"trade_mode": "long_only", "min_signal_margin": 0.09},
    {
        "min_signal_margin": 0.10,
        "max_position_fraction": 0.35,
        "volatility_filter_percentile": 90.0,
    },
    {
        "min_signal_margin": 0.11,
        "max_position_fraction": 0.35,
        "label_min_return": 0.003,
    },
]


def main() -> None:
    pq = default_real_parquet()
    features = features_from_ohlcv_parquet(pq, max_rows=8000)
    journal = BacktestResultsJournal()
    best_sh = -1e9
    best_rid = None

    for i, patch in enumerate(CANDIDATES, 1):
        params = {**BASE, **patch, "use_risk_bridge": False}
        params["_label"] = f"sharpe_push_{i}"
        report = run_single_trial(
            features, params, parquet_label=str(pq.resolve())
        )
        report["max_rows"] = 8000
        rid = journal.append_run(
            report,
            label=params["_label"],
            params={k: v for k, v in params.items() if not k.startswith("_")},
        )
        s = report["summary"]
        sh = float(s.get("mean_sharpe") or -999)
        acc = report["criteria"]["acceptance"]["passed"]
        print(
            f"{params['_label']:16} sharpe={sh:.3f} pf={s.get('mean_profit_factor', 0):.2f} "
            f"wfe={s.get('mean_wfe', 0):.2f} acc={acc} rid={rid}"
        )
        if sh > best_sh:
            best_sh = sh
            best_rid = rid
            if acc:
                print("  *** ACCEPTANCE PASSED ***")
                break

    print(f"\nBest sharpe {best_sh:.4f} run_id={best_rid}")


if __name__ == "__main__":
    main()
