#!/usr/bin/env python3
"""
Diploma ablation: single → dual → 4 equal → 4 + regime_adaptive + decision.

Writes JSON summary and journal entries with ``profile: canonical_4model``.

Usage (from repo root):
  python scripts/run_ablation.py --parquet data/ohlcv/demo_BTC-USDT_1h.parquet --max-rows 2200
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.results_journal import BacktestResultsJournal
from orchestration.benchmark_runner import build_report, orchestrator_config_from_yaml, run_wfo_on_features
from orchestration.canonical_pipeline import journal_profile_tag
from orchestration.glue import default_horizon_labels
from orchestration.real_data_benchmark import features_from_ohlcv_parquet


def _ablation_variants(cfg):
    mk4 = ["lgb", "gru", "xgb", "cnn"]
    w4 = {k: 0.25 for k in mk4}
    w2 = {"lgb": 0.5, "xgb": 0.5}
    return [
        ("lgb_only", replace(cfg, model_keys=["lgb"], trend_weights={"lgb": 1.0}, range_weights={"lgb": 1.0}, breakout_weights={"lgb": 1.0}, ensemble_mode="fixed_range", apply_decision_pipeline=False)),
        ("lgb_xgb", replace(cfg, model_keys=["lgb", "xgb"], trend_weights=w2, range_weights=w2, breakout_weights=w2, ensemble_mode="fixed_range", apply_decision_pipeline=False)),
        ("four_equal", replace(cfg, model_keys=mk4, trend_weights=w4, range_weights=w4, breakout_weights=w4, ensemble_mode="fixed_range", apply_decision_pipeline=False)),
        ("four_regime_adaptive", replace(cfg, model_keys=mk4, apply_decision_pipeline=True, ensemble_mode="regime_adaptive")),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run 4-model ablation suite")
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--config", type=Path, default="config/profiles/canonical_4model.yaml")
    p.add_argument("--max-rows", type=int, default=2200)
    p.add_argument("--dl-epochs", type=int, default=2)
    p.add_argument("--journal-root", type=Path, default="docs/backtest_journal")
    p.add_argument("--out", type=Path, default="docs/reports/ablation_canonical.json")
    args = p.parse_args(argv)

    pq = args.parquet or (ROOT / "data" / "ohlcv" / "demo_BTC-USDT_1h.parquet")
    if not pq.is_file():
        pq = ROOT / "data" / "ohlcv" / "BTC-USDT_1h.parquet"
    if not pq.is_file():
        print(f"No parquet at {pq}", file=sys.stderr)
        return 2

    profile = journal_profile_tag(args.config)
    features = features_from_ohlcv_parquet(pq, max_rows=args.max_rows)
    base_cfg = orchestrator_config_from_yaml(args.config)
    base_cfg = replace(
        base_cfg,
        train_window_size=min(base_cfg.train_window_size, max(400, len(features) // 3)),
        test_window_size=min(base_cfg.test_window_size, 150),
        walk_forward_step=min(base_cfg.walk_forward_step, 200),
    )

    journal = BacktestResultsJournal(args.journal_root)
    rows = []

    for name, cfg in _ablation_variants(base_cfg):
        print(f"--- {name} model_keys={cfg.model_keys} ensemble={cfg.ensemble_mode} ---")
        try:
            folds = run_wfo_on_features(
                features,
                cfg,
                use_risk_bridge=bool(getattr(cfg, "use_risk_bridge", True)),
                dl_epochs=args.dl_epochs,
                config_path=args.config,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"variant": name, "error": str(exc)})
            continue

        report = build_report(
            folds,
            parquet=str(pq.resolve()),
            feature_rows=len(features),
            max_rows=args.max_rows,
            config_path=args.config,
            label=f"ablation/{name}",
            params={
                "variant": name,
                "model_keys": list(cfg.model_keys),
                "ensemble_mode": cfg.ensemble_mode,
                "apply_decision_pipeline": cfg.apply_decision_pipeline,
                "profile": profile,
                "config_path": str(args.config.resolve()),
            },
            stage="ablation",
        )
        report["profile"] = profile
        rid = journal.append_run(report, label=f"ablation/{name}", params=report["params"])
        s = report["summary"]
        row = {
            "variant": name,
            "run_id": rid,
            "model_keys": list(cfg.model_keys),
            "mean_sharpe": s.get("mean_sharpe"),
            "mean_pf": s.get("mean_profit_factor"),
            "mean_wfe": s.get("mean_wfe"),
            "acceptance_passed": report["criteria"]["acceptance"]["passed"],
        }
        rows.append(row)
        print(f"  sharpe={row['mean_sharpe']} pf={row['mean_pf']} acc={row['acceptance_passed']} run_id={rid}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profile": profile, "parquet": str(pq), "variants": rows}
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
