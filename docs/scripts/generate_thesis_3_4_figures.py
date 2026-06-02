"""
Thesis figures for section 3.4 (backtesting & validation).

Output: docs/thesis/3_4/figures/
  fig_3_13_walk_forward_scheme.png
  fig_3_14_equity_curve.png
  fig_3_15_metrics_table.png
  fig_3_15_metrics_table.csv
  fig_3_16_drawdown.png
  fig_3_13_walk_forward_scheme.mmd

Usage:
  py -3 docs/scripts/generate_thesis_3_4_figures.py --journal docs/backtest_journal/runs/20260523T155940Z_7068eb24.json
  py -3 docs/scripts/generate_thesis_3_4_figures.py --run-wfo --light --max-folds 4 --max-rows 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT_DIR = ROOT / "docs" / "thesis" / "3_4" / "figures"
DEFAULT_FEATURES = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
DEFAULT_PROFILE = ROOT / "config" / "profiles" / "canonical_4model.yaml"
DEFAULT_JOURNAL = ROOT / "docs" / "backtest_journal" / "runs" / "20260523T155940Z_7068eb24.json"


def _ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def fig_3_13_wfo_scheme(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    y = 2.4
    w_train, w_test, w_gap = 2.4, 1.0, 0.35
    colors = {"train": "#C8E6C9", "test": "#BBDEFB", "gap": "#FFECB3"}

    def block(x, w, label, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.55),
                w,
                1.1,
                boxstyle="round,pad=0.03",
                facecolor=color,
                edgecolor="#37474F",
            )
        )
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    x0 = 0.4
    block(x0, w_train, "Train\n(fit models)", colors["train"])
    block(x0 + w_train, w_gap, "purge H\n+embargo E", colors["gap"])
    block(x0 + w_train + w_gap, w_test, "Test\n(OOS backtest)", colors["test"])
    ax.annotate(
        "",
        xy=(x0 + w_train + w_gap + w_test + 0.15, y),
        xytext=(x0 + w_train + w_gap + w_test - 0.05, y),
        arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.2),
    )
    ax.text(x0 + w_train + w_gap + w_test + 0.55, y, "shift\nwindow →", ha="left", va="center", fontsize=9)

    x1 = 5.0
    block(x1, w_train, "Train", colors["train"])
    block(x1 + w_train, w_gap, "purge+embargo", colors["gap"])
    block(x1 + w_train + w_gap, w_test, "Test", colors["test"])
    ax.annotate(
        "",
        xy=(x1 + w_train + w_gap + w_test + 0.15, y),
        xytext=(x1 + w_train + w_gap + w_test - 0.05, y),
        arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.2),
    )
    ax.text(x1 + w_train + w_gap + w_test + 0.55, y, "…", ha="left", va="center", fontsize=12)

    ax.text(
        6.5,
        0.75,
        "H = prediction_horizon (12)  ·  E = embargo_period (5)  ·  окна: train_window_size, test_window_size, walk_forward_step",
        ha="center",
        fontsize=8.5,
        color="#546E7A",
    )

    path = out / "fig_3_13_walk_forward_scheme.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_mermaid(out: Path) -> Path:
    mmd = """flowchart LR
  T1[Train fold k] --> P[purge H bars]
  P --> E[embargo E bars]
  E --> OOS[Test OOS]
  OOS --> S[shift window]
  S --> T2[Train fold k+1]
"""
    path = out / "fig_3_13_walk_forward_scheme.mmd"
    path.write_text(mmd, encoding="utf-8")
    return path


def _load_journal(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_from_journal(journal: dict[str, Any]) -> pd.Series:
    folds = journal.get("fold_metrics") or []
    if not folds:
        s = journal.get("summary") or {}
        return pd.Series(
            {
                "Sharpe Ratio": s.get("mean_sharpe"),
                "Max Drawdown (%)": s.get("worst_max_drawdown_pct"),
                "Win Rate (%)": float("nan"),
                "Profit Factor": s.get("mean_profit_factor"),
                "Total Return (%)": s.get("mean_total_return_pct"),
                "Walk-Forward Efficiency": s.get("mean_wfe"),
                "Trade Events": s.get("total_trade_events"),
            }
        )
    df = pd.DataFrame(folds)
    from backtesting.criteria_evaluator import summarize_wfo_folds

    summary = summarize_wfo_folds(df)
    wr = _safe_col_mean(df, "Win Rate (%)")
    return pd.Series(
        {
            "Sharpe Ratio": summary.get("mean_sharpe"),
            "Max Drawdown (%)": summary.get("worst_max_drawdown_pct"),
            "Win Rate (%)": wr,
            "Profit Factor": summary.get("mean_profit_factor"),
            "Total Return (%)": summary.get("mean_total_return_pct"),
            "Walk-Forward Efficiency": summary.get("mean_wfe"),
            "Trade Events": summary.get("total_trade_events"),
            "Folds (n)": summary.get("n_folds"),
        }
    )


def _safe_col_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def fig_3_15_metrics_table(out: Path, metrics: pd.Series, subtitle: str = "") -> tuple[Path, Path]:
    rows = [
        ("Sharpe Ratio", metrics.get("Sharpe Ratio")),
        ("Max Drawdown (%)", metrics.get("Max Drawdown (%)")),
        ("Win Rate (%)", metrics.get("Win Rate (%)")),
        ("Profit Factor", metrics.get("Profit Factor")),
        ("Total Return (%)", metrics.get("Total Return (%)")),
        ("Walk-Forward Efficiency", metrics.get("Walk-Forward Efficiency")),
        ("Trade Events", metrics.get("Trade Events")),
    ]
    if metrics.get("Folds (n)") is not None:
        rows.append(("Folds (n)", metrics.get("Folds (n)")))

    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.4f}"

    table_df = pd.DataFrame([{"Metric": a, "Value": fmt(b)} for a, b in rows])
    csv_path = out / "fig_3_15_metrics_table.csv"
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axis("off")
    if subtitle:
        ax.set_title(subtitle, fontsize=9, loc="left", color="#546E7A")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    png_path = out / "fig_3_15_metrics_table.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, csv_path


def stitched_oos_equity(
    features: pd.DataFrame,
    targets: pd.Series,
    cfg,
    *,
    max_folds: Optional[int],
    dl_epochs: int,
    light_only: bool,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Склейка OOS ``cum_strategy_returns`` по фолдам WFO (как в ``walk_forward_backtest``)."""
    from backtesting.backtester import Backtester
    from backtesting.metrics_config import load_backtesting_config
    from orchestration import TrainingOrchestrator
    from orchestration.glue import (
        MomentumRegimeDetector,
        build_orchestration_models,
        meta_weighting_from_config,
    )
    from risk_management import OrchestratorRiskBridge

    if light_only:
        mk = ["lgb", "xgb"]
        w = {k: 1.0 / len(mk) for k in mk}
        cfg = replace(
            cfg,
            model_keys=mk,
            trend_weights=w.copy(),
            range_weights=w.copy(),
            breakout_weights=w.copy(),
        )

    models = build_orchestration_models(cfg, features, dl_epochs=dl_epochs)
    orch = TrainingOrchestrator(cfg)
    orch.initialize(
        models=models,
        regime_detector=MomentumRegimeDetector(),
        meta_weighting=meta_weighting_from_config(cfg),
        risk_manager=OrchestratorRiskBridge(),
    )

    bt_cfg = load_backtesting_config()
    bt = Backtester(
        commission=bt_cfg.metrics.commission,
        slippage=bt_cfg.metrics.slippage,
    )

    splits = orch.leakage_preventer.safe_walk_forward_split(
        features,
        targets,
        cfg.train_window_size,
        cfg.test_window_size,
        cfg.walk_forward_step,
        cfg.prediction_horizon,
    )

    scale = 1.0
    equity_parts: list[pd.Series] = []
    bench_parts: list[pd.Series] = []
    dd_parts: list[pd.Series] = []
    fold_rows: list[dict] = []

    for fold_idx, (train_f, train_y, test_f, test_y) in enumerate(splits):
        if max_folds is not None and fold_idx >= max_folds:
            break

        train_mask = ~train_y.isna()
        train_f = train_f[train_mask]
        train_y = train_y[train_mask]

        for model in orch.models.values():
            if hasattr(model, "fit"):
                model.fit(train_f, train_y)

        orch._volatility_atr_threshold = orch._compute_atr_threshold(train_f)
        train_res = orch.run_pipeline(train_f)
        orch._runtime_train_meta_threshold = orch._calibrate_test_meta_threshold(
            train_res.meta_probabilities
        )
        test_res = orch.run_pipeline(test_f)
        orch._runtime_train_meta_threshold = None
        orch._volatility_atr_threshold = None

        perf = bt.run(
            test_f[["close"]],
            test_res.final_signals,
            position_size=test_res.position_sizes,
        )
        from backtesting.performance_metrics import PerformanceMetrics

        m = PerformanceMetrics(perf).calculate_metrics().to_dict()
        m["Fold"] = fold_idx + 1
        fold_rows.append(m)

        cum = perf["cum_strategy_returns"] * scale
        bench = perf["cum_market_returns"] * scale
        scale = float(cum.iloc[-1]) if len(cum) else scale
        equity_parts.append(cum)
        bench_parts.append(bench)
        dd_parts.append(perf["drawdown"])

    if not equity_parts:
        raise RuntimeError("No WFO folds produced; check feature rows and window sizes")

    equity = pd.concat(equity_parts)
    benchmark = pd.concat(bench_parts)
    drawdown = pd.concat(dd_parts)
    return equity, drawdown, pd.DataFrame(fold_rows)


def fig_3_14_equity(out: Path, equity: pd.Series, benchmark: pd.Series, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(equity))
    ax.plot(x, equity.values, color="#1565C0", linewidth=1.5, label="Strategy (OOS stitched)")
    ax.plot(x, benchmark.values, color="#9E9E9E", linewidth=1.0, linestyle="--", label="Buy & Hold (OOS)")
    ax.set_ylabel("Cumulative return (× capital)")
    ax.set_xlabel("OOS bar index (concatenated folds)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    if label:
        ax.text(0.01, 0.02, label, transform=ax.transAxes, fontsize=8, color="#546E7A")
    path = out / "fig_3_14_equity_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_16_drawdown(out: Path, drawdown: pd.Series, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(drawdown))
    ax.fill_between(x, drawdown.values * 100, 0, color="#EF5350", alpha=0.45)
    ax.plot(x, drawdown.values * 100, color="#C62828", linewidth=0.9)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("OOS bar index")
    ax.grid(True, alpha=0.25)
    if label:
        ax.text(0.01, 0.05, label, transform=ax.transAxes, fontsize=8, color="#546E7A")
    path = out / "fig_3_16_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--run-wfo", action="store_true", help="Recompute stitched OOS equity (slow)")
    parser.add_argument("--light", action="store_true", help="WFO with lgb+xgb only (faster)")
    parser.add_argument("--max-folds", type=int, default=0, help="Cap folds (0 = all)")
    parser.add_argument("--max-rows", type=int, default=0, help="Tail rows of features (0 = all)")
    parser.add_argument("--dl-epochs", type=int, default=3)
    args = parser.parse_args()

    out = _ensure_out()
    paths: list[Path] = [fig_3_13_wfo_scheme(out), write_mermaid(out)]

    journal = None
    if args.journal.is_file():
        journal = _load_journal(args.journal)
        metrics = _metrics_from_journal(journal)
        sub = f"Source: {args.journal.name} (OOS fold aggregates)"
        p15, pcsv = fig_3_15_metrics_table(out, metrics, subtitle=sub)
        paths.extend([p15, pcsv])
    else:
        print(f"Journal not found: {args.journal} — skip metrics table")

    if args.run_wfo:
        from orchestration.glue import default_horizon_labels, load_feature_table_from_parquet
        from orchestration.orchestrator_config import OrchestratorConfig

        cfg = OrchestratorConfig.from_yaml(args.profile)
        df = load_feature_table_from_parquet(args.features)
        if args.max_rows > 0:
            df = df.tail(args.max_rows)
        y = default_horizon_labels(df["close"], cfg.prediction_horizon)
        max_f = args.max_folds if args.max_folds > 0 else None
        from backtesting.backtester import Backtester
        from backtesting.metrics_config import load_backtesting_config
        from orchestration import TrainingOrchestrator

        equity, dd, fold_df = stitched_oos_equity(
            df, y, cfg, max_folds=max_f, dl_epochs=args.dl_epochs, light_only=args.light
        )
        bt_cfg = load_backtesting_config()
        bt = Backtester(
            commission=bt_cfg.metrics.commission,
            slippage=bt_cfg.metrics.slippage,
        )
        scale = 1.0
        bh_parts = []
        orch_tmp = TrainingOrchestrator(cfg)
        splits = orch_tmp.leakage_preventer.safe_walk_forward_split(
            df, y, cfg.train_window_size, cfg.test_window_size, cfg.walk_forward_step, cfg.prediction_horizon
        )
        for fold_idx, (_, _, test_f, _) in enumerate(splits):
            if max_f is not None and fold_idx >= max_f:
                break
            perf = bt.run(test_f[["close"]], np.zeros(len(test_f)))
            c = perf["cum_market_returns"] * scale
            scale = float(c.iloc[-1])
            bh_parts.append(c)
        benchmark = pd.concat(bh_parts) if bh_parts else equity

        tag = "lgb+xgb" if args.light else "4-model"
        label = f"WFO stitched OOS · {tag} · {len(df)} bars"
        paths.append(fig_3_14_equity(out, equity, benchmark, label))
        paths.append(fig_3_16_drawdown(out, dd, label))
        if journal is None:
            from backtesting.criteria_evaluator import summarize_wfo_folds

            summ = summarize_wfo_folds(fold_df)
            m = pd.Series(
                {
                    "Sharpe Ratio": summ.get("mean_sharpe"),
                    "Max Drawdown (%)": summ.get("worst_max_drawdown_pct"),
                    "Win Rate (%)": _safe_col_mean(fold_df, "Win Rate (%)"),
                    "Profit Factor": summ.get("mean_profit_factor"),
                    "Total Return (%)": summ.get("mean_total_return_pct"),
                    "Walk-Forward Efficiency": summ.get("mean_wfe"),
                    "Trade Events": summ.get("total_trade_events"),
                    "Folds (n)": summ.get("n_folds"),
                }
            )
            p15, pcsv = fig_3_15_metrics_table(out, m, subtitle=label)
            paths.extend([p15, pcsv])
    else:
        print(
            "Equity/drawdown: pass --run-wfo (or use journal-only metrics).\n"
            "  py -3 docs/scripts/generate_thesis_3_4_figures.py --run-wfo --light --max-folds 4"
        )

    print("Generated:")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
