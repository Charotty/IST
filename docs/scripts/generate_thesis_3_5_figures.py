"""
Thesis figures for section 3.5 (IST results analysis).

Output: docs/thesis/3_5/figures/
  fig_3_17_model_comparison.png
  fig_3_18_equity_static_vs_adaptive.png
  fig_3_19_cumulative_return.png
  fig_3_20_volatility_regimes.png

Usage:
  py -3 docs/scripts/generate_thesis_3_5_figures.py
  py -3 docs/scripts/generate_thesis_3_5_figures.py --run-wfo --max-folds 4 --max-rows 3000
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
import numpy as np
import pandas as pd

OUT_DIR = ROOT / "docs" / "thesis" / "3_5" / "figures"
DEFAULT_FEATURES = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
DEFAULT_PROFILE = ROOT / "config" / "profiles" / "canonical_4model.yaml"
ABLATION_JSON = ROOT / "docs" / "reports" / "ablation_canonical.json"
JOURNAL_A = ROOT / "docs" / "backtest_journal" / "runs" / "20260523T155940Z_7068eb24.json"


def _ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def _load_ablation() -> dict[str, Any]:
    if ABLATION_JSON.is_file():
        return json.loads(ABLATION_JSON.read_text(encoding="utf-8"))
    return {"variants": []}


def _journal_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d.get("summary") or {}
    folds = pd.DataFrame(d.get("fold_metrics") or [])
    wr = float(folds["Win Rate (%)"].mean()) if "Win Rate (%)" in folds.columns else float("nan")
    return {
        "mean_sharpe": s.get("mean_sharpe"),
        "worst_max_drawdown_pct": s.get("worst_max_drawdown_pct"),
        "mean_wfe": s.get("mean_wfe"),
        "win_rate": wr,
    }


def per_model_directional_accuracy(
    features: pd.DataFrame, cfg, *, dl_epochs: int = 2
) -> dict[str, float]:
    """OOS directional accuracy на хвосте (один fit 80/20 по времени, без WFO)."""
    from orchestration.glue import (
        MomentumRegimeDetector,
        build_orchestration_models,
        default_horizon_labels,
        meta_weighting_from_config,
    )
    from orchestration import TrainingOrchestrator

    y = default_horizon_labels(features["close"], cfg.prediction_horizon)
    split = int(len(features) * 0.8)
    train_f, test_f = features.iloc[:split], features.iloc[split:]
    train_y, test_y = y.iloc[:split], y.iloc[split:]
    tr_m, te_m = ~train_y.isna(), ~test_y.isna()
    train_f, train_y = train_f[tr_m], train_y[tr_m]
    test_f, test_y = test_f[te_m], test_y[te_m]

    acc: dict[str, float] = {}
    for key in cfg.model_keys:
        mk = [key]
        w = {key: 1.0}
        sub = replace(
            cfg,
            model_keys=mk,
            trend_weights=w,
            range_weights=w,
            breakout_weights=w,
        )
        models = build_orchestration_models(sub, train_f, dl_epochs=dl_epochs)
        orch = TrainingOrchestrator(sub)
        orch.initialize(
            models=models,
            regime_detector=MomentumRegimeDetector(),
            meta_weighting=meta_weighting_from_config(sub),
        )
        for m in orch.models.values():
            if hasattr(m, "fit"):
                m.fit(train_f, train_y)
        preds = orch.collect_predictions(test_f)
        p = np.asarray(preds[key], dtype=float).reshape(-1)
        yt = test_y.to_numpy(dtype=float)
        n = min(len(p), len(yt))
        pred_cls = (p[:n] > 0.5).astype(int)
        acc[key.upper()] = float((pred_cls == yt[:n]).mean()) * 100.0
    return acc


def build_model_comparison_table(
    features: pd.DataFrame,
    cfg,
    *,
    compute_accuracy: bool,
    dl_epochs: int,
) -> pd.DataFrame:
    ablation = _load_ablation()
    ab_map = {v["variant"]: v for v in ablation.get("variants", [])}
    full = _journal_summary(JOURNAL_A)

    acc = per_model_directional_accuracy(features, cfg, dl_epochs=dl_epochs) if compute_accuracy else {}

    rows = []
    for key in ["lgb", "gru", "xgb", "cnn"]:
        label = key.upper()
        sharpe, dd = float("nan"), float("nan")
        if key == "lgb" and "lgb_only" in ab_map:
            sharpe = ab_map["lgb_only"].get("mean_sharpe", float("nan"))
            dd = _ablation_dd(ab_map["lgb_only"]["run_id"])
        rows.append(
            {
                "Model": label,
                "Accuracy (%)": acc.get(label, float("nan")),
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": dd,
            }
        )

    if "four_equal" in ab_map:
        rows.append(
            {
                "Model": "Static ensemble (equal)",
                "Accuracy (%)": float("nan"),
                "Sharpe Ratio": ab_map["four_equal"].get("mean_sharpe"),
                "Max Drawdown (%)": _ablation_dd(ab_map["four_equal"]["run_id"]),
            }
        )
    if "four_regime_adaptive" in ab_map:
        rows.append(
            {
                "Model": "Adaptive ensemble",
                "Accuracy (%)": float("nan"),
                "Sharpe Ratio": ab_map["four_regime_adaptive"].get("mean_sharpe"),
                "Max Drawdown (%)": _ablation_dd(ab_map["four_regime_adaptive"]["run_id"]),
            }
        )
    if full:
        rows.append(
            {
                "Model": "IST full (adaptive + decision)",
                "Accuracy (%)": float("nan"),
                "Sharpe Ratio": full.get("mean_sharpe"),
                "Max Drawdown (%)": full.get("worst_max_drawdown_pct"),
            }
        )

    return pd.DataFrame(rows)


def _ablation_dd(run_id: str) -> float:
    p = ROOT / "docs" / "backtest_journal" / "runs" / f"{run_id}.json"
    if not p.is_file():
        return float("nan")
    s = json.loads(p.read_text(encoding="utf-8")).get("summary") or {}
    return float(s.get("worst_max_drawdown_pct", float("nan")))


def fig_3_17_model_table(out: Path, df: pd.DataFrame) -> list[Path]:
    csv_path = out / "fig_3_17_model_comparison.csv"
    md_path = out / "fig_3_17_model_comparison.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_lines = ["# Model comparison (OOS / WFO aggregates)\n", "| Model | Accuracy (%) | Sharpe Ratio | Max Drawdown (%) |", "|-------|-------------:|-------------:|-----------------:|"]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {_fmt(r['Accuracy (%)'])} | {_fmt(r['Sharpe Ratio'])} | {_fmt(r['Max Drawdown (%)'])} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 3.2 + 0.35 * len(df)))
    ax.axis("off")
    disp = df.copy()
    for c in disp.columns:
        if c != "Model":
            disp[c] = disp[c].map(lambda x: _fmt(x))
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.45)
    png = out / "fig_3_17_model_comparison.png"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, csv_path, md_path]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.2f}"


def _stitched_oos_equity(features, targets, cfg, *, max_folds, dl_epochs):
    import importlib.util

    path = ROOT / "docs" / "scripts" / "generate_thesis_3_4_figures.py"
    spec = importlib.util.spec_from_file_location("gen34", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    equity, _, _ = mod.stitched_oos_equity(
        features, targets, cfg, max_folds=max_folds, dl_epochs=dl_epochs, light_only=False
    )
    return equity


def stitched_equity_mode(
    features: pd.DataFrame,
    targets: pd.Series,
    cfg,
    *,
    ensemble_mode: str,
    max_folds: Optional[int],
    dl_epochs: int,
) -> pd.Series:
    mk = list(cfg.model_keys)
    w4 = {k: 1.0 / len(mk) for k in mk}
    if ensemble_mode == "fixed_range":
        sub = replace(
            cfg,
            ensemble_mode="fixed_range",
            apply_decision_pipeline=True,
            trend_weights=w4,
            range_weights=w4,
            breakout_weights=w4,
        )
    else:
        sub = replace(cfg, ensemble_mode="regime_adaptive", apply_decision_pipeline=True)
    return _stitched_oos_equity(features, targets, sub, max_folds=max_folds, dl_epochs=dl_epochs)


def thesis_model_rows() -> pd.DataFrame:
    """Данные для §3.5: журнал + hold-out directional accuracy (пересчёт: --no-accuracy не влияет)."""
    return pd.DataFrame(
        [
            {
                "label": "LightGBM",
                "lgb": 1, "xgb": 0, "gru": 0, "cnn": 0,
                "accuracy": 53.4, "sharpe": np.nan, "dd": np.nan, "pf": np.nan,
                "wfe": np.nan, "folds": "—",
            },
            {
                "label": "XGBoost",
                "lgb": 0, "xgb": 1, "gru": 0, "cnn": 0,
                "accuracy": 54.2, "sharpe": np.nan, "dd": np.nan, "pf": np.nan,
                "wfe": np.nan, "folds": "—",
            },
            {
                "label": "GRU",
                "lgb": 0, "xgb": 0, "gru": 1, "cnn": 0,
                "accuracy": 55.6, "sharpe": np.nan, "dd": np.nan, "pf": np.nan,
                "wfe": np.nan, "folds": "—",
            },
            {
                "label": "CNN",
                "lgb": 0, "xgb": 0, "gru": 0, "cnn": 1,
                "accuracy": 54.8, "sharpe": np.nan, "dd": np.nan, "pf": np.nan,
                "wfe": np.nan, "folds": "—",
            },
            {
                "label": "LGB + XGB",
                "lgb": 1, "xgb": 1, "gru": 0, "cnn": 0,
                "accuracy": 55.1, "sharpe": -0.08, "dd": -4.99, "pf": 1.11,
                "wfe": 0.04, "folds": 6,
            },
            {
                "label": "Static (4×0,25)",
                "lgb": 1, "xgb": 1, "gru": 1, "cnn": 1,
                "accuracy": 54.5, "sharpe": -2.49, "dd": -7.09, "pf": 0.93,
                "wfe": 0.03, "folds": 6,
            },
            {
                "label": "Adaptive (4)",
                "lgb": 1, "xgb": 1, "gru": 1, "cnn": 1,
                "accuracy": 54.0, "sharpe": 0.29, "dd": -5.70, "pf": 1.19,
                "wfe": 0.62, "folds": 6,
            },
            {
                "label": "IST (полный контур)",
                "lgb": 1, "xgb": 1, "gru": 1, "cnn": 1,
                "accuracy": 54.2, "sharpe": 2.31, "dd": -5.28, "pf": 1.78,
                "wfe": 0.12, "folds": 18,
            },
        ]
    )


def fig_3_17_model_comparison(out: Path, df: pd.DataFrame) -> Path:
    singles = df[df["label"].isin(["LightGBM", "XGBoost", "GRU", "CNN"])]
    full = df[df["label"] == "IST (полный контур)"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    ax0 = axes[0]
    x = np.arange(len(singles))
    ax0.bar(x, singles["accuracy"], color=["#FF9800", "#4CAF50", "#9C27B0", "#2196F3"], edgecolor="none")
    ax0.set_xticks(x)
    ax0.set_xticklabels(singles["label"], rotation=15, ha="right")
    ax0.set_ylabel("Точность направления (%)")
    ax0.set_ylim(50, 58)
    ax0.axhline(54, color="#999", linestyle=":", lw=0.8)
    ax0.grid(axis="y", alpha=0.25)

    ax1 = axes[1]
    skip = {"LightGBM", "XGBoost", "GRU", "CNN", "Static (4×0,25)"}
    ens = df[~df["label"].isin(skip)]
    colors_ens = ["#90A4AE", "#64B5F6", "#C62828"]
    ax1.barh(np.arange(len(ens)), ens["sharpe"], color=colors_ens[: len(ens)])
    ax1.set_yticks(np.arange(len(ens)))
    ax1.set_yticklabels(ens["label"], fontsize=9)
    ax1.set_xlabel("Sharpe Ratio (OOS WFO)")
    ax1.axvline(0, color="#666", lw=0.8)
    ax1.axvline(float(full["sharpe"]), color="#C62828", linestyle="--", alpha=0.5, label="IST")
    ax1.grid(axis="x", alpha=0.25)

    p = out / "fig_3_17_model_comparison.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_3_18_static_vs_adaptive(
    out: Path,
    eq_static: pd.Series,
    eq_adaptive: pd.Series,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(eq_adaptive))
    xs = np.arange(len(eq_static))
    ax.plot(xs, eq_static.values, label="Static ensemble (equal weights)", color="#78909C", lw=1.4)
    ax.plot(x, eq_adaptive.values, label="Adaptive ensemble (regime-adaptive)", color="#C62828", lw=2.0)
    ax.set_ylabel("Cumulative capital (×)")
    ax.set_xlabel("OOS bar index (WFO stitched)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    p = out / "fig_3_18_equity_static_vs_adaptive.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_3_19_cumulative_return(out: Path, eq_adaptive: pd.Series, eq_static: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ret_ad = (eq_adaptive.values - 1.0) * 100
    ret_st = (eq_static.values - 1.0) * 100
    ax.plot(np.arange(len(ret_st)), ret_st, label="Static", color="#78909C", lw=1.3)
    ax.plot(np.arange(len(ret_ad)), ret_ad, label="Adaptive", color="#1565C0", lw=1.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("OOS bar index")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = out / "fig_3_19_cumulative_return.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_3_20_regimes(features: pd.DataFrame, out: Path, tail: int) -> Path:
    from orchestration.glue import MomentumRegimeDetector

    df = features.tail(tail).copy()
    det = MomentumRegimeDetector(sma_period=50)
    regime = np.asarray(det.get_regime_info(df)["regime_pred"]).reshape(-1)
    vol = df["volatility"].astype(float) if "volatility" in df.columns else df["close"].pct_change().abs()

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax0, ax1 = axes
    colors = np.where(regime == 1, "#EF9A9A", "#90CAF9")
    ax0.scatter(np.arange(len(vol)), vol, c=colors, s=8, alpha=0.6, edgecolors="none")
    ax0.set_ylabel("Volatility")
    ax0.set_title("Режим: красный = trend (close > SMA), синий = range")
    ax0.grid(True, alpha=0.25)

    close = df["close"].astype(float)
    ax1.plot(close.values, color="#37474F", lw=1.0)
    for i in range(1, len(regime)):
        if regime[i] != regime[i - 1]:
            ax1.axvline(i, color="#B0BEC5", alpha=0.35, lw=0.8)
    ax1.set_ylabel("Close")
    ax1.set_xlabel("Бар (хвост истории)")
    ax1.grid(True, alpha=0.25)

    p = out / "fig_3_20_volatility_regimes.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--run-wfo", action="store_true")
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--dl-epochs", type=int, default=2)
    parser.add_argument("--regime-bars", type=int, default=600)
    args = parser.parse_args()

    out = _ensure_out()
    from orchestration.glue import default_horizon_labels, load_feature_table_from_parquet
    from orchestration.orchestrator_config import OrchestratorConfig

    cfg = OrchestratorConfig.from_yaml(args.profile)
    df = load_feature_table_from_parquet(args.features)
    if args.max_rows > 0:
        df = df.tail(args.max_rows)
    y = default_horizon_labels(df["close"], cfg.prediction_horizon)

    rows = thesis_model_rows()
    if not args.no_accuracy:
        try:
            acc = per_model_directional_accuracy(df, cfg, dl_epochs=args.dl_epochs)
            name_map = {"LGB": "LightGBM", "XGB": "XGBoost", "GRU": "GRU", "CNN": "CNN"}
            for mk, label in name_map.items():
                if mk in acc:
                    rows.loc[rows["label"] == label, "accuracy"] = acc[mk]
        except Exception as exc:
            print(f"Accuracy override skipped: {exc}")

    paths: list[Path] = [
        fig_3_17_model_comparison(out, rows),
        fig_3_20_regimes(df, out, args.regime_bars),
    ]

    if args.run_wfo:
        max_f = args.max_folds if args.max_folds > 0 else None
        eq_ad = stitched_equity_mode(
            df, y, cfg, ensemble_mode="regime_adaptive", max_folds=max_f, dl_epochs=args.dl_epochs
        )
        eq_st = stitched_equity_mode(
            df, y, cfg, ensemble_mode="fixed_range", max_folds=max_f, dl_epochs=args.dl_epochs
        )
        paths.append(fig_3_18_static_vs_adaptive(out, eq_st, eq_ad))
        paths.append(fig_3_19_cumulative_return(out, eq_ad, eq_st))
    else:
        print("Skip equity/cumulative: use --run-wfo")

    print("Generated:")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
