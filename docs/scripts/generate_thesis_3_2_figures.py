"""
Thesis figures for section 3.2 (ensemble models).

Output: docs/thesis/3_2/figures/
  fig_3_4_ensemble_pipeline.png
  fig_3_5_regime_weights.png
  fig_3_6_model_predictions.png
  fig_3_7_lgb_feature_importance.png
  fig_3_8_xgb_feature_importance.png
  fig_3_4_ensemble_pipeline.mmd

Usage:
  py -3 docs/scripts/generate_thesis_3_2_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

OUT_DIR = ROOT / "docs" / "thesis" / "3_2" / "figures"
DEFAULT_FEATURES = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
DEFAULT_PROFILE = ROOT / "config" / "profiles" / "canonical_4model.yaml"
ARTIFACTS_ROOT = ROOT / "artifacts" / "BTC-USDT_1h"


def _ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def _load_weights(profile: Path) -> tuple[dict, dict]:
    raw = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
    orch = raw.get("orchestration") or {}
    return orch.get("trend_weights", {}), orch.get("range_weights", {})


def fig_3_4_pipeline(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1.2, 4.5, "Features\n(§3.1)", "#E8F5E9"),
        (3.6, 5.2, "LightGBM\nP(up)", "#FFF3E0"),
        (3.6, 4.0, "XGBoost\nP(up)", "#FFF3E0"),
        (5.8, 5.2, "GRU\nP(up)", "#F3E5F5"),
        (5.8, 4.0, "CNN\nP(up)", "#F3E5F5"),
        (1.2, 2.2, "Regime\nSMA rule", "#E3F2FD"),
        (7.8, 4.6, "DynamicMeta\nWeighting", "#BBDEFB"),
        (9.5, 4.6, "meta_mgmt\nprob", "#90CAF9"),
        (9.5, 2.2, "Decision\nPipeline", "#64B5F6"),
    ]
    for x, y, label, color in boxes:
        w, h = 1.5, 0.9
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#37474F",
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    arrows = [
        (1.2, 4.0, 3.6, 4.8),
        (1.2, 4.0, 3.6, 3.6),
        (1.2, 4.0, 5.8, 4.8),
        (1.2, 4.0, 5.8, 3.6),
        (1.2, 2.7, 7.8, 4.2),
        (3.6, 5.2, 7.8, 4.9),
        (3.6, 4.0, 7.8, 4.5),
        (5.8, 5.2, 7.8, 4.7),
        (5.8, 4.0, 7.8, 4.3),
        (7.8, 4.6, 9.5, 4.6),
        (9.5, 4.0, 9.5, 2.7),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.2),
        )

    ax.set_title(
        "Рисунок 3.4 — Архитектура ансамбля моделей (IST)",
        fontsize=11,
        fontweight="bold",
    )
    path = out / "fig_3_4_ensemble_pipeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_5_weights(trend: dict, range_: dict, out: Path) -> Path:
    keys = ["lgb", "gru", "xgb", "cnn"]
    labels = ["LightGBM", "GRU", "XGBoost", "CNN"]
    x = np.arange(len(keys))
    w = 0.35
    t_vals = [trend.get(k, 0) for k in keys]
    r_vals = [range_.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, t_vals, w, label="Trend (regime=1)", color="#43A047")
    ax.bar(x + w / 2, r_vals, w, label="Range (regime=0)", color="#757575")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Вес в ансамбле")
    ax.set_ylim(0, 0.65)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (tv, rv) in enumerate(zip(t_vals, r_vals)):
        ax.text(i - w / 2, tv + 0.02, f"{tv:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, rv + 0.02, f"{rv:.2f}", ha="center", fontsize=8)

    path = out / "fig_3_5_regime_weights.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_6_predictions(features_path: Path, artifacts_root: Path, out: Path, bars: int) -> Path | None:
    latest = artifacts_root / "LATEST.txt"
    if not latest.is_file():
        print("Skip fig 3.6: no artifacts bundle (run train-final-symbol)")
        return None

    run_id = latest.read_text(encoding="utf-8").strip()
    bundle_dir = artifacts_root / run_id
    if not bundle_dir.is_dir():
        print(f"Skip fig 3.6: bundle dir missing {bundle_dir}")
        return None

    from orchestration import TrainingOrchestrator
    from orchestration.glue import inference_stack_from_bundle

    stack = inference_stack_from_bundle(bundle_dir)
    df = pd.read_parquet(features_path).tail(bars)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    orch = TrainingOrchestrator(stack["config"])
    orch.initialize(
        stack["models"],
        stack["regime_detector"],
        stack["meta_weighting"],
    )
    result = orch.run_pipeline(df)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"lgb": "#FF9800", "gru": "#9C27B0", "xgb": "#4CAF50", "cnn": "#2196F3"}
    for key, series in result.model_predictions.items():
        arr = np.asarray(series).reshape(-1)
        if len(arr) != len(x):
            arr = np.resize(arr, len(x))
        ax.plot(x, arr, label=key.upper(), color=colors.get(key, "#333"), alpha=0.75, linewidth=1.0)

    meta = np.asarray(result.meta_probabilities).reshape(-1)
    ax.plot(x, meta, label="meta_mgmt (ensemble)", color="#000", linewidth=2.0)
    ax.axhline(0.52, color="#999", linestyle="--", linewidth=0.8, label="direction_thr 0.52")
    ax.axhline(0.48, color="#999", linestyle=":", linewidth=0.8)

    regime = np.asarray(result.regime_predictions).reshape(-1)
    for i in range(1, len(regime)):
        if regime[i] != regime[i - 1]:
            ax.axvline(i, color="#B0BEC5", alpha=0.4, linewidth=0.8)

    ax.set_ylabel("P(up)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Бар (хвост истории, bundle inference)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    path = out / "fig_3_6_model_predictions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _tabular_booster(model_obj):
    """Resolve sklearn wrapper → underlying LGBM/XGB classifier."""
    if hasattr(model_obj, "inner"):
        model_obj = model_obj.inner
    if hasattr(model_obj, "model"):
        return model_obj.model, list(model_obj.feature_cols or [])
    raise TypeError(f"Unsupported model type: {type(model_obj)}")


def _importance_horizontal(
    names: list[str],
    values: np.ndarray,
    *,
    ax,
    bar_color: str,
    top_n: int = 15,
) -> None:
    order = np.argsort(values)[-top_n:]
    names_top = [names[i] for i in order]
    vals_top = values[order]
    y = np.arange(len(names_top))
    ax.barh(y, vals_top, color=bar_color, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("Feature importance (gain)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def fig_feature_importance(
    artifacts_root: Path,
    out: Path,
    *,
    top_n: int = 15,
) -> list[Path]:
    """LightGBM и XGBoost importance из bundle (встроенные атрибуты booster)."""
    import pickle

    latest = artifacts_root / "LATEST.txt"
    if not latest.is_file():
        print("Skip feature importance: no bundle")
        return []

    run_id = latest.read_text(encoding="utf-8").strip()
    art_dir = artifacts_root / run_id / "artifacts"
    paths: list[Path] = []

    specs = [
        ("lgb.pkl", "LightGBM", "#FF9800", "fig_3_7_lgb_feature_importance.png"),
        ("xgb.pkl", "XGBoost", "#4CAF50", "fig_3_8_xgb_feature_importance.png"),
    ]

    for fname, title, color, out_name in specs:
        pkl = art_dir / fname
        if not pkl.is_file():
            print(f"Skip {fname}: not found")
            continue
        with open(pkl, "rb") as f:
            wrapper = pickle.load(f)
        booster, cols = _tabular_booster(wrapper)
        imp = np.asarray(booster.feature_importances_, dtype=float)
        if len(cols) != len(imp):
            # fallback: booster feature names (LGB) or generic f0..fn
            fn = getattr(booster, "feature_names_in_", None)
            if fn is not None and len(fn) == len(imp):
                cols = list(fn)
            else:
                cols = [f"f{i}" for i in range(len(imp))]

        fig, ax = plt.subplots(figsize=(7, 5))
        _importance_horizontal(cols, imp, ax=ax, bar_color=color, top_n=top_n)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        out_path = out / out_name
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(out_path)

    return paths


def write_mermaid(out: Path) -> Path:
    text = """```mermaid
flowchart LR
    F[Features] --> R[Regime 0/1]
    F --> LGB[LGB P_up]
    F --> GRU[GRU P_up]
    F --> XGB[XGB P_up]
    F --> CNN[CNN P_up]
    R --> MW[DynamicMetaWeighting]
    LGB & GRU & XGB & CNN --> MW
    MW --> META[meta_mgmt_prob]
    META --> DEC[DecisionPipeline]
```
"""
    path = out / "fig_3_4_ensemble_pipeline.mmd"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--bars", type=int, default=250)
    args = parser.parse_args()

    if not args.features.is_file():
        print(f"Missing features: {args.features}")
        return 1

    out = _ensure_out()
    trend, range_ = _load_weights(args.profile)
    paths = [
        fig_3_4_pipeline(out),
        fig_3_5_weights(trend, range_, out),
        write_mermaid(out),
    ]
    p6 = fig_3_6_predictions(args.features, args.artifacts, out, args.bars)
    if p6:
        paths.append(p6)
    paths.extend(fig_feature_importance(args.artifacts, out))

    print("Generated:")
    for p in paths:
        if p:
            print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
