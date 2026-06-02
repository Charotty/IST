"""
Thesis figures for section 3.3 (trading decision system).

Output: docs/thesis/3_3/figures/
  fig_3_9_decision_pipeline.png
  fig_3_10_confidence_filtering.png
  fig_3_11_execution_logic.png
  fig_3_12_trading_signals_example.png
  fig_3_9_decision_pipeline.mmd
  fig_3_11_execution_logic.mmd

Usage:
  py -3 docs/scripts/generate_thesis_3_3_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT_DIR = ROOT / "docs" / "thesis" / "3_3" / "figures"
DEFAULT_FEATURES = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
ARTIFACTS_ROOT = ROOT / "artifacts" / "BTC-USDT_1h"


def _ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def fig_3_9_pipeline(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1.5, 4.8, "meta_mgmt_prob\n(§3.2)", "#BBDEFB"),
        (4.2, 4.8, "DecisionPipeline\nτ_dir=0.52", "#90CAF9"),
        (6.9, 4.8, "final_signal\n{-1,0,1}", "#64B5F6"),
        (4.2, 2.8, "Confidence filters\nmargin · trade_mode\nATR vol", "#FFE082"),
        (9.2, 4.8, "position_sizes\n(risk bridge)", "#C8E6C9"),
        (9.2, 2.2, "Backtester\n(WFO)", "#A5D6A7"),
        (6.9, 1.2, "InferenceOrchestrator\n(predict)", "#E1BEE7"),
        (9.2, 0.5, "ExecutionManager\npaper | live", "#CE93D8"),
    ]
    for x, y, label, color in boxes:
        w, h = 2.0, 1.0
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
        (2.5, 4.8, 3.2, 4.8),
        (5.2, 4.8, 5.9, 4.8),
        (4.2, 4.3, 4.2, 3.3),
        (4.2, 2.3, 5.9, 4.5),
        (7.9, 4.8, 8.2, 4.8),
        (9.2, 4.3, 9.2, 2.7),
        (6.9, 0.7, 8.2, 0.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.2),
        )

    path = out / "fig_3_9_decision_pipeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_11_execution_logic(out: Path) -> Path:
    """
    Рис. 3.11 — логика исполнения (из архитектуры IST, не скрин GUI).

    Signal → Position Sizing → ExecutionManager → Broker (Paper / Live).
    """
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    steps = [
        (
            1.5,
            2.6,
            "Trading Signal\nfinal_signal ∈ {-1,0,1}\nInferenceOrchestrator",
            "#BBDEFB",
        ),
        (
            4.5,
            2.6,
            "Position Sizing\nOrchestratorRiskBridge\nATR · risk_per_trade",
            "#C8E6C9",
        ),
        (
            7.5,
            2.6,
            "ExecutionManager\nexecute_signal()\nOrderManager",
            "#F3E5F5",
        ),
        (
            10.5,
            2.6,
            "Broker\nPaperBroker (paper)\nOKXBroker (live)",
            "#FFCCBC",
        ),
    ]
    for x, y, label, color in steps:
        w, h = 2.2, 1.35
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#37474F",
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 1.1
        x2 = steps[i + 1][0] - 1.1
        ax.annotate(
            "",
            xy=(x2, 2.6),
            xytext=(x1, 2.6),
            arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.4),
        )

    ax.text(
        6.0,
        0.85,
        "signal = 0 → ордер не создаётся  ·  × risk_multiplier (rl_layer)  ·  paper_loop / paper_evidence",
        ha="center",
        fontsize=8.5,
        color="#546E7A",
    )

    path = out / "fig_3_11_execution_logic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _run_pipeline_tail(features_path: Path, artifacts_root: Path, bars: int):
    latest = artifacts_root / "LATEST.txt"
    if not latest.is_file():
        return None
    run_id = latest.read_text(encoding="utf-8").strip()
    bundle_dir = artifacts_root / run_id
    if not bundle_dir.is_dir():
        return None

    from orchestration import TrainingOrchestrator
    from orchestration.glue import inference_stack_from_bundle
    from utils.data_leakage_prevention import compute_safe_threshold

    stack = inference_stack_from_bundle(bundle_dir)
    df = pd.read_parquet(features_path).tail(bars)
    orch = TrainingOrchestrator(stack["config"])
    orch.initialize(
        stack["models"],
        stack["regime_detector"],
        stack["meta_weighting"],
    )
    result = orch.run_pipeline(df)
    meta = np.asarray(result.meta_probabilities, dtype=float).reshape(-1)
    thr_mode = "median"
    if orch.config.meta_threshold_mode in ("median", "mean"):
        thr_mode = orch.config.meta_threshold_mode
    rolling_thr = compute_safe_threshold(
        meta,
        thr_mode,
        getattr(orch.config, "threshold_rolling_window", 100) or 100,
        False,
        None,
    )
    if isinstance(rolling_thr, pd.Series):
        thr_arr = rolling_thr.to_numpy(dtype=float)
    else:
        thr_arr = np.full_like(meta, float(rolling_thr))
    margin = float(getattr(orch.config, "min_signal_margin", 0.0) or 0.0)
    return df, meta, thr_arr, np.asarray(result.final_signals).reshape(-1), margin


def fig_3_10_confidence(
    features_path: Path, artifacts_root: Path, out: Path, bars: int
) -> Path | None:
    data = _run_pipeline_tail(features_path, artifacts_root, bars)
    if data is None:
        print("Skip fig 3.10: no bundle")
        return None
    _df, meta, thr_arr, _sig, margin = data
    x = np.arange(len(meta))
    conf = np.abs(meta - 0.5) * 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes

    ax0.plot(x, meta, color="#1565C0", linewidth=1.2, label="meta_mgmt_prob")
    ax0.plot(x, thr_arr, color="#E65100", linewidth=1.0, linestyle="--", label="meta threshold (safe rolling)")
    ax0.fill_between(x, thr_arr, 1.0, where=meta > thr_arr, alpha=0.15, color="#4CAF50", label="pass meta gate")
    ax0.axhline(0.52, color="#9E9E9E", linestyle=":", linewidth=0.8)
    ax0.axhline(0.48, color="#9E9E9E", linestyle=":", linewidth=0.8)
    if margin > 0:
        ax0.axhspan(0.5 - margin, 0.5 + margin, alpha=0.12, color="#F44336", label=f"margin zone ±{margin}")
    ax0.set_ylabel("P(up)")
    ax0.set_ylim(0, 1)
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.grid(True, alpha=0.25)

    ax1.fill_between(x, 0, conf, color="#7E57C2", alpha=0.5)
    ax1.plot(x, conf, color="#4527A0", linewidth=0.9, label="|p−0.5|×2 (UI confidence)")
    ax1.set_ylabel("confidence")
    ax1.set_xlabel("Бар (хвост истории)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    path = out / "fig_3_10_confidence_filtering.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_12_signals_example(
    features_path: Path, artifacts_root: Path, out: Path, bars: int
) -> Path | None:
    """Доп. иллюстрация: пример ряда сигналов (не рис. 3.11 по подписи ВКР)."""
    data = _run_pipeline_tail(features_path, artifacts_root, bars)
    if data is None:
        print("Skip fig 3.12: no bundle")
        return None
    _df, meta, _thr, sig, _margin = data
    x = np.arange(len(meta))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(x, meta, color="#1565C0", alpha=0.6, linewidth=1.0, label="meta_mgmt_prob")
    ax1.set_ylabel("P(up)", color="#1565C0")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.52, color="#BDBDBD", linestyle="--", linewidth=0.7)
    ax1.axhline(0.48, color="#BDBDBD", linestyle="--", linewidth=0.7)

    ax2.step(x, sig, where="mid", color="#C62828", linewidth=1.5, label="final_signal")
    ax2.set_ylabel("signal", color="#C62828")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_ylim(-1.4, 1.4)

    ax1.set_xlabel("Бар (хвост истории, bundle)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    path = out / "fig_3_12_trading_signals_example.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_mermaid(out: Path) -> list[Path]:
    paths = []
    m9 = """flowchart LR
  E[meta_mgmt_prob] --> DP[DecisionPipeline]
  DP --> F[margin / trade_mode / ATR]
  F --> S[final_signal]
  S --> R[position_sizes]
  R --> BT[Backtester]
  S --> IO[InferenceOrchestrator]
  IO --> EM[ExecutionManager]
"""
    p9 = out / "fig_3_9_decision_pipeline.mmd"
    p9.write_text(m9, encoding="utf-8")
    paths.append(p9)

    m11 = """flowchart LR
  SIG[Trading Signal\nfinal_signal] --> PS[Position Sizing\nOrchestratorRiskBridge]
  PS --> EM[ExecutionManager\nexecute_signal]
  EM --> BR[Broker\nPaperBroker / OKXBroker]
"""
    p11 = out / "fig_3_11_execution_logic.mmd"
    p11.write_text(m11, encoding="utf-8")
    paths.append(p11)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--bars", type=int, default=400)
    args = parser.parse_args()

    out = _ensure_out()
    paths = [
        fig_3_9_pipeline(out),
        fig_3_11_execution_logic(out),
        *write_mermaid(out),
    ]
    p10 = fig_3_10_confidence(args.features, args.artifacts, out, args.bars)
    if p10:
        paths.append(p10)
    p12 = fig_3_12_signals_example(args.features, args.artifacts, out, args.bars)
    if p12:
        paths.append(p12)

    print("Generated:")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
