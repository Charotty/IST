"""Figures for diploma section 3.7 (adaptive ensemble / dynamic weighting)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_7"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
ABLATION = ROOT / "docs" / "reports" / "ablation_canonical.json"

TREND_W = {"lgb": 0.10, "xgb": 0.10, "gru": 0.45, "cnn": 0.35}
RANGE_W = {"lgb": 0.55, "xgb": 0.25, "gru": 0.10, "cnn": 0.10}
STATIC_W = {k: 0.25 for k in TREND_W}


def regime_pred_from_adx(df: pd.DataFrame, adx_thr: float = 25.0) -> np.ndarray:
    adx = df["adx"].astype(float)
    return (adx > adx_thr).astype(int).to_numpy()


def weights_over_time(regime: np.ndarray) -> pd.DataFrame:
    n = len(regime)
    out = {m: np.zeros(n) for m in TREND_W}
    for i, r in enumerate(regime):
        w = TREND_W if r == 1 else RANGE_W
        for m in out:
            out[m][i] = w[m]
    return pd.DataFrame(out, index=df_index)


def main() -> None:
    global df_index
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    seg = df.loc["2023-01-01":"2023-08-01"]
    if len(seg) < 200:
        seg = df.iloc[-2500:]
    df_index = seg.index
    regime = regime_pred_from_adx(seg)
    wdf = weights_over_time(regime)

    # Line chart — adaptive weights
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = {"lgb": "#2ca02c", "xgb": "#1f77b4", "gru": "#ff7f0e", "cnn": "#9467bd"}
    labels = {"lgb": "LightGBM", "xgb": "XGBoost", "gru": "GRU", "cnn": "CNN"}
    x = np.arange(len(wdf))
    for m in wdf.columns:
        ax.plot(x, wdf[m].values, label=labels[m], color=colors[m], linewidth=1.0, alpha=0.9)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Вес $w_i$")
    ax.set_xlabel("Индекс бара (BTC/USDT, 1h)")
    ax.set_title("Динамика весов моделей (regime-adaptive ensemble)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "weights_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Static vs adaptive — flat lines
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for m in STATIC_W:
        axes[0].plot(x, np.full(len(x), STATIC_W[m]), color=colors[m], label=labels[m])
    axes[0].set_ylabel("Вес")
    axes[0].set_title("Статический ансамбль (равные веса $w_i = 0{,}25$)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].set_ylim(0, 0.35)
    axes[0].grid(True, alpha=0.3)

    for m in wdf.columns:
        axes[1].plot(x, wdf[m].values, color=colors[m], label=labels[m])
    axes[1].set_ylabel("Вес")
    axes[1].set_xlabel("Индекс бара")
    axes[1].set_title("Адаптивный ансамбль (dynamic weighting)")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].set_ylim(0, 0.65)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "static_vs_adaptive_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Metrics comparison from ablation JSON
    if ABLATION.is_file():
        data = json.loads(ABLATION.read_text(encoding="utf-8"))
        static = next(v for v in data["variants"] if v["variant"] == "four_equal")
        adaptive = next(v for v in data["variants"] if v["variant"] == "four_regime_adaptive")
        metrics = ["mean_sharpe", "mean_pf", "mean_wfe"]
        titles = ["Sharpe (OOS)", "Profit Factor", "WFE"]
        s_vals = [static[m] for m in metrics]
        a_vals = [adaptive[m] for m in metrics]

        fig, ax = plt.subplots(figsize=(7, 4))
        xpos = np.arange(len(metrics))
        w = 0.35
        ax.bar(xpos - w / 2, s_vals, w, label="Статический (равные веса)", color="#9e9e9e")
        ax.bar(xpos + w / 2, a_vals, w, label="Адаптивный (regime-adaptive)", color="#2ca02c")
        ax.set_xticks(xpos)
        ax.set_xticklabels(titles)
        ax.set_ylabel("Значение метрики")
        ax.set_title("Сравнение ансамблей (WFO, 4 модели)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "static_vs_adaptive_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
