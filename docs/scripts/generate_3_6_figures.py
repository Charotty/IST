"""Generate regime labeling chart for diploma section 3.6."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_6"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"

ADX_TREND = 25.0
VOL_SPIKE_Q = 0.85


def classify_regimes(df: pd.DataFrame) -> pd.Series:
    """Rule-based regime labels for visualization (ADX / volatility)."""
    adx = df["adx"].astype(float)
    vol = df["volatility"].astype(float)
    vol_thr = vol.rolling(100, min_periods=20).quantile(VOL_SPIKE_Q)

    labels = pd.Series("range", index=df.index, dtype=object)
    labels.loc[adx > ADX_TREND] = "trend"
    spike = vol > vol_thr
    labels.loc[spike] = "volatility_spike"
    return labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    seg = df.loc["2023-03-01":"2023-06-01"].copy()
    if len(seg) < 100:
        seg = df.iloc[-2000:].copy()

    regimes = classify_regimes(seg)
    colors = {
        "trend": "#c6efce",
        "range": "#dce6f1",
        "volatility_spike": "#ffc7ce",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(seg))
    close = seg["close"].astype(float).values

    # Background spans by regime
    start = 0
    cur = regimes.iloc[0]
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes.iloc[i] != cur:
            ax.axvspan(start, i, facecolor=colors.get(cur, "#eeeeee"), alpha=0.55, linewidth=0)
            if i < len(regimes):
                start = i
                cur = regimes.iloc[i]

    ax.plot(x, close, color="#1a1a1a", linewidth=0.9, label="Close")
    ax.set_xlim(0, len(seg) - 1)
    ax.set_ylabel("Цена, USDT")
    ax.set_xlabel("Индекс бара (BTC/USDT, 1h)")
    ax.set_title("Разметка рыночных режимов на фрагменте выборки (2023)")

    patches = [
        mpatches.Patch(facecolor=colors["trend"], alpha=0.7, label="Trend (ADX > 25)"),
        mpatches.Patch(facecolor=colors["range"], alpha=0.7, label="Range"),
        mpatches.Patch(
            facecolor=colors["volatility_spike"],
            alpha=0.7,
            label=f"Volatility spike (σ > q{int(VOL_SPIKE_Q*100)})",
        ),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "regime_labeling_candles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Distribution bar
    counts = regimes.value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    order = ["trend", "range", "volatility_spike"]
    vals = [counts.get(k, 0) for k in order]
    ax.bar(order, vals, color=[colors[k] for k in order], edgecolor="white")
    ax.set_ylabel("Число баров")
    ax.set_title("Распределение режимов (фрагмент)")
    fig.tight_layout()
    fig.savefig(OUT / "regime_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
