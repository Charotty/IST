"""Generate figures for diploma section 3.3 (feature engineering)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering.config import DIRECTION_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_3"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sample = df.loc["2022-06-01":"2022-08-01"].copy()

    # --- Indicator charts ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("BTC/USDT 1h — технические индикаторы (фрагмент выборки)", fontsize=11)

    axes[0].plot(sample.index, sample["close"], color="#1f77b4", linewidth=0.9, label="Close")
    axes[0].plot(sample.index, sample["ema_fast"], color="#ff7f0e", linewidth=0.8, label="EMA(20)")
    axes[0].plot(sample.index, sample["ema_slow"], color="#2ca02c", linewidth=0.8, label="EMA(50)")
    axes[0].set_ylabel("Цена")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Тренд: цена и EMA")

    axes[1].plot(sample.index, sample["rsi"], color="#9467bd", linewidth=0.9)
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.6, alpha=0.7)
    axes[1].set_ylabel("RSI")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Импульс: RSI(14)")

    axes[2].plot(sample.index, sample["atr"], color="#d62728", linewidth=0.9)
    axes[2].set_ylabel("ATR")
    axes[2].set_xlabel("Дата (UTC)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Волатильность: ATR(14)")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "indicators_rsi_atr_ema.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Scaling before / after ---
    feats = df[DIRECTION_FEATURE_COLUMNS].dropna().iloc[500:900]
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(feats),
        columns=feats.columns,
        index=feats.index,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Нормализация признаков (StandardScaler)", fontsize=11)

    for col in ["rsi", "ema_slope", "atr"]:
        if col == "atr" and col not in feats.columns:
            continue
    # plot subset of direction features (atr not in direction set — use macd_hist)
    plot_cols = ["rsi", "ema_slope", "macd_hist"]
    for ax, data, title in [
        (axes[0], feats[plot_cols], "До нормализации (исходный масштаб)"),
        (axes[1], scaled[plot_cols], "После нормализации (μ=0, σ=1)"),
    ]:
        for c in plot_cols:
            ax.plot(data.index, data[c], linewidth=0.8, label=c)
        ax.set_ylabel("Значение")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Дата (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "scaling_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Add ATR panel separately for scaling doc (optional combined)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    atr_raw = df["atr"].dropna().iloc[500:900]
    atr_sc = (atr_raw - atr_raw.mean()) / atr_raw.std()
    axes[0].hist(atr_raw, bins=40, color="#d62728", alpha=0.75, edgecolor="white")
    axes[0].set_title("ATR — распределение до нормализации")
    axes[0].set_xlabel("ATR")
    axes[1].hist(atr_sc, bins=40, color="#2ca02c", alpha=0.75, edgecolor="white")
    axes[1].set_title("ATR — после z-score")
    axes[1].set_xlabel("z(ATR)")
    fig.tight_layout()
    fig.savefig(OUT / "atr_scaling_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
