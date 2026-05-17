"""Figures for diploma section 3.8 (decision system)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_8"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"


def proxy_ensemble_prob(df: pd.DataFrame) -> pd.Series:
    """Causal proxy for meta_mgmt_prob from features (illustration only)."""
    rsi_n = (df["rsi"].astype(float) / 100.0).clip(0.05, 0.95)
    slope = df["ema_slope"].astype(float).fillna(0.0)
    z = 2.2 * (rsi_n - 0.5) + 80.0 * slope
    p = 1.0 / (1.0 + np.exp(-z))
    return pd.Series(p, index=df.index)


def rolling_median_threshold(series: pd.Series, window: int = 100) -> pd.Series:
    return series.rolling(window, min_periods=20).median()


def generate_signals(
    p: pd.Series,
    direction_thr: float = 0.52,
    meta_thr: float | pd.Series | None = None,
    use_fixed_meta: float | None = None,
) -> pd.Series:
    if use_fixed_meta is not None:
        mt = use_fixed_meta
    elif meta_thr is None:
        mt = rolling_median_threshold(p)
    else:
        mt = meta_thr

    direction_leg = np.where(
        p > direction_thr,
        1,
        np.where(p < (1.0 - direction_thr), -1, 0),
    )
    if isinstance(mt, pd.Series):
        meta_pass = p > mt
    else:
        meta_pass = p > float(mt)

    sig = np.where(meta_pass & (direction_leg != 0), direction_leg, 0)
    return pd.Series(sig, index=p.index, dtype=int)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    seg = df.loc["2023-04-01":"2023-06-15"].copy()
    if len(seg) < 100:
        seg = df.iloc[-1500:].copy()

    p = proxy_ensemble_prob(seg)
    sig_canon = generate_signals(p, direction_thr=0.52)
    sig_strict = generate_signals(p, direction_thr=0.65, use_fixed_meta=0.55)

    close = seg["close"].astype(float)
    x = np.arange(len(seg))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(x, close.values, color="#333333", linewidth=0.9, label="Close")

    buy = sig_canon == 1
    sell = sig_canon == -1
    ax.scatter(x[buy], close.values[buy], marker="^", color="#2ca02c", s=42, label="BUY", zorder=5)
    ax.scatter(x[sell], close.values[sell], marker="v", color="#d62728", s=42, label="SELL", zorder=5)

    ax.set_ylabel("Цена, USDT")
    ax.set_title("BTC/USDT 1h — торговые сигналы BUY / SELL / HOLD (Decision Pipeline)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.plot(x, p.values, color="#1f77b4", linewidth=0.8, label=r"$\hat{P}_t$")
    ax2.axhline(0.52, color="#2ca02c", linestyle="--", linewidth=0.7, label=r"$\tau_{buy}=0{,}52$")
    ax2.axhline(0.48, color="#d62728", linestyle="--", linewidth=0.7, label=r"$\tau_{sell}=0{,}48$")
    mt = rolling_median_threshold(p)
    ax2.plot(x, mt.values, color="#ff7f0e", linewidth=0.7, alpha=0.8, label=r"$\tau_{meta}$ (rolling median)")
    ax2.set_ylim(0.2, 0.85)
    ax2.set_ylabel(r"$\hat{P}_t$")
    ax2.set_xlabel("Индекс бара")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "signals_buy_sell_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example table
    examples = []
    idx = np.where((sig_canon != 0))[0][:12]
    for i in idx:
        row = seg.iloc[i]
        s = int(sig_canon.iloc[i])
        label = "BUY" if s == 1 else "SELL" if s == -1 else "HOLD"
        examples.append(
            {
                "timestamp": str(seg.index[i])[:19],
                "P_hat": round(float(p.iloc[i]), 4),
                "signal": label,
                "close": round(float(row["close"]), 1),
            }
        )
    pd.DataFrame(examples).to_csv(OUT / "signal_examples.csv", index=False)

    # Strict threshold panel
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, close.values, color="#333", lw=0.8)
    b2 = sig_strict == 1
    s2 = sig_strict == -1
    ax.scatter(x[b2], close.values[b2], marker="^", c="#2ca02c", s=50, label="BUY (τ=0,65)")
    ax.scatter(x[s2], close.values[s2], marker="v", c="#d62728", s=50, label="SELL")
    ax.axhline(close.mean(), color="gray", ls=":", lw=0.5)
    ax.set_title("Усиленные пороги: BUY при P̂ > 0,65")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "signals_strict_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUT}; signals: buy={buy.sum()} sell={sell.sum()} hold={(sig_canon==0).sum()}")


if __name__ == "__main__":
    main()
