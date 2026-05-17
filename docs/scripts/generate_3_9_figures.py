"""Figures for diploma section 3.9 (risk management)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_9"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"


def proxy_prob(df: pd.DataFrame) -> pd.Series:
    rsi_n = (df["rsi"].astype(float) / 100.0).clip(0.05, 0.95)
    slope = df["ema_slope"].astype(float).fillna(0.0)
    z = 2.2 * (rsi_n - 0.5) + 80.0 * slope
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=df.index)


def signals_from_prob(p: pd.Series, thr: float = 0.52) -> pd.Series:
    mt = p.rolling(100, min_periods=20).median()
    leg = np.where(p > thr, 1, np.where(p < (1 - thr), -1, 0))
    sig = np.where((p > mt) & (leg != 0), leg, 0)
    return pd.Series(sig, index=p.index, dtype=int)


def main() -> None:
    from risk_management import RiskPipeline

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    seg = df.loc["2022-08-01":"2024-01-01"].copy()
    if len(seg) < 500:
        seg = df.iloc[-4000:].copy()

    p = proxy_prob(seg)
    seg["final_signal"] = signals_from_prob(p)

    cfg = {
        "position_sizer": {
            "risk_per_trade": 0.02,
            "account_size": 10000,
            "atr_stop_multiplier": 2.0,
        },
        "trailing_stop": {"enabled": True, "atr_mult": 3.0},
    }
    rp = RiskPipeline(config=cfg)
    seg = rp.apply_pipeline(seg, signal_col="final_signal")

    sig = seg["combined_signal"] if "combined_signal" in seg.columns else seg["final_signal"]
    ret = seg["close"].astype(float).pct_change().fillna(0.0)
    account = float(cfg["position_sizer"]["account_size"])
    frac = (seg["final_pos_size"].astype(float) * seg["close"].astype(float) / account).clip(0, 1)
    strat_ret = sig.astype(float) * ret * frac
    equity = (1 + strat_ret).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(seg))

    axes[0].plot(x, equity.values, color="#1f77b4", linewidth=1.0, label="Equity (с риск-контуром)")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    axes[0].set_ylabel("Норм. капитал")
    axes[0].set_title("Кривая капитала стратегии с ATR sizing и trailing stop (BTC/USDT 1h)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(x, drawdown.values * 100, 0, color="#d62728", alpha=0.45, label="Drawdown")
    axes[1].set_ylabel("Просадка, %")
    axes[1].set_xlabel("Индекс бара")
    mdd = float(drawdown.min() * 100)
    axes[1].set_title(f"Просадка (max drawdown ≈ {mdd:.1f}%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stop levels sample
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = seg.iloc[500:700]
    xi = np.arange(len(sub))
    ax.plot(xi, sub["close"].values, label="Close", color="#333", lw=0.9)
    long = sub["final_signal"] == 1
    if long.any() and sub.loc[long, "trailing_stop"].notna().any():
        ax.plot(xi, sub["trailing_stop"].values, label="ATR trailing stop (long)", color="#2ca02c", lw=0.8, alpha=0.8)
    ax.set_title("ATR trailing stop относительно цены (фрагмент)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "atr_trailing_stop_fragment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUT}; max_drawdown={mdd:.2f}%")


if __name__ == "__main__":
    main()
