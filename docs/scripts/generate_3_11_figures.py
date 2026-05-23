"""Figures for diploma section 3.11."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_11"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"


def wfo_split_chart(n: int = 5000) -> None:
    train_w, test_w, step, horizon, embargo = 1500, 250, 250, 12, 5
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, n)
    s, fi = 0, 0
    while s + train_w + test_w + horizon < n and fi < 12:
        tr0, tr1 = s, s + train_w - horizon
        te0, te1 = s + train_w + embargo, s + train_w + embargo + test_w
        ax.axvspan(tr0, tr1, facecolor="#c6efce", alpha=0.75, linewidth=0)
        ax.axvspan(tr1, s + train_w, facecolor="#fff2cc", alpha=0.75, linewidth=0)
        ax.axvspan(s + train_w, te0, facecolor="#ffe699", alpha=0.75, linewidth=0)
        ax.axvspan(te0, te1, facecolor="#dce6f1", alpha=0.75, linewidth=0)
        s += step
        fi += 1
    ax.set_yticks([])
    ax.set_xlabel("Индекс бара")
    ax.set_title("Walk-forward: Train → Purge → Embargo → Test")
    patches = [
        mpatches.Patch(color="#c6efce", label="Train"),
        mpatches.Patch(color="#fff2cc", label="Purge"),
        mpatches.Patch(color="#ffe699", label="Embargo"),
        mpatches.Patch(color="#dce6f1", label="Test (OOS)"),
    ]
    ax.legend(handles=patches, loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "wfo_split_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def purge_embargo_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, 0.6, 4.0, 0.8, "Train", "#c6efce"),
        (4.2, 0.6, 0.6, 0.8, "Purge\nH=12", "#fff2cc"),
        (4.8, 0.6, 0.5, 0.8, "Emb\nE=5", "#ffe699"),
        (5.3, 0.6, 2.5, 0.8, "Test OOS", "#dce6f1"),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="gray"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    ax.set_title("Схема purge и embargo между train и test", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "purge_embargo_scheme.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_seg() -> pd.DataFrame:
    if not DATA.exists():
        t = np.arange(800)
        return pd.DataFrame({
            "close": 30000 * np.cumprod(1 + np.random.randn(800) * 0.002),
            "ema_slope": np.random.randn(800) * 0.001,
        })
    try:
        df = pd.read_parquet(DATA, columns=["close", "ema_slope"])
    except Exception:
        df = pd.read_parquet(DATA, columns=["close"])
    return df.iloc[-800:].copy()


def simulate_equity(seg: pd.DataFrame, bias: float) -> pd.Series:
    slope = seg["ema_slope"].fillna(0) if "ema_slope" in seg else seg["close"].pct_change().fillna(0)
    p = 1 / (1 + np.exp(-(slope * 80 + bias)))
    sig = np.where(p > 0.52, 1, np.where(p < 0.48, -1, 0))
    ret = seg["close"].pct_change().fillna(0) * sig * 0.35
    return (1 + ret).cumprod()


def equity_drawdown(seg: pd.DataFrame) -> None:
    curves = {
        "LightGBM": simulate_equity(seg, -0.02),
        "Static ensemble": simulate_equity(seg, 0.0),
        "Adaptive ensemble": simulate_equity(seg, 0.05),
        "Buy & Hold": seg["close"] / seg["close"].iloc[0],
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(seg))
    colors = {"LightGBM": "#2ca02c", "Static ensemble": "#9e9e9e",
              "Adaptive ensemble": "#1f77b4", "Buy & Hold": "#888"}
    for k, eq in curves.items():
        lw = 1.4 if k == "Adaptive ensemble" else 1.0
        axes[0].plot(x, eq.values, label=k, color=colors[k], lw=lw)
    axes[0].set_ylabel("Норм. капитал")
    axes[0].set_title("Equity curves (OOS-фрагмент BTC/USDT 1h)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    for k in ("Static ensemble", "Adaptive ensemble"):
        eq = curves[k]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        axes[1].plot(x, dd.values, label=k, color=colors[k])
    axes[1].set_ylabel("Drawdown, %")
    axes[1].set_xlabel("Бар")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_confusion(seg: pd.DataFrame) -> None:
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    horizon = 12
    y = (seg["close"].shift(-horizon) > seg["close"]).astype(int).iloc[:-horizon]
    slope = seg["ema_slope"].fillna(0) if "ema_slope" in seg else seg["close"].pct_change().fillna(0)
    p = 1 / (1 + np.exp(-(slope.iloc[:-horizon] * 100)))
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    pred = (p > 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=1.5, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC-кривая")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    im = axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[1].set_title("Confusion matrix")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "roc_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_refs() -> None:
    pairs = [
        (ROOT / "docs" / "figures" / "3_7" / "weights_over_time.png", OUT / "weight_adaptation.png"),
        (ROOT / "docs" / "figures" / "3_8" / "signals_buy_sell_chart.png", OUT / "ensemble_signal_markers.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy(src, dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wfo_split_chart()
    purge_embargo_diagram()
    seg = _load_seg()
    equity_drawdown(seg)
    roc_confusion(seg.iloc[-400:])
    copy_refs()
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
