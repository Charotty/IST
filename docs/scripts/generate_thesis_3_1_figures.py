"""
Generate thesis figures for section 3.1 (market data processing).

Output: docs/thesis/3_1/figures/
  fig_3_1_preprocessing_pipeline.png
  fig_3_2_ohlcv_ema_rsi.png
  fig_3_3_dataset_structure.png

Usage:
  py -3 docs/scripts/generate_thesis_3_1_figures.py
  py -3 docs/scripts/generate_thesis_3_1_figures.py --features data/features/BTC-USDT_1h.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
OUT_DIR = ROOT / "docs" / "thesis" / "3_1" / "figures"
HORIZON = 12


def _ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def fig_3_1_pipeline(out: Path) -> Path:
    """Рисунок 3.1 — общая схема preprocessing pipeline."""
    steps = [
        "OKX API\n(ccxt REST)",
        "Raw OHLCV\nParquet",
        "Validation\nindex, dedup, trim",
        "Multi-Timeframe\nMerge (15m, 4h → 1h)",
        "Feature\nEngineering",
        "Dataset\nFormation",
    ]
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    n = len(steps)
    box_w, box_h = 1.85, 1.35
    gap = (14 - n * box_w) / (n + 1)
    y = 1.5
    xs = [gap + i * (box_w + gap) + box_w / 2 for i in range(n)]
    colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5"]

    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor=color,
            edgecolor="#1565C0",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2 - 0.05, y),
                xytext=(x + box_w / 2 + 0.05, y),
                arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=2),
            )

    ax.set_title(
        "Рисунок 3.1 — Общая схема preprocessing pipeline (IST)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    path = out / "fig_3_1_preprocessing_pipeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _load_tail(features_path: Path, bars: int = 120) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
    return df.tail(bars).copy()


def fig_3_2_ohlcv_indicators(df: pd.DataFrame, out: Path) -> Path:
    """Рисунок 3.2 — свечи BTC-USDT + EMA + RSI."""
    required = {"open", "high", "low", "close", "ema_fast", "ema_slow", "rsi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for fig 3.2: {missing}")

    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    width = 0.6
    for i, row in enumerate(df.itertuples()):
        o, h, l, c = row.open, row.high, row.low, row.close
        color = "#26A69A" if c >= o else "#EF5350"
        ax1.vlines(i, l, h, color=color, linewidth=0.8)
        bottom = min(o, c)
        height = max(abs(c - o), (h - l) * 0.02)
        ax1.add_patch(
            mpatches.Rectangle((i - width / 2, bottom), width, height, facecolor=color, edgecolor=color)
        )

    ax1.plot(x, df["ema_fast"].values, color="#FF9800", linewidth=1.2, label="EMA(20)")
    ax1.plot(x, df["ema_slow"].values, color="#7B1FA2", linewidth=1.2, label="EMA(50)")
    ax1.set_ylabel("Price (USDT)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(
        "Рисунок 3.2 — BTC-USDT 1h: OHLCV, EMA и RSI (фрагмент)",
        fontsize=11,
        fontweight="bold",
    )

    ax2.plot(x, df["rsi"].values, color="#1976D2", linewidth=1.0)
    ax2.axhline(70, color="#999", linestyle="--", linewidth=0.8)
    ax2.axhline(30, color="#999", linestyle="--", linewidth=0.8)
    ax2.fill_between(x, df["rsi"].values, 50, alpha=0.08, color="#1976D2")
    ax2.set_ylabel("RSI(14)")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Бар (последние N часовых свечей)")
    ax2.grid(True, alpha=0.25)

    ticks = np.linspace(0, len(df) - 1, min(8, len(df)), dtype=int)
    labels = [df.index[i].strftime("%Y-%m-%d\n%H:%M") for i in ticks]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=7)

    path = out / "fig_3_2_ohlcv_ema_rsi.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_3_3_dataset_structure(df: pd.DataFrame, out: Path) -> Path:
    """Рисунок 3.3 — фрагмент DataFrame + shape + target."""
    close = df["close"].astype(float)
    target = (close.shift(-HORIZON) > close).astype("Int64")
    target.iloc[-HORIZON:] = pd.NA

    display_cols = [
        c
        for c in [
            "close",
            "rsi",
            "macd_hist",
            "ema_slope",
            "adx",
            "rsi_15m",
            "volatility",
            "target_up_h12",
        ]
        if c in df.columns or c == "target_up_h12"
    ]
    sample = df.tail(200).copy()
    sample["target_up_h12"] = target.reindex(sample.index)

    show_cols = [c for c in display_cols if c in sample.columns]
    head = sample[show_cols].tail(8).iloc[::-1].round(4)

    n_feat = len(
        [
            c
            for c in df.columns
            if c not in {"open", "high", "low", "close", "volume"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    )
    shape_text = (
        f"Dataset shape: ({len(df):,} rows × {len(df.columns)} columns)\n"
        f"Feature columns (numeric, excl. OHLCV): {n_feat}\n"
        f"Target: target_up_h12 = 1 if close[t+{HORIZON}] > close[t], else 0\n"
        f"Index: UTC DatetimeIndex (1h bars)\n"
        f"Source: {DEFAULT_FEATURES.name}"
    )

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.2], hspace=0.35)
    ax_info = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_info.axis("off")
    ax_table.axis("off")

    ax_info.text(
        0.02,
        0.95,
        shape_text,
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor="#2E7D32"),
    )
    ax_info.set_title(
        "Рисунок 3.3 — Структура сформированного датасета (фрагмент)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )

    cell_text = head.astype(str).values.tolist()
    col_labels = list(head.columns)
    row_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in head.index]

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    path = out / "fig_3_3_dataset_structure.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = out / "fig_3_3_dataset_sample.csv"
    head.to_csv(csv_path)
    return path


def write_mermaid_source(out: Path) -> Path:
    """Mermaid-исходник для экспорта в draw.io / mermaid.live."""
    text = """```mermaid
flowchart LR
    OKX["OKX API\\n(ccxt REST)"]
    RAW["Raw OHLCV\\nParquet"]
    VAL["Validation\\ndedup, UTC index"]
    MTF["Multi-Timeframe Merge\\n15m + 4h → 1h"]
    FE["Feature Engineering\\nFeatureManager"]
    DS["Dataset Formation\\nfeatures.parquet + target"]

    OKX --> RAW --> VAL --> MTF --> FE --> DS
```
"""
    path = out / "fig_3_1_preprocessing_pipeline.mmd"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Thesis §3.1 figures")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to features parquet (default: data/features/BTC-USDT_1h.parquet)",
    )
    parser.add_argument("--bars", type=int, default=120, help="Bars for OHLCV chart")
    args = parser.parse_args()

    if not args.features.is_file():
        print(f"Features file not found: {args.features}")
        print("Run: py -3 -m orchestration prepare-symbol --symbol BTC-USDT --timeframe 1h")
        return 1

    out = _ensure_out()
    df = _load_tail(args.features, bars=max(args.bars, 200))

    p1 = fig_3_1_pipeline(out)
    p2 = fig_3_2_ohlcv_indicators(_load_tail(args.features, bars=args.bars), out)
    p3 = fig_3_3_dataset_structure(pd.read_parquet(args.features), out)
    mmd = write_mermaid_source(out)

    print("Generated:")
    for p in (p1, p2, p3, mmd):
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
