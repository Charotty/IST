#!/usr/bin/env python3
"""Plot OKX matrix benchmark report.

Reads JSON/CSV from okx_matrix_benchmark.py and generates:
- Heatmaps (models x timeframes) for test F1 macro and test accuracy
- Overfitting gap heatmap: (train_f1_macro - test_f1_macro)
- Optional per-symbol splits (one figure per symbol)

Usage:
  python okx_plot_report.py --report okx_reports/okx_matrix_YYYYMMDD_HHMMSS.csv
  python okx_plot_report.py --report okx_reports/okx_matrix_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_df(report_path: Path):
    import pandas as pd

    if report_path.suffix.lower() == ".csv":
        return pd.read_csv(report_path)
    if report_path.suffix.lower() == ".json":
        data = json.loads(report_path.read_text(encoding="utf-8"))
        # flatten like CSV output
        rows = []
        for r in data.get("results", []):
            row = {
                "symbol": r.get("symbol"),
                "timeframe": r.get("timeframe"),
                "model": r.get("model"),
                "status": r.get("status"),
            }
            if r.get("status") == "ok":
                tr = (r.get("train") or {}).get("classification") or {}
                te = (r.get("test") or {}).get("classification") or {}
                row.update({
                    "train_accuracy": tr.get("accuracy"),
                    "train_f1_macro": tr.get("f1_macro"),
                    "test_accuracy": te.get("accuracy"),
                    "test_f1_macro": te.get("f1_macro"),
                })
            else:
                row["reason"] = r.get("reason") or r.get("error")
            rows.append(row)
        return pd.DataFrame(rows)

    raise ValueError("Unsupported report format")


def _heatmap(df, value_col: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = df.pivot(index="model", columns="timeframe", values=value_col)

    plt.figure(figsize=(12, max(6, 0.35 * len(pivot.index))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _lineplot_models_over_timeframes(df, value_col: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    order = ["1m", "5m", "15m", "1h", "4h", "1d"]
    df2 = df.copy()
    df2["timeframe"] = df2["timeframe"].astype(str)
    df2 = df2[df2["timeframe"].isin(order)]
    df2["timeframe"] = pd.Categorical(df2["timeframe"], categories=order, ordered=True)
    df2 = df2.sort_values(["model", "timeframe"])

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df2, x="timeframe", y=value_col, hue="model", marker="o")
    plt.title(title)
    plt.xlabel("Timeframe")
    plt.ylabel(value_col)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _barplot_top_models_per_timeframe(df, value_col: str, title: str, out_path: Path, top_n: int = 5):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    order = ["1m", "5m", "15m", "1h", "4h", "1d"]
    df2 = df.copy()
    df2 = df2[df2["timeframe"].isin(order)]
    df2["timeframe"] = pd.Categorical(df2["timeframe"], categories=order, ordered=True)

    parts = []
    for tf in order:
        sub = df2[df2["timeframe"] == tf].sort_values(value_col, ascending=False).head(top_n)
        parts.append(sub)
    if not parts:
        return
    df_top = pd.concat(parts, axis=0)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_top, x="timeframe", y=value_col, hue="model")
    plt.title(title)
    plt.xlabel("Timeframe")
    plt.ylabel(value_col)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _scatter_train_vs_test(df, metric_col: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_col = f"train_{metric_col}"
    test_col = f"test_{metric_col}"
    df2 = df[["model", "timeframe", train_col, test_col]].dropna().copy()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df2, x=train_col, y=test_col, hue="timeframe", style="model")
    mn = float(min(df2[train_col].min(), df2[test_col].min()))
    mx = float(max(df2[train_col].max(), df2[test_col].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray", linewidth=1)
    plt.title(title)
    plt.xlabel(train_col)
    plt.ylabel(test_col)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _gap_distribution(df, gap_col: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df2 = df[["model", "timeframe", gap_col]].dropna().copy()
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df2, x="timeframe", y=gap_col)
    sns.stripplot(data=df2, x="timeframe", y=gap_col, color="black", alpha=0.35, size=3)
    plt.title(title)
    plt.xlabel("Timeframe")
    plt.ylabel(gap_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OKX benchmark report")
    parser.add_argument("--report", type=str, required=True, help="Path to okx_matrix_*.csv or .json")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    df = _load_df(report_path)

    # Keep only successful rows
    df_ok = df[df["status"] == "ok"].copy()

    # Convert to numeric
    for col in ["train_accuracy", "train_f1_macro", "test_accuracy", "test_f1_macro"]:
        if col in df_ok.columns:
            df_ok[col] = df_ok[col].astype(float)

    df_ok["f1_gap"] = df_ok["train_f1_macro"] - df_ok["test_f1_macro"]
    df_ok["acc_gap"] = df_ok["train_accuracy"] - df_ok["test_accuracy"]

    out_dir = report_path.parent / (report_path.stem + "_plots")
    out_dir.mkdir(exist_ok=True)

    # One set per symbol
    for symbol in sorted(df_ok["symbol"].unique()):
        sub = df_ok[df_ok["symbol"] == symbol]

        _heatmap(
            sub,
            "test_f1_macro",
            f"OKX: Test F1-macro (models x timeframes) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_test_f1_macro.png",
        )
        _heatmap(
            sub,
            "test_accuracy",
            f"OKX: Test Accuracy (models x timeframes) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_test_accuracy.png",
        )
        _heatmap(
            sub,
            "f1_gap",
            f"OKX: Overfitting proxy (train_f1 - test_f1) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_f1_gap.png",
        )

        _lineplot_models_over_timeframes(
            sub,
            "test_f1_macro",
            f"OKX: Test F1-macro across timeframes | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_line_test_f1_macro.png",
        )
        _lineplot_models_over_timeframes(
            sub,
            "test_accuracy",
            f"OKX: Test accuracy across timeframes | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_line_test_accuracy.png",
        )
        _barplot_top_models_per_timeframe(
            sub,
            "test_f1_macro",
            f"OKX: Top models per timeframe (Test F1-macro) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_bar_top_test_f1_macro.png",
            top_n=5,
        )
        _scatter_train_vs_test(
            sub,
            "f1_macro",
            f"OKX: Train vs Test F1-macro (overfitting diagnostic) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_scatter_train_vs_test_f1_macro.png",
        )
        _scatter_train_vs_test(
            sub,
            "accuracy",
            f"OKX: Train vs Test accuracy (overfitting diagnostic) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_scatter_train_vs_test_accuracy.png",
        )
        _gap_distribution(
            sub,
            "f1_gap",
            f"OKX: Distribution of generalization gap (F1) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_dist_f1_gap.png",
        )
        _gap_distribution(
            sub,
            "acc_gap",
            f"OKX: Distribution of generalization gap (Accuracy) | {symbol}",
            out_dir / f"{symbol.replace('/', '_')}_dist_acc_gap.png",
        )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
