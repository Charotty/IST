"""
Проверка OOS walk-forward метрик против порогов (а) acceptance / (б) target.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .metrics_config import BacktestingConfig, CriteriaThresholds, load_backtesting_config


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _safe_min(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.min()) if len(vals) else float("nan")


def summarize_wfo_folds(fold_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Агрегаты по фолдам WFO (OOS колонки без префикса IS_)."""
    if fold_metrics.empty:
        return {"n_folds": 0}

    oos_cols = [c for c in fold_metrics.columns if not str(c).startswith("IS_")]
    df = fold_metrics[oos_cols]

    summary: Dict[str, Any] = {
        "n_folds": int(len(df)),
        "mean_sharpe": _safe_mean(df.get("Sharpe Ratio", pd.Series(dtype=float))),
        "mean_profit_factor": _safe_mean(df.get("Profit Factor", pd.Series(dtype=float))),
        "mean_total_return_pct": _safe_mean(df.get("Total Return (%)", pd.Series(dtype=float))),
        "mean_wfe": _safe_mean(df.get("Walk-Forward Efficiency", pd.Series(dtype=float))),
        "mean_wfe_sharpe": _safe_mean(
            df.get("Walk-Forward Efficiency (Sharpe)", pd.Series(dtype=float))
        ),
        "mean_calmar": _safe_mean(df.get("Calmar Ratio", pd.Series(dtype=float))),
        "mean_sortino": _safe_mean(df.get("Sortino Ratio", pd.Series(dtype=float))),
        "worst_max_drawdown_pct": _safe_min(df.get("Max Drawdown (%)", pd.Series(dtype=float))),
        "mean_recovery_factor": _safe_mean(df.get("Recovery Factor", pd.Series(dtype=float))),
        "total_trade_events": int(
            pd.to_numeric(df.get("Trade Events", 0), errors="coerce").fillna(0).sum()
        ),
        "folds_positive_return": int(
            (pd.to_numeric(df.get("Total Return (%)", 0), errors="coerce") > 0).sum()
        ),
    }
    if "Sharpe Ratio" in df.columns:
        summary["sharpe_std_across_folds"] = float(
            pd.to_numeric(df["Sharpe Ratio"], errors="coerce").std()
        )
    return summary


def evaluate_criteria(
    fold_metrics: pd.DataFrame,
    thresholds: CriteriaThresholds,
    *,
    level_name: str = "acceptance",
) -> Dict[str, Any]:
    """
    Проверка агрегированных OOS-метрик против порогов.

    :return: dict с ключами level, passed, checks (list of {name, passed, detail})
    """
    summary = summarize_wfo_folds(fold_metrics)
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": passed, "detail": detail})

    n_folds = summary.get("n_folds", 0)
    add(
        "min_folds",
        n_folds >= thresholds.min_folds,
        f"n_folds={n_folds}, required>={thresholds.min_folds}",
    )

    mean_pf = summary.get("mean_profit_factor", float("nan"))
    pf_ok = mean_pf >= thresholds.min_oos_profit_factor if np.isfinite(mean_pf) else False
    add(
        "min_oos_profit_factor",
        pf_ok,
        f"mean_pf={mean_pf:.4f}, required>={thresholds.min_oos_profit_factor}",
    )

    mean_sharpe = summary.get("mean_sharpe", float("nan"))
    sharpe_ok = (
        mean_sharpe > thresholds.min_oos_sharpe if np.isfinite(mean_sharpe) else False
    )
    add(
        "min_oos_sharpe",
        sharpe_ok,
        f"mean_sharpe={mean_sharpe:.4f}, required>{thresholds.min_oos_sharpe}",
    )

    mean_wfe = summary.get("mean_wfe", float("nan"))
    wfe_ok = mean_wfe >= thresholds.min_wfe if np.isfinite(mean_wfe) else False
    add(
        "min_wfe",
        wfe_ok,
        f"mean_wfe={mean_wfe:.4f}, required>={thresholds.min_wfe}",
    )

    worst_dd = summary.get("worst_max_drawdown_pct", float("nan"))
    dd_ok = (
        worst_dd >= thresholds.max_drawdown_pct if np.isfinite(worst_dd) else False
    )
    add(
        "max_drawdown_pct",
        dd_ok,
        f"worst_dd={worst_dd:.2f}%, limit>={thresholds.max_drawdown_pct}%",
    )

    mean_rf = summary.get("mean_recovery_factor", float("nan"))
    rf_ok = (
        mean_rf >= thresholds.min_recovery_factor if np.isfinite(mean_rf) else False
    )
    add(
        "min_recovery_factor",
        rf_ok,
        f"mean_recovery={mean_rf:.4f}, required>={thresholds.min_recovery_factor}",
    )

    mean_ret = summary.get("mean_total_return_pct", float("nan"))
    ret_ok = (
        mean_ret > thresholds.min_total_return_pct if np.isfinite(mean_ret) else False
    )
    add(
        "min_total_return_pct",
        ret_ok,
        f"mean_return={mean_ret:.2f}%, required>{thresholds.min_total_return_pct}%",
    )

    trades = summary.get("total_trade_events", 0)
    add(
        "min_trade_events",
        trades >= thresholds.min_trade_events,
        f"total_trade_events={trades}, required>={thresholds.min_trade_events}",
    )

    if thresholds.min_calmar is not None:
        mean_calmar = summary.get("mean_calmar", float("nan"))
        calmar_ok = (
            mean_calmar >= thresholds.min_calmar if np.isfinite(mean_calmar) else False
        )
        add(
            "min_calmar",
            calmar_ok,
            f"mean_calmar={mean_calmar:.4f}, required>={thresholds.min_calmar}",
        )

    if thresholds.min_sortino is not None:
        mean_sortino = summary.get("mean_sortino", float("nan"))
        sortino_ok = (
            mean_sortino >= thresholds.min_sortino
            if np.isfinite(mean_sortino)
            else False
        )
        add(
            "min_sortino",
            sortino_ok,
            f"mean_sortino={mean_sortino:.4f}, required>={thresholds.min_sortino}",
        )

    passed = all(c["passed"] for c in checks)
    return {
        "level": level_name,
        "passed": passed,
        "summary": summary,
        "thresholds": asdict(thresholds),
        "checks": checks,
    }


def evaluate_backtest_levels(
    fold_metrics: pd.DataFrame,
    config: BacktestingConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or load_backtesting_config()
    return {
        "acceptance": evaluate_criteria(
            fold_metrics, cfg.acceptance, level_name="acceptance"
        ),
        "target": evaluate_criteria(fold_metrics, cfg.target, level_name="target"),
        "summary": summarize_wfo_folds(fold_metrics),
    }
