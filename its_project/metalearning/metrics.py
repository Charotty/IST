from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class MLMetrics:
    """Machine learning metrics."""

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculate all ML metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
            except ValueError:
                metrics["roc_auc"] = 0.0
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        return metrics

    @staticmethod
    def calculate_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ["sell", "hold", "buy"],
    ) -> str:
        """Detailed classification report."""
        return classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )


class TradingMetrics:
    """Trading strategy metrics."""

    @staticmethod
    def calculate_returns(positions: np.ndarray, price_returns: np.ndarray) -> np.ndarray:
        """Calculate strategy returns."""
        return positions[:-1] * price_returns[1:]

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """Sharpe Ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0
        return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Maximum Drawdown."""
        cumulative = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative) / cumulative
        return abs(drawdown.min())

    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Win rate percentage."""
        if len(returns) == 0:
            return 0.0
        winning_trades = (returns > 0).sum()
        return winning_trades / len(returns)

    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Profit Factor (gross profit / gross loss)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def calculate_all(
        positions: np.ndarray, price_returns: np.ndarray, equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all trading metrics."""
        strategy_returns = TradingMetrics.calculate_returns(positions, price_returns)
        return {
            "total_return": equity_curve[-1] / equity_curve[0] - 1,
            "sharpe_ratio": TradingMetrics.calculate_sharpe_ratio(strategy_returns),
            "max_drawdown": TradingMetrics.calculate_max_drawdown(equity_curve),
            "win_rate": TradingMetrics.calculate_win_rate(strategy_returns),
            "profit_factor": TradingMetrics.calculate_profit_factor(strategy_returns),
            "num_trades": len(strategy_returns),
            "avg_return": strategy_returns.mean(),
            "volatility": strategy_returns.std(),
        }
