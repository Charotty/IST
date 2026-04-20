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
    def calculate(returns: np.ndarray, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate trading performance metrics.

        Args:
            returns: Price returns array
            predictions: Trading signals (0=sell, 1=hold, 2=buy)
            probabilities: Prediction probabilities
        """
        if len(returns) != len(predictions):
            raise ValueError("Returns and predictions must have same length")

        # Convert predictions to trading positions
        # 0 -> sell (-1), 1 -> hold (0), 2 -> buy (1)
        positions = np.where(predictions == 2, 1, np.where(predictions == 0, -1, 0))

        # Calculate trading returns (position * price return)
        trading_returns = positions * returns

        # Calculate additional metrics
        gross_returns = trading_returns[trading_returns != 0]

        return {
            "total_return": np.sum(trading_returns),
            "sharpe_ratio": TradingMetrics._calculate_sharpe_ratio(trading_returns),
            "sortino_ratio": TradingMetrics._calculate_sortino_ratio(trading_returns),
            "calmar_ratio": TradingMetrics._calculate_calmar_ratio(trading_returns),
            "max_drawdown": TradingMetrics._calculate_max_drawdown(trading_returns),
            "max_drawdown_duration": TradingMetrics._calculate_drawdown_duration(trading_returns),
            "win_rate": np.mean(gross_returns > 0) if len(gross_returns) > 0 else 0,
            "loss_rate": np.mean(gross_returns < 0) if len(gross_returns) > 0 else 0,
            "profit_factor": TradingMetrics._calculate_profit_factor(gross_returns),
            "avg_win": np.mean(gross_returns[gross_returns > 0]) if np.any(gross_returns > 0) else 0,
            "avg_loss": np.mean(gross_returns[gross_returns < 0]) if np.any(gross_returns < 0) else 0,
            "avg_win_loss_ratio": TradingMetrics._calculate_avg_win_loss_ratio(gross_returns),
            "num_trades": len(gross_returns),
            "num_wins": np.sum(gross_returns > 0),
            "num_losses": np.sum(gross_returns < 0),
            "avg_return": np.mean(trading_returns),
            "volatility": np.std(trading_returns) * np.sqrt(252),
            "downside_deviation": TradingMetrics._calculate_downside_deviation(trading_returns),
            "var_95": np.percentile(trading_returns, 5),
            "cvar_95": np.mean(trading_returns[trading_returns <= np.percentile(trading_returns, 5)]),
        }

    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        max_dd = TradingMetrics._calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        return annual_return / abs(max_dd)

    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def _calculate_drawdown_duration(returns: np.ndarray) -> float:
        """Calculate maximum drawdown duration in periods."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative < running_max

        # Find consecutive drawdown periods
        if not np.any(drawdown):
            return 0.0

        # Count consecutive True values
        durations = []
        current_duration = 0

        for is_dd in drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        # Add final duration if ending in drawdown
        if current_duration > 0:
            durations.append(current_duration)

        return max(durations) if durations else 0.0

    @staticmethod
    def _calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        return gross_profit / (gross_loss + 1e-10)

    @staticmethod
    def _calculate_avg_win_loss_ratio(returns: np.ndarray) -> float:
        """Calculate average win to loss ratio."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        return avg_win / (avg_loss + 1e-10)

    @staticmethod
    def _calculate_downside_deviation(returns: np.ndarray) -> float:
        """Calculate downside deviation."""
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        return np.std(downside_returns) * np.sqrt(252)
