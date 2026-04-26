"""
Model Evaluator Module

Provides comprehensive evaluation metrics for trading models:
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Regression metrics: R², MAE, RMSE, MAPE
- Trading-specific metrics: Sharpe Ratio, Max Drawdown, Cumulative Return
- Visualization: ROC curves, Precision-Recall curves, Confusion Matrix, Equity Curve
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import warnings

# Try to import sklearn metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, precision_recall_curve,
        confusion_matrix, mean_squared_error, mean_absolute_error,
        r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ModelEvaluator:
    """Comprehensive model evaluator for trading models."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification model with comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            class_names: Names of classes (optional)

        Returns:
            Dictionary with all metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                # For multi-class, use one-vs-rest
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
                    metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            except Exception as e:
                metrics["roc_auc_error"] = str(e)

        metrics["evaluated_at"] = datetime.now().isoformat()
        return metrics

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate regression model with comprehensive metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with all metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        metrics = {}

        # Basic metrics
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE (Mean Absolute Percentage Error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics["mape"] = mape

        # Direction accuracy (for trading)
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics["direction_accuracy"] = np.mean(direction_true == direction_pred)

        metrics["evaluated_at"] = datetime.now().isoformat()
        return metrics

    def evaluate_trading_performance(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate trading-specific metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)

        Returns:
            Dictionary with trading metrics
        """
        metrics = {}

        # Cumulative return
        cumulative_return = np.prod(1 + returns) - 1
        metrics["cumulative_return"] = cumulative_return

        # Sharpe Ratio (annualized)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
            # Annualize (assuming daily returns)
            metrics["sharpe_ratio"] = sharpe * np.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0.0

        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics["max_drawdown"] = np.min(drawdown)

        # Win Rate
        metrics["win_rate"] = np.mean(returns > 0)

        # Profit Factor
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        metrics["profit_factor"] = profits / (losses + 1e-10)

        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_cumulative = np.prod(1 + benchmark_returns) - 1
            metrics["benchmark_return"] = benchmark_cumulative
            metrics["excess_return"] = cumulative_return - benchmark_cumulative

        metrics["evaluated_at"] = datetime.now().isoformat()
        return metrics

    def plot_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot classification metrics: Confusion Matrix, ROC Curve, Precision-Recall.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib Figure or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Classification Model Evaluation", fontsize=16, fontweight="bold")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        im = axes[0, 0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")
        fig.colorbar(im, ax=axes[0, 0])

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, format(cm[i, j], "d"),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")

        # ROC Curve (if probabilities available)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            axes[0, 1].plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
            axes[0, 1].plot([0, 1], [0, 1], "k--")
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, "ROC Curve\n(Probabilities not available)",
                          ha="center", va="center", transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("ROC Curve")

        # Precision-Recall Curve
        if y_proba is not None and len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            axes[1, 0].plot(recall, precision)
            axes[1, 0].set_xlabel("Recall")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].set_title("Precision-Recall Curve")
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, "Precision-Recall Curve\n(Probabilities not available)",
                          ha="center", va="center", transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Precision-Recall Curve")

        # Per-class metrics bar chart
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        x = np.arange(len(class_names))
        width = 0.25

        axes[1, 1].bar(x - width, precision_per_class, width, label="Precision")
        axes[1, 1].bar(x, recall_per_class, width, label="Recall")
        axes[1, 1].bar(x + width, f1_per_class, width, label="F1")
        axes[1, 1].set_xlabel("Class")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Per-Class Metrics")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot regression metrics: Actual vs Predicted, Residuals, Error Distribution.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib Figure or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Regression Model Evaluation", fontsize=16, fontweight="bold")

        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0, 0].set_xlabel("Actual")
        axes[0, 0].set_ylabel("Predicted")
        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].grid(True)

        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residual Plot")
        axes[0, 1].grid(True)

        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor="black")
        axes[1, 0].axvline(x=0, color="r", linestyle="--")
        axes[1, 0].set_xlabel("Residual")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Residual Distribution")
        axes[1, 0].grid(True)

        # Metrics summary
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        metrics_text = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nMAPE = {mape:.2f}%"
        axes[1, 1].text(0.5, 0.5, metrics_text,
                       ha="center", va="center",
                       transform=axes[1, 1].transAxes,
                       fontsize=14,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axes[1, 1].set_title("Metrics Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_equity_curve(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot equity curve with drawdowns.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib Figure or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Trading Performance: Equity Curve & Drawdown", fontsize=16, fontweight="bold")

        # Equity curve
        cumulative = np.cumprod(1 + returns)
        axes[0].plot(cumulative, label="Portfolio", linewidth=2)

        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            axes[0].plot(benchmark_cumulative, label="Benchmark", linewidth=2, alpha=0.7)

        axes[0].set_ylabel("Cumulative Return")
        axes[0].set_title("Equity Curve")
        axes[0].legend()
        axes[0].grid(True)

        # Drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color="red")
        axes[1].plot(drawdown, color="red", linewidth=1)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Time")
        axes[1].set_title("Drawdown")
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def print_metrics_report(self, metrics: Dict[str, Any], model_name: str = "Model") -> None:
        """Print a formatted metrics report."""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'='*60}")

        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        # Classification metrics
        if "accuracy" in metrics:
            print(f"\nClassification Metrics:")
            print(f"  Accuracy:           {metrics['accuracy']:.4f}")
            print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
            print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")

            if "roc_auc" in metrics:
                print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
            elif "roc_auc_ovr" in metrics:
                print(f"  ROC-AUC (OVR):      {metrics['roc_auc_ovr']:.4f}")

            print(f"\n  Per-Class Metrics:")
            for i, (p, r, f) in enumerate(zip(
                metrics['precision_per_class'],
                metrics['recall_per_class'],
                metrics['f1_per_class']
            )):
                print(f"    Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

        # Regression metrics
        if "r2" in metrics:
            print(f"\nRegression Metrics:")
            print(f"  R²:                 {metrics['r2']:.4f}")
            print(f"  MAE:                {metrics['mae']:.4f}")
            print(f"  RMSE:               {metrics['rmse']:.4f}")
            print(f"  MAPE:               {metrics['mape']:.2f}%")
            print(f"  Direction Accuracy:  {metrics['direction_accuracy']:.4f}")

        # Trading metrics
        if "cumulative_return" in metrics:
            print(f"\nTrading Metrics:")
            print(f"  Cumulative Return:  {metrics['cumulative_return']:.4f}")
            print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown:       {metrics['max_drawdown']:.4f}")
            print(f"  Win Rate:           {metrics['win_rate']:.4f}")
            print(f"  Profit Factor:      {metrics['profit_factor']:.4f}")

            if "benchmark_return" in metrics:
                print(f"  Benchmark Return:   {metrics['benchmark_return']:.4f}")
                print(f"  Excess Return:     {metrics['excess_return']:.4f}")

        print(f"\nEvaluated at: {metrics.get('evaluated_at', 'N/A')}")
        print(f"{'='*60}\n")
