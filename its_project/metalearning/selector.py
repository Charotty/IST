from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from its_project.models.base import BaseModel
from its_project.metalearning.metrics import MLMetrics, TradingMetrics


class ModelSelector:
    """Automatic model selection based on metrics."""

    def __init__(
        self,
        models: List[BaseModel],
        metrics: List[str] = ["accuracy", "sharpe_ratio"],
        weights: Optional[List[float]] = None,
    ) -> None:
        self.models = models
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.results: Optional[pd.DataFrame] = None

    def evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        price_returns: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Evaluate all models."""
        results = []
        for i, model in enumerate(self.models):
            print(f"Evaluating model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            ml_metrics = MLMetrics.calculate_all(y_test, y_pred, y_proba)
            trading_metrics = {}
            if price_returns is not None:
                positions = self._predictions_to_positions(y_pred)
                equity = self._calculate_equity_curve(positions, price_returns)
                trading_metrics = TradingMetrics.calculate_all(
                    positions, price_returns, equity
                )
            model_result = {"model": model.__class__.__name__, **ml_metrics, **trading_metrics}
            results.append(model_result)

        self.results = pd.DataFrame(results)
        return self.results

    def select_best(self) -> Tuple[BaseModel, Dict[str, float]]:
        """Select best model by weighted score."""
        if self.results is None:
            raise RuntimeError("Call evaluate_all() first")
        normalized = self.results.copy()
        for metric in self.metrics:
            if metric in normalized.columns:
                min_val = normalized[metric].min()
                max_val = normalized[metric].max()
                if max_val - min_val > 0:
                    normalized[f"{metric}_norm"] = (
                        (normalized[metric] - min_val) / (max_val - min_val)
                    )
                else:
                    normalized[f"{metric}_norm"] = 0.0
        normalized["weighted_score"] = 0.0
        for metric, weight in zip(self.metrics, self.weights):
            if f"{metric}_norm" in normalized.columns:
                normalized["weighted_score"] += weight * normalized[f"{metric}_norm"]
        best_idx = normalized["weighted_score"].argmax()
        best_model = self.models[best_idx]
        best_scores = self.results.iloc[best_idx].to_dict()
        return best_model, best_scores

    def get_ranking(self) -> pd.DataFrame:
        """Get model ranking."""
        if self.results is None:
            raise RuntimeError("Call evaluate_all() first")
        return self.results.sort_values(by=self.metrics[0], ascending=False)

    @staticmethod
    def _predictions_to_positions(predictions: np.ndarray) -> np.ndarray:
        return predictions - 1

    @staticmethod
    def _calculate_equity_curve(
        positions: np.ndarray, returns: np.ndarray
    ) -> np.ndarray:
        strategy_returns = positions[:-1] * returns[1:]
        equity = np.cumprod(1 + strategy_returns)
        equity = np.insert(equity, 0, 1.0)
        return equity
