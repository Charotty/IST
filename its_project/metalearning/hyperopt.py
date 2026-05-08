from __future__ import annotations

from typing import Dict, Any, Type, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

try:
    import optuna
except ImportError:  # pragma: no cover - exercised only when optional dep is absent
    optuna = None

from its_project.models.base import BaseModel
from its_project.metalearning.metrics import TradingMetrics


class HyperparameterOptimizer:
    """Hyperparameter optimization with Optuna."""

    def __init__(
        self,
        model_class: Type[BaseModel],
        param_space: Dict[str, Any],
        metric: str = "sharpe_ratio",
        n_trials: int = 100,
    ) -> None:
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.n_trials = n_trials
        self.best_params: Optional[Dict[str, Any]] = None

    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        price_returns: Optional[np.ndarray] = None,
    ) -> float:
        """Optuna objective."""
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
        model = self.model_class(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        if self.metric == "accuracy":
            score = accuracy_score(y_val, y_pred)
        elif self.metric == "f1":
            score = f1_score(y_val, y_pred, average="weighted")
        elif self.metric == "sharpe_ratio":
            if price_returns is None:
                raise ValueError("price_returns required for sharpe_ratio metric")
            positions = self._predictions_to_positions(y_pred)
            strategy_returns = TradingMetrics.calculate_returns(positions, price_returns)
            score = TradingMetrics.calculate_sharpe_ratio(strategy_returns)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return score

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        price_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run optimization."""
        if optuna is None:
            raise RuntimeError("optuna is required for HyperparameterOptimizer.optimize")
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, price_returns),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        self.best_params = study.best_params
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }

    @staticmethod
    def _predictions_to_positions(predictions: np.ndarray) -> np.ndarray:
        """Convert predictions to positions (-1, 0, 1)."""
        return predictions - 1
