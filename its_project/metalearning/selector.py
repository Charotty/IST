from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from its_project.models.base import BaseModel
from its_project.metalearning.metrics import MLMetrics, TradingMetrics

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    """Configuration for auto-retraining."""
    enabled: bool = False
    retrain_interval_hours: float = 24.0
    min_samples_for_retrain: int = 1000
    performance_threshold: float = 0.5  # Minimum metric to trigger retrain
    retrain_on_degradation: bool = True
    degradation_threshold: float = 0.1  # 10% degradation triggers retrain


@dataclass
class RetrainState:
    """State tracking for auto-retraining."""
    last_retrain_time: Optional[datetime] = None
    last_performance: Optional[Dict[str, float]] = None
    retrain_count: int = 0
    samples_since_last_retrain: int = 0


class ModelSelector:
    """Automatic model selection based on metrics with auto-retrain support."""

    def __init__(
        self,
        models: List[BaseModel],
        metrics: List[str] = ["accuracy", "sharpe_ratio"],
        weights: Optional[List[float]] = None,
        retrain_config: Optional[RetrainConfig] = None,
    ) -> None:
        self.models = models
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.results: Optional[pd.DataFrame] = None
        self.retrain_config = retrain_config or RetrainConfig()
        self.retrain_state = RetrainState()
        self._retrain_task: Optional[asyncio.Task] = None
        self._is_running = False

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

    def should_retrain(self, current_performance: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if retraining is needed based on configuration and state.
        
        Args:
            current_performance: Current model performance metrics
            
        Returns:
            True if retraining should occur
        """
        if not self.retrain_config.enabled:
            return False
        
        # Check minimum samples
        if self.retrain_state.samples_since_last_retrain < self.retrain_config.min_samples_for_retrain:
            return False
        
        # Check time interval
        if self.retrain_state.last_retrain_time:
            time_since_retrain = datetime.now() - self.retrain_state.last_retrain_time
            if time_since_retrain < timedelta(hours=self.retrain_config.retrain_interval_hours):
                return False
        
        # Check performance degradation
        if self.retrain_config.retrain_on_degradation and current_performance and self.retrain_state.last_performance:
            for metric in self.metrics:
                if metric in current_performance and metric in self.retrain_state.last_performance:
                    last_val = self.retrain_state.last_performance[metric]
                    current_val = current_performance[metric]
                    if last_val > 0:
                        degradation = (last_val - current_val) / last_val
                        if degradation > self.retrain_config.degradation_threshold:
                            logger.warning(f"Performance degradation detected: {metric} degraded by {degradation:.2%}")
                            return True
        
        # Check performance threshold
        if current_performance:
            for metric in self.metrics:
                if metric in current_performance and current_performance[metric] < self.retrain_config.performance_threshold:
                    logger.info(f"Performance below threshold: {metric} = {current_performance[metric]:.4f}")
                    return True
        
        return False

    async def retrain(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        price_returns: Optional[np.ndarray] = None,
        on_retrain_complete: Optional[Callable] = None,
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Retrain all models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            price_returns: Price returns for trading metrics
            on_retrain_complete: Callback after retrain completes
            
        Returns:
            Best model and its scores
        """
        logger.info("Starting model retraining...")
        
        # Evaluate all models
        results = self.evaluate_all(X_train, y_train, X_test, y_test, price_returns)
        
        # Select best model
        best_model, best_scores = self.select_best()
        
        # Update state
        self.retrain_state.last_retrain_time = datetime.now()
        self.retrain_state.last_performance = best_scores
        self.retrain_state.retrain_count += 1
        self.retrain_state.samples_since_last_retrain = 0
        
        logger.info(f"Retraining complete. Best model: {best_model.__class__.__name__}")
        logger.info(f"Best scores: {best_scores}")
        
        # Call callback if provided
        if on_retrain_complete:
            try:
                if asyncio.iscoroutinefunction(on_retrain_complete):
                    await on_retrain_complete(best_model, best_scores)
                else:
                    on_retrain_complete(best_model, best_scores)
            except Exception as e:
                logger.error(f"Error in retrain callback: {e}")
        
        return best_model, best_scores

    def update_sample_count(self, count: int) -> None:
        """Update the count of samples since last retrain."""
        self.retrain_state.samples_since_last_retrain += count

    async def start_auto_retrain(
        self,
        get_training_data: Callable,
        on_retrain_complete: Optional[Callable] = None,
    ) -> None:
        """
        Start automatic retraining loop.
        
        Args:
            get_training_data: Async function that returns (X_train, y_train, X_test, y_test, price_returns)
            on_retrain_complete: Callback after each retrain
        """
        if not self.retrain_config.enabled:
            logger.warning("Auto-retrain is not enabled")
            return
        
        if self._is_running:
            logger.warning("Auto-retrain is already running")
            return
        
        self._is_running = True
        logger.info("Starting auto-retrain loop")
        
        async def retrain_loop():
            while self._is_running:
                try:
                    # Wait for check interval
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Get current performance (could be from recent predictions)
                    current_performance = None
                    if self.results is not None and len(self.results) > 0:
                        best_idx = self.results[self.metrics[0]].argmax()
                        current_performance = self.results.iloc[best_idx].to_dict()
                    
                    # Check if retrain is needed
                    if self.should_retrain(current_performance):
                        logger.info("Triggering automatic retraining")
                        
                        # Get training data
                        X_train, y_train, X_test, y_test, price_returns = await get_training_data()
                        
                        # Perform retraining
                        await self.retrain(
                            X_train, y_train, X_test, y_test, price_returns, on_retrain_complete
                        )
                    
                except asyncio.CancelledError:
                    logger.info("Auto-retrain loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in auto-retrain loop: {e}")
                    await asyncio.sleep(60)  # Wait before retry
        
        self._retrain_task = asyncio.create_task(retrain_loop())

    async def stop_auto_retrain(self) -> None:
        """Stop automatic retraining loop."""
        if not self._is_running:
            return
        
        logger.info("Stopping auto-retrain loop")
        self._is_running = False
        
        if self._retrain_task:
            self._retrain_task.cancel()
            try:
                await self._retrain_task
            except asyncio.CancelledError:
                pass
            self._retrain_task = None

    def get_retrain_state(self) -> RetrainState:
        """Get current retrain state."""
        return self.retrain_state
