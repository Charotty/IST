from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd

from its_project.models.base import BaseModel
from its_project.metalearning.metrics import TradingMetrics

logger = logging.getLogger(__name__)


@dataclass
class WeightedEnsembleConfig:
    """Configuration for weighted ensemble with dynamic model selection."""
    
    # Score formula weights
    alpha: float = 0.4      # Weight for Sharpe ratio
    beta: float = 0.4        # Weight for PnL
    gamma: float = 0.2       # Weight for Drawdown (penalty)
    
    # Dynamic selection parameters
    window_size: int = 100        # Sliding window size for performance calculation
    min_trades_for_switch: int = 20  # Minimum trades before considering model switch
    switch_threshold: float = 0.1   # Performance degradation threshold for switching
    
    # Smoothing parameters
    score_smoothing: bool = True
    smoothing_window: int = 10
    
    # Model weights for ensemble prediction
    ensemble_weights: Optional[List[float]] = None  # If None, use equal weights
    
    # Confidence filtering parameters
    confidence_threshold: float = 0.6  # Minimum confidence for valid predictions
    use_confidence_filter: bool = True  # Enable/disable confidence filtering
    confidence_strategy: str = "max_proba"  # Strategy: "max_proba", "entropy", "margin"


@dataclass
class ModelPerformance:
    """Track individual model performance over sliding window."""
    
    model_name: str
    sharpe_history: deque = field(default_factory=lambda: deque(maxlen=200))
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=200))
    drawdown_history: deque = field(default_factory=lambda: deque(maxlen=200))
    score_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    current_score: float = 0.0
    best_score: float = 0.0
    trades_count: int = 0
    
    def update_performance(
        self, 
        sharpe: float, 
        pnl: float, 
        drawdown: float,
        config: WeightedEnsembleConfig
    ) -> None:
        """Update model performance with new metrics."""
        
        # Add to history
        self.sharpe_history.append(sharpe)
        self.pnl_history.append(pnl)
        self.drawdown_history.append(drawdown)
        self.trades_count += 1
        
        # Calculate current score using formula: Score = α·Sharpe + β·PnL - γ·DD
        current_score = (
            config.alpha * sharpe + 
            config.beta * pnl - 
            config.gamma * drawdown
        )
        
        self.score_history.append(current_score)
        
        # Update best score
        if current_score > self.best_score:
            self.best_score = current_score
        
        # Apply smoothing if enabled
        if config.score_smoothing and len(self.score_history) >= config.smoothing_window:
            recent_scores = list(self.score_history)[-config.smoothing_window:]
            self.current_score = np.mean(recent_scores)
        else:
            self.current_score = current_score
    
    def get_window_metrics(self, window_size: int) -> Dict[str, float]:
        """Get metrics over sliding window."""
        if len(self.score_history) == 0:
            return {"score": 0.0, "sharpe": 0.0, "pnl": 0.0, "drawdown": 0.0}
        
        window_scores = list(self.score_history)[-window_size:]
        window_sharpe = list(self.sharpe_history)[-window_size:]
        window_pnl = list(self.pnl_history)[-window_size:]
        window_drawdown = list(self.drawdown_history)[-window_size:]
        
        return {
            "score": np.mean(window_scores) if window_scores else 0.0,
            "sharpe": np.mean(window_sharpe) if window_sharpe else 0.0,
            "pnl": np.mean(window_pnl) if window_pnl else 0.0,
            "drawdown": np.mean(window_drawdown) if window_drawdown else 0.0,
            "trades": len(window_scores)
        }


class WeightedEnsemble(BaseModel):
    """
    Weighted ensemble with dynamic model selection based on:
    Score = α·Sharpe + β·PnL - γ·DD
    
    Features:
    - Real-time model switching based on performance degradation
    - Sliding window performance calculation
    - Configurable score formula weights
    - Smoothed score calculation
    """
    
    def __init__(
        self,
        models: List[BaseModel],
        config: WeightedEnsembleConfig = None,
        **kwargs
    ) -> None:
        super().__init__(config or {})
        
        self.models = models
        self.config = config or WeightedEnsembleConfig()
        
        # Initialize performance tracking for each model
        self.model_performances: Dict[str, ModelPerformance] = {}
        for model in models:
            self.model_performances[model.__class__.__name__] = ModelPerformance(
                model_name=model.__class__.__name__
            )
        
        # Current active model
        self.current_model_idx: int = 0
        self.current_model_name: str = models[0].__class__.__name__
        
        # Ensemble weights (for weighted prediction)
        if self.config.ensemble_weights is None:
            self.ensemble_weights = [1.0 / len(models)] * len(models)
        else:
            self.ensemble_weights = self.config.ensemble_weights
        
        # Performance tracking
        self.total_trades: int = 0
        self.last_switch_trade: int = 0
        
        logger.info(f"WeightedEnsemble initialized with {len(models)} models")
        logger.info(f"Score weights: α={self.config.alpha}, β={self.config.beta}, γ={self.config.gamma}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedEnsemble":
        """Fit all models in the ensemble."""
        logger.info(f"Fitting {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(X, y)
        
        self._is_fitted = True
        logger.info("All models fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using current best model or weighted ensemble."""
        if not self._is_fitted:
            raise RuntimeError("WeightedEnsemble not fitted")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Use weighted ensemble for final prediction
        weighted_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, self.ensemble_weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted ensemble."""
        if not self._is_fitted:
            raise RuntimeError("WeightedEnsemble not fitted")
        
        # Get probabilities from all models
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)
        
        # Use weighted ensemble for final probabilities
        weighted_proba = np.zeros_like(all_probas[0])
        for proba, weight in zip(all_probas, self.ensemble_weights):
            weighted_proba += weight * proba
        
        return weighted_proba
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores based on p_hat thresholding.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self._is_fitted:
            raise RuntimeError("WeightedEnsemble not fitted")
        
        # Get probabilities
        probas = self.predict_proba(X)
        
        # Calculate confidence scores based on strategy
        confidence_scores = self._calculate_confidence(probas)
        
        # Apply confidence filtering if enabled
        if self.config.use_confidence_filter:
            predictions = self._filter_by_confidence(probas, confidence_scores)
        else:
            predictions = np.argmax(probas, axis=1)
        
        return predictions, confidence_scores
    
    def _calculate_confidence(self, probas: np.ndarray) -> np.ndarray:
        """
        Calculate confidence scores from probabilities.
        
        Args:
            probas: Probability array of shape (n_samples, n_classes)
            
        Returns:
            Confidence scores array of shape (n_samples,)
        """
        if self.config.confidence_strategy == "max_proba":
            # Maximum probability (most confident class)
            confidence = np.max(probas, axis=1)
            
        elif self.config.confidence_strategy == "entropy":
            # Inverse entropy (lower entropy = higher confidence)
            entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
            # Normalize entropy to [0, 1] confidence
            max_entropy = np.log(probas.shape[1])  # Maximum possible entropy
            confidence = 1.0 - (entropy / max_entropy)
            
        elif self.config.confidence_strategy == "margin":
            # Margin between top two probabilities
            sorted_probas = np.sort(probas, axis=1)
            if sorted_probas.shape[1] >= 2:
                confidence = sorted_probas[:, -1] - sorted_probas[:, -2]
            else:
                confidence = sorted_probas[:, -1]
                
        else:
            raise ValueError(f"Unknown confidence strategy: {self.config.confidence_strategy}")
        
        return confidence
    
    def _filter_by_confidence(self, probas: np.ndarray, confidence_scores: np.ndarray) -> np.ndarray:
        """
        Filter predictions based on confidence threshold.
        
        Args:
            probas: Probability array
            confidence_scores: Confidence scores
            
        Returns:
            Filtered predictions (low confidence predictions become HOLD class)
        """
        predictions = np.argmax(probas, axis=1)
        
        # Apply confidence filter
        low_confidence_mask = confidence_scores < self.config.confidence_threshold
        
        # Convert low confidence predictions to HOLD (class 1)
        predictions[low_confidence_mask] = 1
        
        return predictions
    
    def get_confidence_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get confidence statistics for predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with confidence statistics
        """
        probas = self.predict_proba(X)
        confidence_scores = self._calculate_confidence(probas)
        
        stats = {
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "min_confidence": np.min(confidence_scores),
            "max_confidence": np.max(confidence_scores),
            "above_threshold": np.sum(confidence_scores >= self.config.confidence_threshold),
            "below_threshold": np.sum(confidence_scores < self.config.confidence_threshold),
            "threshold_ratio": np.mean(confidence_scores >= self.config.confidence_threshold),
            "strategy": self.config.confidence_strategy,
            "threshold": self.config.confidence_threshold
        }
        
        # Add class distribution
        predictions = np.argmax(probas, axis=1)
        for class_id in range(probas.shape[1]):
            class_mask = predictions == class_id
            if np.any(class_mask):
                stats[f"class_{class_id}_mean_confidence"] = np.mean(confidence_scores[class_mask])
                stats[f"class_{class_id}_count"] = np.sum(class_mask)
        
        return stats
    
    def update_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        price_returns: Optional[np.ndarray] = None
    ) -> None:
        """
        Update performance metrics for all models after trading period.
        
        Args:
            y_true: True labels
            y_pred: Predictions from current active model
            price_returns: Price returns for trading metrics calculation
        """
        if price_returns is None:
            logger.warning("No price returns provided for performance update")
            return
        
        # Calculate trading metrics for current predictions
        trading_metrics = TradingMetrics.calculate(y_true, y_pred, price_returns)
        
        # Update performance for current model
        current_perf = self.model_performances[self.current_model_name]
        current_perf.update_performance(
            sharpe=trading_metrics.get("sharpe_ratio", 0.0),
            pnl=trading_metrics.get("total_return", 0.0),
            drawdown=trading_metrics.get("max_drawdown", 0.0),
            config=self.config
        )
        
        self.total_trades += len(y_true)
        
        # Check if we should switch models
        self._check_and_switch_model()
    
    def _check_and_switch_model(self) -> None:
        """Check performance and switch to best model if needed."""
        
        # Need minimum trades before considering switch
        if self.total_trades - self.last_switch_trade < self.config.min_trades_for_switch:
            return
        
        # Get current window metrics for all models
        model_scores = []
        for model_name, perf in self.model_performances.items():
            window_metrics = perf.get_window_metrics(self.config.window_size)
            
            # Need minimum trades in window
            if window_metrics["trades"] < self.config.min_trades_for_switch:
                model_scores.append((model_name, float('-inf')))  # Invalid score
                continue
            
            model_scores.append((model_name, window_metrics["score"]))
        
        if not model_scores:
            return
        
        # Find best model
        best_model_name, best_score = max(model_scores, key=lambda x: x[1])
        current_score = self.model_performances[self.current_model_name].current_score
        
        # Switch if significant improvement
        if (best_score > current_score + self.config.switch_threshold and
            best_model_name != self.current_model_name):
            
            old_model = self.current_model_name
            self.current_model_name = best_model_name
            
            # Update current model index
            for i, model in enumerate(self.models):
                if model.__class__.__name__ == best_model_name:
                    self.current_model_idx = i
                    break
            
            self.last_switch_trade = self.total_trades
            
            logger.info(f"Switched model: {old_model} -> {best_model_name}")
            logger.info(f"Score change: {current_score:.4f} -> {best_score:.4f}")
            logger.info(f"Improvement: {(best_score - current_score):.4f}")
    
    def get_current_model(self) -> BaseModel:
        """Get currently active model."""
        return self.models[self.current_model_idx]
    
    def get_model_rankings(self) -> pd.DataFrame:
        """Get current rankings of all models."""
        rankings = []
        
        for model_name, perf in self.model_performances.items():
            window_metrics = perf.get_window_metrics(self.config.window_size)
            
            rankings.append({
                "model": model_name,
                "current_score": perf.current_score,
                "best_score": perf.best_score,
                "window_score": window_metrics["score"],
                "window_sharpe": window_metrics["sharpe"],
                "window_pnl": window_metrics["pnl"],
                "window_drawdown": window_metrics["drawdown"],
                "window_trades": window_metrics["trades"],
                "total_trades": perf.trades_count,
                "is_current": model_name == self.current_model_name
            })
        
        df = pd.DataFrame(rankings)
        return df.sort_values("window_score", ascending=False)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "current_model": self.current_model_name,
            "total_trades": self.total_trades,
            "model_switches": self.total_trades,
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "window_size": self.config.window_size,
                "switch_threshold": self.config.switch_threshold,
                "confidence_threshold": self.config.confidence_threshold,
                "confidence_strategy": self.config.confidence_strategy,
                "use_confidence_filter": self.config.use_confidence_filter
            }
        }
        
        # Add individual model summaries
        model_summaries = {}
        for model_name, perf in self.model_performances.items():
            window_metrics = perf.get_window_metrics(self.config.window_size)
            model_summaries[model_name] = {
                "current_score": perf.current_score,
                "best_score": perf.best_score,
                "window_score": window_metrics["score"],
                "window_sharpe": window_metrics["sharpe"],
                "window_pnl": window_metrics["pnl"],
                "window_drawdown": window_metrics["drawdown"],
                "trades_count": perf.trades_count
            }
        
        summary["models"] = model_summaries
        return summary
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking for all models."""
        for perf in self.model_performances.values():
            perf.sharpe_history.clear()
            perf.pnl_history.clear()
            perf.drawdown_history.clear()
            perf.score_history.clear()
            perf.current_score = 0.0
            perf.best_score = 0.0
            perf.trades_count = 0
        
        self.total_trades = 0
        self.last_switch_trade = 0
        logger.info("Performance tracking reset")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "ensemble_type": "weighted_dynamic",
            "num_models": len(self.models),
            "current_model": self.current_model_name,
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "window_size": self.config.window_size,
                "switch_threshold": self.config.switch_threshold
            },
            "ensemble_weights": self.ensemble_weights,
            "total_trades": self.total_trades
        })
        return metadata
