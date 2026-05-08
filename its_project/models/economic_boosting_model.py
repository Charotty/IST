from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from its_project.models.base import BaseModel
except ImportError:
    from .base import BaseModel

logger = logging.getLogger(__name__)


class EconomicBoostingModel(BaseModel):
    """
    Gradient Boosting model with economic considerations.
    
    Features:
    - Class weight handling for imbalanced data
    - Custom loss function options
    - Early stopping
    - Economic evaluation metrics
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Model parameters
        self.n_estimators = config.get("n_estimators", 200)
        self.learning_rate = config.get("learning_rate", 0.05)
        self.max_depth = config.get("max_depth", 4)
        self.min_samples_split = config.get("min_samples_split", 10)
        self.min_samples_leaf = config.get("min_samples_leaf", 5)
        self.subsample = config.get("subsample", 0.8)
        self.max_features = config.get("max_features", "sqrt")
        self.random_state = config.get("random_state", 42)
        
        # Economic parameters
        self.use_class_weights = config.get("use_class_weights", True)
        self.class_weights = config.get("class_weights", None)
        self.early_stopping_rounds = config.get("early_stopping_rounds", 10)
        self.validation_split = config.get("validation_split", 0.2)
        
        # Loss function
        self.loss_function = config.get("loss_function", "log_loss")  # log_loss, exponential
        
        # Initialize model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state,
            loss=self.loss_function,
            validation_fraction=self.validation_split,
            n_iter_no_change=self.early_stopping_rounds,
            tol=1e-4
        )
        
        self._fitted = False
        self._feature_names = None
        self._class_weights = None
        self._training_history = []
    
    def _calculate_class_weights(self, y: np.ndarray) -> Optional[Dict[int, float]]:
        """Calculate class weights for imbalanced data."""
        if not self.use_class_weights:
            return None
        
        if self.class_weights:
            # Use provided weights
            return self.class_weights
        
        # Calculate inverse frequency weights
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Calculated class weights: {class_weights}")
        return class_weights
    
    def _create_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Create sample weights for training."""
        if self._class_weights is None:
            return np.ones(len(y))
        
        sample_weights = np.array([self._class_weights[cls] for cls in y])
        return sample_weights
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EconomicBoostingModel":
        """Fit the model with class weights and early stopping."""
        self.validate_input(X, y)
        
        # Ensure 2D input for sklearn
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Calculate class weights
        self._class_weights = self._calculate_class_weights(y)
        
        # Create sample weights
        sample_weights = self._create_sample_weights(y)
        
        # Fit model
        self.model.fit(X, y, sample_weight=sample_weights)
        self._fitted = True
        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        
        # Log training info
        n_estimators_used = self.model.n_estimators_
        logger.info(f"Model fitted with {n_estimators_used} estimators")
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-5:]
            logger.info(f"Top 5 feature indices: {top_features}")
        
        # Store class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        self._training_history.append({
            'class_distribution': class_distribution,
            'class_weights': self._class_weights,
            'n_estimators_used': n_estimators_used
        })
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict_proba(X)
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence (max probability)."""
        probabilities = self.predict_proba(X)
        return np.max(probabilities, axis=1)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def evaluate_economic_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate economic performance metrics."""
        if not self._fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Basic metrics
        accuracy = np.mean(y_pred == y)
        
        # Class-wise metrics
        unique_classes = np.unique(y)
        class_accuracies = {}
        for cls in unique_classes:
            mask = y == cls
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == cls)
                class_accuracies[f"accuracy_class_{cls}"] = class_acc
        
        # Confidence metrics
        confidences = self.get_confidence(X)
        avg_confidence = np.mean(confidences)
        
        # Economic simulation (simplified)
        # Assume BUY=2, SELL=0, HOLD=1
        buy_mask = (y_pred == 2)
        sell_mask = (y_pred == 0)
        
        # Simulate returns (would need actual price data in practice)
        buy_accuracy = np.mean(y[buy_mask] == 2) if np.sum(buy_mask) > 0 else 0
        sell_accuracy = np.mean(y[sell_mask] == 0) if np.sum(sell_mask) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'buy_signal_accuracy': buy_accuracy,
            'sell_signal_accuracy': sell_accuracy,
            'buy_signal_frequency': np.mean(buy_mask),
            'sell_signal_frequency': np.mean(sell_mask),
            **class_accuracies
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation with economic metrics."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        from sklearn.model_selection import cross_validate, StratifiedKFold
        
        # Custom scoring for economic metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }
        
        cv_results = cross_validate(
            self.model, X, y, 
            cv=StratifiedKFold(cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            return_estimator=True,
            n_jobs=-1
        )
        
        # Calculate economic metrics for each fold
        economic_metrics = []
        for i, estimator in enumerate(cv_results['estimator']):
            # Get test indices for this fold
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(cv, shuffle=True, random_state=self.random_state)
            _, test_idx = list(skf.split(X, y))[i]
            
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Create temporary model instance for evaluation
            temp_model = EconomicBoostingModel(self.config)
            temp_model.model = estimator
            temp_model._fitted = True
            
            fold_metrics = temp_model.evaluate_economic_metrics(X_test, y_test)
            economic_metrics.append(fold_metrics)
        
        # Aggregate economic metrics
        avg_economic_metrics = {}
        for key in economic_metrics[0].keys():
            values = [m[key] for m in economic_metrics]
            avg_economic_metrics[f"avg_{key}"] = np.mean(values)
            avg_economic_metrics[f"std_{key}"] = np.std(values)
        
        return {
            'cv_scores': {k: v.tolist() for k, v in cv_results.items()},
            'economic_metrics': avg_economic_metrics,
            'mean_accuracy': cv_results['test_accuracy'].mean(),
            'std_accuracy': cv_results['test_accuracy'].std()
        }
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and statistics."""
        if not self._training_history:
            return {}
        
        return {
            'training_history': self._training_history,
            'class_weights': self._class_weights,
            'feature_importance': self.get_feature_importance(),
            'model_params': self.get_params()
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'max_features': self.max_features,
            'loss_function': self.loss_function,
            'use_class_weights': self.use_class_weights,
            'class_weights': self._class_weights
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
