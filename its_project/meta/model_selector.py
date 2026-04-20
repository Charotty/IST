from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from its_project.metalearning.metrics import MLMetrics, TradingMetrics

logger = logging.getLogger(__name__)


class ModelSelector:
    """Enhanced model selector with trading metrics focus."""
    
    def __init__(
        self,
        models: List[Any],
        metrics: List[str],
        weights: Optional[List[float]] = None,
        trading_returns: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.models = models
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.trading_returns = trading_returns
        self.config = config or {}
        
        # Validate weights
        if len(self.weights) != len(self.metrics):
            raise ValueError("Weights length must match metrics length")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.results_df = None
        self.best_model = None
        self.best_scores = None
    
    def evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_returns: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Evaluate all models with both ML and trading metrics."""
        results = []
        
        for i, model in enumerate(self.models):
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate ML metrics
                ml_metrics = MLMetrics.calculate(y_test, y_pred, y_proba)
                
                # Calculate trading metrics if returns are provided
                trading_metrics = {}
                if test_returns is not None or self.trading_returns is not None:
                    returns_to_use = test_returns if test_returns is not None else self.trading_returns
                    if len(returns_to_use) == len(y_pred):
                        trading_metrics = TradingMetrics.calculate(returns_to_use, y_pred, y_proba)
                
                # Combine results
                result = {
                    'model_index': i,
                    'model_name': model.__class__.__name__,
                    'model_type': type(model).__name__,
                    **ml_metrics
                }
                
                # Add trading metrics with prefix
                for key, value in trading_metrics.items():
                    result[f'trading_{key}'] = value
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating model {i}: {e}")
                # Add failed result
                result = {
                    'model_index': i,
                    'model_name': model.__class__.__name__,
                    'model_type': type(model).__name__,
                    'error': str(e)
                }
                results.append(result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def select_best(self, metric_focus: str = "trading_sharpe_ratio") -> Tuple[Any, Dict[str, float]]:
        """Select best model focusing on trading metrics."""
        if self.results_df is None:
            raise ValueError("Must call evaluate_all() first")
        
        # Filter out failed models
        valid_results = self.results_df[self.results_df['error'].isna()].copy()
        
        if valid_results.empty:
            raise ValueError("No models successfully evaluated")
        
        # Calculate composite score
        for metric, weight in zip(self.metrics, self.weights):
            if metric in valid_results.columns:
                # Normalize metric to 0-1 range (higher is better)
                metric_values = valid_results[metric]
                if metric_values.std() > 0:
                    normalized = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())
                else:
                    normalized = np.ones_like(metric_values)
                
                valid_results[f'normalized_{metric}'] = normalized * weight
        
        # Calculate composite score
        score_columns = [f'normalized_{m}' for m in self.metrics if f'normalized_{m}' in valid_results.columns]
        valid_results['composite_score'] = valid_results[score_columns].sum(axis=1)
        
        # Select best model
        best_idx = valid_results['composite_score'].idxmax()
        best_row = valid_results.loc[best_idx]
        
        self.best_model = self.models[int(best_row['model_index'])]
        self.best_scores = best_row.to_dict()
        
        logger.info(f"Best model: {best_row['model_name']} with score {best_row['composite_score']:.4f}")
        
        return self.best_model, self.best_scores
    
    def get_ranking(self) -> pd.DataFrame:
        """Get ranked model results."""
        if self.results_df is None:
            raise ValueError("Must call evaluate_all() first")
        
        # Filter out failed models
        valid_results = self.results_df[self.results_df['error'].isna()].copy()
        
        if valid_results.empty:
            return self.results_df
        
        # Calculate composite score if not already done
        if 'composite_score' not in valid_results.columns:
            self.select_best()
            valid_results = self.results_df[self.results_df['error'].isna()].copy()
        
        # Sort by composite score
        valid_results = valid_results.sort_values('composite_score', ascending=False)
        
        # Select columns to display
        display_columns = [
            'model_name', 'composite_score'
        ] + [m for m in self.metrics if m in valid_results.columns]
        
        # Add trading metrics
        trading_cols = [col for col in valid_results.columns if col.startswith('trading_')]
        display_columns.extend(trading_cols)
        
        return valid_results[display_columns].reset_index(drop=True)
    
    def compare_models(self, metric: str = "trading_sharpe_ratio") -> pd.DataFrame:
        """Compare models on a specific metric."""
        if self.results_df is None:
            raise ValueError("Must call evaluate_all() first")
        
        # Filter out failed models
        valid_results = self.results_df[self.results_df['error'].isna()].copy()
        
        if metric not in valid_results.columns:
            raise ValueError(f"Metric {metric} not found in results")
        
        # Create comparison table
        comparison = valid_results[['model_name', metric]].copy()
        comparison = comparison.sort_values(metric, ascending=False)
        
        # Add rank
        comparison['rank'] = range(1, len(comparison) + 1)
        
        return comparison
    
    def ensemble_selection(self, top_k: int = 3) -> List[Any]:
        """Select top-k models for ensemble."""
        if self.results_df is None:
            raise ValueError("Must call evaluate_all() first")
        
        ranking = self.get_ranking()
        top_indices = ranking.head(top_k)['model_index'].astype(int).tolist()
        
        return [self.models[i] for i in top_indices]
