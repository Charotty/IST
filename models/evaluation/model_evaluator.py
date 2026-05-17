"""
Model Evaluation Module

Evaluate profitability, robustness, and regime specialization.
Key metrics: Sharpe Ratio, Profit Factor, Max Drawdown, CAGR, Precision, Recall, Calibration Quality.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict, Any, List


class ModelEvaluator:
    """
    Model evaluator for financial and ML metrics.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics_history = []
    
    def calculate_ml_metrics(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """
        Calculate ML classification metrics.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param y_prob: Predicted probabilities (optional)
        :return: Dict with ML metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            # Add calibration quality metric if probabilities available
            metrics['avg_confidence'] = np.mean(y_prob)
        
        return metrics
    
    def calculate_financial_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate financial metrics (delegates to ``backtesting.PerformanceMetrics``).
        """
        from backtesting.performance_metrics import PerformanceMetrics

        cum = (1 + returns.fillna(0)).cumprod()
        peak = cum.expanding(min_periods=1).max()
        frame = pd.DataFrame(
            {
                "net_returns": returns.fillna(0),
                "cum_strategy_returns": cum,
                "drawdown": (cum - peak) / peak,
                "market_returns": returns.fillna(0) * 0,
                "cum_market_returns": pd.Series(1.0, index=returns.index),
            }
        )
        m = PerformanceMetrics(frame).calculate_metrics()
        return {
            "sharpe_ratio": float(m["Sharpe Ratio"]),
            "sortino_ratio": float(m["Sortino Ratio"]),
            "calmar_ratio": float(m["Calmar Ratio"]),
            "profit_factor": float(m["Profit Factor"]),
            "max_drawdown": float(m["Max Drawdown (%)"]) / 100.0,
            "total_return": float(m["Total Return (%)"]) / 100.0,
            "cagr": float(m["CAGR (%)"]) / 100.0,
            "recovery_factor": float(m["Recovery Factor"]),
        }
    
    def evaluate_regime_attribution(self, df: pd.DataFrame, regime_col: str, metric_col: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate which model performs best in which regime.
        
        :param df: DataFrame with regime and metric columns
        :param regime_col: Column name for regime
        :param metric_col: Column name for metric (e.g., returns)
        :return: Dict with regime-specific metrics
        """
        regime_metrics = {}
        
        for regime in df[regime_col].unique():
            regime_data = df[df[regime_col] == regime]
            regime_metrics[regime] = {
                'mean_return': regime_data[metric_col].mean(),
                'std_return': regime_data[metric_col].std(),
                'count': len(regime_data)
            }
        
        return regime_metrics
    
    def evaluate_stability(self, metrics_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Evaluate model stability over time.
        
        :param metrics_history: List of metric dicts over time
        :return: Dict with stability metrics
        """
        if not metrics_history:
            return {}
        
        # Calculate coefficient of variation for key metrics
        sharpe_values = [m.get('sharpe_ratio', 0) for m in metrics_history]
        pf_values = [m.get('profit_factor', 0) for m in metrics_history]
        
        stability_metrics = {
            'sharpe_cv': (np.std(sharpe_values) / np.mean(sharpe_values)) if np.mean(sharpe_values) != 0 else 0,
            'profit_factor_cv': (np.std(pf_values) / np.mean(pf_values)) if np.mean(pf_values) != 0 else 0,
            'num_periods': len(metrics_history)
        }
        
        return stability_metrics
    
    def comprehensive_evaluation(self, y_true, y_pred, y_prob, returns: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param y_prob: Predicted probabilities
        :param returns: Series of returns
        :return: Dict with all metrics
        """
        ml_metrics = self.calculate_ml_metrics(y_true, y_pred, y_prob)
        financial_metrics = self.calculate_financial_metrics(returns)
        
        return {
            'ml_metrics': ml_metrics,
            'financial_metrics': financial_metrics
        }
