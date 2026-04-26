"""
Evaluation module for trading models.

Provides comprehensive evaluation metrics and visualization:
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Regression metrics (R², MAE, RMSE, MAPE, Direction Accuracy)
- Trading metrics (Sharpe Ratio, Max Drawdown, Cumulative Return, Win Rate)
- Visualization (ROC curves, Confusion Matrix, Equity Curves)
"""

from .model_evaluator import ModelEvaluator

__all__ = ["ModelEvaluator"]
