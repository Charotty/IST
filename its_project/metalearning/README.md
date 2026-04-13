# Meta-Learning Layer

## Purpose
Automatically select the best model, optimize hyperparameters, perform cross-validation, and evaluate performance by metrics.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| ML metrics | `metrics.py` | Done | Accuracy, precision, recall, F1, ROC AUC, confusion matrix |
| Trading metrics | `metrics.py` | Done | Returns, Sharpe, max drawdown, win rate, profit factor |
| Time series CV | `cv.py` | Done | TimeSeriesSplitter with gap, WalkForwardValidator |
| Hyperopt | `hyperopt.py` | Done | Optuna-based HyperparameterOptimizer |
| Model selector | `selector.py` | Done | Evaluate all models, select best by weighted metrics |
| Stacking ensemble | `stacking.py` | Done | StackingEnsemble with base/meta models |
| Package init | `__init__.py` | Done | Exports all components |

## Data contracts

### ML metrics output
- Dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix
- All values are floats or list (confusion matrix)

### Trading metrics output
- Dict with keys: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, num_trades, avg_return, volatility
- All values are floats

### Model selector input
- X: 2D np.ndarray (n_samples, n_features)
- y: 1D np.ndarray (n_samples,)
- price_returns: 1D np.ndarray (n_samples,) for trading metrics

## Usage example

```python
from its_project.metalearning import (
    ModelSelector,
    HyperparameterOptimizer,
    TimeSeriesSplitter,
    MLMetrics,
    TradingMetrics,
)

# 1) Time series cross-validation
splitter = TimeSeriesSplitter(n_splits=5, gap=60)
for train_idx, test_idx in splitter.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train/evaluate models...

# 2) Hyperparameter optimization
param_space = {
    "hidden_size": {"type": "int", "low": 64, "high": 512},
    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
}
optimizer = HyperparameterOptimizer(
    model_class=LSTMModel,
    param_space=param_space,
    metric="sharpe_ratio",
    n_trials=100,
)
result = optimizer.optimize(X_train, y_train, X_val, y_val)

# 3) Model selection
models = [LSTMModel(cfg1), TransformerModel(cfg2), EnsembleModel(cfg3)]
selector = ModelSelector(
    models=models,
    metrics=["accuracy", "sharpe_ratio"],
    weights=[0.4, 0.6],
)
results_df = selector.evaluate_all(X_train, y_train, X_test, y_test, returns)
best_model, best_scores = selector.select_best()
ranking = selector.get_ranking()

# 4) Stacking ensemble
from its_project.models import EnsembleModel
base_models = [LSTMModel(cfg1), TransformerModel(cfg2)]
meta_model = EnsembleModel({})
stacking = StackingEnsemble(base_models, meta_model)
stacking.fit(X, y)
preds = stacking.predict(X)
```

## Integration in pipeline

- `metalearning_task` runs every 5 minutes, loads 24h of data, performs model selection with TimeSeriesSplitter, publishes best model info to `metalearning_results`.
- `predictor_task` listens to `metalearning_results` and dynamically updates the prediction model when a better one is available.
- All metrics are computed objectively; no manual model selection.

## Next improvements (without breaking layer)
- Add more sophisticated hyperparameter spaces (Bayesian optimization)
- Add online learning/updating of models
- Add feature importance analysis
- Add model interpretability tools
- Add ensemble of ensembles (blending)
