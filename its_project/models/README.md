# Model Layer

## Purpose
Train ML models on features, predict signals (buy/sell/hold), estimate confidence, and serialize/deserialize models.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interface | `base.py` | Done | `BaseModel` (fit/predict/predict_proba/confidence/save/load) |
| LSTM model | `lstm.py` | Done | PyTorch LSTM with configurable layers/hidden size |
| Transformer model | `transformer.py` | Done | PyTorch Transformer with positional encoding |
| Ensemble model | `ensemble.py` | Done | sklearn VotingClassifier (LR + RF + GB) |
| Model registry | `registry.py` | Done | Register/create/list/load models |
| Package init | `__init__.py` | Done | Exports and registers models |

## Data contracts

### Input features
- Shape: `(n_samples, n_features)` for sklearn models
- Shape: `(n_samples, seq_len, n_features)` for deep models (LSTM/Transformer)
- Feature names list must match column order for validation

### Output predictions
- `predict`: class labels (0=sell, 1=hold, 2=buy)
- `predict_proba`: probabilities per class, shape `(n_samples, 3)`
- `get_confidence`: max probability per sample

## Usage example

```python
from its_project.models import ModelRegistry
import numpy as np

# 1) Create model via registry
model = ModelRegistry.get_model("lstm", {
    "input_size": 50,
    "hidden_size": 128,
    "num_layers": 2,
    "epochs": 100,
    "batch_size": 32,
})

# 2) Train (X shape: (n_samples, seq_len, n_features))
model.fit(X_train, y_train)

# 3) Predict
preds = model.predict(X_test)
proba = model.predict_proba(X_test)
conf = model.get_confidence(X_test)

# 4) Serialize
ModelRegistry.save_model(model, Path("models/lstm_v1.pkl"))
loaded = ModelRegistry.load_model(Path("models/lstm_v1.pkl"))
```

## Next improvements (without breaking layer)
- Add hyperparameter tuning integration
- Add model evaluation metrics (accuracy, F1, AUC)
- Add online learning/updating
- Add GPU memory optimizations
- Add model versioning and experiment tracking
