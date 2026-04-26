from .base import BaseModel
from .registry import ModelRegistry

# Non-torch models (should always be importable)
from .ensemble import EnsembleModel
from .regression_model import RegressionModel
from .boosting_model import BoostingModel

# Register non-torch models
ModelRegistry.register("ensemble")(EnsembleModel)
ModelRegistry.register("regression")(RegressionModel)
ModelRegistry.register("boosting")(BoostingModel)

# Torch models are optional (may fail on Windows if torch DLLs are missing)
LSTMModel = None
TransformerModel = None
GRUModel = None
CNNLOBModel = None

try:
    from .lstm import LSTMModel as _LSTMModel
    LSTMModel = _LSTMModel
    ModelRegistry.register("lstm")(LSTMModel)
except Exception:
    pass

try:
    from .transformer import TransformerModel as _TransformerModel
    TransformerModel = _TransformerModel
    ModelRegistry.register("transformer")(TransformerModel)
except Exception:
    pass

try:
    from .gru_model import GRUModel as _GRUModel
    GRUModel = _GRUModel
    ModelRegistry.register("gru")(GRUModel)
except Exception:
    pass

try:
    from .cnn_lob_model import CNNLOBModel as _CNNLOBModel
    CNNLOBModel = _CNNLOBModel
    ModelRegistry.register("cnn_lob")(CNNLOBModel)
except Exception:
    pass

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "EnsembleModel",
    "RegressionModel",
    "BoostingModel",
]

if LSTMModel is not None:
    __all__.append("LSTMModel")
if TransformerModel is not None:
    __all__.append("TransformerModel")
if GRUModel is not None:
    __all__.append("GRUModel")
if CNNLOBModel is not None:
    __all__.append("CNNLOBModel")