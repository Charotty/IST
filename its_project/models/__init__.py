from .base import BaseModel
from .lstm import LSTMModel
from .transformer import TransformerModel
from .ensemble import EnsembleModel
from .registry import ModelRegistry

# Register models
ModelRegistry.register("lstm")(LSTMModel)
ModelRegistry.register("transformer")(TransformerModel)
ModelRegistry.register("ensemble")(EnsembleModel)

__all__ = [
    "BaseModel",
    "LSTMModel",
    "TransformerModel",
    "EnsembleModel",
    "ModelRegistry",
]