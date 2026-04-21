from __future__ import annotations

from typing import Dict, Type, List
from pathlib import Path

try:
    from its_project.models.base import BaseModel
except ImportError:
    from .base import BaseModel


class ModelRegistry:
    """Registry for all available models."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model."""
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str, config: Dict) -> BaseModel:
        """Create model instance by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered")
        model_class = cls._models[name]
        return model_class(config)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(cls._models.keys())

    @classmethod
    def save_model(cls, model: BaseModel, path: Path) -> None:
        """Convenience wrapper for model.save."""
        model.save(path)

    @classmethod
    def load_model(cls, path: Path) -> BaseModel:
        """Convenience wrapper for BaseModel.load."""
        return BaseModel.load(path)
