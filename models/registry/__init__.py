"""
Registry Module

Реестр моделей для управления версиями и метаданными.
"""

from .model_registry import ModelRegistry
from .version_manager import VersionManager

__all__ = ['ModelRegistry', 'VersionManager']
