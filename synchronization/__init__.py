"""
Synchronization Layer

Синхронизация разнородных потоков данных от различных источников в единый временной ряд.
"""

from .sync_manager import SyncManager

__all__ = ['SyncManager']
