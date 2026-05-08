"""Storage layer exports.

Keep package import lightweight: database-specific modules are imported on
demand so a Parquet-only workflow does not fail when TimescaleDB dependencies
are not installed.
"""

from .base import BaseStorage
from .writer import batch_writer_task

__all__ = [
    "BaseStorage",
    "batch_writer_task",
    "ParquetStorage",
    "TimescaleStorage",
    "StorageReadAPI",
    "RawParquetStorage",
    "ParquetStorageConfig",
    "create_raw_parquet_storage",
    "AggregatedTimescaleStorage",
    "TimescaleDBConfig",
    "create_aggregated_timescale_storage",
    "StorageManager",
    "StorageManagerConfig",
    "StorageMode",
    "create_storage_manager",
]


def __getattr__(name: str):
    if name == "ParquetStorage":
        from .parquet import ParquetStorage

        return ParquetStorage
    if name == "TimescaleStorage":
        from .timescale import TimescaleStorage

        return TimescaleStorage
    if name == "StorageReadAPI":
        from .api import StorageReadAPI

        return StorageReadAPI
    if name in {"RawParquetStorage", "ParquetStorageConfig", "create_raw_parquet_storage"}:
        from .raw_parquet_storage import RawParquetStorage, ParquetStorageConfig, create_raw_parquet_storage

        return {
            "RawParquetStorage": RawParquetStorage,
            "ParquetStorageConfig": ParquetStorageConfig,
            "create_raw_parquet_storage": create_raw_parquet_storage,
        }[name]
    if name in {"AggregatedTimescaleStorage", "TimescaleDBConfig", "create_aggregated_timescale_storage"}:
        from .aggregated_timescale_storage import (
            AggregatedTimescaleStorage,
            TimescaleDBConfig,
            create_aggregated_timescale_storage,
        )

        return {
            "AggregatedTimescaleStorage": AggregatedTimescaleStorage,
            "TimescaleDBConfig": TimescaleDBConfig,
            "create_aggregated_timescale_storage": create_aggregated_timescale_storage,
        }[name]
    if name in {"StorageManager", "StorageManagerConfig", "StorageMode", "create_storage_manager"}:
        from .storage_manager import StorageManager, StorageManagerConfig, StorageMode, create_storage_manager

        return {
            "StorageManager": StorageManager,
            "StorageManagerConfig": StorageManagerConfig,
            "StorageMode": StorageMode,
            "create_storage_manager": create_storage_manager,
        }[name]
    raise AttributeError(name)
