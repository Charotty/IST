"""Data layer exports with lazy optional components."""

from .base import BaseDataSource

__all__ = [
    "BaseDataSource",
    "CCXTDataSource",
    "CCXTMultiExchangeSource",
    "create_binance_source",
    "create_kraken_source",
    "create_coinbase_source",
    "create_bybit_source",
    "OKXDataSource",
    "OKXConfig",
    "create_okx_source",
    "BackfillManager",
    "BackfillConfig",
    "BackfillJob",
    "create_backfill_manager",
    "AutoDownloader",
    "AutoDownloadConfig",
    "DownloadTask",
    "create_auto_downloader",
    "GapDetector",
    "GapDetectionConfig",
    "GapStatistics",
    "create_gap_detector",
    "ReconnectionManager",
    "ReconnectionConfig",
    "ConnectionState",
    "create_reconnection_manager",
    "DataPipeline",
    "DataPipelineConfig",
    "PipelineStats",
    "create_data_pipeline",
]


def __getattr__(name: str):
    if name in {
        "CCXTDataSource",
        "CCXTMultiExchangeSource",
        "create_binance_source",
        "create_kraken_source",
        "create_coinbase_source",
        "create_bybit_source",
    }:
        from .ccxt_source import (
            CCXTDataSource,
            CCXTMultiExchangeSource,
            create_binance_source,
            create_kraken_source,
            create_coinbase_source,
            create_bybit_source,
        )

        return locals()[name]
    if name in {"OKXDataSource", "OKXConfig", "create_okx_source"}:
        from .okx_source import OKXDataSource, OKXConfig, create_okx_source

        return locals()[name]
    if name in {"BackfillManager", "BackfillConfig", "BackfillJob", "create_backfill_manager"}:
        from .backfill_manager import BackfillManager, BackfillConfig, BackfillJob, create_backfill_manager

        return locals()[name]
    if name in {"AutoDownloader", "AutoDownloadConfig", "DownloadTask", "create_auto_downloader"}:
        from .auto_downloader import AutoDownloader, AutoDownloadConfig, DownloadTask, create_auto_downloader

        return locals()[name]
    if name in {"GapDetector", "GapDetectionConfig", "GapStatistics", "create_gap_detector"}:
        from .gap_detector import GapDetector, GapDetectionConfig, GapStatistics, create_gap_detector

        return locals()[name]
    if name in {"ReconnectionManager", "ReconnectionConfig", "ConnectionState", "create_reconnection_manager"}:
        from .reconnection_manager import ReconnectionManager, ReconnectionConfig, ConnectionState, create_reconnection_manager

        return locals()[name]
    if name in {"DataPipeline", "DataPipelineConfig", "PipelineStats", "create_data_pipeline"}:
        from .data_pipeline import DataPipeline, DataPipelineConfig, PipelineStats, create_data_pipeline

        return locals()[name]
    raise AttributeError(name)
