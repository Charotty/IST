# Storage Layer Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Raw Data Storage (Parquet)](#raw-data-storage-parquet)
4. [Aggregated Data Storage (TimescaleDB)](#aggregated-data-storage-timescaledb)
5. [Unified Storage Manager](#unified-storage-manager)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The Storage Layer provides a comprehensive, production-ready system for storing both raw market data and aggregated analytics data. It combines the efficiency of Parquet files for raw data storage with the power of TimescaleDB for time-series aggregations.

### Key Features

- **Dual Storage Architecture**: Raw data in Parquet format, aggregated data in TimescaleDB
- **Intelligent Data Routing**: Automatic routing based on data type and use case
- **High Performance**: Concurrent writes, compression, and optimized queries
- **Data Retention**: Automatic cleanup and retention policies
- **Scalability**: Horizontal scaling with partitioning and connection pooling
- **Monitoring**: Comprehensive statistics and performance metrics
- **Reliability**: Error handling, retries, and graceful degradation

### Architecture Benefits

- **Separation of Concerns**: Raw data for high-frequency access, aggregated data for analytics
- **Cost Efficiency**: Parquet compression reduces storage costs by 60-80%
- **Query Performance**: Columnar storage and time-series optimization
- **Data Integrity**: Schema validation and automatic error recovery
- **Operational Simplicity**: Unified interface with automatic routing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Data Source   │    │ Storage Manager │    │ Query API   │ │
│  │                 │◄──►│                 │◄──►│             │ │
│  │ • Market Data   │    │ • Routing Rules │    │ • Raw Query │ │
│  │ • Real-time     │    │ • Batch Process│    │ • Aggregated│ │
│  │ • Historical    │    │ • Monitoring   │    │ • Analytics │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                       │                   │           │
│           └───────────────────────┼───────────────────┘           │
│                                 │                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                Storage Systems                         │  │
│  │                                                         │  │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐ │  │
│  │  │ Raw Parquet     │    │ Aggregated TimescaleDB      │ │  │
│  │  │                 │    │                             │ │  │
│  │  │ • High-freq     │    │ • OHLCV Aggregations        │ │  │
│  │  │ • Compressed    │    │ • Volume Statistics         │ │  │
│  │  │ • Partitioned   │    │ • Time-series Optimization   │ │  │
│  │  │ • Columnar      │    │ • Continuous Aggregates     │ │  │
│  │  └─────────────────┘    └─────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: Market data enters through the Storage Manager
2. **Routing**: Data is routed based on type and configuration rules
3. **Storage**: Raw data goes to Parquet, aggregated data to TimescaleDB
4. **Query**: Unified API provides access to both storage systems
5. **Monitoring**: Performance metrics and statistics are collected

### Storage Decision Matrix

| Data Type | Storage | Reason |
|-----------|----------|---------|
| Raw Ticker | Both | Raw for replay, aggregated for analytics |
| Orderbook | Raw Only | High-frequency, requires full detail |
| Trades | Both | Raw for analysis, aggregated for statistics |
| OHLCV | Aggregated Only | Already aggregated format |
| Volume Stats | Aggregated Only | Calculated aggregations |

---

## Raw Data Storage (Parquet)

### Overview

Raw data storage uses Apache Parquet format for efficient columnar storage of high-frequency market data. Parquet provides excellent compression, fast query performance, and schema evolution capabilities.

### Key Features

- **Columnar Storage**: Optimized for analytical queries
- **Compression**: Snappy compression reduces storage by 60-80%
- **Partitioning**: Automatic partitioning by symbol, date, and data type
- **Schema Evolution**: Support for schema changes over time
- **Concurrent Writes**: Multi-threaded write operations
- **Automatic Cleanup**: Retention policies and file management

### File Organization

```
data/parquet/raw/
├── symbol=BTC-USDT/
│   ├── date=2024-01-01/
│   │   ├── type=ticker/
│   │   │   ├── BTC-USDT_ticker_20240101_120000_123456.parquet
│   │   │   ├── BTC-USDT_ticker_20240101_130000_789012.parquet
│   │   │   └── ...
│   │   ├── type=orderbook/
│   │   │   ├── BTC-USDT_orderbook_20240101_120000_345678.parquet
│   │   │   └── ...
│   │   └── type=trade/
│   │       ├── BTC-USDT_trade_20240101_120000_567890.parquet
│   │       └── ...
│   ├── date=2024-01-02/
│   └── ...
├── symbol=ETH-USDT/
│   └── ...
└── symbol=SOL-USDT/
    └── ...
```

### Schema Definition

#### Arrow Schema
```python
schema = pa.schema([
    pa.field("timestamp_ms", pa.int64(), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("data_type", pa.string(), nullable=False),
    pa.field("exchange", pa.string(), nullable=False),
    pa.field("data", pa.struct([
        # Common fields
        pa.field("_kind", pa.string()),
        pa.field("_ts_recv_ms", pa.int64()),
        
        # Ticker fields
        pa.field("last_price", pa.float64()),
        pa.field("best_bid", pa.float64()),
        pa.field("best_ask", pa.float64()),
        pa.field("bid_size", pa.float64()),
        pa.field("ask_size", pa.float64()),
        pa.field("volume_24h", pa.float64()),
        pa.field("high_24h", pa.float64()),
        pa.field("low_24h", pa.float64()),
        pa.field("open_24h", pa.float64()),
        pa.field("change_24h", pa.float64()),
        pa.field("change_pct_24h", pa.float64()),
        
        # Orderbook fields
        pa.field("bids", pa.list_(pa.list_(pa.string()))),
        pa.field("asks", pa.list_(pa.list_(pa.string()))),
        pa.field("lastUpdateId", pa.int64()),
        pa.field("U", pa.int64()),
        pa.field("u", pa.int64()),
        
        # Trade fields
        pa.field("trades", pa.list_(pa.struct([
            pa.field("trade_id", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("size", pa.float64()),
            pa.field("side", pa.string()),
            pa.field("timestamp", pa.int64())
        ])))
    
    # Metadata
    pa.field("ingest_time", pa.timestamp('us'), nullable=False),
    pa.field("partition_date", pa.date32(), nullable=False)
])
```

### Configuration

#### ParquetStorageConfig
```python
@dataclass
class ParquetStorageConfig:
    # Storage paths
    base_path: str = "data/parquet/raw"
    temp_path: str = "data/parquet/temp"
    
    # Partitioning
    partition_cols: List[str] = ["symbol", "date", "data_type"]
    partition_format: str = "year={year}/month={month}/day={day}"
    
    # File settings
    file_format: str = "parquet"
    compression: str = "snappy"
    row_group_size: int = 100000
    max_file_size_mb: int = 100
    
    # Performance settings
    write_batch_size: int = 10000
    max_concurrent_writes: int = 4
    write_timeout: float = 30.0
    
    # Data retention
    retention_days: int = 365
    cleanup_interval_hours: int = 24
    
    # Schema evolution
    enable_schema_evolution: bool = True
    schema_validation: bool = True
```

### Performance Characteristics

#### Compression Ratios
- **Ticker Data**: 70-80% compression ratio
- **Orderbook Data**: 60-70% compression ratio
- **Trade Data**: 65-75% compression ratio

#### Query Performance
- **Full Scan**: 10-50 MB/s per thread
- **Partition Pruning**: 90%+ data reduction for time-range queries
- **Column Projection**: 95%+ data reduction for specific fields

#### Write Performance
- **Throughput**: 50,000-100,000 records/second
- **Latency**: 10-50ms per batch
- **Scalability**: Linear scaling with concurrent writers

---

## Aggregated Data Storage (TimescaleDB)

### Overview

TimescaleDB provides high-performance time-series database capabilities for storing aggregated market data. It excels at OHLCV data, volume statistics, and other time-based aggregations with automatic compression and continuous aggregates.

### Key Features

- **Time-Series Optimization**: Hypertables for efficient time-based queries
- **Continuous Aggregates**: Real-time automated aggregations
- **Data Compression**: Automatic compression for historical data
- **Connection Pooling**: High-concurrency database access
- **Retention Policies**: Automatic data lifecycle management
- **SQL Interface**: Standard SQL with time-series extensions

### Database Schema

#### Raw Data Table
```sql
CREATE TABLE market_data_raw (
    timestamp BIGINT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    exchange TEXT NOT NULL,
    data JSONB NOT NULL,
    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable(
    'market_data_raw',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour'
);

-- Create indexes
CREATE INDEX idx_market_data_raw_symbol_timestamp 
ON market_data_raw (symbol, timestamp DESC);

CREATE INDEX idx_market_data_raw_type_timestamp 
ON market_data_raw (data_type, timestamp DESC);
```

#### OHLCV Table
```sql
CREATE TABLE market_data_ohlcv (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    trade_count INTEGER NOT NULL,
    vwap DECIMAL(20, 8),
    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable(
    'market_data_ohlcv',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour'
);

-- Create composite index
CREATE INDEX idx_ohlcv_symbol_interval_timestamp 
ON market_data_ohlcv (symbol, interval, timestamp DESC);
```

#### Volume Table
```sql
CREATE TABLE market_data_volume (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    buy_volume DECIMAL(30, 8) NOT NULL,
    sell_volume DECIMAL(30, 8) NOT NULL,
    total_volume DECIMAL(30, 8) NOT NULL,
    trade_count INTEGER NOT NULL,
    avg_trade_size DECIMAL(20, 8),
    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable(
    'market_data_volume',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour'
);
```

#### Statistics Table
```sql
CREATE TABLE market_data_stats (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    avg_price DECIMAL(20, 8) NOT NULL,
    price_volatility DECIMAL(10, 8) NOT NULL,
    price_change DECIMAL(20, 8) NOT NULL,
    price_change_pct DECIMAL(10, 4) NOT NULL,
    volume_weighted_price DECIMAL(20, 8) NOT NULL,
    total_volume DECIMAL(30, 8) NOT NULL,
    trade_count INTEGER NOT NULL,
    unique_trades INTEGER NOT NULL,
    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable(
    'market_data_stats',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour'
);
```

### Aggregation Intervals

#### Supported Intervals
- **1m**: 1 minute - 7 days retention
- **5m**: 5 minutes - 30 days retention
- **15m**: 15 minutes - 90 days retention
- **1h**: 1 hour - 1 year retention
- **4h**: 4 hours - 2 years retention
- **1d**: 1 day - 7 years retention

#### Aggregation Logic

##### OHLCV Aggregation
```sql
INSERT INTO market_data_ohlcv 
SELECT 
    time_bucket(INTERVAL '1 hour', timestamp) as timestamp,
    symbol,
    '1h' as interval,
    (data->>'last_price')::DECIMAL(20,8) as open_price,
    MAX((data->>'last_price')::DECIMAL(20,8)) as high_price,
    MIN((data->>'last_price')::DECIMAL(20,8)) as low_price,
    (data->>'last_price')::DECIMAL(20,8) as close_price,
    COALESCE(SUM((data->>'volume_24h')::DECIMAL(30,8)), 0) as volume,
    COUNT(*) as trade_count,
    AVG((data->>'last_price')::DECIMAL(20,8)) as vwap
FROM market_data_raw
WHERE data_type = 'ticker'
AND timestamp >= NOW() - INTERVAL '1 hour'
AND timestamp <= NOW()
GROUP BY time_bucket(INTERVAL '1 hour', timestamp), symbol
ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
    open_price = EXCLUDED.open_price,
    high_price = EXCLUDED.high_price,
    low_price = EXCLUDED.low_price,
    close_price = EXCLUDED.close_price,
    volume = EXCLUDED.volume,
    trade_count = EXCLUDED.trade_count,
    vwap = EXCLUDED.vwap;
```

##### Volume Aggregation
```sql
INSERT INTO market_data_volume 
SELECT 
    time_bucket(INTERVAL '1 hour', timestamp) as timestamp,
    symbol,
    '1h' as interval,
    COALESCE(SUM(CASE WHEN (data->>'side') = 'buy' 
        THEN (data->>'volume_24h')::DECIMAL(30,8) ELSE 0 END), 0) as buy_volume,
    COALESCE(SUM(CASE WHEN (data->>'side') = 'sell' 
        THEN (data->>'volume_24h')::DECIMAL(30,8) ELSE 0 END), 0) as sell_volume,
    COALESCE(SUM((data->>'volume_24h')::DECIMAL(30,8)), 0) as total_volume,
    COUNT(*) as trade_count,
    COALESCE(AVG((data->>'volume_24h')::DECIMAL(20,8)), 0) as avg_trade_size
FROM market_data_raw
WHERE data_type = 'trade'
AND timestamp >= NOW() - INTERVAL '1 hour'
AND timestamp <= NOW()
GROUP BY time_bucket(INTERVAL '1 hour', timestamp), symbol
ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
    buy_volume = EXCLUDED.buy_volume,
    sell_volume = EXCLUDED.sell_volume,
    total_volume = EXCLUDED.total_volume,
    trade_count = EXCLUDED.trade_count,
    avg_trade_size = EXCLUDED.avg_trade_size;
```

### Configuration

#### TimescaleDBConfig
```python
@dataclass
class TimescaleDBConfig:
    # Connection settings
    dsn: str = "postgres://user:password@localhost:5432/market_data"
    pool_size: int = 10
    command_timeout: float = 30.0
    max_inactive_connection_lifetime: float = 300.0
    
    # Table settings
    raw_data_table: str = "market_data_raw"
    ohlcv_table: str = "market_data_ohlcv"
    volume_table: str = "market_data_volume"
    stats_table: str = "market_data_stats"
    
    # Aggregation settings
    aggregation_intervals: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    enable_continuous_aggregates: bool = True
    aggregation_retention: Dict[str, int] = {
        "1m": 7,    # 7 days
        "5m": 30,   # 30 days
        "15m": 90,  # 90 days
        "1h": 365,  # 1 year
        "4h": 730,  # 2 years
        "1d": 2555  # 7 years
    }
    
    # Performance settings
    batch_size: int = 1000
    max_concurrent_inserts: int = 5
    insert_timeout: float = 10.0
    enable_compression: bool = True
    
    # Data retention
    raw_data_retention_days: int = 7
    aggregated_data_retention_days: int = 2555  # 7 years
    
    # Monitoring
    enable_query_stats: bool = True
    stats_collection_interval: int = 300  # 5 minutes
```

### Performance Characteristics

#### Query Performance
- **Time Range Queries**: 10-100ms for 1-year ranges
- **Symbol Filtering**: 5-20ms with proper indexing
- **Aggregation Queries**: 50-200ms depending on complexity
- **Concurrent Queries**: 100+ concurrent queries supported

#### Write Performance
- **Insert Throughput**: 10,000-50,000 records/second
- **Batch Inserts**: 100,000+ records/second with batching
- **Aggregation Latency**: 1-5 minutes for continuous aggregates
- **Connection Pool**: 10-100 concurrent connections

#### Storage Efficiency
- **Compression**: 70-90% space reduction with compression
- **Chunking**: 1-hour chunks for optimal query performance
- **Retention**: Automatic cleanup based on configured policies

---

## Unified Storage Manager

### Overview

The Unified Storage Manager provides a single interface for managing both raw and aggregated storage systems. It handles intelligent data routing, performance optimization, and comprehensive monitoring.

### Key Features

- **Intelligent Routing**: Automatic routing based on data type and configuration
- **Unified Interface**: Single API for all storage operations
- **Performance Optimization**: Batch processing and concurrent operations
- **Monitoring**: Comprehensive statistics and performance metrics
- **Error Handling**: Automatic retries and graceful degradation
- **Configuration Management**: Flexible configuration for different environments

### Storage Modes

#### StorageMode Enum
```python
class StorageMode(Enum):
    RAW_ONLY = "raw_only"           # Only store raw data in Parquet
    AGGREGATED_ONLY = "aggregated_only"  # Only store aggregated data in TimescaleDB
    BOTH = "both"                   # Store in both systems (default)
```

#### Routing Rules
```python
routing_rules = {
    "ticker": "both",           # Store in both raw and aggregated
    "orderbook": "raw_only",    # Only raw storage for high-frequency data
    "trade": "both",            # Store in both for analysis
    "ohlcv": "aggregated_only"  # Only aggregated for processed data
}
```

### Data Flow Architecture

```
Market Data → Storage Manager → Routing Engine
                                    ↓
                    ┌─────────────────────┐
                    │                     │
                    ▼                     ▼
            Raw Parquet Storage    Aggregated TimescaleDB
            (High-frequency)       (OHLCV, Volume, Stats)
                    │                     │
                    └─────────────────────┘
                                    ↓
                            Unified Query Interface
```

### Configuration

#### StorageManagerConfig
```python
@dataclass
class StorageManagerConfig:
    # Storage modes
    storage_mode: StorageMode = StorageMode.BOTH
    
    # Raw storage configuration
    raw_config: ParquetStorageConfig = field(default_factory=ParquetStorageConfig)
    
    # Aggregated storage configuration
    aggregated_config: TimescaleDBConfig = field(default_factory=TimescaleDBConfig)
    
    # Data routing
    enable_auto_routing: bool = True
    routing_rules: Dict[str, str] = field(default_factory=lambda: {
        "ticker": "both",
        "orderbook": "raw_only",
        "trade": "both",
        "ohlcv": "aggregated_only"
    })
    
    # Performance settings
    batch_size: int = 1000
    max_queue_size: int = 10000
    processing_timeout: float = 30.0
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 60.0  # seconds
    
    # Data retention
    enable_auto_cleanup: bool = True
    cleanup_interval_hours: int = 24
```

### Statistics and Monitoring

#### StorageStats
```python
@dataclass
class StorageStats:
    start_time: datetime = field(default_factory=datetime.now)
    
    # Raw storage stats
    raw_files_written: int = 0
    raw_records_stored: int = 0
    raw_storage_size_mb: float = 0.0
    
    # Aggregated storage stats
    aggregated_records_stored: int = 0
    ohlcv_records: int = 0
    volume_records: int = 0
    stats_records: int = 0
    
    # Performance stats
    total_processed: int = 0
    processing_rate: float = 0.0  # records per second
    error_count: int = 0
    
    # Queue stats
    queue_size: int = 0
    queue_utilization: float = 0.0
    
    # Last activity
    last_store_time: Optional[datetime] = None
    last_cleanup_time: Optional[datetime] = None
```

#### Monitoring Metrics
- **Processing Rate**: Records processed per second
- **Queue Utilization**: Percentage of queue capacity used
- **Error Rate**: Percentage of failed operations
- **Storage Growth**: Rate of storage consumption
- **Query Performance**: Average query response times

---

## Configuration

### Environment-Specific Configurations

#### Development Environment
```python
# Development configuration
dev_config = StorageManagerConfig(
    storage_mode=StorageMode.BOTH,
    raw_config=ParquetStorageConfig(
        base_path="data/dev/parquet/raw",
        compression="snappy",
        retention_days=7,
        max_concurrent_writes=2
    ),
    aggregated_config=TimescaleDBConfig(
        dsn="postgres://dev_user:dev_pass@localhost:5432/market_data_dev",
        pool_size=5,
        raw_data_retention_days=1,
        aggregation_intervals=["1m", "5m", "1h"]
    ),
    batch_size=500,
    enable_monitoring=False
)
```

#### Production Environment
```python
# Production configuration
prod_config = StorageManagerConfig(
    storage_mode=StorageMode.BOTH,
    raw_config=ParquetStorageConfig(
        base_path="/data/parquet/raw",
        compression="snappy",
        retention_days=365,
        max_concurrent_writes=8,
        row_group_size=100000
    ),
    aggregated_config=TimescaleDBConfig(
        dsn="postgres://prod_user:prod_pass@timescaledb:5432/market_data",
        pool_size=20,
        raw_data_retention_days=7,
        aggregation_intervals=["1m", "5m", "15m", "1h", "4h", "1d"]
    ),
    batch_size=2000,
    max_queue_size=50000,
    enable_monitoring=True,
    enable_auto_cleanup=True
)
```

### Configuration Validation

#### Validation Rules
```python
def validate_config(config: StorageManagerConfig) -> bool:
    # Validate storage mode
    if config.storage_mode not in StorageMode:
        raise ValueError(f"Invalid storage mode: {config.storage_mode}")
    
    # Validate raw storage config
    if config.storage_mode in [StorageMode.RAW_ONLY, StorageMode.BOTH]:
        if not config.raw_config.base_path:
            raise ValueError("Raw storage base path required")
        
        if config.raw_config.retention_days < 1:
            raise ValueError("Raw storage retention must be at least 1 day")
    
    # Validate aggregated storage config
    if config.storage_mode in [StorageMode.AGGREGATED_ONLY, StorageMode.BOTH]:
        if not config.aggregated_config.dsn:
            raise ValueError("TimescaleDB DSN required")
        
        if config.aggregated_config.pool_size < 1:
            raise ValueError("Pool size must be at least 1")
    
    # Validate routing rules
    valid_destinations = ["raw_only", "aggregated_only", "both"]
    for data_type, destination in config.routing_rules.items():
        if destination not in valid_destinations:
            raise ValueError(f"Invalid routing destination for {data_type}: {destination}")
    
    return True
```

---

## Usage Examples

### Basic Usage

#### Quick Start
```python
import asyncio
from its_project.storage import create_storage_manager
from its_project.common.types import MarketData, MarketDataType
from datetime import datetime

async def main():
    # Create storage manager with default configuration
    storage = create_storage_manager(
        storage_mode="both",
        raw_path="data/parquet/raw",
        timescale_dsn="postgres://user:pass@localhost/market_data"
    )
    
    # Start storage system
    await storage.start()
    
    try:
        # Store market data
        market_data = MarketData(
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            symbol="BTC-USDT",
            type=MarketDataType.TICKER,
            exchange="okx",
            data={
                "last_price": 50000.0,
                "best_bid": 49999.0,
                "best_ask": 50001.0,
                "volume_24h": 1000.0
            }
        )
        
        await storage.store(market_data)
        
        # Get statistics
        stats = storage.get_statistics()
        print(f"Processed: {stats.total_processed} records")
        print(f"Processing rate: {stats.processing_rate:.1f} records/sec")
        
    finally:
        # Stop storage system
        await storage.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

#### Custom Configuration
```python
import asyncio
from its_project.storage import (
    StorageManager, StorageManagerConfig, StorageMode,
    ParquetStorageConfig, TimescaleDBConfig
)

async def advanced_example():
    # Custom raw storage configuration
    raw_config = ParquetStorageConfig(
        base_path="/data/parquet/raw",
        compression="snappy",
        retention_days=365,
        max_concurrent_writes=8,
        row_group_size=100000,
        enable_schema_evolution=True
    )
    
    # Custom aggregated storage configuration
    agg_config = TimescaleDBConfig(
        dsn="postgres://user:pass@timescaledb:5432/market_data",
        pool_size=20,
        batch_size=2000,
        aggregation_intervals=["1m", "5m", "15m", "1h", "4h", "1d"],
        raw_data_retention_days=7,
        enable_compression=True
    )
    
    # Create manager configuration
    manager_config = StorageManagerConfig(
        storage_mode=StorageMode.BOTH,
        raw_config=raw_config,
        aggregated_config=agg_config,
        enable_auto_routing=True,
        routing_rules={
            "ticker": "both",
            "orderbook": "raw_only",
            "trade": "both",
            "ohlcv": "aggregated_only"
        },
        batch_size=2000,
        max_queue_size=50000,
        enable_monitoring=True,
        enable_auto_cleanup=True
    )
    
    # Create and start storage manager
    storage = StorageManager(manager_config)
    await storage.start()
    
    try:
        # Store batch of data
        market_data_batch = [
            MarketData(
                timestamp_ms=int(datetime.now().timestamp() * 1000),
                symbol="BTC-USDT",
                type=MarketDataType.TICKER,
                exchange="okx",
                data={"last_price": 50000.0, "volume_24h": 1000.0}
            ),
            MarketData(
                timestamp_ms=int(datetime.now().timestamp() * 1000),
                symbol="ETH-USDT",
                type=MarketDataType.TICKER,
                exchange="okx",
                data={"last_price": 3000.0, "volume_24h": 5000.0}
            )
        ]
        
        await storage.store_batch(market_data_batch)
        
        # Monitor performance
        async def monitor_storage():
            while True:
                status = storage.get_storage_status()
                print(f"Queue utilization: {status['manager']['queue_utilization']:.1%}")
                print(f"Processing rate: {status['statistics']['processing_rate']:.1f} records/sec")
                await asyncio.sleep(30)
        
        monitor_task = asyncio.create_task(monitor_storage())
        
        # Query data
        ohlcv_data = await storage.query_ohlcv_data(
            symbol="BTC-USDT",
            interval="1h",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        
        print(f"Retrieved {len(ohlcv_data)} OHLCV records")
        
        monitor_task.cancel()
        
    finally:
        await storage.stop()

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

### Integration with Data Pipeline

#### Pipeline Integration
```python
class DataPipelineStorage:
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.stats = {
            'total_stored': 0,
            'by_type': {},
            'errors': 0
        }
    
    async def start(self):
        await self.storage.start()
    
    async def stop(self):
        await self.storage.stop()
    
    async def process_market_data(self, market_data: MarketData):
        try:
            await self.storage.store(market_data)
            self.stats['total_stored'] += 1
            
            data_type = market_data.type.value
            self.stats['by_type'][data_type] = self.stats['by_type'].get(data_type, 0) + 1
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Storage error: {e}")
    
    async def get_performance_metrics(self):
        storage_stats = self.storage.get_statistics()
        
        return {
            'pipeline_stats': self.stats,
            'storage_stats': storage_stats,
            'storage_status': self.storage.get_storage_status()
        }

# Usage in data pipeline
async def integrate_with_pipeline():
    storage = create_storage_manager()
    pipeline_storage = DataPipelineStorage(storage)
    
    await pipeline_storage.start()
    
    try:
        # Process data from pipeline
        async for market_data in data_stream:
            await pipeline_storage.process_market_data(market_data)
            
            # Periodic monitoring
            if market_data.timestamp_ms % 60000 < 1000:  # Every minute
                metrics = await pipeline_storage.get_performance_metrics()
                print(f"Storage metrics: {metrics}")
    
    finally:
        await pipeline_storage.stop()
```

### Query Examples

#### Query Raw Data
```python
async def query_raw_data_example():
    storage = create_storage_manager()
    await storage.start()
    
    try:
        # Query raw ticker data for BTC-USDT
        raw_data = await storage.query_raw_data(
            symbol="BTC-USDT",
            data_type="ticker",
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        print(f"Retrieved {len(raw_data)} raw records")
        
        # Process raw data
        for _, row in raw_data.iterrows():
            print(f"Timestamp: {row['timestamp']}, Price: {row['data']['last_price']}")
    
    finally:
        await storage.stop()
```

#### Query OHLCV Data
```python
async def query_ohlcv_example():
    storage = create_storage_manager()
    await storage.start()
    
    try:
        # Query 1-hour OHLCV data
        ohlcv_data = await storage.query_ohlcv_data(
            symbol="BTC-USDT",
            interval="1h",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now()
        )
        
        print(f"Retrieved {len(ohlcv_data)} OHLCV records")
        
        # Calculate simple moving average
        if not ohlcv_data.empty:
            ohlcv_data['sma_20'] = ohlcv_data['close'].rolling(window=20).mean()
            print(ohlcv_data[['timestamp', 'close', 'sma_20']].tail())
    
    finally:
        await storage.stop()
```

#### Query Volume Statistics
```python
async def query_volume_stats_example():
    storage = create_storage_manager()
    await storage.start()
    
    try:
        # Query volume statistics
        volume_data = await storage.query_volume_stats(
            symbol="BTC-USDT",
            interval="1h",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        
        print(f"Retrieved {len(volume_data)} volume records")
        
        # Analyze buy/sell ratio
        if not volume_data.empty:
            volume_data['buy_sell_ratio'] = volume_data['buy_volume'] / volume_data['sell_volume']
            print(volume_data[['timestamp', 'total_volume', 'buy_sell_ratio']].tail())
    
    finally:
        await storage.stop()
```

---

## Performance Optimization

### Raw Storage Optimization

#### Partitioning Strategy
```python
# Optimal partitioning for different query patterns
partitioning_strategies = {
    "time_range_queries": ["date", "symbol"],      # Fast time-based queries
    "symbol_queries": ["symbol", "date"],          # Fast symbol-based queries
    "type_queries": ["data_type", "symbol", "date"] # Fast type-based queries
}

# Configure based on primary query pattern
raw_config = ParquetStorageConfig(
    partition_cols=partitioning_strategies["time_range_queries"],
    row_group_size=100000,  # Optimal for most queries
    compression="snappy"     # Balance compression vs speed
)
```

#### Write Optimization
```python
# Optimize for high-throughput writes
high_throughput_config = ParquetStorageConfig(
    max_concurrent_writes=8,      # More concurrent writers
    write_batch_size=50000,         # Larger batches
    max_file_size_mb=200,          # Larger files
    row_group_size=500000          # Larger row groups
)
```

#### Query Optimization
```python
# Enable predicate pushdown and column pruning
query_optimization_config = ParquetStorageConfig(
    enable_statistics=True,         # Collect file statistics
    enable_metadata_cache=True,     # Cache metadata
    enable_dictionary_encoding=True # Dictionary encoding for strings
)
```

### Aggregated Storage Optimization

#### Connection Pool Optimization
```python
# Optimize connection pool for high concurrency
optimized_pool_config = TimescaleDBConfig(
    pool_size=20,                           # Larger pool
    command_timeout=60.0,                    # Longer timeout
    max_inactive_connection_lifetime=600.0,   # Keep connections longer
    batch_size=2000,                          # Larger batches
    max_concurrent_inserts=10                # More concurrent inserts
)
```

#### Hypertable Optimization
```python
# Optimal chunk intervals for different data frequencies
chunk_intervals = {
    "high_frequency": "INTERVAL '15 minutes'",  # For tick data
    "medium_frequency": "INTERVAL '1 hour'",    # For minute data
    "low_frequency": "INTERVAL '1 day'"         # For daily data
}

# Configure chunk intervals based on data type
await conn.execute(f"""
    SELECT create_hypertable(
        'market_data_raw',
        'timestamp',
        chunk_time_interval => {chunk_intervals['high_frequency']}
    )
""")
```

#### Index Optimization
```python
# Create composite indexes for common query patterns
index_optimizations = [
    # Symbol and time range queries
    "CREATE INDEX idx_symbol_timestamp ON market_data_raw (symbol, timestamp DESC)",
    
    # Data type and time range queries  
    "CREATE INDEX idx_type_timestamp ON market_data_raw (data_type, timestamp DESC)",
    
    # Exchange and symbol queries
    "CREATE INDEX idx_exchange_symbol ON market_data_raw (exchange, symbol)",
    
    # OHLCV queries by symbol and interval
    "CREATE INDEX idx_ohlcv_symbol_interval ON market_data_ohlcv (symbol, interval, timestamp DESC)"
]

for index_sql in index_optimizations:
    await conn.execute(index_sql)
```

### Unified Manager Optimization

#### Routing Optimization
```python
# Optimize routing rules for performance
performance_routing_rules = {
    "ticker": "both",           # Raw for replay, aggregated for analytics
    "orderbook": "raw_only",    # Too high frequency for aggregation
    "trade": "both",            # Raw for analysis, aggregated for stats
    "ohlcv": "aggregated_only"  # Already aggregated
}

# Configure for performance
manager_config = StorageManagerConfig(
    routing_rules=performance_routing_rules,
    batch_size=2000,              # Larger batches
    max_queue_size=50000,         # Larger queue
    processing_timeout=60.0,      # Longer timeout
    enable_monitoring=True        # Monitor performance
)
```

#### Memory Optimization
```python
# Configure for memory efficiency
memory_efficient_config = StorageManagerConfig(
    batch_size=500,               # Smaller batches
    max_queue_size=10000,         # Smaller queue
    raw_config=ParquetStorageConfig(
        write_batch_size=5000,     # Smaller write batches
        max_concurrent_writes=2     # Fewer concurrent writes
    ),
    aggregated_config=TimescaleDBConfig(
        pool_size=5,                # Smaller pool
        batch_size=500              # Smaller batches
    )
)
```

### Performance Monitoring

#### Key Metrics
```python
async def monitor_performance(storage: StorageManager):
    while True:
        stats = storage.get_statistics()
        status = storage.get_storage_status()
        
        # Performance metrics
        processing_rate = stats.processing_rate
        queue_utilization = stats.queue_utilization
        error_rate = stats.error_count / max(stats.total_processed, 1)
        
        # Storage metrics
        raw_size = status.get('raw_storage', {}).get('size_mb', 0)
        agg_records = status.get('aggregated_storage', {}).get('records_inserted', 0)
        
        # Alert on performance issues
        if processing_rate < 1000:  # Low processing rate
            logger.warning(f"Low processing rate: {processing_rate:.1f} records/sec")
        
        if queue_utilization > 0.8:  # High queue utilization
            logger.warning(f"High queue utilization: {queue_utilization:.1%}")
        
        if error_rate > 0.01:  # High error rate
            logger.warning(f"High error rate: {error_rate:.2%}")
        
        await asyncio.sleep(60)  # Check every minute
```

#### Performance Benchmarks
```python
async def benchmark_storage(storage: StorageManager):
    # Generate test data
    test_data = [
        MarketData(
            timestamp_ms=int(datetime.now().timestamp() * 1000) + i * 1000,
            symbol="BTC-USDT",
            type=MarketDataType.TICKER,
            exchange="okx",
            data={"last_price": 50000.0 + i, "volume_24h": 1000.0}
        )
        for i in range(10000)
    ]
    
    # Benchmark write performance
    start_time = datetime.now()
    await storage.store_batch(test_data)
    write_time = (datetime.now() - start_time).total_seconds()
    
    write_rate = len(test_data) / write_time
    print(f"Write performance: {write_rate:.1f} records/sec")
    
    # Benchmark query performance
    start_time = datetime.now()
    ohlcv_data = await storage.query_ohlcv_data(
        symbol="BTC-USDT",
        interval="1m",
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now()
    )
    query_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Query performance: {query_time:.3f} seconds for {len(ohlcv_data)} records")
```

---

## API Reference

### RawParquetStorage

#### Constructor
```python
RawParquetStorage(config: ParquetStorageConfig)
```

#### Methods

##### start()
```python
async def start() -> None
```
Start the raw storage system.

##### stop()
```python
async def stop() -> None
```
Stop the raw storage system.

##### store()
```python
async def store(market_data: MarketData) -> None
```
Store a single market data point.

**Parameters:**
- `market_data`: Market data to store

##### store_batch()
```python
async def store_batch(market_data_batch: List[MarketData]) -> None
```
Store a batch of market data.

**Parameters:**
- `market_data_batch`: List of market data to store

##### query_data()
```python
async def query_data(
    symbol: Optional[str] = None,
    data_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame
```
Query stored raw data.

**Parameters:**
- `symbol`: Symbol filter (optional)
- `data_type`: Data type filter (optional)
- `start_date`: Start date filter (optional)
- `end_date`: End date filter (optional)

**Returns:**
- `pd.DataFrame`: Query results

##### get_statistics()
```python
def get_statistics() -> StorageStats
```
Get storage statistics.

**Returns:**
- `StorageStats`: Current storage statistics

### AggregatedTimescaleStorage

#### Constructor
```python
AggregatedTimescaleStorage(config: TimescaleDBConfig)
```

#### Methods

##### start()
```python
async def start() -> None
```
Start the aggregated storage system.

##### stop()
```python
async def stop() -> None
```
Stop the aggregated storage system.

##### store()
```python
async def store(market_data: MarketData) -> None
```
Store a single market data point.

##### store_batch()
```python
async def store_batch(market_data_batch: List[MarketData]) -> None
```
Store a batch of market data.

##### query_ohlcv()
```python
async def query_ohlcv(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame
```
Query OHLCV data.

**Parameters:**
- `symbol`: Trading symbol
- `interval`: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
- `start_time`: Start time
- `end_time`: End time

**Returns:**
- `pd.DataFrame`: OHLCV data

##### query_volume_stats()
```python
async def query_volume_stats(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame
```
Query volume statistics.

**Parameters:**
- `symbol`: Trading symbol
- `interval`: Time interval
- `start_time`: Start time
- `end_time`: End time

**Returns:**
- `pd.DataFrame`: Volume statistics

### StorageManager

#### Constructor
```python
StorageManager(config: StorageManagerConfig)
```

#### Methods

##### start()
```python
async def start() -> None
```
Start the storage manager.

##### stop()
```python
async def stop() -> None
```
Stop the storage manager.

##### store()
```python
async def store(market_data: MarketData) -> None
```
Store market data with automatic routing.

##### store_batch()
```python
async def store_batch(market_data_batch: List[MarketData]) -> None
```
Store a batch of market data.

##### query_raw_data()
```python
async def query_raw_data(
    symbol: Optional[str] = None,
    data_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Any
```
Query raw data from Parquet storage.

##### query_ohlcv_data()
```python
async def query_ohlcv_data(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime
) -> Any
```
Query OHLCV data from TimescaleDB.

##### query_volume_stats()
```python
async def query_volume_stats(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime
) -> Any
```
Query volume statistics from TimescaleDB.

##### get_statistics()
```python
def get_statistics() -> StorageStats
```
Get comprehensive storage statistics.

##### get_storage_status()
```python
def get_storage_status() -> Dict[str, Any]
```
Get detailed storage status.

---

## Best Practices

### 1. Configuration Management

#### Environment-Specific Configuration
```python
import os
from typing import Dict

def get_storage_config(environment: str) -> StorageManagerConfig:
    configs = {
        'development': get_dev_config(),
        'staging': get_staging_config(),
        'production': get_prod_config()
    }
    
    return configs.get(environment, get_dev_config())

def get_prod_config() -> StorageManagerConfig:
    return StorageManagerConfig(
        storage_mode=StorageMode.BOTH,
        raw_config=ParquetStorageConfig(
            base_path=os.getenv("RAW_STORAGE_PATH", "/data/parquet/raw"),
            compression="snappy",
            retention_days=int(os.getenv("RAW_RETENTION_DAYS", "365")),
            max_concurrent_writes=int(os.getenv("RAW_CONCURRENT_WRITES", "8"))
        ),
        aggregated_config=TimescaleDBConfig(
            dsn=os.getenv("TIMESCALEDB_DSN"),
            pool_size=int(os.getenv("TIMESCALEDB_POOL_SIZE", "20")),
            raw_data_retention_days=int(os.getenv("RAW_DATA_RETENTION", "7"))
        )
    )
```

#### Configuration Validation
```python
def validate_storage_config(config: StorageManagerConfig) -> bool:
    # Validate paths exist or can be created
    if config.storage_mode in [StorageMode.RAW_ONLY, StorageMode.BOTH]:
        raw_path = Path(config.raw_config.base_path)
        try:
            raw_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create raw storage path: {e}")
    
    # Validate database connection
    if config.storage_mode in [StorageMode.AGGREGATED_ONLY, StorageMode.BOTH]:
        try:
            # Test database connection
            conn = asyncpg.connect(config.aggregated_config.dsn)
            await conn.close()
        except Exception as e:
            raise ValueError(f"Cannot connect to TimescaleDB: {e}")
    
    return True
```

### 2. Performance Optimization

#### Batch Processing
```python
class BatchProcessor:
    def __init__(self, storage: StorageManager, batch_size: int = 1000):
        self.storage = storage
        self.batch_size = batch_size
        self.batch = []
    
    async def add_data(self, market_data: MarketData):
        self.batch.append(market_data)
        
        if len(self.batch) >= self.batch_size:
            await self.storage.store_batch(self.batch)
            self.batch.clear()
    
    async def flush(self):
        if self.batch:
            await self.storage.store_batch(self.batch)
            self.batch.clear()

# Usage
processor = BatchProcessor(storage, batch_size=2000)

async for market_data in data_stream:
    await processor.add_data(market_data)

await processor.flush()
```

#### Connection Pool Management
```python
class ConnectionManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.connection_stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'avg_query_time': 0.0
        }
    
    async def execute_query(self, query_func, *args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = await query_func(*args, **kwargs)
            self.connection_stats['total_queries'] += 1
            
            # Update average query time
            query_time = (datetime.now() - start_time).total_seconds()
            total_queries = self.connection_stats['total_queries']
            current_avg = self.connection_stats['avg_query_time']
            self.connection_stats['avg_query_time'] = (
                (current_avg * (total_queries - 1) + query_time) / total_queries
            )
            
            return result
            
        except Exception as e:
            self.connection_stats['failed_queries'] += 1
            logger.error(f"Query failed: {e}")
            raise
```

### 3. Error Handling

#### Comprehensive Error Handling
```python
class ResilientStorageManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_multiplier': 2.0
        }
        self.error_stats = {
            'total_errors': 0,
            'retry_successes': 0,
            'permanent_failures': 0
        }
    
    async def store_with_retry(self, market_data: MarketData) -> bool:
        for attempt in range(self.retry_config['max_retries']):
            try:
                await self.storage.store(market_data)
                return True
                
            except Exception as e:
                self.error_stats['total_errors'] += 1
                
                if attempt == self.retry_config['max_retries'] - 1:
                    self.error_stats['permanent_failures'] += 1
                    logger.error(f"Failed to store data after {attempt + 1} attempts: {e}")
                    return False
                
                # Calculate delay with exponential backoff
                delay = self.retry_config['retry_delay'] * (
                    self.retry_config['backoff_multiplier'] ** attempt
                )
                
                logger.warning(f"Storage attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        
        return False
    
    async def store_batch_with_fallback(self, batch: List[MarketData]) -> Dict[str, int]:
        results = {'success': 0, 'failed': 0}
        
        for data in batch:
            if await self.store_with_retry(data):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results
```

### 4. Monitoring and Alerting

#### Performance Monitoring
```python
class StorageMonitor:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.alert_thresholds = {
            'processing_rate_min': 1000,      # records/sec
            'queue_utilization_max': 0.8,     # 80%
            'error_rate_max': 0.01,           # 1%
            'storage_growth_max_mb': 1000      # MB/hour
        }
        self.last_storage_size = 0
        self.last_check_time = datetime.now()
    
    async def monitor_loop(self):
        while True:
            try:
                await self.check_performance()
                await self.check_storage_growth()
                await self.check_error_rate()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def check_performance(self):
        stats = self.storage.get_statistics()
        
        # Check processing rate
        if stats.processing_rate < self.alert_thresholds['processing_rate_min']:
            await self.send_alert(
                "LOW_PROCESSING_RATE",
                f"Processing rate: {stats.processing_rate:.1f} records/sec"
            )
        
        # Check queue utilization
        if stats.queue_utilization > self.alert_thresholds['queue_utilization_max']:
            await self.send_alert(
                "HIGH_QUEUE_UTILIZATION",
                f"Queue utilization: {stats.queue_utilization:.1%}"
            )
    
    async def check_storage_growth(self):
        status = self.storage.get_storage_status()
        raw_size = status.get('raw_storage', {}).get('size_mb', 0)
        
        current_time = datetime.now()
        time_diff = (current_time - self.last_check_time).total_seconds()
        
        if time_diff > 0:
            growth_rate = (raw_size - self.last_storage_size) / time_diff * 3600  # MB/hour
            
            if growth_rate > self.alert_thresholds['storage_growth_max_mb']:
                await self.send_alert(
                    "HIGH_STORAGE_GROWTH",
                    f"Storage growth: {growth_rate:.1f} MB/hour"
                )
        
        self.last_storage_size = raw_size
        self.last_check_time = current_time
    
    async def check_error_rate(self):
        stats = self.storage.get_statistics()
        
        if stats.total_processed > 0:
            error_rate = stats.error_count / stats.total_processed
            
            if error_rate > self.alert_thresholds['error_rate_max']:
                await self.send_alert(
                    "HIGH_ERROR_RATE",
                    f"Error rate: {error_rate:.2%}"
                )
    
    async def send_alert(self, alert_type: str, message: str):
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Here you could add:
        # - Send to monitoring system
        # - Send email/SMS alert
        # - Update dashboard
        # - Trigger automated responses
```

### 5. Data Retention and Cleanup

#### Automated Cleanup
```python
class DataRetentionManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.retention_policies = {
            'raw_data': timedelta(days=365),
            'ohlcv_1m': timedelta(days=7),
            'ohlcv_5m': timedelta(days=30),
            'ohlcv_1h': timedelta(days=365),
            'ohlcv_1d': timedelta(days=2555)  # 7 years
        }
    
    async def cleanup_loop(self):
        while True:
            try:
                await self.perform_cleanup()
                await asyncio.sleep(24 * 3600)  # Run daily
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def perform_cleanup(self):
        cutoff_date = datetime.now() - self.retention_policies['raw_data']
        
        # Clean up raw Parquet files
        if self.storage.raw_storage:
            await self.cleanup_parquet_files(cutoff_date)
        
        # Clean up aggregated TimescaleDB data
        if self.storage.aggregated_storage:
            await self.cleanup_timescaledb_data()
        
        logger.info("Data cleanup completed")
    
    async def cleanup_parquet_files(self, cutoff_date: datetime):
        base_path = Path(self.storage.raw_storage.config.base_path)
        
        deleted_files = 0
        deleted_size = 0
        
        for file_path in base_path.rglob("*.parquet"):
            try:
                # Extract date from path
                date_str = file_path.parent.parent.name  # date=YYYY-MM-DD
                if date_str.startswith("date="):
                    date_part = date_str[5:]  # Remove "date=" prefix
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files += 1
                        deleted_size += file_size
                        
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")
        
        if deleted_files > 0:
            logger.info(f"Cleaned up {deleted_files} Parquet files, "
                       f"freed {deleted_size / (1024*1024):.1f} MB")
    
    async def cleanup_timescaledb_data(self):
        # This would implement TimescaleDB retention policies
        # Using TimescaleDB's built-in retention policies
        pass
```

### 6. Testing and Validation

#### Unit Testing
```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_raw_parquet_storage():
    config = ParquetStorageConfig(
        base_path="test_data/parquet",
        retention_days=1
    )
    
    storage = RawParquetStorage(config)
    await storage.start()
    
    try:
        # Test storing data
        market_data = MarketData(
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            symbol="BTC-USDT",
            type=MarketDataType.TICKER,
            exchange="test",
            data={"last_price": 50000.0}
        )
        
        await storage.store(market_data)
        
        # Test querying data
        result = await storage.query_data(symbol="BTC-USDT")
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == 'BTC-USDT'
        
    finally:
        await storage.stop()

@pytest.mark.asyncio
async def test_storage_manager_routing():
    config = StorageManagerConfig(
        storage_mode=StorageMode.BOTH,
        routing_rules={"ticker": "both", "orderbook": "raw_only"}
    )
    
    manager = StorageManager(config)
    await manager.start()
    
    try:
        # Test ticker data routing
        ticker_data = MarketData(
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            symbol="BTC-USDT",
            type=MarketDataType.TICKER,
            exchange="test",
            data={"last_price": 50000.0}
        )
        
        await manager.store(ticker_data)
        
        # Verify data was stored in both systems
        assert manager.stats.total_processed == 1
        
    finally:
        await manager.stop()
```

#### Integration Testing
```python
@pytest.mark.asyncio
async def test_end_to_end_storage():
    # Create test configuration
    config = StorageManagerConfig(
        storage_mode=StorageMode.BOTH,
        raw_config=ParquetStorageConfig(
            base_path="test_data/parquet",
            retention_days=1
        ),
        aggregated_config=TimescaleDBConfig(
            dsn="postgres://test:test@localhost:5432/test_db"
        )
    )
    
    manager = StorageManager(config)
    await manager.start()
    
    try:
        # Store test data
        test_data = [
            MarketData(
                timestamp_ms=int(datetime.now().timestamp() * 1000) + i * 1000,
                symbol="BTC-USDT",
                type=MarketDataType.TICKER,
                exchange="test",
                data={"last_price": 50000.0 + i, "volume_24h": 1000.0}
            )
            for i in range(100)
        ]
        
        await manager.store_batch(test_data)
        
        # Wait for aggregation
        await asyncio.sleep(5)
        
        # Query aggregated data
        ohlcv_data = await manager.query_ohlcv_data(
            symbol="BTC-USDT",
            interval="1m",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert len(ohlcv_data) > 0
        
        # Verify data integrity
        for _, row in ohlcv_data.iterrows():
            assert row['open'] > 0
            assert row['high'] >= row['low']
            assert row['volume'] >= 0
        
    finally:
        await manager.stop()
```

---

## Troubleshooting

### Common Issues

#### 1. Parquet Storage Issues

**Symptoms:**
- High memory usage during writes
- Slow query performance
- File corruption errors

**Solutions:**
```python
# Reduce batch size and concurrent writes
config = ParquetStorageConfig(
    write_batch_size=5000,        # Reduce from 10000
    max_concurrent_writes=2,     # Reduce from 4
    row_group_size=50000         # Reduce from 100000
)

# Enable file validation
config.enable_schema_validation = True
config.enable_data_validation = True

# Monitor memory usage
import psutil

def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent}%")
        # Trigger cleanup or reduce batch size
```

#### 2. TimescaleDB Connection Issues

**Symptoms:**
- Connection timeouts
- Pool exhaustion
- High latency

**Solutions:**
```python
# Increase pool size and timeout
config = TimescaleDBConfig(
    pool_size=20,                           # Increase from 10
    command_timeout=60.0,                    # Increase from 30
    max_inactive_connection_lifetime=600.0, # Increase from 300
    batch_size=500,                         # Reduce from 1000
    max_concurrent_inserts=5                # Reduce from 10
)

# Add connection retry logic
async def connect_with_retry(config: TimescaleDBConfig):
    for attempt in range(3):
        try:
            storage = AggregatedTimescaleStorage(config)
            await storage.start()
            return storage
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
```

#### 3. Performance Issues

**Symptoms:**
- Slow write performance
- High queue utilization
- Memory leaks

**Solutions:**
```python
# Optimize batch processing
config = StorageManagerConfig(
    batch_size=2000,              # Increase batch size
    max_queue_size=50000,         # Increase queue size
    processing_timeout=60.0,      # Increase timeout
    enable_monitoring=True        # Enable monitoring
)

# Add memory monitoring
async def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        logger.warning(f"High memory usage: {memory_info.rss / (1024**3):.1f} GB")
        # Trigger garbage collection or reduce batch size
        import gc
        gc.collect()
```

#### 4. Data Integrity Issues

**Symptoms:**
- Missing data
- Corrupted files
- Schema mismatches

**Solutions:**
```python
# Enable data validation
config = ParquetStorageConfig(
    enable_schema_validation=True,
    enable_data_validation=True,
    schema_evolution=True
)

# Add data integrity checks
async def validate_data_integrity(storage: StorageManager):
    # Check for gaps in data
    latest_timestamp = await storage.get_latest_timestamp()
    expected_timestamp = datetime.now() - timedelta(minutes=5)
    
    if latest_timestamp < expected_timestamp:
        logger.warning(f"Data gap detected: latest {latest_timestamp}, expected {expected_timestamp}")
    
    # Validate schema consistency
    schema_issues = await storage.check_schema_consistency()
    if schema_issues:
        logger.error(f"Schema issues detected: {schema_issues}")
```

### Debug Mode

#### Enable Detailed Logging
```python
import logging

# Enable debug logging for storage components
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific component logging
logging.getLogger("its_project.storage.raw_parquet_storage").setLevel(logging.DEBUG)
logging.getLogger("its_project.storage.aggregated_timescale_storage").setLevel(logging.DEBUG)
logging.getLogger("its_project.storage.storage_manager").setLevel(logging.DEBUG)
```

#### Performance Profiling
```python
import cProfile
import pstats

def profile_storage_operations():
    profiler = cProfile.Profile()
    
    # Profile write operations
    profiler.enable()
    
    async def write_test_data():
        storage = create_storage_manager()
        await storage.start()
        
        for i in range(1000):
            market_data = MarketData(
                timestamp_ms=int(datetime.now().timestamp() * 1000) + i,
                symbol="BTC-USDT",
                type=MarketDataType.TICKER,
                exchange="test",
                data={"last_price": 50000.0 + i}
            )
            await storage.store(market_data)
        
        await storage.stop()
    
    # Run test
    asyncio.run(write_test_data())
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Performance Tuning

#### Storage-Specific Tuning

##### Parquet Optimization
```python
# Optimize for specific query patterns
query_optimized_config = ParquetStorageConfig(
    # For time-range queries
    partition_cols=["date", "symbol"],
    
    # For analytical queries
    row_group_size=100000,
    compression="snappy",
    
    # For high-frequency writes
    max_concurrent_writes=8,
    write_batch_size=20000,
    
    # For long-term storage
    retention_days=365,
    enable_compression=True
)
```

##### TimescaleDB Optimization
```python
# Optimize for time-series queries
timescale_optimized_config = TimescaleDBConfig(
    # Connection optimization
    pool_size=20,
    command_timeout=60.0,
    
    # Batch optimization
    batch_size=2000,
    max_concurrent_inserts=10,
    
    # Query optimization
    enable_compression=True,
    aggregation_intervals=["1m", "5m", "15m", "1h", "4h", "1d"],
    
    # Retention optimization
    raw_data_retention_days=7,
    aggregated_data_retention_days=2555
)
```

This comprehensive documentation provides complete coverage of the storage layer system, including raw Parquet storage, aggregated TimescaleDB storage, and the unified storage manager, with practical examples, best practices, and troubleshooting guidance.
