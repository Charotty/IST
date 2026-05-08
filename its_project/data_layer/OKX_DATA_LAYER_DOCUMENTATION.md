# OKX Data Layer Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Data Storage](#data-storage)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The OKX Data Layer provides comprehensive market data collection and management capabilities for the ITS project. It includes real-time data streaming, automatic historical data downloads, backfill systems, and data validation.

### Key Features

- **Real-time WebSocket streaming** for ticker, orderbook, and trades
- **Automatic historical data download** with configurable timeframes
- **Backfill system** for filling gaps in historical data
- **Data validation** and integrity monitoring
- **Configurable storage** with compression support
- **Production-ready reliability** with error handling and retries

---

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OKX Source    │    │  Backfill Manager │    │ Auto-Downloader │
│                 │    │                  │    │                 │
│ • WebSocket     │◄──►│ • Job Queue      │◄──►│ • Scheduler     │
│ • REST API      │    │ • Gap Detection  │    │ • Validation    │
│ • Real-time     │    │ • Progress Track │    │ • Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Data Storage  │
                    │                 │
                    │ • CSV/Parquet   │
                    │ • Compression   │
                    │ • SQLite DB     │
                    └─────────────────┘
```

---

## Components

### 1. OKXDataSource

Main data source for OKX exchange with real-time streaming and historical data capabilities.

#### Key Methods

```python
# Connection management
async def connect() -> None
async def disconnect() -> None
async def is_alive() -> bool

# Data streaming
async def subscribe(symbols: List[str]) -> AsyncIterator[MarketData]
async def fetch(symbol: str, data_type: str) -> MarketData

# Historical data
async def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> List[List[float]]
async def fetch_trades(symbol: str, limit: int) -> List[Dict[str, Any]]

# Backfill requests
def request_backfill(symbol: str, data_type: str, start_time: datetime, end_time: datetime)
```

#### Data Types Supported

- **Ticker**: Real-time price information
- **Orderbook**: Order book depth with bids/asks
- **Trades**: Recent trade executions
- **OHLCV**: Historical candlestick data

### 2. BackfillManager

Comprehensive backfill system for filling gaps in historical data.

#### Features

- **Job Queue Management**: Priority-based task scheduling
- **Gap Detection**: Automatic identification of missing data
- **Progress Tracking**: Real-time progress monitoring
- **Error Handling**: Retry mechanism with exponential backoff
- **Database Persistence**: SQLite-based job tracking

#### Key Methods

```python
# Job management
async def create_backfill_job(symbol: str, exchange: str, data_type: str, 
                             start_time: datetime, end_time: datetime) -> str
async def detect_gaps(symbol: str, exchange: str, data_type: str) -> List[str]

# Control
async def start() -> None
async def stop() -> None
def cancel_job(job_id: str) -> bool

# Monitoring
def get_job_status(job_id: str) -> Optional[BackfillJob]
def get_queue_status() -> Dict[str, Any]
```

### 3. AutoDownloader

Automated system for downloading and maintaining historical market data.

#### Features

- **Scheduled Downloads**: Configurable automatic downloads
- **Incremental Updates**: Efficient data maintenance
- **Data Validation**: Completeness and continuity checks
- **Multiple Formats**: CSV and Parquet support
- **Compression**: Optional data compression

#### Key Methods

```python
# Task management
async def schedule_download(symbol: str, exchange: str, data_type: str, 
                           timeframe: Optional[str] = None) -> str

# Control
async def start() -> None
async def stop() -> None

# Monitoring
def get_download_status(task_id: str) -> Optional[DownloadTask]
def get_statistics() -> Dict[str, Any]
```

---

## Configuration

### OKXConfig

```python
@dataclass
class OKXConfig:
    # API configuration
    api_key: Optional[str] = None
    secret: Optional[str] = None
    passphrase: Optional[str] = None
    sandbox: bool = False
    
    # Data configuration
    symbols: List[str] = ["BTC-USDT", "ETH-USDT"]
    data_types: List[str] = ["ticker", "orderbook", "trades"]
    orderbook_depth: int = 20
    
    # Auto-download configuration
    enable_auto_download: bool = True
    download_timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"]
    max_download_days: int = 365
    download_interval: int = 3600  # 1 hour
    
    # Backfill configuration
    enable_backfill: bool = True
    backfill_gap_threshold: int = 300  # 5 minutes
    backfill_batch_size: int = 1000
    
    # Storage configuration
    storage_path: str = "data/okx"
    enable_compression: bool = True
```

### BackfillConfig

```python
@dataclass
class BackfillConfig:
    max_concurrent_jobs: int = 3
    job_timeout: int = 3600  # 1 hour
    retry_delay: int = 60  # 1 minute
    database_path: str = "data/backfill.db"
    enable_scheduling: bool = True
    schedule_interval: int = 3600  # 1 hour
    gap_threshold: int = 300  # 5 minutes
```

### AutoDownloadConfig

```python
@dataclass
class AutoDownloadConfig:
    enabled: bool = True
    max_concurrent_downloads: int = 2
    download_timeout: int = 1800  # 30 minutes
    
    # Scheduling
    enable_scheduling: bool = True
    schedule_interval: int = 3600  # 1 hour
    incremental_updates: bool = True
    update_window_hours: int = 24
    
    # Data settings
    default_timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"]
    max_history_days: int = 365
    
    # Storage settings
    base_storage_path: str = "data/market_data"
    enable_compression: bool = True
    file_format: str = "parquet"  # "csv" or "parquet"
    
    # Validation settings
    enable_validation: bool = True
    validate_completeness: bool = True
    validate_continuity: bool = True
```

---

## Usage Examples

### Basic Setup

```python
from its_project.data_layer import (
    OKXDataSource, OKXConfig,
    BackfillManager, BackfillConfig,
    AutoDownloader, AutoDownloadConfig
)

# Create OKX source
okx_config = OKXConfig(
    symbols=["BTC-USDT", "ETH-USDT"],
    enable_auto_download=True,
    enable_backfill=True
)
okx_source = OKXDataSource(okx_config)
await okx_source.connect()
```

### Real-time Data Streaming

```python
# Subscribe to real-time data
async for market_data in okx_source.subscribe():
    if market_data.type == MarketDataType.TICKER:
        print(f"Ticker: {market_data.symbol} - ${market_data.data['last_price']}")
    elif market_data.type == MarketDataType.TRADE:
        for trade in market_data.data['trades']:
            print(f"Trade: {trade['side']} {trade['size']} @ {trade['price']}")
```

### Historical Data Download

```python
# Create backfill manager
backfill_config = BackfillConfig(max_concurrent_jobs=3)
backfill_manager = BackfillManager(backfill_config)
backfill_manager.add_data_source("okx", okx_source)
await backfill_manager.start()

# Request historical data
job_id = await backfill_manager.create_backfill_job(
    symbol="BTC-USDT",
    exchange="okx",
    data_type="ohlcv",
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    timeframe="1h"
)
```

### Automated Downloading

```python
# Create auto-downloader
download_config = AutoDownloadConfig(
    max_concurrent_downloads=2,
    enable_scheduling=True,
    base_storage_path="data/market_data"
)
auto_downloader = AutoDownloader(download_config)
auto_downloader.add_data_source("okx", okx_source)
await auto_downloader.start()

# Schedule downloads for all symbols and timeframes
for symbol in okx_config.symbols:
    for timeframe in okx_config.download_timeframes:
        await auto_downloader.schedule_download(
            symbol=symbol,
            exchange="okx",
            data_type="ohlcv",
            timeframe=timeframe
        )
```

### Complete Integration

```python
import asyncio
from datetime import datetime, timedelta

async def main():
    # Initialize components
    okx_config = OKXConfig(
        symbols=["BTC-USDT", "ETH-USDT"],
        enable_auto_download=True,
        enable_backfill=True
    )
    okx_source = OKXDataSource(okx_config)
    
    backfill_config = BackfillConfig(max_concurrent_jobs=3)
    backfill_manager = BackfillManager(backfill_config)
    backfill_manager.add_data_source("okx", okx_source)
    
    download_config = AutoDownloadConfig(
        max_concurrent_downloads=2,
        enable_scheduling=True
    )
    auto_downloader = AutoDownloader(download_config)
    auto_downloader.add_data_source("okx", okx_source)
    
    # Start all systems
    await okx_source.connect()
    await backfill_manager.start()
    await auto_downloader.start()
    
    # Subscribe to real-time data
    async for market_data in okx_source.subscribe():
        print(f"Real-time: {market_data.symbol} {market_data.type}")
        
        # Check system status periodically
        if datetime.now().second % 60 == 0:  # Every minute
            print(f"Backfill queue: {backfill_manager.get_queue_status()}")
            print(f"Download stats: {auto_downloader.get_statistics()}")

# Run the system
asyncio.run(main())
```

---

## Data Storage

### File Organization

```
data/
├── okx/                          # OKX-specific data
│   ├── BTC-USDT/
│   │   ├── ohlcv/
│   │   │   ├── 1m_20240101.parquet
│   │   │   ├── 5m_20240101.parquet
│   │   │   ├── 1h_20240101.parquet
│   │   │   └── 1d_20240101.parquet
│   │   ├── ticker/
│   │   │   └── ticker_20240101.parquet
│   │   └── trades/
│   │       └── trades_20240101.parquet
│   └── ETH-USDT/
│       └── ...
├── market_data/                  # Auto-downloader data
│   ├── okx/
│   │   └── BTC-USDT/
│   │       ├── ohlcv/
│   │       └── ticker/
├── backfill.db                   # Backfill job database
└── auto_download.db              # Auto-download database
```

### Data Formats

#### Parquet Format (Recommended)
- **Compression**: Snappy compression
- **Performance**: Fast read/write operations
- **Schema**: Structured column storage
- **Size**: Efficient storage footprint

#### CSV Format
- **Compression**: GZIP compression
- **Compatibility**: Universal support
- **Readability**: Human-readable format
- **Size**: Larger storage footprint

### Database Schema

#### Backfill Jobs Table
```sql
CREATE TABLE backfill_jobs (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    data_type TEXT NOT NULL,
    timeframe TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    priority INTEGER DEFAULT 1,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    progress REAL DEFAULT 0.0
);
```

#### Download Tasks Table
```sql
CREATE TABLE download_tasks (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    data_type TEXT NOT NULL,
    timeframe TEXT,
    priority INTEGER DEFAULT 1,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    file_path TEXT,
    file_hash TEXT,
    file_size INTEGER DEFAULT 0,
    records_count INTEGER DEFAULT 0
);
```

---

## API Reference

### OKXDataSource

#### Constructor
```python
OKXDataSource(config: OKXConfig)
```

#### Methods

##### connect()
```python
async def connect() -> None
```
Connect to OKX API and WebSocket.

##### subscribe()
```python
async def subscribe(symbols: Optional[List[str]] = None) -> AsyncIterator[MarketData]
```
Subscribe to real-time market data.

**Parameters:**
- `symbols`: List of symbols to subscribe to

**Yields:**
- `MarketData`: Real-time market data

##### fetch()
```python
async def fetch(symbol: str, data_type: str = "ticker", **params) -> MarketData
```
Fetch current market data.

**Parameters:**
- `symbol`: Trading symbol (e.g., "BTC-USDT")
- `data_type`: Data type ("ticker", "orderbook", "trades")

**Returns:**
- `MarketData`: Current market data

##### request_backfill()
```python
def request_backfill(
    symbol: str,
    data_type: str,
    start_time: datetime,
    end_time: datetime,
    timeframe: Optional[str] = None,
    reason: str = "manual"
) -> None
```
Request backfill for missing data.

### BackfillManager

#### Constructor
```python
BackfillManager(config: BackfillConfig)
```

#### Methods

##### create_backfill_job()
```python
async def create_backfill_job(
    symbol: str,
    exchange: str,
    data_type: str,
    start_time: datetime,
    end_time: datetime,
    timeframe: Optional[str] = None,
    priority: int = 1
) -> str
```
Create a new backfill job.

**Returns:**
- `str`: Job ID

##### detect_gaps()
```python
async def detect_gaps(symbol: str, exchange: str, data_type: str) -> List[str]
```
Detect gaps in data and create backfill jobs.

**Returns:**
- `List[str]`: List of created job IDs

### AutoDownloader

#### Constructor
```python
AutoDownloader(config: AutoDownloadConfig)
```

#### Methods

##### schedule_download()
```python
async def schedule_download(
    symbol: str,
    exchange: str,
    data_type: str,
    timeframe: Optional[str] = None,
    priority: int = 1
) -> str
```
Schedule a download task.

**Returns:**
- `str`: Task ID

##### get_statistics()
```python
def get_statistics() -> Dict[str, Any]
```
Get download statistics.

**Returns:**
- `Dict[str, Any]`: Statistics including total downloads, success rate, etc.

---

## Best Practices

### 1. Configuration Management

```python
# Use environment variables for sensitive data
import os

okx_config = OKXConfig(
    api_key=os.getenv("OKX_API_KEY"),
    secret=os.getenv("OKX_SECRET"),
    passphrase=os.getenv("OKX_PASSPHRASE"),
    sandbox=os.getenv("OKX_SANDBOX", "false").lower() == "true"
)
```

### 2. Error Handling

```python
try:
    async for market_data in okx_source.subscribe():
        # Process data
        pass
except Exception as e:
    logger.error(f"Data streaming error: {e}")
    # Implement appropriate error handling
```

### 3. Resource Management

```python
# Always cleanup resources
try:
    await okx_source.connect()
    await backfill_manager.start()
    await auto_downloader.start()
    
    # Run main logic
    pass
    
finally:
    await backfill_manager.stop()
    await auto_downloader.stop()
    await okx_source.disconnect()
```

### 4. Performance Optimization

```python
# Configure appropriate concurrency
backfill_config = BackfillConfig(
    max_concurrent_jobs=3,  # Don't exceed API limits
    job_timeout=1800,       # Reasonable timeout
    retry_delay=60          # Avoid rapid retries
)

download_config = AutoDownloadConfig(
    max_concurrent_downloads=2,  # Conservative concurrency
    batch_size=1000,             # Efficient batch size
    enable_compression=True       # Save storage space
)
```

### 5. Monitoring

```python
# Regular status checks
async def monitor_system():
    while True:
        backfill_status = backfill_manager.get_queue_status()
        download_stats = auto_downloader.get_statistics()
        
        logger.info(f"Backfill: {backfill_status['active_jobs']} active jobs")
        logger.info(f"Downloads: {download_stats['successful_downloads']} successful")
        
        await asyncio.sleep(60)  # Check every minute
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Problems

**Symptoms:**
- Failed to connect to OKX API
- WebSocket connection drops

**Solutions:**
```python
# Check API credentials
okx_config = OKXConfig(
    api_key="your_api_key",
    secret="your_secret",
    passphrase="your_passphrase",
    sandbox=True  # Try sandbox first
)

# Test connection
try:
    await okx_source.connect()
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

#### 2. Rate Limiting

**Symptoms:**
- API rate limit errors
- Slow download speeds

**Solutions:**
```python
# Reduce concurrency
backfill_config = BackfillConfig(
    max_concurrent_jobs=1,  # Reduce from 3
    job_timeout=3600,       # Increase timeout
    retry_delay=300         # Longer retry delay
)

# Add delays between requests
await asyncio.sleep(0.1)  # 100ms delay
```

#### 3. Data Validation Errors

**Symptoms:**
- Validation failures
- Incomplete data

**Solutions:**
```python
# Adjust validation settings
download_config = AutoDownloadConfig(
    enable_validation=True,
    validate_completeness=False,  # Disable strict completeness
    validate_continuity=False,    # Disable strict continuity
    max_gap_minutes=120           # Increase gap tolerance
)
```

#### 4. Storage Issues

**Symptoms:**
- File write errors
- Disk space issues

**Solutions:**
```python
# Check storage path
import os
storage_path = "data/okx"
os.makedirs(storage_path, exist_ok=True)

# Check disk space
import shutil
total, used, free = shutil.disk_usage(storage_path)
print(f"Free space: {free // (1024**3)} GB")

# Use compression
okx_config = OKXConfig(
    enable_compression=True,  # Enable compression
    storage_path=storage_path
)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("its_project.data_layer.okx_source")

# Enable detailed logging
okx_config = OKXConfig(
    # ... other config
)
okx_source = OKXDataSource(okx_config)

# Monitor detailed logs
async for market_data in okx_source.subscribe():
    logger.debug(f"Received data: {market_data}")
```

### Performance Monitoring

```python
# Monitor system performance
def monitor_performance():
    import psutil
    import time
    
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    logger.info(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
    
    # Monitor queue sizes
    backfill_status = backfill_manager.get_queue_status()
    download_stats = auto_downloader.get_statistics()
    
    logger.info(f"Backfill queue: {backfill_status['queue_size']}")
    logger.info(f"Download queue: {download_stats['queue_size']}")
```

---

## Integration Examples

### 1. Integration with Trading System

```python
class TradingSystem:
    def __init__(self):
        self.okx_source = OKXDataSource(OKXConfig())
        self.backfill_manager = BackfillManager(BackfillConfig())
        self.auto_downloader = AutoDownloader(AutoDownloadConfig())
        
        # Trading data cache
        self.market_data_cache = {}
    
    async def start(self):
        await self.okx_source.connect()
        await self.backfill_manager.start()
        await self.auto_downloader.start()
        
        # Start data processing
        asyncio.create_task(self.process_market_data())
    
    async def process_market_data(self):
        async for market_data in self.okx_source.subscribe():
            # Cache latest data
            self.market_data_cache[market_data.symbol] = market_data
            
            # Trigger trading logic
            if market_data.type == MarketDataType.TICKER:
                await self.analyze_market(market_data)
```

### 2. Integration with Analytics Pipeline

```python
class AnalyticsPipeline:
    def __init__(self):
        self.okx_source = OKXDataSource(OKXConfig())
        self.auto_downloader = AutoDownloader(AutoDownloadConfig())
        
        # Analytics components
        self.feature_engineer = None
        self.model_predictor = None
    
    async def start_analytics(self):
        # Ensure historical data is available
        await self.ensure_historical_data()
        
        # Start real-time processing
        async for market_data in self.okx_source.subscribe():
            features = self.feature_engineer.extract_features(market_data)
            prediction = self.model_predictor.predict(features)
            
            # Handle prediction
            await self.handle_prediction(prediction)
    
    async def ensure_historical_data(self):
        # Check data availability
        for symbol in ["BTC-USDT", "ETH-USDT"]:
            for timeframe in ["1m", "5m", "1h"]:
                if not await self.check_data_availability(symbol, timeframe):
                    await self.auto_downloader.schedule_download(
                        symbol, "okx", "ohlcv", timeframe
                    )
```

This documentation provides comprehensive coverage of the OKX data layer implementation, including setup, configuration, usage examples, and troubleshooting guidance.
