# Data Pipeline Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Gap Detection System](#gap-detection-system)
5. [Reconnection Management](#reconnection-management)
6. [Integration](#integration)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Monitoring and Statistics](#monitoring-and-statistics)
10. [API Reference](#api-reference)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The Data Pipeline provides a comprehensive, production-ready system for collecting, processing, and monitoring market data with built-in gap detection and automatic reconnection capabilities.

### Key Features

- **Real-time Data Processing**: High-throughput processing of market data streams
- **Gap Detection**: Automatic detection of data gaps and anomalies
- **Reconnection Management**: Intelligent reconnection with circuit breaker pattern
- **Backfill Integration**: Automatic filling of missing historical data
- **Monitoring & Alerting**: Comprehensive monitoring with configurable alerts
- **Performance Optimization**: Configurable concurrency and buffering
- **Production Reliability**: Error handling, retries, and graceful degradation

### Architecture Benefits

- **Modular Design**: Each component can be used independently or together
- **Scalable**: Horizontal scaling with configurable workers
- **Resilient**: Automatic recovery from all failure types
- **Observable**: Comprehensive statistics and monitoring
- **Maintainable**: Clean separation of concerns and interfaces

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ OKX Source  │    │Gap Detector │    │ Reconnection │        │
│  │             │◄──►│             │◄──►│   Manager   │        │
│  │ • WebSocket │    │ • Real-time │    │ • Circuit   │        │
│  │ • REST API  │    │ • Anomalies │    │ • Backoff   │        │
│  │ • Streaming │    │ • Backfill  │    │ • Health    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│           │                   │                   │           │
│           └───────────────────┼───────────────────┘           │
│                               │                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                Processing Layer                           │  │
│  │                                                         │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │  │
│  │  │ Input Queue │    │   Workers    │    │Output Queue │ │  │
│  │  │             │──►│             │──►│             │ │  │
│  │  │ • Buffering │    │ • Validation │    │ • Enriched  │ │  │
│  │  │ • Rate Limit│    │ • Processing │    │ • Metadata  │ │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
│                               │                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Monitoring & Alerting                       │  │
│  │                                                         │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │  │
│  │  │ Statistics  │    │   Alerts    │    │ Dashboard   │ │  │
│  │  │             │    │             │    │             │ │  │
│  │  │ • Rates     │    │ • Gaps      │    │ • Status    │ │  │
│  │  │ • Errors    │    │ • Connections│    │ • Metrics   │ │  │
│  │  │ • Uptime    │    │ • Health    │    │ • History   │ │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: OKX source provides real-time market data
2. **Gap Detection**: Continuous monitoring for data gaps and anomalies
3. **Reconnection Management**: Automatic handling of connection issues
4. **Processing**: Multi-worker validation and enrichment
5. **Output**: Processed data ready for consumption
6. **Monitoring**: Real-time statistics and alerting

---

## Components

### 1. DataPipeline

Main orchestrator that integrates all pipeline components.

#### Key Responsibilities
- Component lifecycle management
- Data flow coordination
- Performance monitoring
- Error handling and recovery

#### Core Methods
```python
async def start() -> None
async def stop() -> None
async def get_data_stream() -> AsyncIterator[MarketData]
def get_pipeline_status() -> Dict[str, Any]
```

### 2. GapDetector

Advanced gap detection system with real-time monitoring.

#### Detection Capabilities
- **Time Gaps**: Missing data between expected timestamps
- **Price Anomalies**: Sudden price movements beyond thresholds
- **Volume Spikes**: Abnormal volume changes
- **Stale Data**: No recent updates
- **Data Quality**: Validation of data integrity

#### Configuration
```python
GapDetectionConfig(
    ticker_gap_threshold=5.0,      # 5 seconds
    trade_gap_threshold=1.0,        # 1 second
    orderbook_gap_threshold=2.0,    # 2 seconds
    ohlcv_gap_threshold=60.0,       # 1 minute
    max_price_jump_pct=10.0,        # 10% price jump
    max_volume_spike_pct=1000.0,    # 1000% volume spike
    auto_backfill=True,
    enable_real_time_detection=True
)
```

### 3. ReconnectionManager

Production-grade reconnection system with circuit breaker pattern.

#### Features
- **Exponential Backoff**: Intelligent retry timing with jitter
- **Circuit Breaker**: Prevents cascading failures
- **Health Monitoring**: Periodic connection validation
- **Statistics Tracking**: Comprehensive connection metrics
- **Alert Integration**: Automatic notifications for issues

#### Configuration
```python
ReconnectionConfig(
    max_reconnect_attempts=10,
    initial_delay=1.0,              # 1 second
    max_delay=300.0,               # 5 minutes
    backoff_multiplier=2.0,
    jitter=True,                    # Add randomness
    enable_circuit_breaker=True,
    failure_threshold=5,            # Failures before opening circuit
    recovery_timeout=60.0,          # Seconds to wait before retry
    enable_health_checks=True,
    health_check_interval=30.0      # seconds
)
```

### 4. BackfillManager

Automatic backfill system for missing historical data.

#### Capabilities
- **Gap Detection**: Automatic identification of missing data
- **Job Queue**: Priority-based backfill scheduling
- **Batch Processing**: Efficient data retrieval
- **Progress Tracking**: Real-time backfill monitoring
- **Error Handling**: Retry mechanism for failed backfills

---

## Gap Detection System

### Detection Types

#### 1. Time Gaps
Missing data between consecutive messages based on expected intervals.

**Example**: Ticker data expected every 1 second, but no data for 15 seconds.

```python
# Gap detection in action
gap_info = await gap_detector.process_market_data(market_data)
if gap_info:
    print(f"Gap detected: {gap_info.symbol} {gap_info.duration_ms}ms")
```

#### 2. Price Anomalies
Sudden price movements beyond configured thresholds.

**Example**: BTC price jumps from $50,000 to $55,000 (10% jump).

```python
# Price jump detection
if price_change_pct > config.max_price_jump_pct:
    gap_info = GapInfo(
        gap_type="price_jump",
        severity="high",
        duration_ms=0
    )
```

#### 3. Volume Spikes
Abnormal volume increases indicating potential issues.

**Example**: Volume suddenly increases by 2000%.

```python
# Volume spike detection
if volume_change_pct > config.max_volume_spike_pct:
    gap_info = GapInfo(
        gap_type="volume_spike",
        severity="medium",
        duration_ms=0
    )
```

### Gap Statistics

#### Comprehensive Tracking
```python
gap_stats = gap_detector.get_gap_statistics()
# {
#     'total_gaps_detected': 25,
#     'gaps_by_type': {
#         'missing': 15,
#         'price_jump': 8,
#         'volume_spike': 2
#     },
#     'gaps_by_severity': {
#         'low': 10,
#         'medium': 12,
#         'high': 3
#     },
#     'average_gap_duration': 12.5,
#     'longest_gap_duration': 180.0,
#     'detection_rate': 0.42  # gaps per hour
# }
```

### Automatic Backfill Integration

#### Gap → Backfill Workflow
1. **Gap Detected**: Real-time gap identification
2. **Alert Triggered**: Notification sent to monitoring
3. **Backfill Request**: Automatic request to backfill manager
4. **Data Retrieval**: Historical data fetched
5. **Validation**: Data integrity verified
6. **Gap Resolution**: Missing data filled

```python
# Automatic backfill request
if gap_info.gap_type == "missing" and backfill_manager:
    await backfill_manager.create_backfill_job(
        symbol=gap_info.symbol,
        exchange=gap_info.exchange,
        data_type="ohlcv",
        start_time=datetime.fromtimestamp(gap_info.start_time / 1000),
        end_time=datetime.fromtimestamp(gap_info.end_time / 1000),
        priority=2
    )
```

---

## Reconnection Management

### Connection States

#### State Machine
```
DISCONNECTED → CONNECTING → CONNECTED
     ↑              ↓           ↓
     └── RECONNECTING ← FAILED ← ┘
           ↓
     CIRCUIT_OPEN
```

#### State Descriptions
- **DISCONNECTED**: No active connection
- **CONNECTING**: Attempting to establish connection
- **CONNECTED**: Connection active and healthy
- **RECONNECTING**: Attempting to restore connection
- **FAILED**: Connection attempt failed
- **CIRCUIT_OPEN**: Circuit breaker preventing connections

### Circuit Breaker Pattern

#### Protection Mechanism
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time_since_failure > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        return True
```

#### Benefits
- **Prevents Cascading Failures**: Stops repeated failed attempts
- **Automatic Recovery**: Gradual return to normal operation
- **Resource Protection**: Conserves system resources during outages
- **Fast Failure**: Quick detection of systemic issues

### Exponential Backoff

#### Retry Strategy
```python
def calculate_delay(attempt: int, initial_delay: float, max_delay: float) -> float:
    delay = min(initial_delay * (2 ** attempt), max_delay)
    
    # Add jitter to prevent thundering herd
    if jitter:
        delay *= (0.5 + random.random() * 0.5)
    
    return delay
```

#### Retry Sequence Example
```
Attempt 1: 1.0s
Attempt 2: 2.0s
Attempt 3: 4.0s
Attempt 4: 8.0s
Attempt 5: 16.0s
Attempt 6: 32.0s
Attempt 7: 64.0s
Attempt 8: 128.0s
Attempt 9: 256.0s
Attempt 10: 300.0s (max_delay)
```

### Health Monitoring

#### Periodic Health Checks
```python
async def health_check_loop(self, source_name: str):
    while self._running:
        try:
            # Wait for interval
            await asyncio.sleep(self.config.health_check_interval)
            
            # Perform health check
            is_healthy = await self._perform_health_check(source_name)
            
            if not is_healthy:
                # Trigger reconnection
                await self.disconnect_source(source_name)
                await self._start_reconnection(source_name)
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
```

#### Health Check Types
- **Connection Validation**: Verify socket is alive
- **Data Flow Check**: Ensure data is being received
- **Response Time**: Measure latency
- **Error Rate**: Monitor failure frequency

---

## Integration

### Component Integration

#### Seamless Coordination
```python
class DataPipeline:
    def _setup_integration(self):
        # Connect gap detector to backfill
        self.gap_detector.set_backfill_manager(self.backfill_manager)
        
        # Add alert callbacks
        self.gap_detector.add_alert_callback(self._on_gap_detected)
        self.reconnection_manager.add_alert_callback(self._on_connection_event)
        
        # Register source with reconnection manager
        self.reconnection_manager.register_source(
            "okx",
            self._connect_okx,
            health_check_callback=self._health_check_okx
        )
```

#### Event Flow
```
Market Data → Gap Detector → Alert → Backfill Request
     ↓              ↓              ↓
Connection Issue → Reconnection → Circuit Breaker → Health Check
```

### Data Processing Pipeline

#### Multi-Stage Processing
```python
async def _process_market_data(self, market_data: MarketData):
    # 1. Buffer management
    if self.config.enable_data_buffering:
        self.data_buffer.append(market_data)
        if len(self.data_buffer) > self.config.buffer_size:
            self.data_buffer.pop(0)
    
    # 2. Gap detection
    gap_info = await self.gap_detector.process_market_data(market_data)
    
    # 3. Queue for processing
    await self.processing_queue.put(market_data)
```

#### Worker Processing
```python
async def _data_processing_worker(self, worker_name: str):
    while self._running:
        market_data = await self.processing_queue.get()
        
        # Validate and enrich
        processed_data = await self._validate_and_enrich_data(market_data)
        
        # Output to consumers
        await self.output_queue.put(processed_data)
        
        # Update statistics
        self.stats.total_messages_processed += 1
```

---

## Configuration

### Complete Configuration Example

```python
# OKX Configuration
okx_config = OKXConfig(
    symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    data_types=["ticker", "orderbook", "trades"],
    enable_auto_download=True,
    enable_backfill=True,
    download_timeframes=["1m", "5m", "15m", "1h", "1d"],
    storage_path="data/okx",
    enable_compression=True
)

# Gap Detection Configuration
gap_config = GapDetectionConfig(
    ticker_gap_threshold=2.0,        # 2 seconds
    trade_gap_threshold=0.5,        # 500ms
    orderbook_gap_threshold=1.0,     # 1 second
    ohlcv_gap_threshold=30.0,        # 30 seconds
    max_price_jump_pct=5.0,         # 5% price jump
    max_volume_spike_pct=500.0,     # 500% volume spike
    enable_real_time_detection=True,
    enable_historical_validation=True,
    auto_backfill=True,
    backfill_priority=2,
    detection_window_size=1000
)

# Reconnection Configuration
reconnect_config = ReconnectionConfig(
    max_reconnect_attempts=15,
    initial_delay=0.5,              # 500ms
    max_delay=180.0,               # 3 minutes
    backoff_multiplier=1.5,
    jitter=True,
    enable_circuit_breaker=True,
    failure_threshold=3,            # Faster circuit opening
    recovery_timeout=30.0,          # 30 seconds recovery
    enable_health_checks=True,
    health_check_interval=15.0,     # 15 seconds
    health_check_timeout=5.0,
    connection_timeout=15.0,
    read_timeout=30.0
)

# Backfill Configuration
backfill_config = BackfillConfig(
    max_concurrent_jobs=5,
    job_timeout=1800,               # 30 minutes
    retry_delay=30,                 # 30 seconds
    enable_scheduling=True,
    schedule_interval=1800,         # 30 minutes
    gap_threshold=180,              # 3 minutes
    batch_size=500,
    enable_compression=True
)

# Pipeline Configuration
pipeline_config = DataPipelineConfig(
    okx_config=okx_config,
    gap_detection_config=gap_config,
    reconnection_config=reconnect_config,
    backfill_config=backfill_config,
    enable_monitoring=True,
    monitoring_interval=30.0,      # 30 seconds
    enable_alerts=True,
    enable_data_validation=True,
    enable_data_buffering=True,
    buffer_size=2000,
    max_concurrent_processing=15,
    processing_timeout=15.0
)
```

### Environment-Specific Configurations

#### Development Environment
```python
dev_config = DataPipelineConfig(
    okx_config=OKXConfig(sandbox=True),
    gap_detection_config=GapDetectionConfig(
        ticker_gap_threshold=10.0,     # More lenient
        enable_real_time_detection=False
    ),
    reconnection_config=ReconnectionConfig(
        max_reconnect_attempts=5,
        initial_delay=1.0
    ),
    enable_monitoring=False,
    enable_alerts=False
)
```

#### Production Environment
```python
prod_config = DataPipelineConfig(
    okx_config=OKXConfig(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT"],
        enable_auto_download=True,
        enable_backfill=True
    ),
    gap_detection_config=GapDetectionConfig(
        ticker_gap_threshold=1.0,     # Strict thresholds
        max_price_jump_pct=3.0,
        auto_backfill=True
    ),
    reconnection_config=ReconnectionConfig(
        max_reconnect_attempts=20,
        initial_delay=0.1,
        enable_circuit_breaker=True,
        failure_threshold=5
    ),
    enable_monitoring=True,
    enable_alerts=True,
    max_concurrent_processing=20
)
```

---

## Usage Examples

### Basic Usage

```python
import asyncio
from its_project.data_layer import create_data_pipeline

async def main():
    # Create pipeline with default configuration
    pipeline = create_data_pipeline(
        symbols=["BTC-USDT", "ETH-USDT"],
        enable_gap_detection=True,
        enable_reconnection=True,
        enable_backfill=True
    )
    
    # Start pipeline
    await pipeline.start()
    
    try:
        # Process data
        async for market_data in pipeline.get_data_stream():
            print(f"Received: {market_data.symbol} {market_data.type}")
            
            # Get pipeline status periodically
            if market_data.timestamp_ms % 60000 < 1000:  # Every minute
                status = pipeline.get_pipeline_status()
                print(f"Status: {status['statistics']['total_messages_processed']} processed")
    
    finally:
        # Stop pipeline
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
import asyncio
from its_project.data_layer import (
    DataPipeline, DataPipelineConfig,
    OKXConfig, GapDetectionConfig, ReconnectionConfig
)

async def advanced_example():
    # Custom configurations
    okx_config = OKXConfig(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        data_types=["ticker", "orderbook", "trades"],
        enable_auto_download=True,
        download_timeframes=["1m", "5m", "15m", "1h"]
    )
    
    gap_config = GapDetectionConfig(
        ticker_gap_threshold=2.0,
        max_price_jump_pct=5.0,
        auto_backfill=True,
        alert_callback=lambda gap: print(f"GAP ALERT: {gap.symbol}")
    )
    
    reconnect_config = ReconnectionConfig(
        max_reconnect_attempts=15,
        enable_circuit_breaker=True,
        failure_threshold=3,
        alert_callback=lambda event: print(f"CONN ALERT: {event.source}")
    )
    
    # Create pipeline
    pipeline_config = DataPipelineConfig(
        okx_config=okx_config,
        gap_detection_config=gap_config,
        reconnection_config=reconnect_config,
        enable_monitoring=True,
        max_concurrent_processing=10
    )
    
    pipeline = DataPipeline(pipeline_config)
    
    await pipeline.start()
    
    # Monitor pipeline
    async def monitor_pipeline():
        while True:
            status = pipeline.get_pipeline_status()
            print(f"Pipeline uptime: {status['pipeline']['uptime_percentage']:.1f}%")
            print(f"Messages processed: {status['statistics']['total_messages_processed']}")
            print(f"Gaps detected: {status['statistics']['total_gaps_detected']}")
            print(f"Reconnections: {status['statistics']['total_reconnections']}")
            await asyncio.sleep(60)
    
    # Run monitoring alongside data processing
    monitor_task = asyncio.create_task(monitor_pipeline())
    
    try:
        async for market_data in pipeline.get_data_stream():
            # Process data
            if market_data.type == MarketDataType.TICKER:
                price = market_data.data.get('last_price', 0)
                print(f"Ticker: {market_data.symbol} = ${price}")
    
    finally:
        monitor_task.cancel()
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

### Integration with Trading System

```python
class TradingSystem:
    def __init__(self):
        # Create data pipeline
        self.pipeline = create_data_pipeline(
            symbols=["BTC-USDT", "ETH-USDT"],
            enable_gap_detection=True,
            enable_reconnection=True
        )
        
        # Trading components
        self.strategy = None
        self.executor = None
        
        # Data cache
        self.latest_data = {}
    
    async def start(self):
        # Start pipeline
        await self.pipeline.start()
        
        # Start data processing
        asyncio.create_task(self._process_market_data())
        
        # Start monitoring
        asyncio.create_task(self._monitor_system())
    
    async def _process_market_data(self):
        async for market_data in self.pipeline.get_data_stream():
            # Cache latest data
            self.latest_data[market_data.symbol] = market_data
            
            # Trigger strategy
            if market_data.type == MarketDataType.TICKER:
                await self._update_strategy(market_data)
    
    async def _update_strategy(self, market_data):
        # Update strategy with new data
        if self.strategy:
            signal = await self.strategy.analyze(market_data)
            if signal:
                await self._execute_signal(signal)
    
    async def _monitor_system(self):
        while True:
            status = self.pipeline.get_pipeline_status()
            
            # Check for issues
            if status['pipeline']['uptime_percentage'] < 95:
                await self._handle_system_issue("Low uptime")
            
            if status['statistics']['error_count'] > 10:
                await self._handle_system_issue("High error rate")
            
            await asyncio.sleep(30)
    
    async def _handle_system_issue(self, issue_type: str):
        print(f"System issue detected: {issue_type}")
        # Implement appropriate response
        # - Reduce trading activity
        # - Send alerts
        # - Switch to backup data sources
    
    async def stop(self):
        await self.pipeline.stop()
```

---

## Monitoring and Statistics

### Real-time Monitoring

#### Pipeline Status
```python
status = pipeline.get_pipeline_status()
# {
#     'pipeline': {
#         'running': True,
#         'uptime_percentage': 98.7,
#         'start_time': '2024-01-01T10:00:00',
#         'last_message_time': '2024-01-01T11:30:00'
#     },
#     'components': {
#         'okx_source': 'connected',
#         'gap_detector': 'running',
#         'reconnection_manager': 'running',
#         'backfill_manager': 'running'
#     },
#     'statistics': {
#         'total_messages_processed': 150000,
#         'total_gaps_detected': 25,
#         'total_reconnections': 3,
#         'total_backfills_triggered': 15,
#         'average_processing_rate': 45.2,  # messages per second
#         'error_count': 2,
#         'queue_sizes': {
#             'processing': 150,
#             'output': 25
#         }
#     }
# }
```

#### Component-Specific Statistics

##### Gap Detection Statistics
```python
gap_stats = gap_detector.get_gap_statistics()
# {
#     'total_gaps_detected': 25,
#     'gaps_by_type': {
#         'missing': 15,
#         'price_jump': 8,
#         'volume_spike': 2
#     },
#     'gaps_by_symbol': {
#         'BTC-USDT': 18,
#         'ETH-USDT': 7
#     },
#     'gaps_by_severity': {
#         'low': 10,
#         'medium': 12,
#         'high': 3
#     },
#     'average_gap_duration': 12.5,
#     'longest_gap_duration': 180.0,
#     'detection_rate': 0.42,  # gaps per hour
#     'last_gap_time': '2024-01-01T11:25:00'
# }
```

##### Connection Statistics
```python
connection_status = reconnection_manager.get_all_status()
# {
#     'okx': {
#         'source': 'okx',
#         'state': 'connected',
#         'is_connected': True,
#         'is_reconnecting': False,
#         'circuit_breaker_state': 'closed',
#         'stats': {
#             'total_connections': 8,
#             'total_disconnections': 2,
#             'total_reconnections': 2,
#             'total_failures': 2,
#             'average_connection_time': 2.3,
#             'longest_connection_time': 5.1,
#             'current_uptime': 3600.0,
#             'last_connection_time': '2024-01-01T10:30:00'
#         },
#         'last_health_check': '2024-01-01T11:28:00'
#     }
# }
```

### Performance Metrics

#### Processing Performance
```python
# Calculate processing rates
elapsed_time = (datetime.now() - pipeline.stats.start_time).total_seconds()
processing_rate = pipeline.stats.total_messages_processed / elapsed_time

# Queue utilization
processing_utilization = pipeline.processing_queue.qsize() / pipeline.config.buffer_size
output_utilization = pipeline.output_queue.qsize() / pipeline.config.buffer_size

# Error rate
error_rate = pipeline.stats.error_count / pipeline.stats.total_messages_processed
```

#### Memory Usage
```python
# Buffer status
buffer_status = pipeline.get_buffer_status()
# {
#     'buffer_size': 1500,
#     'max_buffer_size': 2000,
#     'buffer_utilization': 0.75,
#     'oldest_message_age': 45.2  # seconds
# }

# Gap detector memory
gap_memory_usage = len(gap_detector.detected_gaps)
connection_memory_usage = len(reconnection_manager.connection_events)
```

### Alert System

#### Alert Types
```python
# Gap detection alerts
async def on_gap_detected(gap_info: GapInfo):
    if gap_info.severity in ["high", "critical"]:
        send_alert(
            alert_type="critical_gap",
            message=f"Critical gap: {gap_info.symbol} {gap_info.duration_ms}ms",
            severity=gap_info.severity
        )

# Connection alerts
async def on_connection_event(event: ConnectionEvent):
    if event.state.value in ["failed", "circuit_open"]:
        send_alert(
            alert_type="connection_issue",
            message=f"Connection issue: {event.source} {event.state.value}",
            severity="high"
        )

# Pipeline alerts
async def check_pipeline_health():
    status = pipeline.get_pipeline_status()
    
    # Stale data alert
    if status['pipeline']['uptime_percentage'] < 90:
        send_alert(
            alert_type="pipeline_stale",
            message=f"Pipeline uptime: {status['pipeline']['uptime_percentage']:.1f}%",
            severity="medium"
        )
    
    # High error rate alert
    error_rate = status['statistics']['error_count'] / max(status['statistics']['total_messages_processed'], 1)
    if error_rate > 0.05:  # 5% error rate
        send_alert(
            alert_type="high_error_rate",
            message=f"Error rate: {error_rate:.2%}",
            severity="medium"
        )
```

---

## API Reference

### DataPipeline

#### Constructor
```python
DataPipeline(config: DataPipelineConfig)
```

#### Methods

##### start()
```python
async def start() -> None
```
Start the data pipeline and all components.

##### stop()
```python
async def stop() -> None
```
Stop the data pipeline and cleanup resources.

##### get_data_stream()
```python
async def get_data_stream() -> AsyncIterator[MarketData]
```
Get processed market data stream.

**Yields:**
- `MarketData`: Processed and enriched market data

##### get_pipeline_status()
```python
def get_pipeline_status() -> Dict[str, Any]
```
Get comprehensive pipeline status.

**Returns:**
- `Dict[str, Any]`: Pipeline status including components and statistics

### GapDetector

#### Constructor
```python
GapDetector(config: GapDetectionConfig)
```

#### Methods

##### process_market_data()
```python
async def process_market_data(market_data: MarketData) -> Optional[GapInfo]
```
Process market data and detect gaps.

**Parameters:**
- `market_data`: Market data to analyze

**Returns:**
- `Optional[GapInfo]`: Detected gap information

##### get_gap_statistics()
```python
def get_gap_statistics() -> GapStatistics
```
Get gap detection statistics.

**Returns:**
- `GapStatistics`: Comprehensive gap statistics

### ReconnectionManager

#### Constructor
```python
ReconnectionManager(config: ReconnectionConfig)
```

#### Methods

##### register_source()
```python
def register_source(
    source_name: str,
    connect_callback: Callable,
    disconnect_callback: Optional[Callable] = None,
    health_check_callback: Optional[Callable] = None
) -> None
```
Register a data source for reconnection management.

**Parameters:**
- `source_name`: Unique identifier for the source
- `connect_callback`: Function to connect the source
- `disconnect_callback`: Optional function to disconnect the source
- `health_check_callback`: Optional function to check source health

##### connect_source()
```python
async def connect_source(source_name: str) -> bool
```
Connect a registered data source.

**Parameters:**
- `source_name`: Name of the source to connect

**Returns:**
- `bool`: True if connection successful

##### get_connection_status()
```python
def get_connection_status(source_name: str) -> Dict[str, Any]
```
Get connection status for a specific source.

**Returns:**
- `Dict[str, Any]`: Connection status and statistics

---

## Best Practices

### 1. Configuration Management

#### Environment-Specific Settings
```python
import os
from typing import Dict

def get_config_for_environment(env: str) -> DataPipelineConfig:
    configs = {
        'development': get_dev_config(),
        'staging': get_staging_config(),
        'production': get_prod_config()
    }
    
    return configs.get(env, get_dev_config())

def get_prod_config() -> DataPipelineConfig:
    return DataPipelineConfig(
        okx_config=OKXConfig(
            api_key=os.getenv("OKX_API_KEY"),
            secret=os.getenv("OKX_SECRET"),
            passphrase=os.getenv("OKX_PASSPHRASE"),
            sandbox=False
        ),
        gap_detection_config=GapDetectionConfig(
            ticker_gap_threshold=1.0,  # Strict for production
            auto_backfill=True
        ),
        reconnection_config=ReconnectionConfig(
            max_reconnect_attempts=20,
            enable_circuit_breaker=True
        ),
        enable_monitoring=True,
        enable_alerts=True
    )
```

#### Configuration Validation
```python
def validate_config(config: DataPipelineConfig) -> bool:
    # Validate OKX configuration
    if not config.okx_config.symbols:
        raise ValueError("No symbols configured")
    
    # Validate gap detection thresholds
    if config.gap_detection_config.ticker_gap_threshold <= 0:
        raise ValueError("Invalid gap threshold")
    
    # Validate reconnection settings
    if config.reconnection_config.max_reconnect_attempts <= 0:
        raise ValueError("Invalid reconnection attempts")
    
    return True
```

### 2. Error Handling

#### Comprehensive Error Handling
```python
async def safe_pipeline_operation(pipeline: DataPipeline):
    try:
        await pipeline.start()
        
        async for market_data in pipeline.get_data_stream():
            try:
                # Process data
                await process_data(market_data)
                
            except Exception as e:
                logger.error(f"Data processing error: {e}")
                # Continue processing other data
                continue
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Implement appropriate error response
        await handle_pipeline_error(e)
        
    finally:
        await pipeline.stop()

async def handle_pipeline_error(error: Exception):
    # Classify error type
    if "connection" in str(error).lower():
        # Connection-related error
        await notify_admins("Connection issue detected")
    elif "gap" in str(error).lower():
        # Gap detection error
        await notify_monitoring("Gap detection issue")
    else:
        # General error
        await notify_admins(f"Pipeline error: {error}")
```

#### Graceful Degradation
```python
class ResilientPipeline:
    def __init__(self, config: DataPipelineConfig):
        self.primary_pipeline = DataPipeline(config)
        self.backup_pipeline = None  # Fallback pipeline
        self.circuit_breaker = CircuitBreaker()
    
    async def get_data_stream(self):
        while True:
            try:
                if self.circuit_breaker.can_execute():
                    async for data in self.primary_pipeline.get_data_stream():
                        self.circuit_breaker.call_success()
                        yield data
                else:
                    # Use backup pipeline
                    async for data in self.backup_pipeline.get_data_stream():
                        yield data
                        
            except Exception as e:
                self.circuit_breaker.call_failure()
                logger.error(f"Pipeline error: {e}")
                await asyncio.sleep(10)  # Wait before retry
```

### 3. Performance Optimization

#### Concurrency Configuration
```python
def optimize_config_for_hardware(cpu_count: int, memory_gb: int) -> DataPipelineConfig:
    # Calculate optimal worker count
    optimal_workers = min(cpu_count - 1, 20)  # Leave one CPU for system
    
    # Calculate buffer size based on memory
    buffer_size = min(memory_gb * 100, 5000)  # 100 items per GB
    
    return DataPipelineConfig(
        max_concurrent_processing=optimal_workers,
        buffer_size=buffer_size,
        processing_timeout=30.0
    )
```

#### Memory Management
```python
class MemoryEfficientPipeline:
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.cleanup_interval = 300  # 5 minutes
    
    async def start(self):
        # Start memory monitoring
        asyncio.create_task(self._memory_monitoring_loop())
        
        # Start pipeline
        await self.pipeline.start()
    
    async def _memory_monitoring_loop(self):
        while True:
            memory_usage = self.memory_monitor.get_usage()
            
            if memory_usage > 0.8:  # 80% memory usage
                # Reduce buffer sizes
                await self._reduce_memory_usage()
            
            await asyncio.sleep(self.cleanup_interval)
    
    async def _reduce_memory_usage(self):
        # Clear old data
        self.pipeline.data_buffer.clear()
        
        # Reduce buffer size temporarily
        original_size = self.config.buffer_size
        self.config.buffer_size = original_size // 2
        
        # Restore after cleanup
        await asyncio.sleep(60)
        self.config.buffer_size = original_size
```

### 4. Monitoring and Alerting

#### Comprehensive Monitoring
```python
class PipelineMonitor:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
    
    async def start_monitoring(self):
        # Start various monitoring tasks
        asyncio.create_task(self._performance_monitoring())
        asyncio.create_task(self._health_monitoring())
        asyncio.create_task(self._error_monitoring())
        asyncio.create_task(self._business_metrics())
    
    async def _performance_monitoring(self):
        while True:
            status = self.pipeline.get_pipeline_status()
            
            # Collect performance metrics
            self.metrics_collector.record_metric(
                "processing_rate",
                status['statistics']['average_processing_rate']
            )
            
            self.metrics_collector.record_metric(
                "queue_utilization",
                status['statistics']['queue_sizes']['processing'] / self.pipeline.config.buffer_size
            )
            
            await asyncio.sleep(10)
    
    async def _health_monitoring(self):
        while True:
            status = self.pipeline.get_pipeline_status()
            
            # Check component health
            for component, state in status['components'].items():
                if state != "running" and state != "connected":
                    await self.alert_manager.send_alert(
                        f"Component {component} unhealthy: {state}",
                        severity="high"
                    )
            
            await asyncio.sleep(30)
```

#### Custom Alert Handlers
```python
class CustomAlertHandler:
    def __init__(self, webhook_url: str, email_config: Dict):
        self.webhook_url = webhook_url
        self.email_config = email_config
    
    async def handle_gap_alert(self, gap_info: GapInfo):
        # Format alert message
        message = f"Gap detected: {gap_info.symbol} {gap_info.gap_type} {gap_info.duration_ms}ms"
        
        # Send to monitoring system
        await self.send_webhook_alert("gap_detected", message, gap_info.severity)
        
        # Send email for critical gaps
        if gap_info.severity in ["high", "critical"]:
            await self.send_email_alert(
                f"Critical Gap: {gap_info.symbol}",
                f"A critical gap has been detected:\n\n{message}"
            )
    
    async def handle_connection_alert(self, event: ConnectionEvent):
        message = f"Connection issue: {event.source} {event.state.value}"
        
        if event.state.value in ["failed", "circuit_open"]:
            await self.send_webhook_alert("connection_issue", message, "high")
            await self.send_email_alert("Connection Failure", message)
```

### 5. Testing and Validation

#### Unit Testing
```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_gap_detection():
    # Create gap detector with test config
    config = GapDetectionConfig(ticker_gap_threshold=1.0)
    detector = GapDetector(config)
    
    # Create test market data
    market_data = MarketData(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        symbol="BTC-USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={'last_price': 50000.0}
    )
    
    # Process data
    gap_info = await detector.process_market_data(market_data)
    
    # Assertions
    assert gap_info is None  # No gap expected
    
    # Test gap detection
    future_time = int((datetime.now() + timedelta(seconds=5)).timestamp() * 1000)
    gap_market_data = MarketData(
        timestamp_ms=future_time,
        symbol="BTC-USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={'last_price': 50000.0}
    )
    
    gap_info = await detector.process_market_data(gap_market_data)
    assert gap_info is not None
    assert gap_info.gap_type == "missing"

@pytest.mark.asyncio
async def test_reconnection():
    # Create reconnection manager
    config = ReconnectionConfig(max_reconnect_attempts=3, initial_delay=0.1)
    manager = ReconnectionManager(config)
    
    # Mock connection callback
    connect_callback = AsyncMock()
    connect_callback.side_effect = [
        Exception("Connection failed"),
        Exception("Connection failed"),
        None  # Success on third attempt
    ]
    
    # Register source
    manager.register_source("test_source", connect_callback)
    
    # Start manager
    await manager.start()
    
    # Connect source (should trigger reconnection)
    success = await manager.connect_source("test_source")
    
    # Assertions
    assert success is True
    assert connect_callback.call_count == 3
    
    await manager.stop()
```

#### Integration Testing
```python
@pytest.mark.asyncio
async def test_pipeline_integration():
    # Create test configuration
    config = DataPipelineConfig(
        okx_config=OKXConfig(symbols=["BTC-USDT"], sandbox=True),
        gap_detection_config=GapDetectionConfig(ticker_gap_threshold=1.0),
        reconnection_config=ReconnectionConfig(max_reconnect_attempts=3)
    )
    
    # Create pipeline
    pipeline = DataPipeline(config)
    
    # Start pipeline
    await pipeline.start()
    
    # Wait for startup
    await asyncio.sleep(2)
    
    # Check status
    status = pipeline.get_pipeline_status()
    assert status['pipeline']['running'] is True
    
    # Test data processing
    data_count = 0
    async for market_data in pipeline.get_data_stream():
        assert market_data.symbol == "BTC-USDT"
        data_count += 1
        
        if data_count >= 10:  # Test with 10 messages
            break
    
    # Stop pipeline
    await pipeline.stop()
    
    # Verify cleanup
    status = pipeline.get_pipeline_status()
    assert status['pipeline']['running'] is False
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Problems

**Symptoms:**
- Frequent reconnections
- Circuit breaker opening
- Connection timeouts

**Solutions:**
```python
# Increase connection timeout
reconnect_config = ReconnectionConfig(
    connection_timeout=60.0,  # Increase from 30s
    read_timeout=120.0,       # Increase from 60s
    max_reconnect_attempts=20
)

# Reduce failure threshold
reconnect_config.failure_threshold = 10  # Increase from 5

# Add more jitter to prevent thundering herd
reconnect_config.jitter = True
```

#### 2. Gap Detection False Positives

**Symptoms:**
- Too many gap alerts
- High false positive rate
- Alert fatigue

**Solutions:**
```python
# Adjust thresholds
gap_config = GapDetectionConfig(
    ticker_gap_threshold=10.0,     # Increase from 5s
    trade_gap_threshold=5.0,       # Increase from 1s
    max_price_jump_pct=15.0,      # Increase from 10%
    max_volume_spike_pct=2000.0    # Increase from 1000%
)

# Add minimum gap duration
gap_config.min_gap_duration = 2.0  # Only report gaps > 2s

# Reduce sensitivity
gap_config.detection_window_size = 500  # Reduce from 1000
```

#### 3. Performance Issues

**Symptoms:**
- High memory usage
- Slow processing
- Queue backlog

**Solutions:**
```python
# Reduce buffer sizes
pipeline_config = DataPipelineConfig(
    buffer_size=1000,  # Reduce from 2000
    max_concurrent_processing=5  # Reduce from 10
)

# Increase processing timeout
pipeline_config.processing_timeout = 60.0  # Increase from 30s

# Enable data compression
okx_config.enable_compression = True
backfill_config.enable_compression = True
```

#### 4. Memory Leaks

**Symptoms:**
- Memory usage increasing over time
- Out of memory errors
- Slow performance

**Solutions:**
```python
# Enable automatic cleanup
gap_config.enable_auto_cleanup = True
gap_config.cleanup_interval = 300  # 5 minutes

# Reduce buffer retention
pipeline_config.data_retention_minutes = 60  # Keep data for 1 hour

# Monitor memory usage
import psutil

def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent}%")
        # Trigger cleanup
        pipeline.cleanup_old_data()
```

### Debug Mode

#### Enable Detailed Logging
```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific component logging
logging.getLogger("its_project.data_layer.gap_detector").setLevel(logging.DEBUG)
logging.getLogger("its_project.data_layer.reconnection_manager").setLevel(logging.DEBUG)
logging.getLogger("its_project.data_layer.data_pipeline").setLevel(logging.DEBUG)
```

#### Component-Specific Debugging

##### Gap Detection Debugging
```python
# Enable detailed gap detection logging
gap_config = GapDetectionConfig(
    enable_debug_logging=True,
    log_all_data_points=True,
    log_gap_analysis=True
)

# Add custom gap analysis callback
def debug_gap_analysis(gap_info: GapInfo):
    logger.debug(f"Gap analysis: {gap_info}")
    logger.debug(f"Recent data points: {gap_detector.data_buffer[gap_info.symbol]}")

gap_detector.add_alert_callback(debug_gap_analysis)
```

##### Reconnection Debugging
```python
# Enable detailed connection logging
reconnect_config = ReconnectionConfig(
    enable_debug_logging=True,
    log_all_attempts=True,
    log_circuit_breaker_state=True
)

# Add custom connection event handler
def debug_connection_events(event: ConnectionEvent):
    logger.debug(f"Connection event: {event}")
    if event.error:
        logger.debug(f"Connection error details: {event.error}")

reconnection_manager.add_alert_callback(debug_connection_events)
```

### Performance Monitoring

#### Real-time Performance Metrics
```python
async def monitor_performance():
    while True:
        status = pipeline.get_pipeline_status()
        
        # Calculate performance metrics
        processing_rate = status['statistics']['average_processing_rate']
        queue_utilization = status['statistics']['queue_sizes']['processing'] / pipeline.config.buffer_size
        error_rate = status['statistics']['error_count'] / max(status['statistics']['total_messages_processed'], 1)
        
        # Log performance
        logger.info(f"Performance: {processing_rate:.1f} msg/s, "
                   f"Queue: {queue_utilization:.1%}, "
                   f"Errors: {error_rate:.2%}")
        
        # Alert on performance issues
        if processing_rate < 10:  # Low processing rate
            logger.warning("Low processing rate detected")
        
        if queue_utilization > 0.8:  # High queue utilization
            logger.warning("High queue utilization detected")
        
        if error_rate > 0.05:  # High error rate
            logger.warning("High error rate detected")
        
        await asyncio.sleep(30)
```

#### Resource Usage Monitoring
```python
import psutil
import threading

def monitor_resources():
    def resource_monitor():
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Log resource usage
            logger.info(f"Resources - CPU: {cpu_percent}%, "
                       f"Memory: {memory_percent}%, "
                       f"Disk: {disk_percent}%")
            
            # Alert on resource issues
            if cpu_percent > 90:
                logger.error("High CPU usage detected")
            
            if memory_percent > 90:
                logger.error("High memory usage detected")
            
            if disk_percent > 90:
                logger.error("High disk usage detected")
            
            time.sleep(60)
    
    # Run in separate thread
    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()
```

This comprehensive documentation provides complete coverage of the data pipeline system, including gap detection, reconnection management, integration patterns, and operational guidance for production deployment.
