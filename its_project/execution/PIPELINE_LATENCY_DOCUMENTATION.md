# Pipeline Latency Tracking - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Visualization](#visualization)
7. [Data Export](#data-export)
8. [Integration Examples](#integration-examples)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The Pipeline Latency Tracking system provides comprehensive monitoring and analysis of pipeline performance, enabling real-time tracking of processing times, error rates, and system health.

### Key Features

- ✅ **Real-time latency measurement** with millisecond precision
- ✅ **Statistical analysis** with percentiles and distributions
- ✅ **Error rate monitoring** with configurable thresholds
- ✅ **Alert system** with customizable warning/critical levels
- ✅ **Historical data management** with automatic cleanup
- ✅ **Interactive visualization** with multiple chart types
- ✅ **Data persistence** with multiple export formats
- ✅ **Performance optimization** with caching and batch operations

### Use Cases

- **Trading Systems**: Monitor order processing, execution latency
- **Data Processing**: Track ETL pipeline performance
- **API Services**: Measure endpoint response times
- **Machine Learning**: Monitor model inference latency
- **Microservices**: Track inter-service communication delays

---

## Core Components

### 1. LatencyPoint (`LatencyPoint`)

Single latency measurement with metadata.

```python
@dataclass
class LatencyPoint:
    timestamp: datetime      # When measurement was taken
    pipeline_stage: str       # Pipeline stage name
    latency_ms: float         # Latency in milliseconds
    metadata: Dict[str, Any]  # Additional context data
    success: bool           # Whether operation succeeded
    error_message: Optional[str]  # Error details if failed
```

### 2. LatencyStatistics (`LatencyStatistics`)

Comprehensive statistical analysis for latency data.

```python
@dataclass
class LatencyStatistics:
    stage: str              # Pipeline stage name
    count: int              # Total measurements
    mean_ms: float          # Average latency
    median_ms: float        # Median latency
    std_ms: float           # Standard deviation
    min_ms: float          # Minimum latency
    max_ms: float          # Maximum latency
    p50_ms: float          # 50th percentile
    p95_ms: float          # 95th percentile
    p99_ms: float          # 99th percentile
    error_rate: float       # Error rate (0.0-1.0)
    last_updated: datetime   # Last calculation time
```

### 3. PipelineLatencyTracker (`PipelineLatencyTracker`)

Main tracking system with real-time monitoring.

```python
class PipelineLatencyTracker:
    def __init__(
        self,
        pipeline_name: str,
        config: Optional[PipelineLatencyConfig] = None,
        persistence_manager: Optional[Any] = None
    )
```

### 4. LatencyContextManager (`LatencyContextManager`)

Context manager for automatic latency measurement.

```python
class LatencyContextManager:
    def __init__(
        self,
        tracker: PipelineLatencyTracker,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None
    )
```

---

## Basic Usage

### Creating a Latency Tracker

```python
from its_project.execution import (
    PipelineLatencyTracker, create_latency_tracker,
    PipelineLatencyConfig, create_latency_config
)

# Create with default configuration
tracker = create_latency_tracker("trading_pipeline")

# Create with custom configuration
config = create_latency_config(
    max_history_points=5000,
    warning_threshold_ms=50.0,
    critical_threshold_ms=200.0
)
tracker = PipelineLatencyTracker("data_pipeline", config)
```

### Recording Latency Measurements

```python
# Manual recording
tracker.record_latency(
    stage="order_processing",
    latency_ms=45.2,
    metadata={"order_id": "12345", "symbol": "BTC/USDT"},
    success=True
)

# Using context manager (recommended)
with LatencyContextManager(tracker, "market_data_fetch") as ctx:
    # Your pipeline code here
    market_data = fetch_market_data("BTC/USDT")
    ctx.set_metadata({"symbol": "BTC/USDT", "source": "exchange"})
    
# Latency automatically recorded when context exits
```

### Error Handling

```python
# Record failed operation
tracker.record_latency(
    stage="api_call",
    latency_ms=5000.0,  # 5 second timeout
    success=False,
    error_message="Connection timeout",
    metadata={"endpoint": "/api/v1/orders"}
)

# Using context manager with error
try:
    with LatencyContextManager(tracker, "database_query") as ctx:
        result = execute_complex_query()
except Exception as e:
    # Error automatically recorded with latency
    logger.error(f"Query failed: {e}")
```

### Getting Statistics

```python
# Get statistics for all stages
all_stats = tracker.get_statistics()
for stage, stats in all_stats.items():
    print(f"{stage}:")
    print(f"  Mean: {stats.mean_ms:.2f}ms")
    print(f"  P95: {stats.p95_ms:.2f}ms")
    print(f"  Error Rate: {stats.error_rate:.1%}")

# Get statistics for specific stage
order_stats = tracker.get_statistics("order_processing")
print(f"Order Processing P95: {order_stats.p95_ms:.2f}ms")
```

### Data Export

```python
# Export to DataFrame
df = tracker.get_latency_dataframe()
print(df.head())

# Export by stage
api_df = tracker.get_latency_dataframe(stage="api_calls")

# Export to different formats
tracker.export_data(format="csv", include_statistics=True)
tracker.export_data(format="json", save_path="latency_analysis.json")
tracker.export_data(format="excel", save_path="performance_report.xlsx")
```

---

## Advanced Features

### Real-time Monitoring

```python
# Start monitoring
tracker.start_monitoring()

# Monitoring runs in background
# - Automatic statistics calculation
# - Alert checking
# - Data persistence
# - Memory cleanup

# Stop monitoring
tracker.stop_monitoring()
```

### Alert System

```python
# Configure alert thresholds
config = PipelineLatencyConfig(
    alert_thresholds={
        'warning_ms': 100.0,      # Warning at 100ms
        'critical_ms': 500.0,     # Critical at 500ms
        'error_rate_warning': 0.05,  # 5% error rate warning
        'error_rate_critical': 0.1   # 10% error rate critical
    }
)

# Add custom alert callback
def handle_alert(alert):
    if alert['severity'] == 'critical':
        # Send SMS/email
        send_critical_notification(alert)
    else:
        # Log for monitoring
        logger.warning(f"Performance alert: {alert['message']}")

tracker.add_alert_callback(handle_alert)
```

### Performance Optimization

```python
# Configure for high-frequency tracking
config = PipelineLatencyConfig(
    max_history_points=50000,    # Large history
    statistics_window=5000,        # Larger window
    auto_cleanup=True,             # Automatic cleanup
    enable_persistence=True,        # Background saving
    persistence_interval=30         # Save every 30 seconds
)

# Enable caching for better performance
tracker = PipelineLatencyTracker("high_freq_pipeline", config)
```

---

## Monitoring & Alerts

### Alert Types

#### Latency Alerts

```python
# Warning latency alert
{
    'type': 'warning_latency',
    'stage': 'order_processing',
    'value': 150.0,
    'threshold': 100.0,
    'message': 'High latency: 150.00ms >= 100.00ms',
    'timestamp': datetime.now(),
    'severity': 'warning'
}

# Critical latency alert
{
    'type': 'critical_latency',
    'stage': 'market_data_fetch',
    'value': 750.0,
    'threshold': 500.0,
    'message': 'Critical latency: 750.00ms >= 500.00ms',
    'timestamp': datetime.now(),
    'severity': 'critical'
}
```

#### Error Rate Alerts

```python
# High error rate warning
{
    'type': 'warning_error_rate',
    'stage': 'api_calls',
    'value': 0.08,  # 8%
    'threshold': 0.05,  # 5%
    'message': 'High error rate: 8.0% >= 5.0%',
    'timestamp': datetime.now(),
    'severity': 'warning'
}

# Critical error rate alert
{
    'type': 'critical_error_rate',
    'stage': 'database_operations',
    'value': 0.15,  # 15%
    'threshold': 0.10,  # 10%
    'message': 'Critical error rate: 15.0% >= 10.0%',
    'timestamp': datetime.now(),
    'severity': 'critical'
}
```

### Custom Alert Handling

```python
# Email alert handler
def email_alert_handler(alert):
    import smtplib
    from email.mime.text import MIMEText
    
    subject = f"Pipeline Alert: {alert['severity'].upper()}"
    body = f"""
    Pipeline: {tracker.pipeline_name}
    Stage: {alert['stage']}
    Alert: {alert['message']}
    Time: {alert['timestamp']}
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@company.com'
    msg['To'] = 'ops@company.com'
    
    # Send email
    server = smtplib.SMTP('smtp.company.com', 587)
    server.starttls()
    server.login('alerts@company.com', 'password')
    server.send_message(msg)
    server.quit()

# Slack alert handler
def slack_alert_handler(alert):
    import requests
    
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    payload = {
        "text": f"🚨 {alert['severity'].upper()} Alert",
        "attachments": [{
            "color": "danger" if alert['severity'] == 'critical' else "warning",
            "fields": [
                {"title": "Pipeline", "value": tracker.pipeline_name},
                {"title": "Stage", "value": alert['stage']},
                {"title": "Message", "value": alert['message']},
                {"title": "Time", "value": alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            ]
        }]
    }
    
    requests.post(webhook_url, json=payload)

# Register handlers
tracker.add_alert_callback(email_alert_handler)
tracker.add_alert_callback(slack_alert_handler)
```

---

## Visualization

### Latency Time Series Chart

```python
# Create interactive latency chart
fig = tracker.create_latency_chart(
    stage="order_processing",
    time_range=timedelta(hours=24),  # Last 24 hours
    save_path="latency_chart.html"
)

# Chart includes:
# - Latency over time
# - Warning/critical threshold lines
# - Hover tooltips with details
# - Zoom and pan capabilities
```

### Performance Dashboard

```python
# Create comprehensive dashboard
fig = tracker.create_performance_dashboard(
    save_path="performance_dashboard.html"
)

# Dashboard includes:
# - Mean latency gauge
# - P95 latency gauge
# - Stage performance bars
# - Error rate indicators
# - Recent trends
# - System health score
```

### Multi-Stage Comparison

```python
# Compare multiple stages
stages = ["order_processing", "market_data", "execution"]
fig = go.Figure()

for stage in stages:
    stats = tracker.get_statistics(stage)
    if stage in stats:
        fig.add_trace(go.Scatter(
            x=stats[stage].last_updated,
            y=stats[stage].p95_ms,
            mode='markers',
            name=f'{stage} P95',
            marker=dict(size=8)
        ))

fig.update_layout(
    title='Stage Performance Comparison',
    xaxis_title='Time',
    yaxis_title='P95 Latency (ms)',
    template='plotly_white'
)
```

---

## Data Export

### CSV Export

```python
# Export with statistics
tracker.export_data(
    format="csv",
    include_statistics=True,
    save_path="pipeline_latency_report.csv"
)

# Creates:
# - latency_data.csv (raw measurements)
# - latency_data_statistics.csv (summary statistics)
```

### JSON Export

```python
# Export complete analysis
tracker.export_data(
    format="json",
    include_statistics=True,
    save_path="pipeline_analysis.json"
)

# JSON structure:
{
    "pipeline_name": "trading_pipeline",
    "export_timestamp": "2024-01-15T10:30:00",
    "latency_points": [...],
    "statistics": {
        "order_processing": {...},
        "market_data": {...}
    },
    "alert_history": [...]
}
```

### Excel Export

```python
# Export with multiple sheets
tracker.export_data(
    format="excel",
    include_statistics=True,
    save_path="performance_report.xlsx"
)

# Creates Excel file with sheets:
# - Latency Data
# - Statistics Summary
# - Alert History
```

### DataFrame Integration

```python
# Get data for pandas analysis
df = tracker.get_latency_dataframe()

# Basic analysis
print("Latency Summary:")
print(df.describe())

# Advanced analysis
# Daily patterns
df['hour'] = df.index.hour
hourly_avg = df.groupby('hour')['latency_ms'].mean()

# Stage comparison
stage_comparison = df.groupby('pipeline_stage')['latency_ms'].agg(['mean', 'std', 'count'])

# Error analysis
error_df = df[~df['success']]
error_rate_by_stage = error_df.groupby('pipeline_stage').size() / df.groupby('pipeline_stage').size()

# Visualization with pandas/matplotlib
import matplotlib.pyplot as plt

# Latency histogram
plt.figure(figsize=(10, 6))
plt.hist(df['latency_ms'], bins=50, alpha=0.7)
plt.title('Latency Distribution')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.savefig('latency_distribution.png')
```

---

## Integration Examples

### Trading System Integration

```python
class TradingSystem:
    def __init__(self):
        # Create latency tracker for trading pipeline
        self.latency_tracker = create_latency_tracker(
            "trading_pipeline",
            config=create_latency_config(
                warning_threshold_ms=50.0,
                critical_threshold_ms=200.0
            )
        )
        
        # Setup alert handlers
        self.latency_tracker.add_alert_callback(self.handle_performance_alert)
        
        # Start monitoring
        self.latency_tracker.start_monitoring()
    
    async def process_order(self, order_data):
        """Process order with latency tracking."""
        # Track order validation
        with LatencyContextManager(self.latency_tracker, "order_validation") as ctx:
            ctx.set_metadata({"order_id": order_data['id']})
            if not self.validate_order(order_data):
                raise ValueError("Invalid order data")
        
        # Track market data fetch
        with LatencyContextManager(self.latency_tracker, "market_data_fetch") as ctx:
            ctx.set_metadata({"symbol": order_data['symbol']})
            market_data = await self.fetch_market_data(order_data['symbol'])
        
        # Track order execution
        with LatencyContextManager(self.latency_tracker, "order_execution") as ctx:
            ctx.set_metadata({
                "order_id": order_data['id'],
                "exchange": order_data['exchange']
            })
            result = await self.execute_order(order_data, market_data)
        
        return result
    
    def handle_performance_alert(self, alert):
        """Handle performance alerts."""
        if alert['severity'] == 'critical':
            # Send immediate notification
            self.send_critical_alert(alert)
            
            # Could implement circuit breaker
            if alert['type'] == 'critical_error_rate':
                self.enable_circuit_breaker()
        
        # Log for monitoring
        logger.warning(f"Performance alert: {alert['message']}")
    
    def get_performance_report(self):
        """Generate comprehensive performance report."""
        overview = self.latency_tracker.get_pipeline_overview()
        stats = self.latency_tracker.get_statistics()
        
        return {
            'pipeline_overview': overview,
            'stage_statistics': stats,
            'recommendations': self._generate_recommendations(stats)
        }
```

### API Service Integration

```python
class APIService:
    def __init__(self):
        self.latency_tracker = create_latency_tracker(
            "api_service",
            config=create_latency_config(
                warning_threshold_ms=100.0,
                critical_threshold_ms=500.0
            )
        )
        
        # Add middleware for automatic tracking
        self.app.middleware(self.latency_middleware)
        
        self.latency_tracker.start_monitoring()
    
    def latency_middleware(self, request):
        """Middleware to track API endpoint latency."""
        endpoint = request.endpoint or request.path
        
        with LatencyContextManager(self.latency_tracker, endpoint) as ctx:
            ctx.set_metadata({
                "method": request.method,
                "path": request.path,
                "user_agent": request.headers.get('User-Agent'),
                "ip": request.remote_addr
            })
            
            try:
                response = self.process_request(request)
                ctx.set_metadata({"status_code": response.status_code})
                return response
            except Exception as e:
                ctx.set_metadata({"error": str(e)})
                raise
    
    def process_request(self, request):
        """Process the actual request."""
        # Your API logic here
        return {"status": "success", "data": "processed"}
```

### Data Pipeline Integration

```python
class DataPipeline:
    def __init__(self):
        self.latency_tracker = create_latency_tracker(
            "data_pipeline",
            config=create_latency_config(
                max_history_points=10000,
                warning_threshold_ms=5000.0,  # 5 seconds
                critical_threshold_ms=15000.0  # 15 seconds
            )
        )
        
        self.latency_tracker.start_monitoring()
    
    def process_batch(self, batch_data):
        """Process data batch with stage tracking."""
        results = []
        
        for item in batch_data:
            # Data validation stage
            with LatencyContextManager(self.latency_tracker, "data_validation") as ctx:
                ctx.set_metadata({"item_id": item['id'], "type": item['type']})
                if not self.validate_item(item):
                    raise ValueError(f"Invalid item: {item['id']}")
            
            # Data transformation stage
            with LatencyContextManager(self.latency_tracker, "data_transformation") as ctx:
                ctx.set_metadata({"transformation": "standardization"})
                transformed = self.transform_item(item)
            
            # Data storage stage
            with LatencyContextManager(self.latency_tracker, "data_storage") as ctx:
                ctx.set_metadata({"storage": "database"})
                stored = self.store_item(transformed)
            
            results.append({"id": item['id'], "status": "processed"})
        
        return results
    
    def get_pipeline_health(self):
        """Get overall pipeline health."""
        stats = self.latency_tracker.get_statistics()
        
        health_score = 100
        issues = []
        
        for stage, stage_stats in stats.items():
            # Check for performance issues
            if stage_stats.p95_ms > 10000:  # 10 seconds
                health_score -= 20
                issues.append(f"{stage} has high P95 latency")
            
            if stage_stats.error_rate > 0.05:  # 5%
                health_score -= 30
                issues.append(f"{stage} has high error rate")
        
        return {
            "health_score": max(0, health_score),
            "issues": issues,
            "recommendation": "healthy" if health_score > 80 else "needs_attention"
        }
```

---

## API Reference

### PipelineLatencyTracker

#### Constructor

```python
PipelineLatencyTracker(
    pipeline_name: str,
    config: Optional[PipelineLatencyConfig] = None,
    persistence_manager: Optional[Any] = None
)
```

#### Methods

##### record_latency()

```python
record_latency(
    stage: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> None
```

##### get_statistics()

```python
get_statistics(stage: Optional[str] = None) -> Dict[str, LatencyStatistics]
```

##### get_latency_dataframe()

```python
get_latency_dataframe(
    stage: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame
```

##### create_latency_chart()

```python
create_latency_chart(
    stage: Optional[str] = None,
    time_range: Optional[timedelta] = None,
    save_path: Optional[str] = None
) -> go.Figure
```

##### start_monitoring() / stop_monitoring()

```python
start_monitoring() -> None
stop_monitoring() -> None
```

### LatencyContextManager

#### Constructor

```python
LatencyContextManager(
    tracker: PipelineLatencyTracker,
    stage: str,
    metadata: Optional[Dict[str, Any]] = None
)
```

#### Methods

##### set_metadata()

```python
set_metadata(key: str, value: Any) -> None
```

---

## Best Practices

### Performance Optimization

#### Configuration Tuning

```python
# For high-frequency systems
config = PipelineLatencyConfig(
    max_history_points=50000,    # Large history for analysis
    statistics_window=5000,        # Larger window for stable stats
    auto_cleanup=True,             # Prevent memory issues
    enable_persistence=True,        # Background saving
    persistence_interval=30         # Frequent persistence
)

# For low-frequency systems
config = PipelineLatencyConfig(
    max_history_points=1000,     # Smaller history
    statistics_window=100,         # Smaller window for responsiveness
    auto_cleanup=True,
    enable_persistence=False,       # Manual persistence
    persistence_interval=300       # Less frequent saves
)
```

#### Memory Management

```python
# Regular cleanup
tracker = PipelineLatencyTracker(
    "memory_critical_pipeline",
    config=PipelineLatencyConfig(
        max_history_points=1000,  # Limit history size
        auto_cleanup=True,         # Enable automatic cleanup
        statistics_window=500      # Reasonable window
    )
)

# Monitor memory usage
import psutil

def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
        # Could trigger cleanup or reduce history size

# Schedule regular checks
import asyncio

async def memory_monitor():
    while True:
        check_memory_usage()
        await asyncio.sleep(60)  # Check every minute
```

### Monitoring Setup

#### Production Configuration

```python
# Production-ready configuration
production_config = PipelineLatencyConfig(
    # Alert thresholds
    alert_thresholds={
        'warning_ms': 100.0,      # 100ms warning
        'critical_ms': 500.0,     # 500ms critical
        'error_rate_warning': 0.02,  # 2% error rate warning
        'error_rate_critical': 0.05   # 5% error rate critical
    },
    
    # Data management
    max_history_points=10000,
    statistics_window=1000,
    auto_cleanup=True,
    
    # Persistence
    enable_persistence=True,
    persistence_interval=60,
    
    # Real-time features
    enable_real_time_alerts=True
)

tracker = PipelineLatencyTracker(
    "production_pipeline",
    config=production_config,
    persistence_manager=database_manager
)
```

#### Alert Integration

```python
# Multi-channel alerting
class AlertManager:
    def __init__(self, latency_tracker):
        self.tracker = latency_tracker
        
        # Register multiple alert handlers
        self.tracker.add_alert_callback(self.email_alert)
        self.tracker.add_alert_callback(self.slack_alert)
        self.tracker.add_alert_callback(self.pagerduty_alert)
        self.tracker.add_alert_callback(self.webhook_alert)
    
    def email_alert(self, alert):
        """Send email alert."""
        # Email implementation
        pass
    
    def slack_alert(self, alert):
        """Send Slack notification."""
        # Slack implementation
        pass
    
    def pagerduty_alert(self, alert):
        """Trigger PagerDuty incident."""
        # PagerDuty implementation
        pass
    
    def webhook_alert(self, alert):
        """Send webhook to monitoring system."""
        # Webhook implementation
        pass

# Use in production
alert_manager = AlertManager(tracker)
```

### Data Analysis

#### Statistical Analysis

```python
# Advanced statistical analysis
def analyze_pipeline_performance(tracker):
    """Comprehensive performance analysis."""
    stats = tracker.get_statistics()
    
    analysis = {}
    
    for stage, stage_stats in stats.items():
        # Performance classification
        if stage_stats.mean_ms < 50:
            performance = "excellent"
        elif stage_stats.mean_ms < 100:
            performance = "good"
        elif stage_stats.mean_ms < 500:
            performance = "acceptable"
        else:
            performance = "poor"
        
        # Stability analysis
        cv = stage_stats.std_ms / stage_stats.mean_ms if stage_stats.mean_ms > 0 else 0
        if cv < 0.1:
            stability = "very_stable"
        elif cv < 0.2:
            stability = "stable"
        elif cv < 0.5:
            stability = "moderate"
        else:
            stability = "unstable"
        
        analysis[stage] = {
            'performance': performance,
            'stability': stability,
            'coefficient_of_variation': cv,
            'recommendations': _generate_recommendations(stage_stats)
        }
    
    return analysis

def _generate_recommendations(stats):
    """Generate performance recommendations."""
    recommendations = []
    
    if stats.mean_ms > 200:
        recommendations.append("Consider optimizing critical path")
    
    if stats.p95_ms > 1000:
        recommendations.append("Investigate outlier performance issues")
    
    if stats.error_rate > 0.05:
        recommendations.append("Review error handling and retry logic")
    
    if stats.std_ms / stats.mean_ms > 0.5:
        recommendations.append("Performance is inconsistent - investigate variability")
    
    return recommendations
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage

**Problem**: Memory usage increases over time.

**Solution**:
```python
# Reduce history size
config = PipelineLatencyConfig(
    max_history_points=1000,  # Reduce from default
    auto_cleanup=True,          # Enable cleanup
    statistics_window=500       # Smaller window
)

# Monitor memory usage
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        logger.warning(f"High memory: {memory.percent:.1f}%")
        # Trigger manual cleanup
        tracker._cleanup_old_data()
```

#### Missing Data

**Problem**: Some latency measurements not recorded.

**Solution**:
```python
# Check context manager usage
try:
    with LatencyContextManager(tracker, "stage_name") as ctx:
        result = process_data()
        # Ensure result is used
        ctx.set_metadata({"result_size": len(result)})
except Exception as e:
    logger.error(f"Context manager error: {e}")

# Manual recording fallback
try:
    start_time = time.time()
    result = process_data()
    latency_ms = (time.time() - start_time) * 1000
    tracker.record_latency("stage_name", latency_ms, success=True)
except Exception as e:
    tracker.record_latency("stage_name", 0, success=False, error_message=str(e))
```

#### Alert Fatigue

**Problem**: Too many alerts causing alert fatigue.

**Solution**:
```python
# Implement alert deduplication
class AlertDeduplicator:
    def __init__(self, cooldown_minutes=5):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_alerts = {}
    
    def should_send_alert(self, alert):
        key = f"{alert['type']}_{alert['stage']}"
        
        if key in self.last_alerts:
            time_since_last = datetime.now() - self.last_alerts[key]
            if time_since_last < self.cooldown:
                return False
        
        self.last_alerts[key] = datetime.now()
        return True

# Use in alert handler
deduplicator = AlertDeduplicator(cooldown_minutes=5)

def filtered_alert_handler(alert):
    if deduplicator.should_send_alert(alert):
        # Send actual alert
        send_alert(alert)
```

### Performance Issues

#### Slow Statistics Calculation

**Problem**: Statistics calculation is slow.

**Solution**:
```python
# Optimize configuration
config = PipelineLatencyConfig(
    statistics_window=500,      # Smaller window
    max_history_points=2000,   # Less data to process
    auto_cleanup=True           # Regular cleanup
)

# Use caching effectively
stats = tracker.get_statistics()  # Cached for 1 minute
# Don't recalculate for each request
```

#### Database Bottlenecks

**Problem**: Persistence is slow.

**Solution**:
```python
# Batch persistence operations
config = PipelineLatencyConfig(
    enable_persistence=True,
    persistence_interval=300,   # Save every 5 minutes
    # Reduce database writes
)

# Use async persistence
async def batch_save():
    while True:
        await asyncio.sleep(config.persistence_interval)
        await tracker._save_all_statistics()
```

---

## Conclusion

The Pipeline Latency Tracking system provides enterprise-grade monitoring capabilities for real-time performance analysis. With comprehensive statistical analysis, alert systems, and visualization tools, it enables proactive performance optimization and issue detection for critical trading and data processing pipelines.

Key benefits:
- **Real-time monitoring** with immediate alerting
- **Statistical rigor** with proper percentile analysis
- **Scalable architecture** for high-frequency systems
- **Integration flexibility** for various pipeline types
- **Professional visualization** for stakeholder reporting
- **Production-ready features** with robust error handling
