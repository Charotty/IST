# Pipeline Logging System - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Configuration](#configuration)
4. [Basic Usage](#basic-usage)
5. [Pipeline Stages](#pipeline-stages)
6. [Logging Levels](#logging-levels)
7. [Structured Logging](#structured-logging)
8. [Performance Monitoring](#performance-monitoring)
9. [Correlation Tracking](#correlation-tracking)
10. [Data Export](#data-export)
11. [Integration Examples](#integration-examples)
12. [API Reference](#api-reference)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The Pipeline Logging system provides comprehensive monitoring and analysis for data → execution pipeline stages, enabling real-time tracking of execution performance, error rates, and system health.

### Key Features

- ✅ **Multi-level logging** with structured format
- ✅ **Stage-based execution tracking** with timing and correlation
- ✅ **Performance metrics collection** with statistical analysis
- ✅ **Real-time monitoring** with configurable intervals
- ✅ **Correlation and traceability** across pipeline stages
- ✅ **Data persistence** with multiple export formats
- ✅ **Background aggregation** and cleanup tasks
- ✅ **Alert system** for performance degradation
- ✅ **Thread-safe operations** for concurrent systems

### Use Cases

- **Data Pipelines**: Monitor ETL processes, data validation, transformations
- **ML Pipelines**: Track feature engineering, model training, inference stages
- **API Services**: Log endpoint performance, request processing, response times
- **Trading Systems**: Monitor order processing, execution, confirmation stages
- **Batch Processing**: Track job execution, completion rates, resource usage

---

## Core Components

### 1. PipelineLogger (`PipelineLogger`)

Main logging system with comprehensive monitoring capabilities.

```python
class PipelineLogger:
    def __init__(
        self,
        pipeline_name: str,
        config: Optional[PipelineLoggerConfig] = None
    )
```

**Key Features:**
- **Stage Tracking**: Monitor individual pipeline stages
- **Performance Metrics**: Collect timing, success rates, errors
- **Correlation Tracking**: Link related executions across stages
- **Real-time Monitoring**: Background tasks for aggregation
- **Data Export**: Multiple formats for analysis

### 2. PipelineLogEntry (`PipelineLogEntry`)

Structured log entry with comprehensive metadata.

```python
@dataclass
class PipelineLogEntry:
    timestamp: datetime      # When event occurred
    stage: PipelineStage     # Pipeline stage identifier
    level: LogLevel         # Logging level
    message: str           # Log message
    data: Dict[str, Any] # Additional context data
    execution_id: Optional[str]  # Unique execution identifier
    duration_ms: Optional[float]  # Stage duration in milliseconds
    metadata: Dict[str, Any]  # Additional metadata
    thread_id: Optional[str]  # Thread identifier
    correlation_id: Optional[str]  # Correlation ID
```

### 3. PipelineMetrics (`PipelineMetrics`)

Performance metrics for pipeline stages.

```python
@dataclass
class PipelineMetrics:
    stage: PipelineStage           # Pipeline stage
    total_executions: int          # Total executions
    successful_executions: int      # Successful executions
    failed_executions: int           # Failed executions
    avg_duration_ms: float          # Average duration
    min_duration_ms: float          # Minimum duration
    max_duration_ms: float          # Maximum duration
    p95_duration_ms: float          # 95th percentile
    error_rate: float              # Error rate (0.0-1.0)
    last_execution: Optional[datetime]  # Last execution time
    last_error: Optional[str]         # Last error message
```

### 4. PipelineLoggerConfig (`PipelineLoggerConfig`)

Configuration for pipeline logging system.

```python
@dataclass
class PipelineLoggerConfig:
    log_level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    enable_database_logging: bool = True
    log_directory: str = "logs"
    log_file_prefix: str = "pipeline"
    max_file_size_mb: int = 100
    enable_compression: bool = True
    enable_structured_logging: bool = True
    enable_performance_tracking: bool = True
    metrics_window_size: int = 1000
    enable_correlation_tracking: bool = True
    retention_days: int = 30
```

### 5. PipelineStage (`PipelineStage`)

Enumeration of pipeline stages.

```python
class PipelineStage(Enum):
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_INFERENCE = "model_inference"
    DECISION_MAKING = "decision_making"
    ORDER_GENERATION = "order_generation"
    ORDER_EXECUTION = "order_execution"
    EXECUTION_CONFIRMATION = "execution_confirmation"
    ERROR_HANDLING = "error_handling"
    CLEANUP = "cleanup"
```

### 6. LogLevel (`LogLevel`)

Logging levels for pipeline logging.

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

### 7. PipelineCorrelationTracker (`PipelineCorrelationTracker`)

Track correlations between pipeline stages and executions.

```python
class PipelineCorrelationTracker:
    def __init__(self, pipeline_name: str)
    
    def track_execution_flow(self, execution_id: str, stages: List[str])
    def analyze_bottlenecks(self) -> Dict[str, Any]
    def generate_correlation_matrix(self) -> pd.DataFrame
```

---

## Configuration

### Basic Configuration

```python
from its_project.execution import create_pipeline_logger, create_pipeline_config

# Create with default settings
logger = create_pipeline_logger("data_pipeline")

# Create with custom configuration
config = create_pipeline_config(
    log_level=LogLevel.DEBUG,
    enable_file_logging=True,
    enable_database_logging=False,
    retention_days=60,
    metrics_window_size=2000
)

logger = PipelineLogger("ml_pipeline", config)
```

### Advanced Configuration

```python
# Production configuration
production_config = PipelineLoggerConfig(
    log_level=LogLevel.INFO,
    enable_file_logging=True,
    enable_database_logging=True,
    log_directory="/var/log/pipeline",
    max_file_size_mb=500,
    enable_compression=True,
    enable_structured_logging=True,
    enable_performance_tracking=True,
    metrics_window_size=5000,
    enable_correlation_tracking=True,
    retention_days=90
)

# Development configuration
development_config = PipelineLoggerConfig(
    log_level=LogLevel.DEBUG,
    enable_file_logging=True,
    enable_database_logging=False,
    log_directory="./logs",
    max_file_size_mb=50,
    enable_compression=False,
    enable_structured_logging=True,
    enable_performance_tracking=True,
    metrics_window_size=500,
    enable_correlation_tracking=False,
    retention_days=7
)
```

### Environment Variables

```python
import os

# Configuration from environment
config = PipelineLoggerConfig(
    log_level=getattr(LogLevel, os.getenv('PIPELINE_LOG_LEVEL', 'INFO')),
    enable_file_logging=os.getenv('PIPELINE_ENABLE_FILE_LOGGING', 'true').lower() == 'true',
    enable_database_logging=os.getenv('PIPELINE_ENABLE_DB_LOGGING', 'false').lower() == 'true',
    log_directory=os.getenv('PIPELINE_LOG_DIR', './logs'),
    retention_days=int(os.getenv('PIPELINE_LOG_RETENTION_DAYS', '30'))
)
```

---

## Pipeline Stages

### Data Ingestion Stage

```python
# Start data ingestion stage
execution_id = logger.start_stage(
    stage=PipelineStage.DATA_INGESTION,
    metadata={
        "source": "database",
        "table": "trades",
        "batch_size": 1000
    }
)

# Log ingestion progress
logger.log_event(
    stage=PipelineStage.DATA_INGESTION,
    level=LogLevel.INFO,
    message="Ingesting batch 1000 records",
    data={
        "records_processed": 500,
        "records_remaining": 500,
        "ingestion_rate": 100.0
    }
)

# End stage
logger.end_stage(
    stage=PipelineStage.DATA_INGESTION,
    stage_id=execution_id,
    success=True
)
```

### Data Validation Stage

```python
# Start validation stage
validation_id = logger.start_stage(
    stage=PipelineStage.DATA_VALIDATION,
    correlation_id=execution_id,  # Link to ingestion
    metadata={
        "validation_rules": ["not_null", "data_types", "ranges"]
    }
)

# Validation with automatic timing
with PipelineLoggingContext(logger, PipelineStage.DATA_VALIDATION) as ctx:
    ctx.set_metadata({"record_id": record.id})
    
    if not validate_record(record):
        ctx.set_metadata({"validation_error": "Invalid record format"})
        raise ValueError("Invalid record")
    
    # Validation successful
    ctx.set_metadata({"validation_result": "passed"})

# End stage
logger.end_stage(
    stage=PipelineStage.DATA_VALIDATION,
    stage_id=validation_id,
    success=True
)
```

### Feature Engineering Stage

```python
# Start feature engineering
feature_id = logger.start_stage(
    stage=PipelineStage.FEATURE_ENGINEERING,
    correlation_id=validation_id,
    metadata={
        "features": ["sma", "rsi", "bollinger"],
        "window_sizes": [5, 10, 20]
    }
)

# Feature processing with progress tracking
features_processed = 0
total_features = 100

for feature_batch in feature_batches:
    with PipelineLoggingContext(logger, PipelineStage.FEATURE_ENGINEERING) as ctx:
        ctx.set_metadata({
            "batch_id": feature_batch.id,
            "features_count": len(feature_batch)
        })
        
        processed_features = process_features(feature_batch)
        features_processed += processed_features
        
        ctx.set_metadata({
            "progress": f"{features_processed}/{total_features}",
            "completion_rate": features_processed / total_features
        })

# End stage
logger.end_stage(
    stage=PipelineStage.FEATURE_ENGINEERING,
    stage_id=feature_id,
    success=True
)
```

### Model Inference Stage

```python
# Start model inference
inference_id = logger.start_stage(
    stage=PipelineStage.MODEL_INFERENCE,
    correlation_id=feature_id,
    metadata={
        "model_name": "trading_model_v2",
        "model_version": "1.2.3",
        "batch_size": 100
    }
)

# Inference with timing and resource tracking
with PipelineLoggingContext(logger, PipelineStage.MODEL_INFERENCE) as ctx:
    ctx.set_metadata({
        "input_shape": input_data.shape,
        "model_path": "/models/trading_model_v2.pkl"
    })
    
    start_time = time.time()
    predictions = model.predict(input_data)
    inference_time = (time.time() - start_time) * 1000
    
    ctx.set_metadata({
        "inference_time_ms": inference_time,
        "predictions_count": len(predictions),
        "throughput": len(predictions) / inference_time * 1000
    })

# End stage
logger.end_stage(
    stage=PipelineStage.MODEL_INFERENCE,
    stage_id=inference_id,
    success=True
)
```

---

## Logging Levels

### Log Level Hierarchy

```python
# Log levels in order of severity
# DEBUG: Detailed information for debugging
# INFO: General information about pipeline execution
# WARNING: Potential issues that don't stop execution
# ERROR: Serious issues that may affect results
# CRITICAL: Critical issues that stop pipeline execution

# Set appropriate level
logger.log_event(
    stage=PipelineStage.DATA_VALIDATION,
    level=LogLevel.ERROR,
    message="Critical validation failure",
    data={"error_type": "schema_violation"}
)
```

### Structured Logging

```python
# Enable structured logging
config = PipelineLoggerConfig(
    enable_structured_logging=True
)

# Structured log entry
logger.log_event(
    stage=PipelineStage.DATA_INGESTION,
    level=LogLevel.INFO,
    message="Data ingestion completed",
    data={
        "source": "database",
        "table": "trades",
        "records_count": 1000,
        "processing_time": 15.5,
        "throughput": 64.5,
        "batch_id": "batch_001"
    }
)
```

### Log Filtering

```python
# Get logs by stage
data_ingestion_logs = logger.get_logs_dataframe(
    stage=PipelineStage.DATA_INGESTION,
    level=LogLevel.ERROR
)

# Get logs by time range
recent_logs = logger.get_logs_dataframe(
    start_time=datetime.now() - timedelta(hours=1),
    end_time=datetime.now()
)

# Get logs by level
error_logs = logger.get_logs_dataframe(level=LogLevel.ERROR)

# Get logs with specific data
validation_errors = logger.get_logs_dataframe(
    stage=PipelineStage.DATA_VALIDATION,
    level=LogLevel.ERROR,
    data={"error_type": "schema_violation"}
)
```

---

## Performance Monitoring

### Real-time Metrics

```python
# Get current metrics
current_metrics = logger.get_stage_metrics()
for stage, metrics in current_metrics.items():
    print(f"{stage}:")
    print(f"  Total: {metrics.total_executions}")
    print(f"  Success: {metrics.successful_executions}")
    print(f"  Failed: {metrics.failed_executions}")
    print(f"  Success Rate: {metrics.successful_executions / metrics.total_executions:.1%}")
    print(f"  Avg Duration: {metrics.avg_duration_ms:.2f}ms")
    print(f"  P95 Duration: {metrics.p95_duration_ms:.2f}ms")
    print(f"  Error Rate: {metrics.error_rate:.1%}")
```

### Performance Alerts

```python
# Custom alert handler
def performance_alert(log_entry):
    if log_entry.level == LogLevel.ERROR:
        if log_entry.stage == PipelineStage.MODEL_INFERENCE:
            # Model inference failure alert
            send_alert(
                message="Model inference failure detected",
                severity="high",
                details=log_entry.data
            )
        elif log_entry.stage == PipelineStage.DATA_INGESTION:
            # Data ingestion failure alert
            send_alert(
                message="Data ingestion pipeline failure",
                severity="medium",
                details=log_entry.data
            )

# Register alert handler
logger.add_alert_handler(performance_alert)
```

### Historical Analysis

```python
# Get performance trends
metrics_history = []
for day in range(30):  # Last 30 days
    daily_metrics = logger.get_stage_metrics()
    metrics_history.append(daily_metrics)

# Analyze trends
for stage in PipelineStage:
    stage_metrics = [m for m in metrics_history if stage in m]
    if stage_metrics:
        # Calculate trend
        recent_avg = stage_metrics[-1].avg_duration_ms
        historical_avg = sum(m.avg_duration_ms for m in stage_metrics) / len(stage_metrics)
        
        if recent_avg > historical_avg * 1.2:  # 20% degradation
            print(f"Performance degradation detected in {stage}")
```

---

## Correlation Tracking

### Execution Flow Tracking

```python
# Create correlation tracker
correlation_tracker = PipelineCorrelationTracker("trading_pipeline")

# Track execution flow
execution_id = correlation_tracker.track_execution_flow(
    execution_id="exec_001",
    stages=[
        PipelineStage.DATA_INGESTION,
        PipelineStage.FEATURE_ENGINEERING,
        PipelineStage.MODEL_INFERENCE,
        PipelineStage.DECISION_MAKING
    ]
)

# Get correlation matrix
correlation_matrix = correlation_tracker.generate_correlation_matrix()
print(correlation_matrix.head())
```

### Bottleneck Analysis

```python
# Analyze bottlenecks
bottleneck_analysis = correlation_tracker.analyze_bottlenecks()

for stage, analysis in bottleneck_analysis.items():
    print(f"{stage} Bottleneck Analysis:")
    print(f"  Average Wait Time: {analysis['avg_wait_time']:.2f}ms")
    print(f"  Error Rate: {analysis['error_rate']:.1%}")
    print(f"  Bottleneck Score: {analysis['bottleneck_score']:.2f}")
    print(f"  Recommendation: {analysis['recommendation']}")
```

### Root Cause Analysis

```python
# Find root causes of failures
error_logs = logger.get_logs_dataframe(level=LogLevel.ERROR)

for error_log in error_logs.iterrows():
    if error_log.stage == PipelineStage.DATA_VALIDATION:
        # Analyze validation errors
        error_patterns = error_log.data.get("validation_errors", [])
        for pattern in error_patterns:
            print(f"Validation Error Pattern: {pattern}")
```

---

## Data Export

### Log Export

```python
# Export all logs to CSV
logger.export_logs(
    format="csv",
    save_path="pipeline_logs.csv"
)

# Export specific stage logs
logger.export_logs(
    format="csv",
    stage=PipelineStage.MODEL_INFERENCE,
    save_path="model_inference_logs.csv"
)

# Export with metrics
logger.export_logs(
    format="csv",
    include_metrics=True,
    save_path="complete_pipeline_logs.csv"
)
```

### Metrics Export

```python
# Export performance metrics
metrics_df = logger.get_metrics_dataframe()
metrics_df.to_csv("pipeline_metrics.csv")

# Export with correlation analysis
correlation_df = correlation_tracker.generate_correlation_matrix()
correlation_df.to_csv("pipeline_correlation.csv")

# Export to JSON
export_data = {
    "pipeline_name": "trading_pipeline",
    "export_timestamp": datetime.now().isoformat(),
    "stage_metrics": {
        stage.value: asdict(metrics) 
        for stage, metrics in logger.get_stage_metrics().items()
    },
    "correlation_matrix": correlation_df.to_dict(),
    "bottleneck_analysis": correlation_tracker.analyze_bottlenecks()
}

with open("pipeline_analysis.json", "w") as f:
    json.dump(export_data, f, indent=2, default=str)
```

### Excel Export

```python
# Export to Excel with multiple sheets
with pd.ExcelWriter("pipeline_report.xlsx") as writer:
    # Logs sheet
    logs_df = logger.get_logs_dataframe()
    logs_df.to_excel(writer, sheet_name="Pipeline Logs", index=False)
    
    # Metrics sheet
    metrics_df = logger.get_metrics_dataframe()
    metrics_df.to_excel(writer, sheet_name="Performance Metrics", index=False)
    
    # Correlation sheet
    correlation_df = correlation_tracker.generate_correlation_matrix()
    correlation_df.to_excel(writer, sheet_name="Stage Correlations", index=False)
```

---

## Integration Examples

### Complete Pipeline Integration

```python
class TradingPipeline:
    def __init__(self):
        # Initialize pipeline logger
        self.logger = create_pipeline_logger(
            "trading_pipeline",
            config=create_pipeline_config(
                enable_file_logging=True,
                enable_performance_tracking=True,
                enable_correlation_tracking=True
            )
        )
        
        # Initialize correlation tracker
        self.correlation_tracker = PipelineCorrelationTracker("trading_pipeline")
        
        # Start monitoring
        self.logger.start_monitoring()
    
    async def process_trading_data(self, raw_data):
        """Process trading data through pipeline."""
        # Data ingestion stage
        ingestion_id = self.logger.start_stage(
            stage=PipelineStage.DATA_INGESTION,
            metadata={"source": "market_data", "records": len(raw_data)}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.DATA_INGESTION, ingestion_id) as ctx:
            ctx.set_metadata({"batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"})
            
            # Ingest data
            processed_data = await self.ingest_data(raw_data)
            
            ctx.set_metadata({
                "records_processed": len(processed_data),
                "data_quality_score": self.assess_data_quality(processed_data)
            })
        
        self.logger.end_stage(
            stage=PipelineStage.DATA_INGESTION,
            stage_id=ingestion_id,
            success=True
        )
        
        # Feature engineering stage
        feature_id = self.logger.start_stage(
            stage=PipelineStage.FEATURE_ENGINEERING,
            correlation_id=ingestion_id,
            metadata={"features": ["sma", "rsi", "bollinger"]}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.FEATURE_ENGINEERING, feature_id) as ctx:
            features = await self.engineer_features(processed_data)
            
            ctx.set_metadata({
                "features_generated": len(features),
                "feature_dimensions": features.shape if hasattr(features, 'shape') else None
            })
        
        self.logger.end_stage(
            stage=PipelineStage.FEATURE_ENGINEERING,
            stage_id=feature_id,
            success=True
        )
        
        # Model inference stage
        inference_id = self.logger.start_stage(
            stage=PipelineStage.MODEL_INFERENCE,
            correlation_id=feature_id,
            metadata={"model_name": "trading_model_v2"}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.MODEL_INFERENCE, inference_id) as ctx:
            predictions = await self.run_inference(features)
            
            ctx.set_metadata({
                "predictions_count": len(predictions),
                "confidence_score": self.calculate_confidence(predictions)
            })
        
        self.logger.end_stage(
            stage=PipelineStage.MODEL_INFERENCE,
            stage_id=inference_id,
            success=True
        )
        
        return predictions
```

### ML Pipeline Integration

```python
class MLPipeline:
    def __init__(self):
        self.logger = create_pipeline_logger(
            "ml_pipeline",
            config=create_pipeline_config(
                log_level=LogLevel.DEBUG,
                enable_performance_tracking=True,
                enable_correlation_tracking=True
            )
        )
        
        # Track experiment correlations
        self.experiment_tracker = {}
    
    def run_experiment(self, experiment_config):
        """Run ML experiment with full tracking."""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Track experiment flow
        self.experiment_tracker[experiment_id] = {
            "config": experiment_config,
            "stages": [],
            "start_time": datetime.now()
        }
        
        # Data preparation
        data_id = self.logger.start_stage(
            stage=PipelineStage.DATA_TRANSFORMATION,
            execution_id=experiment_id,
            metadata={"experiment_config": experiment_config}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.DATA_TRANSFORMATION, data_id) as ctx:
            transformed_data = self.prepare_data(experiment_config)
            
            self.experiment_tracker[experiment_id]["stages"].append({
                "stage": "data_transformation",
                "start_time": datetime.now(),
                "end_time": datetime.now()
            })
        
        self.logger.end_stage(
            stage=PipelineStage.DATA_TRANSFORMATION,
            stage_id=data_id,
            success=True
        )
        
        # Model training
        training_id = self.logger.start_stage(
            stage=PipelineStage.MODEL_INFERENCE,
            execution_id=experiment_id,
            correlation_id=data_id,
            metadata={"model_type": experiment_config["model_type"]}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.MODEL_INFERENCE, training_id) as ctx:
            model = await self.train_model(transformed_data, experiment_config)
            
            self.experiment_tracker[experiment_id]["stages"].append({
                "stage": "model_training",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "model_performance": model.get_metrics()
            })
        
        self.logger.end_stage(
            stage=PipelineStage.MODEL_INFERENCE,
            stage_id=training_id,
            success=True
        )
        
        # Evaluation
        eval_id = self.logger.start_stage(
            stage=PipelineStage.DECISION_MAKING,
            execution_id=experiment_id,
            correlation_id=training_id,
            metadata={"evaluation_metrics": experiment_config["evaluation_metrics"]}
        )
        
        with PipelineLoggingContext(self.logger, PipelineStage.DECISION_MAKING, eval_id) as ctx:
            evaluation = await self.evaluate_model(model, experiment_config)
            
            self.experiment_tracker[experiment_id]["stages"].append({
                "stage": "model_evaluation",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "evaluation_results": evaluation
            })
        
        self.logger.end_stage(
            stage=PipelineStage.DECISION_MAKING,
            stage_id=eval_id,
            success=True
        )
        
        # Complete experiment
        self.experiment_tracker[experiment_id]["end_time"] = datetime.now()
        
        return {
            "experiment_id": experiment_id,
            "total_duration": (self.experiment_tracker[experiment_id]["end_time"] - 
                           self.experiment_tracker[experiment_id]["start_time"]).total_seconds(),
            "stage_details": self.experiment_tracker[experiment_id]["stages"]
        }
```

---

## API Reference

### PipelineLogger

#### Constructor

```python
PipelineLogger(
    pipeline_name: str,
    config: Optional[PipelineLoggerConfig] = None
)
```

#### Methods

##### start_stage()

```python
start_stage(
    stage: PipelineStage,
    execution_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

##### end_stage()

```python
end_stage(
    stage: PipelineStage,
    stage_id: str,
    success: bool,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

##### log_event()

```python
log_event(
    stage: PipelineStage,
    level: LogLevel,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

##### get_stage_metrics()

```python
get_stage_metrics(stage: Optional[PipelineStage] = None) -> Dict[PipelineStage, PipelineMetrics]
```

##### get_logs_dataframe()

```python
get_logs_dataframe(
    stage: Optional[PipelineStage] = None,
    level: Optional[LogLevel] = None,
    limit: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame
```

##### export_logs()

```python
export_logs(
    format: str = "csv",
    stage: Optional[PipelineStage] = None,
    level: Optional[LogLevel] = None,
    include_metrics: bool = True,
    save_path: Optional[str] = None
) -> str
```

##### start_monitoring() / stop_monitoring()

```python
start_monitoring() -> None
stop_monitoring() -> None
```

### PipelineLoggingContext

#### Constructor

```python
PipelineLoggingContext(
    logger: PipelineLogger,
    stage: PipelineStage,
    execution_id: Optional[str] = None,
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

### Configuration Management

#### Environment-based Configuration

```python
import os
from its_project.execution import create_pipeline_config

# Load configuration from environment
config = create_pipeline_config(
    log_level=getattr(LogLevel, os.getenv('PIPELINE_LOG_LEVEL', 'INFO')),
    enable_file_logging=os.getenv('PIPELINE_ENABLE_FILE_LOGGING', 'true').lower() == 'true',
    enable_database_logging=os.getenv('PIPELINE_ENABLE_DB_LOGGING', 'false').lower() == 'true',
    log_directory=os.getenv('PIPELINE_LOG_DIR', './logs'),
    retention_days=int(os.getenv('PIPELINE_LOG_RETENTION_DAYS', '30'))
)

# Use configuration
logger = PipelineLogger("production_pipeline", config)
```

#### Stage-based Configuration

```python
# Different configurations for different stages
stage_configs = {
    PipelineStage.DATA_INGESTION: PipelineLoggerConfig(
        log_level=LogLevel.INFO,
        enable_performance_tracking=True,
        metrics_window_size=2000  # Larger window for ingestion
    ),
    PipelineStage.MODEL_INFERENCE: PipelineLoggerConfig(
        log_level=LogLevel.DEBUG,
        enable_performance_tracking=True,
        metrics_window_size=1000  # Smaller window for inference
    ),
    PipelineStage.ERROR_HANDLING: PipelineLoggerConfig(
        log_level=LogLevel.ERROR,
        enable_performance_tracking=False,  # Focus on errors only
        metrics_window_size=500
    )
}
```

### Performance Optimization

#### Efficient Logging

```python
# Use appropriate log levels
# DEBUG: Only during development
# INFO: Production monitoring
# WARNING: Potential issues
# ERROR: Serious issues
# CRITICAL: System-stopping issues

# Batch logging for high-frequency operations
def log_batch_events(logger, events):
    """Log multiple events efficiently."""
    with PipelineLoggingContext(logger, stage, batch_id) as ctx:
        for event in events:
            ctx.set_metadata(event.metadata)
        
        # Log batch completion
        logger.log_event(
            stage=stage,
            level=LogLevel.INFO,
            message=f"Processed batch of {len(events)} events",
            data={"batch_size": len(events)}
        )
```

#### Memory Management

```python
# Configure appropriate retention
config = PipelineLoggerConfig(
    retention_days=30,  # Keep 30 days of logs
    max_file_size_mb=100,  # Limit file size
    enable_compression=True  # Compress old logs
)

# Monitor memory usage
import psutil

def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        logger.log_event(
            stage=PipelineStage.ERROR_HANDLING,
            level=LogLevel.WARNING,
            message=f"High memory usage: {memory_percent:.1f}%",
            data={"memory_percent": memory_percent}
        )
```

### Error Handling

#### Comprehensive Error Tracking

```python
# Error categorization
class PipelineError(Exception):
    def __init__(self, stage: PipelineStage, error_code: str, message: str, details: Dict[str, Any]):
        self.stage = stage
        self.error_code = error_code
        self.message = message
        self.details = details
        super().__init__(f"[{stage.value}] {error_code}: {message}")

# Error handling with context
def handle_pipeline_error(error: Exception, stage: PipelineStage, context: Dict[str, Any]):
    logger.log_event(
        stage=stage,
        level=LogLevel.ERROR,
        message=f"Pipeline error: {str(error)}",
        data={
            "error_type": type(error).__name__,
            "error_details": str(error),
            "context": context
        }
    )
```

### Monitoring Integration

#### Real-time Alerting

```python
# Custom alert handler
def pipeline_alert_handler(log_entry):
    if log_entry.level == LogLevel.CRITICAL:
        # Critical alert - immediate notification
        send_critical_notification(
            message=f"Critical pipeline failure: {log_entry.message}",
            stage=log_entry.stage,
            details=log_entry.data
        )
    elif log_entry.level == LogLevel.ERROR:
        # Error alert - monitoring dashboard update
        update_monitoring_dashboard({
            "alert_type": "pipeline_error",
            "stage": log_entry.stage,
            "message": log_entry.message,
            "timestamp": log_entry.timestamp
        })

# Register alert handler
logger.add_alert_handler(pipeline_alert_handler)
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage

**Problem**: Memory usage increases over time due to log accumulation.

**Solution**:
```python
# Configure appropriate retention
config = PipelineLoggerConfig(
    retention_days=7,  # Shorter retention
    max_file_size_mb=50,  # Smaller files
    enable_compression=True  # Compress old logs
)

# Enable automatic cleanup
logger = PipelineLogger("pipeline", config)
logger.start_monitoring()  # Starts cleanup task
```

#### Performance Degradation

**Problem**: Pipeline performance degrades over time.

**Solution**:
```python
# Monitor performance trends
def monitor_performance_degradation(logger):
    metrics_history = []
    
    for day in range(30):  # Last 30 days
        daily_metrics = logger.get_stage_metrics()
        metrics_history.append(daily_metrics)
    
    # Analyze trends for each stage
    for stage in PipelineStage:
        stage_metrics = [m for m in metrics_history if stage in m]
        if len(stage_metrics) > 1:
            recent_avg = stage_metrics[-1].avg_duration_ms
            historical_avg = sum(m.avg_duration_ms for m in stage_metrics) / len(stage_metrics)
            
            if recent_avg > historical_avg * 1.3:  # 30% degradation
                logger.log_event(
                    stage=PipelineStage.ERROR_HANDLING,
                    level=LogLevel.WARNING,
                    message=f"Performance degradation detected in {stage}",
                    data={
                        "recent_avg": recent_avg,
                        "historical_avg": historical_avg,
                        "degradation_pct": ((recent_avg / historical_avg) - 1) * 100
                    }
                )
```

#### Log File Issues

**Problem**: Log files become too large or corrupted.

**Solution**:
```python
# Configure file rotation
config = PipelineLoggerConfig(
    max_file_size_mb=100,  # Rotate at 100MB
    enable_compression=True,  # Compress rotated files
    retention_days=30  # Clean up old files
)

# Monitor disk usage
import shutil

def check_disk_usage(log_directory):
    total_size = sum(f.stat().st_size for f in Path(log_directory).glob("*.log*"))
    
    if total_size > 10 * 1024 * 1024 * 1024:  # 10GB
        logger.log_event(
            stage=PipelineStage.ERROR_HANDLING,
            level=LogLevel.WARNING,
            message=f"High disk usage: {total_size / (1024**3):.1f}GB",
            data={"total_size_gb": total_size / (1024**3)}
        )
```

---

## Conclusion

The Pipeline Logging system provides enterprise-grade monitoring capabilities for data → execution pipelines with comprehensive tracking, correlation analysis, and performance optimization. Key benefits include:

- **Real-time monitoring** with automatic aggregation
- **Stage-based tracking** with correlation analysis
- **Structured logging** with flexible configuration
- **Performance optimization** with bottleneck detection
- **Comprehensive export** capabilities for analysis
- **Production-ready features** with alerting and monitoring

The system is designed for high-frequency trading and ML pipelines where performance monitoring and traceability are critical for system reliability and optimization.
