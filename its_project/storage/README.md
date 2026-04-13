# Storage Layer

## Components

- `BaseStorage` (abstract interface)
- `TimescaleStorage` (warm storage, time-series)
- `ParquetStorage` (cold storage, partitioned by date)
- `batch_writer_task` (async batching with backpressure)

## Usage

```python
from its_project.storage import TimescaleStorage, ParquetStorage, batch_writer_task
from its_project.pipeline.queues import create_queues

# Warm storage (TimescaleDB)
warm = TimescaleStorage(dsn="postgres://user:pass@localhost/tsdb")
await warm.connect()

# Cold storage (Parquet)
cold = ParquetStorage(base_path="data/parquet")

# Queues
queues = create_queues(maxsize=10000)

# Start batch writers
stop_event = asyncio.Event()
tasks = [
    asyncio.create_task(
        batch_writer_task(
            name="price_warm",
            storage=warm,
            batch_size=500,
            max_interval_s=5.0,
            in_queue=queues.price_raw,
            stop_event=stop_event,
        )
    ),
    asyncio.create_task(
        batch_writer_task(
            name="lob_cold",
            storage=cold,
            batch_size=1000,
            max_interval_s=10.0,
            in_queue=queues.lob_raw,
            stop_event=stop_event,
        )
    ),
]

# Later...
stop_event.set()
await asyncio.gather(*tasks)
await warm.close()
```

## Schema

TimescaleDB table `market_data`:

| Column | Type |
|--------|------|
| timestamp_ms | BIGINT |
| symbol | TEXT |
| type | TEXT |
| exchange | TEXT |
| data | JSONB |

Parquet files are partitioned by date and symbol.
