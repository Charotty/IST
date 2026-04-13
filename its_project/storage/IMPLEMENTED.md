# Storage Layer Implementation Status

## Purpose
Persist raw `MarketData` streams: warm storage (TimescaleDB) for recent/fast access, cold storage (Parquet) for long-term/archival.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interface | `base.py` | Done | `BaseStorage` (write/write_batch/read/get_latest/close) |
| TimescaleDB client | `timescale.py` | Done | asyncpg pool, batch insert, read with indexes |
| Parquet client | `parquet.py` | Done | pyarrow, partitioned by date, append mode |
| Batch writer task | `writer.py` | Done | Async batching with max interval & backpressure |
| Read API wrapper | `api.py` | Done | Warm-first with fallback to cold |
| Init SQL schema | `init.sql` | Done | Hypertable, indexes, compression policy |
| Package init | `__init__.py` | Done | Exports main classes |
| Documentation | `README.md` | Done | Usage example and schema |

## Data contracts

### Table schema (TimescaleDB)
```sql
CREATE TABLE market_data (
    timestamp_ms BIGINT NOT NULL,
    symbol TEXT NOT NULL,
    type TEXT NOT NULL,
    exchange TEXT NOT NULL,
    data JSONB NOT NULL
);
```

### Parquet layout
- Partitioned by date (`YYYY-MM-DD.parquet`)
- Columns: timestamp_ms, symbol, type, exchange, data (JSON string)
- Optional: partitioning by symbol (pyarrow dataset mode)

## Integration notes
- Use `batch_writer_task` to avoid flooding storage with single writes.
- Warm storage preferred for reads; fallback to cold if warm fails.
- `MarketData.type.value` stored as TEXT (enum string).

## Next improvements
- Configurable TTL/cleanup for warm storage
- Parquet read optimization (metadata, partition pruning)
- Metrics/health for storage writers
- Schema versioning support
