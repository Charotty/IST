-- TimescaleDB schema for market_data (warm storage)
-- Run this once after creating the database

CREATE TABLE IF NOT EXISTS market_data (
    timestamp_ms BIGINT NOT NULL,
    symbol TEXT NOT NULL,
    type TEXT NOT NULL,
    exchange TEXT NOT NULL,
    data JSONB NOT NULL
);

-- Convert to hypertable (time-series)
SELECT create_hypertable('market_data', 'timestamp_ms', chunk_time_interval => 86400000);

-- Indexes for fast reads
CREATE INDEX IF NOT EXISTS idx_symbol_time ON market_data (symbol, timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_type_time ON market_data (type, timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_exchange_time ON market_data (exchange, timestamp_ms DESC);

-- Optional: compression policy for older data (7+ days)
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, type'
);
SELECT add_compression_policy('market_data', INTERVAL '7 days');
