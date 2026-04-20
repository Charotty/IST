from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


class ParquetStore:
    """Parquet storage for raw LOB and high-frequency data."""
    
    def __init__(
        self,
        base_path: str | Path,
        partition_cols: List[str] = None,
        compression: str = "snappy",
        row_group_size: int = 100000
    ) -> None:
        self.base_path = Path(base_path)
        self.partition_cols = partition_cols or ["symbol", "date"]
        self.compression = compression
        self.row_group_size = row_group_size
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset
        self._initialize_dataset()
    
    def _initialize_dataset(self) -> None:
        """Initialize the Parquet dataset schema."""
        schema = pa.schema([
            pa.field("timestamp_ms", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("data_type", pa.string()),
            pa.field("exchange", pa.string()),
            pa.field("data", pa.struct([
                pa.field("_kind", pa.string()),
                pa.field("_reconstructed", pa.bool_()),
                pa.field("lastUpdateId", pa.int64()),
                pa.field("bids", pa.list_(pa.list_(pa.string()))),
                pa.field("asks", pa.list_(pa.list_(pa.string()))),
                pa.field("_ts_recv_ms", pa.int64()),
                pa.field("p", pa.string()),
                pa.field("q", pa.string()),
                pa.field("t", pa.int64())
            ]))
        ])
        
        # Create a dummy dataset to establish schema
        sample_data = pa.Table.from_arrays([
            [0],  # timestamp_ms
            ["BTCUSDT"],  # symbol
            ["orderbook"],  # data_type
            ["binance"],  # exchange
            [{}]  # data
        ], schema=schema)
        
        try:
            ds.write_dataset(
                sample_data,
                base_dir=self.base_path,
                format="parquet",
                partitioning=self.partition_cols,
                existing_data_behavior="overwrite_or_ignore"
            )
        except Exception as e:
            logger.debug(f"Dataset initialization note: {e}")
    
    def write_raw_lob(self, data: MarketData) -> None:
        """Write raw LOB data to Parquet."""
        # Add date partition column
        date_str = datetime.fromtimestamp(data.timestamp_ms / 1000).strftime("%Y-%m-%d")
        
        # Convert to Arrow Table
        table = self._marketdata_to_table([data], [date_str])
        
        # Write to dataset
        ds.write_dataset(
            table,
            base_dir=self.base_path,
            format="parquet",
            partitioning=self.partition_cols,
            existing_data_behavior="overwrite_or_ignore"
        )
    
    def write_batch_raw_lob(self, data_list: Sequence[MarketData]) -> None:
        """Write batch of raw LOB data."""
        if not data_list:
            return
        
        # Add date partition column for each record
        date_strs = [
            datetime.fromtimestamp(md.timestamp_ms / 1000).strftime("%Y-%m-%d")
            for md in data_list
        ]
        
        # Convert to Arrow Table
        table = self._marketdata_to_table(data_list, date_strs)
        
        # Write to dataset
        ds.write_dataset(
            table,
            base_dir=self.base_path,
            format="parquet",
            partitioning=self.partition_cols,
            existing_data_behavior="overwrite_or_ignore"
        )
        
        logger.info(f"Wrote {len(data_list)} records to Parquet")
    
    def read_raw_lob(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[List[Dict[str, Any]]] = None
    ) -> List[MarketData]:
        """Read raw LOB data from Parquet."""
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        
        # Build filters
        base_filters = [
            ("symbol", "=", symbol),
            ("date", ">=", start_date),
            ("date", "<=", end_date)
        ]
        
        if filters:
            base_filters.extend(filters)
        
        # Read dataset
        dataset = ds.dataset(self.base_path, format="parquet", partitioning=self.partition_cols)
        table = dataset.to_table(filter=ds.field("timestamp_ms") >= int(start_time.timestamp() * 1000))
        table = table.filter(ds.field("timestamp_ms") <= int(end_time.timestamp() * 1000))
        
        # Convert to MarketData objects
        return self._table_to_marketdata(table)
    
    def read_latest_raw_lob(
        self,
        symbol: str,
        limit: int = 1000
    ) -> List[MarketData]:
        """Read latest raw LOB data for symbol."""
        # Read latest partition
        latest_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            dataset = ds.dataset(self.base_path, format="parquet", partitioning=self.partition_cols)
            table = dataset.to_table(
                filter=(
                    (ds.field("symbol") == symbol) & 
                    (ds.field("date") == latest_date)
                )
            )
            
            # Sort by timestamp and limit
            table = table.sort_by("timestamp_ms", order="descending")
            table = table.slice(0, limit)
            
            return self._table_to_marketdata(table)
            
        except Exception as e:
            logger.error(f"Error reading latest raw LOB: {e}")
            return []
    
    def _marketdata_to_table(self, data_list: Sequence[MarketData], date_strs: List[str]) -> pa.Table:
        """Convert MarketData list to Arrow Table."""
        timestamps = [md.timestamp_ms for md in data_list]
        symbols = [md.symbol for md in data_list]
        data_types = [md.type.value for md in data_list]
        exchanges = [md.exchange for md in data_list]
        
        # Convert data dict to struct
        data_structs = []
        for md in data_list:
            data_dict = dict(md.data)
            # Ensure all required fields are present
            data_dict.setdefault("_kind", "")
            data_dict.setdefault("_reconstructed", False)
            data_dict.setdefault("lastUpdateId", 0)
            data_dict.setdefault("bids", [])
            data_dict.setdefault("asks", [])
            data_dict.setdefault("_ts_recv_ms", md.timestamp_ms)
            data_dict.setdefault("p", "")
            data_dict.setdefault("q", "")
            data_dict.setdefault("t", 0)
            
            data_structs.append(data_dict)
        
        # Create struct array
        data_array = pa.array(data_structs, type=pa.struct([
            pa.field("_kind", pa.string()),
            pa.field("_reconstructed", pa.bool_()),
            pa.field("lastUpdateId", pa.int64()),
            pa.field("bids", pa.list_(pa.list_(pa.string()))),
            pa.field("asks", pa.list_(pa.list_(pa.string()))),
            pa.field("_ts_recv_ms", pa.int64()),
            pa.field("p", pa.string()),
            pa.field("q", pa.string()),
            pa.field("t", pa.int64())
        ]))
        
        return pa.Table.from_arrays([
            timestamps,
            symbols,
            data_types,
            exchanges,
            data_array
        ], names=["timestamp_ms", "symbol", "data_type", "exchange", "data"])
    
    def _table_to_marketdata(self, table: pa.Table) -> List[MarketData]:
        """Convert Arrow Table to MarketData list."""
        results = []
        
        for i in range(len(table)):
            timestamp_ms = table.column("timestamp_ms")[i].as_py()
            symbol = table.column("symbol")[i].as_py()
            data_type = table.column("data_type")[i].as_py()
            exchange = table.column("exchange")[i].as_py()
            data_dict = table.column("data")[i].as_py()
            
            md = MarketData(
                timestamp_ms=timestamp_ms,
                symbol=symbol,
                type=MarketDataType(data_type),
                exchange=exchange,
                data=data_dict
            )
            results.append(md)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            dataset = ds.dataset(self.base_path, format="parquet", partitioning=self.partition_cols)
            
            # Get file sizes
            total_size = 0
            file_count = 0
            
            for file_path in self.base_path.rglob("*.parquet"):
                total_size += file_path.stat().st_size
                file_count += 1
            
            # Get row count (sample)
            try:
                sample_table = dataset.head(1000)
                sample_rows = len(sample_table)
                total_rows_estimated = sample_rows * file_count  # Rough estimate
            except:
                total_rows_estimated = 0
            
            # Get unique symbols
            try:
                unique_symbols = set()
                for partition in dataset.partitions.dictionaries:
                    if "symbol" in partition:
                        unique_symbols.update(partition["symbol"])
            except:
                unique_symbols = set()
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count,
                "estimated_total_rows": total_rows_estimated,
                "unique_symbols": len(unique_symbols),
                "base_path": str(self.base_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data partitions."""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        removed_count = 0
        for date_dir in self.base_path.glob("symbol=*"):
            date_path = date_dir / f"date={cutoff_str}"
            if date_path.exists() and date_path.is_dir():
                # Remove old date partition
                import shutil
                shutil.rmtree(date_path)
                removed_count += 1
                logger.info(f"Removed old partition: {date_path}")
        
        logger.info(f"Cleanup completed. Removed {removed_count} old partitions")
    
    def optimize_dataset(self) -> None:
        """Optimize Parquet dataset by consolidating small files."""
        try:
            dataset = ds.dataset(self.base_path, format="parquet", partitioning=self.partition_cols)
            
            # Find partitions with many small files
            for partition in dataset.partitions.partitioning:
                partition_path = self.base_path / partition
                if partition_path.exists():
                    files = list(partition_path.glob("*.parquet"))
                    if len(files) > 10:  # If many small files, consolidate
                        # Read and rewrite the partition
                        table = dataset.to_table(filter=partition)
                        ds.write_dataset(
                            table,
                            base_dir=self.base_path,
                            format="parquet",
                            partitioning=self.partition_cols,
                            existing_data_behavior="overwrite_specific_partition"
                        )
                        logger.info(f"Optimized partition: {partition}")
        
        except Exception as e:
            logger.error(f"Error optimizing dataset: {e}")
