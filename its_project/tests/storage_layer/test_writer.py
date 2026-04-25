"""
Unit tests for batch_writer_task.
"""
import pytest
import asyncio
from its_project.storage.writer import batch_writer_task
from its_project.common.types import MarketData, MarketDataType


# Mock storage for testing
class MockStorage:
    """Mock storage for testing."""
    
    def __init__(self):
        self.written_batches = []
    
    async def write_batch(self, data):
        self.written_batches.append(list(data))
        return len(data)


@pytest.mark.unit
@pytest.mark.storage_layer
class TestBatchWriterTask:
    """Test batch_writer_task functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_size_flush(self):
        """Test flushing when batch size is reached."""
        storage = MockStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=3,
            max_interval_s=10.0,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Add items
        for i in range(3):
            await queue.put(MarketData(
                timestamp_ms=1704067200000 + i,
                symbol="BTC/USDT",
                type=MarketDataType.KLINE,
                exchange="binance",
                data={"price": 42000.0 + i}
            ))
        
        # Wait for batch to be written
        await asyncio.sleep(0.1)
        
        # Stop the task
        stop_event.set()
        await task
        
        assert len(storage.written_batches) == 1
        assert len(storage.written_batches[0]) == 3
    
    @pytest.mark.asyncio
    async def test_interval_flush(self):
        """Test flushing when max interval is reached."""
        storage = MockStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task with short interval
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=10,
            max_interval_s=0.2,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Add one item (less than batch size)
        await queue.put(MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"price": 42000.0}
        ))
        
        # Wait for interval to trigger flush
        await asyncio.sleep(0.3)
        
        # Stop the task
        stop_event.set()
        await task
        
        assert len(storage.written_batches) == 1
        assert len(storage.written_batches[0]) == 1
    
    @pytest.mark.asyncio
    async def test_stop_event_flushes_buffer(self):
        """Test that stop event flushes remaining buffer."""
        storage = MockStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=10,
            max_interval_s=10.0,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Add a few items (less than batch size)
        for i in range(2):
            await queue.put(MarketData(
                timestamp_ms=1704067200000 + i,
                symbol="BTC/USDT",
                type=MarketDataType.KLINE,
                exchange="binance",
                data={"price": 42000.0 + i}
            ))
        
        # Stop the task
        stop_event.set()
        await task
        
        # Buffer should be flushed on stop
        assert len(storage.written_batches) == 1
        assert len(storage.written_batches[0]) == 2
    
    @pytest.mark.asyncio
    async def test_multiple_batches(self):
        """Test writing multiple batches."""
        storage = MockStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=2,
            max_interval_s=10.0,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Add 5 items (should create 3 batches: 2, 2, 1)
        for i in range(5):
            await queue.put(MarketData(
                timestamp_ms=1704067200000 + i,
                symbol="BTC/USDT",
                type=MarketDataType.KLINE,
                exchange="binance",
                data={"price": 42000.0 + i}
            ))
        
        await asyncio.sleep(0.1)
        
        # Stop the task
        stop_event.set()
        await task
        
        assert len(storage.written_batches) == 3
        assert len(storage.written_batches[0]) == 2
        assert len(storage.written_batches[1]) == 2
        assert len(storage.written_batches[2]) == 1
    
    @pytest.mark.asyncio
    async def test_empty_buffer_on_stop(self):
        """Test stopping with empty buffer."""
        storage = MockStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=10,
            max_interval_s=10.0,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Stop immediately without adding items
        stop_event.set()
        await task
        
        # No writes should occur
        assert len(storage.written_batches) == 0
    
    @pytest.mark.asyncio
    async def test_storage_write_error_handling(self):
        """Test that storage write errors are handled gracefully."""
        class FailingStorage:
            def __init__(self):
                self.write_count = 0
            
            async def write_batch(self, data):
                self.write_count += 1
                raise Exception("Write failed")
        
        storage = FailingStorage()
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Create writer task
        task = asyncio.create_task(batch_writer_task(
            name="test",
            storage=storage,
            batch_size=2,
            max_interval_s=10.0,
            in_queue=queue,
            stop_event=stop_event
        ))
        
        # Add items
        for i in range(2):
            await queue.put(MarketData(
                timestamp_ms=1704067200000 + i,
                symbol="BTC/USDT",
                type=MarketDataType.KLINE,
                exchange="binance",
                data={"price": 42000.0 + i}
            ))
        
        await asyncio.sleep(0.1)
        
        # Stop the task
        stop_event.set()
        await task
        
        # Write was attempted despite error
        assert storage.write_count >= 1
