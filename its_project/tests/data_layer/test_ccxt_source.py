"""
Unit tests for CCXT data source.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from its_project.data_layer.ccxt_source import CCXTDataSource, create_binance_source


@pytest.mark.unit
@pytest.mark.data_layer
class TestCCXTDataSource:
    """Test CCXTDataSource functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test data source initialization."""
        source = CCXTDataSource(
            exchange_id='binance',
            api_key='test_key',
            secret='test_secret',
            sandbox=True
        )
        
        assert source.exchange_id == 'binance'
        assert source.api_key == 'test_key'
        assert source.secret == 'test_secret'
        assert source.sandbox is True
        assert source._connected is False
    
    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connection to exchange."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            
            assert source._connected is True
            assert source.exchange is not None
            mock_exchange.load_markets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from exchange."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.close = AsyncMock()
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            await source.disconnect()
            
            assert source._connected is False
            mock_exchange.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_ticker(self):
        """Test fetching ticker data."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.fetch_ticker = AsyncMock(return_value={
                'timestamp': 1704067200000,
                'open': 41900.0,
                'high': 42100.0,
                'low': 41800.0,
                'last': 42000.0,
                'baseVolume': 1000.0,
                'bid': 41990.0,
                'ask': 42010.0,
                'bidVolume': 0.5,
                'askVolume': 0.6
            })
            mock_exchange.fetch_order_book = AsyncMock(return_value={
                'bids': [[41990.0, 0.5], [41980.0, 0.3]],
                'asks': [[42010.0, 0.6], [42020.0, 0.4]]
            })
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            data = await source.fetch('BTC/USDT')
            
            assert data.symbol == 'BTC/USDT'
            assert data.timestamp_ms == 1704067200000
            assert data.data['close'] == 42000.0
            assert data.data['bid'] == 41990.0
            assert data.data['ask'] == 42010.0
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self):
        """Test fetching OHLCV data."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.fetch_ohlcv = AsyncMock(return_value=[
                [1704067200000, 41900.0, 42100.0, 41800.0, 42000.0, 1000.0],
                [1704067260000, 42000.0, 42200.0, 41900.0, 42100.0, 1100.0]
            ])
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            ohlcv = await source.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=100)
            
            assert len(ohlcv) == 2
            assert ohlcv[0][4] == 42000.0  # Close price
    
    @pytest.mark.asyncio
    async def test_fetch_trades(self):
        """Test fetching recent trades."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.fetch_trades = AsyncMock(return_value=[
                {'id': '1', 'price': '42000.0', 'amount': '0.1', 'timestamp': 1704067200000},
                {'id': '2', 'price': '42010.0', 'amount': '0.2', 'timestamp': 1704067260000}
            ])
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            trades = await source.fetch_trades('BTC/USDT', limit=100)
            
            assert len(trades) == 2
            assert trades[0]['price'] == '42000.0'
    
    @pytest.mark.asyncio
    async def test_is_alive(self):
        """Test connection alive check."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.fetch_ticker = AsyncMock(return_value={'timestamp': 1704067200000})
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            is_alive = await source.is_alive()
            
            assert is_alive is True
    
    @pytest.mark.asyncio
    async def test_get_symbols(self):
        """Test getting available symbols."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.markets = {
                'BTC/USDT': {'id': 'BTCUSDT'},
                'ETH/USDT': {'id': 'ETHUSDT'}
            }
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            symbols = await source.get_symbols()
            
            assert 'BTC/USDT' in symbols
            assert 'ETH/USDT' in symbols
    
    @pytest.mark.asyncio
    async def test_get_exchange_info(self):
        """Test getting exchange information."""
        source = CCXTDataSource(exchange_id='binance')
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.id = 'binance'
            mock_exchange.name = 'Binance'
            mock_exchange.has = {'fetchOHLCV': True}
            mock_exchange.timeframes = {'1m': '1m', '5m': '5m'}
            mock_exchange.fees = {'trading': {'maker': 0.001}}
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            info = await source.get_exchange_info()
            
            assert info['id'] == 'binance'
            assert info['name'] == 'Binance'
            assert 'has' in info
            assert 'timeframes' in info
    
    @pytest.mark.asyncio
    async def test_subscribe_with_symbols(self):
        """Test subscribing to symbols."""
        source = CCXTDataSource(exchange_id='binance')
        source._symbols = ['BTC/USDT', 'ETH/USDT']
        
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.fetch_ticker = AsyncMock(return_value={'timestamp': 1704067200000})
            mock_exchange.fetch_order_book = AsyncMock(return_value={'bids': [], 'asks': []})
            mock_exchange_class.return_value = mock_exchange
            
            await source.connect()
            
            # Subscribe should yield data
            data_count = 0
            async for _ in source.subscribe(['BTC/USDT']):
                data_count += 1
                if data_count >= 2:
                    break
            
            assert data_count >= 1


@pytest.mark.unit
@pytest.mark.data_layer
class TestCCXTMultiExchangeSource:
    """Test CCXTMultiExchangeSource functionality."""
    
    @pytest.mark.asyncio
    async def test_multi_exchange_connect(self):
        """Test connecting to multiple exchanges."""
        from its_project.data_layer.ccxt_source import CCXTMultiExchangeSource
        
        configs = [
            {'exchange_id': 'binance', 'api_key': 'key1', 'secret': 'secret1'},
            {'exchange_id': 'kraken', 'api_key': 'key2', 'secret': 'secret2'}
        ]
        
        with patch('ccxt.async_support.binance') as mock_binance, \
             patch('ccxt.async_support.kraken') as mock_kraken:
            
            mock_exchange1 = AsyncMock()
            mock_exchange1.load_markets = AsyncMock()
            mock_binance.return_value = mock_exchange1
            
            mock_exchange2 = AsyncMock()
            mock_exchange2.load_markets = AsyncMock()
            mock_kraken.return_value = mock_exchange2
            
            source = CCXTMultiExchangeSource(exchange_configs=configs)
            await source.connect()
            
            assert source._connected is True
            assert len(source.exchanges) == 2
    
    @pytest.mark.asyncio
    async def test_multi_exchange_disconnect(self):
        """Test disconnecting from multiple exchanges."""
        from its_project.data_layer.ccxt_source import CCXTMultiExchangeSource
        
        configs = [{'exchange_id': 'binance'}]
        
        with patch('ccxt.async_support.binance') as mock_binance:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.close = AsyncMock()
            mock_binance.return_value = mock_exchange
            
            source = CCXTMultiExchangeSource(exchange_configs=configs)
            await source.connect()
            await source.disconnect()
            
            assert source._connected is False
            assert len(source.exchanges) == 0
