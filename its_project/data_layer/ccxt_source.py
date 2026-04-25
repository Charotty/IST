from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional, Any, Dict, List

import ccxt.async_support as ccxt

from its_project.data_layer.base import BaseDataSource
from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


class CCXTDataSource(BaseDataSource):
    """CCXT-based data source supporting 100+ exchanges."""
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.config = config or {}
        self.exchange: Optional[ccxt.Exchange] = None
        self._symbols: List[str] = []
        
    async def connect(self) -> None:
        """Connect to the exchange via CCXT."""
        if self._connected:
            logger.warning(f"Already connected to {self.exchange_id}")
            return
        
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            
            config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                **self.config
            }
            
            if self.sandbox:
                config['sandboxMode'] = True
            
            self.exchange = exchange_class(config)
            
            # Load markets
            await self.exchange.load_markets()
            
            self._connected = True
            logger.info(f"Connected to {self.exchange_id} via CCXT")
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
        
        self._connected = False
        logger.info(f"Disconnected from {self.exchange_id}")
    
    async def subscribe(self, symbols: Optional[List[str]] = None) -> AsyncIterator[MarketData]:
        """Subscribe to market data updates (polling-based for CCXT)."""
        if not self._connected:
            await self.connect()
        
        if symbols:
            self._symbols = symbols
        
        if not self._symbols:
            logger.warning("No symbols to subscribe to")
            return
        
        # CCXT uses polling for most exchanges (WebSocket available for some)
        while self._connected:
            for symbol in self._symbols:
                try:
                    data = await self.fetch(symbol)
                    yield data
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            # Polling interval
            await asyncio.sleep(1.0)
    
    async def fetch(self, symbol: str, **params) -> MarketData:
        """Fetch current market data for a symbol."""
        if not self._connected:
            await self.connect()
        
        try:
            # Fetch ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch order book
            orderbook = await self.exchange.fetch_order_book(symbol, limit=20)
            
            # Create MarketData with correct structure
            from its_project.common.types import MarketDataType
            
            market_data = MarketData(
                timestamp_ms=ticker.get('timestamp', 0),
                symbol=symbol,
                type=MarketDataType.TICKER,
                exchange=self.exchange_id,
                data={
                    'open': ticker.get('open'),
                    'high': ticker.get('high'),
                    'low': ticker.get('low'),
                    'close': ticker.get('last'),
                    'volume': ticker.get('baseVolume'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'bid_volume': ticker.get('bidVolume'),
                    'ask_volume': ticker.get('askVolume'),
                    'orderbook': {
                        'bids': orderbook.get('bids', [])[:10],
                        'asks': orderbook.get('asks', [])[:10]
                    }
                }
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 100,
        since: Optional[int] = None
    ) -> List[List[float]]:
        """Fetch OHLCV (candlestick) data."""
        if not self._connected:
            await self.connect()
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=limit
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise
    
    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 100,
        since: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades."""
        if not self._connected:
            await self.connect()
        
        try:
            trades = await self.exchange.fetch_trades(
                symbol,
                since=since,
                limit=limit
            )
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise
    
    async def is_alive(self) -> bool:
        """Check if connection is alive."""
        if not self._connected or not self.exchange:
            return False
        
        try:
            # Try to fetch a ticker to check connection
            if self._symbols:
                await self.exchange.fetch_ticker(self._symbols[0])
            else:
                # Fetch BTC/USDT as default
                await self.exchange.fetch_ticker('BTC/USDT')
            return True
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False
    
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self._connected:
            await self.connect()
        
        if not self.exchange:
            return []
        
        return list(self.exchange.markets.keys())
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        if not self._connected:
            await self.connect()
        
        if not self.exchange:
            return {}
        
        return {
            'id': self.exchange.id,
            'name': self.exchange.name,
            'has': self.exchange.has,
            'timeframes': self.exchange.timeframes,
            'fees': self.exchange.fees
        }


class CCXTMultiExchangeSource(BaseDataSource):
    """CCXT source supporting multiple exchanges simultaneously."""
    
    def __init__(self, exchange_configs: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.exchange_configs = exchange_configs
        self.exchanges: Dict[str, CCXTDataSource] = {}
        
    async def connect(self) -> None:
        """Connect to all configured exchanges."""
        for config in self.exchange_configs:
            exchange_id = config.get('exchange_id', 'binance')
            source = CCXTDataSource(
                exchange_id=exchange_id,
                api_key=config.get('api_key'),
                secret=config.get('secret'),
                sandbox=config.get('sandbox', False),
                config=config.get('config', {})
            )
            await source.connect()
            self.exchanges[exchange_id] = source
        
        self._connected = True
        logger.info(f"Connected to {len(self.exchanges)} exchanges")
    
    async def disconnect(self) -> None:
        """Disconnect from all exchanges."""
        for source in self.exchanges.values():
            await source.disconnect()
        
        self.exchanges.clear()
        self._connected = False
    
    async def subscribe(self, symbols: Optional[List[str]] = None) -> AsyncIterator[MarketData]:
        """Subscribe to market data from all exchanges."""
        if not self._connected:
            await self.connect()
        
        # Yield data from all exchanges
        while self._connected:
            for exchange_id, source in self.exchanges.items():
                try:
                    async for data in source.subscribe(symbols):
                        # Add exchange_id to data
                        data.exchange_id = exchange_id
                        yield data
                except Exception as e:
                    logger.error(f"Error in {exchange_id}: {e}")
            
            await asyncio.sleep(0.5)
    
    async def fetch(self, symbol: str, exchange_id: Optional[str] = None, **params) -> MarketData:
        """Fetch data from specific exchange or first available."""
        if not self._connected:
            await self.connect()
        
        if exchange_id and exchange_id in self.exchanges:
            return await self.exchanges[exchange_id].fetch(symbol, **params)
        elif self.exchanges:
            # Use first available exchange
            first_exchange = next(iter(self.exchanges.values()))
            return await first_exchange.fetch(symbol, **params)
        else:
            raise RuntimeError("No exchanges available")
    
    async def is_alive(self) -> bool:
        """Check if any exchange is alive."""
        if not self._connected:
            return False
        
        for source in self.exchanges.values():
            if await source.is_alive():
                return True
        
        return False


# Convenience functions for common exchanges
async def create_binance_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False
) -> CCXTDataSource:
    """Create Binance data source via CCXT."""
    source = CCXTDataSource(
        exchange_id="binance",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox
    )
    await source.connect()
    return source


async def create_kraken_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False
) -> CCXTDataSource:
    """Create Kraken data source via CCXT."""
    source = CCXTDataSource(
        exchange_id="kraken",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox
    )
    await source.connect()
    return source


async def create_coinbase_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False
) -> CCXTDataSource:
    """Create Coinbase data source via CCXT."""
    source = CCXTDataSource(
        exchange_id="coinbase",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox
    )
    await source.connect()
    return source


async def create_bybit_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False
) -> CCXTDataSource:
    """Create Bybit data source via CCXT."""
    source = CCXTDataSource(
        exchange_id="bybit",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox
    )
    await source.connect()
    return source
