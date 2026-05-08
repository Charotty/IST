from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import ccxt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealMarketDataLoader:
    """
    Real market data loader for validation.
    
    Sources:
    - CCXT for live exchange data
    - Historical data files
    - Multiple timeframes
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Exchange configuration
        self.exchange_name = config.get("exchange", "binance")
        self.symbols = config.get("symbols", ["BTC/USDT", "ETH/USDT"])
        self.timeframes = config.get("timeframes", ["1m", "5m", "15m", "1h"])
        
        # Data parameters
        self.start_date = config.get("start_date", "2024-01-01")
        self.end_date = config.get("end_date", "2024-12-31")
        self.data_dir = Path(config.get("data_dir", "real_market_data"))
        
        # Validation parameters
        self.min_data_points = config.get("min_data_points", 1000)
        self.max_data_points = config.get("max_data_points", 100000)
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                },
            })
            
            # Test connection
            if not hasattr(exchange, 'fetch_ohlcv'):
                raise ValueError(f"Exchange {self.exchange_name} does not support OHLCV data")
            
            logger.info(f"Initialized {self.exchange_name} exchange")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_name}: {e}")
            raise
    
    def load_historical_data(self, symbol: str, timeframe: str, 
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        # Check if data exists locally
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.parquet"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"Loading cached data from {filepath}")
            return pd.read_parquet(filepath)
        
        # Fetch from exchange
        logger.info(f"Fetching {symbol} {timeframe} data from {self.exchange_name}")
        
        try:
            # Convert dates to timestamps
            since = self.exchange.parse8601(start_date)
            until = self.exchange.parse8601(end_date)
            
            # Fetch data in batches
            all_data = []
            current_since = since
            
            while current_since < until:
                # Fetch batch
                batch_data = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )
                
                if not batch_data:
                    break
                
                all_data.extend(batch_data)
                
                # Update since to last timestamp + 1
                current_since = batch_data[-1][0] + 1
                
                # Rate limiting
                self.exchange.sleep()
                
                # Prevent infinite loop
                if len(all_data) > self.max_data_points:
                    logger.warning(f"Reached max data points limit: {len(all_data)}")
                    break
            
            # Convert to DataFrame
            df = self._ohlcv_to_dataframe(batch_data)
            
            # Validate data
            self._validate_data(df, symbol, timeframe)
            
            # Save to cache
            df.to_parquet(filepath)
            logger.info(f"Saved {len(df)} data points to {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} {timeframe}: {e}")
            raise
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List[List]) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame."""
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort
        df = df.drop_duplicates().sort_index()
        
        # Remove any rows with invalid data
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) & (df['volume'] >= 0)]
        
        # Remove price anomalies
        df = self._remove_price_anomalies(df)
        
        return df
    
    def _remove_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove price anomalies and outliers."""
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        
        # Remove extreme price changes (> 20% in one period)
        extreme_changes = abs(df['price_change']) > 0.2
        if extreme_changes.any():
            logger.warning(f"Removing {extreme_changes.sum()} extreme price changes")
            df = df[~extreme_changes]
        
        # Remove zero volume periods (unless it's expected)
        if df['volume'].eq(0).any():
            logger.warning(f"Found {df['volume'].eq(0).sum()} zero volume periods")
        
        # Drop temporary column
        df = df.drop('price_change', axis=1)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Validate loaded data."""
        if len(df) < self.min_data_points:
            raise ValueError(f"Insufficient data for {symbol} {timeframe}: {len(df)} < {self.min_data_points}")
        
        # Check for gaps
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(self._get_expected_timedelta(timeframe))
        
        gaps = time_diff > expected_diff * 2
        if gaps.any():
            logger.warning(f"Found {gaps.sum()} gaps in {symbol} {timeframe} data")
        
        # Check data quality
        missing_data = df.isnull().sum()
        if missing_data.any():
            logger.warning(f"Missing data in {symbol} {timeframe}: {missing_data.to_dict()}")
        
        logger.info(f"Validated {len(df)} data points for {symbol} {timeframe}")
    
    def _get_expected_timedelta(self, timeframe: str) -> str:
        """Get expected time delta for timeframe."""
        timeframe_map = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day'
        }
        return timeframe_map.get(timeframe, '1 minute')
    
    def load_multiple_symbols(self, symbols: List[str] = None, 
                            timeframe: str = "1m") -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        symbols = symbols or self.symbols
        
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.load_historical_data(symbol, timeframe)
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        return data_dict
    
    def create_validation_dataset(self, symbol: str = "BTC/USDT", 
                               timeframe: str = "1m") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create validation dataset with proper train/test split.
        
        Returns:
            Tuple of (data, metadata)
        """
        # Load data
        data = self.load_historical_data(symbol, timeframe)
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_periods': len(data),
            'train_periods': int(len(data) * 0.7),
            'test_periods': int(len(data) * 0.3),
            'data_quality': self._assess_data_quality(data)
        }
        
        return data, metadata
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        # Basic statistics
        price_stats = {
            'mean_price': df['close'].mean(),
            'std_price': df['close'].std(),
            'min_price': df['close'].min(),
            'max_price': df['close'].max()
        }
        
        # Return statistics
        returns = df['close'].pct_change().dropna()
        return_stats = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max()
        }
        
        # Volume statistics
        volume_stats = {
            'mean_volume': df['volume'].mean(),
            'std_volume': df['volume'].std(),
            'zero_volume_periods': (df['volume'] == 0).sum()
        }
        
        # Data completeness
        completeness = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_periods': df.index.duplicated().sum(),
            'gaps': self._count_data_gaps(df)
        }
        
        return {
            'price_stats': price_stats,
            'return_stats': return_stats,
            'volume_stats': volume_stats,
            'completeness': completeness
        }
    
    def _count_data_gaps(self, df: pd.DataFrame) -> int:
        """Count data gaps."""
        if len(df) < 2:
            return 0
        
        # Expected frequency (assume 1 minute for now)
        expected_freq = pd.Timedelta(minutes=1)
        
        # Calculate actual frequency
        time_diffs = df.index.to_series().diff()
        gaps = (time_diffs > expected_freq * 2).sum()
        
        return gaps
    
    def get_market_statistics(self, symbol: str = "BTC/USDT", 
                            timeframe: str = "1m") -> Dict[str, Any]:
        """Get comprehensive market statistics."""
        data = self.load_historical_data(symbol, timeframe)
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Basic statistics
        basic_stats = {
            'total_periods': len(data),
            'date_range': {
                'start': data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            },
            'price_range': {
                'min': data['close'].min(),
                'max': data['close'].max(),
                'mean': data['close'].mean(),
                'std': data['close'].std()
            }
        }
        
        # Return statistics
        return_stats = {
            'daily_return_mean': returns.mean(),
            'daily_return_std': returns.std(),
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }
        
        # Trading statistics
        trading_stats = {
            'avg_volume': data['volume'].mean(),
            'total_volume': data['volume'].sum(),
            'volume_volatility': data['volume'].std(),
            'price_volume_correlation': data['close'].corr(data['volume'])
        }
        
        # Market efficiency metrics
        efficiency_metrics = {
            'autocorrelation_lag1': returns.autocorr(lag=1),
            'autocorrelation_lag5': returns.autocorr(lag=5),
            'autocorrelation_lag10': returns.autocorr(lag=10),
            'hurst_exponent': self._calculate_hurst_exponent(returns)
        }
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'basic_stats': basic_stats,
            'return_stats': return_stats,
            'trading_stats': trading_stats,
            'efficiency_metrics': efficiency_metrics
        }
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for market efficiency."""
        try:
            # Simplified Hurst exponent calculation
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(returns.values[lag:], returns.values[:-lag]))) for lag in lags]
            
            # Linear fit in log-log space
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            
            return hurst
        except:
            return 0.5  # Random walk
    
    def create_synthetic_data(self, n_samples: int = 10000, 
                           volatility: float = 0.02, 
                           drift: float = 0.0001) -> pd.DataFrame:
        """
        Create synthetic market data for testing.
        
        Args:
            n_samples: Number of samples to generate
            volatility: Daily volatility
            drift: Daily drift
            
        Returns:
            Synthetic OHLCV data
        """
        np.random.seed(42)
        
        # Generate price series
        returns = np.random.normal(drift, volatility, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        high_low_range = 0.02  # 2% typical intraday range
        
        opens = prices[:-1]
        closes = prices[1:]
        
        # Generate high and low
        highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, high_low_range, len(opens)))
        lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, high_low_range, len(opens)))
        
        # Generate volume
        volumes = np.random.lognormal(10, 1, len(opens))
        
        # Create DataFrame
        timestamps = pd.date_range(start='2024-01-01', periods=len(opens), freq='1min')
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=timestamps)
        
        return df
