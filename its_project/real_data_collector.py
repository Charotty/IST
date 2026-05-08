#!/usr/bin/env python3
"""
Сборщик реальных рыночных данных для обучения моделей.
Использует CCXT для сбора данных с бирж.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RealDataCollector:
    """Сборщик реальных рыночных данных."""
    
    def __init__(self, exchange_name: str = "binance"):
        """Инициализация сборщика."""
        self.exchange_name = exchange_name
        self.exchange = None
        self.connect_exchange()
    
    def connect_exchange(self):
        """Подключение к бирже."""
        try:
            ex_name = self.exchange_name.lower()
            exchange_cls = getattr(ccxt, ex_name, None)
            if exchange_cls is None:
                logger.warning(f"Неизвестная биржа '{self.exchange_name}', используется Binance")
                exchange_cls = ccxt.binance

            options: Dict[str, object] = {'enableRateLimit': True}
            # Common defaults
            if ex_name in {"binance", "okx"}:
                options['options'] = {'defaultType': 'spot'}

            self.exchange = exchange_cls(options)
            
            logger.info(f"Подключено к бирже: {self.exchange.name}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка подключения к бирже: {e}")
            return False
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Сбор OHLCV данных."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Собрано {len(df)} свечей для {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка сбора OHLCV данных: {e}")
            return pd.DataFrame()
    
    def fetch_orderbook_data(self, symbol: str, limit: int = 100) -> Dict:
        """Сбор данных order book."""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Ошибка сбора order book: {e}")
            return {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов."""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # SMA и EMA
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Price Change
            df['price_change'] = df['close'].pct_change()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Удаляем NaN значения
            df.dropna(inplace=True)
            
            logger.info(f"Рассчитано технических индикаторов для {len(df)} свечей")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
    
    def create_orderbook_features(self, orderbook: Dict) -> Dict:
        """Создание признаков из order book."""
        if not orderbook:
            return {}
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return {}
            
            # Bid-Ask spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            # Order imbalance
            bid_volume = sum(bid[1] for bid in bids[:10])
            ask_volume = sum(ask[1] for ask in asks[:10])
            total_volume = bid_volume + ask_volume
            order_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Mid price
            mid_price = (best_bid + best_ask) / 2
            
            # Weighted mid price
            weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / total_volume if total_volume > 0 else mid_price
            
            return {
                'bid_ask_spread': spread,
                'bid_ask_spread_pct': spread_pct,
                'order_imbalance': order_imbalance,
                'mid_price': mid_price,
                'weighted_mid_price': weighted_mid,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
        except Exception as e:
            logger.error(f"Ошибка создания order book признаков: {e}")
            return {}
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание sentiment признаков на основе цены."""
        try:
            # Price momentum
            df['price_momentum_5'] = df['close'].pct_change(5, fill_method=None)
            df['price_momentum_10'] = df['close'].pct_change(10, fill_method=None)
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Trend direction (простой sentiment)
            df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
            df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
            
            # RSI sentiment
            df['rsi_sentiment'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
            
            # MACD sentiment
            df['macd_sentiment'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            
            # Combined sentiment score
            df['sentiment_score'] = (
                df['trend_5'] * 0.3 + 
                df['rsi_sentiment'] * 0.4 + 
                df['macd_sentiment'] * 0.3
            )
            
            logger.info("Созданы sentiment признаки")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания sentiment признаков: {e}")
            return df
    
    def create_trading_labels(self, df: pd.DataFrame, profit_threshold: float = 0.01, 
                            loss_threshold: float = -0.01) -> pd.DataFrame:
        """Создание меток для обучения (SELL/HOLD/BUY)."""
        try:
            # Future returns
            df['future_return_5'] = df['close'].shift(-5).pct_change(5, fill_method=None)
            df['future_return_10'] = df['close'].shift(-10).pct_change(10, fill_method=None)
            
            # Создание меток
            conditions = []
            
            # BUY signals
            buy_conditions = (
                (df['future_return_5'] > profit_threshold) |
                (df['future_return_10'] > profit_threshold)
            )
            conditions.append(buy_conditions)
            
            # SELL signals
            sell_conditions = (
                (df['future_return_5'] < loss_threshold) |
                (df['future_return_10'] < loss_threshold)
            )
            conditions.append(sell_conditions)
            
            # HOLD signals (все остальное)
            hold_conditions = ~(buy_conditions | sell_conditions)
            conditions.append(hold_conditions)
            
            # Создаем метки: 0=SELL, 1=HOLD, 2=BUY
            df['label'] = np.select(conditions, [0, 2, 1], default=1)
            
            # Удаляем последние строки с NaN
            df.dropna(inplace=True)
            
            # Статистика меток
            label_counts = df['label'].value_counts()
            total_labels = len(df)
            
            logger.info(f"Созданы метки:")
            logger.info(f"  SELL: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total_labels*100:.1f}%)")
            logger.info(f"  HOLD: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total_labels*100:.1f}%)")
            logger.info(f"  BUY: {label_counts.get(2, 0)} ({label_counts.get(2, 0)/total_labels*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания меток: {e}")
            return df
    
    def collect_training_data(self, symbol: str = "BTC/USDT", timeframe: str = "1h", 
                            days: int = 30) -> pd.DataFrame:
        """Сбор полных данных для обучения."""
        logger.info(f"Сбор данных для {symbol} ({timeframe}) за {days} дней")
        
        # Расчет количества свечей
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = timeframe_minutes.get(timeframe, 60)
        total_candles = min(days * 24 * 60 // minutes, 1000)  # Ограничение CCXT
        
        # Сбор OHLCV данных
        df = self.fetch_ohlcv_data(symbol, timeframe, total_candles)
        if df.empty:
            return df
        
        # Расчет индикаторов
        df = self.calculate_technical_indicators(df)
        
        # Создание sentiment признаков
        df = self.create_sentiment_features(df)
        
        # Создание меток
        df = self.create_trading_labels(df)
        
        # Сохранение данных
        self.save_training_data(df, symbol, timeframe)
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Сохранение данных для обучения."""
        try:
            # Создаем директорию
            data_dir = Path("real_training_data")
            data_dir.mkdir(exist_ok=True)
            
            # Имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.csv"
            filepath = data_dir / filename
            
            # Сохранение
            df.to_csv(filepath)
            logger.info(f"Данные сохранены в {filepath}")
            
            # Также сохраняем как Parquet для эффективности
            parquet_path = filepath.with_suffix('.parquet')
            df.to_parquet(parquet_path)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")
            return None
    
    def get_feature_columns(self) -> List[str]:
        """Получение списка признаков для обучения."""
        return [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'sma_20', 'ema_20', 'atr',
            'price_change', 'volume_ratio', 'price_momentum_5', 'price_momentum_10',
            'volatility', 'trend_5', 'trend_10', 'rsi_sentiment', 'macd_sentiment',
            'sentiment_score'
        ]

def main():
    """Тестовый запуск сборщика данных."""
    logging.basicConfig(level=logging.INFO)
    
    collector = RealDataCollector("binance")
    
    # Сбор данных для BTC/USDT
    data = collector.collect_training_data("BTC/USDT", "1h", days=30)
    
    if not data.empty:
        print(f"Собрано {len(data)} строк данных")
        print(f"Признаки: {len(collector.get_feature_columns())}")
        print("\nПоследние 5 строк:")
        print(data[['close', 'rsi', 'macd', 'sentiment_score', 'label']].tail())
        
        print(f"\nРаспределение меток:")
        print(data['label'].value_counts())

if __name__ == "__main__":
    main()
