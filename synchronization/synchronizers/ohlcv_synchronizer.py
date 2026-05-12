"""
OHLCV Synchronizer

Синхронизация OHLCV данных в единые временные ряды.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_synchronizer import BaseSynchronizer


class OHLCVSynchronizer(BaseSynchronizer):
    """Синхронизатор OHLCV данных"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация OHLCV синхронизатора
        
        Args:
            config: Конфигурация синхронизатора
        """
        super().__init__(config)
        
        # Метод агрегации OHLCV
        self.ohlcv_method = config.get('ohlcv_method', 'standard')
        
        # Текущие агрегированные данные для каждого timestep
        self.current_candles = {}
        
        # История синхронизированных свечей
        self.synchronized_candles = []
    
    async def synchronize(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Синхронизация OHLCV данных
        
        Args:
            data: Входные OHLCV данные
            
        Returns:
            Dict: Синхронизированные данные или None
        """
        try:
            # Валидация данных
            if not self._validate_ohlcv_data(data):
                return None
            
            timestamp = data['timestamp']
            
            # Валидация временной метки
            if not self._validate_timestamp(timestamp):
                self.logger.warning(f"Invalid timestamp: {timestamp}")
                return None
            
            # Округление до базового timestep
            rounded_timestamp = self._round_timestamp(timestamp)
            
            # Детекция пропусков
            if self._last_timestamp and self._detect_gap(rounded_timestamp, self._last_timestamp):
                self._update_metrics('gap')
                await self._handle_gap(rounded_timestamp, self._last_timestamp)
            
            # Агрегация данных в свече
            candle = await self._aggregate_candle(data, rounded_timestamp)
            
            # Обновление метрик
            self._update_metrics('message')
            self._last_timestamp = rounded_timestamp
            
            return candle
            
        except Exception as e:
            self.logger.error(f"Error in OHLCV synchronization: {e}")
            return None
    
    async def get_synchronized_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированного потока OHLCV данных
        
        Yields:
            Dict: Синхронизированные OHLCV данные
        """
        while self._running:
            try:
                # Получение текущих свечей
                current_time = self._round_timestamp(datetime.utcnow())
                
                if current_time in self.current_candles:
                    candle = self.current_candles[current_time]
                    
                    # Проверка completeness свечи
                    if self._is_candle_complete(candle, current_time):
                        yield candle
                        
                        # Удаление из текущих
                        del self.current_candles[current_time]
                        
                        # Добавление в историю
                        self.synchronized_candles.append(candle)
                        
                        # Ограничение истории
                        if len(self.synchronized_candles) > 1000:
                            self.synchronized_candles = self.synchronized_candles[-1000:]
                
                await asyncio.sleep(self.timestep_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in synchronized stream: {e}")
                await asyncio.sleep(1)
    
    def _validate_ohlcv_data(self, data: Dict[str, Any]) -> bool:
        """
        Валидация OHLCV данных
        
        Args:
            data: Данные для валидации
            
        Returns:
            bool: True если данные валидны
        """
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Проверка логических соотношений
        try:
            open_price = float(data['open'])
            high_price = float(data['high'])
            low_price = float(data['low'])
            close_price = float(data['close'])
            volume = float(data['volume'])
            
            # low ≤ high
            if low_price > high_price:
                return False
            
            # open, close должны быть в диапазоне [low, high]
            if not (low_price <= open_price <= high_price):
                return False
            
            if not (low_price <= close_price <= high_price):
                return False
            
            # Объем не отрицательный
            if volume < 0:
                return False
            
            # Цена не нулевая и не отрицательная
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
                return False
            
        except (ValueError, TypeError):
            return False
        
        return True
    
    async def _aggregate_candle(self, data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """
        Агрегация данных в свече
        
        Args:
            data: Входные данные
            timestamp: Временная метка свечи
            
        Returns:
            Dict: Агрегированная свеча
        """
        if timestamp not in self.current_candles:
            # Новая свеча
            self.current_candles[timestamp] = {
                'timestamp': timestamp,
                'open': float(data['open']),
                'high': float(data['high']),
                'low': float(data['low']),
                'close': float(data['close']),
                'volume': float(data['volume']),
                'trades_count': 1,
                'source': data.get('source', 'unknown')
            }
        else:
            # Обновление существующей свечи
            candle = self.current_candles[timestamp]
            
            # Обновление OHLC
            candle['high'] = max(candle['high'], float(data['high']))
            candle['low'] = min(candle['low'], float(data['low']))
            candle['close'] = float(data['close'])
            candle['volume'] += float(data['volume'])
            candle['trades_count'] += 1
        
        return self.current_candles[timestamp]
    
    def _is_candle_complete(self, candle: Dict[str, Any], current_time: datetime) -> bool:
        """
        Проверка завершенности свечи
        
        Args:
            candle: Свеча для проверки
            current_time: Текущее время
            
        Returns:
            bool: True если свеча завершена
        """
        # Свеча считается завершенной если прошло время следующего timestep
        candle_time = candle['timestamp']
        next_candle_time = candle_time + timedelta(seconds=self.timestep_seconds)
        
        return current_time >= next_candle_time
    
    async def _handle_gap(self, current_timestamp: datetime, previous_timestamp: datetime) -> None:
        """
        Обработка пропуска в данных
        
        Args:
            current_timestamp: Текущая временная метка
            previous_timestamp: Предыдущая временная метка
        """
        gap_seconds = (current_timestamp - previous_timestamp).total_seconds()
        gap_steps = int(gap_seconds / self.timestep_seconds)
        
        self.logger.warning(f"Detected gap: {gap_steps} timesteps ({gap_seconds}s)")
        
        # Заполнение пропуска последней известной ценой
        if previous_timestamp in self.current_candles:
            last_candle = self.current_candles[previous_timestamp]
            
            for i in range(1, gap_steps):
                gap_timestamp = previous_timestamp + timedelta(seconds=i * self.timestep_seconds)
                
                # Forward fill
                fill_candle = {
                    'timestamp': gap_timestamp,
                    'open': last_candle['close'],
                    'high': last_candle['close'],
                    'low': last_candle['close'],
                    'close': last_candle['close'],
                    'volume': 0.0,
                    'trades_count': 0,
                    'source': 'gap_fill',
                    'gap_filled': True
                }
                
                self.current_candles[gap_timestamp] = fill_candle
                self._update_metrics('gap_filled')
    
    def get_latest_candles(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Получение последних синхронизированных свечей
        
        Args:
            count: Количество свечей
            
        Returns:
            List[Dict]: Последние свечи
        """
        return self.synchronized_candles[-count:] if self.synchronized_candles else []
    
    def get_candle_at_time(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Получение свечи для указанного времени
        
        Args:
            timestamp: Временная метка
            
        Returns:
            Dict: Свеча или None
        """
        rounded_timestamp = self._round_timestamp(timestamp)
        
        # Поиск в текущих свечах
        if rounded_timestamp in self.current_candles:
            return self.current_candles[rounded_timestamp]
        
        # Поиск в истории
        for candle in reversed(self.synchronized_candles):
            if candle['timestamp'] == rounded_timestamp:
                return candle
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики синхронизации
        
        Returns:
            Dict: Статистика
        """
        if not self.synchronized_candles:
            return {}
        
        prices = [candle['close'] for candle in self.synchronized_candles]
        volumes = [candle['volume'] for candle in self.synchronized_candles]
        
        return {
            'total_candles': len(self.synchronized_candles),
            'price_change': prices[-1] - prices[0] if len(prices) > 1 else 0,
            'price_change_pct': ((prices[-1] / prices[0]) - 1) * 100 if len(prices) > 1 and prices[0] > 0 else 0,
            'avg_volume': np.mean(volumes) if volumes else 0,
            'total_volume': sum(volumes),
            'latest_timestamp': self.synchronized_candles[-1]['timestamp'] if self.synchronized_candles else None
        }
