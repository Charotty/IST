"""
Trades Synchronizer

Синхронизация сделок в единые временные ряды.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_synchronizer import BaseSynchronizer


class TradesSynchronizer(BaseSynchronizer):
    """Синхронизатор сделок"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация синхронизатора сделок
        
        Args:
            config: Конфигурация синхронизатора
        """
        super().__init__(config)
        
        # Метод агрегации сделок
        self.aggregation_method = config.get('aggregation_method', 'vwap')
        
        # Буфер сделок для каждого timestep
        self.trades_buffer = {}
        
        # История агрегированных сделок
        self.aggregated_trades = []
        
        # Фильтрация шумовых сделок
        self.min_trade_size = config.get('min_trade_size', 0.001)
        self.max_price_impact = config.get('max_price_impact', 0.01)  # 1%
    
    async def synchronize(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Синхронизация сделок
        
        Args:
            data: Входные данные сделки
            
        Returns:
            Dict: Синхронизированные данные или None
        """
        try:
            # Валидация данных
            if not self._validate_trade_data(data):
                return None
            
            timestamp = data['timestamp']
            
            # Валидация временной метки (очень мягкая)
            if not self._validate_timestamp(timestamp):
                # Не выводим предупреждение для старых меток - это нормально для кэшированных данных
                return None
            
            # Округление до базового timestep
            rounded_timestamp = self._round_timestamp(timestamp)
            
            # Детекция пропусков
            if self._last_timestamp and self._detect_gap(rounded_timestamp, self._last_timestamp):
                self._update_metrics('gap')
                await self._handle_gap(rounded_timestamp, self._last_timestamp)
            
            # Добавление сделки в буфер
            await self._add_trade_to_buffer(data, rounded_timestamp)
            
            # Обновление метрик
            self._update_metrics('message')
            self._last_timestamp = rounded_timestamp
            
            return None  # Сделки агрегируются в get_synchronized_stream
            
        except Exception as e:
            self.logger.error(f"Error in trades synchronization: {e}")
            return None
    
    async def get_synchronized_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированного потока сделок
        
        Yields:
            Dict: Агрегированные сделки за timestep
        """
        while self._running:
            try:
                current_time = self._round_timestamp(datetime.utcnow())
                
                # Проверка наличия сделок для завершенных timesteps
                completed_timesteps = []
                
                for timestamp, trades in self.trades_buffer.items():
                    if self._is_timestep_complete(timestamp, current_time):
                        # Агрегация сделок
                        aggregated = await self._aggregate_trades(trades, timestamp)
                        if aggregated:
                            yield aggregated
                            completed_timesteps.append(timestamp)
                
                # Удаление завершенных timesteps
                for timestamp in completed_timesteps:
                    del self.trades_buffer[timestamp]
                    self.aggregated_trades.append(timestamp)
                
                # Ограничение истории
                if len(self.aggregated_trades) > 1000:
                    self.aggregated_trades = self.aggregated_trades[-1000:]
                
                await asyncio.sleep(self.timestep_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in synchronized stream: {e}")
                await asyncio.sleep(1)
    
    def _validate_trade_data(self, data: Dict[str, Any]) -> bool:
        """
        Валидация данных сделки
        
        Args:
            data: Данные для валидации
            
        Returns:
            bool: True если данные валидны
        """
        required_fields = ['timestamp', 'price', 'volume', 'side']
        
        for field in required_fields:
            if field not in data:
                return False
        
        try:
            price = float(data['price'])
            volume = float(data['volume'])
            side = data['side']
            
            # Цена положительная
            if price <= 0:
                return False
            
            # Объем положительный
            if volume <= 0:
                return False
            
            # Сторона сделки
            if side not in ['buy', 'sell']:
                return False
            
            # Минимальный размер сделки
            if volume < self.min_trade_size:
                return False
            
        except (ValueError, TypeError):
            return False
        
        return True
    
    async def _add_trade_to_buffer(self, trade: Dict[str, Any], timestamp: datetime) -> None:
        """
        Добавление сделки в буфер
        
        Args:
            trade: Данные сделки
            timestamp: Временная метка timestep
        """
        if timestamp not in self.trades_buffer:
            self.trades_buffer[timestamp] = []
        
        self.trades_buffer[timestamp].append(trade)
        
        # Ограничение размера буфера для каждого timestep
        if len(self.trades_buffer[timestamp]) > 1000:
            self.trades_buffer[timestamp] = self.trades_buffer[timestamp][-1000:]
    
    def _is_timestep_complete(self, timestep: datetime, current_time: datetime) -> bool:
        """
        Проверка завершенности timestep
        
        Args:
            timestep: Временная метка timestep
            current_time: Текущее время
            
        Returns:
            bool: True если timestep завершен
        """
        next_timestep = timestep + timedelta(seconds=self.timestep_seconds)
        return current_time >= next_timestep
    
    async def _aggregate_trades(self, trades: List[Dict[str, Any]], timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Агрегация сделок за timestep
        
        Args:
            trades: Список сделок
            timestamp: Временная метка
            
        Returns:
            Dict: Агрегированные данные
        """
        if not trades:
            return None
        
        prices = [float(trade['price']) for trade in trades]
        volumes = [float(trade['volume']) for trade in trades]
        sides = [trade['side'] for trade in trades]
        
        # Базовая агрегация
        total_volume = sum(volumes)
        trade_count = len(trades)
        
        # VWAP расчет
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume if total_volume > 0 else 0
        
        # TWAP расчет
        twap = np.mean(prices) if prices else 0
        
        # Статистика цен
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        # Анализ сторон
        buy_volume = sum(v for v, s in zip(volumes, sides) if s == 'buy')
        sell_volume = sum(v for v, s in zip(volumes, sides) if s == 'sell')
        buy_count = sides.count('buy')
        sell_count = sides.count('sell')
        
        # Выбор метода агрегации
        if self.aggregation_method == 'vwap':
            aggregated_price = vwap
        elif self.aggregation_method == 'twap':
            aggregated_price = twap
        else:  # 'last' или 'standard'
            aggregated_price = prices[-1] if prices else 0
        
        # Детекция аномалий
        if len(prices) > 1:
            price_impact = abs(max_price - min_price) / min_price if min_price > 0 else 0
            if price_impact > self.max_price_impact:
                self._update_metrics('anomaly')
                self.logger.warning(f"High price impact detected: {price_impact:.4f}")
        
        return {
            'timestamp': timestamp,
            'price': aggregated_price,
            'vwap': vwap,
            'twap': twap,
            'min_price': min_price,
            'max_price': max_price,
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'trade_count': trade_count,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'price_range': price_range,
            'data_type': 'aggregated_trades'
        }
    
    async def _handle_gap(self, current_timestamp: datetime, previous_timestamp: datetime) -> None:
        """
        Обработка пропуска в сделках
        
        Args:
            current_timestamp: Текущая временная метка
            previous_timestamp: Предыдущая временная метка
        """
        gap_seconds = (current_timestamp - previous_timestamp).total_seconds()
        gap_steps = int(gap_seconds / self.timestep_seconds)
        
        self.logger.warning(f"Detected gap in trades: {gap_steps} timesteps ({gap_seconds}s)")
        
        # Заполнение пропуска пустыми данными
        for i in range(1, gap_steps):
            gap_timestamp = previous_timestamp + timedelta(seconds=i * self.timestep_seconds)
            
            # Пустой timestep без сделок
            empty_trades = {
                'timestamp': gap_timestamp,
                'price': 0.0,
                'vwap': 0.0,
                'twap': 0.0,
                'min_price': 0.0,
                'max_price': 0.0,
                'total_volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'trade_count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'price_range': 0.0,
                'data_type': 'aggregated_trades',
                'gap_filled': True
            }
            
            self.trades_buffer[gap_timestamp] = []
            self._update_metrics('gap_filled')
    
    def get_latest_aggregated_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Получение последних агрегированных сделок
        
        Args:
            count: Количество записей
            
        Returns:
            List[Dict]: Последние записи
        """
        # Возвращаем из буфера + история
        all_trades = []
        
        # Добавляем из буфера
        for timestamp in sorted(self.trades_buffer.keys()):
            if self.trades_buffer[timestamp]:
                aggregated = asyncio.create_task(
                    self._aggregate_trades(self.trades_buffer[timestamp], timestamp)
                )
                all_trades.append(aggregated)
        
        return all_trades[-count:] if all_trades else []
    
    def get_trades_at_time(self, timestamp: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        Получение сделок для указанного времени
        
        Args:
            timestamp: Временная метка
            
        Returns:
            List[Dict]: Сделки или None
        """
        rounded_timestamp = self._round_timestamp(timestamp)
        return self.trades_buffer.get(rounded_timestamp, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики синхронизации сделок
        
        Returns:
            Dict: Статистика
        """
        total_trades = sum(len(trades) for trades in self.trades_buffer.values())
        
        if not self.trades_buffer:
            return {'total_trades': 0}
        
        all_prices = []
        all_volumes = []
        
        for trades in self.trades_buffer.values():
            for trade in trades:
                all_prices.append(float(trade['price']))
                all_volumes.append(float(trade['volume']))
        
        return {
            'total_trades': total_trades,
            'avg_price': np.mean(all_prices) if all_prices else 0,
            'price_std': np.std(all_prices) if all_prices else 0,
            'total_volume': sum(all_volumes),
            'avg_volume': np.mean(all_volumes) if all_volumes else 0,
            'active_timesteps': len(self.trades_buffer)
        }
