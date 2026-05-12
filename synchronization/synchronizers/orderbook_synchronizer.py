"""
Order Book Synchronizer

Синхронизация Order Book снэпшотов в единые временные ряды.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_synchronizer import BaseSynchronizer


class OrderBookSynchronizer(BaseSynchronizer):
    """Синхронизатор Order Book"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация синхронизатора Order Book
        
        Args:
            config: Конфигурация синхронизатора
        """
        super().__init__(config)
        
        # Глубина Order Book
        self.depth = config.get('depth', 10)
        
        # Буфер снэпшотов Order Book
        self.orderbook_snapshots = {}
        
        # История синхронизированных Order Book
        self.synchronized_orderbooks = []
        
        # Методы обработки Order Book
        self.calculate_spreads = config.get('calculate_spreads', True)
        self.calculate_imbalance = config.get('calculate_imbalance', True)
    
    async def synchronize(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Синхронизация Order Book
        
        Args:
            data: Входные данные Order Book
            
        Returns:
            Dict: Синхронизированные данные или None
        """
        try:
            # Валидация данных
            if not self._validate_orderbook_data(data):
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
            
            # Сохранение снэпшота Order Book
            await self._save_orderbook_snapshot(data, rounded_timestamp)
            
            # Обновление метрик
            self._update_metrics('message')
            self._last_timestamp = rounded_timestamp
            
            return None  # Order Book обрабатывается в get_synchronized_stream
            
        except Exception as e:
            self.logger.error(f"Error in Order Book synchronization: {e}")
            return None
    
    async def get_synchronized_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированного потока Order Book
        
        Yields:
            Dict: Синхронизированные Order Book за timestep
        """
        while self._running:
            try:
                current_time = self._round_timestamp(datetime.utcnow())
                
                # Проверка наличия Order Book для завершенных timesteps
                completed_timesteps = []
                
                for timestamp in sorted(self.orderbook_snapshots.keys()):
                    if self._is_timestep_complete(timestamp, current_time):
                        # Получение лучшего снэпшота для timestep
                        best_snapshot = await self._get_best_snapshot(timestamp)
                        if best_snapshot:
                            yield best_snapshot
                            completed_timesteps.append(timestamp)
                
                # Удаление завершенных timesteps
                for timestamp in completed_timesteps:
                    del self.orderbook_snapshots[timestamp]
                    self.synchronized_orderbooks.append(timestamp)
                
                # Ограничение истории
                if len(self.synchronized_orderbooks) > 1000:
                    self.synchronized_orderbooks = self.synchronized_orderbooks[-1000:]
                
                await asyncio.sleep(self.timestep_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in synchronized stream: {e}")
                await asyncio.sleep(1)
    
    def _validate_orderbook_data(self, data: Dict[str, Any]) -> bool:
        """
        Валидация данных Order Book
        
        Args:
            data: Данные для валидации
            
        Returns:
            bool: True если данные валидны
        """
        required_fields = ['timestamp', 'bids', 'asks']
        
        for field in required_fields:
            if field not in data:
                return False
        
        try:
            bids = data['bids']
            asks = data['asks']
            
            # Bids и asks должны быть списками
            if not isinstance(bids, list) or not isinstance(asks, list):
                return False
            
            # Проверка формата bid/ask: [[price, volume], ...]
            for bid in bids[:self.depth]:
                if not isinstance(bid, list) or len(bid) != 2:
                    return False
                price, volume = bid
                if float(price) <= 0 or float(volume) < 0:
                    return False
            
            for ask in asks[:self.depth]:
                if not isinstance(ask, list) or len(ask) != 2:
                    return False
                price, volume = ask
                if float(price) <= 0 or float(volume) < 0:
                    return False
            
            # Проверка порядка цен (bids убывают, asks возрастают)
            if len(bids) > 1:
                for i in range(len(bids) - 1):
                    if float(bids[i][0]) < float(bids[i + 1][0]):
                        return False  # Bids должны убывать
            
            if len(asks) > 1:
                for i in range(len(asks) - 1):
                    if float(asks[i][0]) > float(asks[i + 1][0]):
                        return False  # Asks должны возрастать
            
            # Проверка спреда (best bid < best ask)
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                if best_bid >= best_ask:
                    return False
            
        except (ValueError, TypeError, IndexError):
            return False
        
        return True
    
    async def _save_orderbook_snapshot(self, orderbook: Dict[str, Any], timestamp: datetime) -> None:
        """
        Сохранение снэпшота Order Book
        
        Args:
            orderbook: Данные Order Book
            timestamp: Временная метка
        """
        if timestamp not in self.orderbook_snapshots:
            self.orderbook_snapshots[timestamp] = []
        
        # Ограничение глубины
        bids = orderbook['bids'][:self.depth]
        asks = orderbook['asks'][:self.depth]
        
        snapshot = {
            'timestamp': orderbook['timestamp'],
            'bids': bids,
            'asks': asks,
            'symbol': orderbook.get('symbol', 'unknown'),
            'source': orderbook.get('source', 'unknown')
        }
        
        self.orderbook_snapshots[timestamp].append(snapshot)
        
        # Ограничение количества снэпшотов в timestep
        if len(self.orderbook_snapshots[timestamp]) > 10:
            self.orderbook_snapshots[timestamp] = self.orderbook_snapshots[timestamp][-10:]
    
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
    
    async def _get_best_snapshot(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Получение лучшего снэпшота для timestep
        
        Args:
            timestamp: Временная метка
            
        Returns:
            Dict: Лучший снэпшот с метриками
        """
        snapshots = self.orderbook_snapshots.get(timestamp, [])
        
        if not snapshots:
            return None
        
        # Выбираем последний снэпшот (самый свежий)
        best_snapshot = snapshots[-1]
        
        # Расчет метрик
        metrics = await self._calculate_orderbook_metrics(best_snapshot)
        
        # Объединение данных
        result = {
            'timestamp': timestamp,
            'bids': best_snapshot['bids'],
            'asks': best_snapshot['asks'],
            'symbol': best_snapshot['symbol'],
            'data_type': 'synchronized_orderbook',
            'source': best_snapshot['source']
        }
        
        result.update(metrics)
        
        return result
    
    async def _calculate_orderbook_metrics(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Расчет метрик Order Book
        
        Args:
            orderbook: Данные Order Book
            
        Returns:
            Dict: Метрики
        """
        bids = orderbook['bids']
        asks = orderbook['asks']
        
        metrics = {}
        
        if self.calculate_spreads and bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            metrics['best_bid'] = best_bid
            metrics['best_ask'] = best_ask
            metrics['spread'] = best_ask - best_bid
            metrics['spread_pct'] = (metrics['spread'] / best_bid) * 100 if best_bid > 0 else 0
            metrics['mid_price'] = (best_bid + best_ask) / 2
        
        if self.calculate_imbalance and bids and asks:
            # Расчет imbalance для первых 5 уровней
            bid_volume = sum(float(bid[1]) for bid in bids[:5])
            ask_volume = sum(float(ask[1]) for ask in asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                metrics['bid_volume_5'] = bid_volume
                metrics['ask_volume_5'] = ask_volume
                metrics['total_volume_5'] = total_volume
                metrics['imbalance_5'] = (bid_volume - ask_volume) / total_volume
                metrics['buy_pressure'] = bid_volume / total_volume
                metrics['sell_pressure'] = ask_volume / total_volume
        
        # Глубина рынка
        metrics['bid_depth'] = len(bids)
        metrics['ask_depth'] = len(asks)
        metrics['total_depth'] = len(bids) + len(asks)
        
        # Общие объемы
        if bids:
            metrics['total_bid_volume'] = sum(float(bid[1]) for bid in bids)
        if asks:
            metrics['total_ask_volume'] = sum(float(ask[1]) for ask in asks)
        
        return metrics
    
    async def _handle_gap(self, current_timestamp: datetime, previous_timestamp: datetime) -> None:
        """
        Обработка пропуска в Order Book
        
        Args:
            current_timestamp: Текущая временная метка
            previous_timestamp: Предыдущая временная метка
        """
        gap_seconds = (current_timestamp - previous_timestamp).total_seconds()
        gap_steps = int(gap_seconds / self.timestep_seconds)
        
        self.logger.warning(f"Detected gap in Order Book: {gap_steps} timesteps ({gap_seconds}s)")
        
        # Заполнение пропуска последним известным Order Book
        if previous_timestamp in self.orderbook_snapshots and self.orderbook_snapshots[previous_timestamp]:
            last_snapshot = self.orderbook_snapshots[previous_timestamp][-1]
            
            for i in range(1, gap_steps):
                gap_timestamp = previous_timestamp + timedelta(seconds=i * self.timestep_seconds)
                
                # Forward fill с меткой gap_filled
                gap_snapshot = {
                    'timestamp': gap_timestamp,
                    'bids': last_snapshot['bids'],
                    'asks': last_snapshot['asks'],
                    'symbol': last_snapshot['symbol'],
                    'source': 'gap_fill',
                    'gap_filled': True
                }
                
                self.orderbook_snapshots[gap_timestamp] = [gap_snapshot]
                self._update_metrics('gap_filled')
    
    def get_latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """
        Получение последнего Order Book
        
        Returns:
            Dict: Последний Order Book или None
        """
        if not self.orderbook_snapshots:
            return None
        
        latest_timestamp = max(self.orderbook_snapshots.keys())
        snapshots = self.orderbook_snapshots[latest_timestamp]
        
        if not snapshots:
            return None
        
        return snapshots[-1]
    
    def get_orderbook_at_time(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Получение Order Book для указанного времени
        
        Args:
            timestamp: Временная метка
            
        Returns:
            Dict: Order Book или None
        """
        rounded_timestamp = self._round_timestamp(timestamp)
        snapshots = self.orderbook_snapshots.get(rounded_timestamp, [])
        
        return snapshots[-1] if snapshots else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики синхронизации Order Book
        
        Returns:
            Dict: Статистика
        """
        if not self.orderbook_snapshots:
            return {'total_snapshots': 0}
        
        total_snapshots = sum(len(snapshots) for snapshots in self.orderbook_snapshots.values())
        
        # Расчет среднего спреда
        spreads = []
        for snapshots in self.orderbook_snapshots.values():
            for snapshot in snapshots:
                bids = snapshot.get('bids', [])
                asks = snapshot.get('asks', [])
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spreads.append(best_ask - best_bid)
        
        return {
            'total_snapshots': total_snapshots,
            'active_timesteps': len(self.orderbook_snapshots),
            'avg_spread': np.mean(spreads) if spreads else 0,
            'spread_std': np.std(spreads) if spreads else 0,
            'latest_timestamp': max(self.orderbook_snapshots.keys()) if self.orderbook_snapshots else None
        }
