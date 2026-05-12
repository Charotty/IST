"""
Order Book Processor

Обработка данных Order Book (стакана заявок).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging


class OrderBookProcessor:
    """Процессор Order Book данных"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация процессора
        
        Args:
            config: Конфигурация процессора
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры обработки
        self.max_depth = self.config.get('max_depth', 20)
        self.min_spread = self.config.get('min_spread', 0.0001)
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.3)
        
    def process_orderbook_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Обработка сырых данных Order Book
        
        Args:
            raw_data: Список сырых данных
            
        Returns:
            pd.DataFrame: Обработанные данные Order Book
        """
        try:
            if not raw_data:
                return pd.DataFrame()
            
            # Создание DataFrame
            df = pd.DataFrame(raw_data)
            
            # Валидация данных
            df = self._validate_data(df)
            
            # Расчет производных метрик
            df = self._calculate_metrics(df)
            
            # Фильтрация по глубине
            df = self._filter_by_depth(df)
            
            # Сортировка
            df = df.sort_values(['timestamp', 'level']).reset_index(drop=True)
            
            self.logger.debug(f"Processed {len(df)} orderbook records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook data: {e}")
            raise
    
    def calculate_imbalance(self, bids: List[List], asks: List[List]) -> float:
        """
        Расчет дисбаланса Order Book
        
        Args:
            bids: Список бидов [[price, volume], ...]
            asks: Список асков [[price, volume], ...]
            
        Returns:
            float: Индекс дисбаланса (-1 до 1)
        """
        try:
            # Суммарные объемы
            bid_volume = sum(bid[1] for bid in bids[:self.max_depth])
            ask_volume = sum(ask[1] for ask in asks[:self.max_depth])
            
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Индекс дисбаланса
            imbalance = (bid_volume - ask_volume) / total_volume
            
            return imbalance
            
        except Exception as e:
            self.logger.error(f"Error calculating imbalance: {e}")
            return 0.0
    
    def calculate_spread(self, best_bid: float, best_ask: float) -> float:
        """
        Расчет спреда
        
        Args:
            best_bid: Лучшая цена покупки
            best_ask: Лучшая цена продажи
            
        Returns:
            float: Спред
        """
        try:
            if best_bid <= 0 or best_ask <= 0:
                return 0.0
            
            spread = best_ask - best_bid
            relative_spread = spread / best_bid
            
            return {
                'absolute': spread,
                'relative': relative_spread,
                'percentage': relative_spread * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spread: {e}")
            return {'absolute': 0.0, 'relative': 0.0, 'percentage': 0.0}
    
    def calculate_mid_price(self, best_bid: float, best_ask: float) -> float:
        """
        Расчет средней цены
        
        Args:
            best_bid: Лучшая цена покупки
            best_ask: Лучшая цена продажи
            
        Returns:
            float: Средняя цена
        """
        try:
            if best_bid <= 0 or best_ask <= 0:
                return 0.0
            
            return (best_bid + best_ask) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating mid price: {e}")
            return 0.0
    
    def calculate_weighted_mid_price(
        self, 
        bids: List[List], 
        asks: List[List],
        depth: int = 5
    ) -> float:
        """
        Расчет взвешенной средней цены
        
        Args:
            bids: Список бидов
            asks: Список асков
            depth: Глубина для расчета
            
        Returns:
            float: Взвешенная средняя цена
        """
        try:
            # Взвешенные объемами цены
            bid_prices = np.array([bid[0] for bid in bids[:depth]])
            bid_volumes = np.array([bid[1] for bid in bids[:depth]])
            
            ask_prices = np.array([ask[0] for ask in asks[:depth]])
            ask_volumes = np.array([ask[1] for ask in asks[:depth]])
            
            # Расчет взвешенных цен
            weighted_bid = np.average(bid_prices, weights=bid_volumes)
            weighted_ask = np.average(ask_prices, weights=ask_volumes)
            
            return (weighted_bid + weighted_ask) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted mid price: {e}")
            return 0.0
    
    def detect_liquidity_holes(self, bids: List[List], asks: List[List]) -> Dict[str, Any]:
        """
        Детекция ликвидных дыр
        
        Args:
            bids: Список бидов
            asks: Список асков
            
        Returns:
            Dict: Информация о ликвидных дырах
        """
        try:
            holes = {
                'bid_holes': [],
                'ask_holes': [],
                'severity': 'low'
            }
            
            # Детекция дыр в бидах
            for i in range(1, len(bids)):
                price_gap = bids[i][0] - bids[i-1][0]
                if price_gap > self.min_spread * 5:  # Большой разрыв
                    holes['bid_holes'].append({
                        'level': i,
                        'price_gap': price_gap,
                        'gap_percentage': price_gap / bids[i-1][0] * 100
                    })
            
            # Детекция дыр в асках
            for i in range(1, len(asks)):
                price_gap = asks[i-1][0] - asks[i][0]
                if price_gap > self.min_spread * 5:  # Большой разрыв
                    holes['ask_holes'].append({
                        'level': i,
                        'price_gap': price_gap,
                        'gap_percentage': price_gap / asks[i-1][0] * 100
                    })
            
            # Оценка серьезности
            total_holes = len(holes['bid_holes']) + len(holes['ask_holes'])
            if total_holes > 5:
                holes['severity'] = 'high'
            elif total_holes > 2:
                holes['severity'] = 'medium'
            
            return holes
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity holes: {e}")
            return {'bid_holes': [], 'ask_holes': [], 'severity': 'low'}
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация данных Order Book"""
        # Проверка обязательных полей
        required_fields = ['timestamp', 'bids', 'asks', 'symbol']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Фильтрация невалидных записей
        valid_rows = (
            df['bids'].apply(lambda x: isinstance(x, list) and len(x) > 0) &
            df['asks'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        )
        
        return df[valid_rows]
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет производных метрик"""
        df_copy = df.copy()
        
        # Расчет метрик для каждой записи
        metrics = []
        
        for idx, row in df_copy.iterrows():
            bids = row['bids']
            asks = row['asks']
            
            if not bids or not asks:
                metrics.append({})
                continue
            
            # Базовые метрики
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            
            # Расчет метрик
            spread_info = self.calculate_spread(best_bid, best_ask)
            mid_price = self.calculate_mid_price(best_bid, best_ask)
            weighted_mid = self.calculate_weighted_mid_price(bids, asks)
            imbalance = self.calculate_imbalance(bids, asks)
            liquidity_holes = self.detect_liquidity_holes(bids, asks)
            
            # Общая глубина
            total_depth = min(len(bids), len(asks))
            
            metrics.append({
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread_info['absolute'],
                'relative_spread': spread_info['relative'],
                'spread_percentage': spread_info['percentage'],
                'mid_price': mid_price,
                'weighted_mid_price': weighted_mid,
                'imbalance': imbalance,
                'total_depth': total_depth,
                'bid_volume': sum(bid[1] for bid in bids[:total_depth]),
                'ask_volume': sum(ask[1] for ask in asks[:total_depth]),
                'liquidity_holes': liquidity_holes
            })
        
        # Добавление метрик в DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        for col in metrics_df.columns:
            df_copy[col] = metrics_df[col]
        
        return df_copy
    
    def _filter_by_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация по глубине"""
        if 'total_depth' not in df.columns:
            return df
        
        # Оставляем только записи с достаточной глубиной
        return df[df['total_depth'] >= 2]
    
    def get_orderbook_snapshot(self, df: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """
        Получение снэпшота Order Book
        
        Args:
            df: Данные Order Book
            timestamp: Временная метка
            
        Returns:
            Dict: Снэпшот Order Book
        """
        try:
            # Фильтрация по времени
            if 'timestamp' in df.columns:
                df_filtered = df[df['timestamp'] <= timestamp]
            else:
                df_filtered = df
            
            if df_filtered.empty:
                return {'error': 'No data available'}
            
            # Получение последней записи
            latest = df_filtered.iloc[-1]
            
            return {
                'timestamp': timestamp,
                'symbol': latest.get('symbol', ''),
                'best_bid': latest.get('best_bid', 0),
                'best_ask': latest.get('best_ask', 0),
                'spread': latest.get('spread', 0),
                'mid_price': latest.get('mid_price', 0),
                'weighted_mid_price': latest.get('weighted_mid_price', 0),
                'imbalance': latest.get('imbalance', 0),
                'total_depth': latest.get('total_depth', 0),
                'bids': latest.get('bids', []),
                'asks': latest.get('asks', []),
                'bid_volume': latest.get('bid_volume', 0),
                'ask_volume': latest.get('ask_volume', 0),
                'liquidity_holes': latest.get('liquidity_holes', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting orderbook snapshot: {e}")
            return {'error': str(e)}
    
    def analyze_orderbook_trends(
        self, 
        df: pd.DataFrame, 
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Анализ трендов Order Book
        
        Args:
            df: Данные Order Book
            window_minutes: Окно для анализа в минутах
            
        Returns:
            Dict: Анализ трендов
        """
        try:
            if df.empty or 'timestamp' not in df.columns:
                return {'error': 'No valid data'}
            
            # Фильтрация по временному окну
            cutoff_time = df['timestamp'].max() - pd.Timedelta(minutes=window_minutes)
            df_window = df[df['timestamp'] >= cutoff_time]
            
            if df_window.empty:
                return {'error': 'No data in time window'}
            
            # Анализ трендов
            trends = {
                'spread_trend': self._analyze_spread_trend(df_window),
                'depth_trend': self._analyze_depth_trend(df_window),
                'imbalance_trend': self._analyze_imbalance_trend(df_window),
                'liquidity_trend': self._analyze_liquidity_trend(df_window)
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook trends: {e}")
            return {'error': str(e)}
    
    def _analyze_spread_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда спреда"""
        if 'spread' not in df.columns:
            return {}
        
        spread_values = df['spread'].values
        if len(spread_values) < 2:
            return {}
        
        # Расчет тренда
        first_half = np.mean(spread_values[:len(spread_values)//2])
        second_half = np.mean(spread_values[len(spread_values)//2:])
        
        trend = 'stable'
        change_pct = 0
        
        if second_half > first_half * 1.1:
            trend = 'widening'
            change_pct = ((second_half - first_half) / first_half) * 100
        elif second_half < first_half * 0.9:
            trend = 'narrowing'
            change_pct = ((first_half - second_half) / first_half) * 100
        
        return {
            'trend': trend,
            'change_percentage': change_pct,
            'current_spread': spread_values[-1] if len(spread_values) > 0 else 0,
            'avg_spread': np.mean(spread_values)
        }
    
    def _analyze_depth_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда глубины"""
        if 'total_depth' not in df.columns:
            return {}
        
        depth_values = df['total_depth'].values
        if len(depth_values) < 2:
            return {}
        
        first_half = np.mean(depth_values[:len(depth_values)//2])
        second_half = np.mean(depth_values[len(depth_values)//2:])
        
        trend = 'stable'
        change_pct = 0
        
        if second_half > first_half * 1.1:
            trend = 'increasing'
            change_pct = ((second_half - first_half) / first_half) * 100
        elif second_half < first_half * 0.9:
            trend = 'decreasing'
            change_pct = ((first_half - second_half) / first_half) * 100
        
        return {
            'trend': trend,
            'change_percentage': change_pct,
            'current_depth': depth_values[-1] if len(depth_values) > 0 else 0,
            'avg_depth': np.mean(depth_values)
        }
    
    def _analyze_imbalance_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда дисбаланса"""
        if 'imbalance' not in df.columns:
            return {}
        
        imbalance_values = df['imbalance'].values
        if len(imbalance_values) < 2:
            return {}
        
        # Анализ направления дисбаланса
        positive_imbalance = np.sum(imbalance_values > 0)
        negative_imbalance = np.sum(imbalance_values < 0)
        
        total_samples = len(imbalance_values)
        
        return {
            'dominant_side': 'buy' if positive_imbalance > negative_imbalance else 'sell',
            'buy_pressure_pct': (positive_imbalance / total_samples) * 100,
            'sell_pressure_pct': (negative_imbalance / total_samples) * 100,
            'current_imbalance': imbalance_values[-1] if len(imbalance_values) > 0 else 0,
            'avg_imbalance': np.mean(np.abs(imbalance_values))
        }
    
    def _analyze_liquidity_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда ликвидности"""
        if 'bid_volume' not in df.columns or 'ask_volume' not in df.columns:
            return {}
        
        bid_volumes = df['bid_volume'].values
        ask_volumes = df['ask_volume'].values
        
        if len(bid_volumes) == 0 or len(ask_volumes) == 0:
            return {}
        
        total_volume_trend = []
        for i in range(len(bid_volumes)):
            total_volume_trend.append(bid_volumes[i] + ask_volumes[i])
        
        if len(total_volume_trend) < 2:
            return {}
        
        first_half = np.mean(total_volume_trend[:len(total_volume_trend)//2])
        second_half = np.mean(total_volume_trend[len(total_volume_trend)//2:])
        
        trend = 'stable'
        change_pct = 0
        
        if second_half > first_half * 1.1:
            trend = 'increasing'
            change_pct = ((second_half - first_half) / first_half) * 100
        elif second_half < first_half * 0.9:
            trend = 'decreasing'
            change_pct = ((first_half - second_half) / first_half) * 100
        
        return {
            'trend': trend,
            'change_percentage': change_pct,
            'current_total_volume': total_volume_trend[-1] if len(total_volume_trend) > 0 else 0,
            'avg_total_volume': np.mean(total_volume_trend)
        }
