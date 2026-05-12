"""
Depth Features

Признаки глубины order book.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class DepthFeatures(BaseFeatureGenerator):
    """Генератор признаков глубины"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора глубины
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.levels = config.get('levels', [5, 10, 20])
        self.depth_window = config.get('depth_window', 10)
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = []
        
        for level in self.levels:
            names.extend([
                f"depth_ratio_{level}",
                f"bid_depth_{level}",
                f"ask_depth_{level}",
                f"total_depth_{level}",
                f"depth_imbalance_{level}",
                f"volume_weighted_price_{level}",
                f"price_skewness_{level}",
                f"liquidity_concentration_{level}"
            ])
        
        # Общие признаки глубины
        names.extend([
            "depth_trend",
            "depth_volatility",
            "depth_momentum",
            "market_depth_score"
        ])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not self.levels or not isinstance(self.levels, list):
            raise ValueError("levels must be a non-empty list")
        
        if not all(isinstance(l, int) and l > 0 for l in self.levels):
            raise ValueError("All levels must be positive integers")
        
        if not isinstance(self.depth_window, int) or self.depth_window <= 0:
            raise ValueError("depth_window must be a positive integer")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков глубины
        
        Args:
            data: Order Book данные
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            # Конвертация в DataFrame если необходимо
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Проверка наличия нужных полей
            required_fields = ['bids', 'asks']
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                self.logger.warning(f"Missing fields: {missing_fields}")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет признаков для каждой строки
            for idx, row in df.iterrows():
                row_features = self._calculate_row_depth_features(row)
                
                for feature_name, value in row_features.items():
                    if feature_name not in features.columns:
                        features[feature_name] = np.nan
                    features.at[idx, feature_name] = value
            
            # Расчет временных признаков глубины
            if len(df) >= self.depth_window:
                self._calculate_temporal_depth_features(features)
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_row_depth_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Расчет признаков глубины для одной строки
        
        Args:
            row: Строка с order book данными
            
        Returns:
            Dict: Признаки глубины
        """
        features = {}
        
        try:
            bids = row['bids']
            asks = row['asks']
            
            if not bids or not asks:
                return features
            
            # Конвертация в numpy arrays если необходимо
            if isinstance(bids, list):
                bids = np.array(bids)
            if isinstance(asks, list):
                asks = np.array(asks)
            
            # Расчет признаков для разных уровней
            for level in self.levels:
                level_features = self._calculate_level_depth_features(bids, asks, level)
                
                for name, value in level_features.items():
                    features[f"{name}_{level}"] = value
            
        except Exception as e:
            self.logger.error(f"Error calculating row depth features: {e}")
            # Возвращаем нулевые значения при ошибке
            for name in self._feature_names:
                features[name] = 0.0
        
        return features
    
    def _calculate_level_depth_features(self, bids: np.ndarray, asks: np.ndarray, level: int) -> Dict[str, float]:
        """
        Расчет признаков глубины для конкретного уровня
        
        Args:
            bids: Массив bids
            asks: Массив asks
            level: Уровень глубины
            
        Returns:
            Dict: Признаки уровня
        """
        features = {}
        
        try:
            # Ограничение глубины
            bids_limited = bids[:level] if len(bids) >= level else bids
            asks_limited = asks[:level] if len(asks) >= level else asks
            
            # Расчет глубины
            bid_depth = len(bids_limited)
            ask_depth = len(asks_limited)
            total_depth = bid_depth + ask_depth
            
            features['bid_depth'] = float(bid_depth)
            features['ask_depth'] = float(ask_depth)
            features['total_depth'] = float(total_depth)
            
            # Отношение глубины
            if total_depth > 0:
                depth_ratio = bid_depth / total_depth
                depth_imbalance = (bid_depth - ask_depth) / total_depth
            else:
                depth_ratio = 0.5
                depth_imbalance = 0.0
            
            features['depth_ratio'] = depth_ratio
            features['depth_imbalance'] = depth_imbalance
            
            # Volume weighted price
            vwp_bid, vwp_ask = self._calculate_volume_weighted_price(bids_limited, asks_limited)
            features['volume_weighted_price'] = (vwp_bid + vwp_ask) / 2
            
            # Price skewness
            features['price_skewness'] = self._calculate_price_skewness(bids_limited, asks_limited)
            
            # Liquidity concentration
            features['liquidity_concentration'] = self._calculate_liquidity_concentration(bids_limited, asks_limited)
            
        except Exception as e:
            self.logger.error(f"Error calculating level {level} depth features: {e}")
            # Возвращаем нулевые значения
            features = {
                'bid_depth': 0.0,
                'ask_depth': 0.0,
                'total_depth': 0.0,
                'depth_ratio': 0.5,
                'depth_imbalance': 0.0,
                'volume_weighted_price': 0.0,
                'price_skewness': 0.0,
                'liquidity_concentration': 0.0
            }
        
        return features
    
    def _calculate_volume_weighted_price(self, bids: np.ndarray, asks: np.ndarray) -> tuple:
        """
        Расчет volume weighted price
        
        Args:
            bids: Массив bids
            asks: Массив asks
            
        Returns:
            tuple: (VWP bid, VWP ask)
        """
        try:
            # VWP для bids
            if len(bids) > 0 and len(bids[0]) >= 2:
                bid_prices = np.array([float(bid[0]) for bid in bids if len(bid) >= 2])
                bid_volumes = np.array([float(bid[1]) for bid in bids if len(bid) >= 2])
                
                if bid_volumes.sum() > 0:
                    vwp_bid = np.sum(bid_prices * bid_volumes) / bid_volumes.sum()
                else:
                    vwp_bid = bid_prices[0] if len(bid_prices) > 0 else 0.0
            else:
                vwp_bid = 0.0
            
            # VWP для asks
            if len(asks) > 0 and len(asks[0]) >= 2:
                ask_prices = np.array([float(ask[0]) for ask in asks if len(ask) >= 2])
                ask_volumes = np.array([float(ask[1]) for ask in asks if len(ask) >= 2])
                
                if ask_volumes.sum() > 0:
                    vwp_ask = np.sum(ask_prices * ask_volumes) / ask_volumes.sum()
                else:
                    vwp_ask = ask_prices[0] if len(ask_prices) > 0 else 0.0
            else:
                vwp_ask = 0.0
            
            return vwp_bid, vwp_ask
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_price_skewness(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """
        Расчет асимметрии цен
        
        Args:
            bids: Массив bids
            asks: Массив asks
            
        Returns:
            float: Асимметрия цен
        """
        try:
            if len(bids) == 0 or len(asks) == 0:
                return 0.0
            
            bid_prices = np.array([float(bid[0]) for bid in bids if len(bid) >= 2])
            ask_prices = np.array([float(ask[0]) for ask in asks if len(ask) >= 2])
            
            if len(bid_prices) == 0 or len(ask_prices) == 0:
                return 0.0
            
            # Расчет skewness на основе разницы между лучшими ценами
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            
            # Простая метрика skewness
            if mid_price > 0:
                skewness = (best_ask - best_bid) / mid_price
            else:
                skewness = 0.0
            
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_liquidity_concentration(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """
        Расчет концентрации ликвидности
        
        Args:
            bids: Массив bids
            asks: Массив asks
            
        Returns:
            float: Концентрация ликвидности
        """
        try:
            if len(bids) == 0 or len(asks) == 0:
                return 0.0
            
            bid_volumes = np.array([float(bid[1]) for bid in bids if len(bid) >= 2])
            ask_volumes = np.array([float(ask[1]) for ask in asks if len(ask) >= 2])
            
            if len(bid_volumes) == 0 or len(ask_volumes) == 0:
                return 0.0
            
            # Расчет концентрации (доля объема на лучших уровнях)
            total_bid_volume = bid_volumes.sum()
            total_ask_volume = ask_volumes.sum()
            total_volume = total_bid_volume + total_ask_volume
            
            if total_volume > 0:
                # Объем на первых 3 уровнях
                top_bid_volume = bid_volumes[:3].sum()
                top_ask_volume = ask_volumes[:3].sum()
                top_volume = top_bid_volume + top_ask_volume
                
                concentration = top_volume / total_volume
            else:
                concentration = 0.0
            
            return concentration
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_depth_features(self, features: pd.DataFrame) -> None:
        """
        Расчет временных признаков глубины
        
        Args:
            features: DataFrame с признаками
        """
        try:
            # Выбираем столбцы с общей глубиной
            depth_columns = [col for col in features.columns if 'total_depth' in col]
            
            if not depth_columns:
                return
            
            # Общая глубина (сумма по уровням)
            total_depth_series = features[depth_columns].sum(axis=1)
            
            # Скользящие средние
            features['depth_trend'] = total_depth_series.rolling(window=self.depth_window, min_periods=1).mean()
            
            # Волатильность глубины
            features['depth_volatility'] = total_depth_series.rolling(window=self.depth_window, min_periods=1).std()
            
            # Моментум глубины
            features['depth_momentum'] = total_depth_series.pct_change().fillna(0)
            
            # Market depth score (комбинированный показатель)
            depth_score = (features['depth_trend'] / (features['depth_volatility'] + 1e-8)) * (1 - abs(features['depth_momentum']))
            features['market_depth_score'] = depth_score.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal depth features: {e}")
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['orderbook']
