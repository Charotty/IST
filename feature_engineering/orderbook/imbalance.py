"""
Order Book Imbalance Features

Признаки дисбаланса order book.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class OrderBookImbalance(BaseFeatureGenerator):
    """Генератор признаков дисбаланса order book"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора дисбаланса
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.levels = config.get('levels', [5, 10, 20])
        self.imbalance_window = config.get('imbalance_window', 10)
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = []
        
        for level in self.levels:
            names.extend([
                f"imbalance_{level}",
                f"bid_volume_{level}",
                f"ask_volume_{level}",
                f"total_volume_{level}",
                f"buy_pressure_{level}",
                f"sell_pressure_{level}"
            ])
        
        # Дополнительные признаки
        names.extend([
            "spread",
            "spread_pct",
            "mid_price",
            "best_bid",
            "best_ask"
        ])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not self.levels or not isinstance(self.levels, list):
            raise ValueError("levels must be a non-empty list")
        
        if not all(isinstance(l, int) and l > 0 for l in self.levels):
            raise ValueError("All levels must be positive integers")
        
        if not isinstance(self.imbalance_window, int) or self.imbalance_window <= 0:
            raise ValueError("imbalance_window must be a positive integer")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков дисбаланса
        
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
                row_features = self._calculate_row_features(row)
                
                for feature_name, value in row_features.items():
                    if feature_name not in features.columns:
                        features[feature_name] = np.nan
                    features.at[idx, feature_name] = value
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_row_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Расчет признаков для одной строки
        
        Args:
            row: Строка с order book данными
            
        Returns:
            Dict: Признаки
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
            
            # Расчет базовых признаков
            best_bid = float(bids[0][0]) if len(bids) > 0 else 0.0
            best_ask = float(asks[0][0]) if len(asks) > 0 else 0.0
            
            features['best_bid'] = best_bid
            features['best_ask'] = best_ask
            features['mid_price'] = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
            
            # Расчет спреда
            if best_bid > 0 and best_ask > 0:
                features['spread'] = best_ask - best_bid
                features['spread_pct'] = (features['spread'] / best_bid) * 100
            else:
                features['spread'] = 0.0
                features['spread_pct'] = 0.0
            
            # Расчет признаков для разных уровней глубины
            for level in self.levels:
                level_features = self._calculate_level_features(bids, asks, level)
                
                for name, value in level_features.items():
                    features[f"{name}_{level}"] = value
            
        except Exception as e:
            self.logger.error(f"Error calculating row features: {e}")
            # Возвращаем нулевые значения при ошибке
            for name in self._feature_names:
                features[name] = 0.0
        
        return features
    
    def _calculate_level_features(self, bids: np.ndarray, asks: np.ndarray, level: int) -> Dict[str, float]:
        """
        Расчет признаков для конкретного уровня глубины
        
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
            
            # Расчет объемов
            bid_volume = sum(float(bid[1]) for bid in bids_limited if len(bid) >= 2)
            ask_volume = sum(float(ask[1]) for ask in asks_limited if len(ask) >= 2)
            total_volume = bid_volume + ask_volume
            
            features['bid_volume'] = bid_volume
            features['ask_volume'] = ask_volume
            features['total_volume'] = total_volume
            
            # Расчет дисбаланса
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                buy_pressure = bid_volume / total_volume
                sell_pressure = ask_volume / total_volume
            else:
                imbalance = 0.0
                buy_pressure = 0.5
                sell_pressure = 0.5
            
            features['imbalance'] = imbalance
            features['buy_pressure'] = buy_pressure
            features['sell_pressure'] = sell_pressure
            
        except Exception as e:
            self.logger.error(f"Error calculating level {level} features: {e}")
            # Возвращаем нулевые значения
            features = {
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'total_volume': 0.0,
                'imbalance': 0.0,
                'buy_pressure': 0.5,
                'sell_pressure': 0.5
            }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['orderbook']
