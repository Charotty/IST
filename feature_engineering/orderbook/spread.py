"""
Spread Features

Признаки на основе спреда order book.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class SpreadFeatures(BaseFeatureGenerator):
    """Генератор признаков спреда"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора спреда
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.levels = config.get('levels', [5, 10])
        self.spread_window = config.get('spread_window', 10)
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = [
            "spread",
            "spread_pct",
            "spread_zscore",
            "spread_ma",
            "spread_volatility",
            "relative_spread",
            "log_spread",
            "bid_ask_tightness",
            "price_impact_score"
        ]
        
        # Признаки для разных уровней
        for level in self.levels:
            names.extend([
                f"spread_{level}",
                f"spread_pct_{level}",
                f"mid_price_{level}"
            ])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not self.levels or not isinstance(self.levels, list):
            raise ValueError("levels must be a non-empty list")
        
        if not all(isinstance(l, int) and l > 0 for l in self.levels):
            raise ValueError("All levels must be positive integers")
        
        if not isinstance(self.spread_window, int) or self.spread_window <= 0:
            raise ValueError("spread_window must be a positive integer")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков спреда
        
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
            
            # Расчет базовых спредов
            spreads = []
            mid_prices = []
            
            for idx, row in df.iterrows():
                try:
                    bids = row['bids']
                    asks = row['asks']
                    
                    if not bids or not asks:
                        spreads.append(0.0)
                        mid_prices.append(0.0)
                        continue
                    
                    best_bid = float(bids[0][0]) if len(bids) > 0 else 0.0
                    best_ask = float(asks[0][0]) if len(asks) > 0 else 0.0
                    
                    if best_bid > 0 and best_ask > 0:
                        spread = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / 2
                    else:
                        spread = 0.0
                        mid_price = 0.0
                    
                    spreads.append(spread)
                    mid_prices.append(mid_price)
                    
                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {e}")
                    spreads.append(0.0)
                    mid_prices.append(0.0)
            
            # Создание Series для удобства расчетов
            spread_series = pd.Series(spreads, index=df.index)
            mid_price_series = pd.Series(mid_prices, index=df.index)
            
            # Базовые признаки спреда
            features['spread'] = spread_series
            features['mid_price'] = mid_price_series
            
            # Процентный спред
            features['spread_pct'] = (spread_series / mid_price_series) * 100
            features['log_spread'] = np.log(spread_series + 1e-8)  # Избегаем log(0)
            
            # Относительный спред
            features['relative_spread'] = spread_series / mid_price_series
            
            # Скользящие средние спреда
            if len(spread_series) >= self.spread_window:
                features['spread_ma'] = spread_series.rolling(window=self.spread_window, min_periods=1).mean()
                features['spread_volatility'] = spread_series.rolling(window=self.spread_window, min_periods=1).std()
                
                # Z-score спреда
                spread_mean = features['spread_ma']
                spread_std = features['spread_volatility']
                features['spread_zscore'] = (spread_series - spread_mean) / (spread_std + 1e-8)
            else:
                features['spread_ma'] = spread_series
                features['spread_volatility'] = 0.0
                features['spread_zscore'] = 0.0
            
            # Bid-Ask tightness (обратный спред)
            features['bid_ask_tightness'] = 1.0 / (spread_series + 1e-8)
            
            # Price impact score (комбинированный показатель)
            features['price_impact_score'] = features['spread_pct'] * features['spread_volatility']
            
            # Признаки для разных уровней
            for level in self.levels:
                level_features = self._calculate_level_spreads(df, level)
                for name, values in level_features.items():
                    features[name] = values
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_level_spreads(self, df: pd.DataFrame, level: int) -> Dict[str, pd.Series]:
        """
        Расчет спредов для разных уровней глубины
        
        Args:
            df: DataFrame с order book данными
            level: Уровень глубины
            
        Returns:
            Dict: Признаки спредов для уровня
        """
        features = {}
        
        try:
            spreads = []
            mid_prices = []
            
            for idx, row in df.iterrows():
                try:
                    bids = row['bids']
                    asks = row['asks']
                    
                    if not bids or not asks or len(bids) < level or len(asks) < level:
                        spreads.append(0.0)
                        mid_prices.append(0.0)
                        continue
                    
                    level_bid = float(bids[level-1][0]) if len(bids[level-1]) >= 2 else 0.0
                    level_ask = float(asks[level-1][0]) if len(asks[level-1]) >= 2 else 0.0
                    
                    if level_bid > 0 and level_ask > 0:
                        spread = level_ask - level_bid
                        mid_price = (level_bid + level_ask) / 2
                    else:
                        spread = 0.0
                        mid_price = 0.0
                    
                    spreads.append(spread)
                    mid_prices.append(mid_price)
                    
                except Exception as e:
                    self.logger.error(f"Error processing level {level} row {idx}: {e}")
                    spreads.append(0.0)
                    mid_prices.append(0.0)
            
            features[f"spread_{level}"] = pd.Series(spreads, index=df.index)
            features[f"mid_price_{level}"] = pd.Series(mid_prices, index=df.index)
            
            # Процентный спред для уровня
            mid_price_series = features[f"mid_price_{level}"]
            features[f"spread_pct_{level}"] = (features[f"spread_{level}"] / (mid_price_series + 1e-8)) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating level {level} spreads: {e}")
            # Возвращаем нулевые значения
            features[f"spread_{level}"] = pd.Series([0.0] * len(df), index=df.index)
            features[f"mid_price_{level}"] = pd.Series([0.0] * len(df), index=df.index)
            features[f"spread_pct_{level}"] = pd.Series([0.0] * len(df), index=df.index)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['orderbook']
