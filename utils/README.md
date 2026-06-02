# Utils

## Назначение

Общие утилиты и вспомогательные функции, используемые всеми компонентами системы.

**Production-critical:** `data_leakage_prevention.py` — purge/embargo и `safe_walk_forward_split` для WFO (`TrainingOrchestrator`). `compute_safe_threshold` — causal meta-порог в `decision` (без look-ahead).

## Основные категории утилит

### 1. Работа с данными
- Обработка временных рядов
- Валидация данных
- Преобразование форматов
- Агрегация данных

### 2. Математические функции
- Статистические расчеты
- Финансовые формулы
- Технические индикаторы
- Оптимизационные алгоритмы

### 3. Системные утилиты
- Логирование
- Работа с файлами
- Сетевые операции
- Управление процессами

### 4. Финансовые утилиты
- Расчет метрик
- Управление рисками
- Портфельная теория
- Временная стоимость денег

## Структура модуля

```
utils/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── time_series.py        # Работа с временными рядами
│   ├── validators.py        # Валидация данных
│   ├── transformers.py      # Преобразование данных
│   └── aggregators.py      # Агрегация данных
├── math/
│   ├── __init__.py
│   ├── statistics.py       # Статистические функции
│   ├── financial.py        # Финансовые расчеты
│   ├── indicators.py       # Технические индикаторы
│   └── optimization.py    # Оптимизационные алгоритмы
├── system/
│   ├── __init__.py
│   ├── logger.py           # Логирование
│   ├── file_utils.py       # Работа с файлами
│   ├── network.py          # Сетевые операции
│   └── process.py         # Управление процессами
├── finance/
│   ├── __init__.py
│   ├── metrics.py          # Финансовые метрики
│   ├── risk.py            # Управление рисками
│   ├── portfolio.py       # Портфельная теория
│   └── time_value.py      # Временная стоимость денег
├── decorators/
│   ├── __init__.py
│   ├── retry.py           # Retry декоратор
│   ├── cache.py           # Cache декоратор
│   ├── timing.py          # Timing декоратор
│   └── validation.py      # Validation декоратор
├── exceptions/
│   ├── __init__.py
│   ├── base.py            # Базовые исключения
│   ├── data.py            # Исключения данных
│   ├── trading.py         # Торговые исключения
│   └── system.py          # Системные исключения
└── helpers/
    ├── __init__.py
    ├── datetime.py        # Работа с датами
    ├── currency.py        # Работа с валютами
    ├── formatting.py      # Форматирование
    └── encryption.py      # Шифрование
```

## Ключевые компоненты

### TimeSeriesUtils

Утилиты для работы с временными рядами:

```python
import pandas as pd
import numpy as np
from typing import Optional, Union

class TimeSeriesUtils:
    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Ресемплинг OHLCV данных"""
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(freq).agg(ohlc_dict).dropna()
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Расчет доходности"""
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return method: {method}")
    
    @staticmethod
    def rolling_window(data: np.ndarray, window: int) -> np.ndarray:
        """Создание скользящих окон"""
        shape = (data.shape[0] - window + 1, window)
        strides = (data.strides[0], data.strides[0])
        
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
```

### FinancialMath

Финансовые расчеты:

```python
class FinancialMath:
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Расчет Sharpe Ratio"""
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Расчет максимальной просадки"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """Расчет Profit Factor"""
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
```

### TechnicalIndicators

Технические индикаторы:

```python
class TechnicalIndicators:
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
```

### LoggerConfig

Конфигурация логирования:

```python
import logging
import logging.config
from typing import Dict, Any

class LoggerConfig:
    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> None:
        """Настройка логирования"""
        logging.config.dictConfig(config)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Получение логгера"""
        return logging.getLogger(name)
    
    @staticmethod
    def default_config() -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'level': 'INFO',
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard'
                },
                'file': {
                    'level': 'DEBUG',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'trading_system.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': 'DEBUG',
                    'propagate': False
                }
            }
        }
```

### RetryDecorator

Декоратор для повторных попыток:

```python
import time
import functools
from typing import Callable, Type, Tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Декоратор для повторных попыток"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator
```

### CacheDecorator

Декоратор для кэширования:

```python
import functools
import hashlib
import pickle
from typing import Any, Dict

class SimpleCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Any:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
    
    def clear(self) -> None:
        self._cache.clear()

_cache = SimpleCache()

def cache(ttl: int = 3600):
    """Декоратор для кэширования"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Создание ключа кэша
            key_data = pickle.dumps((func.__name__, args, kwargs))
            cache_key = hashlib.md5(key_data).hexdigest()
            
            # Проверка кэша
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Выполнение функции
            result = func(*args, **kwargs)
            
            # Сохранение в кэш
            _cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator
```

## Исключения

### Базовые исключения

```python
class TradingSystemError(Exception):
    """Базовое исключение торговой системы"""
    pass

class DataError(TradingSystemError):
    """Исключение связанное с данными"""
    pass

class ValidationError(TradingSystemError):
    """Исключение валидации"""
    pass

class TradingError(TradingSystemError):
    """Торговое исключение"""
    pass

class RiskError(TradingSystemError):
    """Исключение управления рисками"""
    pass
```

## Валидаторы

### Валидация данных

```python
from typing import Any, List, Dict
import pandas as pd

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Валидация OHLCV данных"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Проверка наличия колонок
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Проверка логических соотношений
        if not (df['low'] <= df['high']).all():
            return False
        
        if not (df['low'] <= df['open']).all():
            return False
        
        if not (df['low'] <= df['close']).all():
            return False
        
        if not (df['open'] <= df['high']).all():
            return False
        
        if not (df['close'] <= df['high']).all():
            return False
        
        # Проверка на отрицательные значения
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            return False
        
        return True
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Валидация цены"""
        return isinstance(price, (int, float)) and price > 0
    
    @staticmethod
    def validate_quantity(quantity: float) -> bool:
        """Валидация количества"""
        return isinstance(quantity, (int, float)) and quantity > 0
```

## Технологии

- **pandas** - обработка данных
- **numpy** - численные расчеты
- **scipy** - научные вычисления
- **logging** - логирование
- **functools** - декораторы
- **hashlib** - хэширование

## Тестирование

- Unit тесты для каждой утилиты
- Integration тесты для комплексных функций
- Performance тесты для вычислений
- Validation тесты для корректности

## Требования

1. **Надежность** - корректная работа во всех условиях
2. **Производительность** - эффективные алгоритмы
3. **Тестируемость** - легкое тестирование
4. **Документация** - понятные описания
5. **Совместимость** - работа с разными типами данных
