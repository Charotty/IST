# Feature Engineering

## Назначение

Формирование информативного признакового пространства для моделей машинного обучения.

## Основные задачи

- Расчет технических индикаторов
- Создание признаков из order book
- Генерация временных признаков
- Нормализация и масштабирование признаков
- Отбор наиболее информативных признаков

## Категории признаков

### 1. Технические индикаторы

#### Moving Averages

```python
# Simple Moving Average
SMA_n = (1/n) * Σ(P_{t-i}) for i=0 to n-1

# Exponential Moving Average  
EMA_t = α * P_t + (1-α) * EMA_{t-1}
```

#### Momentum Indicators

```python
# RSI
RSI = 100 - (100 / (1 + RS))

# MACD
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
```

#### Volatility Indicators

```python
# Bollinger Bands
Upper = SMA_n + k * σ_n
Lower = SMA_n - k * σ_n

# ATR
ATR = Average(True_Range)
```

### 2. Order Book признаки

#### Imbalance Metrics

```python
# Order Book Imbalance
Imbalance = (ΣBidVolume - ΣAskVolume) / (ΣBidVolume + ΣAskVolume)

# Spread
Spread = Ask_best - Bid_best

# Mid Price
MidPrice = (Ask_best + Bid_best) / 2
```

#### Depth Features

```python
# Price Impact
PriceImpact = (MidPrice_t - MidPrice_{t-1}) / Volume

# Liquidity Ratio
Liquidity = ΣVolume_at_best_levels / Total_Volume
```

### 3. Временные признаки

#### Returns

```python
# Simple Return
r_t = (P_t - P_{t-1}) / P_{t-1}

# Log Return
r_t = ln(P_t / P_{t-1})
```

#### Time-based Features

```python
# Time of day
HourOfDay = timestamp.hour
DayOfWeek = timestamp.dayofweek

# Session indicators
IsAsianSession = timezone in ['Tokyo', 'Singapore']
IsEuropeanSession = timezone in ['London', 'Frankfurt']
IsAmericanSession = timezone in ['New York', 'Chicago']
```

### 4. Статистические признаки

#### Rolling Statistics

```python
# Rolling Mean/Volatility
RollingMean_n = mean(P_{t-n:t})
RollingStd_n = std(P_{t-n:t})

# Z-score
ZScore = (P_t - RollingMean_n) / RollingStd_n
```

#### Distribution Features

```python
# Skewness and Kurtosis
Skewness = skew(returns_window)
Kurtosis = kurtosis(returns_window)

# Percentiles
P25 = percentile(returns, 25)
P75 = percentile(returns, 75)
```

## Структура модуля

```
feature_engineering/
├── __init__.py
├── technical/
│   ├── __init__.py
│   ├── moving_averages.py    # SMA, EMA, WMA
│   ├── momentum.py           # RSI, MACD, Stochastic
│   ├── volatility.py         # Bollinger Bands, ATR
│   └── volume.py             # OBV, VWAP, ADL
├── orderbook/
│   ├── __init__.py
│   ├── imbalance.py          # Order book imbalance
│   ├── spread.py             # Spread metrics
│   ├── depth.py              # Depth features
│   └── microstructure.py     # Microstructure features
├── temporal/
│   ├── __init__.py
│   ├── returns.py            # Return calculations
│   ├── time_features.py      # Time-based features
│   └── seasonality.py        # Seasonal patterns
├── statistical/
│   ├── __init__.py
│   ├── rolling_stats.py      # Rolling statistics
│   ├── distribution.py       # Distribution features
│   └── correlation.py        # Correlation features
├── selection/
│   ├── __init__.py
│   ├── importance.py         # Feature importance
│   ├── correlation_filter.py # Correlation filtering
│   └── variance_filter.py    # Variance filtering
├── scaling/
│   ├── __init__.py
│   ├── normalizer.py         # Normalization methods
│   └── scaler.py             # Scaling methods
└── feature_manager.py        # Главный менеджер признаков
```

## Ключевые компоненты

### FeatureManager

Центральный компонент управления признаками:
- Координация всех генераторов признаков
- Управление зависимостями между признаками
- Кэширование вычислений
- Валидация признаков

### BaseFeatureGenerator

Абстрактный базовый класс:
- Стандартизация интерфейсов
- Общие методы валидации
- Обработка ошибок

### FeatureSelector

Отбор наиболее информативных признаков:
- Статистические тесты
- Методы вложений (embedded methods)
- Жадные алгоритмы отбора

### FeatureScaler

Нормализация и масштабирование:
- StandardScaler
- MinMaxScaler  
- RobustScaler
- Custom scaling methods

## Оптимизация вычислений

### Векторизация

Использование numpy/pandas векторизованных операций для максимальной производительности.

### Кэширование

Кэширование вычисленных признаков для избежания повторных расчетов.

### Параллелизация

Параллельный расчет независимых признаков.

### Инкрементальные вычисления

Инкрементальное обновление признаков при поступлении новых данных.

## Технологии

- **pandas** - временные ряды и вычисления
- **numpy** - векторизованные операции
- **scipy** - статистические функции
- **scikit-learn** - масштабирование и отбор признаков
- **numba** - JIT компиляция для ускорения
- **dask** - параллельные вычисления

## Конфигурация

```yaml
feature_engineering:
  features:
    technical:
      moving_averages:
        periods: [5, 10, 20, 50, 200]
        types: ["SMA", "EMA"]
      momentum:
        rsi_periods: [14, 21]
        macd_params: [12, 26, 9]
      volatility:
        bollinger_periods: [20]
        bollinger_std: [2.0]
    
    orderbook:
      levels: [5, 10, 20]
      imbalance_window: 10
    
    temporal:
      return_periods: [1, 5, 15, 60]
      seasonal_features: true
    
    statistical:
      rolling_windows: [10, 20, 50]
      percentiles: [25, 75, 90]

  selection:
    method: "mutual_info"  # variance, correlation, mutual_info
    max_features: 100
    correlation_threshold: 0.95

  scaling:
    method: "standard"  # standard, minmax, robust
    feature_range: [0, 1]
```

## Метрики качества

### Информативность признаков
- Mutual Information
- Feature Importance
- Correlation with target

### Стабильность признаков
- Feature stability over time
- Out-of-sample performance
- Computational efficiency

## Интеграция

Feature Engineering получает данные от:
- **Synchronization Layer** - синхронизированные данные

И передает признаки в:
- **Models Layer** - для обучения и предсказания
- **Meta-Learning** - для адаптивного выбора признаков

## Требования к реализации

1. **Производительность** - быстрые вычисления в real-time
2. **Масштабируемость** - поддержка тысяч признаков
3. **Надежность** - обработка ошибок и аномальных данных
4. **Гибкость** - легкое добавление новых признаков
5. **Оптимизация** - минимальное использование памяти

## Тестирование

- Unit тесты для каждого признака
- Integration тесты для pipeline
- Performance тесты для скорости вычислений
- Validation тесты для корректности расчетов
