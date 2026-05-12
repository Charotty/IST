# Data Layer

## Назначение

Получение и потоковая обработка рыночных данных из различных источников.

## Основные задачи

- Сбор исторических и потоковых данных
- Нормализация и валидация данных
- Обработка разрывов и пропусков
- Кэширование и оптимизация доступа

## Источники данных

### Основные
- **OKX REST API** - исторические данные (OHLCV, trades)
- **OKX WebSocket** - потоковые данные (real-time)

### Дополнительные
- **Glassnode** - ончейн-метрики

## Типы данных

### OHLCV данные
```
OHLCV_t = (Open_t, High_t, Low_t, Close_t, Volume_t)
```

### Order Book
```
LOB_t = {(bid_i, volume_i), (ask_i, volume_i)}
```

### Trades
```
Trade_t = (price, volume, side, timestamp)
```

## Структура модуля

```
data_layer/
├── __init__.py
├── connectors/
│   ├── __init__.py
│   ├── base_connector.py      # Базовый класс коннектора
│   ├── okx_connector.py       # OKX API коннектор
│   └── glassnode_connector.py # Glassnode коннектор
├── processors/
│   ├── __init__.py
│   ├── ohlcv_processor.py     # Обработка OHLCV данных
│   ├── orderbook_processor.py # Обработка order book
│   └── trades_processor.py    # Обработка сделок
├── storage/
│   ├── __init__.py
│   ├── parquet_storage.py     # Хранение в Parquet
│   └── timescale_storage.py   # Хранение в TimescaleDB
├── streamers/
│   ├── __init__.py
│   ├── websocket_streamer.py  # WebSocket потоковый клиент
│   └── rest_streamer.py       # REST опрос данных
└── data_manager.py            # Главный менеджер данных
```

## Ключевые компоненты

### BaseConnector
Абстрактный базовый класс для всех коннекторов:
- Стандартизация интерфейсов
- Обработка ошибок и реконнект
- Лимиты запросов и rate limiting

### DataManager
Центральный компонент управления данными:
- Координация всех коннекторов
- Обработка конфликтов данных
- Управление кэшированием
- Валидация данных

### StreamProcessor
Обработка потоковых данных в реальном времени:
- Фильтрация шумов
- Обнаружение аномалий
- Агрегация данных

## Технологии

- **asyncio** - асинхронная обработка
- **websockets** - WebSocket клиенты
- **ccxt** - унифицированный API к биржам
- **pandas** - обработка данных
- **pyarrow** - работа с Parquet
- **asyncpg** - асинхронный PostgreSQL

## Конфигурация

```yaml
data_layer:
  connectors:
    okx:
      api_key: "${OKX_API_KEY}"
      secret_key: "${OKX_SECRET_KEY}"
      sandbox: true
  
  storage:
    parquet_path: "./data/parquet"
    timescale_url: "postgresql://user:pass@localhost/trading"
  
  streaming:
    buffer_size: 10000
    batch_size: 100
    flush_interval: 5  # seconds
```

## Требования к реализации

1. **Асинхронность** - все операции должны быть неблокирующими
2. **Отказоустойчивость** - автоматические реконнекты и обработка ошибок
3. **Масштабируемость** - поддержка множественных источников
4. **Производительность** - минимизация задержек для потоковых данных
5. **Надежность** - валидация и проверка целостности данных

## Интеграция

Data Layer передает данные в:
- **Synchronization Layer** - для синхронизации потоков
- **Feature Engineering** - для расчета признаков
- **Storage** - для долгосрочного хранения

## Тестирование

- Unit тесты для каждого коннектора
- Integration тесты для потоковой обработки
- Load тесты для производительности
- Mock тесты для API endpoints
