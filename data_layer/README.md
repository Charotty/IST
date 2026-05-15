# Data Layer

## Назначение

Загрузка исторических OHLCV с биржи OKX через REST (ccxt) с пагинацией. Опционально — несколько таймфреймов за один запуск для downstream MTF-пайплайна.

## Статус

| Компонент | Статус |
|-----------|--------|
| REST ccxt, пагинация OHLCV | **Реализовано** (`ist.py` → `OKXDataLoader`) |
| Мульти-таймфрейм при включении | **Реализовано** (флаг/список TF) |
| WebSocket, Glassnode, trades, Parquet, Timescale | **Не в scope** (позже при необходимости) |

## Источники данных

- **OKX REST API** — только OHLCV (история)
- Ключи API не обязательны для публичных `fetch_ohlcv`

## Типы данных

### OHLCV

```
OHLCV_t = (timestamp, open, high, low, close, volume)
```

Индекс — `timestamp` (UTC), дедупликация по времени, обрезка по `end_date`.

## Эталонная реализация (Colab / `ist.py`)

### OKXDataLoader

```python
class OKXDataLoader:
    def __init__(self):
        self.exchange = ccxt.okx({'enableRateLimit': True})

    def fetch_all_ohlcv(self, symbol, timeframe, start_str, end_str):
        # Пагинация: since = last_ts + 1, rate limit sleep
        # Колонки: timestamp, open, high, low, close, volume
```

### Параметры по умолчанию

| Параметр | Значение |
|----------|----------|
| `symbol` | `BTC/USDT` |
| `timeframe` (базовый) | `1h` |
| `start_date` | `2020-01-01` |
| `end_date` | `2026-01-01` |

### Мульти-таймфрейм

При `multi_timeframe: true` (или списке TF) загружаются дополнительные интервалы, например:

- `15m` — микро-контекст  
- `4h` — макро-контекст  

Базовый ряд остаётся основным (обычно `1h`). Слияние — в **Synchronization** (`MultiTimeframeEngine`).

## Структура модуля (целевая)

```
data_layer/
├── __init__.py
├── loaders/
│   └── okx_ohlcv_loader.py    # OKXDataLoader
├── validators/
│   └── ohlcv_validator.py     # пропуски, дубликаты, обрезка по end
└── config.py                  # symbol, timeframes, dates
```

## Конфигурация

```yaml
data_layer:
  exchange: okx
  symbol: "BTC/USDT"
  timeframe: "1h"              # базовый TF
  start_date: "2020-01-01 00:00:00"
  end_date: "2026-01-01 00:00:00"
  multi_timeframe:
    enabled: true
    timeframes: ["15m", "4h"]  # дополнительные к базовому
  rate_limit: true
```

## Требования к реализации

1. **Пагинация** — устойчивый цикл до `end_ts`, защита от зацикливания на одном баре  
2. **Rate limit** — `enableRateLimit` + sleep по `exchange.rateLimit`  
3. **Идемпотентность** — `drop_duplicates(subset=['timestamp'])`  
4. **Опциональный MTF** — один loader, несколько вызовов `fetch_all_ohlcv` по списку TF  

## Интеграция

Data Layer передаёт сырые `DataFrame` в:

- **Synchronization** — `MultiTimeframeEngine` (merge на базовый TF)  
- **Feature Engineering** — если MTF отключён, напрямую в `FeatureEngine`  

## Тестирование

- Unit: пагинация, дедуп, обрезка по `end_date`  
- Integration: сравнение длины ряда с ожидаемым числом баров для TF  
