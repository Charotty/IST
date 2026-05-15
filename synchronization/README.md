# Synchronization Layer

## Назначение

Приведение нескольких OHLCV-таймфреймов к **единому базовому индексу** (по умолчанию `1h`): расчёт MTF-признаков, `resample` + `ffill`, merge с основным рядом.

Реализация **не** отдельный микросервис с субсекундными тиками — это слой выравнивания MTF внутри research/production pipeline (эталон: `MultiTimeframeEngine` в `ist.py`).

## Статус

| Компонент | Статус |
|-----------|--------|
| `MultiTimeframeEngine` (resample + ffill → 1h) | **Реализовано** |
| Gap fill (forward fill) | **Реализовано** |
| Субсекундная / LOB / trades sync | **Опционально позже** (не ломает текущий дизайн) |

## Базовый timestep

**Целевой индекс:** таймфрейм основного ряда (например `1h`).

Дополнительные TF (`15m`, `4h`) агрегируются к нему:

```python
df_15_resampled = df_15[cols].resample('1h').last().ffill()
df_4h_resampled = df_4h[cols].resample('1h').last().ffill()
df = base_df.join(df_15_resampled).join(df_4h_resampled)
return df.dropna()
```

## Эталонная реализация

### MultiTimeframeEngine

```python
class MultiTimeframeEngine:
    def __init__(self, loader, symbol, start_date, end_date):
        ...

    def get_multi_data(self):
        self.df_15m = loader.fetch_all_ohlcv(symbol, '15m', ...)
        self.df_4h = loader.fetch_all_ohlcv(symbol, '4h', ...)
        return self

    def compute_and_merge(self, base_df):
        # 15m: rsi_15m, ema_slope_15m
        # 4h:  rsi_4h, adx_4h
        # resample('1h').last().ffill() → join → dropna()
```

### MTF-колонки (после merge)

| Колонка | Источник |
|---------|----------|
| `rsi_15m` | RSI(14) на 15m |
| `ema_slope_15m` | pct_change EMA(20) на 15m |
| `rsi_4h` | RSI(14) на 4h |
| `adx_4h` | ADX(14) на 4h |

## Методы обработки пропусков

### Forward Fill (основной)

```python
x_t = x_{t-1}  # после resample().last().ffill()
```

Используется для MTF-признаков на сетке базового TF.

### Linear / Adaptive Fill (roadmap)

Допустимы для **других** источников (LOB, funding) без изменения `MultiTimeframeEngine`:

- короткие пропуски — `ffill`  
- длинные — linear interpolation или пометка `is_gap`  

Не применять к OHLCV merge без отдельного ADR.

## Структура модуля (целевая)

```
synchronization/
├── __init__.py
├── multi_timeframe_engine.py   # эталон из ist.py
└── gap_handler.py              # опционально: adaptive fill для live
```

## Конфигурация

```yaml
synchronization:
  base_timeframe: "1h"
  auxiliary_timeframes: ["15m", "4h"]
  resample_rule: "1h"          # = base_timeframe
  fill_method: "ffill"
  drop_na_after_merge: true
```

## Расширения (не ломают MTF)

При появлении L2/trades:

- отдельный `orderbook_synchronizer` → снэпшоты на `base_timeframe`  
- `gap_handler`: `max_gap_size`, `method: adaptive`  
- метрики: gap ratio, consistency score  

## Интеграция

| Откуда | Куда |
|--------|------|
| **Data Layer** | сырые OHLCV по TF |
| **Feature Engineering** | единый `df` с MTF-колонками |
| **Models** | те же строки, что и базовый 1h |

## Тестирование

- Длина индекса после merge = длина `base_df` (минус `dropna`)  
- Нет look-ahead: только `resample().last()` на прошлых барах вспомогательного TF  
- Стабильность `ffill` при редких барах 4h  
