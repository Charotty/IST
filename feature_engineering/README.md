# Feature Engineering

## Назначение

Расчёт признаков для ML/DL: базовые индикаторы на основном TF, MTF-колонки после sync, микроструктура (OBI, spread). Каталог ниже включает **реализованные в `ist.py`** и **целевые** из roadmap.

## Статус

| Блок | Статус |
|------|--------|
| `FeatureEngine` (pandas_ta) | **Реализовано** |
| MTF-колонки (через sync) | **Реализовано** |
| OBI / spread | **Симуляция в research** → **реальный L2** (см. ниже) |
| Bollinger, сессии, skew/kurtosis, selection | **Roadmap** (каталог сохранён) |

## Реализованные признаки (`FeatureEngine`)

### Тренд

| Признак | Формула / параметры |
|---------|---------------------|
| `ema_fast` | EMA(close, 20) |
| `ema_slow` | EMA(close, 50) |
| `ema_slope` | Δema_fast / ema_fast.shift(1) |

### Импульс

| Признак | Параметры |
|---------|-----------|
| `rsi` | RSI(14) |
| `macd`, `macd_signal`, `macd_hist` | MACD(12, 26, 9) |

### Волатильность

| Признак | Параметры |
|---------|-----------|
| `atr` | ATR(14) |
| `log_ret` | ln(close / close.shift(1)) |
| `volatility` | std(log_ret, 20) × √24 (часовые бары) |

### Режим

| Признак | Параметры |
|---------|-----------|
| `adx` | ADX(14) |

### MTF (после `MultiTimeframeEngine`)

`rsi_15m`, `ema_slope_15m`, `rsi_4h`, `adx_4h`

### Микроструктура

| Признак | Research (`ist.py`) | Production |
|---------|---------------------|------------|
| `order_book_imbalance` | `simulate_l2_features()` | OKX L2 WebSocket / REST books |
| `bid_ask_spread` | proxy от `volatility` | `(ask - bid) / bid` |

## Переход с симуляции на реальный OBI / spread

### Проблема

В Colab OBI генерировался как `tanh(ema_slope * 100 + noise)` — только для RL-экспериментов.

### Целевой путь (без смены контракта колонок)

1. **Data Layer (live)** — подписка OKX `books` / `books5`, depth N (например 20).  
2. **`OrderBookMicrostructure`** (уже в `ist.py`):

```python
def calculate_obi(self, bids, asks):
    bid_vol = sum(b[1] for b in bids[:self.depth])
    ask_vol = sum(a[1] for a in asks[:self.depth])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def get_spread(self, bids, asks):
    return (asks[0][0] - bids[0][0]) / bids[0][0]
```

3. **Synchronization** — resample OBI/spread на `1h` (`.last()` или mean за час).  
4. **Feature flag** — `microstructure_mode: simulated | live`; при `live` симулятор не вызывается.  
5. **Бэктест** — хранить исторические L2-снэпшоты (Parquet) или отключать OBI в чистом OHLCV-бэктесте.

## Каталог признаков (roadmap / расширение)

Сохраняется для поэтапного добавления; не все входят в текущий training set.

### Технические

- **MA:** SMA, EMA (доп. периоды 5, 10, 200)  
- **Momentum:** Stochastic, доп. RSI(21)  
- **Volatility:** Bollinger(20, 2σ)  
- **Volume:** OBV, VWAP, ADL  

### Order book (при live L2)

```python
Imbalance = (ΣBidVol - ΣAskVol) / (ΣBidVol + ΣAskVol)
MidPrice = (best_ask + best_bid) / 2
Spread = best_ask - best_bid
```

Depth: price impact, liquidity ratio — по мере накопления L2.

### Временные

- `hour`, `dayofweek`, сессии (Asia / EU / US)  
- returns: `r_t`, log-return на горизонтах 1, 5, 15, 60 баров  

### Статистические

- rolling mean/std, z-score  
- skew, kurtosis, percentiles на окнах 10/20/50  

### Нормализация (DL)

- `StandardScaler` на окне последовательностей (`WINDOW_SIZE=24`) — в pipeline моделей, не в `FeatureEngine`.

## Входной набор для Direction / DL (эталон)

```python
features = [
    'rsi', 'macd_hist', 'ema_slope', 'adx',
    'rsi_15m', 'ema_slope_15m', 'rsi_4h', 'adx_4h'
]
```

## Структура модуля (целевая)

```
feature_engineering/
├── __init__.py
├── feature_engine.py           # FeatureEngine
├── microstructure/
│   ├── order_book.py           # OrderBookMicrostructure
│   └── l2_adapter.py           # OKX WS → OBI, spread
├── technical/                  # roadmap: bollinger, volume, ...
└── feature_manager.py
```

## Конфигурация

```yaml
feature_engineering:
  base_indicators:
    ema_fast: 20
    ema_slow: 50
    rsi_length: 14
    atr_length: 14
    macd: [12, 26, 9]
    vol_window: 20
  microstructure:
    mode: "live"               # simulated | live | off
    depth: 20
  mtf_columns: ["rsi_15m", "ema_slope_15m", "rsi_4h", "adx_4h"]
```

## Интеграция

| Откуда | Куда |
|--------|------|
| **Data Layer** + **Synchronization** | OHLCV + MTF |
| **Models / Meta / RL** | матрица признаков + производные колонки моделей |

## Тестирование

- Unit: формулы RSI/ATR vs эталон  
- Нет NaN после `dropna()` в конце pipeline  
- Live OBI: сравнение с ручным расчётом на снэпшоте стакана  
