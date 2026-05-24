# 2. Модели · 3. Ensemble · 4–7. Regime / веса / confidence / trading · 8. Features

---

## 2. Какие модели РЕАЛЬНО реализованы

### 2.1. Сводная таблица (canonical production)

Конфиг: `config/profiles/canonical_4model.yaml` → `orchestration.model_keys: [lgb, gru, xgb, cnn]`.

Фабрика: `orchestration/model_factory.py` → `build_orchestration_models()`.

| Название | Key | Обучается? | Ensemble | Prediction P(up) | Inference | Backtest/WFO |
|----------|-----|------------|----------|------------------|-----------|--------------|
| **LightGBM** | `lgb` | **Да** | **Да** | **Да** | **Да** | **Да** |
| **XGBoost** | `xgb` | **Да** | **Да** | **Да** | **Да** | **Да** |
| **GRU** | `gru` | **Да** (TF/Keras) | **Да** | **Да** | **Да** | **Да** |
| **CNN** | `cnn` | **Да** (TF/Keras) | **Да** | **Да** | **Да** | **Да** |

**Общий target для всех четырёх:** бинарный forward direction, `prediction_horizon: 12` баров (см. `orchestration/glue.default_horizon_labels`). Специализация «trend / mean-reversion / volatility» — **архитектурная и через веса ансамбля**, не через разные y в WFO.

### 2.2. Что есть в коде, но НЕ в canonical pipeline

| Название | Файл / место | Статус |
|----------|--------------|--------|
| **LSTM** | `ist.py`, README meta_learning | Legacy Colab; веса в `dynamic_meta.py` по умолчанию — ключ `lstm`, **не** `gru` |
| **Transformer** | `ist.py` `DLModelBuilder.build_transformer` | Только legacy; **не** в `model_factory` |
| **ML RegimeDetector** | `models/regime/regime_detector.py` | Обучается отдельно; **не** prod regime |
| **ModelRouter** | `models/router/model_router.py` | Legacy: одна модель на режим |
| **InferenceEngine** | `models/inference/inference_engine.py` | Deprecated; заменён `InferenceOrchestrator` |
| **EnsembleAggregator** | `meta_learning/ensemble.py` | Baseline average; ключи `lstm`/`trans` — **не** prod |

### 2.3. Детали по каждой canonical-модели

#### LightGBM (`lgb`)

| Поле | Значение |
|------|----------|
| Файл | `models/tabular/lightgbm_tabular_model.py` |
| Вход | таблица `(n_bars × n_features)` |
| Выход | `predict_proba[:, 1]` → P(up) |
| Вес trend / range | **0.10 / 0.55** |

#### XGBoost (`xgb`)

| Поле | Значение |
|------|----------|
| Файл | `models/mean_reversion/xgboost_model.py` |
| Legacy target | `prepare_labels()` mean-reversion — **не** вызывается orchestration |
| Вес trend / range | **0.10 / 0.25** |

#### GRU (`gru`)

| Поле | Значение |
|------|----------|
| Файл | `models/trend/gru_model.py` |
| Вход | окно **24 × n_features** (`feature_window_size`) |
| Архитектура | GRU(64) → GRU(32) → Dense sigmoid |
| Обучение | epochs=5 (profile), temporal val split 10%, EarlyStopping |
| Вес trend / range | **0.45 / 0.10** |

#### CNN (`cnn`)

| Поле | Значение |
|------|----------|
| Файл | `models/volatility/cnn_model.py` |
| Вход | окно 24 × n_features |
| Архитектура | Conv1D → MaxPool → Conv1D → Dense sigmoid |
| Вес trend / range | **0.35 / 0.10** |

### 2.4. Признаки для обучения

`infer_training_feature_columns(df)`:

- **Включает:** все числовые колонки
- **Исключает:** OHLCV, подстроки `meta_prob`, `signal`, `target`, `future_`, `order_book`, `obi`

Обычно **15+ колонок** после FeatureEngine + MTF (не только эталонные 8).

---

## 3. Как устроен ensemble (ядро ВКР)

### 3.1. Класс и место в pipeline

**Класс:** `meta_learning.dynamic_meta.DynamicMetaWeighting`  
**Создание:** `orchestration/model_factory.meta_weighting_from_config(OrchestratorConfig)`  
**Вызов:** `TrainingOrchestrator.run_pipeline()` → `apply_dynamic_weighting(predictions, regime_pred)`

Все четыре модели дают словарь `predictions = {lgb: array, gru: array, …}`.

### 3.2. Формула итогового prediction

Для каждого бара \(t\):

\[
\text{meta\_mgmt\_prob}[t] = \sum_{m \in \{lgb,gru,xgb,cnn\}} w_m(t) \cdot p_m(t)
\]

где \(p_m(t) \in [0,1]\) — P(up) модели \(m\).

**Псевдокод (как в коде):**

```python
weights_dict = meta.get_weights(regime_pred, mode="regime_adaptive")
ensemble_prob = np.zeros(n)
for model in ("lgb", "gru", "xgb", "cnn"):
    ensemble_prob += weights_dict[model] * predictions[model]
```

### 3.3. Static vs dynamic weighting

| Тип | Реализован? | Описание |
|-----|-------------|----------|
| **Равные веса** | Да (baseline) | `meta_learning/ensemble.py` — `simple_average`; не prod |
| **Static regime weights** | **Да (prod)** | Две фиксированные таблицы в YAML; режим `fixed_trend` / `fixed_range` |
| **Regime-adaptive** | **Да (prod)** | На каждом баре выбор таблицы по `regime_pred` |
| **Online adaptive по PnL/accuracy** | **Нет в prod** | `WeightUtils.adaptive_weight_update()` — утилита без вызова из orchestrator |

### 3.4. Веса в production (`canonical_4model.yaml`)

**Trend** (`regime_pred == 1`):

| lgb | gru | xgb | cnn |
|-----|-----|-----|-----|
| 0.10 | **0.45** | 0.10 | **0.35** |

**Range** (`regime_pred == 0`):

| lgb | gru | xgb | cnn |
|-----|-----|-----|-----|
| **0.55** | 0.10 | 0.25 | 0.10 |

Сумма весов = 1.0 в каждом режиме.

**Режимы ensemble** (`ensemble_mode`):

- `regime_adaptive` — default
- `fixed_trend` — всегда trend-таблица
- `fixed_range` — всегда range-таблица

### 3.5. direction_soft

После ансамбля (`training_orchestrator.run_pipeline`):

```python
direction_soft = np.where(
    meta_probabilities > direction_threshold, 1,
    np.where(meta_probabilities < (1 - direction_threshold), -1, 0)
)
# direction_threshold = 0.52 по умолчанию
```

Это **не** отдельная модель «DirectionModel» в modular path — порогование `meta_mgmt_prob`.

### 3.6. Сравнение с legacy `ist.py`

В Colab-скрипте веса заданы для **`lgb, lstm, cnn, trans`**. В modular prod — **`lgb, gru, xgb, cnn`**. Текст ВКР должен использовать **фактические ключи из YAML**, не LSTM/Transformer, если описывается текущая система.

---

## 4. Regime Detection

### 4.1. Реализовано ли?

| Подход | Статус | Где |
|--------|--------|-----|
| **Rule-based (prod)** | **Да** | `orchestration/glue.MomentumRegimeDetector` |
| **ML classifier** | Код есть, **не prod** | `models/regime/regime_detector.py` |
| **Третий режим breakout** | Ключи в YAML возможны | **Не** участвуют в `DynamicMetaWeighting` (только 0/1) |

### 4.2. Как определяется режим (production)

```python
# MomentumRegimeDetector (sma_period=50)
sma = close.rolling(50).mean()
regime_pred = (close > sma).astype(int)   # 1 = trend, 0 = range
```

- **Режимы:** бинарные — **trend** (1) / **range** (0), не отдельный класс «volatility».
- **Признак volatility** (`atr`, `volatility`) идёт в **модели**, но не в отдельный regime label в prod.

### 4.3. Где используется

1. **Веса ансамбля** — `DynamicMetaWeighting._regime_adaptive_weights`
2. **Интерпретация в GUI** — `inference.regime_series`, overlay на графике
3. **Не используется** для выбора одной модели (legacy router отключён)

### 4.4. Влияние на веса

**Прямое:** при `regime_pred[t]=1` применяется строка `trend_weights`, иначе `range_weights`.  
**Нет:** обучения весов по режиму из данных; веса заданы экспертно/тюнингом в конфиге.

---

## 5. Adaptive Weighting (критично для ВКР)

### 5.1. Честная формулировка

| Утверждение | Верно? |
|-------------|--------|
| Веса меняются **от бара к бару** | **Да**, если меняется `regime_pred` |
| Веса **обучаются** на истории PnL | **Нет** в prod pipeline |
| Веса **фиксированы** внутри режима | **Да** (константы из YAML) |
| Есть ablation static vs adaptive | **Да** — режимы `fixed_trend` / `regime_adaptive` в конфиге |

**Для текста ВКР:** корректный термин — **«режимно-адаптивное взвешивание» (regime-adaptive static weights)**, а не «online meta-learning весов по доходности».

### 5.2. Где update weights

| Механизм | Файл | В prod? |
|----------|------|---------|
| Переключение таблицы по regime | `dynamic_meta._regime_adaptive_weights` | **Да** |
| Ручное `update_weights(trend, range)` | `DynamicMetaWeighting.update_weights` | Только если вызвать вручную |
| `adaptive_weight_update(performance)` | `meta_learning/utils/weight_utils.py` | **Нет** |

### 5.3. Pipeline адаптации (regime-adaptive)

```text
close[t] → SMA(50) → regime_pred[t] ∈ {0,1}
       → pick trend_weights or range_weights
       → w_lgb(t), w_gru(t), … (константы внутри режима)
       → meta_mgmt_prob[t] = Σ w_m(t) * p_m(t)
```

### 5.4. Если в дипломе заявлен «полностью обучаемый meta-learner весов»

Потребуется **доработка кода** или **смягчение формулировки** до regime routing с фиксированными весами + опциональный тюнинг offline (`tune-thesis`, `orchestration/tuning_loop.py`).

---

## 6. Confidence Filtering

### 6.1. Есть ли confidence score?

| Механизм | Есть? | Как |
|----------|-------|-----|
| **Meta threshold filter** | **Да** | `DecisionPipeline` + `compute_integrated_signal` |
| **Scalar confidence** | **Да** (inference) | `InferenceOrchestrator`: `confidence = abs(meta_prob - 0.5) * 2` |
| **Отдельный MetaFilterModel** | Legacy / roadmap | не отдельный pickle в canonical bundle |

### 6.2. Как считается порог (второй фильтр после направления)

**Рекомендуемый путь:** `signal_source: integrated`

```python
# decision/signal_rules.compute_integrated_signal
direction_signal = sign(meta_mgmt vs 0.52)  # -1, 0, 1
meta_threshold = median(meta_mgmt_prob)  # на train; OOS — train_threshold_override
integrated_signal = direction_signal if (meta_mgmt > meta_threshold) and direction != 0 else 0
```

| Параметр | Default |
|----------|---------|
| `direction_threshold` | 0.52 |
| `meta_threshold_mode` | `median` (на train fold); causal rolling window 100 в safe mode |
| OOS без утечки | `_calibrate_test_meta_threshold(train_meta_probs)` — только train |

**Влияние на сделки:** сигнал **0 (flat)**, если meta_mgmt не выше порога → **меньше входов**, фильтрация «уверенности» ансамбля.

### 6.3. min_signal_margin

В `OrchestratorConfig`: `min_signal_margin: 0.0` — по умолчанию **выключен**.

---

## 7. Trading Logic

### 7.1. BUY / SELL / HOLD

В коде — **целые сигналы**, не строки:

| Значение | Смысл |
|----------|--------|
| **+1** | Long (аналог BUY) |
| **-1** | Short (аналог SELL) |
| **0** | Flat (HOLD) |

Источник: `DecisionPipeline` → `final_signal` / `integrated_signal`.

`trade_mode: both` в конфиге — разрешены long и short в бэктесте; paper execution **long-focused** (short без позиции может отклоняться).

### 7.2. Position management

| Элемент | Реализация |
|---------|------------|
| Размер позиции | `risk_management` — ATR-based `PositionSizer`, `OrchestratorRiskBridge` |
| Масштаб в бэктесте | `Backtester.run(..., position_size=)` |
| Trailing stop | `apply_atr_trailing_stop` — **опционально** (`apply_atr_trailing: false` в canonical) |
| Take-profit отдельным правилом | **Нет** в эталоне (README risk) |
| Stop-loss | через ATR trailing / обнуление сигнала volatility filter |

### 7.3. Risk controls

- `use_risk_bridge: true` — связка orchestrator ↔ risk
- Volatility filter: обнуление сигнала при экстремальном ATR на test (часто выключен порогом 0)
- **RL layer:** `rl_layer` — множитель риска DQN, **не** предсказание направления

### 7.4. Исполнение в бэктесте

```python
# backtesting/backtester.py
strategy_returns = signal.shift(1) * position_size.shift(1) * market_returns
costs = |Δ(signal * position_size)| * (commission + slippage)
# commission=0.0006, slippage=0.0002
```

Сигнал на баре \(t-1\) → доходность на баре \(t\) (как в vector backtest).

---

## 8. Feature Engineering

### 8.1. Базовые индикаторы (1h) — `feature_engineering/feature_engine.py`

| Признак | Параметры |
|---------|-----------|
| `ema_fast`, `ema_slow` | EMA 20, 50 |
| `ema_slope` | относительное изменение ema_fast |
| `rsi` | RSI 14 |
| `macd`, `macd_signal`, `macd_hist` | 12 / 26 / 9 |
| `atr` | ATR 14 |
| `log_ret` | log-returns |
| `volatility` | rolling std(log_ret, 20) × √24 |
| `adx` | ADX 14 |

### 8.2. Multi-timeframe (`synchronization/mtf_features.py`)

| Колонка | TF |
|---------|-----|
| `rsi_15m` | 15m |
| `ema_slope_15m` | 15m |
| `rsi_4h` | 4h |
| `adx_4h` | 4h |

### 8.3. Эталонный минимальный набор (8 колонок)

`feature_engineering/config.py` → `DIRECTION_FEATURE_COLUMNS`:

```text
rsi, macd_hist, ema_slope, adx,
rsi_15m, ema_slope_15m, rsi_4h, adx_4h
```

Orchestration **часто обучает на расширенном** наборе (все числовые engineered cols).

### 8.4. Что НЕ в canonical features

- Явные lag-returns (`return_5`, `return_15`) — roadmap
- Live order book / OBI — `microstructure.mode: off`
- Simulated OBI — только при `mode: simulated`

### 8.5. Scaling

- В `FeatureEngine` **нет** StandardScaler
- GRU/CNN получают сырые значения в окне 24
- Legacy `ist.py` использовал `StandardScaler` — не modular path
