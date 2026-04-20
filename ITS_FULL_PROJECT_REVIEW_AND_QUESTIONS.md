# Intelligent Trading System (ITS) — Полный анализ документации, статуса реализации и вопросы к архитектуре

Дата: 2026-04-19

Этот документ составлен на основе файлов:

- `COMPREHENSIVE_REVIEW_GUIDE.md`
- `PROJECT_STATUS.md`
- `PLAN.MD`
- README по слоям в `its_project/*/README.md`
- фактического кода (в первую очередь `its_project/app/main.py`, `pipeline/queues.py` и реализаций слоев)

Цель документа:

1) Подробно объяснить, **что уже реализовано** в проекте (архитектура, подход, поток данных, контракты).
2) Подробно объяснить, **что планируется** (как задумано по плану/гайду; что отсутствует/недоделано по коду).
3) Дать **полный набор сильных вопросов** по каждому модулю/слою, которые ставят под сомнение решения и помогают составить план поиска знаний (книги/статьи/практики трейдинга, рынков, ML, инженерии).

---

## 1) Что реализовано уже сейчас

### 1.1. Архитектура и ключевые принципы

#### Слоистая архитектура
В проекте зафиксирована последовательность слоев (по `PROJECT_STATUS.md`):

```
Data Layer -> Storage -> Features -> Models -> Meta-Learning -> Decision -> Execution -> Interface
```

Фактическая реализация (по `app/main.py`) уже строится как асинхронный pipeline, связанный очередями.

#### Главные правила проекта (заявленные)

- Разделение ответственности: Data Layer делает только I/O и упаковывание в `MarketData`.
- Коммуникация через контракты и очереди: этапы обмениваются через `asyncio.Queue`.
- Асинхронность: сетевые операции через `aiohttp`/`websockets`.
- Конфигурация извне: `common/config.py` читает `config.json`.
- Логирование: в большинстве мест используется `logging`, но в отдельных местах остались `print` (см. ниже в вопросах/рисках).

#### Единый формат события MarketData
Контракт Data Layer:

```python
MarketData(
    timestamp_ms: int,
    symbol: str,
    type: MarketDataType,
    exchange: str,
    data: dict[str, Any],
)
```

Дополнительные внутренние маркеры в `data`:

- `_kind`: `trade | snapshot | delta`
- `_ts_recv_ms`: timestamp получения (для контроля лагов)

Это полезно для:

- отладки задержек/потерь,
- диагностики качества канала,
- последующей реконструкции порядка событий.

---

### 1.2. Реализованные слои (по модульной структуре)

Ниже — слой, его назначение, и что конкретно реализовано.

#### Stage 1 — Data Layer (`its_project/data_layer`)

**Назначение:** сбор сырых данных и унификация формата, без вычислений признаков/аналитики.

**Реализовано:**

- `BaseDataSource` — базовый интерфейс источника.
- `RateLimiter` — ограничение запросов.
- `binance_ws.py` — Binance WebSocket клиент:
  - подписка на `trade` и `depth` стримы,
  - преобразование входящих сообщений в `MarketData` через `make_trade` / `make_orderbook_delta`,
  - опциональная публикация snapshot через REST перед началом стрима (`fetch_snapshot_with_retries`).
- `binance_rest.py` — REST клиент для snapshot и klines (spot/futures), асинхронный, с `RateLimiter`.
- `glassnode.py` — клиент Glassnode (ончейн метрики), опрос по расписанию.
- `sentiment_x.py` — клиент X/Twitter, recent search, bearer token.
- `polling.py` — универсальная polling задача.

**Инварианты:**

- Data Layer не должен «чинить» стакан, не должен делать «агрегации», иначе это утечка ответственности.

#### Stage 2 — Storage Layer (`its_project/storage`)

**Назначение:** персистентность сырых потоков, warm+cold storage.

**Реализовано:**

- `BaseStorage` — абстракция `write/write_batch/read/get_latest/close`.
- `TimescaleStorage` (warm) — asyncpg pool, batch insert, чтение по интервалу.
- `ParquetStorage` (cold) — pyarrow dataset, партиционирование по дате.
- `writer.py` — `batch_writer_task` (батчи, backpressure).
- `api.py` — `StorageReadAPI` (warm-first fallback).
- `init.sql` — схема Timescale/Hypertable/индексы/компрессия (по описанию).

**Инварианты:**

- запись по батчам предпочтительнее одиночных записей.
- тип события хранится строкой `MarketData.type.value`.

#### Stage 3 — Preprocessing/Synchronization (`its_project/features/synchronizer.py`, `preprocessing.py`)

**Назначение:** синхронизация разных потоков, ресемплинг, обработка пропусков.

**Реализовано:**

- `marketdata_to_dataframe` — преобразование списка `MarketData` в DataFrame.
- `align_timestamps` — выравнивание к общей частоте (`ffill` или `interpolate`).
- `synchronize_marketdata` — сборка общего synced DataFrame.
- `extract_ohlcv_from_synced` — построение OHLCV.
- `handle_missing` — ffill/interpolate/drop.
- `normalize_features`, `denormalize_features` — чистые методы.
- `resample_to_uniform` — агрегация OHLCV.

**Инварианты:**

- Синхронизация не должна создавать look-ahead (на уровне ресемплинга/агрегации важно аккуратно выбирать методы).

#### Stage 4 — Feature Engineering (`its_project/features/*`)

**Назначение:** вычисление признаков, фиксированная размерность, детерминизм.

**Реализовано:**

- `BaseFeature` — интерфейс feature calculator (валидация OHLCV, чистые функции).
- `TechnicalFeatures` — RSI, MACD, Bollinger, ATR, Stoch.
- `OrderBookFeatures` — OFI, spread, imbalance, depth, VWAP, density.
- `MicrostructureFeatures` — Roll, VPIN, realized vol, Amihud, Kyle lambda.
- `FeaturePipeline` — иммутабельная композиция.
- `WindowedFeatures` — генерация окон.
- `FeatureScaler` — масштабирование.

**Инварианты:**

- фиксированная форма выходных тензоров,
- одинаковые имена признаков в том же порядке.

#### Stage 5 — Models (`its_project/models`)

**Назначение:** fit/predict/predict_proba/confidence, сериализация.

**Реализовано:**

- `BaseModel` — контракт.
- LSTM (PyTorch), Transformer (PyTorch), Ensemble (sklearn voting).
- `ModelRegistry` — create/list/load/save.

**Факт по коду:**

- Registry присутствует, но вопрос: действительно ли модели регистрируются декораторами (в ранних тестах мы видели отсутствие `@ModelRegistry.register` в файлах моделей; возможно регистрация реализована иначе через `__init__.py`). Это надо подтвердить запуском/импортом.

#### Stage 6 — Meta-learning (`its_project/metalearning`)

**Назначение:** авто-выбор модели, гипероптимизация, CV.

**Реализовано:**

- `MLMetrics` и `TradingMetrics`.
- `TimeSeriesSplitter`, `WalkForwardValidator`.
- Optuna hyperopt.
- ModelSelector + Stacking.

#### Stage 7 — Decision (`its_project/decision`)

**Назначение:** преобразование модельного сигнала в решение, риск, сайзинг.

**Реализовано:**

- Action/Signal/Decision контракты.
- SimpleDecisionMaker.
- RiskManager.
- PositionSizer (fixed/fraction/Kelly/risk-based).
- PortfolioManager.
- TradingDecisionEngine (Signal → Decision → Risk → Sizing → Final).

#### Stage 8 — Execution (`its_project/execution`)

**Назначение:** исполнение решений, paper/live, SL/TP, безопасность.

**Реализовано в коде:**

- BaseExecutor, Order/Position контракты.
- PaperTradingExecutor.
- LiveExecutor (ccxt).
- OrderManager.

**Разночтение с документацией:**

- В `PROJECT_STATUS.md` Execution помечен как PENDING, но в коде слой уже присутствует и довольно полный.

#### Stage 9 — Backtesting (`its_project/backtesting`)

**Назначение:** историческое моделирование, метрики, walk-forward.

**Реализовано:**

- BaseBacktester + cost accounting.
- SimpleBacktester (хронологический прогон).
- PerformanceAnalyzer (метрики/отчет).
- WalkForwardValidator (rolling validation, multi-asset).

---

### 1.3. Реальная связка слоев (как оно сейчас «течет»)

В `pipeline/queues.py` определены очереди:

- `price_raw`, `lob_raw`, `onchain_raw`, `sentiment_raw` — сырые потоки.
- `features_ready` — готовые окна/тензоры признаков.
- `predictions` — предсказания модели.
- `metalearning_results` — результаты выбора модели.
- `decisions` — решения для трейдинга.
- `orders` — результаты исполнения.

`app/main.py` (по смыслу) содержит задачи:

- публикация WS → очереди,
- batch writer → storage,
- preprocessing_task → чтение из storage → sync → features → `features_ready`,
- predictor_task → `features_ready` → модель → `predictions` (и слушает `metalearning_results`),
- metalearning_task → читает storage → оценивает модели → `metalearning_results`,
- decision_task → `predictions` → `decisions`,
- execution_task → `decisions` → `orders`.

Это уже очень близко к полноценной работающей системе.

---

## 2) Что планируется / что не закрыто полностью

Ниже — не «хотелки», а именно то, что вытекает из документации + наблюдений по коду.

### 2.1. Планируемые расширения по `PLAN.MD` и README

- Добавление health/metrics по источникам и storage writers.
- Улучшение reconnect логики.
- Интеграционные тесты внешних API.
- Оптимизации parquet read (partition pruning/metadata).
- Расширение набора моделей (GRU/CNN for LOB/XGBoost/LightGBM).
- Улучшение экспериментов/треккинга моделей.
- Расширение execution: частичные fills, продвинутые типы ордеров.
- Интерфейсный слой (дашборд/бот/CLI).
- Мониторинг/алертинг (Prometheus/Grafana и т.п.).

### 2.2. Выявленные “дыры”/несостыковки в текущем коде (как «планируемое исправление»)

1) **Конфиг ключей и секретов**
   - В проекте есть `glassnode_api_key` и `twitter_bearer_token` в `common/config.py`.
   - Для Binance ключи для live execution не подключены к `AppConfig` (по крайней мере, в текущем виде парсинга `load_config`).
   - Следовательно, чтобы реально включить LiveExecutor, нужно определить, где и как будут храниться `binance_api_key/binance_api_secret` и как они попадут в `LiveExecutor`.

2) **Зависимости vs реальное окружение**
   - pandas/numpy/pyarrow/asyncpg — обязательно, иначе часть модулей не импортируется.

3) **Look-ahead риск в некоторых местах**
   - В meta-learning в `app/main.py` формирование target использует `shift(-300)`.
   - Это нормально как "label future", но важно строго отделить:
     - что используется только для меток,
     - а признаки берутся только из прошлого.
   - Надо убедиться, что при преобразованиях/нормализациях не просачивается будущее.

4) **Observability**
   - Есть логирование, но нет метрик: лаги, длина очередей, drop rate, ошибки.

5) **Execution/Backtesting консистентность**
   - Реальный execution (ccxt) и backtesting должны иметь сопоставимую модель комиссий/проскальзывания/частичных исполнений — иначе переносимость стратегии сомнительна.

---

## 3) Полный блок вопросов по каждому модулю/слою (сильные вопросы-сомнения)

Вопросы сделаны так, чтобы:

- вскрывать скрытые допущения,
- находить слабые места в инженерии и трейдинге,
- задавать направления для чтения статей/книг.

### 3.1. Data Layer — источники данных

#### Про достоверность и микроструктуру
- Какие именно события Binance depth вы используете: `depth@100ms`/`@250ms` или raw depth updates? Насколько это достаточно для микроструктурных фич?
- Мы **восстанавливаем стакан** (order book reconstruction) или используем deltas как есть? Если не восстанавливаем — насколько корректны OFI/imbalance признаки?
- Как вы гарантируете корректный порядок deltas относительно snapshot? (lastUpdateId + U/u). Сейчас snapshot публикуется, но есть ли downstream логика "apply deltas only after lastUpdateId"?

#### Про качество и потери
- Что происходит при переполнении очередей `asyncio.Queue(maxsize=...)`? Мы блокируемся (backpressure) — это хорошо для целостности, но плохо для реального времени. Какой режим предпочтительнее?
- Как измеряем лаг: разницу между `_ts_recv_ms` и `timestamp_ms`? Где хранится/логируется?
- Есть ли дедупликация событий (например, повторные deltas после reconnect)?

#### Про API лимиты и стабильность
- RateLimiter учитывает ли особенности Binance (веса запросов, разные лимиты по endpoint)?
- Что будет при бане IP/HTTP 429? Есть ли глобальная стратегия backoff?

#### Про on-chain и sentiment
- Как вы выбираете on-chain метрики, чтобы они были **каузальны** и не опаздывали (реальные on-chain данные часто публикуются с задержкой)?
- Sentiment: насколько стабильна корреляция твитов и доходности? Как избежать "переобучения на шуме"?

---

### 3.2. Storage Layer — Timescale + Parquet

#### Про схему и ключи
- Какие гарантии уникальности записей? `ON CONFLICT DO NOTHING` — на каком уникальном индексе? Он вообще создан?
- Хранение `data` как JSONB удобно, но как будет происходить быстрый поиск по полям? Нужны ли GIN индексы?

#### Про warm/cold
- По каким правилам данные мигрируют из warm в cold? Это делается тем же writer’ом или отдельным процессом?
- Что важнее: скорость чтения или полнота? Как решаем конфликт "warm не доступен"?

#### Про целостность и тестируемость
- Как вы тестируете, что writer не теряет события при остановке? Есть ли graceful shutdown и drain очередей?
- Как воспроизводить датасеты для обучения (версионирование выгрузок)?

---

### 3.3. Features / Preprocessing

#### Про look-ahead
- Где именно вычисляются параметры нормализации (mean/std)? На всем интервале или только на train части? Как это будет обеспечено при backtesting и online?
- `align_timestamps(... ffill limit=1)` — почему limit=1? Что если пропуски длиннее? Мы выбрасываем информацию или вводим NaN?

#### Про контракты
- `extract_ohlcv_from_synced`: логика выбора price колонок и `data` поля — не ломается ли это при реальном `synced`, где `data` может быть не одна колонка, а prefixed?
- OHLCV ресемплинг 1s: почему 1s оптимально? Для LOB микроструктуры часто нужна 100ms/10ms.

#### Про производительность
- Какие операции самые тяжелые? OFI/VPIN могут быть дорогими.
- Нужна ли numba/polars уже сейчас, или сначала подтвердить, что стратегия работает?

---

### 3.4. Models

#### Про постановку задачи
- Почему 3 класса (sell/hold/buy) и фиксированные пороги? Это соответствует цели максимизации Sharpe/utility?
- Если цель торговая, почему оптимизируем accuracy? (accuracy почти всегда плохо коррелирует с PnL)

#### Про вход и форму данных
- Для LSTM/Transformer ожидается `(n, seq_len, n_features)`. В pipeline `predictor_task` добавляет `np.newaxis`, но правильно ли формируется seq_len?
- Где гарантируется, что `feature_names` согласованы между train и inference?

#### Про обучение
- Где хранится train loop, early stopping, seeds, воспроизводимость?
- Как защищаемся от data leakage при формировании train/test?

---

### 3.5. Meta-learning

- Почему веса метрик именно `[0.4, 0.6]` (accuracy vs sharpe)? Есть ли обоснование?
- Как корректно посчитать Sharpe на сигнале классификатора? Нужна симуляция стратегии, а не просто метрика по классам.
- CV: `TimeSeriesSplitter` — как выбирается test_size? нет ли слишком маленьких окон?
- Hyperopt: оптимизируем на одном сплите или на нескольких? Как избегаем overfitting на validation?

---

### 3.6. Decision

- RiskManager считает риск как 2% от position value — почему именно так? Где stop-loss логика связана с риском?
- Сайзер: Kelly в трейдинге чувствителен к оценке win_rate/avg_win/avg_loss — откуда эти оценки берутся online?
- PortfolioManager: поддерживает ли несколько активов и корреляцию? Сейчас корреляция = "тот же symbol" — достаточно ли?
- Где проходит граница между Decision и Execution? Кто отвечает за выставление SL/TP: decision или execution?

---

### 3.7. Execution

- PaperTradingExecutor не обновляет позиции (`_update_position` = pass). Значит ли это, что risk/portfolio в runtime не будет знать реальную позицию?
- LiveExecutor через ccxt: как обрабатываем частичные исполнения, проскальзывание, задержки, отмены?
- Какие circuit breakers предусмотрены? (max daily loss, max orders/min, emergency stop)
- Где хранятся API ключи? Сейчас есть риск держать их в `config.json`.

---

### 3.8. Backtesting

- Соответствует ли модель исполнения в backtest реальному execution? Сейчас комиссии/проскальзывание очень упрощены.
- Walk-forward: как гарантируется, что модель в каждом окне обучается только на train, и что нет leakage через feature scaling?
- Метрики: как интерпретируем profit factor = inf (нет убыточных сделок) — это может быть артефакт малого числа сделок.

---

### 3.9. App / Pipeline

- `app/main.py` использует pandas (`pd.Timestamp.now`) — проект предполагает обязательную зависимость pandas в runtime. Это нормально?
- Какие правила остановки задач? Stop_event есть, но где гарантия, что очереди будут корректно drained?
- Как вы будете запускать в проде: один процесс, несколько процессов, контейнеры? asyncio-очереди в одном процессе не масштабируются горизонтально.

---

### 3.10. Common / Config / Security

- Конфиг: почему часть секретов в `config.json`, а не в env vars/секрет-менеджере?
- Есть ли валидация конфигурации при старте? Если ключ отсутствует — сейчас просто `return None` в fetch_onchain/fetch_sentiment.

---

## 4) Рекомендуемая «карта чтения» (чтобы закрывать вопросы)

Это не список ссылок, а категории знаний, которые стоит покрыть для ответа на вопросы выше.

- Микроструктура рынка: order book dynamics, OFI, VPIN, Kyle lambda, влияние latency.
- Data leakage в временных рядах: walk-forward, purged CV, embargo.
- Оценка стратегий: дефляция Sharpe, multiple testing, overfitting, reality check.
- Position sizing: Kelly pitfalls, risk parity, volatility targeting.
- Execution: partial fills, slippage modeling, limit order vs market order impact.
- On-chain и sentiment: задержки данных, устойчивость сигналов, причинность.

---

## 5) Приложение: что нужно уточнить/подтвердить в проекте перед углублением

1) Как вы хотите хранить секреты: `config.json` для dev, env vars для prod?
2) Нужна ли поддержка multi-asset в runtime pipeline или только в backtesting?
3) Цель оптимизации: PnL/Sharpe/Sortino/Calmar, или accuracy как proxy?
4) Какая частота принятия решений: 1s, 5s, 1m? От этого зависит вся архитектура фич и execution.

---

### Статус документа

Документ создан и предназначен быть живым: по мере чтения книг/статей можно брать любой вопрос выше и:

- фиксировать вывод,
- обновлять требования,
- превращать вопрос в конкретную задачу/эксперимент.
