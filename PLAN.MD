Ок. Перестроено с учётом того, что архитектура уже определена. Только: **блок → что сделать → результат → стек**.

---

# ЭТАП 1 — Реализация сбора данных (Data Layer)

### Что сделать

* Реализовать подключение к бирже через REST (исторические данные)
* Реализовать WebSocket клиент для LOB
* Реализовать асинхронный сбор потоков:

  * цены
  * LOB
  * ончейн
  * sentiment
* Привести данные к единому формату (структура события)
* Реализовать обработку:

  * reconnect
  * rate limit
  * ошибки API

### Статус (выполнено / учесть / тонкости)

#### Выполнено

* Приведение данных к единому формату: `MarketData(timestamp_ms, symbol, type, exchange, data)`
* Асинхронные очереди `asyncio.Queue` для потоков: `price_raw`, `lob_raw`, `onchain_raw`, `sentiment_raw`
* Binance WebSocket:
  * trade → `price_raw`
  * depth (spot: `depth@100ms`, futures: `depth@250ms`) → `lob_raw`
  * события помечаются `_kind`: `trade` / `delta`
* Binance REST:
  * snapshot orderbook top-50 (публикуется в `lob_raw` как `_kind: snapshot`)
  * исторические свечи (klines) для spot и futures (async через `aiohttp`)
* Onchain (Glassnode): async polling задача (если задан `glassnode_api_key`)
* Sentiment (X/Twitter): async polling задача (если задан `twitter_bearer_token`)
* Базовый интерфейс источника данных (`BaseDataSource`) + rate limiter

#### Что учитывать

* Data Layer делает только I/O и унификацию формата. Не восстанавливает стакан и не считает признаки.
* Для LOB в downstream важно сохранять поля из Binance:
  * snapshot: `lastUpdateId`
  * delta: `U`, `u`, `b`, `a`
  Эти поля сейчас приходят в `data` как сырой payload Binance и должны сохраняться без изменений.
* При публикации snapshot используется backpressure-safe `await queue.put(...)`, чтобы snapshot не терялся при нагрузке.
* Для X API endpoint/домен зависит от выданного доступа (может потребоваться заменить `api.x.com` на `api.twitter.com`).
* Для Glassnode конкретные метрики и частоты обновления должны быть согласованы с реальными лимитами API.

#### Что можно сделать дальше (без выхода за слой)

* Добавить health/metrics для источников (alive, reconnect_count, queue_size) в отдельный объект состояния.
* Улучшить reconnect: exponential backoff + ограничение максимальной паузы + логирование причин.
* Добавить интеграционные тесты для:
  * Binance REST snapshot
  * Binance WS subscription (парсинг и маршрутизация)
* Вынести параметры polling/stream names полностью в конфиг (включая enable/disable флаги).

### Результат

* Потоки данных:

  * ( D^{LOB}, D^{price}, D^{onchain}, D^{sentiment} )
* Унифицированный формат данных (единая структура записи)
* Непрерывный streaming pipeline

### Стек

* Python 3.10+
* asyncio
* aiohttp
* websockets
* ccxt

---

# ЭТАП 0 — Каркас проекта + контракты + очереди + lifecycle (добавлено)

### Что сделать

* Создать каркас проекта (пакеты по слоям)
* Зафиксировать единый формат данных для Data Layer
* Реализовать in-process очереди `asyncio.Queue` между слоями
* Реализовать lifecycle запуска/остановки async-задач (graceful shutdown)

### Результат

* Минимально запускаемое приложение (entrypoint), которое:
  * создает очереди
  * запускает async-задачи источников
  * корректно останавливается

### Стек

* Python 3.10+
* asyncio

---

# ЭТАП 2 — Реализация хранилища (Storage Layer)

### Что сделать

* Реализовать сохранение данных в time-series формате
* Реализовать раздельное хранение:

  * streaming данных
  * исторических данных
* Реализовать быстрый доступ:

  * по времени
  * по инструменту
* Реализовать экспорт в формат для обучения (dataset dump)

### Результат

* Хранилище с доступом:

  * ( S(t, i) \rightarrow x_t^{(i)} )
* Возможность формировать выборки для обучения
* Исторический архив данных

### Стек

* InfluxDB / TimescaleDB
* Parquet
* pandas / polars

---

# ЭТАП 3 — Реализация синхронизации и preprocessing

### Что сделать

* Реализовать оператор синхронизации:

[
X_t = \Phi(D^{LOB}_t, D^{price}_t, D^{onchain}_t, D^{sentiment}_t)
]

* Привести данные к единой временной сетке
* Реализовать:

  * обработку пропусков
  * интерполяцию
  * нормализацию
* Реализовать формирование временных окон:

[
\mathbf{X}_t^{(L)}
]

### Результат

* Синхронизированный поток данных ( X_t )
* Готовые последовательности для моделей
* Pipeline preprocessing

### Стек

* numpy
* pandas / polars
* numba

---

# ЭТАП 4 — Реализация Feature Engineering

### Что сделать

* Реализовать расчет признаков:

  * OFI
  * агрегаты LOB (bid/ask imbalance, depth)
  * технические индикаторы
  * ончейн признаки
* Реализовать масштабирование признаков
* Реализовать формирование финального входного тензора

### Результат

* Полный набор признаков:

  * ( \mathbf{X}_t^{(L)} )
* Feature pipeline (переиспользуемый)

### Стек

* numpy
* polars
* ta
* numba

---

# ЭТАП 5 — Реализация Model Layer

### Что сделать

* Реализовать модели:

  * GRU
  * Transformer
  * CNN для LOB
  * XGBoost / LightGBM
* Реализовать обучение моделей
* Реализовать сохранение/загрузку моделей
* Реализовать единый интерфейс предсказания:

[
f_\theta(\mathbf{X}_t^{(L)}) \rightarrow (\hat{y}_t, \hat{p}_t, \widehat{\Delta P_t})
]

### Результат

* Набор моделей ( \mathcal{F} )
* Унифицированный API моделей
* Обученные веса моделей

### Стек

* PyTorch
* scikit-learn
* xgboost / lightgbm

---

# ЭТАП 6 — Реализация оптимизации (Hyperparameter tuning)

### Что сделать

* Задать search space для моделей
* Реализовать оптимизацию гиперпараметров
* Реализовать логирование результатов экспериментов

### Результат

* Оптимальные конфигурации моделей
* Повышенное качество моделей

### Стек

* optuna

---

# ЭТАП 7 — Реализация Meta-Learning Layer

### Что сделать

* Реализовать расчет метрик для моделей:

  * accuracy
  * RMSE
  * trading metrics
* Реализовать функцию выбора:

[
f^* = \arg\max \mathcal{M}(f_{\theta_i})
]

* Реализовать механизм обновления выбора модели

### Результат

* Модуль выбора модели
* Адаптивный выбор ( f^* )

### Стек

* numpy
* sklearn.metrics

---

# ЭТАП 8 — Реализация Decision Layer

### Что сделать

* Реализовать правило принятия решений:

[
a_t =
\begin{cases}
1, -1, 0
\end{cases}
]

* Реализовать настройку порога ( \tau )
* Реализовать базовый риск-менеджмент:

  * ограничение позиции
  * фильтрация сигналов

### Результат

* Поток торговых сигналов
* Фильтр по уверенности

### Стек

* Python

---

# ЭТАП 9 — Реализация Backtesting

### What was done

* Implemented BaseBacktester with cost accounting (commission, slippage)
* Implemented SimpleBacktester with strict no look-ahead bias
* Implemented Trade and BacktestResult dataclasses
* Implemented PerformanceAnalyzer with metrics and visualization
* Implemented WalkForwardValidator for rolling window validation
* Added multi-asset support and cross-asset comparison

### Status

#### Completed

* Base interfaces: Trade, BacktestResult, BaseBacktester
* Simple Backtester: No look-ahead, chronological order, cost accounting
* Performance Analyzer: Metrics, equity curve, drawdown, visualization
* Walk Forward Validator: Rolling window validation, multi-asset support
* Comprehensive validation checklist and best practices

#### What to consider

* CRITICAL: No look-ahead bias - only past data used for decisions
* Data MUST be sorted by timestamp
* Real trading costs included: commission (0.1%), slippage (0.05%)
* Walk-forward validation prevents overfitting
* Performance metrics: Sharpe, drawdown, win rate, profit factor

### Result

* Robust backtesting framework with no look-ahead bias
* Comprehensive performance analysis and visualization
* Walk-forward validation for strategy stability testing
* Multi-asset validation capabilities
* Integration ready with existing models and decision makers

### Stack

* Python
* pandas/numpy
* matplotlib/seaborn
* dataclasses

---

# ЭТАП 10 — Реализация Execution Layer

### Что сделать

* Реализовать paper trading
* Реализовать исполнение ордеров
* Реализовать журнал сделок

### Результат

* Исполняемые торговые сигналы
* История операций

### Стек

* ccxt

---

# ЭТАП 11 — Реализация интерфейса

### Что сделать

* Реализовать desktop UI:

  * графики
  * состояние системы
  * сигналы
* Реализовать Telegram-бот:

  * уведомления
  * команды управления

### Результат

* GUI приложения
* Telegram интерфейс

### Стек

* PyQt6
* aiogram

---

# ЭТАП 12 — Оптимизация и стабилизация

### Что сделать

* Ускорение feature pipeline (Numba)
* Подключение GPU для моделей
* Оптимизация задержек streaming
* Добавление логирования и мониторинга

### Результат

* Стабильная real-time система
* Оптимизированная производительность

### Стек

* numba
* PyTorch CUDA
* multiprocessing
* dask
