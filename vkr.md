Вот **полный план трансформации системы из исследовательской в production-уровень**
(структура: блок → подблок → конкретные действия → результат)

---

# 1. DATA LAYER (УЛУЧШЕНИЕ ДАННЫХ)

## 1.1 Качество данных

* Проверка целостности потоков (WebSocket reconnect, gap detection)
* Реализация replay механизма
* Валидация данных (NaN, spikes, outliers)

**Результат:** стабильный поток без пропусков

---

## 1.2 LOB данные

* Увеличение глубины (top 50–100)
* Нормализация цен/объёмов
* Выделение microprice

**Результат:** информативные high-frequency признаки

---

## 1.3 Расширение источников

* Funding rates
* Open Interest
* Liquidations
* Market indices

**Результат:** усиление сигнала

---

## 1.4 Data sampling

* Переход:

  * 1s → multi-timeframe (1s, 5s, 1m)
* Resampling pipeline

**Результат:** устойчивость моделей

---

# 2. FEATURE ENGINEERING (КЛЮЧЕВОЙ БЛОК)

## 2.1 Базовые признаки (обязательно)

* Returns:

  * ( r_t = \frac{P_t - P_{t-1}}{P_{t-1}} )
* Log returns
* Lag features (t-1 … t-n)

**Результат:** базовый сигнал для моделей

---

## 2.2 Волатильность

* Rolling std
* ATR
* Realized volatility

**Результат:** учет рыночных режимов

---

## 2.3 LOB признаки (расширение)

* Order flow imbalance
* Queue imbalance
* Depth ratio
* Liquidity slope

**Результат:** микроструктурный edge

---

## 2.4 Regime detection

* Trending / ranging / volatile
* Кластеризация или rule-based

**Результат:** адаптация стратегии

---

## 2.5 Feature selection

* Correlation filtering
* SHAP / feature importance
* Drop noise features

**Результат:** снижение переобучения

---

## 2.6 Scaling

* Online normalization (rolling window)
* Separate scaling train/test

**Результат:** корректная работа моделей

---

# 3. TARGET ENGINEERING

## 3.1 Переформулировка задачи

* Классификация:

  * up / down / flat
* Regression:

  * future return

---

## 3.2 Horizon tuning

* Проверка:

  * +5s
  * +30s
  * +1m

---

## 3.3 Threshold labeling

* Убрать noise:

  * flat зона

**Результат:** обучаемый таргет

---

# 4. MODEL LAYER (УПРОЩЕНИЕ И УСИЛЕНИЕ)

## 4.1 Базовые модели (приоритет)

* LightGBM / XGBoost (основа)
* Logistic Regression (baseline)

---

## 4.2 Deep Learning (ограничить)

* Оставить:

  * 1–2 модели (GRU или Transformer)
* Убрать лишние

---

## 4.3 Обучение

* Walk-forward training
* Cross-validation (time-series)

---

## 4.4 Регуляризация

* Dropout
* Early stopping
* L1/L2

---

## 4.5 Калибровка вероятностей

* Platt scaling
* Isotonic regression

**Результат:** корректные p_hat

---

# 5. META-LEARNING (РЕАЛЬНОЕ УСИЛЕНИЕ)

## 5.1 Weighted ensemble

* Веса по Sharpe / PnL

---

## 5.2 Regime-based selection

* Разные модели для разных режимов

---

## 5.3 Sliding window evaluation

* Online обновление весов

---

## 5.4 Auto-retrain

* Переобучение:

  * по расписанию
  * при деградации

**Результат:** адаптивная система

---

# 6. DECISION LAYER (КРИТИЧНЫЙ БЛОК)

## 6.1 Логика сигнала

(выбранный вариант 2)

```
action = sign(ΔP_hat)
if p_hat < τ → HOLD
if |ΔP_hat| < min_move → HOLD
```

---

## 6.2 Threshold optimization

* Grid search
* Walk-forward tuning

---

## 6.3 Signal filtering

* Cooldown между сделками
* Minimum confidence
* Volatility filter

---

## 6.4 Position sizing

* Fixed fraction
* Kelly approximation
* Volatility scaling

---

## 6.5 Risk management

* Stop-loss
* Take-profit
* Max drawdown limit

**Результат:** контролируемая торговля

---

# 7. BACKTESTING (ПРИВЕСТИ К РЕАЛЬНОСТИ)

## 7.1 Реалистичные условия

* Комиссии (реальные)
* Slippage (динамический)
* Latency

---

## 7.2 Метрики

* Sharpe
* Sortino
* Calmar
* Max DD

---

## 7.3 Walk-forward (обязательно)

* Train → Test → Shift

---

## 7.4 Benchmark

* Buy & Hold
* Random strategy

**Результат:** честная оценка

---

# 8. EXECUTION (ПЕРЕХОД К REAL TRADING)

## 8.1 Order execution

* Smart order routing
* Partial fills handling

---

## 8.2 Latency

* Измерение:

  * data → signal → order

---

## 8.3 Risk limits

* Max position
* Max loss/day

---

## 8.4 Fail-safe

* Kill switch
* Disconnect handling

---

# 9. SYSTEM (INFRASTRUCTURE)

## 9.1 Pipeline

* Финализировать orchestrator
* Централизованные очереди

---

## 9.2 Monitoring

* Latency
* PnL
* Errors

---

## 9.3 Logging

* Structured logs
* Error tracking

---

## 9.4 Config system

* Полная параметризация

---

# 10. UI (ПРОДУКТОВАЯ ЧАСТЬ)

## 10.1 Dashboard

* Цена + сигналы
* PnL
* Позиции

---

## 10.2 Контроль

* Start/Stop
* Выбор модели
* Параметры

---

## 10.3 Метрики

* Sharpe
* Win rate

---

# 11. MLOPS (ОБЯЗАТЕЛЬНО ДЛЯ PROD)

## 11.1 Versioning

* Данные
* Модели

---

## 11.2 Experiment tracking

* MLflow / аналог

---

## 11.3 Drift detection

* Data drift
* Model decay

---

## 11.4 Auto pipeline

* Training → Evaluation → Deploy

---