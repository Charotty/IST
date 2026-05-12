# Meta-Learning Layer

## Назначение

Адаптивный выбор и комбинирование моделей в зависимости от рыночных условий и режимов.

## Основные задачи

- Детекция рыночных режимов
- Адаптивное взвешивание моделей
- Выбор оптимального ансамбля
- Online адаптация к изменениям рынка

## Рыночные режимы

### 1. Trending

Направленное движение цены:
- Сильные тренды вверх или вниз
- Высокая корреляция с направлением
- Низкая волатильность относительно тренда

### 2. Ranging

Боковой рынок:
- Цена колеблется в диапазоне
- Низкая направленность
- Частые развороты

### 3. Volatile

Высокая волатильность:
- Большие ценовые движения
- Высокий риск
- Быстрые изменения

### 4. Neutral

Нейтральные условия:
- Отсутствие явных паттернов
- Случайные движения
- Низкая предсказуемость

## Ансамблирование моделей

### Weighted Ensemble

```python
ŷ = Σ(w_i * ŷ_i) for i=1 to N
```

где:
- ŷ - финальное предсказание
- w_i - вес i-ой модели
- ŷ_i - предсказание i-ой модели
- Σw_i = 1 (ограничение на веса)

### Dynamic Weighting

Адаптивное изменение весов в зависимости от:
- Текущей производительности моделей
- Рыночного режима
- Волатильности
- Времени суток

### Stacking Ensemble

Многоуровневое ансамблирование:
- Level 1: базовые модели
- Level 2: мета-модель для комбинирования

## Структура модуля

```
meta_learning/
├── __init__.py
├── regime_detection/
│   ├── __init__.py
│   ├── regime_detector.py     # Детекция режимов
│   ├── market_classifier.py   # Классификация рынка
│   └── regime_analyzer.py     # Анализ режимов
├── ensemble/
│   ├── __init__.py
│   ├── weighted_ensemble.py   # Взвешенный ансамбль
│   ├── stacking_ensemble.py   # Stacking ансамбль
│   └── dynamic_ensemble.py    # Динамический ансамбль
├── adaptation/
│   ├── __init__.py
│   ├── online_learner.py      # Online обучение
│   ├── drift_detector.py      # Детекция дрифта
│   └── adaptive_weights.py    # Адаптивные веса
├── selection/
│   ├── __init__.py
│   ├── model_selector.py      # Выбор моделей
│   ├── feature_selector.py     # Выбор признаков
│   └── hyperparameter_adapter.py # Адаптация гиперпараметров
├── optimization/
│   ├── __init__.py
│   ├── weight_optimizer.py     # Оптимизация весов
│   ├── ensemble_optimizer.py   # Оптимизация ансамбля
│   └── performance_tracker.py  # Отслеживание производительности
└── meta_manager.py             # Главный менеджер meta-learning
```

## Ключевые компоненты

### RegimeDetector

Детекция и классификация рыночных режимов:
- Статистические тесты
- Методы машинного обучения
- Временные паттерны

### WeightedEnsemble

Управление взвешенным ансамблем:
- Расчет оптимальных весов
- Динамическая адаптация
- Ограничения и регуляризация

### OnlineLearner

Адаптивное обучение в реальном времени:
- Инкрементальное обновление
- Детекция концепт дрифта
- Быстрая адаптация

### ModelSelector

Интеллектуальный выбор моделей:
- Оценка производительности
- Контекстный выбор
- Оптимизация ансамбля

## Алгоритмы детекции режимов

### Statistical Methods

```python
def detect_regime_statistical(returns, window):
    # ADF test for trend detection
    adf_result = adfuller(returns)
    
    # Volatility analysis
    volatility = returns.rolling(window).std()
    
    # Trend strength
    trend_strength = abs(returns.mean())
    
    return classify_regime(adf_result, volatility, trend_strength)
```

### Machine Learning Methods

```python
def detect_regime_ml(features, model):
    # Pretrained regime classifier
    regime_probabilities = model.predict_proba(features)
    
    # Most probable regime
    regime = model.predict(features)
    
    return regime, regime_probabilities
```

### Hybrid Approach

Комбинация статистических и ML методов для повышения точности.

## Оптимизация весов ансамбля

### Convex Optimization

```python
def optimize_weights(predictions, targets, constraints):
    # Quadratic programming for optimal weights
    # Minimize: ||predictions @ w - targets||²
    # Subject to: Σw_i = 1, w_i ≥ 0
    
    result = solve_qp(objective, constraints)
    return result.x
```

### Bayesian Optimization

Использование байесовской оптимизации для поиска оптимальных весов.

### Reinforcement Learning

RL агент для обучения оптимального взвешивания моделей.

## Адаптация к изменениям

### Concept Drift Detection

```python
def detect_concept_drift(model_performance, threshold):
    # Statistical test for performance degradation
    recent_performance = model_performance[-window:]
    historical_performance = model_performance[:-window]
    
    drift_score = ks_test(recent_performance, historical_performance)
    
    return drift_score > threshold
```

### Online Model Updates

Инкрементальное обновление моделей при детекции дрифта.

### Adaptive Thresholds

Адаптивная настройка порогов для принятия решений.

## Технологии

- **scikit-learn** - ML алгоритмы
- **scipy** - статистические тесты
- **cvxpy** - выпуклая оптимизация
- **optuna** - байесовская оптимизация
- **river** - online machine learning
- **pandas** - обработка временных рядов

## Конфигурация

```yaml
meta_learning:
  regime_detection:
    method: "hybrid"  # statistical, ml, hybrid
    window_size: 100
    update_frequency: 3600  # seconds
    
  ensemble:
    type: "weighted"  # weighted, stacking, dynamic
    rebalance_frequency: 300  # seconds
    min_weight: 0.05
    max_weight: 0.5
    
  adaptation:
    drift_detection_method: "ks_test"
    drift_threshold: 0.05
    adaptation_rate: 0.1
    
  optimization:
    method: "convex"  # convex, bayesian, rl
    optimization_frequency: 1800  # seconds
    regularization: "l2"
```

## Метрики оценки

### Quality of Regime Detection

- Accuracy of regime classification
- Transition detection accuracy
- Stability of regime assignments

### Ensemble Performance

- Ensemble vs individual model performance
- Weight stability over time
- Adaptation speed

### Adaptation Effectiveness

- Time to detect regime changes
- Performance improvement after adaptation
- Overfitting prevention

## Интеграция

Meta-Learning Layer получает данные от:
- **Models Layer** - предсказания базовых моделей
- **Feature Engineering** - признаки для детекции режимов

И передает результаты в:
- **Decision Layer** - финальные предсказания
- **RL Layer** - контекст для RL агента

## Требования к реализации

1. **Адаптивность** - быстрая реакция на изменения рынка
2. **Стабильность** - избежание резких переключений
3. **Производительность** - быстрые вычисления в real-time
4. **Надежность** - обработка ошибок и fallback
5. **Интерпретируемость** - понятные решения

## Тестирование

- Unit тесты для каждого компонента
- Integration тесты для pipeline
- Simulation тесты для различных рыночных условий
- Performance тесты для скорости адаптации
