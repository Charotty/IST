# Decision Layer

## Назначение

Центральный узел принятия решений, который интегрирует предсказания от ML моделей, RL политик и управление риском для генерации финальных торговых сигналов.

## Pipeline принятия решений

```text
Predictions (ML + RL)
    ↓
Threshold Filtering
    ↓
Ensemble Logic
    ↓
Risk Management
    ↓
Position Sizing
    ↓
Final Action
```

## Основные компоненты

### 1. Probability Filtering

Фильтрация сигналов по порогам вероятности:

```python
trade_if p̂ > τ
```

где:
- p̂ - предсказанная вероятность
- τ - пороговое значение

### 2. Ensemble Logic

Комбинирование предсказаний:

```python
Action = ML + RL + RiskFilter
```

### 3. Risk Management

Применение ограничений и правил управления риском.

### 4. Position Sizing

Расчет оптимального размера позиции.

## Структура модуля

```
decision/
├── __init__.py
├── filters/
│   ├── __init__.py
│   ├── probability_filter.py   # Фильтрация по вероятностям
│   ├── confidence_filter.py    # Фильтрация по уверенности
│   └── regime_filter.py        # Фильтрация по режимам
├── ensemble/
│   ├── __init__.py
│   ├── voting_ensemble.py      # Голосование
│   ├── weighted_ensemble.py    # Взвешенное комбинирование
│   └── stacking_ensemble.py    # Stacking ансамбль
├── risk_integration/
│   ├── __init__.py
│   ├── risk_filter.py          # Применение риск-фильтров
│   ├── position_limiter.py     # Ограничение позиций
│   └── exposure_controller.py   # Контроль экспозиции
├── sizing/
│   ├── __init__.py
│   ├── kelly_sizer.py          # Kelly criterion
│   ├── volatility_sizer.py     # Volatility-based sizing
│   └── risk_parity_sizer.py    # Risk parity sizing
├── execution/
│   ├── __init__.py
│   ├── signal_generator.py      # Генерация сигналов
│   ├── order_builder.py        # Построение ордеров
│   └── timing_optimizer.py     # Оптимизация тайминга
└── decision_engine.py           # Главный движок принятия решений
```

## Ключевые компоненты

### DecisionEngine

Центральный движок принятия решений:
- Координация всех компонентов
- Управление pipeline принятия решений
- Логирование и аудит решений
- Обработка исключительных ситуаций

### ProbabilityFilter

Фильтрация сигналов на основе вероятностей:
- Динамические пороги
- Адаптивная фильтрация
- Контекстные правила

### EnsembleLogic

Интеллектуальное комбинирование предсказаний:
- Взвешивание по надежности
- Учет рыночного контекста
- Динамическая адаптация весов

### RiskIntegrator

Интеграция управления риском:
- Применение стоп-лоссов
- Контролирование просадки
- Ограничение экспозиции

## Алгоритмы принятия решений

### Threshold-based Decision

```python
def threshold_decision(probabilities, thresholds):
    actions = []
    for prob in probabilities:
        if prob > thresholds['buy']:
            actions.append('BUY')
        elif prob < thresholds['sell']:
            actions.append('SELL')
        else:
            actions.append('HOLD')
    return actions
```

### Weighted Voting

```python
def weighted_voting(predictions, weights):
    weighted_sum = sum(pred * weight for pred, weight in zip(predictions, weights))
    total_weight = sum(weights)
    
    if weighted_sum / total_weight > buy_threshold:
        return 'BUY'
    elif weighted_sum / total_weight < sell_threshold:
        return 'SELL'
    else:
        return 'HOLD'
```

### Risk-adjusted Decision

```python
def risk_adjusted_decision(signal, risk_metrics, risk_limits):
    # Проверка риск-метрик
    if risk_metrics['drawdown'] > risk_limits['max_drawdown']:
        return 'HOLD'  # Принудительный HOLD при риске
    
    # Корректировка размера позиции
    adjusted_size = calculate_position_size(signal, risk_metrics)
    
    return signal, adjusted_size
```

## Position Sizing Algorithms

### Kelly Criterion

```python
def kelly_sizing(win_rate, avg_win, avg_loss):
    """
    Kelly Criterion: f* = (bp - q) / b
    где:
    - b = avg_win / avg_loss (win/loss ratio)
    - p = win_rate
    - q = 1 - p
    """
    b = avg_win / avg_loss if avg_loss != 0 else 1
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    return max(0, min(kelly_fraction, 0.25))  # Ограничение 25%
```

### Volatility-based Sizing

```python
def volatility_sizing(base_size, current_vol, target_vol, max_size):
    """
    Размер позиции на основе волатильности
    """
    volatility_adjustment = target_vol / current_vol
    adjusted_size = base_size * volatility_adjustment
    
    return min(adjusted_size, max_size)
```

### Risk Parity Sizing

```python
def risk_parity_sizing(portfolio_vol, target_vol, max_size):
    """
    Risk parity sizing
    """
    risk_adjustment = target_vol / portfolio_vol
    adjusted_size = risk_adjustment
    
    return min(adjusted_size, max_size)
```

## Управление порогами

### Adaptive Thresholds

Динамическая адаптация порогов на основе:
- Волатильности рынка
- Производительности стратегии
- Времени суток
- Рыночного режима

### Confidence-based Thresholds

Адаптация порогов на основе уверенности моделей:
- Высокая уверенность → низкие пороги
- Низкая уверенность → высокие пороги

## Технологии

- **numpy** - численные вычисления
- **pandas** - обработка данных
- **scipy** - оптимизация и статистика
- **asyncio** - асинхронная обработка
- **pydantic** - валидация данных

## Конфигурация

```yaml
decision:
  filtering:
    probability_thresholds:
      buy: 0.6
      sell: 0.4
      hold: 0.5
    
    confidence_threshold: 0.7
    regime_filtering: true
  
  ensemble:
    method: "weighted"  # voting, weighted, stacking
    weights:
      ml_models: 0.6
      rl_policy: 0.4
    
    rebalance_frequency: 3600  # seconds
  
  risk_integration:
    max_position_size: 0.1
    max_portfolio_risk: 0.02
    max_drawdown_limit: 0.05
  
  sizing:
    method: "volatility"  # kelly, volatility, risk_parity
    base_size: 0.05
    max_size: 0.2
    target_volatility: 0.15
```

## Метрики принятия решений

### Decision Quality

- **Accuracy**: точность решений
- **Precision**: точность положительных сигналов
- **Recall**: полнота обнаружения возможностей
- **F1-Score**: баланс precision/recall

### Risk Metrics

- **Position Size Consistency**: стабильность размеров позиций
- **Risk-adjusted Returns**: доходность с поправкой на риск
- **Drawdown Control**: эффективность контроля просадки

### Performance Metrics

- **Decision Latency**: задержка принятия решений
- **Signal Quality**: качество генерируемых сигналов
- **Execution Efficiency**: эффективность исполнения

## Интеграция

Decision Layer получает данные от:
- **Models Layer** - предсказания и вероятности
- **Meta-Learning** - ансамблированные предсказания
- **RL Layer** - RL политика действий
- **Risk Management** - риск-метрики и ограничения

И передает решения в:
- **Execution Layer** - финальные ордера для исполнения
- **Risk Management** - обновление риск-метрик

## Требования к реализации

1. **Скорость** - минимальные задержки принятия решений
2. **Надежность** - обработка ошибок и fallback механизмы
3. **Гибкость** - легкая настройка правил и параметров
4. **Тестируемость** - возможность бэктестинга решений
5. **Аудит** - полное логирование всех решений

## Тестирование

- Unit тесты для каждого компонента
- Integration тесты для pipeline
- Simulation тесты для различных сценариев
- Performance тесты для скорости принятия решений
