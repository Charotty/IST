# Decision Layer

## Назначение

Тонкий слой **правил сборки торгового решения** из выходов models + meta_learning. В `ist.py` отдельного `DecisionEngine` нет — логика вынесена сюда как контракт для production.

> Основная «тяжёлая» логика (мета-фильтр, ансамбль, пороги) живёт в **meta_learning**. Decision — финальные правила и точка входа для risk / backtest.

## Pipeline

```text
direction_soft_signal  (DirectionModel, τ=0.52)
meta_prob | meta_mgmt_prob  (MetaFilter | Dynamic Ensemble)
    ↓
threshold rules  →  final_signal | integrated_signal
    ↓
[risk_management] ATR stop, PositionSizer, RL multiplier
    ↓
[backtesting] / [execution — позже]
```

## Правила (эталон)

### Вариант A — MetaFilter + MTF (`final_signal`)

```python
meta_threshold = meta_prob.median()
final_signal = np.where(
    (meta_prob > meta_threshold) & (direction_soft_signal != 0),
    direction_soft_signal,
    0
)
```

### Вариант B — Regime-Adaptive Ensemble (`integrated_signal`) — **рекомендуемый**

```python
meta_threshold = meta_mgmt_prob.median()
integrated_signal = np.where(
    (meta_mgmt_prob > meta_threshold) & (direction_soft_signal != 0),
    direction_soft_signal,
    0
)
```

### Параметры

| Параметр | Значение |
|----------|----------|
| `direction_threshold` | **0.52** |
| `meta_threshold_mode` | **median** на текущей выборке |

### Сигналы

| Значение | Смысл |
|----------|--------|
| `1` | Long |
| `-1` | Short |
| `0` | Flat / skip |

## Что не входит в Decision

- Обучение моделей → **models**  
- Веса ансамбля → **meta_learning**  
- ATR stop, size, RL → **risk_management** / **rl_layer**  
- Симуляция PnL → **backtesting**  

## Структура модуля (целевая)

```
decision/
├── __init__.py
├── signal_rules.py      # final_signal, integrated_signal
└── decision_pipeline.py # выбор варианта A/B из config
```

## Конфигурация

```yaml
decision:
  signal_source: "integrated"    # final | integrated
  direction_threshold: 0.52
  meta_threshold_mode: "median"
```

## Интеграция

| Вход | Выход |
|------|--------|
| meta_learning | `final_signal` или `integrated_signal` |
| risk_management | тот же ряд после stop/sizing |
| backtesting | колонка `signal` для `Backtester.run()` |

## Roadmap

- Асимметричные пороги long/short  
- Фиксированный `meta_threshold` vs rolling median  
- Единый `DecisionPipeline` class вместо inline `np.where`  
