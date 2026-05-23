# 3.7 Реализация adaptive ensemble и dynamic weighting

Режимно-адаптивный ансамбль — центральный мета-уровень ИТС: на каждом баре объединяются вероятности четырёх моделей (п. 3.5) с весами, зависящими от классификации рынка (п. 3.6). Реализация сосредоточена в `meta_learning/dynamic_meta.py` (класс `DynamicMetaWeighting`); создание экземпляра по конфигурации — `orchestration/model_factory.meta_weighting_from_config()`. Ниже — особенности кода и сравнение со статическим усреднением; формулы не приводятся.

## 3.7.1. Конвейер агрегирования

Последовательность вызовов в оркестраторе (`TrainingOrchestrator`, `InferenceOrchestrator`):

1. `collect_predictions(features)` — словарь `{lgb, xgb, gru, cnn: массив вероятностей}`;
2. `regime_detector` — массив `regime_pred` (1 = trend, 0 = range);
3. `DynamicMetaWeighting.get_weights(regime_pred, mode)` — для каждой модели массив весов по барам;
4. `apply_dynamic_weighting(predictions, regime_pred, mode)` — итоговая вероятность ансамбля `meta_mgmt_prob`;
5. при необходимости `get_integrated_signal(...)` или передача в `DecisionPipeline` (п. 3.8).

Схема потока данных:

```
lgb, xgb, gru, cnn  →  predict_proba
         ↓
regime_pred (trend/range)
         ↓
DynamicMetaWeighting  →  meta_mgmt_prob  →  DecisionPipeline
```

Базовый бенчмарк без режимной логики — `meta_learning/ensemble.py` (`EnsembleAggregator`: простое или фиксированное взвешенное среднее). В прод-контуре используется только `DynamicMetaWeighting`.

## 3.7.2. Режимы `ensemble_mode`

Параметр `OrchestratorConfig.ensemble_mode` (профиль `canonical_4model.yaml`):

| Режим | Поведение в коде |
|-------|------------------|
| `regime_adaptive` | На каждом баре: если `regime_pred == 1` — строка `trend_weights`, иначе — `range_weights` |
| `fixed_trend` | Все бары: веса из `trend_weights` (`_fixed_weights`) |
| `fixed_range` | Все бары: веса из `range_weights` |

Метод `_regime_adaptive_weights` заполняет массив весов модели поэлементно: для индексов с trend подставляется коэффициент из `trend_weights[model]`, для range — из `range_weights[model]`. Сумма весов по активным моделям в каждой строке YAML нормируется до 1 при загрузке конфигурации (`_renormalize_weight_subset` в `orchestrator_config.py`).

## 3.7.3. Таблицы весов в конфигурации

**Таблица 3.16 — `trend_weights` (канонический профиль)**

| Модель | Вес |
|--------|-----|
| lgb | 0,10 |
| gru | 0,45 |
| xgb | 0,10 |
| cnn | 0,35 |

**Таблица 3.17 — `range_weights` (канонический профиль)**

| Модель | Вес |
|--------|-----|
| lgb | 0,55 |
| gru | 0,10 |
| xgb | 0,25 |
| cnn | 0,10 |

В тренде усиливаются последовательные модели (GRU, CNN); во флэте — табличные (LightGBM, XGBoost). Имена ключей должны совпадать с `model_keys`; в docstring `DynamicMetaWeighting` по умолчанию ещё указаны устаревшие ключи `lstm` / `trans` — в оркестраторе используются только `lgb`, `gru`, `xgb`, `cnn`.

Пересчёт весов в рантайме: `update_weights(trend_weights=..., range_weights=...)` с нормализацией суммы к 1.

## 3.7.4. Динамика весов во времени

На рис. 3.16 показаны траектории весов моделей на фрагменте истории: при смене `regime_pred` веса **ступенчато** переключаются между табл. 3.16 и 3.17.

**Рисунок 3.16 — Изменение весов моделей во времени**

![Рис. 3.16 — weights over time](figures/3_7/weights_over_time.png)

**Рисунок 3.17 — Статический vs адаптивный ансамбль: веса во времени**

![Рис. 3.17 — static vs adaptive weights](figures/3_7/static_vs_adaptive_weights.png)

Верхняя панель рис. 3.17 — постоянные веса 0,25 (режим `four_equal` / fixed); нижняя — regime-adaptive.

## 3.7.5. Сравнение со статическим ансамблем (ablation)

Скрипт `scripts/run_ablation.py`, профиль `canonical_4model`, демо-выборка. Варианты:

| Вариант | `ensemble_mode` / веса | Смысл |
|---------|------------------------|--------|
| `four_equal` | равные 0,25 на все модели | статический бенчмарк |
| `four_regime_adaptive` | табл. 3.16–3.17 | основной режим ИТС |

**Таблица 3.18 — OOS-метрики ablation (среднее по фолдам WFO)**

| Метрика | Статический (four_equal) | Адаптивный (regime_adaptive) |
|---------|--------------------------|------------------------------|
| Mean Sharpe | −2,49 | **0,29** |
| Mean Profit Factor | 0,93 | **1,18** |
| Mean WFE | 0,03 | **0,62** |

Источник: `docs/reports/ablation_canonical.json`.

**Рисунок 3.18 — Сравнение метрик: статический и адаптивный ансамбль**

![Рис. 3.18 — static vs adaptive metrics](figures/3_7/static_vs_adaptive_metrics.png)

Адаптивное взвешивание улучшает переносимость стратегии с обучения на тест (WFE) и риск-скорректированную доходность относительно равномерного усреднения без учёта режима.

## 3.7.6. Связь с decision layer и статистикой режима

- **Итоговая вероятность** после `apply_dynamic_weighting` передаётся в `DecisionPipeline` как `meta_mgmt_prob` (пороги — п. 3.8).
- **Устаревший путь:** `get_integrated_signal` совмещает `meta_mgmt_prob`, `direction_soft_signal` и фиксированный `meta_threshold` через `np.where`; в каноническом профиле предпочтителен отдельный `DecisionPipeline` с каузальным медианным порогом.
- **Диагностика:** `get_regime_statistics(regime_pred)` возвращает доли баров trend/range для отчётов и GUI.

## 3.7.7. Выводы по разделу

1. Реализован класс `DynamicMetaWeighting` с тремя режимами агрегации и конфигурируемыми таблицами весов trend/range.
2. Оркестратор на каждом баре вызывает все модели и применяет режимно-зависимые веса, а не маршрутизацию одной модели (`ModelRouter`).
3. Ablation подтверждает преимущество `regime_adaptive` над статическим равным ансамблем по Sharpe, profit factor и WFE.
4. Иллюстрации (рис. 3.16–3.18) отражают переключение весов и сравнение метрик без отдельного математического описания.

Агрегирование прогнозов передаётся в тестирование системы (п. 3.11) и в подсистему принятия решений (п. 3.8).
