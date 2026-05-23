# 3.8 Реализация системы принятия решений

Слой принятия решений (decision layer) преобразует вероятность адаптивного ансамбля `meta_mgmt_prob` (п. 3.7) и вспомогательный направленный сигнал `direction_soft_signal` в дискретные команды **BUY** (+1), **SELL** (−1) или **HOLD** (0). Реализация — пакет `decision`: правила в `signal_rules.py`, фасад — `DecisionPipeline` в `decision_pipeline.py`. Оркестратор подключает пайплайн при `apply_decision_pipeline: true` (`TrainingOrchestrator._build_decision_pipeline`). Формулы порогов не приводятся; ниже — логика кода и конфигурация.

## 3.8.1. Коды сигналов и два варианта правил

**Таблица 3.19 — Кодировка торгового сигнала**

| Значение | Действие | Колонка в DataFrame |
|----------|----------|---------------------|
| +1 | BUY (long) | `signal` |
| −1 | SELL (short) | `signal` |
| 0 | HOLD (flat) | `signal` |

В `DecisionPipeline` задаётся `signal_source`:

| Вариант | Функция | Вход «мета»-вероятности |
|---------|---------|-------------------------|
| `final` | `compute_final_signal` | `meta_prob` (MetaFilter, legacy) |
| `integrated` | `compute_integrated_signal` | `meta_mgmt_prob` (DynamicMetaWeighting) |

В каноническом профиле используется **`integrated`** — согласован с regime-adaptive ансамблем.

Общая логика обоих вариантов в `signal_rules.py`:

1. по `direction_soft_signal` и `direction_threshold` формируется направление: выше порога → +1, ниже отрицательного порога → −1, иначе 0;
2. вычисляется **каузальный** мета-порог по истории `meta_mgmt_prob` (или `meta_prob`) — режим `median` / `mean` / `fixed`, окно по умолчанию 100 бар (`threshold_window`);
3. сделка разрешена, если мета-вероятность **строго выше** порога на этом баре и направление ненулевое; иначе 0.

При `safe_mode: true` порог считается через `utils.data_leakage_prevention.compute_safe_threshold` (только прошлые бары, без look-ahead). На OOS допускается фиксированный `train_threshold` с обучающего фолда.

## 3.8.2. Класс DecisionPipeline

**Точка входа:** `DecisionPipeline.generate_signal(...)` или `add_signal_to_df(df, signal_column='signal')`.

Параметры из конфигурации (словарь или YAML-секция `decision`):

| Параметр | Значение в `canonical_4model` | Назначение |
|----------|-------------------------------|------------|
| `signal_source` | `integrated` | выбор `compute_integrated_signal` |
| `direction_threshold` | 0,52 | порог направления относительно «нейтрали» 0,5 |
| `meta_threshold_mode` | `median` | скользящая медиана `meta_mgmt_prob` |
| `meta_threshold` | 0,5 | запасное значение для режима `fixed` |
| `threshold_rolling_window` | 100 | длина окна мета-порога |
| `safe_mode` | true | каузальный расчёт порога |
| `use_asymmetric_thresholds` | false | отдельные long/short пороги (roadmap) |

Дополнительные фильтры задаются в **оркестраторе**, не в `DecisionPipeline`:

- `min_signal_margin` — «мёртвая зона» вокруг 0,5 по вероятности;
- `trade_mode` — `both` / `long_only` / `short_only`;
- `volatility_filter_percentile` — отсечение по волатильности.

Диагностика блокировки сигнала: `orchestration/introspect.py` (поле `why_blocked`).

## 3.8.3. Конвейер в оркестраторе

```
meta_mgmt_prob, direction_soft_signal
         ↓
   DecisionPipeline.generate_signal
         ↓
   final_signals  →  risk bridge (п. 3.9)  →  Backtester
```

`TrainingOrchestrator` при инициализации создаёт `DecisionPipeline` из полей `OrchestratorConfig` (`direction_threshold`, `meta_threshold_mode`, `threshold_rolling_window`, …). Результат попадает в `TrainingResult.final_signals`.

Устаревный путь: `DynamicMetaWeighting.get_integrated_signal` — дублирует фильтр порога внутри meta-модуля; в новых прогонах предпочтителен отдельный `DecisionPipeline`.

## 3.8.4. Пороги и консервативный режим

**Таблица 3.20 — Пороги в конфигурации ИТС**

| Условие | Параметр | Канон / примечание |
|---------|----------|-------------------|
| Long | `direction_threshold` | 0,52 (симметрично: short при значении ниже `1 − 0,52`) |
| Уверенность ансамбля | `meta_threshold_mode` + окно 100 | медиана `meta_mgmt_prob` по прошлому |
| Усиленная торговля (эксперимент) | пороги 0,65 / 0,35 | отдельные прогоны в журнале (`direction_threshold: 0.6`) |

Функция `apply_asymmetric_thresholds` в `signal_rules.py` поддерживает разные `long_threshold` и `short_threshold`; в каноническом YAML отключена (`use_asymmetric_thresholds: false`).

## 3.8.5. Примеры сигналов на графике

**Рисунок 3.19 — Маркеры BUY/SELL на свечном графике (канонические пороги)**

![Рис. 3.19 — signals buy/sell](figures/3_8/signals_buy_sell_chart.png)

**Рисунок 3.20 — Сигналы при ужесточённых порогах (сравнение)**

![Рис. 3.20 — strict thresholds](figures/3_8/signals_strict_threshold.png)

Фрагмент журнала сигналов для таблиц в приложении: `docs/figures/3_8/signal_examples.csv` (колонки `timestamp`, `P_hat`, `signal`, `close`).

## 3.8.6. Статистика и выводы

`DecisionPipeline.get_signal_stats(signal)` возвращает число и доли long / short / flat — используется в отчётах и GUI.

**Выводы по разделу:**

1. Реализован единый пайплайн с двумя источниками правил; в прод-контуре — вариант **integrated** + `meta_mgmt_prob`.
2. Мета-порог считается **каузально** (`safe_mode`, rolling median), что согласовано с WFO и purge/embargo (п. 3.11).
3. Дискретный сигнал передаётся в риск-менеджмент и бэктест; дополнительные ограничения — на уровне оркестратора.
4. Рис. 3.19–3.20 иллюстрируют различие частоты входов при стандартных и ужесточённых порогах.

Далее — подсистема управления рисками (п. 3.9).
