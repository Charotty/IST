# 3.10 Архитектура программного обеспечения

Интеллектуальная торговая система (ИТС) реализована как модульный Python-проект с разделением по слоям: данные, признаки, модели, мета-уровень, решения, риски, бэктест и опционально GUI/исполнение. Сквозной контракт пайплайна зафиксирован в `OrchestratorConfig` и `TrainingOrchestrator` (пакет `orchestration`). Ниже — структура репозитория, назначение модулей и конфигурация без UML-формализмов.

## 3.10.1. Дерево проекта

Актуальный снимок каталогов (уровни 1–3) хранится в `docs/figures/3_10/project_tree.txt` (генерация: `docs/scripts/generate_3_10_tree.py`). Фрагмент корневой структуры:

```
IST/
|-- data_layer/          # загрузка OHLCV, Parquet
|-- synchronization/     # MTF, слияние таймфреймов
|-- feature_engineering/ # индикаторы, feature_engine
|-- models/              # LGB, XGB, GRU, CNN, regime, router (legacy)
|-- meta_learning/       # DynamicMetaWeighting, ensemble baseline
|-- orchestration/       # WFO, фабрика моделей, glue
|-- decision/            # DecisionPipeline, signal_rules
|-- risk_management/     # RiskPipeline, OrchestratorRiskBridge
|-- backtesting/         # Backtester, метрики, журнал
|-- execution/           # paper/OKX broker (опционально)
|-- gui/                 # PyQt6, IstGuiClient
|-- config/              # profiles, symbols, archive
|-- artifacts/           # сохранённые модели по символу
+-- docs/                # разделы ВКР, figures, backtest_journal
```

**Рисунок 3.23 — Дерево проекта ИТС (полный текст)**

Полное дерево — в файле `figures/3_10/project_tree.txt` (при оформлении ВКР вставляется как рисунок или листинг в приложении).

## 3.10.2. Назначение основных пакетов

**Таблица 3.22 — Модули ИТС и точки входа**

| Пакет | Ключевые файлы | Роль |
|-------|----------------|------|
| `data_layer` | `okx_ohlcv_loader.py`, `cli.py` | CCXT/OKX, валидация OHLCV, Parquet |
| `synchronization` | `multi_timeframe_engine.py` | 15m/1h/4h → базовый 1h |
| `feature_engineering` | `feature_engine.py`, `indicators.py` | матрица признаков для ML |
| `models` | `tabular/`, `trend/`, `volatility/`, `regime/` | обучаемые предикторы и детектор режима |
| `meta_learning` | `dynamic_meta.py`, `ensemble.py` | взвешивание прогнозов (п. 3.7) |
| `orchestration` | `training_orchestrator.py`, `model_factory.py`, `symbol_pipeline.py` | WFO, сборка пайплайна |
| `decision` | `decision_pipeline.py` | BUY/SELL/HOLD (п. 3.8) |
| `risk_management` | `risk_pipeline.py`, `orchestrator_risk_bridge.py` | размер позиции (п. 3.9) |
| `backtesting` | `backtester.py`, `results_journal.py` | симуляция и критерии acceptance |
| `gui` | `app/main_window.py`, `api/client.py` | просмотр прогонов и графиков |
| `utils` | `data_leakage_prevention.py` | purge, embargo, safe thresholds |

Legacy-контур `models.router.ModelRouter` + `InferenceEngine` (одна модель на бар) сохранён для сравнения; **прод-путь** — `TrainingOrchestrator` / `InferenceOrchestrator` со вызовом всех `model_keys`.

## 3.10.3. Сквозной пайплайн оркестратора

Контракт из docstring `OrchestratorConfig`:

```text
features → regime → {p_lgb, p_xgb, p_gru, p_cnn} → meta_mgmt_prob → decision → risk → backtest
```

**Рисунок 3.24 — Логическая схема оркестратора (поток данных)**

```
[Parquet features]
       |
       v
 TrainingOrchestrator.walk_forward_backtest
       |
       +-- DataLeakagePreventer (purge / embargo)
       +-- build_orchestration_models (model_factory)
       +-- RegimeDetector -> regime_pred
       +-- collect_predictions -> dict probabilities
       +-- DynamicMetaWeighting.apply_dynamic_weighting
       +-- DecisionPipeline.generate_signal
       +-- OrchestratorRiskBridge.calculate_position_sizes
       +-- Backtester.run + PerformanceMetrics
       v
 docs/backtest_journal/, docs/reports/
```

Типовые CLI/сценарии:

- `python -m orchestration validate-config --config config/profiles/canonical_4model.yaml`;
- `orchestration/glue.py` — `run_wfo_backtest_from_parquet`;
- `orchestration/symbol_pipeline.py` — подготовка символа end-to-end;
- `scripts/run_ablation.py` — сравнение вариантов ансамбля.

## 3.10.4. Конфигурация

Конфигурация — YAML, без жёсткой привязки к коду.

| Файл | Назначение |
|------|------------|
| `config.yaml` | корень: data_layer, backtesting, orchestration |
| `config/profiles/canonical_4model.yaml` | канон: 4 модели, WFO, regime weights |
| `config/symbols/BTC-USDT_1h.yaml` | переопределения по символу (тюнинг) |
| `config/archive/discussion/` | архив экспериментов (не default) |

Загрузка: `OrchestratorConfig.from_yaml(path)` (`orchestrator_config.py`, dataclass). Веса trend/range нормируются под активный список `model_keys` (`_renormalize_weight_subset`).

Фрагмент секции orchestration (канон):

```yaml
orchestration:
  model_keys: [lgb, gru, xgb, cnn]
  ensemble_mode: regime_adaptive
  apply_decision_pipeline: true
  direction_threshold: 0.52
  train_window_size: 1500
  test_window_size: 250
  walk_forward_step: 250
  use_risk_bridge: true
  trend_weights: { lgb: 0.10, gru: 0.45, xgb: 0.10, cnn: 0.35 }
  range_weights: { lgb: 0.55, gru: 0.10, xgb: 0.25, cnn: 0.10 }
```

Переменные окружения для API: `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE` (см. `data_layer` connectors).

## 3.10.5. Паттерны реализации

| Паттерн | Где проявляется |
|---------|------------------|
| **Factory** | `build_orchestration_models`, `meta_weighting_from_config` |
| **Dataclass config** | `OrchestratorConfig`, `LeakageConfig`, `MetricsConfig` |
| **Adapter** | `_TrainCallableAdapter`, `OrchestratorRiskBridge` |
| **Lazy import** | `models/__init__.py` (`__getattr__`), тяжёлый TF только для gru/cnn |
| **Journal / artifacts** | `backtesting/results_journal.py`, `orchestration/artifact_bundle.py` |

Тесты: каталог `tests/` (`test_orchestration_four_models.py`, `test_wfo_backtest.py`, `test_data_leakage` и др.).

## 3.10.6. GUI и исполнение

**GUI** (`gui/`): PyQt6, `IstGuiClient` агрегирует API (`backtests_api`, `inference_api`, `jobs_api`). Экраны: `overview_view`, `chart_view`, `backtests_view`, `models_view`, `execution_view`, `settings_view`. Журнал WFO читается из `docs/backtest_journal/`.

**Execution** (`execution/`): `paper_broker`, `okx_broker`, `execution_manager` — вне основного WFO-контура диплома, заготовка под бумажную торговлю.

## 3.10.7. Выводы по разделу

1. ИТС организована слоями с явным оркестратором и единым YAML-конфигом.
2. Прод-пайплайн объединяет все модели, meta-weighting, decision и risk в одном WFO-цикле.
3. Дерево проекта и схема потока (рис. 3.23–3.24) задают карту кода для сопровождения и ВКР.
4. GUI и execution — вспомогательные подсистемы поверх того же API и журналов экспериментов.

Комплексное тестирование и метрики — п. 3.11.
