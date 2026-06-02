# Что ещё требует интеграции / связи для полной проверки и работы

Файл-справочник: разрывы между слоями и статус закрытия. **Обновлено:** после внедрения фабрики моделей, bundle-артефактов, валидации YAML, glue CLI и E2E-тестов.

**Легенда:** ✅ закрыто / есть рабочий путь в коде · ⚠️ частично · ⬜ открыто / вне текущего объёма

---

## Сводка по разделам

| № | Тема | Статус |
|---|------|--------|
| 1 | Мёртвые импорты WFO | ✅ `walk_forward.py` → оркестратор; старые функции — `NotImplementedError` |
| 2 | Сквозная сборка | ✅ `orchestration/glue.py`, CLI `python -m orchestration from-parquet` |
| 3 | Backtester ↔ RiskPipeline | ✅ `OrchestratorRiskBridge`, `TrainingResult.position_sizes`, WFO-бэктест |
| 4 | Фабрика моделей | ✅ `orchestration/model_factory.py` (`lgb`, `gru`, `xgb`, `cnn`) |
| 5 | YAML ↔ код | ✅ `OrchestratorConfig.from_yaml` игнор. неизв. ключи; `validate_pipeline_config`; секция `feature_engineering` в `config.yaml` |
| 6 | Артефакты / bundle | ✅ `orchestration/artifact_bundle.py` + сверка схемы фич |
| 7 | Execution loop | ⚠️ `execution/paper_loop.run_inference_execution_step`; полный scheduler — вне репо |
| 8 | Данные / микроструктура | ⚠️ live L2 по-прежнему `NotImplemented` (сообщение указывает на adapter); MTF/OKX — не одна команда |
| 9 | E2E / pytest | ✅ `tests/test_integration_gaps_closure.py` |
| 10 | Прочее | ✅ `gui/` PyQt6; `rl_layer` overlay backtest; ⚠️ `SignalAssembler` parallel к `DecisionPipeline` |

---

## 1. Обрывы в коде («нельзя просто запустить»)

| Обрыв | Суть | Статус |
|--------|------|--------|
| **`backtesting/walk_forward.py`** | Импорты `models.direction_model` и `models.dl_model_builder` | ✅ Удалены; `run_orchestrator_walk_forward_backtest`, `TrainingOrchestrator.walk_forward_backtest` |
| **`models/__init__.py`** | Два контура inference | ⚠️ Докстринг + `InferenceEngine.initialize` → `DeprecationWarning` (legacy); прод: **`InferenceOrchestrator`** |

---

## 2. Нет единой сквозной сборки (data → сигнал → бэктест / ордер)

Реализовано:

- **`orchestration/glue.py`**: `load_feature_table_from_parquet`, `run_wfo_backtest_from_parquet` (`light_only` / полные `model_keys`, override конфига для тестов).
- **CLI:** `python -m orchestration smoke` · `validate-config` · `from-parquet <path> [--raw-ohlcv] [--full-models]`.

Полная цепочка OKX → storage → MTF по-прежнему собирается из модулей `data_layer` / `synchronization` (см. §8).

---

## 3. Бэктест и риск не согласованы с оркестратором

✅ `RiskPipeline` через **`OrchestratorRiskBridge.calculate_position_sizes`**, тот же контракт, что у mock risk в тестах; **`Backtester.run(..., position_size=...)`**; ATR trailing совместим с pandas CoW (`atr_trailing_stop.py`).

---

## 4. Оркестратор ↔ реальные модели

✅ **`orchestration/model_factory.py`**: `build_orchestration_models`, `infer_training_feature_columns`, `meta_weighting_from_config` (ленивые импорты TF только для `gru`/`cnn`).

⚠️ **`DynamicMetaWeighting`** использует только **trend/range** (0/1); ключи **`breakout_weights`** в YAML задаются, но в весе по режиму не участвуют — нужно расширение meta-слоя, если нужен третий режим.

⚠️ **`meta_learning/ensemble.py`**: исторические имена `lstm`/`trans`; в прод-конфиге — **`gru`/`cnn`** (комментарий в модуле).

---

## 5. Конфиг YAML и код

✅ **`OrchestratorConfig.from_yaml` / `from_dict`**: неизвестные ключи отбрасываются с предупреждением (в т.ч. `safe_threshold_mode`, `threshold_window`, … — до появления полей в dataclass или явного маппинга).

✅ **`feature_engineering/config.py`**: `FeatureEngineeringConfig.from_yaml` читает только секцию `feature_engineering`, при отсутствии — `{}`.

✅ **`orchestration/config_validate.py`**: `validate_pipeline_config`, `raise_if_invalid` для старта пайплайна.

---

## 6. Persistence и prod

✅ **`orchestration/artifact_bundle.py`**: `save_orchestrator_bundle` / `load_orchestrator_bundle`, `validate_bundle_feature_schema`, `feature_schema_hash` в manifest.

Загрузка в inference: **`orchestration/glue.inference_stack_from_bundle`** → заполнить **`InferenceOrchestrator.initialize`** (при необходимости выставить `train_meta_threshold` в `OrchestratorConfig` / `DecisionPipeline` вручную).

---

## 7. Execution — ручной мост

⚠️ **`execution/paper_loop.py`**: один шаг `predict` → `execute_signal`. Внешний цикл по барам — у движка данных/биржи.

---

## 8. Данные и микроструктура

⚠️ **`FeatureManager`**: `microstructure.mode == "live"` — `NotImplementedError` с отсылкой к L2 adapter и документу.

⚠️ **OKX → parquet → MTF → features** — не объединены в одну команду; используйте CLI слоёв данных и `glue` для финального parquet с признаками.

---

## 9. Тесты и проверка

✅ **`tests/test_integration_gaps_closure.py`**: валидация `config.yaml`, roundtrip bundle, фабрика, **parquet → WFO → метрики**, deprecation `InferenceEngine`.

✅ Существующие: `tests/test_orchestration.py`, `tests/test_wfo_backtest.py`.

---

## 10. Прочее (архитектурные «дыры»)

| Пункт | Статус |
|--------|--------|
| **`gui/`** | ✅ PyQt6 (`gui/app/`), API (`gui/api/`), paper + reconcile + CLI jobs |
| **`rl_layer/`** | ✅ интеграция с оркестратором: `rl_layer/integration.py`, `run_rl_overlay_backtest` + общий `Backtester` |
| **`meta_learning/signal_assembler.py`** | ⚠️ В модульном докстринге: канон **`DecisionPipeline`**; assembler — параллельный путь порогов |

---

## Связанные документы

| Файл | Назначение |
|------|------------|
| `docs/AUDIT_ISSUES_REFERENCE.md` | Замечания аудита (утечки, regime_pred, пороги и т.д.) |
| `docs/TEST_RESULTS_LOG.md` | Полный вывод pytest и скриптов для анализа |

---

## Приоритет (остаточный)

1. ⬜ Расширить **DynamicMetaWeighting** до `breakout` / маппинг `regime_pred` → три режима при необходимости.
2. ⬜ Явный маппинг полей YAML (`threshold_window` → `threshold_rolling_window` и т.д.) или добавление полей в **`OrchestratorConfig`**.
3. ⬜ Live **L2** + join к барам; единая CLI-команда data→features при необходимости.
4. ⬜ Консолидация **`SignalAssembler`** vs **`DecisionPipeline`** (один источник правил порогов).

---

*Документ обновлён: фиксирует статус интеграции после закрытия блоков фабрики, артефактов, валидации конфига, glue и E2E.*
