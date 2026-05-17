# Справочник замечаний по аудиту (engineering + алгоритмы)

Файл для быстрого обращения при разборе проблем. Статусы ниже можно обновлять по мере фиксов (например: `FIXED`, `PARTIAL`, `OPEN`).

---

## Критично для продакшена (алгоритмы)

### A1 — `RegimeDetector` не возвращает `regime_pred`

- **Что**: `TrainingOrchestrator` / `InferenceOrchestrator` берут `regime_info.get('regime_pred', zeros)`, а реальный `get_regime_info()` не задаёт массив `regime_pred` той же длины, что признаки.
- **Эффект**: адаптивные веса ensemble фактически не переключаются по режиму (нули или fallback).
- **Где**: `models/regime/regime_detector.py`, потребители в `orchestration/training_orchestrator.py`, `orchestration/inference_orchestrator.py`.
- **Статус**: **FIXED** — `get_regime_info()` возвращает `regime_pred` длины `len(df)` (баровые метки из `predict`), поправлен разбор `predict_proba` через NumPy.

---

### A2 — `safe_mode=True` без `train_threshold` / без `window` не гарантирует порог без look-ahead

- **Что**: в `compute_safe_threshold` без `train_threshold`, без `window` и с `expanding=False` возможен fallback на статистику по **всему** переданному ряду (в т.ч. test или весь исторический блок).
- **Эффект**: завышение метрик в бэктесте, расхождение с live.
- **Где**: `utils/data_leakage_prevention.py`, потребление в `decision/signal_rules.py`, дефолты в `decision/decision_pipeline.py`.
- **Статус**: **FIXED** — небезопасная ветка «глобальная медиана по ряду» удалена; без `train_threshold` требуется `window` или `expanding=True`. В `signal_rules` по умолию подставляется rolling `window=100`; в `DecisionPipeline` дефолт `threshold_window: 100`. Для калибровки по train-only по-прежнему `compute_train_threshold` + `train_threshold`.

---

### A3 — `SignalAssembler` обходит safe-threshold API

- **Что**: `_calculate_meta_threshold` всё ещё использует `np.median` и перцентили по переданному массиву целиком.
- **Эффект**: любой путь через `SignalAssembler` подвержен тем же ошибкам, что были в старом коде порогов.
- **Где**: `meta_learning/signal_assembler.py`.
- **Статус**: **FIXED** — при `meta_threshold_safe=True` используется `compute_safe_threshold` (или явный `train_threshold` / rolling / expanding; при `False` сохранён research-режим с глобальной статистикой).

---

### A4 — Orchestrator не связан с `DecisionPipeline` / контрактом integrated signal

- **Что**: в orchestrator порогование direction идёт от `meta_probabilities` напрямую (`direction_threshold` / `signal_threshold`), без сборки через `DecisionPipeline` и без явного разделения `meta_prob` vs `meta_mgmt_prob` как в README.
- **Эффект**: расхождение документации и прод-поведения; сложнее согласовать WFO и live.
- **Где**: `orchestration/training_orchestrator.py`, `orchestration/inference_orchestrator.py`, сравнить с `decision/decision_pipeline.py`.
- **Статус**: **FIXED** — при `OrchestratorConfig.apply_decision_pipeline=True` (по умолчанию) оркестраторы собирают integrated-сигнал через `DecisionPipeline`; инференс подставляет `train_threshold` из `config.meta_threshold` для bar-level гейта; отключение: `apply_decision_pipeline=False`.

---

### A5 — Meta-labels и `direction_prob` на одном баре

- **Что**: `create_safe_meta_labels` может использовать колонку `direction_prob` с тем же временным индексом без гарантии OOS-предикта direction-модели.
- **Эффект**: переобучение и циркулярная связь между direction и meta.
- **Где**: `utils/data_leakage_prevention.py`, `models/meta/logistic_filter.py`.
- **Статус**: **PARTIAL** — по умолчанию `direction_prob` сдвигается на 1 бар (`direction_prob_same_bar_allowed=False`); поддерживается колонка `direction_prob_oos`. Для строгого прод-пайплайна по-прежнему рекомендуются nested folds / отдельный этап OOS-предиктов.

---

### A6 — WFO в orchestrator: порог мета-слоя только на train

- **Что**: на фолде обучаются модели на train; на test считаются сигналы. Не зафиксировано обязательное правило «порог (median/mean) считать только на train и применять к test».
- **Эффект**: риск утечки порога, если где-то по цепочке передаётся полный ряд без разбиения.
- **Где**: `orchestration/training_orchestrator.py`.
- **Статус**: **FIXED** — в `walk_forward_optimization` и `train_test_split` после прогона train вызывается `compute_train_threshold` по `meta_probabilities` train (или фиксированный порог при `meta_threshold_mode=fixed`), затем для test в `DecisionPipeline` передаётся `train_threshold_override`.

---

## Инженерия

### E1 — Два входа в инференс

- **Что**: `InferenceOrchestrator` (все модели → meta weights) параллельно старый `InferenceEngine` + `ModelRouter` (одна активная модель).
- **Эффект**: риск подключить не тот entry point в проде или в экспериментах.
- **Где**: `orchestration/`, `models/inference/inference_engine.py`.
- **Статус**: **FIXED** (документация) — в модульных docstring указано: канонический multi-model путь — `InferenceOrchestrator`; legacy — `InferenceEngine` + router; не смешивать без адаптера.

---

### E2 — Бэктест не совпадает с risk-слоем

- **Что**: `Backtester` не умножает на `final_pos_size`, не воспроизводит trailing stop из `RiskPipeline` в том же контракте.
- **Эффект**: метрики WFO ≠ ожидания после risk.
- **Где**: `backtesting/backtester.py`, `risk_management/risk_pipeline.py`.
- **Статус**: **PARTIAL** — `Backtester.run(..., position_size=...)` принимает ряд множителей позиции (как `final_pos_size`), издержки при смене **эффективной экспозиции** (`signal * position_size`). Полный parity с ATR trailing + всеми шагами `RiskPipeline` остаётся интеграционной задачей (см. `apply_pipeline` в risk).

---

### E3 — Два класса `PositionSizer`

- **Что**: правила sizing в `models/sizing/` и в `risk_management/position_sizer.py` семантически разные (доля/уверенность vs ATR-risk units).
- **Эффект**: несостыковка inference vs backtest/live.
- **Статус**: **FIXED** (документация) — в `models/sizing` указано `ConfidencePositionSizer` vs ATR `risk_management/PositionSizer`; кросс-ссылка в `risk_management/position_sizer.py`.

---

### E4 — Тесты orchestration на mock

- **Что**: `tests/test_orchestration.py` использует заглушки; не проверяет реальные формы предиктов и отсутствие `regime_pred` у RegimeDetector.
- **Эффект**: регрессии типа «веса режима всегда нулевые» не ловятся.
- **Статус**: **PARTIAL** — добавлены `_meta_matching_config` (веса совпадают с `model_keys`), тест на длину `regime_pred` с `RegimeDetector` (pytest `importorskip("lightgbm")`). Без LightGBM тест пропускается.

---

### E5 — `create_safe_meta_labels` и цепочка `pct_change` после безопасного shift

- **Что**: стоит отдельно проверять экономический смысл доходности (горизонт, выравнивание индексов).
- **Где**: `utils/data_leakage_prevention.py` (`create_safe_meta_labels`).
- **Статус**: **FIXED** — доходность: `pct_change(horizon)` на `get_safe_horizon_shift(close, horizon)`; unit-тест `tests/test_audit_fixes.py` (лаг `direction_prob`).

---

### E6 — `ModelRegistry` без сохранения весов

- **Что**: в основном JSON-метаданные; воспроизводимость артефактов моделей ограничена.
- **Где**: `models/registry/model_registry.py`.
- **Статус**: **PARTIAL** — у `register_model` добавлен опциональный аргумент `artifact_manifest` (сид, гиперпараметры, отпечатки данных, пути); pickle весов по-прежнему через `save_weights`.

---

### E7 — `rank_ensemble` в мета-слое (хрупкость для 1D)

- **Что**: при включении rank-ensemble возможны ошибки на одномерных массивах.
- **Где**: `meta_learning/ensemble.py`.
- **Статус**: **FIXED** — `_to_ranks` поддерживает 1D и 2D батчи.

---

### E8 — Тесты окружения: pandas `freq='H'` vs `'h'`

- **Что**: в некоторых фикстурах `freq='H'` даёт ошибку на актуальных pandas.
- **Где**: `tests/test_orchestration.py`.
- **Статус**: **FIXED** — `freq='h'`.

---

### E9 — `OrchestratorConfig` и сокращённый `model_keys`

- **Что**: если задать `model_keys=['lgb','xgb']`, дефолтные веса могут содержать `gru`, `cnn` → рассогласование в `__post_init__`.
- **Где**: `orchestration/orchestrator_config.py`, `tests/test_orchestration.py`.
- **Статус**: **FIXED** — дефолтные веса режимов нормализуются через подмножество ключей `_renormalize_weight_subset`; тест проверяет ключи весов.

---

## Связь с журналом тестов

Полный вывод прогонов pytest и скриптов сохранять в **`docs/TEST_RESULTS_LOG.md`**, а имена прогона и решения по пунктам этого файла здесь можно кратко ссылками добавлять (например: «прогон #2 — исправлен E8»).

**Прогон (локально)**: `tests/test_orchestration.py` + `tests/test_audit_fixes.py` — 25 passed, 1 skipped (RegimeDetector без lightgbm).

---

## Версионирование

- **Последнее обновление содержания**: 2026-05-16 — статусы синхронизированы с кодовой базой по запросу пользователя.
- При существенном исправлении пункта: обновите **Статус** и при желании добавьте строку **Исправлено в коммите: …**.
