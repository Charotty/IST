# План реализации: ускорение и качество обучения IST (4 модели)

**Статус:** реализованы §2–§4 и §1 (базовый pipeline); Optuna — опционально (`requirements-tune.txt`). ETH/BTC acceptance на confirm — запускается вручную.

**Связанные документы:**

- [`THESIS_CONTEXT.md`](THESIS_CONTEXT.md) — цель, окружение, две задачи (скорость / качество)
- [`THESIS_4MODEL_STEPS.md`](THESIS_4MODEL_STEPS.md) — текущие CLI-шаги диплома
- [`BACKTESTING_CRITERIA_REFERENCE.md`](BACKTESTING_CRITERIA_REFERENCE.md) — пороги acceptance

**Уже есть в коде (частичная база, не дублировать при реализации):**

| Компонент | Файл | Что делает сейчас |
|-----------|------|-------------------|
| Пресеты fast/quality | `orchestration/tuning_config.py` | `THESIS_FAST_PRESET`, `THESIS_QUALITY_PRESET`, `merge_thesis_params` |
| Лимит фолдов WFO | `orchestration/orchestrator_config.py` → `max_wfo_folds` | Обрезка splits в `training_orchestrator.walk_forward_backtest` |
| Скрипты подбора | `scripts/thesis_push_4model.py`, `thesis_push_4model_phase2.py` | Ручной grid патчей, без Optuna |
| Фичи из OHLCV | `orchestration/real_data_benchmark.features_from_ohlcv_parquet` | FeatureEngine на каждый вызов |
| Canonical features path | `orchestration/canonical_pipeline.py` → `data/features/<slug>.parquet` | Сохранение при `prepare-symbol` / `build_canonical_features` |
| DL epochs | `orchestration/model_factory.py` → `_DLTrainAdapter` | `dl_epochs`, `batch_size=64` |
| Табличные модели | `models/tabular/lightgbm_tabular_model.py`, `models/mean_reversion/xgboost_model.py` | CPU, без GPU-параметров |

---

## Цели реализации (напоминание)

1. **Скорость:** сократить время одного trial WFO (целевой ориентир: с ~30 мин до ~5–10 мин на этапе подбора; финальный отчёт может оставаться длинным).
2. **Качество:** добиться `acceptance_passed: true` с **`model_keys: [lgb, gru, xgb, cnn]`** на BTC и перенос методики на ETH/другие пары.
3. **Без смены архитектуры диплома:** контракт пайплайна `features → regime → 4 модели → meta → decision → risk` сохраняется.

---

## Порядок внедрения (рекомендуемый)

Зависимости между блоками:

```text
[2] Кэш фичей ──┬──► [1] Многоуровневая оптимизация (быстрые trials)
                │
[3] Табличные GPU/потоки ──► ускоряет каждый фолд внутри trial
[4] GRU/CNN mixed precision ──► ускоряет каждый фолд внутри trial
                │
                └──► [1] Фаза quality: полный WFO + acceptance
```

| Этап | Блок | Приоритет | Зависимости |
|------|------|-----------|-------------|
| A | §2 Ускорение данных и фичей | P0 | — |
| B | §4 GRU/CNN (mixed precision, batch) | P1 | окружение TF+GPU |
| C | §3 Табличные модели | P1 | опционально CUDA-сборки LGB/XGB |
| D | §1 Многоуровневая оптимизация | P0 | §2 желателен до Optuna |
| E | Интеграция CLI + тесты + доки | P2 | A–D |

---

## §1. Многоуровневая оптимизация

### 1.1. Назначение

Заменить/дополнить ручной перебор в `thesis_push_4model.py` **формальной многостадийной схемой**:

- **Уровень 0 (fast):** мало фолдов, мало эпох DL, урезанный хвост данных — только отсев плохих гиперпараметров.
- **Уровень 1 (refine):** средний WFO на shortlist (top-K из уровня 0).
- **Уровень 2 (confirm):** полный WFO (`max_wfo_folds=0`, полный `max_rows`, `dl_epochs` из quality-профиля) — единственный результат для диплома и `config/symbols/*.yaml`.

### 1.2. Требования (функциональные)

| ID | Требование | Критерий приёмки |
|----|------------|------------------|
| M1 | Единая точка входа CLI | `python -m orchestration tune-thesis` или `scripts/thesis_tune_multilevel.py` с `--symbol`, `--phase fast\|refine\|confirm\|all` |
| M2 | Ранний стоп trial | Trial прерывается, если после ≥2 фолдов `mean_sharpe` ниже медианы завершённых trials (или интеграция **Optuna** `MedianPruner` / `SuccessiveHalvingPruner`) |
| M3 | Shortlist между уровнями | После fast сохраняется `docs/thesis_tune_shortlist.json` (top-K params, K=5 по умолчанию) |
| M4 | Random + локальный refine | Не только сетка 30 патчей: TPE или random 20 точек в окрестности `THESIS_QUALITY_PRESET` |
| M5 | Двухэтапный model_keys (опционально) | Флаг `--tabular-first`: уровни 0–1 только `lgb+xgb`; уровень 2 — полные 4 модели (ускорение без изменения финального claim) |
| M6 | Журнал | Каждый trial → `BacktestResultsJournal` с полями `tune_level`, `symbol`, `stage`, `elapsed_sec` |
| M7 | Финал только confirm | `acceptance_passed` для записи в YAML учитывается **только** с `tune_level=confirm` |
| M8 | Совместимость | `prepare-symbol --max-trials N` делегирует в ту же логику или документируется как legacy |

### 1.3. Требования (нефункциональные)

- Не ломать `run_single_trial` / `walk_forward_backtest` контракт утечек (purge, embargo).
- Параллельные trials на GPU: **не более одного** trial с GRU/CNN одновременно на 6 GB VRAM (очередь или `CUDA_VISIBLE_DEVICES`).
- CPU: допустимо `n_jobs` параллельных tabular-only trials без DL.

### 1.4. Предлагаемая реализация (модули)

| Компонент | Действие |
|-----------|----------|
| `orchestration/thesis_tuning.py` (новый) | Класс/функции: `run_fast_search`, `run_refine`, `run_confirm`, `should_prune_trial` |
| `orchestration/tuning_loop.py` | Опциональный callback `on_fold_done(fold_metrics)` для pruning; параметр `tune_level` в report |
| Зависимость | `optuna>=3.0` в `requirements.txt` (optional extra `requirements-tune.txt`) |
| `scripts/thesis_push_4model.py` | Становится thin-wrapper над `thesis_tuning` или deprecated с redirect |
| `orchestration/__main__.py` | Subcommand `tune-thesis` |
| `config/profiles/thesis_tuning.yaml` (новый) | Пороги: `fast_max_folds`, `fast_max_rows`, `refine_max_rows`, `top_k`, pruner type |

### 1.5. Параметры уровней (дефолты для спеки)

```yaml
# thesis_tuning.yaml (черновик, не создан)
levels:
  fast:
    max_rows: 6000
    max_wfo_folds: 8
    walk_forward_step: 500
    dl_epochs: 2
    n_trials: 24
    pruner: median
  refine:
    max_rows: 8000
    max_wfo_folds: 12
    walk_forward_step: 400
    dl_epochs: 2
    n_trials: 12
    seed_from: fast_shortlist
  confirm:
    max_rows: 8000
    max_wfo_folds: 0
    walk_forward_step: 360
    dl_epochs: 3
    n_trials: 3
    seed_from: refine_best
```

### 1.6. Риски

| Риск | Митигация |
|------|-----------|
| Pruning отсекает trial, который хорош на полном WFO | Confirm всегда без pruning; fast только для отсева |
| Optuna + TF memory leak | `clear_session()` Keras между trials |
| Переобучение под fast | Явная маркировка в журнале `tune_level`; диплом цитирует только `confirm` |

### 1.7. Тесты (при реализации)

- Unit: shortlist top-K из mock reports.
- Integration (малый parquet, 1 trial): fast завершается < N минут, journal содержит `tune_level`.
- Smoke: confirm с `max_wfo_folds=2` на demo parquet — форма отчёта без падения.

---

## §2. Ускорение данных и фичей

### 2.1. Назначение

Исключить повторное построение признаков и MTF при каждом trial и при каждом запуске `report-real` / `tune_until`.

### 2.2. Текущее состояние

| Путь | Поведение |
|------|-----------|
| `features_from_ohlcv_parquet` | `read_parquet(OHLCV)` → `FeatureEngine.add_indicators()` каждый раз |
| `build_features` + `save_to` | Сохранение в `data/features/<slug>.parquet` только при явном `prepare-symbol` / canonical |
| `run_single_trial` | Принимает уже готовый `DataFrame` — кэш на уровне скрипта один раз за процесс |
| WFO внутри trial | На каждом фолде заново `fit` всех моделей (кэша моделей нет) |

### 2.3. Требования (функциональные)

| ID | Требование | Критерий приёмки |
|----|------------|------------------|
| D1 | Feature store по символу | `data/features/<slug>.parquet` с manifest: `built_at`, `config_hash`, `row_count`, `columns_hash` |
| D2 | CLI build-features | `python -m orchestration build-features --symbol BTC/USDT --timeframe 1h [--force]` |
| D3 | Загрузка в tuning | `thesis_tuning` и `report-real` при `--use-feature-cache` читают features parquet; при отсутствии — fallback build с записью |
| D4 | Инвалидация кэша | Если изменился `config/profiles/canonical_4model.yaml` (hash секций `feature_engineering` + `synchronization`) — предупреждение и пересборка |
| D5 | Хвост без пересчёта | `max_rows` = slice после load: `df.iloc[-max_rows:]` |
| D6 | MTF один раз | Canonical path: `MultiTimeframeEngine` + `FeatureManager` только в `build-features`, не в trial |
| D7 | Optional memory map | Большие parquet: `pd.read_parquet(columns=...)` только нужные колонки для tabular/DL |

### 2.4. Предлагаемая реализация (модули)

| Компонент | Действие |
|-----------|----------|
| `orchestration/feature_store.py` (новый) | `build_if_needed`, `load_features`, `manifest_path`, `is_stale` |
| `orchestration/real_data_benchmark.py` | `features_from_ohlcv_parquet` → делегат в feature_store |
| `orchestration/canonical_pipeline.py` | Единый вызов `feature_store.build` |
| `data/features/<slug>.manifest.json` | Метаданные |
| `orchestration/__main__.py` | `build-features` |

### 2.5. Требования (инфраструктура)

- Рекомендация в доке: проект на `~/IST` в WSL, не `/mnt/d/`, для I/O.
- Размер features parquet: оценка ~50–200 MB на символ 1h — приемлемо.

### 2.6. Вне scope (явно не в первой итерации)

- Инкрементальное обновление фичей при догрузке OHLCV (только full rebuild).
- Feature store в Redis/PostgreSQL.

### 2.7. Тесты

- Build + load даёт тот же `shape` и те же колонки, что inline `FeatureEngine`.
- Stale detection при смене hash конфига.

---

## §3. Ускорение табличных моделей (LGB, XGB)

### 3.1. Назначение

Ускорить **повторяющееся** обучение LGB/XGB на каждом WFO-фолде (основная CPU-нагрузка наряду с DL).

### 3.2. Текущее состояние

```python
# LightGBMTabularModel — LGBMClassifier без device, num_threads, max_bin
# XGBoostMeanReversionModel — XGBClassifier без tree_method, device
```

Обучение: полный train-фолд, все feature columns, каждый fold с нуля.

### 3.3. Требования (функциональные)

| ID | Требование | Критерий приёмки |
|----|------------|------------------|
| T1 | Профиль ускорения в YAML | `config/profiles/thesis_tuning.yaml` → секция `tabular_accel` |
| T2 | LightGBM threads | `num_threads` = min(физические ядра, 16); документировать в README |
| T3 | LightGBM GPU (опционально) | Если сборка LGB с GPU: `device_type=gpu`, `max_bin=63`; fallback CPU при ImportError/RuntimeError |
| T4 | XGBoost GPU | `device=cuda`, `tree_method=hist`; fallback CPU; проверка на WSL + `nvidia-smi` |
| T5 | Меньше деревьев на fast-level | `n_estimators_fast: 100` vs `n_estimators_confirm: 200` — только при `tune_level=fast` |
| T6 | Единый фабричный путь | `model_factory.build_orchestration_models(..., tabular_profile="fast"\|"confirm")` |
| T7 | Optuna pruning для boosters | Callback `LightGBMPruningCallback` / `XGBoostPruningCallback` на уровне fast (если §1 внедрён) |
| T8 | Не менять контракт predict | `predict(DataFrame) -> ndarray` без изменений для orchestrator |

### 3.4. Параметры (черновик спеки)

```yaml
tabular_accel:
  lightgbm:
    num_threads: -1          # -1 → os.cpu_count() capped
    max_bin: 63              # для GPU; CPU может 127/255
    device_type: cpu         # gpu если доступно
    n_estimators_fast: 100
    n_estimators_confirm: 200
  xgboost:
    tree_method: hist
    device: cuda             # cpu fallback
    n_estimators_fast: 100
    n_estimators_confirm: 200
```

### 3.5. Предлагаемая реализация (модули)

| Файл | Изменение |
|------|-----------|
| `models/tabular/lightgbm_tabular_model.py` | Конструктор принимает `**lgb_params` или `TabularTrainingProfile` dataclass |
| `models/mean_reversion/xgboost_model.py` | Аналогично `device`, `tree_method` |
| `orchestration/model_factory.py` | Проброс `tabular_profile`, `tune_level` |
| `orchestration/thesis_tuning.py` | Передаёт profile по уровню |
| `docs/THESIS_CONTEXT.md` | Таблица «когда GPU tabular выгоден» |

### 3.6. Риски

| Риск | Митигация |
|------|-----------|
| LGB GPU медленнее CPU на ~8k строк | Авто-benchmark при старте `build-features` или флаг `--benchmark-tabular` |
| Разные предсказания CPU vs GPU | Confirm на CPU если расхождение AUC > ε |
| XGB CUDA OOM | `device=cpu` fallback |

### 3.7. Тесты

- Mock/fit на 1k строк: GPU path не падает или graceful fallback.
- Один WFO fold: время fit LGB+XGB логируется в journal metadata.

---

## §4. Ускорение GRU / CNN (TensorFlow)

### 4.1. Назначение

Сократить время **переобучения Keras-моделей на каждом фолде** при сохранении 4-model контура.

### 4.2. Текущее состояние

| Параметр | Значение |
|----------|----------|
| `GRUTrendModel` / `CNNVolatilityModel` | `Sequential`, Adam, binary CE |
| `train()` | `epochs` из `_DLTrainAdapter`, `batch_size=64`, `EarlyStopping(patience=5)` |
| Precision | float32 по умолчанию Keras |
| GPU | Используется если TF видит GPU; mixed precision не включён |
| Sequences | Строятся заново каждый `fit` |

### 4.3. Требования (функциональные)

| ID | Требование | Критерий приёмки |
|----|------------|------------------|
| G1 | Mixed precision policy | Глобально или на уровне train: `tf.keras.mixed_precision.set_global_policy('mixed_float16')` для `tune_level in (fast, refine)`; `float32` для `confirm` (опционально тоже mixed) |
| G2 | LossScale | Output layer / loss совместим с policy (sigmoid + float32 head при необходимости) |
| G3 | Настраиваемый batch_size | `dl_batch_size` в `thesis_tuning.yaml`: fast=128, confirm=64 (подбор под 6 GB) |
| G4 | Epochs по уровню | Уже через `dl_epochs`; зафиксировать в §1 levels |
| G5 | Кэш sequences (опционально v2) | Для одного fold: `prepare_sequences` один раз, если несколько эпох — уже так; между trials не кэшировать |
| G6 | `tf.function` predict (опционально) | Ускорение batch predict в `run_pipeline` — только если профилирование покажет bottleneck |
| G7 | Очистка VRAM | `tf.keras.backend.clear_session()` между trials в multilevel tune |
| G8 | XLA (опционально) | `TF_XLA_FLAGS` / `jit_compile=True` в fit — за флагом, т.к. может ломать debug |
| G9 | GRU/CNN arch freeze на tune | Не менять число units без ADR; только training policy |

### 4.4. Ограничения железа (GTX 1660 Ti)

- Compute capability **7.5** — Tensor Cores есть, mixed precision релевантен ([TF Mixed Precision](https://www.tensorflow.org/guide/mixed_precision)).
- **6 GB VRAM:** при OOM — уменьшить `batch_size`, `dl_epochs`, или `max_wfo_folds` на fast.
- Один активный Keras train на GPU.

### 4.5. Предлагаемая реализация (модули)

| Файл | Изменение |
|------|-----------|
| `orchestration/dl_training.py` (новый) | `setup_mixed_precision(enable: bool)`, `clear_tf_session()`, `fit_dl_model(model, ..., profile)` |
| `models/trend/gru_model.py` | Вызов setup из dl_training; опционально `dtype` в Input |
| `models/volatility/cnn_model.py` | Аналогично |
| `orchestration/model_factory.py` | `dl_batch_size`, `mixed_precision` из profile |
| `orchestration/thesis_tuning.py` | Профиль по уровню |

### 4.6. Требования (качество)

- После включения mixed precision: сравнить `mean_sharpe` на confirm BTC с float32 baseline (допуск ±10% relative или ручной sign-off в журнале).
- Не снижать `acceptance` только ради скорости на confirm-уровне.

### 4.7. Тесты

- GPU smoke: один fit GRU 100 sequences, 2 epochs, mixed precision on — без NaN loss.
- `test_tuning_config.py` расширить metadata profile.

---

## Сводная матрица: что ускоряет trial

| Механизм | Блок | Оценка вклада в trial ~30 min | Влияние на качество метрик |
|----------|------|------------------------------|----------------------------|
| `max_wfo_folds=8` | §1 + уже есть | −50…60% | Только на fast/refine |
| `dl_epochs: 5→2` | §1, §4 | −20…30% DL часть | Только fast/refine |
| Feature cache | §2 | −1…3 мин на trial (I/O+FE) | Нет |
| XGB GPU | §3 | −10…40% tabular часть | Минимальное при hist |
| LGB num_threads | §3 | −10…20% CPU | Нет |
| Mixed precision GRU/CNN | §4 | −15…35% DL | Проверить на confirm |
| Optuna prune | §1 | −30…70% числа trials | Нет на confirm |
| Tabular-first levels 0–1 | §1 | −40% если без DL | N/A на confirm |

---

## Конфигурация и CLI (целевое состояние после всех §)

```bash
# 1) Один раз на символ
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h

# 2) Многоуровневый tune
python -m orchestration tune-thesis --symbol BTC/USDT --phase all

# 3) Финальный отчёт
python -m orchestration report-real \
  --symbol BTC/USDT --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_btc_4model_acceptance.json
```

---

## Критерии готовности всего плана (Definition of Done)

- [ ] §2: повторный `tune-thesis` не вызывает `FeatureEngine` на OHLCV при валидном кэше.
- [ ] §3: логи trial содержат `tabular_device` (cpu/cuda) и время fit tabular.
- [ ] §4: mixed precision включается по флагу; confirm-профиль воспроизводим.
- [ ] §1: pipeline fast → refine → confirm с journal `tune_level`.
- [ ] BTC: `acceptance_passed: true`, 4 `model_keys`, `n_folds >= 5` на confirm.
- [ ] ETH: тот же pipeline, отдельный `config/symbols/ETH-USDT_1h.yaml`.
- [ ] Документация: обновлены `THESIS_4MODEL_STEPS.md`, `THESIS_CONTEXT.md` (ссылка на этот план).
- [ ] Тесты: минимум unit + один integration smoke на demo/small parquet.

---

## Явно не входит в этот план

- Переписывание `TrainingOrchestrator` / замена WFO на CPCV.
- Снижение порогов `backtesting.acceptance` в `config.yaml`.
- Обучение только 2 моделей как основной claim диплома.
- Распределённый Ray/Dask cluster.
- Автоторговля / live execution.

---

## История документа

| Дата | Изменение |
|------|-----------|
| 2026-05-22 | Первая версия: спецификация §1–§4, без реализации в коде |

---

*При начале разработки: реализовывать по этапам A→B→C→D из раздела «Порядок внедрения», открывать отдельные PR/коммиты по § с обновлением чеклиста DoD.*
