# Команды проекта IST и проверка GUI

Справочник **всех основных команд** для обучения, валидации, данных и десктоп-GUI. Используйте его как чеклист: каждая строка в таблице «GUI» — что должно отразиться во вкладке после успешного CLI/API-прогона.

**Статус:** актуально для репозитория `D:/IST` (Windows / WSL). Пути — от корня репозитория.

**Связанные документы:**

- [`GUI_LAYOUT_REFERENCE.md`](GUI_LAYOUT_REFERENCE.md) — **описание окон, расположения, сценарии, empty-states, mermaid-потоки, UX-аномалии** (с cross-ref на §7 ниже)
- [`GUI_VERIFICATION_PLAN_WSL.md`](GUI_VERIFICATION_PLAN_WSL.md) — пошаговая проверка GUI + команды WSL
- [`GUI_USER_GUIDE.md`](GUI_USER_GUIDE.md) — краткое руководство по вкладкам
- [`gui/README.md`](../gui/README.md) — архитектура API и экранов
- [`THESIS_4MODEL_STEPS.md`](THESIS_4MODEL_STEPS.md) — дипломный контур по шагам
- [`TUNE_THESIS_SPEED.md`](TUNE_THESIS_SPEED.md) — профили `tune-thesis`

---

## 1. Подготовка окружения

### 1.1. Виртуальное окружение и зависимости

```bash
cd /mnt/d/IST          # Windows: cd D:\IST
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gui.txt   # только для GUI
export TF_CPP_MIN_LOG_LEVEL=2       # меньше шума TensorFlow
```

### 1.2. Быстрая проверка Python / GPU / библиотек

```bash
python scripts/thesis_4model_steps.py --step 0
```

или вручную:

```bash
python -c "import sys; print(sys.version)"
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPU', tf.config.list_physical_devices('GPU'))"
python -c "import lightgbm, xgboost, sklearn; print('OK')"
```

### 1.3. Минимальные данные для полного контура

| Файл | Назначение |
|------|------------|
| `data/ohlcv/BTC-USDT_1h.parquet` | BTC, WFO / tune / GUI |
| `data/ohlcv/ETH-USDT_1h.parquet` | ETH, второй символ |
| `data/features/BTC-USDT_1h.parquet` | после `build-features` |
| `artifacts/BTC-USDT_1h/` | bundle + `manifest.json` |
| `docs/backtest_journal/runs.jsonl` | журнал WFO |

Скачать OHLCV (OKX):

```bash
python -m data_layer -s BTC/USDT -t 1h --from 2022-01-01 --to 2026-01-01
python -m data_layer -s ETH/USDT -t 1h --from 2022-01-01 --to 2026-01-01
```

С MTF (15m, 4h):

```bash
python -m data_layer -s BTC/USDT -t 1h --mtf --from 2022-01-01
```

---

## 2. Главная точка входа: `python -m orchestration`

Список подкоманд:

```bash
python -m orchestration --help
```

### 2.1. Диагностика и конфиг

| Команда | Назначение | Типичный артефакт |
|---------|------------|-------------------|
| `smoke` | Синтетический WFO без внешних данных | stdout |
| `validate-config --config <yaml>` | Проверка секций YAML | exit 0 / список ERROR |

```bash
python -m orchestration smoke
python -m orchestration validate-config --config config/profiles/canonical_4model.yaml
python -m orchestration validate-config --config config.yaml
```

**GUI:** прямой кнопки нет; косвенно — **Конфигурация** (merged config) и отсутствие ошибок при Jobs/отчётах.

---

### 2.2. Данные и признаки

| Команда | Назначение |
|---------|------------|
| `build-features` | Канонический parquet признаков + manifest |
| `list-symbols` | JSON: пары, parquet, bundle, статус |
| `manifest-show` | `artifacts/<slug>/manifest.json` |

```bash
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h --force
python -m orchestration list-symbols
python -m orchestration manifest-show --symbol BTC/USDT --timeframe 1h
```

**GUI:**

| Вкладка | Что проверить после CLI |
|---------|-------------------------|
| **Задачи** | Чеклист: шаг «Признаки» ✓; `data_health.features_rows` > 0 |
| **Задачи → Признаки** | та же команда из GUI |
| **Модели** | после bundle — `feature_columns` в manifest |

---

### 2.3. WFO и отчёты (ядро диплома)

#### `report-real` — OHLCV → признаки → 4 модели → WFO → acceptance/target

```bash
# Базовый прогон (путь к parquet явно)
python -m orchestration report-real \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --symbol BTC/USDT \
  --timeframe 1h \
  --dl-epochs 3 \
  --json-out docs/thesis_report_BTC.json

# С семенем из config/symbols + кэш фичей
python -m orchestration report-real \
  --symbol BTC/USDT \
  --timeframe 1h \
  --use-tuning-best \
  --use-feature-cache \
  --max-rows 0 \
  --dl-epochs 3

# Явно все 4 модели (TensorFlow)
python -m orchestration report-real \
  --symbol BTC/USDT \
  --use-tuning-best \
  --use-feature-cache \
  --full-models
```

| Флаг | Смысл |
|------|--------|
| `--max-rows 0` | хвост из `orchestration_tuning_best.max_rows` или все строки |
| `--use-tuning-best` | параметры из `config/symbols/<slug>.yaml` |
| `--use-feature-cache` | `data/features/<slug>.parquet` |
| `--config` | YAML с `backtesting.acceptance` (по умолчанию `config.yaml`) |
| `--json-out` | JSON отчёт; без флага — запись в `docs/backtest_journal/` |

#### `from-parquet` — уже готовый feature-parquet

```bash
python -m orchestration from-parquet data/features/BTC-USDT_1h.parquet \
  --config config/profiles/canonical_4model.yaml \
  --full-models \
  --dl-epochs 3 \
  --json-out docs/from_parquet_test.json
```

`--raw-ohlcv` — если на входе сырой OHLCV (прогон FeatureManager).

**GUI:**

| Вкладка | Что проверить |
|---------|----------------|
| **Бэктесты** | новая строка в таблице; `acceptance_passed`; детали run |
| **Бэктесты** | equity по фолдам (если есть `fold_metrics`) |
| **Обзор** | toolbar: `acc:PASS/FAIL` по последнему журналу |
| **Задачи → Отчёт WFO** | лог CLI; после завершения — refresh бэктестов |
| **Конфигурация → Отчёт acceptance** | `docs/thesis_<slug>_acceptance_gui.json` |

---

### 2.4. Подбор гиперпараметров

#### `tune-thesis` — fast → refine → confirm

```bash
python -m orchestration tune-thesis --symbol BTC/USDT --timeframe 1h --phase all
python -m orchestration tune-thesis --symbol BTC/USDT --phase fast
python -m orchestration tune-thesis --symbol BTC/USDT --phase refine
python -m orchestration tune-thesis --symbol BTC/USDT --phase confirm

# Быстрый профиль (см. TUNE_THESIS_SPEED.md)
python -m orchestration tune-thesis --symbol BTC/USDT --profile turbo --phase all

# Без кэша фичей (медленнее)
python -m orchestration tune-thesis --symbol BTC/USDT --no-feature-cache

# Свой файл уровней
python -m orchestration tune-thesis --symbol BTC/USDT \
  --tuning-yaml config/profiles/thesis_tuning.yaml
```

По умолчанию в CLI: `--config config/profiles/canonical_4model.yaml`.

**Важно:** кнопка **«Тюнинг thesis»** в GUI собирает команду через `gui/api/cli_api.py` с `--config config/profiles/thesis_tuning.yaml` (отличие от чистого CLI). Для сравнения GUI и терминала используйте одинаковый `--config`.

**GUI:** **Задачи → Тюнинг thesis**; фаза из комбобокса; журнал `docs/backtest_journal/` с `tune_level`; **Бэктесты** — новые trials.

#### `tune-until` — legacy grid/refine до acceptance

```bash
python -m orchestration tune-until \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --max-trials 80 \
  --mode refine \
  --journal-root docs/backtest_journal

python -m orchestration tune-until --require-target --apply-best
```

**GUI:** отдельной кнопки нет (только CLI / скрипты).

---

### 2.5. Сквозной pipeline по символу

#### `prepare-symbol` — download → features → tune → holdout → train-final → manifest

```bash
python -m orchestration prepare-symbol BTC/USDT 1h --download
python -m orchestration prepare-symbol --symbol BTC/USDT --timeframe 1h \
  --config config/profiles/canonical_4model.yaml \
  --max-trials 30 \
  --holdout-fraction 0.2

# Без финального обучения
python -m orchestration prepare-symbol BTC/USDT 1h --skip-final

# Разрешить запись 2-model keys в YAML (не для диплома 4-model)
python -m orchestration prepare-symbol BTC/USDT 1h --allow-two-model
```

**GUI:**

| Вкладка | Что проверить |
|---------|----------------|
| **Задачи → Подготовить символ** | чекбокс «Скачать OHLCV» |
| **Задачи** | чеклист pipeline; лог `active/*.jsonl` |
| **Задачи** | combo «Лог задачи» + автообновление |
| **Модели / Обзор** | после успеха — bundle, explain |

#### `train-final-symbol` — принудительный финальный fit

```bash
python -m orchestration train-final-symbol --symbol BTC/USDT --timeframe 1h --force
```

**GUI:** **Задачи → Финальное обучение** (`--force` всегда).

---

### 2.6. Инференс и режим (без полного WFO)

```bash
python -m orchestration explain --symbol BTC/USDT --timeframe 1h --window 256
python -m orchestration regime-history --symbol BTC/USDT --timeframe 1h --step 24
python -m orchestration regime-history --symbol BTC/USDT --json-out docs/regime_BTC.json
```

**GUI:**

| Вкладка | API / источник |
|---------|----------------|
| **Обзор** | `inference.explain()` — сигнал, 4 prob, regime, why_blocked |
| **График** | `chart_payload()` / `regime_series()` — фон trend/range |
| **Модели** | веса из merged config + последний explain |

---

## 3. Слой данных: `python -m data_layer`

Эквивалент скачивания внутри `prepare-symbol --download`.

```bash
python -m data_layer --help
python -m data_layer -s BTC/USDT -t 1h --from 2022-01-01 --to 2026-01-01
python -m data_layer -s BTC/USDT -t 1h -o data/ohlcv/BTC-USDT_1h.parquet
python -m data_layer -s BTC/USDT -t 1h --format csv --output-dir data/ohlcv
python -m data_layer -s BTC/USDT -t 1h --mtf --mtf-timeframes 15m 4h
python -m data_layer -s BTC/USDT -t 1h -q
```

**GUI:** косвенно — **Задачи** (data health: строки OHLCV, last timestamp); скачивание через **Подготовить символ** + «Скачать OHLCV».

---

## 4. Скрипты `scripts/` (диплом и исследования)

| Скрипт | Команда | GUI |
|--------|---------|-----|
| Пошаговый контур | `python scripts/thesis_4model_steps.py --list` | — |
| | `python scripts/thesis_4model_steps.py --step 0` … `6` | — |
| | `python scripts/thesis_4model_steps.py --step all` | — |
| Push 4-model | `python scripts/thesis_push_4model.py --parquet data/ohlcv/BTC-USDT_1h.parquet --mode quality` | — |
| | `--mode fast` / `fast+quality` | — |
| | `--write-symbol-yaml` (только при PASS) | обновляет **Конфигурация** |
| Phase 2 | `python scripts/thesis_push_4model_phase2.py --parquet ... --write-symbol-yaml` | — |
| Ablation | `python scripts/run_ablation.py --max-rows 8000 --out docs/reports/ablation_canonical.json` | — |
| WFE compare | `python scripts/compare_wfe_reports.py baseline.json experiment.json` | — |
| WFE analyze | `python scripts/analyze_wfe.py` | — |
| Sharpe trials | `python scripts/sharpe_push_trials.py` | — |

Пример шага 3 из обёртки:

```bash
python scripts/thesis_4model_steps.py --step 3 --symbol BTC/USDT --max-rows 8000 --dl-epochs 3
```

---

## 5. Тесты (проверка логики без GUI)

```bash
# Быстрый набор (как в thesis_4model_steps --step 1)
pytest tests/test_orchestration.py tests/test_backtesting_criteria.py \
  tests/test_wfo_backtest.py -q --tb=line -m "not integration"

# Все unit-тесты репозитория (без integration)
pytest tests/ -m "not integration" -q

# Медленно: реальные модели на parquet
pytest tests/test_orchestration_real_models.py -m integration
pytest tests/test_orchestration_four_models.py -m integration

# Тюнинг / feature store
pytest tests/test_tuning_config.py tests/test_feature_store.py tests/test_thesis_tuning.py -q
pytest tests/test_symbol_pipeline.py -q
```

**GUI:** отдельного экрана нет; падение тестов → риск ошибок в **Обзор / Бэктесты / Задачи**.

---

## 6. GUI: запуск и режимы

### 6.1. Запуск приложения

```bash
# Полный режим (нужны parquet, bundle, журнал)
python -m gui.app

# Демо: синтетические данные, CLI отключён
python -m gui.app --demo
# или
set IST_GUI_DEMO=1
python -m gui.app
```

Windows:

```powershell
cd D:\IST
py -3 -m gui.app
py -3 -m gui.app --demo
```

Зависимости: `requirements-gui.txt`.

### 6.2. API без окна (smoke для всех вкладок)

```bash
python -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(c.symbols.list_symbols()[:3])"
python -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(c.inference.explain('BTC/USDT','1h'))"
python -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(len(c.backtests.list_runs(limit=5)))"
python -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(c.jobs.pipeline_status('BTC/USDT','1h'))"
python -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(c.bundles.bundle_info('BTC/USDT','1h'))"
```

Демо-API:

```bash
python -c "from gui.api import IstGuiClient; c=IstGuiClient(demo=True); print(c.inference.explain('BTC/USDT','1h').signal)"
```

### 6.3. Снимки для диплома (3.11)

```bash
python docs/scripts/capture_gui_screenshots.py
```

Требуется графическая сессия; выход: `docs/figures/3_11/gui/*.png`.

---

## 7. Матрица: CLI ↔ вкладка GUI ↔ проверка

Используйте как **чеклист приёмки GUI**. Колонка «Ожидание» — что должно быть видно после успешного прогона (не в `--demo`). Подробная карта экранов и сценарии S1–S5: [`GUI_LAYOUT_REFERENCE.md`](GUI_LAYOUT_REFERENCE.md).

| # | CLI / действие | Вкладка GUI | Кнопка / элемент | Ожидание после успеха |
|---|----------------|-------------|------------------|------------------------|
| 1 | `list-symbols` | Toolbar | Combo символ/TF | Список пар, статус OHLCV/feat/bundle |
| 2 | `data_layer` / `prepare --download` | Задачи | Data health | `ohlcv_rows` > 0, актуальный `last_ts` |
| 3 | `build-features` | Задачи | «Признаки» | features_rows > 0; шаг признаков ✓ |
| 4 | `tune-thesis` | Задачи | «Тюнинг thesis» | записи в журнале; shortlist/confirm в stdout |
| 5 | `train-final-symbol --force` | Задачи | «Финальное обучение» | `artifacts/<slug>/`, manifest обновлён |
| 6 | `report-real` + cache + tuning-best | Задачи | «Отчёт WFO» | новый run; PASS/FAIL в toolbar |
| 7 | `report-real` | Бэктесты | таблица | stage, sharpe, PF, WFE, acceptance |
| 8 | `report-real` | Бэктесты | детали run | fold_metrics, criteria checks |
| 9 | `report-real` | Бэктесты | equity chart | кривая по фолдам |
| 10 | `prepare-symbol` | Задачи | «Подготовить символ» | чеклист всех шагов; лог задачи |
| 11 | `explain` | Обзор | ExplainCard | direction, signal, 4×P(up), regime |
| 12 | `explain` | Обзор | why_blocked | текст при HOLD |
| 13 | `regime-history` / API `regime_series` | **Режим** + **График IST** | таблица режимов; фон trend/range на локальном графике |
| 14 | bundle + manifest | Модели | таблица | model_keys: lgb, gru, xgb, cnn |
| 15 | `merged_config` | Конфигурация | форма + JSON | пороги, margin, dl_epochs |
| 16 | save config | Конфигурация | «Сохранить» | `config/symbols/<slug>.yaml` |
| 17 | acceptance report CLI | Конфигурация | «Отчёт acceptance» | `docs/thesis_*_acceptance_gui.json` |
| 18 | Paper session | Исполнение | Connect / Step | позиции, ордера, шаг inference |
| 19 | `validate-config` | **Задачи** | Validate config | stdout в лог CLI |
| 20 | `smoke` | **Задачи** | Smoke | stdout в лог CLI |
| 21 | `tune-until` | **Задачи** | Tune-until | журнал + refresh |
| 22 | `from-parquet` | **Задачи** | From-parquet | выбор файла → WFO |
| 23 | `list-symbols` | **Задачи** / **Конфигурация** | List symbols / кнопка | JSON / текст |
| 24 | `manifest-show` | **Задачи** / **Конфигурация** | Manifest / вкладка | manifest JSON |
| 25 | `backtests.best_acceptance` | **Бэктесты** | Лучший PASS | выбор строки |
| 26 | `bundles` schema validate | **Модели** | Проверить схему | schema valid/invalid |

### 7.1. Исполнение (без отдельного CLI)

Вкладка **Исполнение** использует только Python API:

- `api.execution.create_paper_session()` → connect
- `PaperExecutionSession.run_step()` — один бар inference + paper order
- emergency stop, таймер автошага

Проверка: подключиться → 1–5 шагов → история ордеров; согласованность сигнала с **Обзор**.

### 7.2. Режим `--demo`

| Функция | demo | production |
|---------|------|------------|
| Обзор / График / Модели / Бэктесты | синтетика | реальные parquet + journal |
| Jobs CLI-кнопки | **отключены** | subprocess как в таблице |
| Конфигурация → acceptance | **отключён** | `report-real` subprocess |

Для полной приёмки GUI всегда запускайте **без** `--demo`.

---

## 8. Точные команды, которые строит GUI (`gui/api/cli_api.py`)

Для сверки лога в окне «CLI» на вкладке **Задачи**:

```text
# Подготовить символ (+ --download если отмечено)
python -m orchestration prepare-symbol <SYMBOL> <TF> --config config/profiles/canonical_4model.yaml --max-trials 30

# Признаки
python -m orchestration build-features --symbol <SYMBOL> --timeframe <TF> --config config/profiles/canonical_4model.yaml

# Тюнинг (фаза из UI: all | fast | refine | confirm)
python -m orchestration tune-thesis --symbol <SYMBOL> --timeframe <TF> --phase <PHASE> --config config/profiles/thesis_tuning.yaml

# Финальное обучение
python -m orchestration train-final-symbol --symbol <SYMBOL> --timeframe <TF> --force

# Отчёт WFO (флаги из чекбоксов Jobs)
python -m orchestration report-real --symbol <SYMBOL> --parquet data/ohlcv/<slug>.parquet \
  --config config.yaml --max-rows <N|0> [--use-tuning-best] [--use-feature-cache] [--full-models]
```

**Конфигурация → Отчёт acceptance:**

```text
python -m orchestration report-real ... --max-rows 0 --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_<slug>_acceptance_gui.json
```

---

## 9. Рекомендуемые сценарии end-to-end

### 9.1. Минимальная проверка GUI за 15–30 мин (BTC)

```bash
python -m orchestration validate-config --config config/profiles/canonical_4model.yaml
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h
python -m orchestration report-real --symbol BTC/USDT --use-tuning-best --use-feature-cache --max-rows 8000 --dl-epochs 3
python -m gui.app
```

В GUI: BTC/USDT 1h → **Обзор** → **Бэктесты** (последний run) → **Задачи** (health) → **Модели** (если есть bundle).

### 9.2. Полный дипломный контур (CLI, затем сверка GUI)

```bash
python scripts/thesis_4model_steps.py --step all
python -m gui.app
```

### 9.3. Полный контур только через orchestration

```bash
python -m orchestration prepare-symbol BTC/USDT 1h --download --max-trials 10
python -m orchestration tune-thesis --symbol BTC/USDT --phase confirm
python -m orchestration train-final-symbol --symbol BTC/USDT --force
python -m orchestration report-real --symbol BTC/USDT --use-tuning-best --use-feature-cache --max-rows 0
python -m orchestration explain --symbol BTC/USDT --timeframe 1h
```

---

## 10. Артефакты и пути (где смотреть результат)

| Путь | Содержимое |
|------|------------|
| `data/ohlcv/<slug>.parquet` | OHLCV |
| `data/features/<slug>.parquet` | признаки + `.manifest.json` |
| `config/symbols/<slug>.yaml` | overrides + `orchestration_tuning_best` |
| `config/reference/thesis_4model_reference.yaml` | эталон BTC |
| `artifacts/<slug>/manifest.json` | bundle, ready_for_paper |
| `artifacts/<slug>/<run_id>/` | обученные модели |
| `docs/backtest_journal/runs.jsonl` | все WFO/tune runs |
| `docs/backtest_journal/runs/<run_id>.json` | снимок одного run |
| `docs/thesis_*_acceptance_gui.json` | отчёт из GUI Settings |
| `docs/reports/` | ablation и прочие JSON |

---

## 11. Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `TF_CPP_MIN_LOG_LEVEL=2` | меньше логов TF |
| `IST_GUI_DEMO=1` | демо-GUI без CLI |
| `OKX_API_KEY`, `OKX_SECRET`, `OKX_PASSPHRASE` | live OKX (будущее; paper не требует) |
| `CUDA_VISIBLE_DEVICES=0` | выбор GPU в WSL |

---

## 12. Частые проблемы

| Симптом | Команда / решение |
|---------|-------------------|
| `n_folds=0` | увеличить `--max-rows` (≥8000 для 1h) |
| GUI CLI серые кнопки | запуск без `--demo` |
| Нет explain / bundle | `train-final-symbol` или `prepare-symbol` без `--skip-final` |
| TF GPU пустой | Python 3.11 + `tensorflow[and-cuda]`; `nvidia-smi` в WSL |
| Медленный tune | `TUNE_THESIS_SPEED.md`, `--profile turbo`, `build-features` + cache |
| Расхождение GUI vs терминал tune | выровнять `--config` (см. §7.1 и §8) |

---

## История

| Дата | Изменение |
|------|-----------|
| 2026-05-22 | Первая версия: полный справочник CLI + матрица проверки GUI |
