# План самопроверки: per-symbol пайплайн и полный набор моделей

Цель документа — **не нагружая ассистента**, локально проверить весь IST: от журнала метрик до **GRU/CNN** через существующий CLI.

Окружение: **Windows**, репозиторий `d:\IST` (при необходимости поправьте путь).

---

## 1. Подготовка окружения (каждый сеанс PowerShell)

```powershell
Set-Location D:\IST
$env:PYTHONPATH = "D:\IST"
py -3 -c "import sys; print('python', sys.executable)"
```

Проверка TensorFlow **до** долгих прогонов (импорт может занять минуты при первой загрузке):

```powershell
py -3 -c "import tensorflow as tf; print('TF', tf.__version__)"
```

Если здесь ошибка — команды с `--full-models` из раздела 6 не использовать, пока не установите совместимый TensorFlow под ваш Python.

---

## 2. Быстрый регресс (без сети, без TF-тяжести)

```powershell
Set-Location D:\IST
$env:PYTHONPATH = "D:\IST"
py -3 -m pytest tests/test_symbol_pipeline.py tests/test_performance_metrics.py tests/test_backtesting_criteria.py tests/test_orchestration_real_models.py tests/test_integration_gaps_closure.py tests/test_wfo_backtest.py -q
```

Минимум только по новому контуру:

```powershell
py -3 -m pytest tests/test_symbol_pipeline.py -q
```

Встроенный дым оркестратора (без parquet):

```powershell
py -3 -m orchestration smoke
```

Валидация `config.yaml`:

```powershell
py -3 -m orchestration validate-config --config config.yaml
```

---

## 3. Per-symbol контур (LGB+XGB, без загрузки с биржи)

Ниже команды считают, что уже есть parquet, например `data\ohlcv\BTC-USDT_1h.parquet`.

### Список известных пар (JSON для будущего GUI)

```powershell
py -3 -m orchestration list-symbols
```

### Полный конвейер: признаки → локальный tune → holdout → manifest  
(может быть **долгим**: часы при больших `max-trials` и длинном ряду)

Лёгкий прогон (мало проб):

```powershell
py -3 -m orchestration prepare-symbol --symbol BTC/USDT --timeframe 1h --max-trials 12 --holdout-fraction 0.2
```

Если parquet ещё нет — качнуть историю через data layer (OKX):

```powershell
py -3 -m data_layer -s BTC/USDT -t 1h --from 2022-01-01 --to 2026-01-01 -o data/ohlcv/BTC-USDT_1h.parquet
py -3 -m orchestration prepare-symbol --symbol BTC/USDT --timeframe 1h --download --max-trials 20
```

Показать итоговый manifest символа:

```powershell
py -3 -m orchestration manifest-show --symbol BTC/USDT --timeframe 1h
```

Если после `prepare-symbol` финальный bundle **не** создался из-за падения acceptance на holdout, но нужно собрать артефакты для **`explain`/paper-разработки** (явно «принудительно»):

```powershell
py -3 -m orchestration train-final-symbol --symbol BTC/USDT --timeframe 1h --force
```

### Интроспекция («как система видит сейчас»)

После наличия **последнего** artifact bundle для пары (`artifacts\<slug>\LATEST.txt`):

```powershell
py -3 -m orchestration explain --symbol BTC/USDT --timeframe 1h --window 256
```

Лента режимов для GUI-оверлея (выбор диапазона UTC, как в вашем индексе):

```powershell
py -3 -m orchestration regime-history --symbol BTC/USDT --timeframe 1h --start 2025-12-01 --end 2026-01-01 --step 12
```

Выгрузка в JSON для фронта:

```powershell
py -3 -m orchestration regime-history --symbol BTC/USDT --timeframe 1h --start 2025-12-01 --end 2026-01-01 --step 24 --json-out docs/regime_sample.json
```

**Важно:** этапа `prepare-symbol` с GRU/CNN **пока нет** — подбор там идёт через существующий `run_single_trial` (обычно LGB+XGB). Полный состав моделей проверяется отдельно командой **`from-parquet --full-models`** (раздел 6).

Журнал прогонов: `docs/backtest_journal/runs.jsonl`, снимки `docs/backtest_journal/runs\*.json`, таблица `docs/backtest_journal/INDEX.md`. В записях появились поля **`symbol`, `timeframe`, `stage`**.

Лог задачи `prepare-symbol`:  
`artifacts\<SYMBOL_TF>\active\<task_id>.jsonl`

---

## 4. Отчёт WFO только на LGB+XGB (`report-real`)

Параметры из `config/symbols\<slug>.yaml` → секция **`orchestration_tuning_best`**, если файл есть; иначе fallback на корневой `config.yaml`.

Хвост 8000 баров (быстрый смоук после тюнинга):

```powershell
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 8000 --json-out docs/e2e_last_metrics.json
```

Вся история (очень медленно, много фолдов):

```powershell
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 0 --json-out docs/e2e_full_metrics.json
```

Игнорировать `orchestration_tuning_best`, задать окна явно:

```powershell
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 8000 --no-tuning-best --train-window 1500 --test-window 180 --wfo-step 360
```

---

## 5. Подбор параметров без `prepare-symbol` (`tune-until`)

На хвосте N баров, локальный **refine** вокруг baseline:

```powershell
py -3 -m orchestration tune-until --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 8000 --max-trials 40 --mode refine
```

Широкий grid (дорого):

```powershell
py -3 -m orchestration tune-until --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 8000 --max-trials 80 --mode grid --apply-best
```

`--apply-best` перепишет **корневой** `config.yaml`; для нескольких пар предпочтительнее сохранять лучшие параметры через `prepare-symbol`, который пишет `config/symbols\<slug>.yaml`.

---

## 6. Полный набор моделей (LGB + XGB + **GRU** + **CNN**)

Это главная команда проверки **полного ансамбля** из `config.yaml` → `orchestration.model_keys`.

Требования: **TensorFlow**, достаточно RAM/GPU времени эпох `dl_epochs`.

### Вариант A — демо parquet (обычно меньше строк, быстрее для первой проверки TF)

Убедиться, что файл есть (`demo_BTC-USDT_1h.parquet`). Мало эпох:

```powershell
py -3 -m orchestration from-parquet data/ohlcv/demo_BTC-USDT_1h.parquet --config config.yaml --full-models --dl-epochs 2
```

### Вариант B — ваш основной BTC 1h (долго, реалистично)

```powershell
py -3 -m orchestration from-parquet data/ohlcv/BTC-USDT_1h.parquet --config config.yaml --full-models --dl-epochs 3
```

### Вариант C — только «лёгкие» модели (без TF, контроль)

```powershell
py -3 -m orchestration from-parquet data/ohlcv/demo_BTC-USDT_1h.parquet --config config.yaml --dl-epochs 3
```

(флаг **`--full-models` не указан** → только LGB+XGB)

### Если вход — сырой OHLCV без готовых фичей

Укажите `--raw-ohlcv` (путь должен быть совместим с `FeatureManager`):

```powershell
py -3 -m orchestration from-parquet path/to/raw.parquet --config config.yaml --raw-ohlcv --full-models --dl-epochs 2
```

**Остановка без ассистента:** если процесс висит после «первого барa» TF — первым делом проверить дескрипторы GPU/CUDA или уменьшить `--dl-epochs` и размер parquet.

---

## 7. Вспомогательный скрипт узкого Sharpe-подбора (опционально)

Если уже зафиксирован baseline в `orchestration_tuning_best`:

```powershell
Set-Location D:\IST
$env:PYTHONPATH = "D:\IST"
py -3 scripts/sharpe_push_trials.py
```

Результаты попадают в тот же `docs/backtest_journal`.

---

## 8. Интеграционные маркеры pytest (если используете)

```powershell
py -3 -m pytest -m integration -q
```

(точное имя маркера смотрите в `pytest.ini` / документации проекта, если маркеры отключены — используйте явный список файлов из раздела 2).

---

## 9. Краткая таблица «что проверить»

| Что проверить | Команда / артефакт |
|---------------|---------------------|
| Per-symbol конфиг на диске | `config/symbols\<SYMBOL_TF>.yaml` |
| Manifest готовности | `py -3 -m orchestration manifest-show ...` или `artifacts\<slug>\manifest.json` |
| Bundle для inference | `artifacts\<slug>\LATEST.txt` → каталог с `manifest.json` + `artifacts\*.pkl` |
| Журнал метрик | `docs/backtest_journal/` |
| Быстрые модели только | `report-real`, `prepare-symbol`, `from-parquet` без `--full-models` |
| **Все модели** | **`from-parquet ... --full-models --dl-epochs …`** |

---

## 10. Честные ограничения текущей версии

- `prepare-symbol` не подмешивает GRU/CNN в цикл подбора; для проверки **полного** стека используйте раздел **6**.
- `--force` на `train-final-symbol` делает деплой-friendly bundle при неверифицированном holdout — для реального живого включения нужен пройденный acceptance на вашей политике holdout/paper.

При появлении новых команд обновите этот файл локально или попросите ассистента синхронизировать разделы.
