# IST — журнал тестирования проекта

Единое место для **автоматических прогонов**, **ручных/скриптовых проверок** и **метрик качества моделей / бэктестов**. После каждого значимого прогона добавляйте секцию **«Прогон #N»** или дополняйте таблицы ниже.

---

## 1. Что входит в «полное» тестирование репозитория

| Область | Что проверяется | Как фиксируем результат |
|---------|-----------------|-------------------------|
| Данные и загрузка | OHLCV, гэпы, OKX-loader, валидация | pytest `tests/test_*ohlcv*`, `test_gap_handler.py` |
| Фичи и мультитаймфрейм | `FeatureEngine`, order book sim, MTF | pytest `test_feature_engine.py`, `test_multi_timeframe_engine.py` |
| Оркестрация | конфиг, train/inference orchestrator, контракт пайплайна | pytest `tests/test_orchestration.py` |
| Утечки и пороги | safe threshold, метки meta | pytest `test_audit_fixes.py`, скрипт `test_threshold_leakage.py` |
| WFO и бэктест | walk-forward + `Backtester` + расширенные метрики / критерии (а)/(б) | pytest `tests/test_wfo_backtest.py`, `test_performance_metrics.py`, `test_backtesting_criteria.py` |
| Интеграция «зазоров» | YAML, bundle, glue, parquet glue | pytest `tests/test_integration_gaps_closure.py` (**LightGBM + XGBoost + sklearn**) |
| Качество моделей / E2E | реальные OHLCV → FeatureEngine → **LGB+XGB** → WFO | pytest `-m integration`, CLI `report-real`, §6 |
| CLI оркестратора | smoke WFO, валидация config | `python -m orchestration smoke`, `validate-config` |

**Готовность «полного green»**: все строки матрицы выше либо `PASS`, либо явно помечены `SKIP` с причиной (нет данных, нет GPU/TF, нет библиотек).

---

## 2. Зависимости окружения

Запишите версию Python и наличие опций перед прогоном «модельного» уровня:

| Компонент | Обязательно для | Проверка |
|-----------|-----------------|----------|
| Python 3.11+ | pytest | `py -3 --version` |
| pandas, numpy, pytest | большинство тестов | уже в `requirements.txt` |
| **lightgbm**, **xgboost**, **sklearn** | табличные модели + `report-real` | `py -3 -c "import lightgbm, xgboost, sklearn"` |
| **tensorflow** | GRU/CNN в CLI `--full-models` | по необходимости |

---

## 3. Команды (копировать в метаданные прогона)

### 3.1 Автоматические тесты

```powershell
Set-Location D:\IST

# Быстрый контур (без интеграции на demo parquet / без обучения GBDT)
py -3 -m pytest tests/ -v --tb=short -m "not integration"

# Полный контур, включая реальные модели на ``data/ohlcv/demo_BTC-USDT_1h.parquet``
py -3 -m pytest tests/ -v --tb=short

# Только E2E без мок-моделей (обучение LGB+XGB)
py -3 -m pytest tests/test_orchestration_real_models.py -v --tb=short -m integration

# Только оркестрация (unit + mock — быстро)
py -3 -m pytest tests/test_orchestration.py -v --tb=short

# Интеграция зазоров (YAML, bundle, glue)
py -3 -m pytest tests/test_integration_gaps_closure.py -v --tb=short
```

### 3.2 Скрипт порогов (не pytest; нужен parquet)

```powershell
py -3 test_threshold_leakage.py
```

Требует файл `data/ohlcv/demo_BTC-USDT_1h.parquet` (или поправьте путь в скрипте).

### 3.3 CLI оркестратора

```powershell
py -3 -m orchestration smoke
py -3 -m orchestration validate-config --config config.yaml
# При наличии данных:
# py -3 -m orchestration from-parquet path\to\features.parquet --config config.yaml
```

### 3.4 Реальные модели на OHLCV (без Mock): отчёт по фолдам

Использует ``FeatureEngine`` + **LightGBM + XGBoost** + leakage-safe WFO (конфиг окон — ``orchestration/real_data_benchmark.py``). Режим по умолчанию: хвост **2200** баров после индикаторов (ускорение).

```powershell
py -3 -m orchestration report-real --max-rows 2200 --json-out docs/e2e_last_metrics.json
# полный BTC 1h (рекомендуется для оценки критериев):
py -3 -m orchestration report-real --parquet data\ohlcv\BTC-USDT_1h.parquet --max-rows 6000 --json-out docs\e2e_last_metrics.json
# все бары после FeatureEngine:
py -3 -m orchestration report-real --parquet data\ohlcv\BTC-USDT_1h.parquet --max-rows 0 --json-out docs\e2e_last_metrics.json
```

Метрики по фолдам + **ACCEPTANCE / TARGET** (из `config.yaml` → `backtesting.acceptance` / `target`) — в stdout и JSON; для журнала — §5 прогон #3 и §6.2.

Справочник порогов: `docs/BACKTESTING_CRITERIA_REFERENCE.md`.

---

## 4. Сводка последнего полного прогона (обновлять после каждого забега)

| Дата | Python | pytest `tests/` (полный) | `-m "not integration"` | threshold script | smoke | report-real |
|------|--------|--------------------------|-------------------------|------------------|-------|-------------|
| 2026-05-16 | 3.11.0 | 66+ passed | — | OK (demo parquet) | OK | OK — см. §5 прогон #3 (BTC 6k, критерии) |

---

## 5. Прогон #1 — baseline полного охвата (2026-05-16)

### Метаданные

| Поле | Значение |
|------|----------|
| Дата | 2026-05-16 |
| Commit | `9712897` |
| ОС | Windows |
| Команда pytest | `py -3 -m pytest tests/ -v --tb=short` |

### Итог pytest (`tests/`)

| Passed | Failed | Errors | Skipped |
|--------|--------|--------|---------|
| 46 | 0 | 0 | 2 |

**Пропущено (2):** `tests/test_orchestration.py::TestRegimePredIntegration::test_pipeline_uses_regime_pred_length` — помечен как `SKIP` в тесте.

**Не входит в этот прогон:** модуль `tests/test_integration_gaps_closure.py` при `collect` даёт **0 тестов / модуль пропущен**, если в окружении нет `lightgbm` (установите пакеты и перезапустите).

### Сырой вывод pytest (сокращённо — полный лог храните при необходимости отдельно)

```
46 passed, 2 skipped in ~2.7s
```

Охват файлов: `test_audit_fixes.py`, `test_feature_engine.py`, `test_gap_handler.py`, `test_multi_timeframe_engine.py`, `test_ohlcv_validator.py`, `test_okx_ohlcv_loader.py`, `test_orchestration.py`, `test_wfo_backtest.py`.

### `test_threshold_leakage.py`

**Статус:** SUCCESS (демо BTC parquet, safe vs unsafe threshold, expanding/rolling без look-ahead).

Ключевые числа из прогона: unsafe median ~0.6834 на всём ряду; train-only median ~0.9508; предупреждение о большой разнице между unsafe и safe — ожидаемо для демонстрации утечки.

### `python -m orchestration smoke`

**Статус:** OK — synthetic WFO, 4 фолда.

Пример метрик smoke (не эталон качества моделей, а проверка цепочки):

| Fold | Sharpe | Total Return % | Max DD % |
|------|--------|----------------|----------|
| 1 | -20.17 | -1.01 | -1.08 |
| 2 | -32.10 | -1.51 | -1.51 |
| 3 | -25.29 | -1.23 | -1.38 |
| 4 | -17.29 | -0.90 | -1.32 |

### `validate-config`

**Статус:** OK для `config.yaml`.

**Замечание:** предупреждение об игнорируемых ключах YAML (`safe_threshold_mode`, `threshold_expanding`, `threshold_window`, `train_threshold`) — они не входят в `OrchestratorConfig`; при необходимости синхронизировать схему конфига и код.

### Выводы по прогону #1

- Автотесты ядра (`tests/`) зелёные; интеграционный файл с LightGBM/XGBoost нужно прогнать после установки зависимостей.
- Качество «моделей» на smoke-Negative Sharpe ожидаемо для заглушечных предсказаний — отдельный учёт в §6.

---

## Прогон #2 — реальные модели (demo BTC 1h), без Mock

### Метаданные

| Поле | Значение |
|------|----------|
| Дата | 2026-05-16 |
| Данные | ``data/ohlcv/demo_BTC-USDT_1h.parquet`` → ``FeatureEngine``, хвост 2200 баров |
| Модели | LightGBM + XGBoost (ключи ``lgb``, ``xgb``) |
| Режим | ``SimpleHourlyRegimeStub`` (детерминированный ``regime_pred``); контур мета + риск как в прод-пайплайне |

### Команды

- Интеграционные тесты: `py -3 -m pytest tests/test_orchestration_real_models.py -m integration -v`
- Отчёт + JSON: `py -3 -m orchestration report-real --max-rows 2200 --json-out docs/e2e_last_metrics.json`

### Итог pytest

Входит в полный прогон ``tests/``: **+4** теста ``test_orchestration_real_models.py`` (маркер ``integration``).

### Метрики WFO (пример текущего прогона)

Копия из ``docs/e2e_last_metrics.json``:

| Fold | Sharpe | PF | Win % | Max DD % | Total Return % |
|------|--------|-----|-------|----------|----------------|
| 1 | 2.647 | 1.096 | 40.35 | -3.90 | 1.092 |
| 2 | -5.308 | 0.796 | 45.13 | -6.08 | -2.983 |
| 3 | -0.809 | 0.966 | 40.21 | -2.32 | -0.399 |
| 4 | -8.055 | 0.705 | 41.05 | -5.92 | -3.715 |

_Значения не являются «целевым качеством» стратегии — это зафиксированный снимок контура «данные → обучение → бэктест»._

### Заметки

- Unit-тесты ``tests/test_orchestration.py`` и ``tests/test_wfo_backtest.py`` по-прежнему используют лёгкие заглушки моделей для скорости; реальный контур вынесен в ``test_orchestration_real_models.py`` и CLI ``report-real``.
- GRU/CNN не включены (TensorFlow); для них — ``from-parquet --full-models`` после установки TF.

---

## Прогон #3 — расширенные метрики + критерии (а)/(б), BTC-USDT 1h

### Метаданные

| Поле | Значение |
|------|----------|
| Дата | 2026-05-16 |
| Данные | `data/ohlcv/BTC-USDT_1h.parquet` (31441 raw rows), хвост **6000** баров после `FeatureEngine` |
| Модели | LightGBM + XGBoost (`lgb`, `xgb`), без Mock |
| WFO | train=900, test=180, step=360 → **14 OOS-фолдов** |
| Издержки | commission=0.0006, slippage=0.0002 (`config.yaml` → `backtesting.simulation`) |
| Код | `backtesting/performance_metrics.py`, `criteria_evaluator.py`, `metrics_config.py` |

### Команды

```powershell
py -3 -m pytest tests/test_performance_metrics.py tests/test_backtesting_criteria.py tests/test_wfo_backtest.py -v
py -3 -m pytest tests/test_orchestration_real_models.py -m integration -v
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 6000 --json-out docs/e2e_last_metrics.json
```

### pytest (новые тесты метрик)

| Набор | Результат |
|-------|-----------|
| `test_performance_metrics.py` + `test_backtesting_criteria.py` + `test_wfo_backtest.py` | **4 passed** |
| `test_orchestration_real_models.py` (integration) | **5 passed** (~38s) |

### Новые метрики в `PerformanceMetrics` (OOS и IS_*)

Sortino, Calmar, CAGR (%), Annualized Volatility, Ulcer Index, Time Underwater (bars), Max Consecutive Losses, Trade Events, Benchmark Return / Alpha / Benchmark Sharpe; по фолдам — **Walk-Forward Efficiency**, IS/OOS annualized return.

### Агрегаты WFO (снимок)

| Метрика | Значение |
|---------|----------|
| n_folds | 14 |
| mean Sharpe (OOS) | -3.83 |
| mean Profit Factor | 0.93 |
| mean Total Return % | -2.03 |
| mean WFE | -0.014 |
| worst Max DD % | -10.48 |
| total trade events | 1521 |
| folds with positive return | 4 / 14 |

### Критерии из `config.yaml`

| Уровень | Результат | Прошло / всего проверок |
|---------|-----------|-------------------------|
| **(а) acceptance** | **FAIL** | 3 / 8 (OK: min_folds, max_drawdown, min_trade_events) |
| **(б) target** | **FAIL** | 4 / 10 (доп. OK: min_calmar — при отрицательной доходности Calmar может быть обманчив) |

Не прошли: mean PF, mean Sharpe, mean WFE, mean recovery, mean total return; target sortino.

**Вывод:** контур «данные → LGB+XGB → WFO → метрики → автопроверка» работает на **реальном** BTC 1h; стратегия **ещё не** на уровне (а) «можно торговать» — нужна итерация обучения/фич/порогов (не снижать пороги без обоснования).

Полный JSON: `docs/e2e_last_metrics.json` (`acceptance_passed`, `target_passed`, `folds[]`).

---

## 6. Качество моделей и E2E (заполнять по мере появления данных)

Используйте после реальных обучений / parquet-пайплайна.

### 6.1 Чеклист приёмки E2E

- [x] OHLCV parquet → признаки (`FeatureEngine`) → `TrainingOrchestrator` / WFO без ошибок формы — **прогон #2**, ``report-real``
- [x] Для каждого фолда WFO: train без утечки — встроено в ``walk_forward_backtest`` (`validate_no_leakage`)
- [ ] Порог сигнала строго causal на всех ветках meta/assembler — частично (см. аудит порогов в коде)
- [x] Метрики записаны — таблица ниже + ``docs/e2e_last_metrics.json``
- [x] Автопроверка acceptance/target — **прогон #3** (пока FAIL на BTC 6k, LGB+XGB)

### 6.2 Метрики по фолдам (ручная таблица)

| Прогон | Fold | Train bars | Test bars | Model suite | Acc/AUC / PR | Sharpe | PF | MaxDD % | Trades | Примечание |
|--------|------|------------|-----------|-------------|--------------|--------|-----|---------|--------|------------|
| #2 demo 2200 | 1–4 | 900 | 180 | lgb+xgb | — | см. JSON | см. JSON | см. JSON | — | старый снимок |
| #3 BTC 6k | 1–14 | 900 | 180 | lgb+xgb | — | mean -3.83 | 0.93 | -10.5 | 1521 events | acceptance **FAIL** |
| _TODO_ | | | | + GRU+CNN | | | | | | TF, ``from-parquet --full-models`` |
| _TODO_ | | | | all keys | | | | | | полный parquet ``--max-rows 0`` |

### 6.3 Регрессии и известные риски (живой список)

Обновляйте после код-ревью и прогонов.

| ID | Тема | Статус | Комментарий |
|----|------|--------|-------------|
| M1 | `tests/test_integration_gaps_closure.py` без GBDT | FIXED при наличии pip-пакетов | нужны `lightgbm`, `xgboost`, `sklearn` |
| M2 | YAML ключи порогов не в `OrchestratorConfig` | OPEN | WARN при `validate-config` |
| M3 | Реальные модели vs Mock | PARTIAL | mock остаётся в быстрых тестах; E2E — ``test_orchestration_real_models.py`` + ``report-real`` |
| M4 | Качество smoke-моделей | INFO | отрицательный Sharpe на синтетике нормален для smoke |

_(При необходимости перенесите сюда строки из старого аудита A1–A18 из истории git.)_

---

## 7. Шаблон нового прогона (копировать)

```markdown
## Прогон #N

### Метаданные
| Поле | Значение |
|------|----------|
| Дата | |
| Commit | `git rev-parse --short HEAD` |
| Python | |
| Команды | (pytest / скрипты / CLI) |

### pytest
| Passed | Failed | Errors | Skipped |

### Другие проверки
- threshold script: 
- orchestration smoke: 
- validate-config: 

### Сырой вывод (при необходимости полный)
\`\`\`
\`\`\`

### Заметки
-
```

---

## 8. Сравнение прогонов (кратко)

| Прогон | Дата | pytest | Заметка |
|--------|------|--------|---------|
| #1 | 2026-05-16 | 46p / 2sk | исторический baseline (до правок deps / gaps) |
| #2 | 2026-05-16 | 57p | + demo parquet + реальный LGB+XGB WFO; см. ``e2e_last_metrics.json`` |
| #3 | 2026-05-16 | +9 тестов метрик | BTC-USDT 1h, 14 folds, расширенные метрики + criteria; acceptance/target FAIL |
