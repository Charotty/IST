# 10. Backtesting · 11. WFO · 12. Результаты · 13. GUI · 14–15. Пробелы и статусы · 17. Артефакты

---

## 10. Backtesting System

### 10.1. Полноценный ли backtest?

**Да.** Модуль `backtesting/`:

| Компонент | Файл | Назначение |
|-----------|------|------------|
| Симуляция PnL | `backtester.py` | `Backtester.run()` |
| Метрики | `performance_metrics.py` | Sharpe, DD, PF, Calmar, … |
| Критерии приёмки | `criteria_evaluator.py` | acceptance vs target |
| Журнал | `results_journal.py` | `docs/backtest_journal/runs.jsonl` |

### 10.2. Equity curve

```python
results['cum_strategy_returns'] = (1 + net_returns).cumprod()
results['drawdown'] = (cum - expanding_max) / expanding_max
```

Колонки: `market_returns`, `strategy_returns`, `net_returns`, `costs`, `trades`.

### 10.3. Метрики (основные)

Из `PerformanceMetrics.calculate_metrics()`:

| Метрика | Есть |
|---------|------|
| Total Return (%) | Да |
| CAGR (%) | Да |
| **Sharpe Ratio** | Да |
| Sortino, Calmar | Да |
| **Max Drawdown (%)** | Да |
| Profit Factor | Да |
| Ulcer Index, time underwater | Да |
| Walk-Forward Efficiency (WFE) | Да (в WFO-таблице фолдов) |
| Benchmark Sharpe (buy & hold) | Да |

`bars_per_year: 8760` для 1h (в `backtesting` config).

### 10.4. Комиссия и проскальзывание

| Параметр | Значение (canonical) |
|----------|----------------------|
| `commission` | **0.0006** (0.06%) |
| `slippage` | **0.0002** (0.02%) |
| Учёт | при изменении эффективной экспозиции `|Δ(signal × position_size)|` |

Согласовано с `execution` defaults (см. docstring `Backtester`).

### 10.5. Risk-aware backtest

WFO вызывает:

```python
bt.run(test_features[["close"]], test_result.final_signals, position_size=test_result.position_sizes)
```

Размер позиции из `OrchestratorRiskBridge` / `RiskPipeline`.

---

## 11. Walk-Forward Validation / WFO

### 11.1. Есть ли WFO?

**Да** — основной путь валидации:

- `TrainingOrchestrator.walk_forward_backtest()`
- `backtesting/walk_forward.run_orchestrator_walk_forward_backtest()`

**Нет** как единственного метода — также есть простой `TimeSeriesSplitter` без ML (`run_simple_signal_wfo`) для готовых сигналов.

### 11.2. Параметры окон (canonical)

| Параметр | Значение |
|----------|----------|
| `train_window_size` | **1500** баров |
| `test_window_size` | **250** |
| `walk_forward_step` | **250** |
| `prediction_horizon` | **12** (purge labels) |
| `enable_purge` | **true** |
| `embargo_period` | **5** баров |

### 11.3. Purge / embargo / OOS

Реализовано в `utils/data_leakage_prevention.py` → `TrainingOrchestrator.leakage_preventer`:

```python
splits = leakage_preventer.safe_walk_forward_split(
    features, targets,
    train_window_size, test_window_size, walk_forward_step,
    prediction_horizon,
)
# + validate_no_leakage() на каждом фолде
```

| Механизм | Статус |
|----------|--------|
| Rolling/expanding WFO | **Rolling** с шагом 250 |
| Purge (horizon) | **Да** |
| Embargo между train/test | **Да** (5 баров) |
| OOS evaluation | **Да** — метрики на test fold после train-only fit |
| Meta threshold на test | Train-only: `_calibrate_test_meta_threshold` |

**Простой splitter** `backtesting/time_series_splitter.py` — **без** purge/embargo (только вспомогательный).

### 11.4. Критерии приёмки (acceptance)

`config/profiles/canonical_4model.yaml` → `backtesting.acceptance`:

| Критерий | Порог |
|----------|-------|
| min_folds | 5 |
| min_oos_sharpe | 0.5 |
| min_oos_profit_factor | 1.2 |
| max_drawdown_pct | -35% |
| min_wfe | 0.5 |
| min_trade_events | 30 |

Проверка: `backtesting/criteria_evaluator.py` → статусы **PASS/FAIL** в журнале.

### 11.5. Упрощение глав 2–3, если бы WFO не было

В проекте WFO **есть**; главы можно строить вокруг `walk_forward_backtest` + журнала. Старые функции `run_walk_forward_validation` / `run_integrated_wfo` — **удалены** (`NotImplementedError`).

---

## 12. Какие результаты реально есть

### 12.1. Журнал бэктестов

- **Индекс:** `docs/backtest_journal/INDEX.md` (сотни/тысячи прогонов)
- **Строки:** `docs/backtest_journal/runs.jsonl`
- **Детали:** `docs/backtest_journal/runs/<run_id>.json`

Типичные колонки сводки: `mean Sharpe`, `mean PF`, `mean WFE`, `mean ret %`, `acceptance`, `target`.

### 12.2. Пример недавних прогонов (BTC-USDT 1h, report-real)

| run_id (кратко) | folds | acc | mean Sharpe | mean PF | mean WFE | mean ret % |
|-----------------|-------|-----|-------------|---------|----------|------------|
| 20260523T155940Z | 18 | **FAIL** | 2.32 | 1.78 | 0.12 | 0.58 |
| 20260523T150959Z | 18 | **FAIL** | 2.53 | 1.79 | 0.20 | 0.75 |
| 20260523T145438Z | 147 | **FAIL** | 0.68 | 1.17 | 0.05 | 0.24 |

**Вывод для ВКР:** средний OOS Sharpe может быть **> 0.5**, но **acceptance FAIL** (часто из‑за WFE, DD, trade count или target-критериев). Не писать «система прошла приёмку», не сверив `criteria_evaluator` по конкретному run.

### 12.3. Сравнение моделей / ensemble / adaptive vs static

| Сравнение | Как получить |
|-----------|--------------|
| Отдельные модели | Ablation: убрать ключи из `model_keys`, `scripts/run_ablation.py` |
| Ensemble vs single | WFO с одним ключом vs четырьмя |
| Regime-adaptive vs static | `ensemble_mode: regime_adaptive` vs `fixed_trend` / `fixed_range` |
| Тюнинг | `orchestration/tuning_loop.py`, `tune-thesis` |

Графики сравнения ensemble — генерировать заново (`scripts/run_ablation.py`) или из журнала WFO; каталог `docs/figures/3_*` снят с репозитория.

### 12.4. Screenshots / logs

- GUI: `docs/figures/3_11/gui/*.png`
- Equity/DD: `docs/figures/3_9/equity_drawdown.png`, `3_11/equity_drawdown_comparison.png`
- WFO схема: `docs/figures/3_11/wfo_split_visualization.png`, `purge_embargo_scheme.png`
- CLI logs: `artifacts/.../active/*.jsonl`

---

## 13. GUI

### 13.1. Mockup или рабочий?

**Рабочий desktop (PyQt6)**, не mockup.

| Слой | Путь | Статус |
|------|------|--------|
| API без Qt | `gui/api/` (`IstGuiClient`) | Реализован |
| UI | `gui/app/` | Реализован |
| HTTP backend | — | Не нужен (прямой вызов Python API) |

Запуск: `py -3 -m gui.app` (+ `requirements-gui.txt`).

### 13.2. Функционал (кратко)

| Вкладка / зона | Данные |
|----------------|--------|
| Обзор | `explain` — сигнал, meta P(up), режим, веса |
| График | OHLCV, regime overlay, сигналы, meta probability |
| Модели | статус bundle, manifest |
| Бэктесты | журнал WFO, equity по run |
| Исполнение | paper connect, шаг по бару, сверка с Backtester |
| Задачи | CLI: prepare-symbol, train-final, report-real, tune |
| Настройки | пути, acceptance report |

Документация: `gui/README.md`, `docs/GUI_USER_GUIDE.md`, `docs/GUI_LAYOUT_REFERENCE.md`.

### 13.3. Screenshots

См. `docs/figures/3_11/gui/`:

- `gui_overview_explain.png`
- `gui_chart_signals.png`
- `gui_backtests_journal.png`
- `gui_models_bundle.png`

---

## 14. Что НЕ реализовано (честный список)

```text
planned / discussed but NOT in canonical production:

- Transformer как 5-я модель в orchestration
- LSTM под ключом lstm в prod ensemble (заменён на gru)
- Online обучение весов ансамбля по rolling PnL
- MetaFilter как отдельная обученная модель (только threshold rules)
- ML RegimeDetector в inference pipeline
- Третий режим breakout в DynamicMetaWeighting
- Live LOB / microstructure mode: live
- Полный autonomous live trading на OKX
- Единая команда «OKX → trade» без ручной сборки
- Reinforcement learning для НАПРАВЛЕНИЯ (RL только risk multiplier)
- Distributed training
- Meta-learning в смысле MAML / few-shot
- Binance как primary connector
- HTTP REST API для внешних клиентов
- Take-profit как отдельное правило (только trailing в risk)
```

**Legacy только в `ist.py`:** Transformer, LSTM naming, полный RL trading env, ensemble Colab.

---

## 15. Демо vs полноценные компоненты

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| 4-model WFO + backtest | **Full** | основной research path |
| Regime-adaptive ensemble | **Full** | фиксированные таблицы весов |
| Performance-based weight learning | **Stub** | утилита без wiring |
| Rule-based regime (SMA) | **Full** | prod |
| ML regime | **Prototype** | код есть |
| Decision + meta threshold | **Full** | leakage-safe OOS |
| Risk bridge + ATR sizing | **Full** | trailing опционален |
| Backtest journal + criteria | **Full** | |
| Artifact bundles | **Full** | `artifact_bundle.py` |
| Paper execution | **Partial** | шаговый loop, long bias |
| Live trading | **Not** | |
| GUI | **Full** (research desktop) | |
| Simulated microstructure | **Optional** | off in canonical |
| RL risk overlay | **Partial** | не главный signal path |
| `ist.py` end-to-end | **Legacy / demo** | не синхронизирован с gru/xgb keys |

---

## 17. Графики и артефакты

### 17.1. Каталог фигур

Ранее: `docs/figures/3_3` … `3_11` (удалены). Для диплома:

| Тип | Как получить |
|-----|----------------|
| GUI | `python docs/scripts/capture_gui_screenshots.py` |
| WFO / equity | экспорт из `docs/backtest_journal/runs/<id>.json` |
| Индикаторы §3.1 | построить из `data/features/*.parquet` |

### 17.2. Runtime-артефакты

| Артефакт | Путь |
|----------|------|
| Обученные модели | `artifacts/<symbol_tf>/<run_id>/artifacts/*.pkl` |
| Manifest | `artifacts/.../manifest.json`, `LATEST.txt` |
| Feature parquet | `data/features/*.parquet` |
| OHLCV parquet | `data/ohlcv/*.parquet` |
| Журнал WFO | `docs/backtest_journal/` |
| Active training log | `artifacts/.../active/*.jsonl` |

### 17.3. Чего может не хватать без перегенерации

- Актуальные скриншоты GUI под **текущую** ветку UI
- Единая таблица «adaptive vs static» с числами из **последнего** `report-real` (взять из `runs.jsonl`)
- Веса по времени в интерактивном виде (есть PNG, не обязательно live в GUI)

### 17.4. Как воспроизвести ключевые артефакты

```powershell
py -3 -m orchestration report-real --symbol BTC-USDT --timeframe 1h
# → обновит docs/backtest_journal/

py -3 -m gui.app
# → скриншоты вручную или из docs/figures/3_11/gui/
```

---

## Связанные документы в репозитории

| Документ | Зачем |
|----------|-------|
| `docs/DIPLOMA_SYSTEM_DOCUMENTATION.md` | Полный справочник A–J |
| `docs/INTEGRATION_GAPS_REFERENCE.md` | Разрывы интеграции |
| `docs/BACKTESTING_CRITERIA_REFERENCE.md` | Критерии метрик |
| `docs/backtest_journal/INDEX.md` | Таблица всех прогонов |
| `config/profiles/canonical_4model.yaml` | Эталонные числа |
