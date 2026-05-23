# GUI Layer (PyQt6)

## Назначение

Десктоп-приложение для **мониторинга и исследования** Intelligent Trading System (IST):

- визуализация pipeline (данные → 4 модели → regime → ансамбль → decision → risk);
- карточка решения на последнем баре (`explain`);
- журнал WFO / бэктестов;
- статус подготовки символа (parquet, features, bundle);
- позже — paper/live через **execution**.

Research по-прежнему доступен через CLI (`python -m orchestration …`) и журнал `docs/backtest_journal/`. Справочник всех команд и приёмки GUI: [`docs/PROJECT_CLI_AND_GUI_COMMANDS.md`](../docs/PROJECT_CLI_AND_GUI_COMMANDS.md).

Конфиг: эталон [`config/reference/thesis_4model_reference.yaml`](../config/reference/thesis_4model_reference.yaml), пары — `baseline_ref` + `orchestration_overrides` (см. [`docs/THESIS_REFERENCE.md`](../docs/THESIS_REFERENCE.md)). Руководство: [`docs/GUI_USER_GUIDE.md`](../docs/GUI_USER_GUIDE.md).  
Описание окон и расположения виджетов: [`docs/GUI_LAYOUT_REFERENCE.md`](../docs/GUI_LAYOUT_REFERENCE.md).

## Статус

| Компонент | Статус |
|-----------|--------|
| `gui/api/` | ✅ API-слой (обёртки orchestration) |
| PyQt6 UI (`gui/app/`) | ✅ 3 вкладки: **График** (chart+решение/режим/bundle), **Задачи** (pipeline/журнал/paper), **Конфигурация** |
| CLI из Jobs | ✅ prepare, build-features, tune-thesis, train-final, report-real + smoke, validate, tune-until, from-parquet, regime-history, list-symbols, manifest-show |
| Вкладка «Режим» | ✅ `inference.regime_series` |
| График IST | ✅ `chart_payload`: regime overlay, сигналы, meta P(up) |
| HTTP backend | ⬜ не нужен для desktop (прямой вызов `gui.api`) |
| Live OKX trading | ⬜ (paper только) |

### CLI ↔ GUI (Jobs / Settings)

| CLI | GUI |
|-----|-----|
| `build-features` | Кнопка «Признаки» |
| `tune-thesis --phase …` | «Тюнинг thesis» + фаза |
| `report-real --use-tuning-best --use-feature-cache` | «Отчёт WFO»; Settings → «Отчёт acceptance» |
| `train-final-symbol` | «Финальное обучение» |

## Технологии

| Слой | Стек |
|------|------|
| UI | **PyQt6** (Widgets + optional QtCharts / pyqtgraph для свечей) |
| API | Python-модуль `gui.api.IstGuiClient` — без Qt-зависимостей |
| Данные | Parquet, `artifacts/`, `docs/backtest_journal/` |
| Тяжёлые вызовы | `QThread` / `QRunnable` + сигналы в UI |

Зависимости UI (добавить в `requirements-gui.txt` при реализации):

```text
PyQt6>=6.6
pyqtgraph>=0.13
```

## Архитектура

```text
┌─────────────────────────────────────────────────────────┐
│  PyQt6 MainWindow (gui/app/)                            │
│  Views: Overview | Chart | Models | Backtests | Jobs    │
└───────────────────────────┬─────────────────────────────┘
                            │ IstGuiClient (sync / threads)
┌───────────────────────────▼─────────────────────────────┐
│  gui/api/                                               │
│  symbols | bundles | inference | backtests | jobs       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  orchestration · backtesting · feature_engineering        │
└─────────────────────────────────────────────────────────┘
```

**Правило:** виджеты **не импортируют** LightGBM/TensorFlow напрямую — только через `gui.api`.

---

## API-слой (`gui/api/`)

Единая точка входа:

```python
from gui.api import IstGuiClient

api = IstGuiClient()
entry = api.symbols.resolve("BTC/USDT", "1h")
card = api.inference.explain("BTC/USDT", "1h", window=256)
runs = api.backtests.list_runs(symbol="BTC-USDT", timeframe="1h", limit=50)
```

### Модули API

| Модуль | Класс | Назначение |
|--------|-------|------------|
| `client.py` | `IstGuiClient` | Facade, подключает все под-API |
| `symbols_api.py` | `SymbolsApi` | Список пар, пути, `merged_config`, `data_health` |
| `bundles_api.py` | `BundlesApi` | `manifest.json`, schema hash, model_keys |
| `inference_api.py` | `InferenceApi` | `explain`, `regime_series`, `chart_bars` |
| `backtests_api.py` | `BacktestsApi` | `runs.jsonl`, снимки `runs/<id>.json` |
| `jobs_api.py` | `JobsApi` | `active/*.jsonl`, чеклист pipeline |
| `types.py` | dataclasses | `ExplainSnapshot`, `ChartBar`, … — для Qt models |

### Типизированные ответы (`gui/api/types.py`)

| Тип | Для какого экрана |
|-----|------------------|
| `SymbolEntry` | Combo symbol / TF, статус bundle |
| `DataHealth` | Jobs — строки OHLCV/features, last timestamp |
| `BundleInfo` | Settings / Models — колонки фич, run_id |
| `ExplainSnapshot` | **Overview** — главная карточка |
| `RegimeBar` | **Chart** — фон trend/range |
| `ChartBar` | **Chart** — OHLCV + signal/meta_prob |
| `BacktestRunSummary` | **Backtests** — таблица прогонов |
| `JobEvent` | **Jobs** — лог `prepare-symbol` |
| `PipelineStepStatus` | **Jobs** — галочки этапов |

### Потоки в PyQt6

```python
from PyQt6.QtCore import QThread, pyqtSignal

class ExplainWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, api: IstGuiClient, symbol: str, tf: str):
        super().__init__()
        self._api = api
        self._symbol = symbol
        self._tf = tf

    def run(self):
        try:
            self.finished.emit(self._api.inference.explain(self._symbol, self._tf))
        except Exception as e:
            self.failed.emit(str(e))
```

Тяжёлые методы: `inference.explain`, `inference.chart_bars(..., include_signals=True)`, `bundles.bundle_info` (schema check).

---

## Структура каталога (целевая)

```text
gui/
├── README.md                 # этот файл
├── __init__.py
├── api/                      # ✅ реализовано
│   ├── client.py
│   ├── types.py
│   ├── symbols_api.py
│   ├── bundles_api.py
│   ├── inference_api.py
│   ├── backtests_api.py
│   └── jobs_api.py
└── app/                      # ⬜ PyQt6
    ├── main.py               # QApplication entry
    ├── main_window.py        # QMainWindow + QStackedWidget / tabs
    ├── workers.py            # QThread wrappers над IstGuiClient
    ├── widgets/
    │   ├── symbol_toolbar.py # QComboBox symbol, TF, bundle
    │   ├── explain_card.py
    │   ├── model_weights_bar.py
    │   ├── criteria_badge.py
    │   ├── chart_widget.py   # pyqtgraph candlestick + regime
    │   ├── backtest_table.py
    │   └── job_log_view.py
    └── views/
        ├── overview_view.py
        ├── chart_view.py
        ├── models_view.py
        ├── backtests_view.py
        ├── pipeline_view.py
        └── settings_view.py
```

Запуск:

```powershell
pip install -r requirements-gui.txt
py -3 -m gui.app
```

---

## Экраны PyQt6 — что для чего

### 0. Shell — `MainWindow` + `SymbolToolbar`

| Элемент | Виджет | API / данные |
|---------|--------|----------------|
| Символ | `QComboBox` | `api.symbols.list_symbols()` |
| Таймфрейм | `QComboBox` | из `SymbolEntry.timeframe` |
| Bundle | `QLabel` / combo run_id | `api.bundles.latest_bundle_dir()` |
| Статус | `QLabel` ● Ready / No bundle | `SymbolEntry.has_bundle` |
| Refresh | `QPushButton` | перезагрузка активной вкладки |

**Зачем:** единый контекст для всех вкладок; пути только через `orchestration.symbols` (без «магических» строк в UI).

---

### 1. Overview — «состояние системы сейчас»

**Назначение:** главный экран для диплома и оператора — что система думает **на последнем баре**.

| Блок UI | Поля | API |
|---------|------|-----|
| Карточка решения | direction, signal, meta_probability, confidence | `inference.explain()` → `ExplainSnapshot` |
| Цена / время | last_close, as_of | то же |
| Режим | trend / range | `regime`, `regime_int` |
| Почему HOLD | `why_blocked` (красный текст) | то же |
| Таблица моделей | lgb, gru, xgb, cnn → P(up) | `model_probs` |
| Таблица весов | активные веса ансамбля | `active_weights` |
| Позиция | position_size_frac | то же |
| Config snippet | direction_threshold, trade_mode, ensemble_mode | `config` |
| Bundle path | run_id, schema_valid | `bundles.bundle_info()` |

**Виджеты:** `ExplainCard` (`QGroupBox` + `QFormLayout`), `ModelWeightsBar` (`QProgressBar` × 4).

**Обновление:** кнопка Refresh + опциональный `QTimer` (`refresh_interval_sec` из конфига).

---

### 2. Chart — свечи и оверлеи

**Назначение:** визуальная проверка сигналов и режима на истории.

| Слой | Отображение | API |
|------|-------------|-----|
| Свечи OHLCV | pyqtgraph `CandlestickItem` | `inference.chart_bars()` → `ChartBar` |
| Фон regime | цветовая полоса trend/range | `inference.regime_series(start, end, step)` |
| Маркеры сделок | стрелки при `signal != 0` | `ChartBar.signal` (хвост окна, см. API) |
| Линия meta prob | вторая ось / subplot | `ChartBar.meta_probability` |
| Диапазон дат | `QDateTimeEdit` start/end | параметры `regime_series` / slice |

**Зачем `step`:** разрежение для длинной истории (как CLI `regime-history --step 24`).

**Производительность:** `max_bars` по умолчанию 2000; inference на хвосте `window=512` — не пересчитывать всю историю в v1.

---

### 3. Models — четыре модели + ансамбль

**Назначение:** объяснение раздела «C. Все 4 модели» диплома.

| Блок | Содержание | API |
|------|------------|-----|
| Статическая таблица | архитектура, window=24 для DL | README + `BundleInfo.model_keys` |
| Sparklines P(up) | последние N баров | повторный `chart_bars` или отдельный batch (v2) |
| Stacked weights | вклад weight×prob | `ExplainSnapshot` на последнем баре; история — v2 |
| Список признаков | `feature_columns` | `bundles.bundle_info()` |
| Переключатель regime | показать trend vs range weights из YAML | `symbols.merged_config()["orchestration"]` |

**Виджет:** `QTableWidget` + `QTabWidget` (Trend weights / Range weights).

---

### 4. Regime (вкладка или подпанель Chart)

**Назначение:** прозрачность rule-based детектора (SMA momentum).

| Элемент | Детали |
|---------|--------|
| Лента режима | `RegimeBar[]` |
| Статистика | % trend / range за период |
| Badge | «Production: MomentumRegimeDetector (close > SMA)» |
| SMA overlay | линия на графике (v2: вычислить из close) |

**API:** `inference.regime_series()`.

---

### 5. Decision & Risk (панель Overview или отдельная вкладка)

**Назначение:** пошаговое «почему такое решение».

| Шаг | Индикатор | Источник |
|-----|-----------|----------|
| 1. Ensemble prob | `meta_probability` vs 0.52 | `ExplainSnapshot` |
| 2. Direction | long/short/flat | `signal`, `direction` |
| 3. Meta filter | pass/fail | `why_blocked is None` |
| 4. Margin | dead-zone | `config.min_signal_margin` |
| 5. Trade mode | both / long_only | `config.trade_mode` |
| 6. Risk size | fraction | `position_size_frac` |

**Виджет:** `QListWidget` с иконками ✓/✗ (stepper).

Конфиг risk (trailing, ATR mult) — read-only из `merged_config` / `BundleInfo.orchestrator_config`.

---

### 6. Backtests & Experiments

**Назначение:** сравнение WFO-прогонов, критерии acceptance.

| UI | API |
|----|-----|
| `QTableView` прогонов | `backtests.list_runs(symbol, timeframe, limit=100)` |
| Фильтры stage / PASS | client-side |
| Детали прогона | `backtests.get_run(run_id)` — folds, criteria |
| Equity curve | построить из `fold_metrics` / summary (v1: таблица метрик) |
| Badge PASS/FAIL | `CriteriaBadge` по `acceptance_passed` |

**Действия (v2):** кнопки запуска subprocess `python -m orchestration report-real` с прогрессом из `jobs.read_task_log`.

---

### 7. Pipeline / Jobs

**Назначение:** статус данных и длительных задач `prepare-symbol`.

| UI | API |
|----|-----|
| Чеклист этапов | `jobs.pipeline_status()` → `PipelineStepStatus[]` |
| Data health | `symbols.data_health()` |
| Список задач | `jobs.list_active_tasks()` |
| Лог задачи | `QPlainTextEdit` ← `jobs.read_task_log(task_id)` |
| Auto-refresh log | `QTimer` 2s при активной задаче |

**Не в v1:** запуск download/train из GUI (только CLI); в v2 — `QProcess` + tail jsonl.

---

### 8. Orders & Execution (фаза 2)

**Статус:** `execution/` — scaffold. Вкладка показывает заглушку «Backtest-only mode».

| Панель | Когда |
|--------|-------|
| Paper positions / orders | после `PaperBroker` |
| Emergency stop | после `execution_manager` |
| Mode switch paper/live | с подтверждением `QMessageBox` |

---

### 9. Settings

| Раздел | Поведение v1 |
|--------|----------------|
| Effective config | `QTextEdit` JSON — `symbols.merged_config()` read-only |
| GUI prefs | `refresh_interval_sec`, default symbol — `QSettings` |
| Paths | показать `repo_root`, journal, artifacts |

---

## Конфигурация GUI

```yaml
# config.yaml (черновик, секция gui)
gui:
  enabled: true
  refresh_interval_sec: 30
  default_symbol: "BTC/USDT"
  default_timeframe: "1h"
  chart_max_bars: 2000
  chart_inference_window: 512
  journal_root: "./docs/backtest_journal"
```

В PyQt6: `QSettings("IST", "Desktop")` для пользовательских prefs.

---

## Roadmap

| Фаза | Содержание |
|------|------------|
| **0** ✅ | `gui/api/` — IstGuiClient |
| **1** ✅ | PyQt6: Overview + Chart + SymbolToolbar |
| **2** ✅ | Backtests + Jobs + Models + Settings |
| **3** ✅ | CLI jobs (prepare/train/report), regime overlay, fold equity chart |
| **4** ✅ | Execution tab — paper connect, one-bar step, positions/orders |
| **5** | Live + алерты (drawdown, disconnect) |

### Фаза 4 (реализовано)

| Функция | Где | API |
|---------|-----|-----|
| Paper connect / disconnect | Execution | `execution.create_paper_session()` |
| One-bar inference + order | Execution | `PaperExecutionSession.run_step()` |
| Positions & order history | Execution | `account_snapshot()`, `order_history()` |
| Auto-step timer | Execution | 30s / 60s |
| Emergency stop | Execution | `emergency_stop()` |

Paper: long-only (short closes long). Live mode — когда заданы `OKX_*` env (UI подсказка).

### Фаза 3 (реализовано)

| Функция | Где | API / worker |
|---------|-----|----------------|
| Regime overlay на графике | Chart | `inference.chart_payload()`, `ChartPayloadWorker` |
| Equity по фолдам WFO | Backtests | `backtests.fold_equity_curve()`, `FoldEquityChart` |
| CLI prepare / train / report | Jobs | `api.cli.*`, `CliProcessWorker` |

Запуск длительных команд — вкладка **Jobs**; лог в окне; по завершении обновляются символы и активная вкладка.

### MVP для защиты диплома (минимум UI)

1. **Overview** + `ExplainCard`  
2. **Chart** + regime overlay  
3. **Backtests** — таблица + детали одного run  
4. **Pipeline** — статическая схема + подсветка шага из `ExplainSnapshot`

---

## Тестирование

```powershell
# API без UI
py -3 -c "from gui.api import IstGuiClient; c=IstGuiClient(); print(c.symbols.list_symbols()[:3])"
```

План:

- unit-тесты API на synthetic paths (mock parquet — опционально);
- `pytest-qt` для виджетов после появления `app/`;
- smoke: открыть Overview при наличии `artifacts/.../LATEST.txt`.

---

## Связь с остальными README

| Модуль | Элемент GUI |
|--------|-------------|
| `orchestration/introspect.py` | Overview, Chart tooltip |
| `orchestration/symbol_pipeline.py` | Jobs |
| `backtesting/results_journal.py` | Backtests |
| `decision/README.md` | Decision stepper |
| `meta_learning/README.md` | Models / weights |
| `execution/README.md` | Orders (фаза 4) |
| `docs/DIPLOMA_SYSTEM_DOCUMENTATION.md` | Текст для главы «Реализация» |

---

## Запуск API из корня репозитория

```powershell
cd D:\IST
py -3 -c "from gui.api import IstGuiClient; api=IstGuiClient(); print(api.inference.explain('BTC/USDT','1h'))"
```

Требуется: parquet, artifact bundle для пары (см. `docs/SYMBOL_PIPELINE_TEST_PLAN.md`).
