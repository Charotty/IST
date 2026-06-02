# Документация для ВКР (IST) — навигация

**Назначение:** честное описание системы **как реализовано в репозитории** (май 2026), без маркетинга. Для корректировки текста диплома, защиты и списка «что не обещать».

**Источники:** код (`orchestration/`, `meta_learning/`, `backtesting/`, …), `config/profiles/canonical_4model.yaml`, журнал `docs/backtest_journal/`, расширенный справочник `docs/DIPLOMA_SYSTEM_DOCUMENTATION.md`.

---

## Файлы в этой папке

| Файл | Темы (п. требований) |
|------|----------------------|
| [01-architecture-and-data.md](01-architecture-and-data.md) | **1** Архитектура · **9** Data pipeline · **16** Библиотеки |
| [02-ml-ensemble-decision-features.md](02-ml-ensemble-decision-features.md) | **2** Модели · **3** Ensemble · **4** Regime · **5** Adaptive weighting · **6** Confidence · **7** Trading · **8** Features |
| [03-validation-results-gaps-artifacts.md](03-validation-results-gaps-artifacts.md) | **10** Backtest · **11** WFO · **12** Результаты · **13** GUI · **14** Не реализовано · **15** Статусы · **17** Артефакты |

---

## Краткий вердикт (для текста ВКР)

| Вопрос | Ответ по коду |
|--------|----------------|
| 4 модели в прод-pipeline? | **Да:** `lgb`, `gru`, `xgb`, `cnn` — обучаются и inference через `TrainingOrchestrator` / `InferenceOrchestrator` |
| Transformer / LSTM в проде? | **Нет** — только в legacy `ist.py` и старых именах в `meta_learning/ensemble.py` |
| Ensemble — ядро? | **Да:** `DynamicMetaWeighting` + `DecisionPipeline` (`integrated`) |
| Веса «адаптивные»? | **Частично:** переключение **по режиму** (trend/range), веса **фиксированы в YAML**, не обучаются online по PnL |
| Regime detection? | **Да (rule-based):** `MomentumRegimeDetector` (SMA); ML `RegimeDetector` — **не** canonical prod |
| WFO с purge/embargo? | **Да:** `TrainingOrchestrator.walk_forward_backtest` + `DataLeakagePreventer` |
| Полноценный backtest? | **Да:** equity, Sharpe, Max DD, комиссия, slippage |
| Критерии приёмки WFO | Часто **FAIL** в журнале при положительном mean Sharpe (см. файл 03) |

---

## Статус компонентов (п. 15)

| Компонент | Статус | Комментарий |
|-----------|--------|-------------|
| 4-model training + WFO | **Full** | CLI `orchestration`, `symbol-pipeline` |
| Regime-adaptive ensemble | **Full** | `meta_learning/dynamic_meta.py` + YAML веса |
| Performance-based online weights | **Stub** | `WeightUtils.adaptive_weight_update` не в prod pipeline |
| Rule-based regime (SMA) | **Full** | `glue.MomentumRegimeDetector` |
| ML regime classifier | **Prototype** | `models/regime/regime_detector.py` |
| Decision + meta threshold | **Full** | `decision/`, train-only calibration OOS |
| Risk (ATR sizing, bridge) | **Full** | `risk_management/`, опционально trailing |
| Backtester + metrics + journal | **Full** | `backtesting/`, `docs/backtest_journal/` |
| Paper execution | **Partial** | `execution/paper_loop.py`, long-focused |
| Live OKX trading | **Not** | только загрузка данных / sandbox API |
| GUI PyQt6 | **Full** (desktop) | `gui/app/`, не mockup |
| Microstructure live LOB | **Not** | `mode: live` → `NotImplementedError` |
| RL direction / Transformer prod | **Legacy only** | `ist.py`, Colab-наследие |

---

## Запуск справочных команд

```powershell
# WFO smoke / отчёт
py -3 -m orchestration from-parquet data/features/BTC-USDT_1h.parquet
py -3 -m orchestration report-real --symbol BTC-USDT --timeframe 1h

# GUI
pip install -r requirements-gui.txt
py -3 -m gui.app
```

См. также: `gui/README.md`, `docs/PROJECT_CLI_AND_GUI_COMMANDS.md`, `docs/GUI_USER_GUIDE.md`, `docs/GUI_LAYOUT_REFERENCE.md`.

**Слойные README** (`data_layer/`, `orchestration/`, `meta_learning/`, …) обновлены под canonical pipeline (май 2026); при расхождении приоритет у этого комплекта и кода.
