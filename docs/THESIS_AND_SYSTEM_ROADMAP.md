# Диплом и система IST: единый план изменений

Живой документ для правки перед внедрением. Описывает **текущее состояние**, **целевое состояние** и **все запланированные изменения** в одном контуре: метрики и acceptance, мульти-пара, CLI, GUI, артефакты диплома, документация.

Связанные файлы (уже есть, обновлять по мере выполнения пунктов ниже):

- `config/symbols/BTC-USDT_1h.yaml` — лучшие params (variant B: margin 0.06, vol filter 90)
- `docs/THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md` — реализованный блок ускорения (§1–§4)
- `docs/THESIS_CONTEXT.md` — контекст диплома (сейчас частично устарел)
- `docs/THESIS_4MODEL_STEPS.md` — пошаговые CLI-команды
- `gui/README.md` — архитектура GUI

---

## Текущее состояние (снимок)

**Ядро:** четыре модели (`lgb`, `gru`, `xgb`, `cnn`), regime-adaptive ensemble, decision pipeline, WFO, журнал `docs/backtest_journal/`, кэш фичей, `tune-thesis` (fast → refine → confirm), `report-real` с `--use-tuning-best` и `--use-feature-cache`.

**BTC/USDT 1h (variant B, confirm):** ~18 фолдов, mean Sharpe ~2.48, PF ~1.78, recovery ~1.26, return ~0.74%, ~381 сделок; **acceptance 7/8** — не проходит только `min_wfe` (0.21 vs 0.5). Формально `acceptance_passed: false` в JSON, по смыслу метрики сильные, WFE — точка для диплома (доработка или обоснование).

**Мульти-пара:** инфраструктура (`data/ohlcv/`, `config/symbols/`, `paths_for`, GUI toolbar) готова; **ETH** и второй JSON acceptance не прогонялись с variant B.

**GUI:** PyQt6 (`python -m gui.app`): вкладки **График** (chart + Решение/Bundle), **Режим**, **Задачи** (Pipeline / Журнал WFO / Практика), **Конфигурация**. Jobs: `prepare-symbol`, `build-features`, `tune-thesis`, `train-final-symbol`, `report-real` с чекбоксами `tuning-best` / feature-cache (`jobs_view.py`). Paper + сверка с Backtester — «Практика». См. `gui/README.md`, `docs/GUI_LAYOUT_REFERENCE.md`.

**Дипломные артефакты:** `THESIS_CONTEXT.md` описывает старые цифры (~0.33 Sharpe); нужна синхронизация с variant B и acceleration plan.

---

## Целевое состояние

Система, которую можно показать на защите и использовать дальше:

1. **Воспроизводимый контур** на BTC и ≥1 второй паре (ETH): одинаковая архитектура, отдельный symbol YAML, журнал и JSON отчёты.
2. **Acceptance** либо формально `true` на 4 моделях (все 8 проверок), либо зафиксированный отчёт с явным разбором WFE и остальных метрик в тексте ВКР.
3. **GUI** отражает тот же контур, что CLI: подготовка пары, тюнинг/отчёт с теми же флагами, правка params, визуализация, paper — без расхождения «в терминале PASS, в окне другие цифры».
4. **Документация** (контекст, глава 3.x, шаги) совпадает с последними run_id и YAML.

---

## План изменений (единый список)

### A. Эталон метрик и конфигурация (reference + overrides)

**Принцип:** лучший достигнутый результат живёт в **одном** файле `config/reference/thesis_4model_reference.yaml`. Пары не дублируют полный блок — только `baseline_ref` + `orchestration_overrides` (отличия). Эталон меняется **один раз** при новом лучшем confirm; пары настраиваются **отдельно** поверх него.

| Статус | Пункт |
|--------|--------|
| [x] | Эталон variant B в `config/reference/thesis_4model_reference.yaml` |
| [x] | BTC: `baseline_ref` + override `tune_source` |
| [x] | ETH: `baseline_ref` + overrides (horizon, margin, …) до своего confirm |
| [x] | Merge в `orchestration/symbols.py` (`resolve_tuning_best_block`, `write_symbol_config`) |
| [x] | Документ [`THESIS_REFERENCE.md`](THESIS_REFERENCE.md) |
| [x] | JSON `docs/thesis_btc_4model_acceptance.json` (7/8 acceptance) |
| [ ] | ETH: `tune-thesis --phase all` → overrides + `docs/thesis_eth_4model_acceptance.json` |
| [ ] | WFE: разбор по фолдам; улучшение или обоснование в ВКР (единственный FAIL) |
| [ ] | Ablation / variant A–B в `docs/reports/` для таблиц диплома |
| [ ] | `train-final-symbol` BTC+ETH, smoke inference |
| [ ] | Обновление эталона только при новом лучшем confirm (процедура в THESIS_REFERENCE) |

---

### B. CLI и orchestration

| Статус | Пункт |
|--------|--------|
| [x] | `tune-thesis`, `build-features`, feature cache |
| [x] | `report-real`: `--symbol`, `--use-tuning-best`, `--use-feature-cache`, `max_rows=0` → YAML |
| [x] | `tuning_best_for` через reference + overrides |
| [ ] | `prepare-symbol` как legacy в доке; опционально делегирование в tune-thesis |
| [ ] | Shortlist в GUI (Jobs/Backtests) |
| [ ] | ETH tune confirm → overrides only |

---

### C. GUI — мульти-пара и CLI

| Статус | Пункт |
|--------|--------|
| [x] | `cli_api`: report-real, build-features, tune-thesis |
| [x] | Jobs: признаки, тюнинг, флаги tuning-best / cache, баров 0=YAML |
| [x] | Settings: vol filter, horizon, dl_epochs, эталон, отчёт acceptance |
| [x] | SymbolToolbar: OHLCV/feat/bundle/acc |
| [ ] | Backtests: фильтр symbol, подсветка FAIL-критерия |
| [ ] | Импорт OHLCV / мастер новой пары |
| [x] | `gui/README.md` + [`GUI_USER_GUIDE.md`](GUI_USER_GUIDE.md) |

---

### D. Документация

| Статус | Пункт |
|--------|--------|
| [x] | `THESIS_CONTEXT.md`, `THESIS_REFERENCE.md`, `GUI_USER_GUIDE.md` |
| [ ] | `THESIS_4MODEL_STEPS.md` — reference + overrides |
| [ ] | Разделы `docs/thesis/3_*` (см. `docs/thesis/README.md`) — BTC variant B, ETH |
| [ ] | Слайды / figures GUI |

---

### E. Тесты и надёжность

43. Тест `gui/api/cli_api.py` — аргументы `report_real_cmd` содержат tuning-best при флаге.

44. Интеграционный smoke: mock subprocess или один короткий `report-real --max-rows 500` в CI (опционально, долгий — skip в CI).

45. Регрессия `tests/test_thesis_tuning.py`, `tests/test_feature_store.py` после изменений merge params.

46. Проверка CNN float32 / mixed precision между GRU и CNN после любых правок `dl_training.py`.

---

### F. Модели и пайплайн (точечные улучшения при необходимости)

47. Третья «проблемная» модель из четырёх — отдельный короткий trial только с одним `model_key` для диагностики (если снова деградация ансамбля).

48. Meta threshold `percentile` vs fixed — один confirm A/B на BTC.

49. `ensemble_mode`: regime_weighted vs regime_adaptive — одна строка в ablation.

50. Commission/slippage в `config.yaml` — sensitivity один прогон (уже заданы 0.0006 / 0.0002).

---

### G. Paper и визуализация

| Статус | Пункт |
|--------|--------|
| [x] | Execution: серия шагов, график баланса, сверка Backtester |
| [x] | Chart: подсказка про сигналы в Обзоре/Исполнении |
| [ ] | Overlay сигналов на свечах (локальные features) |
| [ ] | Экспорт equity paper в CSV |

---

## Рекомендуемый порядок внедрения (без разделения на «обязательно/можно»)

Логическая цепочка при правке системы:

```text
Зафиксировать BTC JSON (1–2) → ETH tune + YAML (3) → CLI/report флаги (9–11, 15–19)
    → GUI cli_api + Jobs + Settings (21–25, 34) → THESIS_CONTEXT + шаги (36–41)
    → SymbolToolbar/Backtests polish (26–29) → ablation/диплом таблицы (4, 38, 42)
    → WFE исследование параллельно с (2) → paper/chart (51–53) по желанию после стабилизации
```

---

## Команды-якоря (копировать при работе)

```bash
# Кэш фичей
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h

# Многоуровневый тюнинг
python -m orchestration tune-thesis --symbol BTC/USDT --phase all
python -m orchestration tune-thesis --symbol ETH/USDT --phase all

# Эталонный отчёт (диплом + GUI должен вызывать то же)
python -m orchestration report-real --symbol BTC/USDT --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_btc_4model_acceptance.json

# Финальный bundle
python -m orchestration train-final-symbol --symbol BTC/USDT --timeframe 1h --force

# GUI
py -3 -m gui.app
```

---

## Критерии «система готова к защите» (чеклист для самопроверки)

- [ ] `docs/thesis_btc_4model_acceptance.json` соответствует variant B YAML и последнему confirm run_id  
- [ ] `docs/thesis_eth_4model_acceptance.json` существует и методика описана в тексте  
- [ ] Acceptance: 8/8 или явный подраздел ВКР по WFE с цифрами 7/8  
- [ ] GUI report-real даёт те же флаги, что команда выше (нет расхождения фолдов/params)  
- [ ] Settings сохраняет vol filter и margin; inference OK на BTC и ETH  
- [ ] `THESIS_CONTEXT.md` не противоречит журналу и YAML  
- [ ] Демо на защите: переключение пары → Overview → Backtests → paper один шаг  

---

## Заметки для правки

*(Добавляйте сюда свои правки перед реализацией — приоритеты, отказ от пунктов, новые пары, пороги acceptance.)*

- 
- 
- 

---

*Версия: 2026-05-22. Обновлять по мере закрытия пунктов (можно помечать `[x]` в чеклисте и номерах списка).*
