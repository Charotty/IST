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

**GUI:** PyQt6, выбор пары, Overview/Chart/Models/Backtests/Jobs/Settings/Execution (paper). Jobs запускает `prepare-symbol`, `train-final-symbol`, `report-real` без флагов tuning-best и feature-cache; `tune-thesis` и `build-features` в GUI нет. Settings редактирует часть `orchestration_tuning_best` (margin, direction, ensemble, …), не все поля variant B.

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

### A. Метрики, acceptance и исследование качества

1. Зафиксировать эталонный прогон BTC variant B в JSON:  
   `python -m orchestration report-real --symbol BTC/USDT --use-tuning-best --use-feature-cache --json-out docs/thesis_btc_4model_acceptance.json`  
   Сохранить run_id из журнала в этот документ и в `THESIS_CONTEXT.md`.

2. Разобрать WFE: посчитать по фолдам (train vs OOS Sharpe), выявить фолды-выбросы; при необходимости варьировать `walk_forward_step`, `train_window_size`, `test_window_size`, `volatility_filter_percentile`, `min_signal_margin` только на confirm-фазе.

3. Прогнать `tune-thesis --phase all` для **ETH/USDT** 1h (скачать OHLCV при отсутствии), перенести лучший confirm в `config/symbols/ETH-USDT_1h.yaml`, JSON `docs/thesis_eth_4model_acceptance.json`.

4. Сравнить variant A/B и ablation (regime_adaptive vs fixed, 2 vs 4 модели) — `scripts/run_ablation.py` или узкий grid; результаты в `docs/reports/` для таблиц диплома.

5. Обновить пороги или режим проверки только если комиссия решит менять `config.yaml` → `backtesting.acceptance` (сейчас 8 проверок: Sharpe, PF, WFE, MDD, return, recovery, trades, min_folds); любое изменение — с обоснованием в тексте ВКР.

6. Target-критерии (`backtesting.target`, Sharpe > 1.0) — отдельный прогон quality; не смешивать с acceptance в отчётах.

7. После стабилизации params — `train-final-symbol` для BTC и ETH, проверить bundle `manifest.json` (4 model_keys, schema hash).

8. Inference smoke: `explain` / Settings «Тест inference» на обеих парах после сохранения YAML.

---

### B. CLI и orchestration (доработки поверх уже реализованного)

9. `prepare-symbol` — явно описать в доке как legacy относительно `tune-thesis`; опционально делегировать fast-trials в ту же логику shortlist.

10. `report-real` по умолчанию для GUI и скриптов: `--symbol`, `--use-tuning-best`, `--use-feature-cache`, `max_rows` из symbol YAML (0 = без обрезки по соглашению в коде).

11. Команда или флаг **acceptance-only** в выводе JSON: все 8 checks с фактическими значениями и порогами (удобно для GUI и диплома).

12. `build-features` — вызывать из GUI и из чеклиста Jobs после download OHLCV.

13. `tune-thesis` — кнопки/профили в GUI: fast, refine, confirm, all; лог в `artifacts/<slug>/active/`; по завершении confirm — предложение записать params в symbol YAML.

14. Shortlist `docs/thesis_tune_shortlist.json` — отображать в GUI (топ-K, mean_sharpe) на вкладке Jobs или Backtests.

15. Зафиксировать в symbol YAML поля variant B, которых нет в форме Settings: `volatility_filter_percentile`, `prediction_horizon`, `max_wfo_folds`, `dl_epochs`, `use_risk_bridge`, `tune_source`, `walk_forward_step`, `train_window_size`, `test_window_size`.

16. ETH: тот же `config/profiles/canonical_4model.yaml` и `config/profiles/thesis_tuning.yaml`; при необходимости слегка другие окна WFO под ликвидность.

17. Риск-мост и trailing: если включать `use_risk_bridge` / `apply_atr_trailing` — только после baseline PASS или в отдельной ветке ablation.

18. Feature cache: везде, где WFO/tune/report, единый флаг `--use-feature-cache`; инвалидация при смене OHLCV (manifest stale).

19. Журнал: каждый confirm-run с тегами `tune_level=confirm`, `symbol`, `model_keys`, `ensemble_mode`, `acceptance_passed`.

20. Репозиторий на быстром диске (`~/IST`) — рекомендация в README для длинных tune; не менять код.

---

### C. GUI — функциональная полнота и мульти-пара

21. **`gui/api/cli_api.py` — `report_real_cmd`:** добавить `--symbol`, `--use-tuning-best`, `--use-feature-cache`; `max_rows` из merged config или spinbox «0 = из YAML».

22. **`prepare_symbol_cmd`:** опции `max_trials`, профиль `canonical_4model` / `thesis_tuning`, галочка skip-final; отображать ожидаемое время.

23. **Jobs:** кнопки `build-features`, `tune-thesis` (фаза из комбобокса), прогресс и tail лога; после CLI — автообновление pipeline checklist.

24. **Settings:** поля `volatility_filter_percentile`, `prediction_horizon`, `max_wfo_folds`, `dl_epochs`, `model_keys` (read-only список 4), `use_risk_bridge`; загрузка/сохранение полного блока `orchestration_tuning_best` + YAML tab как сейчас.

25. **Settings → «Отчёт acceptance»:** запуск `report-real` с правильными флагами и вывод 8 checks в `_result`.

26. **SymbolToolbar:** индикаторы по паре — OHLCV ✓, features ✓, bundle ✓, last acceptance PASS/FAIL из последнего journal run.

27. **Backtests:** фильтр по symbol/timeframe; подсветка FAIL по имени критерия (в т.ч. WFE); экспорт equity PNG для слайдов.

28. **Overview / Chart:** при смене пары — перезагрузка explain и свечей; предупреждение, если bundle отсутствует.

29. **Models:** показ весов regime-adaptive и model_keys из manifest текущего bundle.

30. **Execution (paper):** сохранение сессии, несколько шагов подряд, сводка PnL; опционально тот же `min_signal_margin` / vol filter из config.

31. **Демо-режим:** оставить `--demo`; в прод-режиме все CLI-кнопки активны (как сейчас, без регрессии).

32. **Новая пара из GUI:** мастер «Добавить символ» — slug, download OKX, prepare/build-features, ссылка на tune-thesis (без обязательного wizard на один экран — можно пошаговые кнопки в Jobs).

33. **Импорт OHLCV:** file picker → копирование в `data/ohlcv/<slug>.parquet` + пересбор features (для не-OKX данных).

34. **Согласованность:** после Save в Settings — подсказка «запустите report-real / train-final».

35. Обновить `gui/README.md` — таблица «CLI ↔ GUI», статус фаз 3–4.

---

### D. Документация и материалы диплома

36. Переписать `docs/THESIS_CONTEXT.md`: variant B, 7/8 acceptance, команды tune-thesis/report-real, ссылка на acceleration plan как «реализовано».

37. Обновить `docs/THESIS_4MODEL_STEPS.md` под `tune-thesis` вместо только `thesis_push_4model`.

38. `docs/3_11_SYSTEM_TESTING.md` — таблицы с актуальными 4-model прогонами (BTC variant B, ETH когда готов).

39. `docs/BACKTESTING_CRITERIA_REFERENCE.md` — пример JSON с 8 полями checks.

40. Краткий `docs/GUI_USER_GUIDE.md` — сценарий защиты: выбор BTC → Overview → Backtests → Settings → paper step.

41. `THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md` — финальный статус DoD (ETH YAML, acceptance JSON, GUI parity).

42. Слайды / рисунки: схема pipeline, скрин GUI Overview + Backtests PASS/FAIL, таблица ablation.

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

### G. Paper trading и визуализация (продолжение текущей линии)

51. Paper: прогон N баров подряд с записью equity в journal или локальный CSV для графика в Execution.

52. Chart: overlay сигналов BUY/SELL/HOLD на хвосте; легенда regime.

53. Сравнение paper vs backtester на том же окне (уже есть PaperCompareWorker — расширить отчёт в UI).

54. Live OKX — вне scope диплома; в GUI оставить предупреждение и env-переменные без реализации ордеров.

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
