# Walk-Forward Efficiency (WFE) — разбор для BTC variant B

## Определение в IST

В `orchestration/training_orchestrator.py` на каждом фолде WFO:

```text
IS_ann  = годовая доходность бэктеста на train-окне (1500 баров)
OOS_ann = годовая доходность бэктеста на test-окне (170 баров)
WFE     = OOS_ann / IS_ann   (если IS_ann ≠ 0)
```

Acceptance: **mean(WFE) ≥ 0.5** по всем фолдам (`config.yaml` → `backtesting.acceptance.min_wfe`).

Источник порога: [Quant Trading Tools — WFA](https://quanttradingtools.com/walk-forward-analysis/), см. `docs/BACKTESTING_CRITERIA_REFERENCE.md` §4.1.

---

## Факт по `docs/thesis_btc_4model_acceptance.json` (18 фолдов)

| Показатель | Значение |
|------------|----------|
| **mean WFE** | **0.206** (порог 0.5 → **FAIL**) |
| median WFE | 0.075 |
| Фолдов WFE ≥ 0.5 | **3** из 18 |
| Фолдов WFE &lt; 0 | **3** (фолды 1, 3, 18) |
| **mean OOS Sharpe** | **2.48** (порог 0.5 → **PASS**) |
| **mean IS Sharpe** | **8.40** |
| mean OOS Sharpe / mean IS Sharpe | ~0.24 |
| corr(WFE, OOS Sharpe) | ~0.61 |

**Вывод:** OOS по Sharpe, PF, recovery и просадке **сильные**, но WFE низкий не потому, что OOS «плохой», а потому что **IS на train-окне завышен** относительно OOS.

---

## Почему IS завышен

На каждом фолде:

1. Модели **обучаются** на `train_features` (1500 баров).
2. На тех же данных считается `run_pipeline` → сигналы → **бэктест train** → метрики `IS_*`.
3. На `test_features` (170 баров) — переобученные модели дают OOS-бэктест.

Бэктест на train — это **in-sample симуляция** после подгонки на тех же барах. Поэтому типичны:

- `IS_Total Return` **+20…+50%** на окне,
- `IS_Sharpe` **5…10**,
- `IS_CAGR` в сотнях процентов (артефакт годовизации короткого окна).

OOS на 170 барах чаще скромнее или отрицателен на отдельных сегментах рынка → **WFE = OOS_ann / IS_ann** становится малым или отрицательным, даже когда **OOS Sharpe на том же фолде высокий** (на других фолдах).

### Худшие фолды по WFE

| Fold | WFE | OOS Sharpe | IS Sharpe | OOS Return % | IS Return % |
|------|-----|------------|-------------|--------------|-------------|
| 3 | −0.49 | −12.5 | 5.8 | −5.1 | +19.8 |
| 1 | −0.25 | −6.0 | 7.2 | −2.5 | +25.9 |
| 18 | −0.04 | −2.5 | 9.8 | −1.1 | +49.5 |

### Лучшие фолды по WFE

| Fold | WFE | OOS Sharpe | IS Sharpe |
|------|-----|------------|-----------|
| 13 | **1.53** | 5.23 | 7.89 |
| 8 | **1.17** | 8.30 | 8.82 |
| 9 | **0.59** | 6.49 | 7.10 |

---

## Не путать с «плохим» OOS

| Критерий acceptance | Результат |
|---------------------|-----------|
| min_oos_sharpe | PASS (2.48) |
| min_oos_profit_factor | PASS (1.78) |
| min_recovery_factor | PASS (1.26) |
| max_drawdown_pct | PASS |
| min_wfe | **FAIL** (0.21) |

Для текста ВКР: **7/8** — единственный формальный провал из-за определения WFE через отношение **годовых** доходностей при **разной длине** окон и **in-sample** train-бэктесте.

---

## Что можно сделать (по приоритету)

### 1. Диплом / интерпретация (без смены кода)

- Привести таблицу фолдов (WFE, OOS/IS Sharpe, return).
- Объяснить расхождение: **высокий OOS Sharpe при низком WFE** из-за IS inflation.
- Сослаться на остальные 7 критериев и стабильность PF/DD.

### 2. Диагностика

```bash
python scripts/analyze_wfe.py docs/thesis_btc_4model_acceptance.json
```

### 3. Улучшить OOS на слабых фолдах (честный путь к WFE ≥ 0.5)

- Увеличить `test_window_size` (больше OOS баров, меньше шум годовизации).
- Сузить `train_window_size` или увеличить `walk_forward_step` (меньше переобучения на train).
- Донастроить `volatility_filter_percentile` / `min_signal_margin` только на confirm.
- Повторить `tune-thesis` с целевой метрикой **median WFE** или **mean WFE** в shortlist (доработка `thesis_tuning.py`).

### 4. Уточнить метрику в коде (если комиссия согласует)

Дополнительно писать в отчёт:

- `Walk-Forward Efficiency (Sharpe)` = `Sharpe_OOS / Sharpe_IS` (при `Sharpe_IS > ε`),
- `WFE length-neutral` = `(OOS_return × train_bars) / (IS_return × test_bars)`.

**Не заменяет** текущий WFE в acceptance без изменения `config.yaml` и обоснования в ВКР.

### 5. Порог acceptance

Снизить `min_wfe` для профиля крипто 1h (например 0.2) — только с явным обоснованием в главе 3.11; иначе лучше п.1 + п.3.

---

## Рекомендация для IST

1. **Сейчас:** зафиксировать анализ (этот файл + JSON) и использовать **7/8** в дипломе с пояснением WFE.  
2. **Не менять** формулу WFE в acceptance без согласования с критериями из `BACKTESTING_CRITERIA_REFERENCE.md`.

---

## Эксперимент: `test_window_size` 280 (май 2026)

Override в `config/symbols/BTC-USDT_1h.yaml` (`wfe_exp_test280`), остальные params — эталон variant B.  
Прогон: `report-real --use-tuning-best --use-feature-cache`.

| Метрика | Baseline (test **170**) | Эксперимент (test **280**) |
|---------|-------------------------|----------------------------|
| mean OOS Sharpe | **2.48** | 0.86 |
| mean WFE | 0.21 | **0.03** |
| mean PF | **1.78** | 1.31 |
| mean return % | **+0.74** | −0.09 |
| mean recovery | **1.26** | 0.93 |
| folds return &gt; 0 | **13/18** | 10/18 |
| acceptance | **7/8** (только WFE) | **5/8** (+ recovery, return) |

**Вывод:** увеличение OOS-окна **не** подняло WFE и **ухудшило** OOS Sharpe/return. На длинном test (280 бар) стратегия чаще попадает в неблагоприятные участки (фолд 11: −11.1% OOS, 42 сделки; фолд 18: −3.7%). Короткое окно 170 бар «усредняло» сильные микро-OOS сегменты (высокий mean Sharpe), но завышало WFE-иллюзию через IS — компромисс в пользу **baseline 170** для диплома.

Результат **сохранён**: `docs/thesis_btc_wfe_exp_t280.json`, журнал `docs/experiments/WFE_EXPERIMENTS_LOG.md`.

### Эксперимент 2: `test_window_size` **140**

JSON: `docs/thesis_btc_wfe_exp_t140.json`.

| Метрика | Baseline 170 | t140 |
|---------|--------------|------|
| mean OOS Sharpe | **2.48** | 1.49 |
| mean WFE | 0.21 | **0.59 (PASS)** |
| median WFE | ~0.07 | **~0** |
| mean PF | 1.78 | 1.74 |
| mean return % | **0.74** | 0.40 |
| recovery | **1.26** | 0.98 (FAIL) |
| acceptance | 7/8 | **7/8** (recovery) |

**PASS по WFE** достигнут, но:

1. Mean WFE завышен **выбросами** на фолдах 8 и 13 (WFE > 3, CAGR > 1500% на 140 барах — артефакт `OOS_ann / IS_ann`).
2. **Median WFE ≈ 0** — типичный фолд по-прежнему «слабый перенос» по доходности.
3. Mean Sharpe **ниже**, чем на 170 барах — для основного claim диплома **baseline 170 остаётся главным**.

**Практика:** в тексте ВКР — основной ряд **170**; t140 — таблица чувствительности «при укороченном OOS окне формальный порог WFE выполняется, median — нет».

BTC: в `config/symbols/BTC-USDT_1h.yaml` только `tune_source: variant_b_vol_filter_90` (без `test_window_size` — значение 170 из эталона).
