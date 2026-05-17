# План действий: достижение стратегий (а) и (б)

Документ описывает **цели**, **текущий статус IST**, **выводы из внешних источников** (браузер / литература) и **конкретные шаги**, которые выполняются в репозитории для прохождения критериев на **реальных** OHLCV (`data/ohlcv/BTC-USDT_1h.parquet`).

Связанные файлы:

| Документ | Назначение |
|----------|------------|
| [BACKTESTING_CRITERIA_REFERENCE.md](./BACKTESTING_CRITERIA_REFERENCE.md) | Пороги метрик, справочник |
| [backtest_journal/README.md](./backtest_journal/README.md) | Журнал прогонов (`runs.jsonl`, `INDEX.md`) |
| [TEST_RESULTS_LOG.md](./TEST_RESULTS_LOG.md) | Pytest / CI |
| `config.yaml` → `backtesting.acceptance` / `backtesting.target` | Автопроверка |

---

## 1. Цели

### (а) Acceptable — «на этих данных можно торговать»

Доказать **устойчивое OOS-преимущество** после издержек, не случайность на выборке.

| Проверка | Порог (IST, часовой Sharpe `√8760`) |
|----------|-------------------------------------|
| OOS Profit Factor (mean) | ≥ 1.2 |
| OOS Sharpe (mean) | **> 0.5** |
| Walk-Forward Efficiency (mean) | ≥ 0.5 |
| Max Drawdown (worst fold) | ≥ −35% |
| Recovery Factor (mean) | ≥ 1.0 |
| Mean Total Return % | > 0 |
| Trade events (sum) | ≥ 30 |
| WFO folds | ≥ 5 |

### (б) Target — «хороший результат»

| Проверка | Порог |
|----------|-------|
| OOS PF | ≥ 1.6 |
| OOS Sharpe | **> 1.0** |
| WFE | ≥ 0.7 |
| Max DD | ≥ −20% |
| Recovery | ≥ 2.0 |
| Folds | ≥ 8 |
| Trades | ≥ 100 |
| Calmar / Sortino | ≥ 1.0 / ≥ 0.5 |

> **Важно:** внешние статьи часто годализуют Sharpe через `√252` (дневные бары). В IST — **`√8760` (1h)**. Сравнивать абсолютные числа с дневными публикациями напрямую нельзя; сравнивать только внутри одного пайплайна.

---

## 2. Текущий статус (снимок на 2026-05-16)

### Инфраструктура — готова

- Расширенные метрики: `backtesting/performance_metrics.py` (Sharpe, Sortino, Calmar, CAGR, WFE, IS/OOS, Ulcer, alpha vs benchmark).
- Критерии: `backtesting/criteria_evaluator.py` + `config.yaml`.
- Журнал: `docs/backtest_journal/` (~410 прогонов), CLI `report-real`, `tune-until`.
- Leakage-safe WFO: `TrainingOrchestrator.walk_forward_backtest`, purge/embargo.

### Лучший прогон на BTC 1h (реальные данные, LGB+XGB)

Параметры (см. `config.yaml` → `orchestration_tuning_best`), ~8000 баров, 18 OOS-фолдов:

| Метрика | Значение | (а) |
|---------|----------|-----|
| mean Sharpe | **0.36** | **FAIL** (нужно > 0.5) |
| mean PF | **2.11** | OK |
| mean WFE | **0.63** | OK |
| mean return | **+0.18%** | OK |
| worst Max DD | **−11.2%** | OK |
| recovery | **1.37** | OK |
| trade events | **254** | OK |

**Итого: 7/8 по (а).** Ни один из ~410 журнальных прогонов не получил `acceptance_passed: true`. Цель (б) не достигнута.

### Что не сработало

- Слепой grid **240+** вариантов (`sharpe_hunt`) на 10k барах — ухудшил метрики; selection bias на уровне «перебора до зелёного».
- Поднятие Sharpe только порогами сигнала без улучшения edge — PF/WFE падают.

---

## 3. Внешние источники: как добиваются устойчивых результатов

Ниже — выжимка из просмотренных в браузере материалов и исследований. Это **не** оправдание занижать пороги, а ориентиры для **правильного** пути.

### 3.1 Risk-adjusted return и реалистичные ожидания

**[Breaking Alpha — Understanding Sharpe Ratios](https://breakingalpha.io/insights/understanding-sharpe-ratios-selecting-trading-algorithms)**

| Sharpe (годовой, контекст статьи) | Интерпретация |
|-----------------------------------|---------------|
| < 0.5 | Слабо |
| 0.5 – 1.0 | Приемлемо / средний рынок |
| 1.0 – 2.0 | Хорошо (типичная цель systematic) |
| 2.0 – 3.0 | Очень хорошо / институциональный порог |
| > 3.0 на бэктесте | Подозрение на overfit / скрытые риски |

- Бэктест → лайв: закладывать **haircut 30–50%** к Sharpe.
- Для отбора бэктеста без лайва часто требуют **Sharpe 2.0–2.5+** до haircut.
- Смотреть **несколько** метрик (Sortino, Calmar, drawdown), не один Sharpe.

**Для IST (а):** порог Sharpe **> 0.5** на OOS WFO — согласован с «acceptable»; наш **0.36** — объективно близко, но формально недостаточно.

### 3.2 Walk-forward и WFE

**[Quant Trading Tools — Walk-Forward Analysis](https://quanttradingtools.com/walk-forward-analysis/)**

- **5–10** OOS-окон — стандарт; <5 шумно, >15 — короткий IS.
- Соотношение **70/30 или 80/20** (IS/OOS).
- **Rolling** WFO предпочтительнее anchored.
- **Walk-Forward Efficiency** = OOS performance / IS performance; **≥ 0.5** acceptable, **≥ 0.7** strong.
- Оптимизировать не «голый net profit», а **return / max DD** или risk-adjusted objective.
- Не перезапускать WFO, пока не «пройдёт» — meta-level curve fitting.

**Для IST:** WFE **0.63** уже на уровне (а); при доработке Sharpe сохранять WFE ≥ 0.5.

### 3.3 ML / AI бэктест

**[Technical Analysis Pro — AI Backtesting & WFO (2026)](https://www.technical-analysis-pro.com/strategies-ai-backtesting-walk-forward-model-validation/)**

Пять ловушек:

1. Overfitting (train >> test accuracy).
2. Data leakage (нормализация на всём ряду, shuffle времени).
3. Survivorship bias.
4. Заниженные издержки.
5. Selection bias (100 моделей → лучшая случайно).

Практики:

- **Purged CV + embargo** между train и test (у IST: `embargo_period`, `enable_purge`).
- **Deflated Sharpe Ratio (DSR)** и учёт числа trials ([Bailey & López de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
- Sharpe **> 3** на бэктесте AI — «почти наверняка overfit»; реалистичный диапазон **1.0–2.5** у сильных фондов.
- Monte Carlo по сделкам — дополнение к WFO, не замена.

### 3.4 Оптимизация параметров без переобучения

**[Quanthop — Parameter Optimization Without Overfitting](https://quanthop.com/learn/backtesting-optimization/parameter-optimization)**

- Каждый новый параметр **×10** к пространству поиска → рост false discovery.
- Искать **плато устойчивости**, а не один пик на heatmap.
- Фиксировать правила **ex-ante**, затем один WFO-прогон; не подгонять правила под историю.
- Финальный **holdout** период, на котором параметры **не** подбирались.

**Для IST:** `tune-until` логировать в журнал и ограничивать число «официальных» конфигураций; лучший конфиг уже в `orchestration_tuning_best`.

### 3.5 Альтернативы классическому WFO

**[SSRN — Backtest Overfitting in the ML Era](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376)** — CPCV / purged CV иногда лучше предотвращают overfitting, чем один WFO.

**Для IST (фаза 4):** рассмотреть CPCV или combinatorial purged splits поверх текущего `DataLeakagePreventer`.

---

## 4. Диагностика: почему Sharpe «узкое место» при хорошем PF

| Наблюдение | Интерпретация |
|------------|---------------|
| PF ≈ 2.1, return ≈ +0.18% | Edge есть, но **мало съеденной доходности** на фолд |
| Sharpe 0.36 при положительном return | Высокая **часовая волатильность** equity / «шумные» баровые returns |
| 254 trade events | Достаточно сделок; проблема не в sample size |
| std Sharpe across folds ~6.8 | Нестабильность по режимам; один фолд тянет среднее |

**Вывод:** нужно одновременно (1) **снизить шум** returns (реже/увереннее входы, sizing), (2) **улучшить предиктивную силу** моделей/фич, (3) не раздувать IS относительно OOS (WFE).

---

## 5. План действий по фазам

Каждая фаза заканчивается **записью в журнал** и проверкой `acceptance` / `target` через CLI.

```powershell
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 0 --json-out docs/e2e_last_metrics.json
```

(`--max-rows 0` — весь доступный ряд после `FeatureEngine`.)

---

### Фаза 0 — Зафиксировано (выполнено)

| # | Действие | Статус |
|---|----------|--------|
| 0.1 | Метрики Sortino, Calmar, CAGR, WFE, IS_*, benchmark alpha | Done |
| 0.2 | `backtesting.acceptance` / `target` в `config.yaml` | Done |
| 0.3 | `BacktestResultsJournal` + `INDEX.md` | Done |
| 0.4 | `tune-until`, `report-real` | Done |
| 0.5 | Базовый подбор LGB+XGB (momentum regime, decision pipeline) | Done |
| 0.6 | Документ критериев `BACKTESTING_CRITERIA_REFERENCE.md` | Done |

---

### Фаза 1 — Стабилизация Sharpe без подгонки (текущий приоритет)

**Цель:** закрыть **единственный FAIL** по (а): `mean_oos_sharpe > 0.5`, не ломая PF ≥ 1.2 и WFE ≥ 0.5.

| # | Действие | Обоснование (источники) | Где в коде |
|---|----------|-------------------------|------------|
| 1.1 | Зафиксировать **лучший конфиг** как baseline; дальнейший поиск — **локальный** (±2–3 параметра), не grid 200+ | Quanthop: плато, не пик | `orchestration/tuning_loop.py` → режим `refine_baseline` |
| 1.2 | Снизить частоту сделок: `min_signal_margin`, `direction_threshold`, **volatility filter** (не торговать при ATR > p90 train) | Меньше шума → выше Sharpe | `OrchestratorConfig`, `training_orchestrator.py` |
| 1.3 | Целевая метка: `label_min_return` 0.3–0.5% за horizon | Меньше «случайных» up-labels | `glue.default_horizon_labels` |
| 1.4 | `signal_strategy=momentum_confirm` + ML-фильтр | Тренд + prob (структурный edge) | уже есть, расширить тесты |
| 1.5 | ATR sizing: cap `position_size` max fraction | Снижение vol of returns | `OrchestratorRiskBridge` / config cap |
| 1.6 | WFO objective: логировать **return/MDD** по фолдам; подобрать train/test/step под **8+** стабильных фолдов | Quant Trading Tools | `config.yaml` `walk_forward` |
| 1.7 | Прогон на **полной** истории (~31k баров) | Min track record / стабильность | `report-real --max-rows 0` |
| 1.8 | Критерий остановки: `acceptance_passed` в журнале | — | `tune-until` с лимитом trials |

**Критерий успеха фазы 1:** `acceptance_passed: true` в `docs/backtest_journal/runs.jsonl`.

---

### Фаза 2 — Все модели оркестратора (a) на полном контуре

**Цель:** (а) не только LGB+XGB, а `model_keys` из config: **lgb, gru, xgb, cnn**.

| # | Действие | Обоснование | Где |
|---|----------|-------------|-----|
| 2.1 | `from-parquet --full-models` на BTC 1h с реалистичными `dl_epochs` | Regime-adaptive ensemble | `orchestration/glue.py` |
| 2.2 | Обучить **RegimeDetector** вместо `SimpleHourlyRegimeStub` / `MomentumRegimeDetector` | Breaking Alpha: контекст режима | `models/regime/` |
| 2.3 | Веса meta по режимам из `config.yaml` (trend/range) | Уже в `DynamicMetaWeighting` | `meta_learning/dynamic_meta.py` |
| 2.4 | Отдельные строки в журнале **per model** + integrated | Сравнение вкладов | расширить `run_benchmark_report` |
| 2.5 | Провал модели → отключение в ensemble (gating по OOS Sharpe фолда) | Не тащить слабый сигнал | `signal_assembler` / orchestrator |

**Критерий успеха:** integrated pipeline `acceptance_passed` на полном parquet.

---

### Фаза 3 — Статистическая валидность (защита от selection bias)

| # | Действие | Источник |
|---|----------|----------|
| 3.1 | **Deflated Sharpe Ratio** + число trials из журнала | Bailey & López de Prado |
| 3.2 | **PBO** / Min Backtest Length в отчёт | SSRN 2308682 |
| 3.3 | Monte Carlo (bootstrap returns) — P(loss), 5/95 pct | Technical Analysis Pro |
| 3.4 | Запрет «официального» конфига без DSR p-value < 0.05 | Best practice |

Реализация: `backtesting/statistical_validation.py` + секция в JSON журнала.

---

### Фаза 4 — Цель (б) «хороший результат»

После стабильного (а):

| # | Действие | Порог (б) |
|---|----------|-----------|
| 4.1 | Ужесточить только после 2+ полных прогонов с (a) на разных срезах времени | — |
| 4.2 | Sharpe OOS > 1.0, WFE > 0.7, recovery > 2 | `backtesting.target` |
| 4.3 | RL layer (`rl_layer/integration.py`) — dynamic sizing, не direction | Снижение DD, рост Calmar |
| 4.4 | Multi-asset: ETH-USDT и др. | Робастность |
| 4.5 | Paper loop / execution costs audit | Trap 4 (costs) |

---

### Фаза 5 — Продакшен-готовность

| # | Действие |
|---|----------|
| 5.1 | Зафиксировать артефакт: `artifact_bundle` после train на полной истории |
| 5.2 | Inference smoke на хвосте parquet |
| 5.3 | Обновить `TEST_RESULTS_LOG.md` только после (a); target — отдельная секция |
| 5.4 | Не понижать пороги в YAML без записи в журнале (причина + ссылка на DSR) |

---

## 6. Что будет делаться автоматически (агент / CI)

| Ритуал | Команда | Артефакт |
|--------|---------|----------|
| Одиночный отчёт | `py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 0` | журнал + опционально `e2e_last_metrics.json` |
| Локальный подбор | `py -3 -m orchestration tune-until --max-rows 8000 --max-trials 40` | `runs.jsonl`, `INDEX.md` |
| Регрессия метрик | `pytest tests/test_performance_metrics.py tests/test_backtesting_criteria.py tests/test_orchestration_real_models.py -m integration` | pytest |
| После (a) | `--full-models`, фаза 2 | новые run_id в журнале |

**Не делать:**

- Массовый grid сотен комбинаций без учёта trials (selection bias).
- Снижать `min_oos_sharpe` в config «чтобы прошло».
- Считать успехом один фолд с огромным PnL (проверять `folds_positive_return` / std Sharpe).

---

## 7. Матрица «метрика → рычаг в IST»

| Метрика | Рычаги |
|---------|--------|
| **Sharpe** | Реже сделки, cap sizing, лучшие фичи/модели, longer horizon, vol filter, long_only в бычьих режимах |
| **Profit Factor** | Пороги meta/direction, calibration prob, costs model |
| **WFE** | Меньше переобучения IS (regularization LGB/XGB), короче train vs test mismatch, проще модель |
| **Max DD** | Risk pipeline, RL sizing, max leverage |
| **Recovery** | Суммарный return vs depth DD |
| **# trades** | Пороги, decision pipeline, margin |

---

## 8. Ориентиры успеха (чеклист)

### Стратегия (а)

- [ ] `acceptance_passed: true` на `BTC-USDT_1h` full history
- [ ] ≥ 5 OOS folds, без доминирования одного фолда (>70% PnL)
- [ ] DSR / trials учтены (фаза 3)
- [ ] Запись в `docs/backtest_journal/` с label `acceptance_achieved`

### Стратегия (б)

- [ ] `target_passed: true` на том же пайплайне
- [ ] Повтор на втором активе или срезе времени
- [ ] GRU/CNN в ensemble без деградации (a)

---

## 9. Ссылки

| Тема | URL |
|------|-----|
| Sharpe & institutional thresholds | https://breakingalpha.io/insights/understanding-sharpe-ratios-selecting-trading-algorithms |
| Walk-forward & WFE | https://quanttradingtools.com/walk-forward-analysis/ |
| AI backtesting traps & DSR | https://www.technical-analysis-pro.com/strategies-ai-backtesting-walk-forward-model-validation/ |
| Parameter optimization | https://quanthop.com/learn/backtesting-optimization/parameter-optimization |
| Deflated Sharpe Ratio | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 |
| Backtest overfitting (ML era) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376 |
| What to look for in a backtest | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308682 |
| Метрики (внутренний справочник) | [BACKTESTING_CRITERIA_REFERENCE.md](./BACKTESTING_CRITERIA_REFERENCE.md) |

---

*Обновлять этот README после каждого достижения вехи: (a) acceptance, (b) target. Последнее обновление: 2026-05-16.*
