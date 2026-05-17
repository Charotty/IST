# Справочник: критерии и метрики бэктестинга (IST)

Документ фиксирует результаты исследования (май 2026) для двух целей обучения системы:

- **(а) Приемлемый результат** — доказательство, что на данных можно торговать (устойчивое OOS-преимущество после издержек, не случайность).
- **(б) Хороший результат** — сильные risk-adjusted показатели с запасом под деградацию бэктест → лайв.

Источники: [Breaking Alpha — Sharpe & algorithm selection](https://breakingalpha.io/insights/understanding-sharpe-ratios-selecting-trading-algorithms), [Algo Strategy Analyzer — trading metrics (2026)](https://algostrategyanalyzer.com/en/blog/algorithmic-trading-metrics/), [Technical Analysis Pro — AI backtesting & WFO](https://www.technical-analysis-pro.com/strategies-ai-backtesting-walk-forward-model-validation/), [Quant Trading Tools — walk-forward analysis](https://quanttradingtools.com/walk-forward-analysis/), [Optimized Portfolio — Sharpe vs Sortino vs Calmar](https://www.optimizedportfolio.com/risk-adjusted-return/), [Finantic — performance measures](http://www.finantic.de/en/performance-measures-for-trading-systems/), Bailey & López de Prado — *Deflated Sharpe Ratio* ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).

---

## 1. Текущее состояние в репозитории IST

### 1.1 `backtesting/performance_metrics.py` — реализовано

| Метрика | Формула / логика в коде |
|---------|-------------------------|
| **Total Return (%)** | `cum_strategy_returns[-1] - 1` |
| **Sharpe Ratio** | `mean(net_returns) / std * sqrt(8760)` — **часовые** бары |
| **Profit Factor** | `sum(gains) / abs(sum(losses))` по `net_returns` |
| **Win Rate (%)** | доля положительных **баров** с ненулевым return (не обязательно сделки) |
| **Max Drawdown (%)** | `min(drawdown) * 100` |
| **Recovery Factor** | `total_return / abs(min(drawdown))` |

### 1.2 `backtesting/backtester.py` — колонки результата

`market_returns`, `strategy_returns`, `net_returns`, `cum_strategy_returns`, `cum_market_returns`, `drawdown`, `trades`, `costs`, `position_size`.

По умолчанию: `commission=0.0006`, `slippage=0.0002` (согласовано с execution).

### 1.3 `models/evaluation/model_evaluator.py` — дублирует финансовые метрики + ML

Sharpe, PF, max_drawdown, total_return, recovery_factor; ML: accuracy, precision, recall, f1, avg_confidence. В docstring упомянуты CAGR и calibration — **CAGR в коде не считается**.

### 1.4 Конфигурация окон (целевая, `config/README.md`)

```yaml
walk_forward:
  train_window: 252      # дневная логика в доке
  validation_window: 63
  test_window: 63
  step_size: 21
simulation:
  commission: 0.001
  slippage: 0.0005
analysis:
  benchmark: "BTC/USDT"
  risk_free_rate: 0.02
```

`orchestration`: `walk_forward_step: 100` (бары). Бенчмарк на реальных данных: хвост ~2200 баров (`real_data_benchmark.py`).

### 1.5 Gap: метрики из литературы, **не** в `PerformanceMetrics`

Sortino, Calmar, CAGR, Expectancy, Sortino/Calmar, Ulcer Index, UPI (Martin Ratio), Omega, Information Ratio, Treynor, VaR/CVaR, Time Underwater, Max Consecutive Losses, Average Trade, Win/Loss streaks, benchmark alpha, **Walk-Forward Efficiency (WFE)**, **Deflated Sharpe Ratio (DSR)**, **Minimum Backtest Length (MinBTL)**, Monte Carlo bands, parameter stability по фолдам.

---

## 2. Две цели: пороги «приемлемо» vs «хорошо»

> **Важно:** внешние таблицы часто для **дневных** returns (`√252`). В IST Sharpe годализуется через **`√8760`** (1h). Сравнивать абсолютные числа Sharpe с дневными статьями напрямую нельзя — только **относительно** внутри одного пайплайна и одной частоты баров.

### 2.1 Задача (а) — «можно торговать на этих данных»

| Критерий | Ориентир | Источник / смысл |
|----------|----------|------------------|
| **OOS прибыльность** | Суммарный / средний OOS return > 0 по **нескольким** WFO-окнам | WFO как стандарт для ML |
| **Profit Factor** | **> 1.0** обязательно; желательно **≥ 1.3–1.5** | Algo: <1.5 «хрупко» к costs; institutional min **>1.75** для «viable» |
| **Sharpe (год., та же формула)** | **> 0.5–1.0** на OOS | Breaking Alpha: 0.5–1.0 acceptable; Algo: <1 «субоптимально после costs» |
| **Деградация IS→OOS** | WFE **> 0.5** (желательно **> 0.6**) | Quant Trading Tools, StratBase-обзоры |
| **Число независимых OOS** | **≥ 5–10** окон WFO | Quant Trading Tools: <5 шумно, >15 фрагментирует IS |
| **Сделки / решения** | **≥ 30** для чернового вывода, **100+** для устойчивого Win Rate | Algo Strategy Analyzer |
| **Издержки** | Commission + slippage в каждом прогоне | Trap «underestimated costs» |
| **Статистика (желательно)** | Sharpe significance p < 0.05 или DSR > 0 при учёте числа trials | Bailey & López de Prado |

**Практическое правило для IST (а):** walk-forward по всем моделям (`lgb`, `gru`, `xgb`, `cnn` + integrated), OOS PF ≥ 1.2, средний OOS Sharpe > 0.5 (8760), WFE ≥ 0.5, без «весь PnL в одном фолде».

### 2.2 Задача (б) — «хороший результат»

| Критерий | Ориентир (бэктест / OOS) | Комментарий |
|----------|-------------------------|-------------|
| **Sharpe** | **1.0–2.0** good (retail); **2.0–3.0** very good | Breaking Alpha, Algo; бэктест **~1.5** → лайв может **<1** |
| **Sharpe бэктест с запасом** | Целиться **≥ 2.0** на IS/WFO до haircut | Algo: проф. менеджеры часто требуют **≥2** в бэктесте |
| **Profit Factor** | **1.5–2.5** good; **>3** подозрительно | overfit / малый sample |
| **Calmar** | **> 1** хорошо, **> 3** отлично | return / max DD |
| **Recovery Factor** | **2–5** good, **>5** excellent | уже в коде |
| **Max Drawdown** | **< 15–25%** (контекст крипто 1h) | Algo: >25% poor для daily-ориентиров |
| **Согласованность метрик** | Sharpe + PF + DD/Calmar не противоречат | высокий Sharpe + плохой Calmar → tail risk |
| **AI-специфика** | Sharpe **>3** на бэктесте — «почти наверняка overfit» | Technical Analysis Pro |

**Практическое правило для IST (б):** OOS PF ≥ 1.6, OOS Sharpe > 1.0 (8760), Recovery Factor 2–5, Calmar > 1 (после добавления в код), стабильность по фолдам (std метрик между окнами низкая).

### 2.3 Haircut бэктест → лайв

| Фактор | Оценка ухудшения |
|--------|------------------|
| Переход бэктест → live | **−30…50%** к Sharpe и смежным метрикам |
| Множественное тестирование моделей | Deflated Sharpe / MinBTL |
| Ненормальность returns | Sortino, Omega, tail analysis |

---

## 3. Полный каталог метрик бэктестинга

### 3.1 Доходность и эффективность

| Метрика | Формула (типичная) | Интерпретация | В IST |
|---------|-------------------|---------------|-------|
| **Net Profit / Total Return** | конечная equity / начальная − 1 | абсолютный результат | ✅ Total Return (%) |
| **CAGR** | `(Final/Initial)^(1/years) − 1` | сравнение разных горизонтов | ❌ |
| **Average Return (annualized)** | `mean(returns) * periods_per_year` | база для ratios | частично (в Sharpe) |
| **Expectancy** | `E[trade PnL]` или avg profit per trade | жизнеспособность после costs | ❌ |
| **Average Trade** | Net Profit / N trades | intraday: must exceed costs | ❌ |
| **Profit Factor** | gross profit / gross loss | **>1.75** viable (inst.); **1.5–2.5** robust | ✅ |
| **Payoff Ratio** | avg win / avg loss | дополняет Win Rate | ❌ |

### 3.2 Win Rate и сделки

| Метрика | Пороги / заметки | В IST |
|---------|------------------|-------|
| **Win Rate** | 40% + R:R 1:3 может бить 60% + 1:0.5 | ✅ (по барам) |
| **N trades** | ≥30 inference, ≥100 robust | считать из `trades` / сигналов |
| **Max Consecutive Losses** | streak risk | ❌ (есть в AI-backtest гайдах) |
| **Number of trades** | мало сделок → ненадёжные метрики | из `Backtester.trades` |

### 3.3 Риск и просадка

| Метрика | Формула | Пороги (типичные) | В IST |
|---------|---------|-------------------|-------|
| **Max Drawdown** | max peak-to-trough | 10% tolerable … 50% devastating | ✅ |
| **Time Underwater** | время до нового peak | критично для психологии | ❌ |
| **Ulcer Index** | RMS drawdowns | fat tails, duration | ❌ |
| **VaR / CVaR** | квантили хвоста | tail risk | ❌ |
| **Volatility (annualized)** | `std * sqrt(T)` | сравнение стратегий | ❌ |
| **Downside Deviation** | std отриц. returns | база Sortino | ❌ |

### 3.4 Risk-adjusted ratios

| Метрика | Формула | Пороги (daily-литература; адаптировать частоту) | В IST |
|---------|---------|--------------------------------------------------|-------|
| **Sharpe** | `(R − Rf) / σ` annualized | <1 suboptimal; 1–2 good; 2–3 very good; >3 suspicious | ✅ (√8760) |
| **Sortino** | `(R − Rf) / σ_downside` | often ~1.5–3 «good» | ❌ |
| **Calmar / MAR** | `CAGR / |MaxDD|` | >1 good, >3 excellent | ❌ |
| **Sterling** | `APR / (MaxDD + 10%)` | variant Calmar family | ❌ |
| **Recovery Factor** | Net Profit / MaxDD | <1 dangerous; 2–5 good; >5 excellent | ✅ |
| **Omega** | Σ gains above threshold / Σ losses below | >1 profitable | ❌ |
| **Information Ratio** | active return / tracking error | vs benchmark | ❌ (есть `cum_market_returns`) |
| **Treynor** | excess return / beta | portfolio context | ❌ |
| **UPI (Martin Ratio)** | excess return / Ulcer Index | penalizes long underwater | ❌ |
| **Modigliani M2** | Sharpe в % return | intuitive vs peers | ❌ |

### 3.5 Сравнение с бенчмарком

| Метрика | Смысл | В IST |
|---------|-------|-------|
| **Alpha vs Buy & Hold** | `strategy_return − market_return` | колонки есть, метрика не агрегирована |
| **Beta / correlation** | рыночная экспозиция | ❌ |
| **Information Ratio** | risk-adjusted alpha | ❌ |

### 3.6 Кривая капитала (качественные критерии)

Из Algo Strategy Analyzer — проверять **equity curve**, не только числа:

- ✅ постоянный наклон, короткие DD, нет зависимости от 1–2 сделок  
- 🚫 «лестница», длинные плато, вертикальные обвалы, рост только в конце (overfit)

---

## 4. Метрики валидации и walk-forward (не PnL, но обязательны для (а))

### 4.1 Walk-Forward Efficiency (WFE)

```
WFE = mean( OOS_annualized_return / IS_annualized_return )  по окнам
```

| WFE | Интерпретация |
|-----|---------------|
| **> 0.5** | acceptable |
| **> 0.7** | strong |
| **< 0.5** | IS завышает ожидания (~2× и более) |

Источник: [Quant Trading Tools — WFA](https://quanttradingtools.com/walk-forward-analysis/).

### 4.2 Настройка окон WFO

| Параметр | Рекомендация |
|----------|--------------|
| Число окон | **5–10** (не <5, редко >15) |
| IS : OOS | **70/30** или **80/20** |
| Тип | **rolling** предпочтительнее anchored |
| Objective оптимизации | return/maxDD или Sharpe, не голый net profit |
| Стабильность параметров | график optimal params по окнам |

### 4.3 ML / AI-специфика (доп. критерии)

| Метрика | Назначение | Порог (гайд AI backtesting) |
|---------|------------|----------------------------|
| **Fold accuracy std** | робастность классификатора | std < **5%** между фолдами |
| **Train vs test gap** | overfit | train >> test (напр. 75% vs 52%) — плохо |
| **Purged CV + embargo** | leakage | gap ≥ lookback фичей |
| **Deflated Sharpe Ratio** | множественные trials | учитывать `n_trials` |
| **Min Backtest Length** | длина истории vs trials | см. Bailey & López de Prado |
| **Monte Carlo** | path dependency | median, 5th/95th pct, P(loss) |

### 4.4 Статистические тесты Sharpe

- **t-test Sharpe** (Lo, 2002): H0: true Sharpe = 0; p < 0.05  
- **Deflated Sharpe Ratio (DSR)**: коррекция на selection bias и non-normality; при большом числе переборов стратегий observed Sharpe часто ложный.

Пример: после **7** конфигураций на **2-летнем** бэктесте можно получить Sharpe > 1 при истинном SR = 0 ([What to Look for in a Backtest](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308682)).

### 4.5 Monte Carlo (дополнение к WFO)

Перемешивание / bootstrap последовательности сделок или returns:

- median return, 5th / 95th percentile  
- probability of loss  
- не заменяет WFO (другой вопрос: path dependency vs parameter drift)

---

## 5. Объём данных для прогонов

| Контекст | Минимум / ориентир |
|----------|-------------------|
| **WFO окна** | достаточно истории для **5–10** полных IS+OOS циклов |
| **Дневные бары (док config)** | train 252 + val 63 + test 63, step 21 ≈ **~1.5+ года** на цикл; для нескольких циклов — **несколько лет** |
| **Часовые бары (IST)** | те же *торговые дни* × 24; для GRU/CNN — чем длиннее чистая история, тем лучше |
| **Сделки** | ≥30 (грубо), ≥100 (надёжно) |
| **Режимы рынка** | покрыть trend / range / high vol (regime attribution) |
| **Текущий bench IST** | 2200 баров 1h — **ускорение smoke**, не финальный объём для (б) |

Рекомендация: для финальной валидации (а)/(б) использовать **максимально доступную** историю `BTC-USDT_1h.parquet` (и др. активы), единые commission/slippage, лог в `ExperimentLogger`.

---

## 6. Ловушки бэктеста (чеклист перед прогонами)

1. **Overfitting** — особенно DL (тысячи параметров).  
2. **Data leakage** — нормализация на всём датасете, фичи с look-ahead.  
3. **Survivorship bias** — только «живые» активы.  
4. **Underestimated costs** — commission, slippage, funding.  
5. **Selection bias** — выбор лучшей из N моделей без DSR/MinBTL.  
6. **Re-running WFO** до «прохождения» — meta-level curve fitting.  
7. **Несогласованная annualization** — 252 vs 8760 vs per-trade.

---

## 7. Сводная таблица порогов (для копирования в config / тесты)

Предлагаемые **черновые** пороги для автоматической проверки в IST (часовой Sharpe, OOS WFO):

| Метрика | (а) Acceptable | (б) Good |
|---------|----------------|----------|
| OOS Profit Factor | ≥ 1.2 | ≥ 1.6 |
| OOS Sharpe (√8760) | > 0.5 | > 1.0 |
| OOS Total Return | > 0 | — |
| WFE | ≥ 0.5 | ≥ 0.7 |
| Max Drawdown (%) | < 35 (crypto 1h) | < 20 |
| Recovery Factor | > 1.0 | 2–5 |
| WFO folds (OOS+) | ≥ 5 | ≥ 8 |
| Min trades / decisions | ≥ 30 | ≥ 100 |
| Fold metric stability | нет одного доминирующего фолда | low std Sharpe/PF across folds |

*Calmar, Sortino, DSR — добавить в `PerformanceMetrics` и пороги уточнить после первых полных прогонов.*

---

## 8. План внедрения

Пункты 1–3 и журнал прогонов — **выполнены**. Дальнейшие шаги по фазам (Sharpe, все модели, DSR, target): **`docs/BACKTESTING_ACTION_PLAN.md`**.

---

## 9. Ссылки

| Тема | URL |
|------|-----|
| Sharpe & institutional thresholds | https://breakingalpha.io/insights/understanding-sharpe-ratios-selecting-trading-algorithms |
| 8 metrics, PF, DD, realistic values | https://algostrategyanalyzer.com/en/blog/algorithmic-trading-metrics/ |
| AI WFO, Sortino, Calmar, DSR, Monte Carlo | https://www.technical-analysis-pro.com/strategies-ai-backtesting-walk-forward-model-validation/ |
| WFE, window count, IS/OOS | https://quanttradingtools.com/walk-forward-analysis/ |
| Sortino, Calmar, UPI | https://www.optimizedportfolio.com/risk-adjusted-return/ |
| Ulcer, Calmar, VaR, UPI | http://www.finantic.de/en/performance-measures-for-trading-systems/ |
| Deflated Sharpe Ratio | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 |
| What to look for in a backtest | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308682 |

---

*Файл создан для задачи обучения системы до уровней (а) и (б). Обновлять после первых полных WFO-прогонов по всем `model_keys`.*
