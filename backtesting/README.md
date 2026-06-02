# Backtesting Layer

## Назначение

Симуляция стратегии на истории, walk-forward валидация, метрики и журнал экспериментов. Реализация в **`backtesting/`**; семантика согласована с эталоном `ist.py`.

## Статус

| Компонент | Класс / функция | Статус |
|-----------|-----------------|--------|
| Движок бэктеста | `Backtester` | **Реализовано** (`position_size` опционально) |
| Метрики | `PerformanceMetrics` | **Реализовано** |
| WFO + 4 модели + purge/embargo | `TrainingOrchestrator.walk_forward_backtest` | **Реализовано** (canonical) |
| Обертка WFO | `walk_forward.run_orchestrator_walk_forward_backtest` | **Реализовано** |
| Простой WFO по готовому сигналу | `run_simple_signal_wfo` | **Реализовано** |
| WFO split без ML | `TimeSeriesSplitter` | **Вспомогательный** (без purge) |
| `run_walk_forward_validation` / `run_integrated_wfo` | — | **Удалены** → `NotImplementedError` |
| Журнал прогонов | `results_journal` → `docs/backtest_journal/` | **Реализовано** (таблица в GUI «Журнал WFO») |
| Критерии acceptance/target | `criteria_evaluator` | **Реализовано** |
| Monte Carlo, order-level sim | — | Roadmap |

## Backtester

```python
class Backtester:
    def __init__(self, commission=0.0006, slippage=0.0002):
        ...

    def run(self, df, signals, position_size=None):
        # signal ∈ {-1, 0, 1}
        # strategy_returns = signal.shift(1) * position_size.shift(1) * market_returns
        # costs = |Δ(signal * position_size)| * (commission + slippage)
        # net_returns, cum_strategy_returns, drawdown
```

### Параметры research (эталон)

| Параметр | Значение |
|----------|----------|
| `commission` | **0.0006** (0.06%) |
| `slippage` | **0.0002** (0.02%) |

### Допущения

- Без `position_size` — множитель 1.0; с risk bridge — `final_pos_size` из orchestrator  
- Исполнение на **close** следующего бара (`shift(1)`)  
- Издержки при изменении **эффективной** экспозиции `signal × position_size`  
- Long и short симметричны (если `trade_mode: both` в orchestration)  

## PerformanceMetrics

```python
metrics = PerformanceMetrics(backtest_results).calculate_metrics()
```

| Метрика | Описание |
|---------|----------|
| Sharpe Ratio | `mean(net_returns) / std * sqrt(8760)` (часовые бары) |
| Profit Factor | sum(gains) / abs(sum(losses)) |
| Win Rate (%) | доля положительных баров с ненулевым return |
| Max Drawdown (%) | min drawdown |
| Recovery Factor | total_return / abs(max_dd) |
| Total Return (%) | cumulative |

## TimeSeriesSplitter

Expanding window по фолдам:

```python
class TimeSeriesSplitter:
    def __init__(self, n_splits=5, train_size=0.7):
        ...

    def split(self, data):
        # yield train_chunk, test_chunk
```

## Walk-Forward (canonical)

```python
from orchestration import TrainingOrchestrator, OrchestratorConfig
from orchestration.model_factory import build_orchestration_models, meta_weighting_from_config

# после initialize(models, regime_detector, meta_weighting, ...)
fold_metrics = orchestrator.walk_forward_backtest(features, targets)
# purge + embargo via DataLeakagePreventer; train-only meta threshold on OOS
```

CLI: `python -m orchestration report-real`, `from-parquet <features.parquet>`.

Журнал: `docs/backtest_journal/` (`runs.jsonl`, `INDEX.md`).

## Структура модуля (фактическая)

```
backtesting/
├── backtester.py
├── performance_metrics.py
├── metrics_config.py
├── time_series_splitter.py
├── walk_forward.py              # orchestrator wrapper + run_simple_signal_wfo
├── criteria_evaluator.py
└── results_journal.py
```

## Конфигурация

```yaml
backtesting:
  commission: 0.0006
  slippage: 0.0002
  sharpe_annualization_hours: 8760
  walk_forward:
    n_splits: 5
    train_size: 0.7
  integrated_wfo:
    n_splits: 3
    dl_epochs: 3
```

## Интеграция

```text
decision / meta_learning  →  signals
risk_management           →  optional combined_signal (ATR stop)
backtesting               →  perf, metrics, logs
```

| Источник сигнала | Когда |
|------------------|--------|
| `final_signal` | MetaFilter pipeline |
| `integrated_signal` | Regime-adaptive ensemble |
| `combined_signals` | После ATR trailing |
| RL returns | Отдельная оценка через `run_rl_backtest` |

## Roadmap

- Учёт `final_pos_size` в движке  
- Отдельный `walk_forward_engine.py` с единым API  
- Отчёты HTML / сравнение экспериментов из `experiments_log.json`  
