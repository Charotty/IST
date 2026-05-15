# Backtesting Layer

## Назначение

Симуляция стратегии на истории, walk-forward валидация, метрики и лог экспериментов. **Использовать готовую реализацию из `ist.py`** — перенос в модуль `backtesting/` без изменения семантики.

## Статус

| Компонент | Класс / функция | Статус |
|-----------|-----------------|--------|
| Движок бэктеста | `Backtester` | **Эталон** |
| Метрики | `PerformanceMetrics` | **Эталон** |
| WFO split | `TimeSeriesSplitter` | **Эталон** |
| WFO Direction | `run_walk_forward_validation` | **Эталон** |
| WFO Integrated | `run_integrated_wfo` | **Эталон** |
| Логи | `ExperimentLogger` | **Эталон** |
| Monte Carlo, order-level sim | — | Roadmap |

## Backtester

```python
class Backtester:
    def __init__(self, commission=0.0005, slippage=0.0001):
        ...

    def run(self, df, signals):
        # signal ∈ {-1, 0, 1}
        # strategy_returns = signal.shift(1) * market_returns
        # costs = |signal.diff()| * (commission + slippage)
        # net_returns = strategy_returns - costs
        # cum_strategy_returns, drawdown
```

### Параметры research (эталон)

| Параметр | Значение |
|----------|----------|
| `commission` | **0.0006** (0.06%) |
| `slippage` | **0.0002** (0.02%) |

### Допущения

- Позиция **полная** по знаку сигнала (без `final_pos_size` в `run`)  
- Исполнение на **close** следующего бара (`shift(1)`)  
- Издержки только при **смене** сигнала (`trades = signal.diff().abs()`)  
- Long и short симметричны  

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

## Walk-Forward

### Direction only

```python
run_walk_forward_validation(df, n_splits=5)
# на каждом фолде: DirectionModel → get_calibrated_signals(0.52) → Backtester
```

### Integrated system

```python
run_integrated_wfo(df, n_splits=3)
# переобучение LGBM + короткое DL (3 epochs)
# get_dynamic_meta_signal → integrated_signal → Backtester
```

## ExperimentLogger

```python
logger.log_experiment(name, params_dict, metrics_series)
# → experiments_log.json
```

Примеры имён: `Phase_2_MTF_Full_History`, `Integrated_Meta_Optimization_v1`, `Phase_2_with_ATR_Stop`.

## Структура модуля (целевая)

```
backtesting/
├── __init__.py
├── backtester.py              # из ist.py
├── performance_metrics.py
├── time_series_splitter.py
├── walk_forward.py            # run_walk_forward_validation, run_integrated_wfo
└── experiment_logger.py
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
