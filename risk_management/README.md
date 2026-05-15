# Risk Management

## Назначение

Управление размером позиции и выходами: **ATR-based sizing**, **ATR trailing stop**, координация со **RL risk multiplier**. Централизация логики, разбросанной в `ist.py` по `PositionSizer`, `apply_atr_trailing_stop` и RL.

## Статус

| Компонент | Где эталон | Статус |
|-----------|------------|--------|
| `PositionSizer` (ATR) | `ist.py` | **Эталон** |
| `apply_atr_trailing_stop` | `ist.py` | **Эталон** |
| RL `risk_multiplier` | **rl_layer** | **Эталон** (динамический риск) |
| Kelly, VaR, portfolio limits | — | Roadmap |

## 1. Position Sizing (статический)

```python
class PositionSizer:
    def __init__(self, risk_per_trade=0.01, account_size=10000):
        ...

    def calculate_sizes(self, df):
        stop_multiplier = 2.0
        risk_amount = account_size * risk_per_trade
        pos_size = risk_amount / (atr * stop_multiplier)
        final_pos_size = pos_size * abs(final_signal)
```

| Параметр | По умолчанию |
|----------|--------------|
| `risk_per_trade` | 1% капитала |
| `account_size` | 10000 |
| ATR stop mult | 2.0 |

**Выход:** `pos_size`, `final_pos_size` (в BTC для spot-логики).

> Текущий `Backtester` использует сигнал {-1,0,1} без умножения на `final_pos_size` — при интеграции execution нужно связать sizing с исполнением.

## 2. ATR Trailing Stop

```python
def apply_atr_trailing_stop(df, signal_col='final_signal', atr_mult=3.0):
    # long: trailing_stop = max(prev_stop, close - atr * mult)
    # exit когда close < trailing_stop
    # short: симметрично
```

| Параметр | Значение |
|----------|----------|
| `atr_mult` | **3.0** |

**Выход:** `exit_signal` — комбинированный сигнал:

```python
combined = np.where(exit_signal == 1, 0, final_signal)
```

## 3. Динамический риск (RL)

Предпочтительный режим при наличии обученного агента:

```python
strategy_return = final_signal * pct_change.shift(-1) * risk_multiplier
```

`risk_multiplier` ∈ {0.005, 0.01, 0.02} — см. **rl_layer**.

Сравнение в research: RL dynamic vs static 1% multiplier.

## Рекомендуемый порядок применения

```text
Meta-Learning → final_signal / integrated_signal
    → [опционально] ATR trailing → combined_signal
    → PositionSizer → final_pos_size
    → [опционально] RL → risk_multiplier на доходность
    → Backtesting / Execution
```

## Что не в текущем scope

- Kelly, risk parity, portfolio VaR  
- Жёсткие лимиты экспозиции на уровне биржи (→ **execution**, позже)  
- Take-profit отдельным правилом (только trailing stop в эталоне)  

## Структура модуля (целевая)

```
risk_management/
├── __init__.py
├── position_sizer.py
├── atr_trailing_stop.py
└── risk_pipeline.py    # объединяет stop + sizer + rl hook
```

## Конфигурация

```yaml
risk_management:
  position_sizer:
    risk_per_trade: 0.01
    account_size: 10000
    atr_stop_multiplier: 2.0
  trailing_stop:
    enabled: true
    atr_mult: 3.0
    signal_column: "final_signal"
  rl_overlay:
    enabled: true
    source: "rl_layer"
```

## Интеграция

| Слой | Связь |
|------|--------|
| **Meta-Learning** | входной `final_signal` |
| **Models** | `atr`, `vol_spike_prob` |
| **RL Layer** | `risk_multiplier` |
| **Backtesting** | `combined_signals` или RL returns |

## Roadmap

- Связать `final_pos_size` с **execution**  
- Max drawdown circuit breaker  
- Согласовать комиссии с `Backtester` (0.06% + 0.02% slippage)  
