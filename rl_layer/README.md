# Reinforcement Learning Layer

## Назначение

**Не предсказывает направление рынка.** RL выбирает **уровень риска** (размер экспозиции) поверх готового сигнала `final_signal` / ансамбля. Опционально — расширенное состояние с **OBI** (микроструктура).

Эталон: `TradingEnvironment`, `DQNAgent`, `EnsembleTradingEnv`, `MicrostructureRLenv` в `ist.py`.

## Статус

| Компонент | Статус |
|-----------|--------|
| DQN, discrete risk levels | **Эталон** |
| EnsembleTradingEnv | **Эталон** |
| MicrostructureRLenv (+ OBI) | **Эталон** (при live OBI — production) |
| BUY/SELL/HOLD как actions | **Не используется** |
| Continuous position [-1,1] | Roadmap |

## Концепция

```text
final_signal (от Meta-Learning)  ×  risk_multiplier (от RL)  →  фактическая доходность
```

Награда в среде: `reward = signal * price_change * risk_pct * 100`.

## Action Space

**Дискретный — 3 уровня риска:**

| Action | `risk_multiplier` | Смысл |
|--------|-------------------|--------|
| 0 | 0.005 (0.5%) | Low risk |
| 1 | 0.01 (1%) | Medium |
| 2 | 0.02 (2%) | High |

```python
risk_map = {0: 0.005, 1: 0.01, 2: 0.02}
```

## Observation Space

### TradingEnvironment (базовый, 10 dim)

```python
state = [
    rsi, volatility, regime_pred,
    vol_spike_prob, direction_prob, meta_prob,
    ema_slope, adx, rsi_15m, rsi_4h
]
```

### EnsembleTradingEnv (10 dim)

Замена `direction_prob` → **`ensemble_prob`**, остальное как базовый.

### MicrostructureRLenv (11 dim)

Добавлен **`order_book_imbalance`**:

```python
state = [
    rsi, volatility, regime_pred, vol_spike_prob,
    ensemble_prob, meta_prob,
    ema_slope, adx, rsi_15m, rsi_4h,
    order_book_imbalance
]
```

## DQNAgent

| Параметр | Значение |
|----------|----------|
| Сеть | Dense(24) → Dense(24) → Linear(actions) |
| Optimizer | Adam lr=0.001 |
| `gamma` | 0.95 |
| `epsilon` | 1.0 → min 0.01, decay 0.995 |
| Memory | deque maxlen=2000 |
| Batch | 32 |

Обучение: episodic, replay из memory, target Q вручную (как в Colab).

## Среда — step

```python
def step(self, action):
    risk_pct = risk_map[action]
    signal = df.iloc[step]['final_signal']
    price_change = (future_close - current_close) / current_close
    reward = signal * price_change * risk_pct * 100
    ...
```

## Бэктест RL

```python
def run_rl_backtest(df, agent):
    # для каждого бара: action → risk_multiplier
    # strategy_return = final_signal * pct_change.shift(-1) * risk_multiplier
```

Сравнение: **RL dynamic** vs **static 1%** — см. Phase 4 в `ist.py`.

## Выбор реализации

| Сценарий | Среда |
|----------|--------|
| Базовый риск-менеджмент | `TradingEnvironment` + `direction_prob`, `meta_prob` |
| После simple/regime ensemble | `EnsembleTradingEnv` + `ensemble_prob` |
| С ликвидностью (OBI) | `MicrostructureRLenv` — **предпочтительно при live L2** |

## Структура модуля (целевая)

```
rl_layer/
├── __init__.py
├── environment.py           # TradingEnvironment
├── ensemble_environment.py
├── microstructure_environment.py
├── dqn_agent.py
└── rl_backtest.py           # run_rl_backtest
```

## Конфигурация

```yaml
rl_layer:
  algorithm: dqn
  action_size: 3
  risk_multipliers: [0.005, 0.01, 0.02]
  gamma: 0.95
  epsilon_decay: 0.995
  memory_size: 2000
  batch_size: 32
  episodes: 5
  steps_per_episode: 500-2000
  use_ensemble_state: true
  use_obi: false              # true когда live OBI
```

## Интеграция

| Откуда | Что |
|--------|-----|
| **Meta-Learning** | `final_signal`, `ensemble_prob`, `meta_prob` |
| **Models** | `regime_pred`, `vol_spike_prob`, probs |
| **Feature Engineering** | OBI (live) |
| **Risk Management** | статический `PositionSizer` как baseline vs RL |
| **Backtesting** | оценка `strategy_return` с multiplier |

## Roadmap

- Reward: явный штраф за drawdown и costs  
- Double DQN / Prioritized replay  
- Не смешивать с генерацией `final_signal` в одной политике  
