# GUI Layer

## Назначение

Мониторинг и управление торговой системой: графики, сигналы, ордера, метрики в реальном времени.

## Статус

**Не реализовано** — запланировано после **execution** и стабильного live/paper контура.

Research и оценка стратегий — **Colab / `ist.py`** + **backtesting** (`ExperimentLogger`, matplotlib).

## Целевой scope (v1)

### Панели

| Панель | Содержание |
|--------|------------|
| Live Chart | OHLCV, индикаторы, `final_signal` / `integrated_signal`, ATR stop |
| Signals | BUY/SELL/flat, `direction_prob`, `meta_prob`, `meta_mgmt_prob` |
| Orders | активные и исполненные (из **execution**) |
| Metrics | Sharpe, DD, PnL, PF — из последнего бэктеста или live stats |
| Experiments | просмотр `experiments_log.json` |

### Источники данных

- **data_layer** — live OHLCV  
- **feature_engineering** — признаки, OBI  
- **meta_learning** — текущие сигналы  
- **execution** — ордера и позиции  

## Структура (черновик)

```
gui/
├── dashboard/           # web или desktop (TBD)
├── charts/
├── widgets/
└── api_client.py        # REST/WS к backend системы
```

## Конфигурация (черновик)

```yaml
gui:
  enabled: false
  refresh_interval_sec: 5
  symbol: "BTC/USDT"
  timeframe: "1h"
```

## Roadmap

1. Read-only dashboard (сигналы + equity из логов)  
2. Paper trading controls  
3. Live с подтверждением и risk limits  
4. Алерты (просадка, disconnect)  

## Тестирование (план)

- E2E с mock execution  
- Latency отображения vs WebSocket feed  
