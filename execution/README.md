# Execution Layer

## Назначение

Исполнение торговых сигналов на бирже: paper trading и live trading, управление ордерами, мониторинг исполнения.

## Статус

**Не реализовано** — запланировано после стабилизации research pipeline (`ist.py` + слои data → backtest).

Сейчас «исполнение» — только векторный **Backtester** (**backtesting**).

## Зависимости (готовые контракты)

| Вход | Источник |
|------|----------|
| Сигнал {-1, 0, 1} | **decision** / **meta_learning** |
| `final_pos_size` | **risk_management** (`PositionSizer`) |
| `risk_multiplier` | **rl_layer** (опционально) |
| Комиссии / slippage | согласовать с backtest: 0.06% / 0.02% |

## Целевой scope (v1)

### Режимы

- **Paper** — симуляция на live котировках без реальных средств  
- **Live** — OKX через ccxt (тот же брокер, что data REST)  

### Типы ордеров (приоритет)

1. **MARKET** — MVP  
2. **LIMIT** — v1.1  
3. **STOP** (ATR trailing из risk) — v1.2  

### Компоненты (из прежнего PRD, сокращённо)

```
execution/
├── brokers/
│   ├── base_broker.py
│   ├── okx_broker.py
│   └── paper_broker.py
├── orders/
│   └── order_manager.py
├── execution_manager.py
└── monitoring/
    └── execution_monitor.py
```

## Конфигурация (черновик)

```yaml
execution:
  mode: "paper"          # paper | live
  exchange: okx
  sandbox: true
  default_order_type: market
  max_order_size: null   # из risk_management
```

## Интеграция (план)

```text
decision → risk_management → execution → OKX
                ↓
         data_layer (live quotes / L2 для OBI)
```

## Roadmap

1. Paper broker + reconciliation с `Backtester` assumptions  
2. Live market orders, idempotency, rate limits  
3. Limit/stop, emergency stop  
4. Логирование сделок в storage (Parquet — опционально)  

## Тестирование (план)

- Paper vs backtest на одном периоде  
- Sandbox OKX integration tests  
- Fail-safe: отмена всех ордеров, max exposure  
