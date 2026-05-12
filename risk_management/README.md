# Risk Management

## Назначение

Централизованное управление рисками торговой системы, включая ограничение позиций, управление капиталом и контроль просадки.

## Основные задачи

- Ограничение размера позиций
- Управление стоп-лоссами и тейк-профитами
- Контроль портфельных рисков
- Мониторинг просадок
- Динамическая адаптация рисков

## Ключевые компоненты управления риском

### 1. Position Sizing

Расчет оптимального размера позиции:

```python
PositionSize = f(balance, volatility, risk)
```

### 2. Stop-Loss

Ограничение убытков:

```python
SL = EntryPrice - k * σ
```

где:
- EntryPrice - цена входа
- k - множитель волатильности
- σ - волатильность

### 3. Take-Profit

Фиксация прибыли:

```python
TP = EntryPrice + m * σ
```

где:
- m - множитель волатильности для тейк-профита

## Структура модуля

```
risk_management/
├── __init__.py
├── position_sizing/
│   ├── __init__.py
│   ├── kelly_sizer.py         # Kelly criterion
│   ├── volatility_sizer.py     # Volatility-based sizing
│   ├── fixed_fractional.py     # Fixed fractional sizing
│   └── risk_parity_sizer.py    # Risk parity sizing
├── stop_loss/
│   ├── __init__.py
│   ├── atr_stop_loss.py       # ATR-based stop loss
│   ├── volatility_stop_loss.py # Volatility-based stop loss
│   ├── trailing_stop.py       # Trailing stop loss
│   └── time_stop.py          # Time-based stop loss
├── portfolio_risk/
│   ├── __init__.py
│   ├── var_calculator.py      # Value at Risk
│   ├── exposure_controller.py  # Exposure control
│   ├── correlation_monitor.py # Correlation monitoring
│   └── concentration_limiter.py # Concentration limits
├── drawdown_control/
│   ├── __init__.py
│   ├── drawdown_monitor.py    # Drawdown monitoring
│   ├── equity_curve_analyzer.py # Equity curve analysis
│   └── recovery_tracker.py     # Recovery tracking
├── dynamic_adjustment/
│   ├── __init__.py
│   ├── volatility_regime.py   # Volatility regime detection
│   ├── risk_adjuster.py       # Dynamic risk adjustment
│   └── market_condition_adapter.py # Market condition adaptation
└── risk_manager.py            # Главный менеджер рисков
```

## Ключевые компоненты

### RiskManager

Центральный менеджер управления рисками:
- Координация всех риск-компонентов
- Применение риск-правил
- Мониторинг риск-метрик
- Генерация риск-алертов

### PositionSizer

Расчет размеров позиций с учетом:
- Текущего баланса
- Волатильности актива
- Корреляций в портфеле
- Максимальных рисков

### StopLossManager

Управление стоп-лоссами:
- Динамические стопы
- Трейлинг стопы
- Временные стопы
- Групповые стопы

### PortfolioRiskController

Контроль портфельных рисков:
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Корреляционный анализ
- Концентрационные лимиты

## Алгоритмы управления риском

### Kelly Criterion

```python
def kelly_criterion(win_rate, avg_win, avg_loss, kelly_fraction=0.25):
    """
    Kelly Criterion для оптимального размера позиции
    """
    if avg_loss == 0:
        return 0
    
    win_loss_ratio = avg_win / avg_loss
    kelly_percentage = win_rate - (1 - win_rate) / win_loss_ratio
    
    # Ограничение для conservative approach
    return max(0, min(kelly_percentage * kelly_fraction, 0.25))
```

### Volatility-based Position Sizing

```python
def volatility_sizing(account_balance, asset_volatility, target_volatility=0.02):
    """
    Размер позиции на основе волатильности
    """
    volatility_adjustment = target_volatility / asset_volatility
    position_size = account_balance * volatility_adjustment
    
    return position_size
```

### ATR-based Stop Loss

```python
def atr_stop_loss(entry_price, atr, multiplier=2.0, direction='long'):
    """
    Stop Loss на основе ATR
    """
    if direction == 'long':
        stop_loss = entry_price - (atr * multiplier)
    else:
        stop_loss = entry_price + (atr * multiplier)
    
    return stop_loss
```

### Value at Risk (VaR)

```python
def calculate_var(returns, confidence_level=0.05, time_horizon=1):
    """
    Value at Risk calculation
    """
    sorted_returns = np.sort(returns)
    var_index = int(len(sorted_returns) * confidence_level)
    var = sorted_returns[var_index]
    
    # Scale for time horizon
    var_scaled = var * np.sqrt(time_horizon)
    
    return var_scaled
```

## Динамическая адаптация рисков

### Volatility Regime Detection

```python
def detect_volatility_regime(current_vol, historical_vols, window=252):
    """
    Детекция режима волатильности
    """
    vol_percentile = np.percentile(historical_vols, [25, 75])
    
    if current_vol < vol_percentile[0]:
        return 'low_volatility'
    elif current_vol > vol_percentile[1]:
        return 'high_volatility'
    else:
        return 'normal_volatility'
```

### Dynamic Risk Adjustment

```python
def adjust_risk_parameters(base_params, volatility_regime, market_condition):
    """
    Динамическая адаптация риск-параметров
    """
    adjusted_params = base_params.copy()
    
    if volatility_regime == 'high_volatility':
        adjusted_params['position_size'] *= 0.5
        adjusted_params['stop_loss_multiplier'] *= 1.5
    elif volatility_regime == 'low_volatility':
        adjusted_params['position_size'] *= 1.2
        adjusted_params['stop_loss_multiplier'] *= 0.8
    
    return adjusted_params
```

## Мониторинг рисков

### Drawdown Monitoring

```python
def calculate_drawdown(equity_curve):
    """
    Расчет просадки
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    
    return {
        'current_drawdown': drawdown.iloc[-1],
        'max_drawdown': drawdown.min(),
        'drawdown_duration': calculate_drawdown_duration(drawdown)
    }
```

### Risk Alerts

```python
def check_risk_alerts(risk_metrics, risk_limits):
    """
    Проверка риск-алертов
    """
    alerts = []
    
    if risk_metrics['drawdown'] > risk_limits['max_drawdown']:
        alerts.append('MAX_DRAWDOWN_EXCEEDED')
    
    if risk_metrics['position_size'] > risk_limits['max_position']:
        alerts.append('POSITION_SIZE_EXCEEDED')
    
    if risk_metrics['var'] > risk_limits['max_var']:
        alerts.append('VAR_EXCEEDED')
    
    return alerts
```

## Технологии

- **numpy** - численные вычисления
- **pandas** - обработка временных рядов
- **scipy** - статистические функции
- **asyncio** - асинхронный мониторинг
- **pydantic** - валидация данных

## Конфигурация

```yaml
risk_management:
  position_sizing:
    method: "volatility"  # kelly, volatility, fixed_fractional
    base_size: 0.02
    max_size: 0.1
    kelly_fraction: 0.25
    
  stop_loss:
    method: "atr"  # atr, volatility, trailing
    atr_multiplier: 2.0
    volatility_multiplier: 2.0
    trailing_activation: 1.0
    
  portfolio_risk:
    max_portfolio_var: 0.02
    max_position_concentration: 0.3
    correlation_threshold: 0.7
    
  drawdown_control:
    max_drawdown: 0.1
    daily_loss_limit: 0.05
    recovery_mode_threshold: 0.08
    
  dynamic_adjustment:
    volatility_regime_detection: true
    risk_adjustment_frequency: 3600  # seconds
    high_vol_multiplier: 0.5
    low_vol_multiplier: 1.2
```

## Метрики рисков

### Position-level Metrics

- **Position Size**: относительный размер позиции
- **Stop Loss Distance**: расстояние до стоп-лосса
- **Risk/Reward Ratio**: соотношение риска к доходности

### Portfolio-level Metrics

- **Value at Risk (VaR)**: потенциальные убытки
- **Expected Shortfall (ES)**: ожидаемые убытки при VaR breach
- **Portfolio Volatility**: волатильность портфеля
- **Correlation Matrix**: корреляции между активами

### Performance Metrics

- **Maximum Drawdown**: максимальная просадка
- **Sharpe Ratio**: риск-скорректированная доходность
- **Calmar Ratio**: доходность к максимальной просадке
- **Sortino Ratio**: доходность к downside risk

## Интеграция

Risk Management получает данные от:
- **Decision Layer** - торговые сигналы
- **Execution Layer** - информация об исполненных сделках
- **Data Layer** - рыночные данные и волатильность

И передает риск-параметры в:
- **Decision Layer** - ограничения для принятия решений
- **Execution Layer** - параметры ордеров
- **Meta-Learning** - контекст для адаптации

## Требования к реализации

1. **Надежность** - безотказная работа риск-системы
2. **Скорость** - мгновенное применение риск-правил
3. **Масштабируемость** - поддержка множественных стратегий
4. **Мониторинг** - полный контроль над рисками
5. **Гибкость** - легкая настройка риск-параметров

## Тестирование

- Unit тесты для каждого риск-компонента
- Integration тесты для pipeline
- Stress тесты для экстремальных сценариев
- Simulation тесты для различных рыночных условий
