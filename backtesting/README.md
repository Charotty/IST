# Backtesting Layer

## Назначение

Комплексная система бэктестинга торговых стратегий с walk-forward валидацией и анализом производительности.

## Основные задачи

- Симуляция исторической торговли
- Walk-forward валидация
- Анализ производительности стратегий
- Расчет финансовых метрик
- Оптимизация параметров

## Walk-forward Validation

```text
Train → Validation → Test
      ↓
Shift window
      ↓
Repeat
```

Процесс поэтапного тестирования на разных временных интервалах для оценки устойчивости стратегии.

## Метрики оценки

### Sharpe Ratio

```python
Sharpe = (E[R_p - R_f]) / σ_p
```

Риск-скорректированная доходность портфеля.

### Max Drawdown

```python
MDD = (Peak - Trough) / Peak
```

Максимальная просадка капитала.

### Profit Factor

```python
PF = GrossProfit / GrossLoss
```

Отношение общей прибыли к общим убыткам.

## Структура модуля

```
backtesting/
├── __init__.py
├── engines/
│   ├── __init__.py
│   ├── backtest_engine.py     # Основной движок бэктестинга
│   ├── walk_forward_engine.py  # Walk-forward валидация
│   └── monte_carlo_engine.py  # Monte Carlo симуляции
├── simulators/
│   ├── __init__.py
│   ├── market_simulator.py    # Симулятор рынка
│   ├── order_simulator.py      # Симулятор ордеров
│   ├── slippage_simulator.py  # Симулятор проскальзывания
│   └── cost_simulator.py      # Симулятор издержек
├── analysis/
│   ├── __init__.py
│   ├── performance_analyzer.py # Анализ производительности
│   ├── risk_analyzer.py       # Анализ рисков
│   ├── metrics_calculator.py   # Расчет метрик
│   └── report_generator.py    # Генерация отчетов
├── optimization/
│   ├── __init__.py
│   ├── parameter_optimizer.py  # Оптимизация параметров
│   ├── grid_search.py         # Grid search
│   ├── bayesian_optimizer.py   # Bayesian optimization
│   └── genetic_optimizer.py    # Genetic algorithm
├── visualization/
│   ├── __init__.py
│   ├── equity_curve_plotter.py # График кривой капитала
│   ├── drawdown_plotter.py     # График просадок
│   ├── returns_plotter.py      # График доходности
│   └── comparison_plotter.py  # Сравнительные графики
└── backtest_manager.py         # Главный менеджер бэктестинга
```

## Ключевые компоненты

### BacktestEngine

Основной движок для симуляции торговли:
- Исполнение исторических сделок
- Управление позициями и балансом
- Применение комиссий и проскальзывания
- Логирование всех операций

### WalkForwardEngine

Движок walk-forward валидации:
- Разделение данных на окна
- Последовательное обучение и тестирование
- Агрегация результатов
- Оценка устойчивости

### PerformanceAnalyzer

Анализ производительности:
- Расчет финансовых метрик
- Статистический анализ
- Сравнение с бенчмарками
- Генерация отчетов

## Алгоритмы бэктестинга

### Simple Backtesting

```python
def simple_backtest(strategy, data, initial_balance=10000):
    """
    Простой бэктестинг стратегии
    """
    balance = initial_balance
    positions = {}
    trades = []
    
    for timestamp, row in data.iterrows():
        # Получение сигнала от стратегии
        signal = strategy.generate_signal(row)
        
        if signal == 'BUY' and positions.get('symbol', 0) == 0:
            # Открытие длинной позиции
            quantity = calculate_position_size(balance, row['close'])
            cost = quantity * row['close'] * (1 + COMMISSION)
            
            if cost <= balance:
                balance -= cost
                positions['symbol'] = quantity
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': row['close'],
                    'quantity': quantity,
                    'balance': balance
                })
        
        elif signal == 'SELL' and positions.get('symbol', 0) > 0:
            # Закрытие позиции
            quantity = positions['symbol']
            revenue = quantity * row['close'] * (1 - COMMISSION)
            
            balance += revenue
            del positions['symbol']
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': row['close'],
                'quantity': quantity,
                'balance': balance
            })
    
    return trades, balance
```

### Walk-forward Validation

```python
def walk_forward_validation(strategy, data, window_size, step_size):
    """
    Walk-forward валидация
    """
    results = []
    
    for start in range(0, len(data) - window_size * 3, step_size):
        train_end = start + window_size
        val_end = train_end + window_size
        test_end = val_end + window_size
        
        # Разделение данных
        train_data = data[start:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:test_end]
        
        # Обучение стратегии
        strategy.fit(train_data)
        
        # Валидация и оптимизация
        strategy.optimize(val_data)
        
        # Тестирование
        test_results = backtest_strategy(strategy, test_data)
        
        results.append({
            'train_period': (start, train_end),
            'test_period': (val_end, test_end),
            'performance': test_results
        })
    
    return aggregate_results(results)
```

### Monte Carlo Simulation

```python
def monte_carlo_simulation(strategy, data, n_simulations=1000):
    """
    Monte Carlo симуляция
    """
    simulation_results = []
    
    for i in range(n_simulations):
        # Случайное перемешивание данных
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        
        # Бэктестинг на перемешанных данных
        result = backtest_strategy(strategy, shuffled_data)
        
        simulation_results.append(result)
    
    # Анализ распределения результатов
    return analyze_simulation_distribution(simulation_results)
```

## Расчет метрик

### Financial Metrics

```python
def calculate_financial_metrics(trades, initial_balance):
    """
    Расчет финансовых метрик
    """
    equity_curve = build_equity_curve(trades, initial_balance)
    returns = calculate_returns(equity_curve)
    
    metrics = {
        'total_return': (equity_curve[-1] - initial_balance) / initial_balance,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'profit_factor': calculate_profit_factor(trades),
        'win_rate': calculate_win_rate(trades),
        'avg_trade': calculate_average_trade(trades),
        'calmar_ratio': calculate_calmar_ratio(equity_curve, returns),
        'sortino_ratio': calculate_sortino_ratio(returns)
    }
    
    return metrics
```

### Risk Metrics

```python
def calculate_risk_metrics(equity_curve, returns):
    """
    Расчет риск-метрик
    """
    metrics = {
        'volatility': np.std(returns) * np.sqrt(252),  # Annualized
        'var_95': np.percentile(returns, 5),
        'expected_shortfall': np.mean(returns[returns <= np.percentile(returns, 5)]),
        'skewness': scipy.stats.skew(returns),
        'kurtosis': scipy.stats.kurtosis(returns),
        'beta': calculate_beta(returns, market_returns),
        'alpha': calculate_alpha(returns, market_returns)
    }
    
    return metrics
```

## Оптимизация параметров

### Grid Search

```python
def grid_search_optimization(strategy, data, param_grid):
    """
    Grid search оптимизация
    """
    best_score = float('-inf')
    best_params = None
    results = []
    
    # Генерация всех комбинаций параметров
    param_combinations = generate_param_combinations(param_grid)
    
    for params in param_combinations:
        # Установка параметров стратегии
        strategy.set_params(params)
        
        # Бэктестинг
        trades, final_balance = backtest_strategy(strategy, data)
        
        # Расчет метрики
        score = calculate_optimization_score(trades, final_balance)
        
        results.append({
            'params': params,
            'score': score,
            'trades': trades,
            'final_balance': final_balance
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, results
```

### Bayesian Optimization

```python
def bayesian_optimization(strategy, data, param_bounds, n_iterations=50):
    """
    Bayesian optimization
    """
    from skopt import gp_minimize
    
    def objective(params):
        # Преобразование параметров
        param_dict = convert_params_to_dict(params, param_bounds)
        
        # Установка параметров
        strategy.set_params(param_dict)
        
        # Бэктестинг
        trades, final_balance = backtest_strategy(strategy, data)
        
        # Минимизация отрицательной метрики
        score = -calculate_optimization_score(trades, final_balance)
        
        return score
    
    # Оптимизация
    result = gp_minimize(
        func=objective,
        dimensions=list(param_bounds.values()),
        n_calls=n_iterations,
        random_state=42
    )
    
    return result
```

## Технологии

- **pandas** - обработка временных рядов
- **numpy** - численные расчеты
- **scipy** - статистические функции
- **matplotlib** - визуализация
- **seaborn** - продвинутая визуализация
- **scikit-optimize** - оптимизация

## Конфигурация

```yaml
backtesting:
  data:
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    assets: ["BTC/USDT", "ETH/USDT"]
    
  walk_forward:
    train_window: 252  # days
    validation_window: 63  # days
    test_window: 63  # days
    step_size: 21  # days
    
  simulation:
    initial_balance: 10000
    commission: 0.001
    slippage: 0.0005
    latency: 0.1  # seconds
    
  optimization:
    method: "bayesian"  # grid, bayesian, genetic
    objective: "sharpe_ratio"
    n_iterations: 100
    
  analysis:
    benchmark: "BTC/USDT"
    risk_free_rate: 0.02
    confidence_level: 0.95
```

## Визуализация результатов

### Equity Curve

График кривой капитала с отмеченными просадками.

### Drawdown Chart

График просадок с выделением максимальной.

### Returns Distribution

Распределение доходности с статистическими характеристиками.

### Monthly Returns

Тепловая карта месячных доходностей.

## Интеграция

Backtesting Layer получает данные от:
- **Data Layer** - исторические данные
- **Feature Engineering** - признаки для стратегий
- **Models Layer** - обученные модели

И передает результаты в:
- **Decision Layer** - валидированные стратегии
- **Meta-Learning** - метрики для адаптации

## Требования к реализации

1. **Точность** - корректная симуляция торговли
2. **Скорость** - быстрые вычисления для оптимизации
3. **Масштабируемость** - поддержка больших датасетов
4. **Гибкость** - легкое добавление новых метрик
5. **Визуализация** - понятные графики и отчеты

## Тестирование

- Unit тесты для каждого компонента
- Integration тесты для pipeline
- Validation тесты для корректности расчетов
- Performance тесты для скорости
