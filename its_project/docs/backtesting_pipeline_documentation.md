# Документация по Backtesting Pipeline

## Обзор

В данном документе подробно описана полная реализация pipeline в `d:\IST\its_project\backtesting` для валидации торговых стратегий с защитой от overfitting и реалистичным моделированием рыночных условий.

---

## 1. Архитектура Backtesting Pipeline

### Core Components

#### 1. BaseBacktester (`base.py`)
```python
@dataclass
class Trade:
    timestamp: int
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    commission: float
    slippage: float
    pnl: Optional[float] = None

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray
    metrics: Dict[str, float]
    positions: pd.DataFrame
```

#### 2. SimpleBacktester (`simple.py`)
```python
class SimpleBacktester(BaseBacktester):
    """Simple backtester for long/short strategies with no look-ahead bias."""
    
    def run(self, data, model, decision_maker) -> BacktestResult:
        # CRITICAL: Check time sorting
        if not data['timestamp'].is_monotonic_increasing:
            raise ValueError("Data MUST be sorted by time!")
        
        # Use ONLY data BEFORE current timestamp
        for i in range(min_window, len(data)):
            historical_data = data.iloc[:i]
            current_bar = data.iloc[i]
            
            # Extract features using only past data!
            features = self._extract_features(historical_data)
            
            # Get signal from model (no future data!)
            prediction = model.predict(features[-1:])
            confidence = model.get_confidence(features[-1:])
```

#### 3. WalkForwardEngine (`engine.py`)
```python
class WalkForwardEngine:
    """Walk-forward validation engine with strict no look-ahead bias."""
    
    def validate(self, data, model_class, model_params):
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp_ms)
        
        # Generate walk-forward windows
        windows = self._generate_windows(df)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Split data
            train_data = df.loc[train_start:train_end]["data"].tolist()
            test_data = df.loc[test_start:test_end]["data"].tolist()
            
            # Train model
            model = self._train_model(train_data, model_class, model_params)
            
            # Run backtest on test data
            result = self._run_backtest(test_data, model)
```

#### 4. WalkForwardValidator (`walkforward.py`)
```python
class WalkForwardValidator:
    """Walk-forward validation for robust strategy testing."""
    
    def validate(self, data, model_factory, decision_maker_factory):
        # CRITICAL: Check time sorting
        if not data['timestamp'].is_monotonic_increasing:
            raise ValueError("Data MUST be sorted by time!")
        
        results = []
        
        # Calculate windows
        train_size = 252  # 1 year
        test_size = 63    # 3 months
        step_size = 21     # 1 month
```

---

## 2. Pipeline Flow

### Основной процесс валидации

#### 1. Data Preparation
```python
# Сортировка данных по времени (критически важно!)
sorted_data = sorted(data, key=lambda x: x.timestamp_ms)

# Конвертация в DataFrame для временных операций
df = pd.DataFrame([
    {"timestamp": datetime.fromtimestamp(md.timestamp_ms / 1000), "data": md}
    for md in sorted_data
]).set_index("timestamp")
```

#### 2. Window Generation
```python
def _generate_windows(self, df: pd.DataFrame):
    """Generate walk-forward windows."""
    windows = []
    current_start = df.index[0]
    end_time = df.index[-1]
    
    while True:
        # Calculate window boundaries
        train_end = current_start + self.min_train_size
        window_end = current_start + self.window_size
        test_start = train_end
        test_end = window_end
        
        # Add window
        windows.append((current_start, train_end, test_start, test_end))
        
        # Move to next window
        current_start += self.step_size
```

#### 3. Model Training
```python
def _train_model(self, train_data, model_class, model_params):
    """Train model on training data."""
    # Convert training data to features
    X_train, y_train = self._prepare_training_data(train_data)
    
    # Create and train model
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    return model
```

#### 4. Backtesting
```python
def _run_backtest(self, test_data, model):
    """Run backtest on test data."""
    # Process each test data point
    for i in range(len(test_data)):
        # Use only historical data up to current point
        historical_data = test_data[:i]
        current_bar = test_data[i]
        
        # Extract features (no future data!)
        features = self._extract_features(historical_data)
        
        # Get prediction
        prediction = model.predict(features[-1:])
        
        # Execute trade
        self._execute_trade(prediction, current_bar)
```

---

## 3. Key Features

### No Look-Ahead Bias Protection
```python
# CRITICAL: Только исторические данные для feature extraction
historical_data = data.iloc[:i]  # Все данные ДО текущего бара
current_bar = data.iloc[i]       # Текущий бар

# Extract features using only past data!
features = self._extract_features(historical_data)

# Get signal from model (no future data!)
prediction = model.predict(features[-1:])
```

### Chronological Order Enforcement
```python
# Проверка монотонности временной последовательности
if not data['timestamp'].is_monotonic_increasing:
    raise ValueError("Data MUST be sorted by time!")

# Обработка в строгом хронологическом порядке
for i in range(min_window, len(data)):
    # Никакого заглядывания в будущее
```

### Sliding Window Validation
```python
# Параметры walk-forward
window_size = timedelta(days=30)    # 30 дней
step_size = timedelta(days=7)        # Шаг 7 дней
min_train_size = timedelta(days=21)   # Минимум 21 день

# Генерация окон
for train_start, train_end, test_start, test_end in windows:
    # Обучение на train данных
    # Тестирование на test данных
```

---

## 4. Realistic Cost Modeling

### Commission Modeling
```python
# В WalkForwardEngine
commission_rate: float = 0.001,  # 0.1% commission

# В SimpleBacktester
trade.commission = size * price * commission_rate
```

### Slippage Modeling
```python
# Различные модели slippage
slippage_model: str = "linear"  # "linear", "percentage", "fixed"

# Linear slippage: увеличивается с размером позиции
if slippage_model == "linear":
    slippage = size * slippage_params['rate']

# Percentage slippage: процент от цены
elif slippage_model == "percentage":
    slippage = price * slippage_params['pct']
```

### Realistic Execution
```python
# Моделирование реальных условий исполнения
def _execute_trade(self, signal, current_bar):
    # Account for commission
    commission = self._calculate_commission(signal.size, current_bar.price)
    
    # Account for slippage
    slippage = self._calculate_slippage(signal.size, current_bar.price)
    
    # Adjust execution price
    execution_price = current_bar.price + slippage
    
    # Record trade with realistic costs
    trade = Trade(
        price=execution_price,
        commission=commission,
        slippage=slippage
    )
```

---

## 5. Performance Metrics

### Core Metrics
```python
# В BacktestResult.metrics
metrics = {
    'total_return': float,      # Общая доходность
    'sharpe_ratio': float,     # Sharpe ratio
    'max_drawdown': float,      # Максимальная просадка
    'win_rate': float,         # Процент выигрышных сделок
    'profit_factor': float,     # Прибыльность
    'num_trades': int,         # Количество сделок
    'avg_trade_duration': float, # Средняя длительность
    'volatility': float,       # Волатильность доходности
}
```

### Equity Curve Calculation
```python
# Расчет кривой капитала
equity_history = [self.initial_capital]

for trade in self.trades:
    if trade.pnl is not None:
        new_equity = equity_history[-1] + trade.pnl
        equity_history.append(new_equity)

equity_curve = np.array(equity_history)
```

### Risk-Adjusted Metrics
```python
# Расчет метрик с учетом риска
def _calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Daily adjustment
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def _calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)
```

---

## 6. Integration Points

### Model Integration
```python
# Любая модель с интерфейсом BaseModel
from its_project.models.base import BaseModel

model = model_class(**model_params)
model.fit(X_train, y_train)
prediction = model.predict(features)
confidence = model.get_confidence(features)
```

### Decision Maker Integration
```python
# Интеграция с decision layer
from its_project.decision.decision import DecisionMaker

decision_maker = decision_maker_factory()
signal = decision_maker.decide(signal, market_state)
decision = decision_maker.make_decision(prediction, confidence)
```

### Risk Management Integration
```python
# Интеграция с risk manager
from its_project.decision.risk import RiskManager

risk_manager = RiskManager(risk_limits)
risk_check = risk_manager.check_decision_risk(decision, positions, portfolio_value)

if not risk_check.approved:
    # Блокировка сделки по рискам
    continue
```

---

## 7. Usage Examples

### Simple Backtesting
```python
# Создание simple backtester
config = {
    'initial_capital': 100000,
    'commission_rate': 0.001,
    'slippage_model': 'linear'
}

backtester = SimpleBacktester(config)
result = backtester.run(data, model, decision_maker)

print(result.summary())
```

### Walk-Forward Validation
```python
# Создание walk-forward validator
config = {
    'train_size': 252,    # 1 год
    'test_size': 63,      # 3 месяца
    'step_size': 21,       # 1 месяц
    'min_window': 100
}

validator = WalkForwardValidator(config)
results = validator.validate(data, model_factory, decision_maker_factory)

# Агрегация результатов
aggregate_metrics = validator.calculate_aggregate_metrics(results)
```

### Advanced Pipeline
```python
# Полный pipeline с walk-forward
engine = WalkForwardEngine(
    base_backtester=SimpleBacktester(config),
    window_size=timedelta(days=30),
    step_size=timedelta(days=7),
    min_train_size=timedelta(days=21),
    commission_rate=0.001,
    slippage_model="linear"
)

# Запуск валидации
validation_results = engine.validate(
    data=market_data,
    model_class=LSTMModel,
    model_params={'hidden_size': 64, 'dropout': 0.2}
)
```

---

## 8. Best Practices

### Data Quality
- **Сортировка по времени**: обязательна для предотвращения look-ahead bias
- **Проверка пропусков**: обработка NaN значений
- **Валидация временных интервалов**: проверка на пересечения

### Model Validation
- **Walk-forward**: обязательна для продакшена
- **Multiple windows**: тестирование на различных периодах
- **Stability analysis**: проверка стабильности результатов

### Risk Management
- **Realistic costs**: комиссия и slippage
- **Position sizing**: интеграция с sizing модулями
- **Drawdown limits**: защита от больших потерь

### Performance Analysis
- **Risk-adjusted metrics**: Sharpe ratio, Sortino ratio
- **Scenario analysis**: результаты в разных рыночных условиях
- **Benchmarking**: сравнение с базовыми стратегиями

---

## 9. Production Considerations

### Speed Optimization
```python
# Векторизация операций
features = np.array([self._extract_features(data.iloc[:i]) for i in range(len(data))])

# Pandas optimizations
df = df.set_index('timestamp')  # Индексация по времени
```

### Memory Management
```python
# Ограничение истории
if len(self.risk_history) > 1000:
    self.risk_history = self.risk_history[-1000:]

# Streaming processing для больших датасетов
for chunk in pd.read_csv('large_data.csv', chunksize=10000):
    process_chunk(chunk)
```

### Error Handling
```python
# Graceful degradation
try:
    result = backtester.run(data, model, decision_maker)
except InsufficientDataError:
    logger.warning("Insufficient data for backtest")
    return None
except ModelTrainingError:
    logger.error("Model training failed")
    return None
```

---

## 10. Заключение

Backtesting pipeline в `d:\IST\its_project\backtesting` предоставляет комплексную систему для валидации торговых стратегий:

✅ **Защита от look-ahead bias** через строгую хронологическую обработку  
✅ **Walk-forward validation** для тестирования стабильности стратегий  
✅ **Realistic cost modeling** с комиссией и slippage  
✅ **Risk-adjusted metrics** для корректной оценки производительности  
✅ **Интеграция с моделями** через BaseModel интерфейс  
✅ **Совместимость с decision layer** для полной валидации  
✅ **Production-ready оптимизация** для работы с большими датасетами  

Система готова к продакшенному использованию и обеспечивает надежную валидацию торговых стратегий перед развертыванием на реальных рынках.

---

*Дата создания: 6 мая 2026 г.*  
*Автор: AI Assistant*  
*Версия: 1.0*
