# Документация по Weighted Ensemble с Confidence Thresholding

## Обзор

В данном документе подробно описана реализация WeightedEnsemble с динамическим выбором модели и фильтрацией предсказаний на основе уверенности (confidence). Система использует комплексную метрику для оценки моделей и пороговую фильтрацию для повышения качества торговых сигналов.

---

## 1. Основная формула оценки моделей

### Score = α·Sharpe + β·PnL - γ·DD

Для каждой модели (f_i) рассчитывается комплексная оценка:

```
Score_i = α × Sharpe_i + β × PnL_i - γ × DD_i
```

Где:
- **Sharpe_i** — доходность/риск для модели i
- **PnL_i** — прибыль/убыток для модели i  
- **DD_i** — максимальная просадка для модели i
- **α, β, γ** — настраиваемые веса (по умолчанию: α=0.4, β=0.4, γ=0.2)

### Выбор лучшей модели

```
f* = argmax_i(Score_i)
```

---

## 2. Динамический выбор в реальном времени

### Sliding Window подход

Для адаптации к изменяющимся рыночным условиям используется скользящее окно:

```
Score_i(t) = metric over last N trades
```

- **N** — размер окна (по умолчанию 100 трейдов)
- **t** — текущий момент времени
- Метрики рассчитываются по последним N трейдам

### Логика переключения моделей

Если производительность текущей модели ухудшается:

```
if Score_current < Score_best - δ:
    → смена модели
```

Где **δ** — порог переключения (по умолчанию 0.1)

---

## 3. Confidence Thresholding по p_hat

### Концепция

Для фильтрации неуверенных предсказаний используется threshold по предсказанным вероятностям (p_hat):

```
confidence = calculate_confidence(p_hat)
if confidence < confidence_threshold:
    prediction = HOLD (класс 1)
else:
    prediction = argmax(p_hat)
```

### Стратегии расчета уверенности

#### 1. Max Probability Strategy (`max_proba`)

```python
confidence = max(p_hat)
```

**Преимущества:**
- Простота расчета
- Интуитивно понятная метрика
- Хорошие результаты для уверенных предсказаний

**Недостатки:**
- Не учитывает распределение вероятностей
- Чувствительность к выбросам

#### 2. Entropy-based Strategy (`entropy`)

```python
entropy = -sum(p_hat * log(p_hat))
max_entropy = log(num_classes)
confidence = 1 - (entropy / max_entropy)
```

**Преимущества:**
- Учитывает все вероятности
- Более робастная к шуму
- Теоретически обоснована

**Недостатки:**
- Большая вычислительная сложность
- Менее интуитивная интерпретация

#### 3. Margin-based Strategy (`margin`)

```python
sorted_probs = sort(p_hat)
confidence = sorted_probs[-1] - sorted_probs[-2]
```

**Преимущества:**
- Учитывает конкуренцию между классами
- Хорошо работает при близких вероятностях
- Стабильные результаты

**Недостатки:**
- Игнорирует остальные вероятности
- Плохо работает для 2-классовых задач

---

## 4. Архитектура реализации

### WeightedEnsembleConfig

```python
@dataclass
class WeightedEnsembleConfig:
    # Score formula weights
    alpha: float = 0.4          # Вес для Sharpe ratio
    beta: float = 0.4            # Вес для PnL
    gamma: float = 0.2           # Вес для Drawdown (штраф)
    
    # Dynamic selection parameters
    window_size: int = 100               # Размер скользящего окна
    min_trades_for_switch: int = 20     # Минимум трейдов для переключения
    switch_threshold: float = 0.1          # Порог ухудшения
    
    # Smoothing parameters
    score_smoothing: bool = True
    smoothing_window: int = 10
    
    # Confidence filtering parameters
    confidence_threshold: float = 0.6      # Порог уверенности
    use_confidence_filter: bool = True       # Включение фильтрации
    confidence_strategy: str = "max_proba"     # Стратегия расчета
    
    # Ensemble weights
    ensemble_weights: Optional[List[float]] = None
```

### ModelPerformance

```python
@dataclass
class ModelPerformance:
    model_name: str
    
    # История метрик (deque с maxlen=200)
    sharpe_history: deque
    pnl_history: deque
    drawdown_history: deque
    score_history: deque
    
    # Текущие показатели
    current_score: float = 0.0
    best_score: float = 0.0
    trades_count: int = 0
```

### WeightedEnsemble

Основной класс с методами:

#### Core Methods
- `fit(X, y)` — обучение всех моделей
- `predict(X)` — взвешенное предсказание
- `predict_proba(X)` — взвешенные вероятности

#### Confidence Methods
- `predict_with_confidence(X)` — предсказания с уверенностью
- `_calculate_confidence(probas)` — расчет уверенности по стратегии
- `_filter_by_confidence(probas, confidence)` — фильтрация по порогу
- `get_confidence_stats(X)` — статистика уверенности

#### Performance Methods
- `update_performance(y_true, y_pred, price_returns)` — обновление метрик
- `_check_and_switch_model()` — проверка и переключение моделей
- `get_model_rankings()` — рейтинг моделей
- `get_performance_summary()` — сводная статистика

---

## 5. Примеры использования

### Базовое использование

```python
from its_project.metalearning import WeightedEnsemble, WeightedEnsembleConfig

# Создание конфигурации
config = WeightedEnsembleConfig(
    alpha=0.4, beta=0.4, gamma=0.2,
    confidence_threshold=0.7,
    confidence_strategy="max_proba"
)

# Создание ансамбля
models = [model1, model2, model3]
ensemble = WeightedEnsemble(models, config)

# Обучение
ensemble.fit(X_train, y_train)

# Предсказание с уверенностью
predictions, confidence = ensemble.predict_with_confidence(X_test)
```

### Продвинутое использование

```python
# Разные стратегии уверенности
strategies = ["max_proba", "entropy", "margin"]
thresholds = [0.5, 0.7, 0.9]

for strategy in strategies:
    for threshold in thresholds:
        config = WeightedEnsembleConfig(
            confidence_strategy=strategy,
            confidence_threshold=threshold
        )
        
        ensemble = WeightedEnsemble(models, config)
        ensemble.fit(X_train, y_train)
        
        # Анализ результатов
        stats = ensemble.get_confidence_stats(X_test)
        print(f"Strategy: {strategy}, Threshold: {threshold}")
        print(f"HOLD ratio: {1 - stats['threshold_ratio']:.1%}")
```

---

## 6. Параметры оптимизации

### Рекомендуемые настройки

#### Для криптовалютных рынков (высокая волатильность):
```python
config = WeightedEnsembleConfig(
    alpha=0.3,      # Меньший вес Sharpe
    beta=0.5,        # Больший вес PnL
    gamma=0.2,       # Умеренный штраф за просадку
    confidence_threshold=0.65,  # Средний порог уверенности
    window_size=50,    # Меньшее окно для быстрой адаптации
    switch_threshold=0.08  # Более чувствительное переключение
)
```

#### Для фондовых рынков (низкая волатильность):
```python
config = WeightedEnsembleConfig(
    alpha=0.5,      # Больший вес Sharpe
    beta=0.3,        # Меньший вес PnL
    gamma=0.2,       # Стандартный штраф за просадку
    confidence_threshold=0.75,  # Высокий порог уверенности
    window_size=200,   # Большее окно для стабильности
    switch_threshold=0.12  # Менее чувствительное переключение
)
```

### Настройка весов ансамбля

```python
# Для акцентирования на лучших моделях
ensemble_weights = [0.6, 0.3, 0.1]  # Модель 1 имеет 60% веса

# Для равномерного распределения
ensemble_weights = [1.0/3, 1.0/3, 1.0/3]  # Равные веса
```

---

## 7. Метрики и мониторинг

### Ключевые метрики производительности

#### Model-Level Metrics:
- **Current Score**: текущий Score по формуле
- **Best Score**: лучший исторический Score
- **Window Score**: Score за последнее окно
- **Trades Count**: количество обработанных трейдов

#### Confidence Metrics:
- **Mean Confidence**: средняя уверенность предсказаний
- **Confidence Distribution**: распределение по классам
- **Threshold Ratio**: доля предсказаний выше порога
- **Filtering Impact**: влияние фильтрации на HOLD ratio

#### Trading Metrics:
- **Sharpe Ratio**: доходность/риск
- **Total PnL**: общая прибыль/убыток
- **Max Drawdown**: максимальная просадка
- **Win Rate**: процент прибыльных трейдов

### Пример мониторинга

```python
# Получение полной статистики
summary = ensemble.get_performance_summary()

print(f"Active model: {summary['current_model']}")
print(f"Total trades: {summary['total_trades']}")

for model_name, metrics in summary['models'].items():
    print(f"{model_name}:")
    print(f"  Score: {metrics['current_score']:.4f}")
    print(f"  Sharpe: {metrics['window_sharpe']:.4f}")
    print(f"  PnL: {metrics['window_pnl']:.4f}")
    print(f"  Drawdown: {metrics['window_drawdown']:.4f}")
```

---

## 8. Интеграция с торговой системой

### Signal Generation

```python
class TradingSignalGenerator:
    def __init__(self, ensemble: WeightedEnsemble):
        self.ensemble = ensemble
    
    def generate_signal(self, features: np.ndarray) -> Dict:
        # Получение предсказаний с уверенностью
        predictions, confidence = self.ensemble.predict_with_confidence(features)
        
        # Формирование торгового сигнала
        signal = {
            'action': self._prediction_to_action(predictions[-1]),
            'confidence': confidence[-1],
            'model': self.ensemble.current_model_name,
            'filtered': confidence[-1] < self.ensemble.config.confidence_threshold
        }
        
        return signal
    
    def _prediction_to_action(self, prediction: int) -> str:
        if prediction == 0:
            return "SELL"
        elif prediction == 2:
            return "BUY"
        else:
            return "HOLD"
```

### Risk Management Integration

```python
class RiskAwareEnsemble:
    def __init__(self, ensemble: WeightedEnsemble, risk_params: Dict):
        self.ensemble = ensemble
        self.risk_params = risk_params
    
    def get_adjusted_signal(self, features: np.ndarray) -> Dict:
        predictions, confidence = self.ensemble.predict_with_confidence(features)
        
        # Корректировка на основе риска
        if confidence < self.risk_params['min_confidence']:
            return {'action': 'HOLD', 'reason': 'Low confidence'}
        
        if self.ensemble.get_current_drawdown() > self.risk_params['max_drawdown']:
            return {'action': 'HOLD', 'reason': 'Max drawdown exceeded'}
        
        return {
            'action': self._prediction_to_action(predictions[-1]),
            'confidence': confidence[-1],
            'risk_adjusted': True
        }
```

---

## 9. Тестирование и валидация

### Unit Tests

Структура тестов:
- `test_weighted_ensemble.py` — основные функции ансамбля
- `test_confidence_thresholding.py` — фильтрация по уверенности

### Ключевые тестовые сценарии

#### Score Formula Tests:
```python
def test_score_calculation():
    # Тест формулы Score = α·Sharpe + β·PnL - γ·DD
    config = WeightedEnsembleConfig(alpha=0.4, beta=0.4, gamma=0.2)
    perf = ModelPerformance("TestModel")
    
    perf.update_performance(sharpe=2.0, pnl=0.1, drawdown=0.05, config=config)
    expected = 0.4*2.0 + 0.4*0.1 - 0.2*0.05  # = 0.83
    
    assert abs(perf.current_score - expected) < 0.001
```

#### Confidence Strategy Tests:
```python
def test_confidence_strategies():
    # Тест всех стратегий расчета уверенности
    probas = np.array([[0.1, 0.8, 0.1], [0.33, 0.34, 0.33]])
    
    # Max probability
    confidence_max = ensemble._calculate_confidence(probas, "max_proba")
    assert confidence_max[0] == 0.8
    
    # Entropy-based
    confidence_entropy = ensemble._calculate_confidence(probas, "entropy")
    assert confidence_entropy[0] > confidence_entropy[1]  # Low entropy > high entropy
```

### Integration Tests

```python
def test_end_to_end_workflow():
    # Полный тест рабочего процесса
    models = create_test_models()
    config = WeightedEnsembleConfig(
        confidence_threshold=0.7,
        confidence_strategy="max_proba"
    )
    
    ensemble = WeightedEnsemble(models, config)
    ensemble.fit(X_train, y_train)
    
    # Симуляция реальной торговли
    for batch in trading_data:
        predictions, confidence = ensemble.predict_with_confidence(batch.features)
        ensemble.update_performance(batch.y_true, predictions, batch.returns)
    
    # Проверка переключения моделей
    assert ensemble.total_trades > 0
    assert len(ensemble.get_model_rankings()) == len(models)
```

---

## 10. Производительность и оптимизация

### Вычислительная сложность

| Операция | Сложность | Описание |
|-------------|-------------|------------|
| `predict_proba` | O(M × C × K) | M моделей, C классов, K признаков |
| `calculate_confidence` | O(N × C) | N сэмплов, C классов |
| `update_performance` | O(1) | Константное время |
| `model_switching` | O(M) | M моделей |

### Оптимизации

#### Memory Optimization:
- Использование `deque` с ограниченным размером
- Эффективное хранение истории метрик
- Garbage collection для старых данных

#### Compute Optimization:
- Векторизованные операции NumPy
- Кэширование частых расчетов
- Параллельная обработка моделей

### Рекомендации по производительности

1. **Размер окна**: 50-200 для баланса адаптивности/стабильности
2. **Частота обновления**: Каждые 10-50 трейдов
3. **Кэширование**: Предрассчитанные статистики
4. **Параллелизация**: Независимое обучение моделей

---

## 11. Best Practices

### Конфигурация

```python
# Рекомендуемая базовая конфигурация
config = WeightedEnsembleConfig(
    # Сбалансированные веса
    alpha=0.4, beta=0.4, gamma=0.2,
    
    # Адаптивность к рынку
    window_size=100,
    switch_threshold=0.1,
    
    # Фильтрация уверенности
    confidence_threshold=0.65,
    confidence_strategy="max_proba",
    use_confidence_filter=True,
    
    # Сглаживание для стабильности
    score_smoothing=True,
    smoothing_window=10
)
```

### Мониторинг

```python
# Регулярный мониторинг производительности
def monitor_ensemble(ensemble, test_data):
    stats = ensemble.get_confidence_stats(test_data)
    
    # Алерты для проблем
    if stats['mean_confidence'] < 0.5:
        logger.warning("Low average confidence detected")
    
    if stats['below_threshold'] / len(test_data) > 0.5:
        logger.warning("High filtering rate detected")
    
    # Логирование переключений
    current_model = ensemble.current_model_name
    if hasattr(ensemble, '_last_model'):
        if current_model != ensemble._last_model:
            logger.info(f"Model switched: {ensemble._last_model} -> {current_model}")
    
    ensemble._last_model = current_model
```

### Обработка ошибок

```python
def safe_predict(ensemble, features):
    try:
        predictions, confidence = ensemble.predict_with_confidence(features)
        
        # Валидация результатов
        if len(predictions) != len(features):
            raise ValueError("Prediction length mismatch")
        
        if not all(0 <= c <= 1 for c in confidence):
            raise ValueError("Invalid confidence values")
        
        return predictions, confidence
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Fallback к простой стратегии
        return np.full(len(features), 1), np.zeros(len(features))  # All HOLD
```

---

## 12. Заключение

WeightedEnsemble с confidence thresholding предоставляет мощную и гибкую систему для:

✅ **Динамической оптимизации моделей** на основе комплексной метрики  
✅ **Адаптации к рыночным условиям** через sliding windows  
✅ **Фильтрации неуверенных предсказаний** для снижения рисков  
✅ **Настраиваемых стратегий** под разные типы рынков  
✅ **Мониторинга производительности** в реальном времени  

Система готова к использованию в продакшене и может быть легко интегрирована с существующими торговыми стратегиями.

---

## 13. Финальная логика в Decision Module

### Реализованная логика

В `d:\IST\its_project\decision\` реализована финальная логика согласно требованиям:

#### Основная формула:
```python
action = sign(ΔP_hat)
```

#### Фильтрация:
```python
если p_hat < threshold → HOLD
если |ΔP_hat| < min_move → HOLD
```

### Созданные файлы:

#### 1. `enhanced_decision.py` - Улучшенный Decision Maker:
- **EnhancedDecisionMaker**: Основная реализация с полной логикой
- **ConfidenceBasedDecisionMaker**: Альтернативная реализация фокусированная на уверенности

#### 2. Модифицированный `signal_generator.py`:
- Поддержка ΔP_hat в metadata сигналов
- Метод `_delta_p_to_action()` для конвертации знака ΔP_hat
- Улучшенный расчет уверенности на основе ΔP_hat

#### 3. `test_enhanced_decision.py` - Полные тесты:
- Тестирование логики sign(ΔP_hat)
- Тестирование фильтрации по p_hat threshold
- Тестирование фильтрации по минимальному движению
- Интеграционные тесты с WeightedEnsemble

### Ключевые особенности реализации:

#### Enhanced Decision Maker:
- **Конфигурируемые параметры**: confidence_threshold, min_price_move, use_price_sign_logic
- **Умный размер позиции**: учитывает уверенность и величину ΔP_hat
- **Динамические риски**: адаптация stop loss/take profit под ΔP_hat
- **Детальное логирование**: причин принятия решений

#### Signal Generator Integration:
- **ΔP_hat metadata**: передача предсказанного изменения цены
- **p_hat extraction**: максимальная вероятность как уверенность
- **Комбинированная уверенность**: учет классификации и величины ΔP_hat
- **Обратная совместимость**: работа с существующими моделями

#### Тестовое покрытие:
- ✅ Базовые функции и инициализация
- ✅ Логика конвертации знака ΔP_hat
- ✅ Фильтрация по уверенности и движению
- ✅ Расчет размера позиции и рисков
- ✅ Интеграционные сценарии
- ✅ Обновление конфигурации

---

## 15. Расширенные фильтры: Cooldown и Volatility

### Cooldown Filter

#### Концепция
Cooldown фильтр предотвращает слишком частые торговые сигналы для одного и того же актива, обеспечивая период "остывания" между сделками.

#### Реализация
```python
# Конфигурация
cooldown_enabled = True
cooldown_period = 300  # 5 минут в секундах
last_signal_time = {}  # Отслеживание по символам

# Логика работы
if time_since_last_signal < cooldown_period:
    return HOLD (reason: "Cooldown active for {symbol}")
else:
    allow signal and update_cooldown()
```

#### Преимущества
- **Защита от overtrading**: предотвращает избыточную торговую активность
- **Per-symbol tracking**: независимый cooldown для каждого актива
- **Настраиваемый период**: адаптация под разные торговые стратегии
- **Только реальные сделки**: обновляется только для non-HOLD действий

#### Параметры конфигурации
```python
{
    "cooldown_enabled": True,           # Включение фильтра
    "cooldown_period": 300,            # Период в секундах
}
```

### Volatility Filter

#### Концепция
Volatility фильтр блокирует торговые сигналы в периоды высокой рыночной волатильности, когда предсказания моделей становятся менее надежными.

#### Реализация
```python
# Конфигурация
volatility_filter_enabled = True
volatility_window = 20
max_volatility_threshold = 0.05  # 5% максимальная волатильность

# Расчет волатильности
returns = calculate_returns(price_history, window)
volatility = std(returns)

if volatility > max_volatility_threshold:
    return HOLD (reason: "Volatility too high: {vol:.4f} > {threshold}")
else:
    allow signal
```

#### Преимущества
- **Адаптация к рынку**: блокировка сигналов в турбулентные периоды
- **Снижение рисков**: избежание сделок в моменты высокой неопределенности
- **Исторический анализ**: отслеживание волатильности для оптимизации
- **Настраиваемый порог**: гибкая настройка под разные активы

#### Параметры конфигурации
```python
{
    "volatility_filter_enabled": True,    # Включение фильтра
    "volatility_window": 20,            # Окно для расчета волатильности
    "max_volatility_threshold": 0.05,   # Максимальная волатильность
}
```

### Интеграция в Enhanced Decision Maker

#### Обновленный процесс принятия решений
1. **Cooldown check** - проверка периода ожидания
2. **Volatility filter** - проверка волатильности рынка
3. **Confidence filtering** - фильтрация по уверенности
4. **Minimum movement filtering** - фильтрация по минимальному движению
5. **Action determination** - определение действия через sign(ΔP_hat)
6. **Cooldown update** - обновление времени последнего сигнала

#### Пример использования
```python
# Конфигурация с обоими фильтрами
config = {
    "confidence_threshold": 0.7,
    "min_price_move": 0.0005,
    "use_price_sign_logic": True,
    "cooldown_enabled": True,
    "cooldown_period": 300,
    "volatility_filter_enabled": True,
    "volatility_window": 20,
    "max_volatility_threshold": 0.05
}

decision_maker = EnhancedDecisionMaker(config)
decision = decision_maker.decide(signal, market_state)
```

### Тестирование фильтров

#### Тестовые сценарии
- ✅ **Первый сигнал**: должен разрешаться (нет истории)
- ✅ **Слишком быстрый повтор**: должен блокироваться cooldown
- ✅ **Сигнал после периода**: должен разрешаться
- ✅ **Нормальная волатильность**: сигнал разрешается
- ✅ **Высокая волатильность**: сигнал блокируется
- ✅ **Совместная работа**: оба фильтра вместе
- ✅ **Per-symbol tracking**: независимая работа с разными активами
- ✅ **История волатильности**: корректное отслеживание

#### Файлы тестов
- `test_enhanced_filters.py` - исчерпывающие тесты для обоих фильтров
- Покрытие всех граничных случаев и сценариев использования
- Интеграционные тесты с существующими компонентами

### Рекомендации по настройке

#### Cooldown параметры
- **Высокочастотные стратегии**: 60-300 секунд
- **Среднечастотные стратегии**: 300-900 секунд
- **Низкочастотные стратегии**: 900-3600 секунд

#### Volatility параметры
- **Криптовалюты**: 0.03-0.08 (3-8%)
- **Форекс**: 0.01-0.03 (1-3%)
- **Акции**: 0.02-0.05 (2-5%)

---

## 17. Fixed Percentage Position Sizing (1-2%)

### Концепция
В рамках risk management реализован механизм фиксированного процентного sizing (1-2%) для управления размером позиций. Это обеспечивает консервативный подход к управлению капиталом с предопределенными границами риска.

### Реализация

#### Новый метод sizing
```python
FIXED_PERCENTAGE = "fixed_percentage"  # Fixed 1-2% sizing
```

#### Параметры конфигурации
```python
# Fixed percentage sizing parameters (1-2%)
fixed_percentage_min: float = 0.01   # 1% minimum fixed percentage
fixed_percentage_max: float = 0.02   # 2% maximum fixed percentage
fixed_percentage_default: float = 0.015  # 1.5% default fixed percentage
```

#### Алгоритм расчета
```python
def _fixed_percentage_sizing(decision, portfolio_value):
    # Получение фиксированного процента (по умолчанию 1.5%)
    fixed_percentage = config.fixed_percentage_default
    
    # Ограничение в диапазоне 1-2%
    fixed_percentage = max(min(fixed_percentage, 0.02), 0.01)
    
    # Расчет размера позиции
    size_value = portfolio_value * fixed_percentage
    size_units = size_value / current_price
    
    # Учет существующих позиций
    if existing_position:
        fixed_percentage *= 0.5  # Уменьшение при добавлении
```

### Преимущества

#### Риск-менеджмент
- **Фиксированные границы**: 1-2% портфеля на позицию
- **Предсказуемый риск**: консервативный подход к управлению капиталом
- **Защита от over-leveraging**: жесткие ограничения размера позиций
- **Простота расчетов**: прозрачная и понятная логика

#### Гибкость настройки
- **Минимальный процент**: 1% для консервативных стратегий
- **Максимальный процент**: 2% для умеренно-агрессивных стратегий
- **По умолчанию**: 1.5% баланс между риском и доходностью
- **Адаптация под активы**: разные проценты для разных инструментов

### Интеграция с существующей системой

#### Совместимость с PositionSizer
```python
# Добавление в существующий enum
class SizingMethod(Enum):
    FIXED_FRACTION = "fixed_fraction"
    FIXED_PERCENTAGE = "fixed_percentage"  # Новый метод
    KELLY = "kelly"
    # ... другие методы
```

#### Расширенные возможности
- **Комбинированный подход**: использование fixed percentage как базы с другими методами
- **Risk-adjusted sizing**: интеграция с лимитами портфеля
- **Portfolio-level control**: учет общего риска всех позиций

### Примеры использования

#### Консервативная стратегия
```python
config = SizingConfig(
    method=SizingMethod.FIXED_PERCENTAGE,
    fixed_percentage_min=0.01,    # 1%
    fixed_percentage_max=0.015,    # 1.5%
    fixed_percentage_default=0.01    # 1%
)
```

#### Умеренная стратегия
```python
config = SizingConfig(
    method=SizingMethod.FIXED_PERCENTAGE,
    fixed_percentage_min=0.015,   # 1.5%
    fixed_percentage_max=0.02,    # 2%
    fixed_percentage_default=0.0175  # 1.75%
)
```

### Тестирование

#### Созданные тесты
- `test_fixed_percentage_sizing.py` - исчерпывающие тесты
- Тестирование всех диапазонов: 1%, 1.5%, 2%
- Проверка ограничений и адаптации
- Интеграционные тесты с другими компонентами

#### Ключевые тестовые сценарии
- ✅ Минимальный процент (1%): корректный расчет размера
- ✅ Максимальный процент (2%): корректный расчет размера
- ✅ По умолчанию (1.5%): стандартное поведение
- ✅ Ограничение диапазона: clamp в 1-2%
- ✅ Существующие позиции: уменьшение размера при добавлении
- ✅ Расчет риска: правильное вычисление risk amount
- ✅ Метаданные: полная информация о параметрах

---

## 18. Заключение

WeightedEnsemble с confidence thresholding, финальная логика decision, расширенные фильтры и fixed percentage sizing предоставляют комплексную систему:

✅ **Динамической оптимизации моделей** на основе Score = α·Sharpe + β·PnL - γ·DD  
✅ **Адаптации к рыночным условиям** через sliding windows  
✅ **Фильтрации неуверенных предсказаний** для снижения рисков  
✅ **Умной финальной логики** с sign(ΔP_hat) и порогами  
✅ **Защиты от overtrading** через cooldown механизм  
✅ **Адаптации к волатильности** через volatility фильтр  
✅ **Фиксированный процентный sizing** (1-2%) для риск-менеджмента  
✅ **Настраиваемых стратегий** под разные типы рынков  
✅ **Полной интеграции** между WeightedEnsemble и Decision модулями  
✅ **Комплексного тестирования** всех компонентов  

Система готова к продакшенному использованию и предоставляет мощные инструменты для автоматической торговли с многоуровневой защитой от рисков, адаптивной оптимизацией стратегий и консервативным управлением капиталом.

---

*Дата создания: 6 мая 2026 г.*  
*Автор: AI Assistant*  
*Версия: 4.0*
