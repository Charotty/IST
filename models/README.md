# Models Layer

## Назначение

Модели машинного обучения для прогнозирования рыночного поведения и генерации торговых сигналов.

## Основные задачи

- Классификация направления движения цены
- Регрессия для предсказания доходности
- Оценка вероятностей торговых сигналов
- Ансамблевое комбинирование моделей

## Типы задач

### 1. Классификация

Предсказание направления движения:

```python
y_t = {
    BUY,    # r_{t+h} > τ
    SELL,   # r_{t+h} < -τ  
    HOLD    # |r_{t+h}| ≤ τ
}
```

### 2. Регрессия

Предсказание доходности:

```python
ΔP̂_t = f(X_t)
```

### 3. Оценка вероятностей

Вероятность торгового сигнала:

```python
P(BUY) = f(X_t)
P(SELL) = f(X_t)  
P(HOLD) = f(X_t)
```

## Используемые модели

### 1. Базовые модели (Boosting)

#### LightGBM
- Быстрое обучение
- Хорошая производительность на табличных данных
- Встроенная обработка пропусков

#### XGBoost
- Высокая точность
- Регуляризация для предотвращения overfitting
- Параллельное обучение

### 2. Временные модели (Deep Learning)

#### GRU (Gated Recurrent Unit)
- Эффективная обработка временных последовательностей
- Меньше параметров чем LSTM
- Быстрое обучение

#### LSTM (Long Short-Term Memory)
- Долгосрочные зависимости
- Memory cells для информации
- Хорошо для долгосрочных паттернов

#### Transformer
- Attention mechanism
- Параллельная обработка последовательностей
- Хорошо для сложных зависимостей

### 3. Специализированные модели

#### CNN-LOB
- Сверточные сети для order book
- Извлечение пространственных паттернов
- Хорошо для микроструктурных данных

## Структура модуля

```
models/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── base_model.py         # Базовый класс модели
│   ├── model_interface.py    # Интерфейс моделей
│   └── model_utils.py       # Утилиты моделей
├── boosting/
│   ├── __init__.py
│   ├── lightgbm_model.py     # LightGBM реализация
│   ├── xgboost_model.py      # XGBoost реализация
│   └── boosting_utils.py     # Утилиты boosting
├── deep_learning/
│   ├── __init__.py
│   ├── gru_model.py          # GRU реализация
│   ├── lstm_model.py         # LSTM реализация
│   ├── transformer_model.py  # Transformer реализация
│   └── neural_utils.py       # Утилиты нейросетей
├── specialized/
│   ├── __init__.py
│   ├── cnn_lob_model.py      # CNN для Order Book
│   └── ensemble_model.py     # Ансамбль моделей
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Обучение моделей
│   ├── validator.py          # Валидация моделей
│   └── hyperparameter_tuner.py # Оптимизация гиперпараметров
├── registry/
│   ├── __init__.py
│   ├── model_registry.py     # Реестр моделей
│   ├── version_manager.py    # Управление версиями
│   └── model_loader.py       # Загрузка моделей
└── model_manager.py          # Главный менеджер моделей
```

## Ключевые компоненты

### BaseModel

Абстрактный базовый класс для всех моделей:
- Стандартизация интерфейсов
- Общие методы обучения и предсказания
- Валидация и логирование

### ModelManager

Центральный компонент управления моделями:
- Координация всех моделей
- Управление жизненным циклом
- Версионирование моделей
- Мониторинг производительности

### ModelRegistry

Реестр моделей с версионированием:
- Сохранение/загрузка моделей
- Метаданные моделей
- Сравнение версий

### Trainer

Универсальный тренер моделей:
- Обучение с валидацией
- Early stopping
- Callbacks для мониторинга

## Архитектура моделей

### GRU Architecture

```python
class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
```

### Transformer Architecture

```python
class TransformerModel(BaseModel):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers
        )
        self.output_projection = nn.Linear(d_model, output_dim)
```

## Обучение и валидация

### Walk-forward Validation

```python
def walk_forward_validation(model, data, window_size, step_size):
    for i in range(0, len(data) - window_size, step_size):
        train_data = data[i:i+window_size]
        test_data = data[i+window_size:i+window_size+step_size]
        
        model.fit(train_data)
        predictions = model.predict(test_data)
        
        # Оценка производительности
        evaluate(predictions, test_data.target)
```

### Cross-validation

```python
def time_series_cross_validation(model, data, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, val_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        model.fit(train_data)
        predictions = model.predict(val_data)
```

## Метрики оценки

### Классификация

- Accuracy
- Precision/Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Регрессия

- MSE/RMSE
- MAE
- R²
- MAPE

### Финансовые метрики

- Sharpe Ratio
- Max Drawdown
- Profit Factor
- Win Rate

## Технологии

- **PyTorch** - deep learning фреймворк
- **LightGBM** - gradient boosting
- **XGBoost** - gradient boosting
- **scikit-learn** - метрики и утилиты
- **optuna** - оптимизация гиперпараметров
- **mlflow** - эксперименты и версионирование

## Конфигурация

```yaml
models:
  boosting:
    lightgbm:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 100
      early_stopping_rounds: 10
    
    xgboost:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
      subsample: 0.8
  
  deep_learning:
    gru:
      hidden_dim: 128
      num_layers: 2
      dropout: 0.2
      learning_rate: 0.001
    
    transformer:
      d_model: 256
      nhead: 8
      num_layers: 4
      dropout: 0.1
  
  training:
    validation_split: 0.2
    early_stopping_patience: 10
    batch_size: 32
    epochs: 100
```

## Оптимизация гиперпараметров

### Bayesian Optimization

Использование Optuna для оптимального поиска гиперпараметров.

### Grid Search

Систематический перебор параметров для небольших пространств.

### Random Search

Случайный поиск для больших пространств параметров.

## Интеграция

Models Layer получает данные от:
- **Feature Engineering** - признаки для обучения
- **Meta-Learning** - адаптивные параметры

И передает предсказания в:
- **Meta-Learning** - для ансамблирования
- **Decision Layer** - для принятия решений

## Требования к реализации

1. **Производительность** - быстрые предсказания в real-time
2. **Масштабируемость** - поддержка множественных моделей
3. **Надежность** - обработка ошибок и fallback
4. **Версионирование** - контроль версий моделей
5. **Мониторинг** - отслеживание производительности

## Тестирование

- Unit тесты для каждой модели
- Integration тесты для pipeline
- Performance тесты для скорости
- Validation тесты для корректности предсказаний
