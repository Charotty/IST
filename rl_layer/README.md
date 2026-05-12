# Reinforcement Learning Layer

## Назначение

Оптимизация поведения торговой системы с помощью reinforcement learning для управления позицией и исполнением стратегий.

## Основная концепция

RL используется **не** для предсказания рынка, а для **оптимизации поведения системы** на основе предсказаний от ML моделей.

## Observation Space

Наблюдения включают всю релевантную информацию:

```python
o_t = [
    features,        # Технические признаки
    probabilities,   # Вероятности от ML моделей
    volatility,      # Текущая волатильность
    position,        # Текущая позиция
    balance,         # Баланс счета
    PnL,            # Прибыль/убыток
    regime,          # Рыночный режим
    time_features    # Временные признаки
]
```

## Action Space

### Дискретный Action Space

```python
a_t ∈ {BUY, SELL, HOLD}
```

Простые дискретные действия для базовых стратегий.

### Непрерывный Action Space

```python
a_t ∈ [-1, 1]
```

Непрерывные значения для:
- -1: полная короткая позиция
- 0: отсутствие позиции  
- 1: полная длинная позиция

### Multi-dimensional Action Space

```python
a_t = [position_size, stop_loss, take_profit]
```

Сложные действия с параметрами управления риском.

## Reward Function

Комплексная функция вознаграждения:

```python
R_t = PnL_t - λ * Drawdown_t - γ * Costs_t
```

где:
- **PnL_t**: прибыль/убыток за период
- **Drawdown_t**: максимальная просадка
- **Costs_t**: торговые издержки
- **λ, γ**: коэффициенты регуляризации

### Components of Reward

#### Profit Component

```python
profit_reward = (current_value - previous_value) / previous_value
```

#### Risk Component

```python
risk_penalty = λ * max_drawdown + γ * volatility_penalty
```

#### Cost Component

```python
cost_penalty = transaction_costs + slippage_costs
```

## RL Алгоритмы

### PPO (Proximal Policy Optimization)

**Рекомендуемый основной алгоритм:**

- Стабильность обучения
- Хорошая работа с noisy environments
- Простота реализации
- Эффективное использование данных

### SAC (Soft Actor-Critic)

Для непрерывных action spaces:
- Максимальная энтропия
- Стабильное обучение
- Хорошая эксплорация

### DQN (Deep Q-Network)

Для дискретных действий:
- Простота интерпретации
- Хорошие базовые результаты
- Стабильное обучение

### A2C (Advantage Actor-Critic)

Легковесный вариант:
- Быстрое обучение
- Низкие требования к ресурсам
- Хорош для prototyping

## Структура модуля

```
rl_layer/
├── __init__.py
├── environments/
│   ├── __init__.py
│   ├── trading_env.py         # Основная среда торговли
│   ├── portfolio_env.py       # Среда управления портфелем
│   └── custom_envs.py         # Специализированные среды
├── agents/
│   ├── __init__.py
│   ├── ppo_agent.py           # PPO агент
│   ├── sac_agent.py           # SAC агент
│   ├── dqn_agent.py           # DQN агент
│   └── a2c_agent.py           # A2C агент
├── training/
│   ├── __init__.py
│   ├── rl_trainer.py          # Тренер RL агентов
│   ├── experience_buffer.py    # Буфер опыта
│   └── curriculum.py          # Curriculum learning
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # Оценка агентов
│   ├── metrics.py             # Метрики оценки
│   └── visualizer.py          # Визуализация результатов
├── integration/
│   ├── __init__.py
│   ├── tensortrade_adapter.py # Адаптер TensorTrade
│   ├── gym_adapter.py         # Адаптер Gymnasium
│   └── stable_baselines_adapter.py # Адаптер Stable-Baselines3
└── rl_manager.py              # Главный менеджер RL
```

## Ключевые компоненты

### TradingEnvironment

Основная среда для RL обучения:
- Симуляция торговли
- Управление состоянием
- Расчет вознаграждений
- Интеграция с рыночными данными

### RLAgent

Базовый класс для всех RL агентов:
- Стандартизация интерфейсов
- Общие методы обучения
- Валидация и логирование

### RLTrainer

Универсальный тренер для RL агентов:
- Управление процессом обучения
- Curriculum learning
- Early stopping и checkpointing

### ExperienceBuffer

Эффективный буфер опыта:
- Приоритизированный сэмплинг
- Эффективное использование памяти
- Параллельная обработка

## TensorTrade Integration

### Почему TensorTrade?

- Специализированная библиотека для trading RL
- Готовые компоненты для торговли
- Интеграция с major RL фреймворками
- Активное развитие

### Интеграция с Stable-Baselines3

```python
from stable_baselines3 import PPO
from tensortrade.env.default import TradingEnv

# Создание среды
env = TradingEnv(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted"
)

# Обучение PPO агента
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Обучение и валидация

### Curriculum Learning

Постепенное усложнение задач:

1. **Stage 1**: Простая среда с детерминированными данными
2. **Stage 2**: Добавление стохастичности
3. **Stage 3**: Реальные рыночные данные
4. **Stage 4**: Множественные активы
5. **Stage 5**: Live trading с ограничениями

### Walk-forward Validation

```python
def rl_walk_forward_validation(agent, data, window_size, step_size):
    for i in range(0, len(data) - window_size, step_size):
        train_data = data[i:i+window_size]
        test_data = data[i+window_size:i+window_size+step_size]
        
        # Обучение на train_data
        agent.train(train_data)
        
        # Валидация на test_data
        performance = agent.evaluate(test_data)
        
        # Обновление модели
        agent.update(performance)
```

## Технологии

- **Stable-Baselines3** - RL алгоритмы
- **TensorTrade** - trading среды
- **Gymnasium** - RL интерфейсы
- **PyTorch** - deep learning
- **NumPy** - вычисления
- **Pandas** - обработка данных

## Конфигурация

```yaml
rl_layer:
  environment:
    type: "trading"  # trading, portfolio
    initial_balance: 10000
    commission: 0.001
    max_position_size: 1.0
    
  agent:
    algorithm: "PPO"  # PPO, SAC, DQN, A2C
    policy: "MlpPolicy"
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    
  training:
    total_timesteps: 1000000
    eval_freq: 10000
    save_freq: 50000
    curriculum_learning: true
    
  reward:
    type: "risk_adjusted"  # simple, risk_adjusted, sharpe
    profit_weight: 1.0
    risk_weight: 0.5
    cost_weight: 0.1
```

## Метрики оценки

### Financial Metrics

- **Sharpe Ratio**: риск-скорректированная доходность
- **Max Drawdown**: максимальная просадка
- **Win Rate**: процент прибыльных сделок
- **Profit Factor**: отношение прибыли к убытку

### RL Metrics

- **Average Reward**: среднее вознаграждение
- **Episode Length**: длительность эпизодов
- **Exploration**: уровень эксплорации
- **Convergence**: сходимость обучения

### Stability Metrics

- **Out-of-sample Performance**: производительность на новых данных
- **Robustness**: устойчивость к изменениям
- **Transfer Learning**: переносимость между рынками

## Интеграция

RL Layer получает данные от:
- **Meta-Learning** - предсказания и вероятности
- **Feature Engineering** - технические признаки
- **Risk Management** - ограничения и параметры

И передает решения в:
- **Decision Layer** - финальные торговые решения
- **Execution Layer** - параметры исполнения

## Требования к реализации

1. **Стабильность** - надежное обучение RL агентов
2. **Производительность** - быстрые предсказания в real-time
3. **Масштабируемость** - поддержка множественных стратегий
4. **Безопасность** - ограничения на риски
5. **Мониторинг** - отслеживание производительности

## Тестирование

- Unit тесты для компонентов RL
- Integration тесты с TensorTrade
- Simulation тесты для различных сценариев
- Performance тесты для скорости предсказаний
