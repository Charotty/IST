# Configuration

## Назначение

Централизованное управление конфигурацией системы, включая настройки всех компонентов и окружений.

## Структура конфигурации

### Основные файлы

- **config.yaml** - основная конфигурация системы
- **development.yaml** - настройки для разработки
- **production.yaml** - настройки для продакшена
- **testing.yaml** - настройки для тестирования
- **local.yaml** - локальные настройки (не в git)

### Иерархия конфигурации

1. **config.yaml** - базовые настройки
2. **{environment}.yaml** - переопределения для окружения
3. **local.yaml** - локальные переопределения
4. **Environment Variables** - переменные окружения

## Структура модуля

```
config/
├── __init__.py
├── config_manager.py         # Менеджер конфигурации
├── validators.py             # Валидаторы конфигурации
├── loaders.py               # Загрузчики конфигурации
├── schemas.py               # Схемы валидации
├── templates/
│   ├── config.yaml.template  # Шаблон конфигурации
│   ├── development.yaml.template
│   ├── production.yaml.template
│   └── testing.yaml.template
└── README.md               # Этот файл
```

## Основная конфигурация (config.yaml)

```yaml
# Intelligent Trading System Configuration

app:
  name: "Intelligent Trading System"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

# Data Layer Configuration
data_layer:
  connectors:
    okx:
      api_key: "${OKX_API_KEY}"
      secret_key: "${OKX_SECRET_KEY}"
      passphrase: "${OKX_PASSPHRASE}"
      sandbox: true
      rate_limit: 20  # requests per second
    
    binance:
      api_key: "${BINANCE_API_KEY}"
      secret_key: "${BINANCE_SECRET_KEY}"
      sandbox: true
      rate_limit: 20
    
    glassnode:
      api_key: "${GLASSNODE_API_KEY}"
      rate_limit: 10
  
  storage:
    parquet:
      path: "./data/parquet"
      compression: "snappy"
    
    timescaledb:
      host: "${TIMESCALEDB_HOST}"
      port: 5432
      database: "trading"
      username: "${TIMESCALEDB_USER}"
      password: "${TIMESCALEDB_PASSWORD}"
  
  streaming:
    buffer_size: 10000
    batch_size: 100
    flush_interval: 5

# Synchronization Layer Configuration
synchronization:
  base_timestep: "1s"
  gap_handling:
    max_gap_size: 60
    method: "adaptive"
    threshold: 0.1
  aggregation:
    ohlcv_method: "standard"
    volume_method: "sum"
  buffers:
    size: 10000
    flush_interval: 5

# Feature Engineering Configuration
feature_engineering:
  features:
    technical:
      moving_averages:
        periods: [5, 10, 20, 50, 200]
        types: ["SMA", "EMA"]
      momentum:
        rsi_periods: [14, 21]
        macd_params: [12, 26, 9]
      volatility:
        bollinger_periods: [20]
        bollinger_std: [2.0]
    
    orderbook:
      levels: [5, 10, 20]
      imbalance_window: 10
    
    temporal:
      return_periods: [1, 5, 15, 60]
      seasonal_features: true
    
    statistical:
      rolling_windows: [10, 20, 50]
      percentiles: [25, 75, 90]
  
  selection:
    method: "mutual_info"
    max_features: 100
    correlation_threshold: 0.95
  
  scaling:
    method: "standard"
    feature_range: [0, 1]

# Models Layer Configuration
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

# Meta-Learning Configuration
meta_learning:
  regime_detection:
    method: "hybrid"
    window_size: 100
    update_frequency: 3600
  
  ensemble:
    type: "weighted"
    rebalance_frequency: 300
    min_weight: 0.05
    max_weight: 0.5
  
  adaptation:
    drift_detection_method: "ks_test"
    drift_threshold: 0.05
    adaptation_rate: 0.1
  
  optimization:
    method: "convex"
    optimization_frequency: 1800
    regularization: "l2"

# RL Layer Configuration
rl_layer:
  environment:
    type: "trading"
    initial_balance: 10000
    commission: 0.001
    max_position_size: 1.0
  
  agent:
    algorithm: "PPO"
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
    type: "risk_adjusted"
    profit_weight: 1.0
    risk_weight: 0.5
    cost_weight: 0.1

# Decision Layer Configuration
decision:
  filtering:
    probability_thresholds:
      buy: 0.6
      sell: 0.4
      hold: 0.5
    confidence_threshold: 0.7
    regime_filtering: true
  
  ensemble:
    method: "weighted"
    weights:
      ml_models: 0.6
      rl_policy: 0.4
    rebalance_frequency: 3600
  
  risk_integration:
    max_position_size: 0.1
    max_portfolio_risk: 0.02
    max_drawdown_limit: 0.05
  
  sizing:
    method: "volatility"
    base_size: 0.05
    max_size: 0.2
    target_volatility: 0.15

# Risk Management Configuration
risk_management:
  position_sizing:
    method: "volatility"
    base_size: 0.02
    max_size: 0.1
    kelly_fraction: 0.25
  
  stop_loss:
    method: "atr"
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
    risk_adjustment_frequency: 3600
    high_vol_multiplier: 0.5
    low_vol_multiplier: 1.2

# Execution Layer Configuration
execution:
  mode: "paper"
  
  brokers:
    okx:
      api_key: "${OKX_API_KEY}"
      secret_key: "${OKX_SECRET_KEY}"
      sandbox: true
    
    binance:
      api_key: "${BINANCE_API_KEY}"
      secret_key: "${BINANCE_SECRET_KEY}"
      sandbox: true
  
  order_management:
    default_order_type: "market"
    max_order_size: 1000
    order_timeout: 30
  
  risk_control:
    max_position_size: 0.1
    max_total_exposure: 0.5
    emergency_stop_enabled: true
  
  monitoring:
    slippage_threshold: 0.1
    latency_threshold: 1.0
    cost_analysis_enabled: true

# Backtesting Configuration
backtesting:
  data:
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    assets: ["BTC/USDT", "ETH/USDT"]
  
  walk_forward:
    train_window: 252
    validation_window: 63
    test_window: 63
    step_size: 21
  
  simulation:
    initial_balance: 10000
    commission: 0.001
    slippage: 0.0005
    latency: 0.1
  
  optimization:
    method: "bayesian"
    objective: "sharpe_ratio"
    n_iterations: 100
  
  analysis:
    benchmark: "BTC/USDT"
    risk_free_rate: 0.02
    confidence_level: 0.95

# GUI Configuration
gui:
  window:
    title: "Intelligent Trading System"
    width: 1200
    height: 800
    resizable: true
  
  charts:
    update_interval: 1000
    max_points: 1000
    theme: "dark"
  
  panels:
    auto_refresh: true
    refresh_interval: 5000
  
  data:
    cache_size: 10000
    update_timeout: 30
  
  themes:
    default: "dark"
    custom_colors:
      background: "#2b2b2b"
      foreground: "#ffffff"
      accent: "#007acc"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    console:
      enabled: true
      level: "INFO"
    file:
      enabled: true
      level: "DEBUG"
      filename: "trading_system.log"
      max_size: "10MB"
      backup_count: 5
    syslog:
      enabled: false
      address: "/dev/log"

# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    export_interval: 60
    backend: "prometheus"
  
  alerts:
    enabled: true
    channels: ["email", "telegram"]
    thresholds:
      drawdown: 0.05
      error_rate: 0.01
      latency: 1.0
  
  health_checks:
    enabled: true
    interval: 30
    endpoints: ["/health", "/metrics", "/status"]

# Security Configuration
security:
  api_keys:
    encryption: true
    rotation_interval: 86400  # 24 hours
  
  authentication:
    enabled: true
    method: "jwt"
    token_expiry: 3600
  
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation: true
```

## Конфигурации окружений

### Development (development.yaml)

```yaml
app:
  debug: true
  log_level: "DEBUG"

data_layer:
  connectors:
    okx:
      sandbox: true
    binance:
      sandbox: true

execution:
  mode: "paper"

gui:
  window:
    width: 1400
    height: 900

logging:
  handlers:
    console:
      level: "DEBUG"
```

### Production (production.yaml)

```yaml
app:
  debug: false
  log_level: "INFO"

data_layer:
  connectors:
    okx:
      sandbox: false
    binance:
      sandbox: false

execution:
  mode: "real"

monitoring:
  alerts:
    enabled: true
    channels: ["email", "slack", "telegram"]

security:
  authentication:
    enabled: true
```

## ConfigManager

```python
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Загрузка иерархической конфигурации"""
        # Базовая конфигурация
        base_config = self.load_yaml("config.yaml")
        
        # Конфигурация окружения
        env = os.getenv("ENVIRONMENT", "development")
        env_config = self.load_yaml(f"{env}.yaml", {})
        
        # Локальная конфигурация
        local_config = self.load_yaml("local.yaml", {})
        
        # Слияние конфигураций
        self.config = self.merge_configs(base_config, env_config, local_config)
        
        # Подстановка переменных окружения
        self.config = self.substitute_env_vars(self.config)
    
    def load_yaml(self, filename: str, default: Dict = None) -> Dict:
        """Загрузка YAML файла"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            return default or {}
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def merge_configs(self, *configs) -> Dict:
        """Слияние конфигураций"""
        result = {}
        for config in configs:
            if config:
                result = self.deep_merge(result, config)
        return result
    
    def deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Глубокое слияние словарей"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def substitute_env_vars(self, config: Any) -> Any:
        """Подстановка переменных окружения"""
        if isinstance(config, dict):
            return {k: self.substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self.substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def get(self, key: str, default=None):
        """Получение значения по ключу"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
```

## Валидация конфигурации

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class DatabaseConfig(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str

class ConnectorConfig(BaseModel):
    api_key: str
    secret_key: str
    sandbox: bool = True
    rate_limit: int = 20

class DataLayerConfig(BaseModel):
    connectors: Dict[str, ConnectorConfig]
    storage: Dict[str, DatabaseConfig]
    
    @validator('connectors')
    def validate_connectors(cls, v):
        if not v:
            raise ValueError("At least one connector must be configured")
        return v

class AppConfig(BaseModel):
    name: str
    version: str
    debug: bool = False
    log_level: str = "INFO"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v
```

## Использование

```python
from config import ConfigManager

# Инициализация
config = ConfigManager()

# Получение значений
api_key = config.get('data_layer.connectors.okx.api_key')
debug_mode = config.get('app.debug', False)

# Валидация
app_config = AppConfig(**config.get('app'))
data_config = DataLayerConfig(**config.get('data_layer'))
```

## Требования

1. **Безопасность** - шифрование чувствительных данных
2. **Валидация** - проверка конфигурации при загрузке
3. **Гибкость** - поддержка различных окружений
4. **Мониторинг** - отслеживание изменений конфигурации
5. **Резервирование** - бэкап конфигурационных файлов
