# New Workflow for ITS Web Interface

## Workflow Overview

### Phase 1: Data Loading
- **Symbol Selection**: Choose trading pair (BTC/USDT, ETH/USDT, etc.)
- **Timeframe Selection**: Choose timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- **Date Range**: Select start date and end date
- **Data Sources**: Choose data sources (OKX candles, orderbook, trades)
- **Load Button**: Load historical data from OKX REST API
- **Data Preview**: Show loaded data statistics

### Phase 2: Feature Engineering
- **Feature Selection**: Choose features to calculate
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Orderbook features (imbalance, spread, depth)
  - Microstructure features (volume profile, VWAP)
- **Feature Configuration**: Configure parameters for each feature
- **Calculate Button**: Calculate features from loaded data
- **Feature Preview**: Show calculated features statistics

### Phase 3: Model Training
- **Model Selection**: Choose model type
  - LSTM
  - GRU
  - Transformer
  - Ensemble
- **Model Configuration**: Configure hyperparameters
  - Sequence length
  - Hidden layers
  - Dropout
  - Learning rate
  - Epochs
  - Batch size
- **Train/Test Split**: Configure train/test split ratio
- **Train Button**: Train model on prepared data
- **Training Progress**: Show training progress (loss, accuracy)
- **Model Evaluation**: Show evaluation metrics

### Phase 4: Backtesting
- **Strategy Configuration**: Configure trading strategy
  - Entry signals threshold
  - Exit signals threshold
  - Stop loss
  - Take profit
  - Position sizing
- **Backtest Period**: Select period for backtesting
- **Backtest Button**: Run backtesting
- **Backtest Results**: Show
  - Total return
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Profit factor
  - Trade count
- **Equity Curve**: Show equity chart
- **Trade List**: Show individual trades

### Phase 5: Deployment
- **Model Selection**: Choose trained model to deploy
- **Deployment Configuration**: Configure deployment settings
  - Trading mode (paper/live)
  - Position size
  - Risk limits
- **Deploy Button**: Deploy model for real-time trading
- **Monitoring**: Show real-time
  - Current signals
  - Open positions
  - PnL
  - Trade history

## Interface Structure

### Navigation Tabs
1. **Data** - Data loading and feature engineering
2. **Training** - Model training and evaluation
3. **Backtesting** - Strategy backtesting
4. **Deployment** - Model deployment and monitoring

### Data Tab
- Symbol selector
- Timeframe selector
- Date range picker
- Data source checkboxes
- Load button
- Data statistics display
- Feature configuration panel
- Calculate features button
- Feature statistics display

### Training Tab
- Model type selector
- Hyperparameter inputs
- Train/test split slider
- Train button
- Training progress bar
- Loss chart
- Evaluation metrics display
- Model save button

### Backtesting Tab
- Strategy configuration panel
- Backtest period selector
- Backtest button
- Results summary
- Equity chart
- Trade history table

### Deployment Tab
- Model selector (from trained models)
- Deployment configuration
- Deploy button
- Real-time monitoring
  - Current price
  - Current signal
  - Open positions
  - PnL
  - Trade history

## Backend API Changes

### New Endpoints

#### Data Loading
- `POST /api/data/load` - Load historical data
- `GET /api/data/status` - Get data loading status
- `GET /api/data/preview` - Get data preview

#### Feature Engineering
- `POST /api/features/calculate` - Calculate features
- `GET /api/features/list` - Get available features
- `GET /api/features/preview` - Get feature preview

#### Model Training
- `POST /api/models/train` - Train model
- `GET /api/models/status` - Get training status
- `GET /api/models/list` - Get trained models
- `POST /api/models/load` - Load trained model
- `GET /api/models/evaluate` - Get model evaluation

#### Backtesting
- `POST /api/backtest/run` - Run backtest
- `GET /api/backtest/status` - Get backtest status
- `GET /api/backtest/results` - Get backtest results

#### Deployment
- `POST /api/deployment/deploy` - Deploy model
- `GET /api/deployment/status` - Get deployment status
- `POST /api/deployment/stop` - Stop deployment
- `GET /api/deployment/signals` - Get current signals
- `GET /api/deployment/positions` - Get current positions

## Implementation Priority

1. **Data Loading** - Historical data from OKX REST API
2. **Feature Engineering** - Integrate features module
3. **Model Training** - Integrate models module
4. **Backtesting** - Integrate backtesting module
5. **Deployment** - Real-time monitoring with OKX WebSocket
