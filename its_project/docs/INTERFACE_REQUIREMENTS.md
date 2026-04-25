# Interface Requirements for ITS Backend

## Overview
This document describes the interface requirements for the Intelligent Trading System (ITS) backend. The backend provides HTTP REST API for commands and WebSocket for real-time events.

## Backend Architecture

### Implemented Modules

#### 1. **Backend Service** (`backend/service.py`)
- **Purpose**: Main HTTP + WebSocket server (Flask + SocketIO)
- **Port**: 5050 (localhost)
- **Protocol**: HTTP for commands, WebSocket for events

#### 2. **OKX WebSocket Client** (`backend/okx_ws_client.py`)
- **Purpose**: Real-time market data from OKX
- **Channels**: Tickers, Orderbook L2, Candles
- **Protocol**: WebSocket (wss://ws.okx.com:8443/ws/v5/public)

#### 3. **OKX REST Client** (`backend/okx_rest_client.py`)
- **Purpose**: REST API calls to OKX via CCXT
- **Functions**: Market data, order management, account info

#### 4. **Paper Trading Engine** (`backend/paper_trading.py`)
- **Purpose**: Simulated trading without real money
- **Features**: Position tracking, PnL calculation, metrics

#### 5. **Backend Contract** (`common/backend_contract.py`)
- **Purpose**: Data schemas for events and commands
- **Independence**: No backend imports, used by GUI

---

## HTTP REST API Commands

### Connection Management

#### `POST /api/connect`
- **Purpose**: Connect backend to OKX WebSocket
- **Request Body**: None
- **Response**: 
  ```json
  {
    "success": true,
    "message": "Connected to OKX",
    "ts": 1234567890
  }
  ```

#### `POST /api/disconnect`
- **Purpose**: Disconnect backend from OKX
- **Request Body**: None
- **Response**: 
  ```json
  {
    "success": true,
    "message": "Disconnected from OKX",
    "ts": 1234567890
  }
  ```

### Health & Status

#### `GET /api/health`
- **Purpose**: Health check
- **Response**: 
  ```json
  {
    "healthy": true,
    "ts": 1234567890
  }
  ```

#### `GET /api/status`
- **Purpose**: Detailed system status
- **Response**: 
  ```json
  {
    "connected_okx_ws": "connected",
    "connected_okx_rest": "connected",
    "running_ingestion": true,
    "running_trading": false,
    "queues": {
      "price_updates": 100,
      "orderbook_updates": 50,
      "candle_updates": 30,
      "signals": 5,
      "trades": 2
    },
    "last_heartbeat": 1234567890,
    "active_symbols": ["BTC/USDT"],
    "active_channels": ["tickers", "orderbook_l2", "candles"]
  }
  ```

### Data Ingestion

#### `POST /api/ingestion/start`
- **Purpose**: Start market data ingestion
- **Request Body**:
  ```json
  {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "channels": ["tickers", "orderbook_l2", "candles"],
    "depth": 20,
    "candle_timeframes": ["1m", "5m", "1h"]
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Ingestion started for 2 symbols",
    "data": {
      "symbols": ["BTC/USDT", "ETH/USDT"],
      "channels": ["tickers", "orderbook_l2", "candles"],
      "depth": 20,
      "candle_timeframes": ["1m", "5m", "1h"]
    },
    "ts": 1234567890
  }
  ```

#### `POST /api/ingestion/stop`
- **Purpose**: Stop market data ingestion
- **Request Body**: None
- **Response**:
  ```json
  {
    "success": true,
    "message": "Ingestion stopped",
    "ts": 1234567890
  }
  ```

### Trading Control

#### `POST /api/trading/start`
- **Purpose**: Start trading (paper or live)
- **Request Body**:
  ```json
  {
    "mode": "paper"  // or "live"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Trading started in paper mode",
    "ts": 1234567890
  }
  ```

#### `POST /api/trading/stop`
- **Purpose**: Stop trading
- **Request Body**: None
- **Response**:
  ```json
  {
    "success": true,
    "message": "Trading stopped",
    "ts": 1234567890
  }
  ```

### Configuration

#### `POST /api/mode`
- **Purpose**: Set trading mode
- **Request Body**:
  ```json
  {
    "mode": "paper"  // or "live"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Trading mode set to paper",
    "ts": 1234567890
  }
  ```

#### `POST /api/subscriptions`
- **Purpose**: Update data subscriptions
- **Request Body**:
  ```json
  {
    "symbols": ["BTC/USDT"],
    "channels": ["tickers", "orderbook_l2"],
    "depth": 20,
    "candle_timeframes": ["1m"]
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Subscriptions updated",
    "ts": 1234567890
  }
  ```

### Paper Trading

#### `POST /api/paper/reset`
- **Purpose**: Reset paper trading portfolio
- **Request Body**: None
- **Response**:
  ```json
  {
    "success": true,
    "message": "Paper trading portfolio reset",
    "ts": 1234567890
  }
  ```

---

## WebSocket Events

### Event Types

#### `price_update`
- **Purpose**: Real-time price updates from tickers
- **Data**:
  ```json
  {
    "event_type": "price_update",
    "symbol": "BTC/USDT",
    "last": 50000.0,
    "bid": 49999.0,
    "ask": 50001.0,
    "ts_exchange": 1234567890,
    "ts_local": 1234567891,
    "volume_24h": 1000.0,
    "change_24h": 2.5,
    "high_24h": 51000.0,
    "low_24h": 49000.0
  }
  ```

#### `orderbook_update`
- **Purpose**: Real-time orderbook L2 updates
- **Data**:
  ```json
  {
    "event_type": "orderbook_update",
    "symbol": "BTC/USDT",
    "bids": [[50000.0, 1.0], [49999.0, 2.0]],
    "asks": [[50001.0, 1.0], [50002.0, 2.0]],
    "ts_exchange": 1234567890,
    "ts_local": 1234567891,
    "depth": 20,
    "checksum": 12345
  }
  ```

#### `candle_update`
- **Purpose**: Real-time candle/OHLCV updates
- **Data**:
  ```json
  {
    "event_type": "candle_update",
    "symbol": "BTC/USDT",
    "timeframe": "1m",
    "open": 50000.0,
    "high": 50100.0,
    "low": 49900.0,
    "close": 50050.0,
    "volume": 10.0,
    "ts_open": 1234567890,
    "ts_local": 1234567891,
    "is_closed": false
  }
  ```

#### `signal_update`
- **Purpose**: Trading signals from models
- **Data**:
  ```json
  {
    "event_type": "signal_update",
    "symbol": "BTC/USDT",
    "action": "BUY",  // or "SELL", "HOLD"
    "confidence": 0.85,
    "predicted_change": 2.5,
    "model_name": "GRU-LSTM",
    "ts": 1234567890
  }
  ```

#### `paper_trade_update`
- **Purpose**: Paper trade execution updates
- **Data**:
  ```json
  {
    "event_type": "paper_trade_update",
    "id": "trade_123",
    "symbol": "BTC/USDT",
    "side": "BUY",  // or "SELL"
    "qty": 0.1,
    "price": 50000.0,
    "fee": 5.0,
    "status": "OPEN",  // or "CLOSED"
    "ts": 1234567890,
    "exit_price": null,
    "pnl": null
  }
  ```

#### `metrics_update`
- **Purpose**: Portfolio metrics updates
- **Data**:
  ```json
  {
    "event_type": "metrics_update",
    "pnl": 10.5,
    "drawdown": 2.0,
    "exposure": 5000.0,
    "trades_count": 5,
    "win_rate": 60.0,
    "ts": 1234567890,
    "sharpe": 1.5,
    "profit_factor": 2.0
  }
  ```

#### `log_event`
- **Purpose**: Backend log messages
- **Data**:
  ```json
  {
    "event_type": "log_event",
    "level": "info",  // or "warning", "error"
    "message": "Connected to OKX",
    "component": "okx_ws",
    "ts": 1234567890
  }
  ```

#### `system_status`
- **Purpose**: System status updates (heartbeat)
- **Data**:
  ```json
  {
    "event_type": "system_status",
    "connected_okx_ws": "connected",
    "connected_okx_rest": "connected",
    "running_ingestion": true,
    "running_trading": false,
    "queues": {
      "price_updates": 100,
      "orderbook_updates": 50,
      "candle_updates": 30,
      "signals": 5,
      "trades": 2
    },
    "last_heartbeat": 1234567890,
    "ts": 1234567891,
    "active_symbols": ["BTC/USDT"],
    "active_channels": ["tickers", "orderbook_l2", "candles"]
  }
  ```

---

## Interface Requirements

### 1. Connection Panel
- **Status Indicator**: Show backend connection status (Connected/Disconnected/Error)
- **Connect Button**: Trigger `/api/connect`
- **Disconnect Button**: Trigger `/api/disconnect`
- **Health Display**: Show `/api/health` result

### 2. Data Ingestion Panel
- **Symbol Selection**: Multi-select for trading pairs (BTC/USDT, ETH/USDT, etc.)
- **Channel Selection**: Checkboxes for tickers, orderbook_l2, candles
- **Depth Input**: Number input for orderbook depth (default: 20)
- **Timeframe Selection**: Multi-select for candle timeframes (1m, 5m, 1h, etc.)
- **Start Button**: Trigger `/api/ingestion/start`
- **Stop Button**: Trigger `/api/ingestion/stop`
- **Status Display**: Show active symbols and channels

### 3. Trading Control Panel
- **Mode Selection**: Radio buttons for Paper/Live mode
- **Start Trading Button**: Trigger `/api/trading/start`
- **Stop Trading Button**: Trigger `/api/trading/stop`
- **Reset Paper Button**: Trigger `/api/paper/reset`
- **Status Display**: Show trading status and mode

### 4. Market Data Display
- **Price Display**: Show latest price from `price_update` events
- **Orderbook Display**: Show bids/asks from `orderbook_update` events
- **Candle Chart**: Display OHLCV data from `candle_update` events
- **24h Stats**: Show volume, change, high, low from `price_update`

### 5. Signal Display
- **Signal Indicator**: Show current signal (BUY/SELL/HOLD) from `signal_update`
- **Confidence Meter**: Show signal confidence (0-100%)
- **Predicted Change**: Show expected price change
- **Model Name**: Show which model generated the signal

### 6. Portfolio Display
- **Balance**: Show current paper balance
- **PnL**: Show total profit/loss percentage
- **Drawdown**: Show maximum drawdown
- **Exposure**: Show current position exposure
- **Win Rate**: Show percentage of winning trades
- **Trade Count**: Show total number of trades
- **Sharpe Ratio**: Show risk-adjusted return
- **Profit Factor**: Show profit/loss ratio

### 7. Trade History
- **Trade List**: Table showing executed trades from `paper_trade_update`
- **Columns**: ID, Symbol, Side, Qty, Price, Fee, Status, PnL, Timestamp
- **Filter**: Filter by symbol, status, date range

### 8. Log Panel
- **Log Display**: Show log messages from `log_event` events
- **Filter**: Filter by level (info, warning, error)
- **Auto-scroll**: Scroll to latest messages
- **Timestamp**: Show message timestamp

### 9. System Status Panel
- **Component Status**: Show status of each component (OKX WS, OKX REST, Ingestion, Trading)
- **Queue Counts**: Show event queue counts
- **Heartbeat**: Show last heartbeat time
- **Active Subscriptions**: Show active symbols and channels

---

## Data Validation

### Symbol Format
- **GUI Format**: `BTC/USDT`, `ETH/USDT`
- **OKX Format**: `BTCUSDT` (spot), `BTCUSDT-SWAP` (swap)
- **Validation**: Must match pattern `^[A-Z]+/[A-Z]+$`

### Channel Names
- **Valid Values**: `tickers`, `orderbook_l2`, `candles`
- **Validation**: Case-sensitive, must be from predefined list

### Trading Modes
- **Valid Values**: `paper`, `live`
- **Validation**: Case-sensitive, must be from predefined list

### Connection Status
- **Valid Values**: `disconnected`, `connecting`, `connected`, `reconnecting`, `error`
- **Validation**: Case-sensitive, must be from predefined list

---

## Error Handling

### HTTP Errors
- **400 Bad Request**: Invalid request body or parameters
- **500 Internal Server Error**: Backend processing error
- **503 Service Unavailable**: Backend not running

### WebSocket Errors
- **Connection Failed**: Backend not reachable
- **Authentication Failed**: Invalid credentials (if implemented)
- **Rate Limit**: Too many requests

### Client-Side Handling
- Show user-friendly error messages
- Implement retry logic for transient errors
- Log errors for debugging
- Provide recovery options (reconnect, reset)

---

## Performance Considerations

### Event Rate
- **Price Updates**: ~1-10 per second per symbol
- **Orderbook Updates**: ~10-100 per second per symbol
- **Candle Updates**: ~1 per minute per timeframe per symbol
- **Signal Updates**: ~1 per minute per symbol
- **Trade Updates**: As trades are executed

### Latency Requirements
- **Command Latency**: < 100ms for HTTP commands
- **Event Latency**: < 50ms for WebSocket events
- **Heartbeat Interval**: 5 seconds

### Memory Management
- Limit event queue size (e.g., 1000 events per type)
- Implement event expiration (e.g., 1 hour)
- Clean up old data periodically

---

## Security Considerations

### API Keys
- Store API keys securely (environment variables, encrypted storage)
- Never log API keys
- Implement key rotation

### Authentication
- Implement authentication for live trading
- Use API keys or JWT tokens
- Implement rate limiting

### Data Privacy
- Encrypt sensitive data at rest
- Use HTTPS for production
- Implement audit logging

---

## Future Enhancements

### Additional Data Sources
- Binance WebSocket/REST
- Glassnode on-chain data
- Twitter/X sentiment

### Advanced Features
- Backtesting interface
- Strategy optimization
- Model training interface
- Multi-exchange support

### UI Improvements
- Dark/Light theme toggle
- Customizable layouts
- Mobile responsiveness
- Real-time alerts
