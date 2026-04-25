# OKX Integration Plan

## Overview
Based on OKX API documentation, here's what needs to be implemented for OKX integration.

## REST API Authentication

### Required Headers for Private Requests
All private REST requests must include:
- `OK-ACCESS-KEY`: API key as String
- `OK-ACCESS-SIGN`: Base64-encoded signature
- `OK-ACCESS-TIMESTAMP`: ISO 8601 UTC timestamp with millisecond precision (e.g., `2020-12-08T09:08:57.715Z`)
- `OK-ACCESS-PASSPHRASE`: Passphrase specified when creating API key

### Signature Generation
The `OK-ACCESS-SIGN` header is generated as follows:

1. Create pre-hash string: `timestamp + method + requestPath + body`
2. Sign with HMAC SHA256 using SecretKey
3. Encode signature in Base64 format

Example:
```python
sign = base64.b64encode(
    hmac.new(
        secret_key.encode(),
        (timestamp + method + request_path + body).encode(),
        hashlib.sha256
    ).digest()
).decode()
```

### Important Notes
- Timestamp must be in UTC (local timezone offset causes error 50102)
- Server rejects requests where timestamp differs by more than 30 seconds
- Sync time with `GET /api/v5/public/time` before placing orders
- Request body content type must be `application/json`

## WebSocket Connection

### Endpoints
- **Public channels**: `wss://ws.okx.com:8443/ws/v5/public`
- **Private channels**: Requires login (different endpoint)

### Rate Limits
- **Connection limit**: 3 requests per second (based on IP)
- **Subscription limit**: 480 subscribe/unsubscribe/login requests per hour per connection
- **Connection count limit**: 30 WebSocket connections per channel per sub-account

### Connection Stability
- Connection breaks if no data pushed for more than 30 seconds
- To keep connection stable:
  1. Set timer for N seconds when response received (N < 30)
  2. If timer triggers, send string `'ping'`
  3. Expect `'pong'` response
  4. If no response within N seconds, raise error or reconnect

### Public Channels (No Authentication Required)
- **Tickers channel**: Real-time ticker updates
- **Order book channel**: Real-time order book updates
- **Candlesticks channel**: Real-time candlestick updates
- **Trades channel**: Real-time trade updates

### Private Channels (Authentication Required)
- **Orders channel**: Order updates
- **Account channel**: Account updates
- **Positions channel**: Position updates
- **Balance and positions channel**: Balance and position updates

## Implementation Steps

### Phase 1: Configuration (Required First)
1. **Add API key configuration**
   - Create environment variables or config file for:
     - `OKX_API_KEY`
     - `OKX_SECRET_KEY`
     - `OKX_PASSPHRASE`
   - For paper trading/public data, API keys may not be required

### Phase 2: REST API Implementation
1. **Implement authentication in `okx_rest_client.py`**
   - Add signature generation function
   - Add required headers to all private requests
   - Implement time synchronization with server
   - Handle timestamp validation errors

2. **Test public endpoints (no auth required)**
   - `GET /api/v5/market/tickers` - Get tickers
   - `GET /api/v5/market/books` - Get order book
   - `GET /api/v5/market/candles` - Get candlesticks
   - `GET /api/v5/public/time` - Get server time

3. **Test private endpoints (auth required)**
   - `GET /api/v5/account/balance` - Get balance
   - `GET /api/v5/trade/orders-pending` - Get pending orders
   - `POST /api/v5/trade/order` - Place order

### Phase 3: WebSocket Implementation
1. **Implement public WebSocket connection in `okx_ws_client.py`**
   - Connect to `wss://ws.okx.com:8443/ws/v5/public`
   - Implement subscription to public channels
   - Handle connection errors and reconnection

2. **Add ping/pong mechanism**
   - Send `'ping'` every 20-25 seconds
   - Handle `'pong'` response
   - Reconnect if no response within timeout

3. **Parse OKX message formats**
   - Tickers format
   - Order book format
   - Candlesticks format
   - Convert to internal event format

### Phase 4: Integration with Backend
1. **Connect OKX clients to backend service**
   - Wire up OKX WS client callbacks
   - Wire up OKX REST client methods
   - Handle connection state management

2. **Test data flow**
   - OKX → Backend → WebSocket → Web Interface
   - Verify all event types work correctly

### Phase 5: Model Integration
1. **Integrate models module**
   - Load trained models
   - Generate signals from market data
   - Send signals to backend

2. **Integrate decision module**
   - Process signals through decision engine
   - Apply risk management
   - Generate trade decisions

3. **Integrate execution module**
   - Execute trades via OKX REST API
   - Track positions
   - Update portfolio

## Configuration Template

### Environment Variables (.env)
```bash
# OKX API Configuration
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

# OKX Environment (demo or live)
OKX_ENVIRONMENT=demo

# Backend Configuration
BACKEND_HOST=127.0.0.1
BACKEND_PORT=5050
```

### Config File (config.yaml)
```yaml
okx:
  api_key: ${OKX_API_KEY}
  secret_key: ${OKX_SECRET_KEY}
  passphrase: ${OKX_PASSPHRASE}
  environment: demo  # demo or live
  
  websocket:
    public_url: wss://ws.okx.com:8443/ws/v5/public
    private_url: wss://ws.okx.com:8443/ws/v5/private
    ping_interval: 20  # seconds
    reconnect_interval: 5  # seconds
    
  rest:
    base_url: https://www.okx.com
    demo_url: https://www.okx.com
    timeout: 30  # seconds
```

## Testing Strategy

### Unit Tests
- Test signature generation
- Test timestamp formatting
- Test header construction

### Integration Tests
- Test public API endpoints
- Test private API endpoints (with demo keys)
- Test WebSocket connection
- Test WebSocket subscriptions
- Test ping/pong mechanism

### End-to-End Tests
- Test full data flow: OKX → Backend → Web Interface
- Test paper trading with real data
- Test model signal generation with real data

## Security Considerations

1. **Never commit API keys to repository**
2. **Use environment variables or secret management**
3. **Rotate API keys regularly**
4. **Use demo environment for testing**
5. **Implement IP whitelisting if available**
6. **Monitor API usage and rate limits**
7. **Log all API calls for audit trail**

## Common Errors and Solutions

### Error 50102: Timestamp out of range
- **Cause**: Local time differs from server time by more than 30 seconds
- **Solution**: Sync time with `GET /api/v5/public/time` before requests

### Error 50103: Invalid signature
- **Cause**: Incorrect signature generation
- **Solution**: Verify pre-hash string format and HMAC SHA256 implementation

### WebSocket disconnection
- **Cause**: No data for more than 30 seconds
- **Solution**: Implement ping/pong mechanism

### Rate limit exceeded
- **Cause**: Too many requests
- **Solution**: Implement rate limiting and exponential backoff

## Next Steps

1. Add API key configuration to environment/config
2. Implement REST authentication in `okx_rest_client.py`
3. Implement real WebSocket connection in `okx_ws_client.py`
4. Add ping/pong mechanism
5. Test with OKX demo environment
6. Integrate with backend service
7. Test full pipeline
