# Intelligent Trading System (ITS)

A comprehensive, production-ready trading system with real-time data processing, machine learning models, and advanced GUI.

## Features

### Core Trading System
- **Real-time Signal Generation**: ML-powered trading signals with multiple models
- **Paper Trading**: Realistic trade execution with slippage and commissions
- **Risk Management**: Advanced position sizing and risk controls
- **Model Selection**: Automatic model selection based on performance

### Data Processing
- **Real-time Data Streaming**: Live data from Parquet and TimescaleDB
- **Data Quality Monitoring**: Duplicate detection, out-of-order handling
- **Feature Engineering**: Technical indicators and market features
- **Storage Integration**: Unified Parquet + TimescaleDB storage

### User Interface
- **Modern PyQt6 GUI**: Dark theme, real-time updates
- **Multiple GUI Modes**: Basic, Integrated, Production, Async, Real-time
- **Comprehensive Dashboard**: Charts, signals, trades, metrics
- **Async Processing**: Non-blocking UI with background operations

### Advanced Features
- **Async Integration**: Full asyncio support with PyQt
- **Change Detection**: Real-time data change notifications
- **Batch Processing**: Efficient bulk data operations
- **Caching System**: TTL-based caching for performance

## Quick Start

### 1. Setup
```bash
# Clone and navigate to project
cd d:\IST\its_project

# Run initial setup
python run.py setup

# Install dependencies
pip install -r requirements.txt
```

### 2. Health Check
```bash
# Verify system health
python run.py check
```

### 3. Launch GUI
```bash
# Production GUI (recommended)
python run.py gui --mode production

# Real-time GUI with storage integration
python run.py gui --mode realtime

# Demo mode (no backend required)
python run.py gui --demo
```

## GUI Modes

### Basic Mode
- Simple trading interface
- Simulated data
- Basic signal display

### Integrated Mode
- Backend integration
- Real signal generation
- Trade execution

### Production Mode
- Full backend integration
- Error handling
- Performance monitoring

### Async Mode
- Async processing
- Non-blocking operations
- Advanced task management

### Real-time Mode
- Storage system integration
- Real-time data streaming
- Data quality monitoring

## Architecture

### Backend Systems
```
decision/           # Signal generation
  signal_generator.py
  decision.py

execution/          # Trade execution
  trader.py
  base.py

storage/            # Data storage
  data_loader.py
  parquet_store.py
  timescale_client.py

models/             # ML models
  gru_model.py
  cnn_lob_model.py
  boosting_model.py
  base.py

features/           # Feature engineering
  feature_builder.py
  synchronizer.py
  technical.py
```

### GUI System
```
gui/
  main_window.py              # Basic GUI
  integrated_main_window.py   # Integrated GUI
  production_main_window.py   # Production GUI
  async_main_window.py         # Async GUI
  realtime_main_window.py      # Real-time GUI
  
  components/                  # UI components
    sidebar.py
    main_chart.py
    signal_panel.py
    model_panel.py
    bottom_panel.py
  
  styles/                      # UI styling
    dark_theme.py
  
  integration/                 # Backend integration
    backend_bridge.py
    real_backend_bridge.py
    realtime_backend_bridge.py
    data_adapters.py
    format_validators.py
    async_integration.py
    realtime_data_connector.py
```

## Configuration

### Environment Setup
```bash
# Database connection
export TIMESCALE_DSN="postgresql://user:pass@localhost/its"

# Data paths
export PARQUET_PATH="data/parquet"
export MODEL_PATH="models/saved"
```

### Configuration Files
- `config.yaml` - Main system configuration
- `requirements.txt` - Python dependencies
- `.env` - Environment variables

## Trading Features

### Signal Generation
- **GRU-LSTM Model**: Deep learning for time series
- **CNN-Attention Model**: Convolutional neural network
- **Boosting Model**: Gradient boosting ensemble
- **Model Selection**: Automatic best model selection

### Risk Management
- **Position Sizing**: Dynamic position allocation
- **Stop Loss**: Automatic loss limiting
- **Commission Modeling**: Realistic fee structure
- **Slippage Model**: Market impact simulation

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough loss
- **Win Rate**: Success percentage
- **Profit Factor**: Gross profit/loss ratio

## Data Processing

### Real-time Streaming
- **Parquet Monitoring**: File change detection
- **TimescaleDB Listening**: Database notifications
- **Data Validation**: Quality checks and filtering
- **Change Detection**: Real-time change alerts

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Price Features**: Returns, moving averages, volatility
- **Volume Features**: Volume ratios, VWAP
- **Market Features**: Order book metrics

### Storage Systems
- **Parquet Storage**: Columnar format for analytics
- **TimescaleDB**: Time-series database
- **Unified Access**: Single interface for both systems
- **Data Versioning**: Track data changes over time

## Development

### Running Tests
```bash
# All tests
python run.py test

# Unit tests
python run.py test --type unit

# Integration tests
python run.py test --type integration

# GUI tests
python run.py test --type gui
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Health Monitoring
```bash
# Comprehensive health check
python run.py check

# Detailed health report
python scripts/health_check.py
```

## Troubleshooting

### Common Issues

#### GUI Won't Start
```bash
# Check PyQt6 installation
python -c "from PyQt6.QtWidgets import QApplication"

# Try demo mode
python run.py gui --demo
```

#### Backend Connection Issues
```bash
# Check backend components
python -c "from decision.signal_generator import SignalGenerator"

# Verify data paths
ls -la data/parquet/
```

#### Storage System Issues
```bash
# Create data directories
mkdir -p data/parquet data/raw

# Check permissions
chmod 755 data/
```

### Log Analysis
```bash
# System logs
tail -f logs/system.log

# GUI logs
tail -f logs/gui.log

# Filter errors
grep "ERROR" logs/*.log
```

## Performance

### Optimization Features
- **Async Processing**: Non-blocking operations
- **Caching**: TTL-based data caching
- **Batch Processing**: Efficient bulk operations
- **Memory Management**: Automatic cleanup

### Monitoring
- **Real-time Metrics**: Performance tracking
- **Resource Usage**: CPU and memory monitoring
- **Error Tracking**: Comprehensive error logging
- **Data Quality**: Quality metrics and alerts

## Security

### Data Protection
- **Input Validation**: Comprehensive data validation
- **Error Handling**: Secure error reporting
- **Access Controls**: Role-based permissions
- **Encryption**: Data encryption at rest

### Best Practices
- **Environment Variables**: Secure credential storage
- **Regular Updates**: Security patching
- **Audit Logging**: Comprehensive audit trails
- **Backup Systems**: Regular data backups

## Deployment

### Development
```bash
# Setup development environment
python run.py setup
pip install -r requirements.txt
python run.py check
```

### Production
```bash
# Setup production environment
export ENVIRONMENT="production"
export LOG_LEVEL="WARNING"
python run.py gui --mode production
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py", "gui", "--mode", "production"]
```

## Documentation

- [System Verification Guide](SYSTEM_VERIFICATION_GUIDE.md) - Comprehensive system testing
- [API Documentation](docs/api/) - Detailed API reference
- [Development Guide](docs/development/) - Development procedures
- [Deployment Guide](docs/deployment/) - Production deployment

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests: `python run.py test`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the [System Verification Guide](SYSTEM_VERIFICATION_GUIDE.md)
- Review the troubleshooting section
- Check system health: `python run.py check`
- Review logs: `tail -f logs/system.log`

---

**ITS** - Intelligent Trading System  
Built for professional trading with real-time data processing and advanced machine learning.
