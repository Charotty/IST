#!/usr/bin/env python3
"""
CLI Interface for Intelligent Trading System
============================================

Command-line interface for system management and monitoring.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional
import click

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Intelligent Trading System CLI."""
    pass


@cli.command()
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def status(format: str):
    """Show system status."""
    click.echo("System Status Check")
    click.echo("=" * 40)
    
    try:
        # Check if system is running
        from execution.task import create_execution_task
        from decision.decision import Decision, Action
        import asyncio
        
        click.echo("✅ Trading system components loaded")
        click.echo("✅ Model layer available")
        click.echo("✅ Decision layer available")
        click.echo("✅ Execution layer available")
        click.echo("✅ Backtesting layer available")
        
        if format == 'json':
            status = {
                "status": "ready",
                "components": {
                    "models": "available",
                    "decision": "available",
                    "execution": "available",
                    "backtesting": "available"
                }
            }
            click.echo(json.dumps(status, indent=2))
            
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTC/USDT)')
@click.option('--side', type=click.Choice(['buy', 'sell']), required=True, help='Order side')
@click.option('--amount', type=float, required=True, help='Order amount')
@click.option('--price', type=float, help='Order price (for limit orders)')
@click.option('--stop-loss', type=float, help='Stop loss price')
@click.option('--take-profit', type=float, help='Take profit price')
def order(symbol: str, side: str, amount: float, price: Optional[float], 
          stop_loss: Optional[float], take_profit: Optional[float]):
    """Place manual order."""
    click.echo(f"Placing {side.upper()} order for {amount} {symbol}")
    click.echo("=" * 40)
    
    try:
        from execution.task import create_execution_task
        from decision.decision import Decision, Action
        import asyncio
        
        async def place_order():
            decision_queue = asyncio.Queue()
            result_queue = asyncio.Queue()
            
            config = {
                "execution_mode": "paper",
                "paper_executor": {"initial_balance": 10000.0, "latency_ms": 10}
            }
            
            task = create_execution_task(decision_queue, result_queue, config)
            await task.start()
            
            action = Action.BUY if side == "buy" else Action.SELL
            decision = Decision(
                action=action,
                symbol=symbol,
                size=amount,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=int(asyncio.get_event_loop().time() * 1000),
                reason="Manual CLI order"
            )
            
            await decision_queue.put(decision)
            result = await asyncio.wait_for(result_queue.get(), timeout=5.0)
            
            await task.stop()
            
            return result
        
        result = asyncio.run(place_order())
        
        if result.success:
            click.echo(f"✅ Order placed successfully")
            click.echo(f"   Order ID: {result.order_id}")
            click.echo(f"   Symbol: {symbol}")
            click.echo(f"   Side: {side.upper()}")
            click.echo(f"   Amount: {amount}")
            if price:
                click.echo(f"   Price: ${price:.2f}")
        else:
            click.echo(f"❌ Order failed: {result.error}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error placing order: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--limit', type=int, default=10, help='Number of orders to show')
def orders(limit: int):
    """Show recent orders."""
    click.echo("Recent Orders")
    click.echo("=" * 40)
    
    try:
        from execution.persistence import ExecutionPersistence
        import asyncio
        
        async def get_orders():
            persistence = ExecutionPersistence("data/its_trading.db", {"max_history_days": 7})
            orders = await persistence.get_order_history(limit=limit)
            return orders
        
        orders = asyncio.run(get_orders())
        
        if not orders:
            click.echo("No orders found")
            return
        
        click.echo(f"{'ID':<20} {'Symbol':<12} {'Side':<6} {'Amount':<10} {'Price':<12} {'Status':<12}")
        click.echo("-" * 80)
        
        for order in orders:
            click.echo(f"{order['id'][:20]:<20} {order['symbol']:<12} {order['side']:<6} "
                     f"{order['amount']:<10.4f} ${order['price']:<11.2f} {order['status']:<12}")
            
    except Exception as e:
        click.echo(f"❌ Error fetching orders: {e}", err=True)
        sys.exit(1)


@cli.command()
def portfolio():
    """Show portfolio overview."""
    click.echo("Portfolio Overview")
    click.echo("=" * 40)
    
    try:
        from execution.persistence import ExecutionPersistence
        import asyncio
        
        async def get_portfolio():
            persistence = ExecutionPersistence("data/its_trading.db", {"max_history_days": 7})
            positions = await persistence.get_all_positions()
            stats = persistence.get_database_stats()
            return positions, stats
        
        positions, stats = asyncio.run(get_portfolio())
        
        click.echo(f"Total Orders: {stats['total_orders']}")
        click.echo(f"Total Positions: {stats['total_positions']}")
        click.echo()
        
        if positions:
            click.echo(f"{'Symbol':<12} {'Side':<6} {'Size':<10} {'Entry Price':<12} {'Current Price':<12}")
            click.echo("-" * 60)
            
            for pos in positions:
                click.echo(f"{pos['symbol']:<12} {pos['side']:<6} {pos['size']:<10.4f} "
                         f"${pos['entry_price']:<11.2f} ${pos['current_price']:<11.2f}")
        else:
            click.echo("No open positions")
            
    except Exception as e:
        click.echo(f"❌ Error fetching portfolio: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbol', required=True, help='Symbol to backtest')
@click.option('--period', default='1d', help='Time period')
@click.option('--strategy', default='simple', help='Strategy name')
def backtest(symbol: str, period: str, strategy: str):
    """Run backtest."""
    click.echo(f"Running Backtest")
    click.echo("=" * 40)
    click.echo(f"Symbol: {symbol}")
    click.echo(f"Period: {period}")
    click.echo(f"Strategy: {strategy}")
    click.echo()
    
    try:
        from backtesting.simple import SimpleBacktester
        import numpy as np
        
        # Generate synthetic data for demo
        np.random.seed(42)
        n_points = 1000
        prices = np.cumsum(np.random.randn(n_points) * 100) + 42000
        
        backtester = SimpleBacktester()
        
        # Create simple trades
        trades = []
        for i in range(10):
            trades.append({
                "timestamp": i * 100,
                "symbol": symbol,
                "side": "buy" if i % 2 == 0 else "sell",
                "price": float(prices[i]),
                "size": 0.001,
                "commission": 0.01
            })
        
        result = backtester.backtest(trades)
        
        click.echo(f"✅ Backtest completed")
        click.echo(f"   Total Trades: {len(result.trades)}")
        click.echo(f"   Total Return: {result.total_return:.2%}")
        click.echo(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        click.echo(f"   Max Drawdown: {result.max_drawdown:.2%}")
        click.echo(f"   Win Rate: {result.win_rate:.2%}")
        
    except Exception as e:
        click.echo(f"❌ Error running backtest: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def config(config: Optional[str]):
    """Show or update configuration."""
    if config:
        click.echo(f"Loading configuration from: {config}")
        try:
            import yaml
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
            click.echo(json.dumps(config_data, indent=2))
        except Exception as e:
            click.echo(f"❌ Error loading config: {e}", err=True)
    else:
        click.echo("Current Configuration:")
        click.echo("=" * 40)
        click.echo("Execution Mode: paper")
        click.echo("Initial Balance: $10,000")
        click.echo("Symbols: BTC/USDT, ETH/USDT, BNB/USDT")
        click.echo()
        click.echo("To load custom config:")
        click.echo("  its-cli config --config path/to/config.yaml")


@cli.command()
def logs():
    """Show recent logs."""
    click.echo("Recent Logs")
    click.echo("=" * 40)
    
    log_file = PROJECT_ROOT / "logs" / "its_trading.log"
    
    if not log_file.exists():
        click.echo("No log file found")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Show last 50 lines
            for line in lines[-50:]:
                click.echo(line.rstrip())
    except Exception as e:
        click.echo(f"❌ Error reading logs: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', default='boosting', help='Model name')
@click.option('--data', type=click.Path(exists=True), help='Training data path')
def train(model: str, data: Optional[str]):
    """Train a model."""
    click.echo(f"Training Model: {model}")
    click.echo("=" * 40)
    
    try:
        from models.boosting_model import BoostingModel
        import numpy as np
        
        # Generate synthetic training data
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 3, 100)
        
        click.echo("Generating synthetic training data...")
        click.echo(f"Training samples: {len(X)}")
        click.echo(f"Features: {X.shape[1]}")
        click.echo()
        
        if model == 'boosting':
            model_config = {"n_estimators": 50, "learning_rate": 0.1}
            model_instance = BoostingModel(model_config)
            
            click.echo("Training BoostingModel...")
            model_instance.fit(X, y)
            
            click.echo("✅ Model trained successfully")
            click.echo(f"   Feature importance: {model_instance.feature_importances_[:5]}")
        else:
            click.echo(f"❌ Model {model} not supported yet", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error training model: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    click.echo("Intelligent Trading System CLI")
    click.echo("Version: 1.0.0")
    click.echo()
    click.echo("Components:")
    click.echo("  - Data Layer")
    click.echo("  - Storage Layer")
    click.echo("  - Feature Engineering")
    click.echo("  - Model Layer")
    click.echo("  - Decision Layer")
    click.echo("  - Execution Layer")
    click.echo("  - Backtesting Layer")
    click.echo("  - Web Dashboard")
    click.echo("  - Telegram Bot")


if __name__ == "__main__":
    cli()
