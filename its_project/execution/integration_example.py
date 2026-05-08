#!/usr/bin/env python3
"""
Integration Example: Complete Trading System with Persistence
======================================================

Example demonstrating complete integration of PnL tracking,
performance metrics, and data persistence.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from its_project.execution import (
    PnLTracker, AdvancedPerformanceMetrics, RealTimePerformanceDashboard,
    DataPersistenceManager, DatabaseConfig, CSVConfig,
    TradeRecord, PnLRecord, PositionSnapshot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """
    Complete trading system with PnL tracking, performance analysis,
    and data persistence.
    """
    
    def __init__(self, data_dir: str = "trading_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_pnl_tracker()
        self._init_performance_metrics()
        self._init_persistence()
        self._init_dashboard()
        
        # System state
        self.is_running = False
        self.trades_processed = 0
        self.last_export_time = datetime.now()
    
    def _init_pnl_tracker(self) -> None:
        """Initialize PnL tracker."""
        self.pnl_tracker = PnLTracker(
            db_path=str(self.data_dir / "pnl_tracking.db"),
            auto_save=True,
            save_interval=60
        )
        logger.info("PnL tracker initialized")
    
    def _init_performance_metrics(self) -> None:
        """Initialize performance metrics calculator."""
        self.metrics_calculator = AdvancedPerformanceMetrics(
            risk_free_rate=0.02,
            confidence_level=0.95
        )
        logger.info("Performance metrics calculator initialized")
    
    def _init_persistence(self) -> None:
        """Initialize data persistence manager."""
        db_config = DatabaseConfig(
            db_path=str(self.data_dir / "trading_system.db"),
            backup_enabled=True,
            backup_interval=3600,  # 1 hour
            max_backups=24  # Keep 24 hours of backups
        )
        
        csv_config = CSVConfig(
            export_dir=str(self.data_dir / "exports"),
            compression=True,
            batch_size=5000
        )
        
        self.persistence_manager = DataPersistenceManager(
            db_config=db_config,
            csv_config=csv_config,
            auto_export_interval=1800  # 30 minutes
        )
        logger.info("Data persistence manager initialized")
    
    def _init_dashboard(self) -> None:
        """Initialize real-time dashboard."""
        self.dashboard = RealTimePerformanceDashboard(
            pnl_tracker=self.pnl_tracker,
            benchmark_returns=None  # Could add benchmark data
        )
        logger.info("Real-time dashboard initialized")
    
    async def start(self) -> None:
        """Start the integrated trading system."""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting integrated trading system")
        self.is_running = True
        
        # Start dashboard
        await self.dashboard.start()
        
        # Start auto-export (already started in persistence manager)
        logger.info("System started successfully")
    
    async def stop(self) -> None:
        """Stop the integrated trading system."""
        if not self.is_running:
            return
        
        logger.info("Stopping integrated trading system")
        self.is_running = False
        
        # Stop dashboard
        await self.dashboard.stop()
        
        # Close persistence manager
        await self.persistence_manager.close()
        
        # Save final PnL data
        self.pnl_tracker.save()
        
        logger.info("System stopped successfully")
    
    async def process_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_id: Optional[str] = None,
        order_id: Optional[str] = None
    ) -> bool:
        """
        Process a single trade through the complete system.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price
            strategy_id: Strategy identifier
            order_id: Order identifier
            
        Returns:
            True if trade processed successfully
        """
        try:
            # Create trade record
            trade = TradeRecord(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                commission=price * quantity * 0.001,  # 0.1% commission
                fees={"exchange": 1.0, "network": 0.5},
                strategy_id=strategy_id,
                order_id=order_id
            )
            
            # Add to PnL tracker
            self.pnl_tracker.add_trade(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity,
                price=trade.price,
                timestamp=trade.timestamp,
                commission=trade.commission,
                fees=trade.fees,
                strategy_id=trade.strategy_id,
                order_id=trade.order_id
            )
            
            # Save to database
            await self.persistence_manager.save_trade(trade)
            
            self.trades_processed += 1
            
            logger.info(f"Processed trade {trade_id}: {symbol} {side} {quantity}@{price}")
            
            # Periodic export
            if (datetime.now() - self.last_export_time).seconds >= 300:  # 5 minutes
                await self._export_summary_data()
                self.last_export_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing trade {trade_id}: {e}")
            return False
    
    async def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for unrealized PnL calculation."""
        try:
            self.pnl_tracker.update_market_price(symbol, price)
            logger.debug(f"Updated market price for {symbol}: {price}")
        except Exception as e:
            logger.error(f"Error updating market price for {symbol}: {e}")
    
    async def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        try:
            # Get PnL summary
            pnl_summary = {
                'cumulative_pnl': self.pnl_tracker.get_cumulative_pnl(),
                'total_trades': len(self.pnl_tracker.pnl_records),
                'open_positions': len(self.pnl_tracker.positions),
                'trades_processed': self.trades_processed
            }
            
            # Get performance summary
            if self.pnl_tracker.pnl_records:
                trade_pnls = [record.realized_pnl for record in self.pnl_tracker.pnl_records]
                equity_curve = [100000]  # Starting capital
                for pnl in trade_pnls:
                    equity_curve.append(equity_curve[-1] + pnl)
                
                returns = []
                for i in range(1, len(equity_curve)):
                    returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
                
                performance_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                    returns=returns,
                    equity_curve=equity_curve,
                    trade_pnls=trade_pnls
                )
            else:
                performance_metrics = {'summary': {}}
            
            # Get dashboard summary
            dashboard_summary = self.dashboard.get_current_metrics_summary()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'pnl_summary': pnl_summary,
                'performance_metrics': performance_metrics.get('summary', {}),
                'dashboard_summary': dashboard_summary,
                'data_directory': str(self.data_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'error': str(e)
            }
    
    async def export_all_data(self, custom_filename: Optional[str] = None) -> dict:
        """Export all system data to CSV files."""
        try:
            results = await self.persistence_manager.export_all_data(
                include_trades=True,
                include_pnl=True,
                include_positions=True,
                custom_filename=custom_filename
            )
            
            logger.info(f"Exported data to {len(results)} files")
            return results
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {}
    
    async def _export_summary_data(self) -> None:
        """Export summary data for monitoring."""
        try:
            # Get current positions
            positions_data = []
            for symbol, position in self.pnl_tracker.get_all_positions().items():
                positions_data.append({
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'total_pnl': position.total_pnl,
                    'last_price': position.last_price,
                    'trades_count': position.trades_count
                })
            
            # Export positions
            if positions_data:
                await self.persistence_manager.csv_exporter.export_positions(
                    positions_data,
                    filename="current_positions.csv"
                )
            
            logger.debug("Exported summary data")
            
        except Exception as e:
            logger.error(f"Error exporting summary data: {e}")


async def run_integration_example():
    """Run complete integration example."""
    logger.info("Starting integration example")
    
    # Create integrated system
    system = IntegratedTradingSystem("example_trading_data")
    
    try:
        # Start system
        await system.start()
        
        # Simulate trading activity
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        strategies = ["momentum", "mean_reversion", "arbitrage"]
        
        # Simulate 100 trades over time
        for i in range(100):
            # Random symbol and strategy
            symbol = symbols[i % len(symbols)]
            strategy = strategies[i % len(strategies)]
            
            # Generate realistic price
            base_price = {"BTC/USDT": 50000, "ETH/USDT": 3000, "BNB/USDT": 300}[symbol]
            price_variation = (i % 20 - 10) * 0.01  # ±10% variation
            price = base_price * (1 + price_variation)
            
            # Alternate buy/sell
            side = "buy" if i % 2 == 0 else "sell"
            
            # Process trade
            await system.process_trade(
                trade_id=f"example_trade_{i:04d}",
                symbol=symbol,
                side=side,
                quantity=1.0 + (i % 3) * 0.5,  # Variable quantity
                price=price,
                strategy_id=strategy,
                order_id=f"order_{i:04d}"
            )
            
            # Update market prices periodically
            if i % 10 == 0:
                for s in symbols:
                    market_price = {"BTC/USDT": 50000, "ETH/USDT": 3000, "BNB/USDT": 300}[s]
                    market_variation = (i % 20 - 10) * 0.005
                    await system.update_market_price(s, market_price * (1 + market_variation))
            
            # Small delay between trades
            await asyncio.sleep(0.1)
        
        # Let the system run for a bit to process data
        await asyncio.sleep(5)
        
        # Get system status
        status = await system.get_system_status()
        logger.info("System Status:")
        logger.info(f"  Cumulative PnL: ${status['pnl_summary']['cumulative_pnl']:.2f}")
        logger.info(f"  Total Trades: {status['pnl_summary']['total_trades']}")
        logger.info(f"  Open Positions: {status['pnl_summary']['open_positions']}")
        logger.info(f"  Trades Processed: {status['pnl_summary']['trades_processed']}")
        
        if 'overall_sharpe' in status['performance_metrics']:
            logger.info(f"  Sharpe Ratio: {status['performance_metrics']['overall_sharpe']:.2f}")
        if 'max_drawdown' in status['performance_metrics']:
            logger.info(f"  Max Drawdown: {status['performance_metrics']['max_drawdown']:.1%}")
        if 'win_rate' in status['performance_metrics']:
            logger.info(f"  Win Rate: {status['performance_metrics']['win_rate']:.1%}")
        
        # Export all data
        export_results = await system.export_all_data("integration_example_export")
        logger.info(f"Exported data files: {list(export_results.keys())}")
        
        # Export dashboard charts
        system.dashboard.export_charts(str(system.data_dir / "dashboard_charts"))
        logger.info("Exported dashboard charts")
        
        # Keep system running for a while to demonstrate real-time updates
        logger.info("System will continue running for 30 seconds...")
        await asyncio.sleep(30)
        
    except Exception as e:
        logger.error(f"Error in integration example: {e}")
    
    finally:
        # Stop system
        await system.stop()
        logger.info("Integration example completed")


async def run_simple_persistence_example():
    """Run simple persistence example."""
    logger.info("Starting simple persistence example")
    
    # Create persistence manager
    persistence = create_persistence_manager(
        db_path="simple_example.db",
        export_dir="simple_exports"
    )
    
    try:
        # Create sample trades
        trades = [
            TradeRecord(
                trade_id=f"simple_trade_{i:03d}",
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                quantity=1.0,
                price=50000.0 + i * 100,
                timestamp=datetime.now() + timedelta(minutes=i),
                commission=10.0,
                fees={"exchange": 5.0},
                strategy_id="simple_strategy"
            )
            for i in range(10)
        ]
        
        # Save trades
        saved_count = await persistence.save_trades_batch(trades)
        logger.info(f"Saved {saved_count} trades to database")
        
        # Export to CSV
        export_results = await persistence.export_all_data()
        logger.info(f"Exported data: {list(export_results.keys())}")
        
        # Verify data
        retrieved_trades = await persistence.db_manager.get_trades(limit=5)
        logger.info(f"Retrieved {len(retrieved_trades)} trades from database")
        
    finally:
        await persistence.close()
        logger.info("Simple persistence example completed")


if __name__ == "__main__":
    # Run examples
    print("Choose example to run:")
    print("1. Complete integration example")
    print("2. Simple persistence example")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_integration_example())
    elif choice == "2":
        asyncio.run(run_simple_persistence_example())
    else:
        print("Invalid choice")
