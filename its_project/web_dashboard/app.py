#!/usr/bin/env python3
"""
Web Dashboard Application
========================

Real-time monitoring interface for the Intelligent Trading System.
Provides WebSocket connections for live data and REST API for historical data.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np

logger = logging.getLogger(__name__)


class DashboardDataStore:
    """In-memory data store for dashboard real-time data."""
    
    def __init__(self) -> None:
        # Real-time price data
        self.price_data: Dict[str, Dict[str, Any]] = {}
        
        # Order data
        self.orders: List[Dict[str, Any]] = []
        
        # Portfolio data
        self.portfolio: Dict[str, Any] = {
            "total_value": 0.0,
            "positions": [],
            "pnl": 0.0,
            "daily_pnl": 0.0
        }
        
        # System health
        self.system_health: Dict[str, Any] = {
            "status": "running",
            "uptime": 0,
            "last_update": datetime.now().isoformat(),
            "components": {
                "data_layer": "healthy",
                "models": "healthy",
                "execution": "healthy",
                "decision": "healthy"
            }
        }
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_returns": []
        }
    
    def update_price(self, symbol: str, price: float, timestamp: int) -> None:
        """Update price data for a symbol."""
        self.price_data[symbol] = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp,
            "last_update": datetime.now().isoformat()
        }
    
    def add_order(self, order: Dict[str, Any]) -> None:
        """Add new order to the list."""
        order["dashboard_timestamp"] = datetime.now().isoformat()
        self.orders.insert(0, order)
        if len(self.orders) > 100:  # Keep last 100 orders
            self.orders.pop()
    
    def update_portfolio(self, portfolio_data: Dict[str, Any]) -> None:
        """Update portfolio information."""
        self.portfolio.update(portfolio_data)
        self.portfolio["last_update"] = datetime.now().isoformat()
    
    def update_system_health(self, health_data: Dict[str, Any]) -> None:
        """Update system health status."""
        self.system_health.update(health_data)
        self.system_health["last_update"] = datetime.now().isoformat()
    
    def update_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """Update performance metrics."""
        self.metrics.update(metrics_data)
        self.metrics["last_update"] = datetime.now().isoformat()


# Global data store instance
data_store = DashboardDataStore()


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure Flask application."""
    
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configuration
    app.config['SECRET_KEY'] = config.get('secret_key', 'dev-secret-key') if config else 'dev-secret-key'
    app.config['DEBUG'] = config.get('debug', True) if config else True
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Template routes
    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/orders')
    def orders_page():
        """Orders monitoring page."""
        return render_template('orders.html')
    
    @app.route('/portfolio')
    def portfolio_page():
        """Portfolio overview page."""
        return render_template('portfolio.html')
    
    # API routes
    @app.route('/api/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_health": data_store.system_health
        })
    
    @app.route('/api/prices')
    def get_prices():
        """Get current prices."""
        return jsonify(data_store.price_data)
    
    @app.route('/api/orders')
    def get_orders():
        """Get recent orders."""
        limit = request.args.get('limit', 50, type=int)
        return jsonify(data_store.orders[:limit])
    
    @app.route('/api/portfolio')
    def get_portfolio():
        """Get portfolio information."""
        return jsonify(data_store.portfolio)
    
    @app.route('/api/metrics')
    def get_metrics():
        """Get performance metrics."""
        return jsonify(data_store.metrics)
    
    @app.route('/api/system-health')
    def get_system_health():
        """Get system health status."""
        return jsonify(data_store.system_health)
    
    # WebSocket events
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'connected', 'message': 'Connected to dashboard'})
        
        # Send initial data
        emit('price_update', data_store.price_data)
        emit('orders_update', data_store.orders[:20])
        emit('portfolio_update', data_store.portfolio)
        emit('system_health', data_store.system_health)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle subscription to data streams."""
        channels = data.get('channels', [])
        logger.info(f"Client {request.sid} subscribed to: {channels}")
        emit('subscribed', {'channels': channels})
    
    # Simulation data update (for development)
    def update_simulation_data():
        """Generate simulated data for dashboard testing."""
        import random
        
        # Update prices
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        for symbol in symbols:
            base_price = {"BTC/USDT": 42000.0, "ETH/USDT": 2500.0, "BNB/USDT": 400.0}[symbol]
            price = base_price * (1 + random.uniform(-0.001, 0.001))
            data_store.update_price(symbol, price, int(asyncio.get_event_loop().time() * 1000))
        
        # Update portfolio
        total_value = sum(
            data["price"] * random.uniform(0.1, 2.0)
            for data in data_store.price_data.values()
        )
        data_store.update_portfolio({
            "total_value": total_value,
            "positions": [
                {"symbol": "BTC", "size": 0.5, "value": data_store.price_data.get("BTC/USDT", {}).get("price", 42000.0) * 0.5},
                {"symbol": "ETH", "size": 5.0, "value": data_store.price_data.get("ETH/USDT", {}).get("price", 2500.0) * 5.0}
            ],
            "pnl": random.uniform(-100, 200),
            "daily_pnl": random.uniform(-50, 100)
        })
        
        # Update metrics
        data_store.update_metrics({
            "total_trades": len(data_store.orders),
            "win_rate": random.uniform(0.45, 0.65),
            "sharpe_ratio": random.uniform(1.0, 2.5),
            "max_drawdown": random.uniform(0.05, 0.15)
        })
        
        # Broadcast updates
        socketio.emit('price_update', data_store.price_data, broadcast=True)
        socketio.emit('portfolio_update', data_store.portfolio, broadcast=True)
        socketio.emit('metrics_update', data_store.metrics, broadcast=True)
    
    # Start background simulation (only in debug mode)
    if app.config['DEBUG']:
        import threading
        import time
        
        def simulation_thread():
            """Background thread for data simulation."""
            while True:
                try:
                    update_simulation_data()
                    time.sleep(1)  # Update every second
                except Exception as e:
                    logger.error(f"Simulation error: {e}")
                    time.sleep(5)
        
        sim_thread = threading.Thread(target=simulation_thread, daemon=True)
        sim_thread.start()
    
    # Store socketio instance for external access
    app.socketio = socketio
    
    return app


# External interface for updating dashboard data
def update_dashboard_price(symbol: str, price: float, timestamp: int) -> None:
    """Update price data from external system."""
    data_store.update_price(symbol, price, timestamp)


def add_dashboard_order(order: Dict[str, Any]) -> None:
    """Add order from external system."""
    data_store.add_order(order)


def update_dashboard_portfolio(portfolio_data: Dict[str, Any]) -> None:
    """Update portfolio from external system."""
    data_store.update_portfolio(portfolio_data)


def update_system_health(health_data: Dict[str, Any]) -> None:
    """Update system health from external system."""
    data_store.update_system_health(health_data)


def broadcast_update(event_name: str, data: Any) -> None:
    """Broadcast update to all connected clients."""
    # This would need access to the socketio instance
    # For now, it's a placeholder for future implementation
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    socketio = app.socketio
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
