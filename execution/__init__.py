"""
Execution Layer for ITS

This layer handles execution of trading signals on exchanges: paper trading and live trading,
order management, and execution monitoring.

Components:
- Brokers: Base broker interface, OKX broker (live), Paper broker (simulation)
- Orders: Order manager for order lifecycle management
- Execution Manager: Main execution orchestrator
- Monitoring: Execution monitoring and reconciliation
"""

from .brokers.base_broker import BaseBroker
from .brokers.paper_broker import PaperBroker
from .brokers.okx_broker import OKXBroker
from .orders.order_manager import OrderManager
from .execution_manager import ExecutionManager
from .monitoring.execution_monitor import ExecutionMonitor

__all__ = [
    'BaseBroker',
    'PaperBroker',
    'OKXBroker',
    'OrderManager',
    'ExecutionManager',
    'ExecutionMonitor',
]
