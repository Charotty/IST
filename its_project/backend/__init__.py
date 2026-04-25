"""
Backend module for OKX WS + CCXT integration.
"""

from .service import BackendService, get_service, create_service

__all__ = ['BackendService', 'get_service', 'create_service']
