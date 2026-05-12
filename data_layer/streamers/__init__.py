"""
Streamers Module

Потоковая передача данных в реальном времени.
"""

from .websocket_streamer import WebSocketStreamer
from .rest_streamer import RESTStreamer

__all__ = ['WebSocketStreamer', 'RESTStreamer']
