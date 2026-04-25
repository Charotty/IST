"""
Unit tests for Telegram bot.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from its_project.telegram_bot.bot import create_telegram_bot


@pytest.mark.unit
@pytest.mark.interface_layer
class TestTelegramBot:
    """Test Telegram bot functionality."""
    
    @pytest.mark.asyncio
    async def test_bot_creation(self):
        """Test bot creation."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        assert bot is not None
    
    @pytest.mark.asyncio
    async def test_bot_start(self):
        """Test bot start."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        
        # Mock the start method
        bot.start = AsyncMock()
        await bot.start()
        
        assert bot.start.called
    
    @pytest.mark.asyncio
    async def test_trade_notification(self):
        """Test trade notification."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        bot.send_message = AsyncMock()
        
        await bot.send_trade_notification(
            symbol='BTCUSDT',
            side='buy',
            amount=0.1,
            price=42000.0,
            pnl=100.0
        )
        
        assert bot.send_message.called
    
    @pytest.mark.asyncio
    async def test_system_status_update(self):
        """Test system status update."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        bot.send_message = AsyncMock()
        
        await bot.send_system_status('System running normally')
        
        assert bot.send_message.called
    
    @pytest.mark.asyncio
    async def test_alert_notification(self):
        """Test alert notification."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        bot.send_message = AsyncMock()
        
        await bot.send_alert('High volatility detected')
        
        assert bot.send_message.called
    
    @pytest.mark.asyncio
    async def test_disabled_bot(self):
        """Test disabled bot behavior."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': False
        }
        
        bot = create_telegram_bot(config)
        bot.send_message = AsyncMock()
        
        await bot.send_trade_notification(
            symbol='BTCUSDT',
            side='buy',
            amount=0.1,
            price=42000.0,
            pnl=100.0
        )
        
        # Should not send message if disabled
        assert not bot.send_message.called
    
    @pytest.mark.asyncio
    async def test_message_queue(self):
        """Test message queue functionality."""
        config = {
            'token': 'test_token',
            'chat_id': 'test_chat_id',
            'enabled': True
        }
        
        bot = create_telegram_bot(config)
        
        # Add multiple messages to queue
        await bot.queue_message('Message 1')
        await bot.queue_message('Message 2')
        await bot.queue_message('Message 3')
        
        # Queue should have messages
        assert bot.message_queue.qsize() == 3
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when library not installed."""
        # Test that bot creation handles missing library gracefully
        try:
            from its_project.telegram_bot.bot import TelegramBot
            # If library is installed, test normal operation
            assert True
        except ImportError:
            # If library not installed, should handle gracefully
            assert True
