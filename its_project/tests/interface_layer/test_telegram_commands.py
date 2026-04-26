"""Tests for Telegram bot command handlers."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

# Skip all tests if telegram library is not installed
pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from its_project.telegram_bot.bot import TelegramBot, TelegramConfig


@pytest.fixture
def telegram_config():
    """Telegram bot configuration."""
    return TelegramConfig(
        token="test_token",
        chat_id="test_chat",
        enabled=True,
        commands=True,
        whitelist=["123456789"],
        require_whitelist=True,
    )


@pytest.fixture
def telegram_bot(telegram_config):
    """Telegram bot instance."""
    bot = TelegramBot(telegram_config)
    return bot


def test_whitelist_check_allowed(telegram_bot):
    """Test whitelist check for allowed user."""
    assert telegram_bot._check_whitelist("123456789") is True


def test_whitelist_check_denied(telegram_bot):
    """Test whitelist check for denied user."""
    assert telegram_bot._check_whitelist("999999999") is False


def test_whitelist_disabled(telegram_config):
    """Test whitelist when disabled."""
    config = TelegramConfig(
        token="test_token",
        chat_id="test_chat",
        enabled=True,
        require_whitelist=False,
    )
    bot = TelegramBot(config)
    
    # Should allow all when whitelist disabled
    assert bot._check_whitelist("any_user") is True


def test_set_callbacks(telegram_bot):
    """Test setting callbacks."""
    start_callback = AsyncMock()
    stop_callback = AsyncMock()
    
    telegram_bot.set_callbacks(start_callback, stop_callback)
    
    assert telegram_bot._start_callback is start_callback
    assert telegram_bot._stop_callback is stop_callback


def test_set_system_status(telegram_bot):
    """Test setting system status."""
    telegram_bot.set_system_status("running")
    
    assert telegram_bot._system_status == "running"


@pytest.mark.asyncio
async def test_handle_start_command_authorized(telegram_bot):
    """Test start command with authorized user."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Set callback
    start_callback = AsyncMock()
    telegram_bot.set_callbacks(start_callback=start_callback)
    
    # Handle command
    await telegram_bot._handle_start_command(update, context)
    
    # Verify callback was called
    start_callback.assert_called_once()
    
    # Verify status updated
    assert telegram_bot._system_status == "running"


@pytest.mark.asyncio
async def test_handle_start_command_unauthorized(telegram_bot):
    """Test start command with unauthorized user."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with unauthorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 999999999
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Handle command
    await telegram_bot._handle_start_command(update, context)
    
    # Verify unauthorized message sent
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "not authorized" in call_args.lower()


@pytest.mark.asyncio
async def test_handle_stop_command_authorized(telegram_bot):
    """Test stop command with authorized user."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Set status to running first
    telegram_bot.set_system_status("running")
    
    # Set callback
    stop_callback = AsyncMock()
    telegram_bot.set_callbacks(stop_callback=stop_callback)
    
    # Handle command
    await telegram_bot._handle_stop_command(update, context)
    
    # Verify callback was called
    stop_callback.assert_called_once()
    
    # Verify status updated
    assert telegram_bot._system_status == "stopped"


@pytest.mark.asyncio
async def test_handle_status_command_authorized(telegram_bot):
    """Test status command with authorized user."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Set status
    telegram_bot.set_system_status("running")
    
    # Handle command
    await telegram_bot._handle_status_command(update, context)
    
    # Verify status message sent
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args
    assert call_args[1]["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_handle_help_command_authorized(telegram_bot):
    """Test help command with authorized user."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Handle command
    await telegram_bot._handle_help_command(update, context)
    
    # Verify help message sent
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args
    assert call_args[1]["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_handle_start_already_running(telegram_bot):
    """Test start command when already running."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Set status to running
    telegram_bot.set_system_status("running")
    
    # Handle command
    await telegram_bot._handle_start_command(update, context)
    
    # Verify already running message
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "already running" in call_args.lower()


@pytest.mark.asyncio
async def test_handle_stop_already_stopped(telegram_bot):
    """Test stop command when already stopped."""
    from telegram import Update, User
    from telegram.ext import ContextTypes
    
    # Mock update with authorized user
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 123456789
    update.message = Mock()
    update.message.reply_text = AsyncMock()
    
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Status is already stopped
    telegram_bot.set_system_status("stopped")
    
    # Handle command
    await telegram_bot._handle_stop_command(update, context)
    
    # Verify already stopped message
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "already stopped" in call_args.lower()


def test_telegram_config_defaults():
    """Test Telegram config defaults."""
    config = TelegramConfig(
        token="test",
        chat_id="test",
    )
    
    assert config.enabled is True
    assert config.notifications is True
    assert config.commands is True
    assert config.whitelist is None
    assert config.require_whitelist is False


def test_create_telegram_bot_from_dict():
    """Test creating bot from configuration dictionary."""
    config_dict = {
        "token": "test_token",
        "chat_id": "test_chat",
        "enabled": True,
        "notifications": True,
        "commands": True,
        "whitelist": ["123456789"],
        "require_whitelist": True,
    }
    
    from its_project.telegram_bot.bot import create_telegram_bot
    bot = create_telegram_bot(config_dict)
    
    assert isinstance(bot, TelegramBot)
    assert bot.config.token == "test_token"
    assert bot.config.chat_id == "test_chat"
    assert bot.config.whitelist == ["123456789"]
    assert bot.config.require_whitelist is True
