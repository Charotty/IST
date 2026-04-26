#!/usr/bin/env python3
"""
Telegram Bot for Intelligent Trading System
==========================================

Provides notifications and remote control via Telegram.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    token: str
    chat_id: str
    enabled: bool = True
    notifications: bool = True
    commands: bool = True
    whitelist: List[str] = None  # List of allowed user IDs
    require_whitelist: bool = False


class TelegramBot:
    """Telegram bot for trading system notifications and control."""
    
    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        if self.config.whitelist is None:
            self.config.whitelist = []
        self._running = False
        self._bot = None
        self._message_queue = asyncio.Queue()
        self._application = None
        self._system_status = "stopped"  # Current system status
        self._start_callback = None  # Callback for start command
        self._stop_callback = None  # Callback for stop command
        
    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.config.enabled:
            logger.info("Telegram bot is disabled in config")
            return
        
        try:
            # Try to import telegram library
            from telegram import Bot, Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
            
            self._bot = Bot(token=self.config.token)
            self._running = True
            
            # Create application with command handlers
            self._application = Application.builder().token(self.config.token).build()
            
            # Register command handlers
            self._application.add_handler(CommandHandler("start", self._handle_start_command))
            self._application.add_handler(CommandHandler("stop", self._handle_stop_command))
            self._application.add_handler(CommandHandler("status", self._handle_status_command))
            self._application.add_handler(CommandHandler("help", self._handle_help_command))
            
            # Start message processor
            asyncio.create_task(self._process_messages())
            
            # Start the application (polling)
            if self.config.commands:
                await self._application.initialize()
                await self._application.start()
                await self._application.updater.start_polling()
                logger.info("Telegram bot started with command handlers")
            else:
                logger.info("Telegram bot started (commands disabled)")
            
        except ImportError:
            logger.warning("python-telegram-bot not installed, Telegram bot disabled")
            logger.info("Install with: pip install python-telegram-bot")
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            self.config.enabled = False
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        
        if self._application:
            try:
                await self._application.updater.stop()
                await self._application.stop()
                await self._application.shutdown()
            except Exception as e:
                logger.error(f"Error stopping Telegram application: {e}")
        
        logger.info("Telegram bot stopped")
    
    async def send_notification(
        self, 
        message: str, 
        parse_mode: str = "HTML"
    ) -> bool:
        """Send notification to Telegram."""
        if not self.config.enabled or not self.config.notifications:
            return False
        
        try:
            await self._message_queue.put({
                "type": "notification",
                "message": message,
                "parse_mode": parse_mode
            })
            return True
        except Exception as e:
            logger.error(f"Failed to queue notification: {e}")
            return False
    
    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        amount: float,
        price: float,
        pnl: Optional[float] = None
    ) -> bool:
        """Send trade execution notification."""
        message = f"""
🚀 <b>Trade Executed</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Action:</b> {action.upper()}
💰 <b>Amount:</b> {amount}
💵 <b>Price:</b> ${price:.2f}
"""
        
        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            message += f"{pnl_emoji} <b>PnL:</b> ${pnl:.2f}\n"
        
        message += f"⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_notification(message)
    
    async def send_system_status(
        self,
        status: str,
        uptime: float,
        active_orders: int,
        balance: Dict[str, float]
    ) -> bool:
        """Send system status notification."""
        message = f"""
📊 <b>System Status</b>

🔄 <b>Status:</b> {status}
⏱ <b>Uptime:</b> {uptime:.0f}s
📋 <b>Active Orders:</b> {active_orders}
💼 <b>Balance:</b>
"""
        
        for asset, amount in balance.items():
            message += f"   {asset}: {amount:.4f}\n"
        
        message += f"⏰ <b>Last Update:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_notification(message)
    
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING"
    ) -> bool:
        """Send alert notification."""
        severity_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }.get(severity, "ℹ️")
        
        formatted_message = f"""
{severity_emoji} <b>{alert_type}</b>

{message}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_notification(formatted_message)
    
    async def _process_messages(self) -> None:
        """Process message queue and send to Telegram."""
        while self._running:
            try:
                message_data = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                await self._send_message(message_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _send_message(self, message_data: Dict[str, Any]) -> None:
        """Send message to Telegram API."""
        try:
            if self._bot:
                await self._bot.send_message(
                    chat_id=self.config.chat_id,
                    text=message_data["message"],
                    parse_mode=message_data.get("parse_mode", "HTML")
                )
                logger.debug("Message sent to Telegram")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def is_enabled(self) -> bool:
        """Check if bot is enabled."""
        return self.config.enabled
    
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running

    def _check_whitelist(self, user_id: str) -> bool:
        """
        Check if user is in whitelist.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is allowed, False otherwise
        """
        if not self.config.require_whitelist:
            return True
        
        return str(user_id) in self.config.whitelist

    def set_callbacks(self, start_callback=None, stop_callback=None) -> None:
        """
        Set callbacks for start/stop commands.
        
        Args:
            start_callback: Async callback for start command
            stop_callback: Async callback for stop command
        """
        self._start_callback = start_callback
        self._stop_callback = stop_callback

    def set_system_status(self, status: str) -> None:
        """Set current system status."""
        self._system_status = status

    async def _handle_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        user_id = update.effective_user.id
        
        if not self._check_whitelist(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        
        if self._system_status == "running":
            await update.message.reply_text("⚠️ System is already running.")
            return
        
        await update.message.reply_text("🚀 Starting trading system...")
        
        if self._start_callback:
            try:
                await self._start_callback()
                self._system_status = "running"
                await update.message.reply_text("✅ Trading system started successfully.")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to start: {e}")
                logger.error(f"Start callback error: {e}")
        else:
            self._system_status = "running"
            await update.message.reply_text("✅ Trading system started (no callback set).")

    async def _handle_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stop command."""
        user_id = update.effective_user.id
        
        if not self._check_whitelist(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        
        if self._system_status == "stopped":
            await update.message.reply_text("⚠️ System is already stopped.")
            return
        
        await update.message.reply_text("🛑 Stopping trading system...")
        
        if self._stop_callback:
            try:
                await self._stop_callback()
                self._system_status = "stopped"
                await update.message.reply_text("✅ Trading system stopped successfully.")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to stop: {e}")
                logger.error(f"Stop callback error: {e}")
        else:
            self._system_status = "stopped"
            await update.message.reply_text("✅ Trading system stopped (no callback set).")

    async def _handle_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command."""
        user_id = update.effective_user.id
        
        if not self._check_whitelist(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        
        status_emoji = "🟢" if self._system_status == "running" else "🔴"
        
        message = f"""
📊 <b>System Status</b>

{status_emoji} <b>Status:</b> {self._system_status.upper()}
🤖 <b>Bot:</b> {'Running' if self._running else 'Stopped'}
📨 <b>Notifications:</b> {'Enabled' if self.config.notifications else 'Disabled'}
🎛️ <b>Commands:</b> {'Enabled' if self.config.commands else 'Disabled'}
🔒 <b>Whitelist:</b> {'Active' if self.config.require_whitelist else 'Inactive'}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        await update.message.reply_text(message, parse_mode="HTML")

    async def _handle_help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        user_id = update.effective_user.id
        
        if not self._check_whitelist(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        
        help_message = """
🤖 <b>ITS Trading Bot Commands</b>

<b>Available Commands:</b>
/start - Start the trading system
/stop - Stop the trading system
/status - Get current system status
/help - Show this help message

<b>Notifications:</b>
You will receive notifications for:
• Trade executions
• System status changes
• Alerts and warnings

⏰ <b>Time:</b> {time}
""".format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        await update.message.reply_text(help_message, parse_mode="HTML")


class TelegramNotifier:
    """Simple notification interface for trading system."""
    
    def __init__(self, bot: TelegramBot) -> None:
        self.bot = bot
    
    async def notify_trade(
        self,
        symbol: str,
        action: str,
        amount: float,
        price: float,
        pnl: Optional[float] = None
    ) -> None:
        """Notify about trade execution."""
        if self.bot.is_enabled():
            await self.bot.send_trade_notification(symbol, action, amount, price, pnl)
    
    async def notify_status(
        self,
        status: str,
        uptime: float,
        active_orders: int,
        balance: Dict[str, float]
    ) -> None:
        """Notify about system status."""
        if self.bot.is_enabled():
            await self.bot.send_system_status(status, uptime, active_orders, balance)
    
    async def notify_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING"
    ) -> None:
        """Send alert notification."""
        if self.bot.is_enabled():
            await self.bot.send_alert(alert_type, message, severity)


def create_telegram_bot(config: Dict[str, Any]) -> TelegramBot:
    """Create Telegram bot from configuration dictionary."""
    telegram_config = TelegramConfig(
        token=config.get("token", ""),
        chat_id=config.get("chat_id", ""),
        enabled=config.get("enabled", False),
        notifications=config.get("notifications", True),
        commands=config.get("commands", True),
        whitelist=config.get("whitelist", []),
        require_whitelist=config.get("require_whitelist", False)
    )
    
    return TelegramBot(telegram_config)


# Standalone test function
async def test_telegram_bot():
    """Test Telegram bot functionality."""
    import os
    
    # Get credentials from environment
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables required")
        print("Set them with:")
        print("export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("export TELEGRAM_CHAT_ID='your_chat_id'")
        return
    
    config = TelegramConfig(
        token=token,
        chat_id=chat_id,
        enabled=True
    )
    
    bot = TelegramBot(config)
    
    try:
        # Start bot
        await bot.start()
        
        # Send test message
        await bot.send_notification("🚀 <b>Telegram Bot Test</b>\n\nBot is working correctly!")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Send trade notification
        await bot.send_trade_notification(
            symbol="BTC/USDT",
            action="buy",
            amount=0.001,
            price=42000.0,
            pnl=100.0
        )
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Send system status
        await bot.send_system_status(
            status="running",
            uptime=3600.0,
            active_orders=5,
            balance={"USDT": 10000.0, "BTC": 0.5}
        )
        
        print("Test messages sent successfully!")
        
        # Stop bot
        await bot.stop()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_telegram_bot())
