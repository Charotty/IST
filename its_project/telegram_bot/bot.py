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


class TelegramBot:
    """Telegram bot for trading system notifications and control."""
    
    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self._running = False
        self._bot = None
        self._message_queue = asyncio.Queue()
        
    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.config.enabled:
            logger.info("Telegram bot is disabled in config")
            return
        
        try:
            # Try to import telegram library
            from telegram import Bot
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
            
            self._bot = Bot(token=self.config.token)
            self._running = True
            
            # Start message processor
            asyncio.create_task(self._process_messages())
            
            logger.info("Telegram bot started successfully")
            
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
        commands=config.get("commands", True)
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
