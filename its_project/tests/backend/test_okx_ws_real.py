"""
Test real OKX WebSocket connection.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.okx_ws_client import OKXWSClient


async def test_okx_ws_connection():
    """Test OKX WebSocket connection with real data."""
    print("Testing OKX WebSocket connection...")
    
    client = OKXWSClient()
    
    # Track received messages
    received_messages = {
        'price': 0,
        'orderbook': 0,
        'candle': 0,
    }
    
    def on_price(event):
        received_messages['price'] += 1
        print(f"[PRICE] {event.symbol}: {event.last} (bid: {event.bid}, ask: {event.ask})")
    
    def on_orderbook(event):
        received_messages['orderbook'] += 1
        print(f"[ORDERBOOK] {event.symbol}: {len(event.bids)} bids, {len(event.asks)} asks")
    
    def on_candle(event):
        received_messages['candle'] += 1
        print(f"[CANDLE] {event.symbol} {event.timeframe}: O={event.open} H={event.high} L={event.low} C={event.close}")
    
    # Also log all raw messages
    original_handle = client._handle_message
    async def debug_handle(message):
        print(f"[RAW] {message}")
        await original_handle(message)
    client._handle_message = debug_handle
    
    # Also log message loop activity
    original_message_loop = client._message_loop
    async def debug_message_loop():
        print("[DEBUG] Message loop started")
        await original_message_loop()
        print("[DEBUG] Message loop ended")
    client._message_loop = debug_message_loop
    
    # Register callbacks
    client.on_price(on_price)
    client.on_orderbook(on_orderbook)
    client.on_candle(on_candle)
    
    try:
        # Connect
        print("Connecting to OKX WebSocket...")
        connected = await client.connect()
        
        if not connected:
            print("Failed to connect to OKX WebSocket")
            return False
        
        print("Connected successfully!")
        
        # Start message loop in background
        message_loop_task = asyncio.create_task(client._message_loop())
        
        # Wait a moment for message loop to start
        await asyncio.sleep(1)
        
        # Subscribe to tickers for BTC/USDT
        print("Subscribing to tickers for BTC/USDT...")
        await client.subscribe_tickers(['BTC/USDT'])
        
        # Subscribe to orderbook
        print("Subscribing to orderbook for BTC/USDT...")
        await client.subscribe_orderbook(['BTC/USDT'], depth=5)
        
        # Subscribe to candles
        print("Subscribing to candles for BTC/USDT (1m)...")
        await client.subscribe_candles(['BTC/USDT'], ['1m'])
        
        # Wait for messages
        print("Waiting for messages (30 seconds)...")
        for i in range(30):
            await asyncio.sleep(1)
            if i % 5 == 0:
                print(f"Waiting... {i}s elapsed")
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Price updates: {received_messages['price']}")
        print(f"Orderbook updates: {received_messages['orderbook']}")
        print(f"Candle updates: {received_messages['candle']}")
        
        # Unsubscribe
        print("\nUnsubscribing...")
        await client.unsubscribe_all()
        
        # Stop message loop
        print("Stopping message loop...")
        client.running = False
        
        # Wait for message loop to end
        try:
            await asyncio.wait_for(message_loop_task, timeout=5)
        except asyncio.TimeoutError:
            print("Message loop did not stop in time")
        
        # Disconnect
        print("Disconnecting...")
        await client.disconnect()
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure disconnect
        await client.disconnect()


if __name__ == '__main__':
    result = asyncio.run(test_okx_ws_connection())
    sys.exit(0 if result else 1)
