from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LOBLevel:
    """Single level in the Limit Order Book."""
    price: float
    size: float
    orders: int = 1


@dataclass
class LOBSnapshot:
    """Full snapshot of the Limit Order Book."""
    symbol: str
    timestamp_ms: int
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    sequence: int = 0


@dataclass
class LOBDelta:
    """Incremental update to the Limit Order Book."""
    symbol: str
    timestamp_ms: int
    sequence: int
    bids: List[Tuple[float, float, str]] = field(default_factory=list)  # (price, size, action)
    asks: List[Tuple[float, float, str]] = field(default_factory=list)  # (price, size, action)
    # action: 'insert', 'update', 'delete'


class LOBDeltaRecovery:
    """
    Reconstruct Limit Order Book from snapshots and incremental updates.
    
    This class maintains the current state of the order book by applying
    delta updates to the most recent snapshot. It handles sequence gaps
    and provides validation to ensure data integrity.
    """

    def __init__(self, max_depth: int = 20) -> None:
        """
        Initialize LOB delta recovery.
        
        Args:
            max_depth: Maximum number of levels to maintain on each side
        """
        self.max_depth = max_depth
        self._snapshots: Dict[str, LOBSnapshot] = {}
        self._current_lob: Dict[str, Dict[str, List[LOBLevel]]] = {}
        self._last_sequence: Dict[str, int] = {}
        self._sequence_gaps: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def apply_snapshot(self, snapshot: LOBSnapshot) -> None:
        """
        Apply a full snapshot to reset the order book state.
        
        Args:
            snapshot: Full LOB snapshot
        """
        symbol = snapshot.symbol
        
        # Store snapshot
        self._snapshots[symbol] = snapshot
        self._last_sequence[symbol] = snapshot.sequence
        
        # Initialize current LOB from snapshot
        self._current_lob[symbol] = {
            "bids": [LOBLevel(price, size) for price, size in snapshot.bids[:self.max_depth]],
            "asks": [LOBLevel(price, size) for price, size in snapshot.asks[:self.max_depth]]
        }
        
        # Clear sequence gaps
        self._sequence_gaps[symbol] = []
        
        logger.debug(f"Applied snapshot for {symbol} at sequence {snapshot.sequence}")

    def apply_delta(self, delta: LOBDelta) -> bool:
        """
        Apply an incremental delta update to the order book.
        
        Args:
            delta: Incremental LOB update
            
        Returns:
            True if delta was applied successfully, False if rejected
        """
        symbol = delta.symbol
        
        # Check if we have a snapshot
        if symbol not in self._snapshots:
            logger.warning(f"No snapshot available for {symbol}, rejecting delta")
            return False
        
        # Check sequence continuity
        last_seq = self._last_sequence.get(symbol, 0)
        if delta.sequence != last_seq + 1:
            # Sequence gap detected
            gap_start = last_seq + 1
            gap_end = delta.sequence - 1
            if gap_start <= gap_end:
                self._sequence_gaps[symbol].append((gap_start, gap_end))
                logger.warning(
                    f"Sequence gap for {symbol}: expected {last_seq + 1}, got {delta.sequence}. "
                    f"Gap: {gap_start}-{gap_end}"
                )
            # Still apply the delta, but mark as potentially inconsistent
        
        # Apply bid updates
        if symbol not in self._current_lob:
            self._current_lob[symbol] = {"bids": [], "asks": []}
        
        bids = self._current_lob[symbol]["bids"]
        for price, size, action in delta.bids:
            self._apply_level_update(bids, price, size, action, "bid")
        
        # Apply ask updates
        asks = self._current_lob[symbol]["asks"]
        for price, size, action in delta.asks:
            self._apply_level_update(asks, price, size, action, "ask")
        
        # Sort and trim to max depth
        bids.sort(key=lambda x: x.price, reverse=True)  # Bids: highest first
        asks.sort(key=lambda x: x.price)  # Asks: lowest first
        
        self._current_lob[symbol]["bids"] = bids[:self.max_depth]
        self._current_lob[symbol]["asks"] = asks[:self.max_depth]
        
        # Update last sequence
        self._last_sequence[symbol] = delta.sequence
        
        return True

    def _apply_level_update(
        self,
        levels: List[LOBLevel],
        price: float,
        size: float,
        action: str,
        side: str
    ) -> None:
        """
        Apply a single level update.
        
        Args:
            levels: Current levels list
            price: Price level
            size: Size at price level
            action: Update action ('insert', 'update', 'delete')
            side: Side ('bid' or 'ask')
        """
        # Find existing level at this price
        existing_idx = None
        for i, level in enumerate(levels):
            if abs(level.price - price) < 1e-8:  # Float comparison with tolerance
                existing_idx = i
                break
        
        if action == "delete":
            if existing_idx is not None:
                del levels[existing_idx]
        elif action == "update":
            if existing_idx is not None:
                levels[existing_idx].size = size
                if size <= 0:
                    del levels[existing_idx]
            else:
                # Update on non-existent level -> insert
                levels.append(LOBLevel(price, size))
        elif action == "insert":
            if existing_idx is None:
                levels.append(LOBLevel(price, size))
            else:
                # Insert on existing level -> update
                levels[existing_idx].size = size
        else:
            logger.warning(f"Unknown action: {action}")

    def get_current_lob(self, symbol: str) -> Optional[Dict[str, List[Tuple[float, float]]]]:
        """
        Get current reconstructed order book.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with 'bids' and 'asks' as lists of (price, size) tuples
        """
        if symbol not in self._current_lob:
            return None
        
        return {
            "bids": [(level.price, level.size) for level in self._current_lob[symbol]["bids"]],
            "asks": [(level.price, level.size) for level in self._current_lob[symbol]["asks"]]
        }

    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Calculate current bid-ask spread.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread value or None if not available
        """
        lob = self.get_current_lob(symbol)
        if not lob or not lob["bids"] or not lob["asks"]:
            return None
        
        best_bid = lob["bids"][0][0]
        best_ask = lob["asks"][0][0]
        return best_ask - best_bid

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """
        Calculate current mid price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Mid price or None if not available
        """
        lob = self.get_current_lob(symbol)
        if not lob or not lob["bids"] or not lob["asks"]:
            return None
        
        best_bid = lob["bids"][0][0]
        best_ask = lob["asks"][0][0]
        return (best_bid + best_ask) / 2.0

    def has_sequence_gaps(self, symbol: str) -> bool:
        """Check if there are sequence gaps for a symbol."""
        return len(self._sequence_gaps.get(symbol, [])) > 0

    def get_sequence_gaps(self, symbol: str) -> List[Tuple[int, int]]:
        """Get sequence gaps for a symbol."""
        return self._sequence_gaps.get(symbol, [])

    def clear_sequence_gaps(self, symbol: str) -> None:
        """Clear recorded sequence gaps for a symbol."""
        self._sequence_gaps[symbol] = []

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a specific symbol."""
        if symbol in self._snapshots:
            del self._snapshots[symbol]
        if symbol in self._current_lob:
            del self._current_lob[symbol]
        if symbol in self._last_sequence:
            del self._last_sequence[symbol]
        if symbol in self._sequence_gaps:
            del self._sequence_gaps[symbol]

    def reset_all(self) -> None:
        """Reset all state."""
        self._snapshots.clear()
        self._current_lob.clear()
        self._last_sequence.clear()
        self._sequence_gaps.clear()

    def get_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get status information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Status dictionary
        """
        lob = self.get_current_lob(symbol)
        spread = self.get_spread(symbol)
        mid_price = self.get_mid_price(symbol)
        
        return {
            "symbol": symbol,
            "has_snapshot": symbol in self._snapshots,
            "last_sequence": self._last_sequence.get(symbol, 0),
            "has_gaps": self.has_sequence_gaps(symbol),
            "gap_count": len(self._sequence_gaps.get(symbol, [])),
            "bid_levels": len(lob["bids"]) if lob else 0,
            "ask_levels": len(lob["asks"]) if lob else 0,
            "spread": spread,
            "mid_price": mid_price,
        }


def create_lob_delta_from_ws_message(message: Dict[str, Any]) -> Optional[LOBDelta]:
    """
    Create LOBDelta from WebSocket message (OKX format).
    
    Args:
        message: WebSocket message dictionary
        
    Returns:
        LOBDelta object or None if invalid
    """
    try:
        data = message.get("data", [])
        if not data:
            return None
        
        first_data = data[0]
        symbol = first_data.get("instId", "")
        timestamp_ms = int(first_data.get("ts", 0))
        
        # OKX sends deltas in 'bids' and 'asks' arrays
        # Each level is [price, size, orders] or [price, size]
        bids_raw = first_data.get("bids", [])
        asks_raw = first_data.get("asks", [])
        
        bids = []
        for level in bids_raw:
            if len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
                action = "update" if size > 0 else "delete"
                bids.append((price, size, action))
        
        asks = []
        for level in asks_raw:
            if len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
                action = "update" if size > 0 else "delete"
                asks.append((price, size, action))
        
        # Sequence number (OKX uses checksum instead)
        sequence = int(first_data.get("checksum", timestamp_ms))
        
        return LOBDelta(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            sequence=sequence,
            bids=bids,
            asks=asks
        )
        
    except Exception as e:
        logger.error(f"Failed to create LOBDelta from message: {e}")
        return None


def create_lob_snapshot_from_ws_message(message: Dict[str, Any]) -> Optional[LOBSnapshot]:
    """
    Create LOBSnapshot from WebSocket message (OKX format).
    
    Args:
        message: WebSocket message dictionary
        
    Returns:
        LOBSnapshot object or None if invalid
    """
    try:
        data = message.get("data", [])
        if not data:
            return None
        
        first_data = data[0]
        symbol = first_data.get("instId", "")
        timestamp_ms = int(first_data.get("ts", 0))
        
        bids_raw = first_data.get("bids", [])
        asks_raw = first_data.get("asks", [])
        
        bids = [(float(level[0]), float(level[1])) for level in bids_raw if len(level) >= 2]
        asks = [(float(level[0]), float(level[1])) for level in asks_raw if len(level) >= 2]
        
        sequence = int(first_data.get("checksum", timestamp_ms))
        
        return LOBSnapshot(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            bids=bids,
            asks=asks,
            sequence=sequence
        )
        
    except Exception as e:
        logger.error(f"Failed to create LOBSnapshot from message: {e}")
        return None
