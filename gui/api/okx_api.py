"""OKX: список рынков и OHLCV для GUI (ccxt, без Qt)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import ccxt

from gui.api.types import ChartBar

_PRIORITY = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
)

_FALLBACK_SYMBOLS = list(_PRIORITY)
_STANDARD_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def standard_timeframes() -> List[str]:
    return list(_STANDARD_TIMEFRAMES)


def list_usdt_spot_symbols(*, quote: str = "USDT") -> List[str]:
    """Активные spot-пары OKX с котировкой USDT (ccxt unified symbols)."""
    try:
        ex = ccxt.okx({"enableRateLimit": True})
        markets = ex.load_markets()
    except Exception:
        return list(_FALLBACK_SYMBOLS)

    symbols: List[str] = []
    for m in markets.values():
        if not m.get("active", True):
            continue
        if m.get("type") != "spot":
            continue
        if m.get("quote") != quote:
            continue
        sym = m.get("symbol")
        if sym:
            symbols.append(sym)

    symbols = sorted(set(symbols))
    if not symbols:
        return list(_FALLBACK_SYMBOLS)

    pri = [s for s in _PRIORITY if s in symbols]
    rest = [s for s in symbols if s not in pri]
    return pri + rest


def fetch_ohlcv_bars(
    symbol: str,
    timeframe: str,
    *,
    limit: int = 400,
) -> List[ChartBar]:
    """Последние свечи с OKX."""
    ex = ccxt.okx({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe, limit=min(limit, 300))
    bars: List[ChartBar] = []
    for ts_ms, o, h, l, c, v in rows:
        t = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        bars.append(
            ChartBar(
                t=str(t),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )
        )
    return bars
