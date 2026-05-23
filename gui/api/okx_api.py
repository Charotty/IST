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

# OKX public OHLCV cap per REST request (ccxt)

OKX_OHLCV_LIMIT_MAX = 300

OKX_OHLCV_HISTORY_MAX = 3000





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





def _rows_to_bars(rows: list) -> List[ChartBar]:

    bars: List[ChartBar] = []

    for ts_ms, o, h, l, c, v in rows:

        t = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        bars.append(
            ChartBar(
                t=t.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
                ts_ms=int(ts_ms),
            )
        )

    return bars





def _dedupe_rows_by_ts(rows: list) -> list:

    if not rows:

        return []

    out: list = []

    seen: set = set()

    for row in sorted(rows, key=lambda r: r[0]):

        ts = row[0]

        if ts in seen:

            continue

        seen.add(ts)

        out.append(row)

    return out





def fetch_ohlcv_bars(

    symbol: str,

    timeframe: str,

    *,

    limit: int = 400,

) -> List[ChartBar]:

    """Последние ``limit`` свечей с OKX (один запрос)."""

    ex = ccxt.okx({"enableRateLimit": True})

    rows = ex.fetch_ohlcv(

        symbol, timeframe, limit=min(limit, OKX_OHLCV_LIMIT_MAX)

    )

    return _rows_to_bars(rows)





def fetch_ohlcv_history(

    symbol: str,

    timeframe: str,

    *,

    max_bars: int = 900,

) -> List[ChartBar]:

    """История: несколько страниц назад (для прокрутки графика влево)."""

    ex = ccxt.okx({"enableRateLimit": True})

    target = max(1, min(int(max_bars), OKX_OHLCV_HISTORY_MAX))

    tf_ms = int(ex.parse_timeframe(timeframe) * 1000)



    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=OKX_OHLCV_LIMIT_MAX)

    if not ohlcv:

        return []



    while len(ohlcv) < target:

        oldest_ts = ohlcv[0][0]

        since = max(0, oldest_ts - OKX_OHLCV_LIMIT_MAX * tf_ms)

        older = ex.fetch_ohlcv(

            symbol, timeframe, since=since, limit=OKX_OHLCV_LIMIT_MAX

        )

        if not older:

            break

        older = [c for c in older if c[0] < oldest_ts]

        if not older:

            break

        ohlcv = older + ohlcv

        if len(older) < OKX_OHLCV_LIMIT_MAX:

            break



    ohlcv = _dedupe_rows_by_ts(ohlcv)

    return _rows_to_bars(ohlcv[-target:])





def fetch_ohlcv_tail(
    symbol: str,
    timeframe: str,
    *,
    limit: int = 5,
) -> List[ChartBar]:
    """Хвост для live: последние несколько свечей (обновление без полной перезагрузки)."""
    return fetch_ohlcv_bars(symbol, timeframe, limit=max(2, min(limit, OKX_OHLCV_LIMIT_MAX)))


def fetch_okx_live_tail(
    symbol: str,
    timeframe: str,
    *,
    ohlcv_limit: int = 3,
    use_ticker: bool = True,
) -> List[ChartBar]:
    """
    Live-хвост по модели TradingView ``series.update()``:
    OHLCV даёт закрытые + формирующуюся свечу; тикер двигает close последней.
    """
    bars = fetch_ohlcv_tail(symbol, timeframe, limit=ohlcv_limit)
    if not use_ticker or not bars:
        return bars
    try:
        ex = ccxt.okx({"enableRateLimit": True})
        ticker = ex.fetch_ticker(symbol)
        last = ticker.get("last")
        if last is not None:
            bars = _patch_forming_bar_price(bars, float(last))
    except Exception:
        pass
    return bars


def _patch_forming_bar_price(bars: List[ChartBar], last_price: float) -> List[ChartBar]:
    if not bars or last_price <= 0:
        return bars
    last = bars[-1]
    out = list(bars)
    out[-1] = ChartBar(
        t=last.t,
        open=last.open,
        high=max(last.high, last_price),
        low=min(last.low, last_price),
        close=last_price,
        volume=last.volume,
        ts_ms=last.ts_ms,
    )
    return out


