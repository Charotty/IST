"""Merge OHLCV bars by open-time for live chart updates (TradingView update() semantics)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple

from gui.api.types import ChartBar


class LiveBarChange(Enum):
    """Как изменился ряд после live-обновления (аналог series.update)."""

    NONE = "none"
    FORMING = "forming"  # тот же open time — обновление текущей свечи
    NEW_BAR = "new_bar"  # open time больше последнего — новая свеча


@dataclass
class LiveMergeResult:
    bars: List[ChartBar]
    change: LiveBarChange


def bar_open_time_ms(bar: ChartBar) -> int:
    if bar.ts_ms is not None:
        return int(bar.ts_ms)
    s = bar.t.replace("Z", "+00:00")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def merge_chart_bars(
    existing: List[ChartBar],
    incoming: List[ChartBar],
) -> LiveMergeResult:
    """
    Merge tail into history by candle **open time** (ms).

    - Same open time as last bar → replace last (forming candle).
    - Strictly greater open time → append (brand-new candle).
    """
    if not incoming:
        return LiveMergeResult(list(existing), LiveBarChange.NONE)
    if not existing:
        return LiveMergeResult(list(incoming), LiveBarChange.NEW_BAR)

    merged = list(existing)
    by_ms = {bar_open_time_ms(b): i for i, b in enumerate(merged)}
    last_ms = bar_open_time_ms(merged[-1])
    change = LiveBarChange.NONE

    for bar in incoming:
        ms = bar_open_time_ms(bar)
        idx = by_ms.get(ms)
        if idx is not None:
            merged[idx] = bar
            if ms == last_ms:
                change = LiveBarChange.FORMING
        elif ms > last_ms:
            merged.append(bar)
            by_ms[ms] = len(merged) - 1
            last_ms = ms
            change = LiveBarChange.NEW_BAR
        # ms < last_ms: older history chunk — prepend only if needed
        elif ms < bar_open_time_ms(merged[0]):
            merged.insert(0, bar)
            by_ms = {bar_open_time_ms(b): i for i, b in enumerate(merged)}

    merged.sort(key=bar_open_time_ms)
    return LiveMergeResult(merged, change)


def patch_forming_bar_price(bars: List[ChartBar], last_price: float) -> List[ChartBar]:
    """Обновить close/high/low последней (формирующейся) свечи по тикеру."""
    if not bars or last_price <= 0:
        return bars
    last = bars[-1]
    bars = list(bars)
    bars[-1] = ChartBar(
        t=last.t,
        ts_ms=last.ts_ms,
        open=last.open,
        high=max(last.high, last_price),
        low=min(last.low, last_price),
        close=last_price,
        volume=last.volume,
        signal=last.signal,
        regime_int=last.regime_int,
        meta_probability=last.meta_probability,
    )
    return bars
