"""Live L2 → hourly OBI/spread (production path; stub until WS data layer is wired)."""

from __future__ import annotations

import pandas as pd

from feature_engineering.microstructure.order_book import OrderBookMicrostructure


def obi_from_snapshot(bids: list, asks: list, depth: int = 20) -> float:
    """Manual OBI on a single order book snapshot."""
    return OrderBookMicrostructure(depth=depth).calculate_obi(bids, asks)


def spread_from_snapshot(bids: list, asks: list) -> float:
    return OrderBookMicrostructure().get_spread(bids, asks)


def align_l2_series(
    l2_df: pd.DataFrame,
    rule: str = "1h",
    obi_col: str = "order_book_imbalance",
    spread_col: str = "bid_ask_spread",
) -> pd.DataFrame:
    """
  Resample tick/snapshot L2 features to base timeframe (last + ffill).

  ``l2_df`` must have DatetimeIndex and OBI/spread columns.
  """
    from synchronization.gap_handler import align_to_base

    cols = [c for c in (obi_col, spread_col) if c in l2_df.columns]
    if not cols:
        raise ValueError(f"L2 dataframe must contain {obi_col!r} and/or {spread_col!r}")
    return align_to_base(l2_df[cols], rule, "ffill")
