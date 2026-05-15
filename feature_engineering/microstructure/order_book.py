"""Order book microstructure (OBI, spread) — ``ist.py`` reference."""

from __future__ import annotations

from typing import Sequence


class OrderBookMicrostructure:
    def __init__(self, depth: int = 20):
        self.depth = depth

    def calculate_obi(self, bids: Sequence, asks: Sequence) -> float:
        """(BidVol - AskVol) / (BidVol + AskVol) over top ``depth`` levels."""
        bid_vol = sum(b[1] for b in bids[: self.depth])
        ask_vol = sum(a[1] for a in asks[: self.depth])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def get_mid_price(self, bids: Sequence, asks: Sequence) -> float:
        return (bids[0][0] + asks[0][0]) / 2

    def get_spread(self, bids: Sequence, asks: Sequence) -> float:
        return (asks[0][0] - bids[0][0]) / bids[0][0]
