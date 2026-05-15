"""Research-mode synthetic L2 features (``simulate_l2_features`` in ``ist.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_l2_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic ``order_book_imbalance`` and ``bid_ask_spread`` for backtests without L2.

    Requires ``ema_slope`` and ``volatility`` (run ``FeatureEngine`` first).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    if "ema_slope" not in out.columns or "volatility" not in out.columns:
        raise ValueError("simulate_l2_features requires ema_slope and volatility columns")
    out["order_book_imbalance"] = np.tanh(out["ema_slope"] * 100 + rng.normal(0, 0.1, len(out)))
    out["bid_ask_spread"] = out["volatility"] * 0.05 + 0.0001
    return out
