"""Research-mode synthetic L2 features (``simulate_l2_features`` in ``ist.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_l2_features(df: pd.DataFrame, seed: int = 42, mode: str = "research") -> pd.DataFrame:
    """
    Synthetic ``order_book_imbalance`` and ``bid_ask_spread`` for backtests without L2.

    Requires ``ema_slope`` and ``volatility`` (run ``FeatureEngine`` first).

    WARNING: This simulator creates OBI as a function of ema_slope, which creates
    false predictability since direction/regime models also use ema_slope.
    
    Modes:
    - "research": Use for research/backtesting only (default)
    - "off": Disable simulation, return original df
    - "live": Not supported - use real L2 data in production

    :param df: DataFrame with features
    :param seed: Random seed for reproducibility
    :param mode: Simulation mode ("research", "off", "live")
    :return: DataFrame with simulated L2 features
    """
    if mode == "off":
        return df.copy()
    
    if mode == "live":
        raise ValueError("Live mode not supported - use real L2 data in production")
    
    if mode != "research":
        raise ValueError(f"Invalid mode: {mode}. Use 'research', 'off', or 'live'")
    
    out = df.copy()
    rng = np.random.default_rng(seed)
    if "ema_slope" not in out.columns or "volatility" not in out.columns:
        raise ValueError("simulate_l2_features requires ema_slope and volatility columns")
    
    # WARNING: This creates false predictability - OBI correlates with ema_slope
    # which is also used by direction/regime models
    out["order_book_imbalance"] = np.tanh(out["ema_slope"] * 100 + rng.normal(0, 0.1, len(out)))
    out["bid_ask_spread"] = out["volatility"] * 0.05 + 0.0001
    return out
