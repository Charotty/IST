"""
Bridge RiskPipeline ↔ TrainingOrchestrator risk_manager API.

Exposes ``calculate_position_sizes(signals, meta_probabilities, features)`` so the
orchestrator can use ATR sizing + optional trailing stop, then converts
``final_pos_size`` (asset units) into an **unsigned** fraction of equity for
``Backtester.run(..., position_size=...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .risk_pipeline import RiskPipeline


class OrchestratorRiskBridge:
    """
    Wraps ``RiskPipeline`` for orchestrator-compatible position sizing.

    ``meta_probabilities`` is accepted for API compatibility with the orchestrator;
    sizing is driven by ATR + ``final_signal`` (``meta`` may extend sizing later).
    """

    def __init__(
        self,
        risk_pipeline: Optional[RiskPipeline] = None,
        *,
        max_position_fraction: float = 1.0,
    ):
        self.risk_pipeline = risk_pipeline or RiskPipeline()
        self.max_position_fraction = float(max_position_fraction)

    def _ensure_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if "atr" in out.columns:
            return out
        if "volatility" in out.columns and "close" in out.columns:
            out = out.copy()
            out["atr"] = (out["volatility"].astype(float) * out["close"].astype(float)).clip(lower=1e-8)
            return out
        if "close" not in out.columns:
            raise ValueError("features must include 'close' (and ideally 'atr' or 'volatility') for ATR sizing")
        out = out.copy()
        tr = out["close"].pct_change().abs()
        out["atr"] = tr.rolling(14, min_periods=1).mean() * out["close"]
        out["atr"] = out["atr"].bfill().ffill().clip(lower=1e-8)
        return out

    def calculate_position_sizes(
        self,
        signals: np.ndarray,
        meta_probabilities: np.ndarray,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Returns non-negative scale factors: notional / account_size per bar.
        """
        _ = meta_probabilities
        df = features.copy()
        df["final_signal"] = np.asarray(signals, dtype=float).reshape(-1)
        if len(df) != len(df["final_signal"]):
            raise ValueError("signals length must match features rows")
        df = self._ensure_atr(df)
        df = self.risk_pipeline.apply_pipeline(df, signal_col="final_signal")
        account = float(self.risk_pipeline.position_sizer.account_size)
        close = df["close"].astype(float)
        final_sz = df["final_pos_size"].astype(float)
        frac = (final_sz * close / account).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = frac.to_numpy()
        cap = self.max_position_fraction
        if 0 < cap < 1.0:
            return np.clip(arr, 0.0, cap)
        return np.clip(arr, 0.0, None)
