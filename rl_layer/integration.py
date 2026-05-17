"""
Подключение rl_layer к выходам оркестратора / фичам.

Среды ожидают фиксированные колонки; недостающие заполняются безопасными значениями.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.training_orchestrator import TrainingResult


def prepare_df_for_rl_env(
    features: pd.DataFrame,
    final_signal: Union[np.ndarray, pd.Series],
    meta_prob: Union[np.ndarray, pd.Series],
    regime_pred: Union[np.ndarray, pd.Series],
    *,
    direction_prob: Optional[Union[np.ndarray, pd.Series]] = None,
    vol_spike_prob: Union[float, np.ndarray] = 0.5,
    rsi: Optional[Union[np.ndarray, pd.Series]] = None,
    volatility: Optional[Union[np.ndarray, pd.Series]] = None,
    ema_slope: Union[float, np.ndarray] = 0.0,
    adx: Union[float, np.ndarray] = 20.0,
    rsi_15m: Union[float, np.ndarray] = 50.0,
    rsi_4h: Union[float, np.ndarray] = 50.0,
    order_book_imbalance: Union[float, np.ndarray] = 0.0,
) -> pd.DataFrame:
    """
    Копия ``features`` с колонками для ``TradingEnvironment`` / ``EnsembleTradingEnv`` / ``MicrostructureRLenv``.
    """
    df = features.reset_index(drop=True).copy()
    n = len(df)
    sig = np.asarray(final_signal, dtype=float).reshape(-1)
    if sig.shape[0] != n:
        raise ValueError("final_signal length must match features")
    df["final_signal"] = sig
    df["meta_prob"] = np.asarray(meta_prob, dtype=float).reshape(-1)
    df["regime_pred"] = np.asarray(regime_pred, dtype=float).reshape(-1)
    if direction_prob is not None:
        df["direction_prob"] = np.asarray(direction_prob, dtype=float).reshape(-1)
    else:
        df["direction_prob"] = df["meta_prob"]
    if "ensemble_prob" not in df.columns:
        df["ensemble_prob"] = df["direction_prob"]

    vsp = np.asarray(vol_spike_prob, dtype=float)
    if vsp.ndim == 0:
        df["vol_spike_prob"] = float(vsp)
    else:
        df["vol_spike_prob"] = vsp.reshape(-1)

    if rsi is not None:
        df["rsi"] = np.asarray(rsi, dtype=float).reshape(-1)
    elif "rsi" not in df.columns:
        df["rsi"] = 50.0
    if volatility is not None:
        df["volatility"] = np.asarray(volatility, dtype=float).reshape(-1)
    elif "volatility" not in df.columns:
        df["volatility"] = 0.02

    for col, default in [
        ("ema_slope", ema_slope),
        ("adx", adx),
        ("rsi_15m", rsi_15m),
        ("rsi_4h", rsi_4h),
    ]:
        if col not in df.columns:
            v = np.asarray(default, dtype=float)
            df[col] = float(v) if v.ndim == 0 else v.reshape(-1)

    obi = np.asarray(order_book_imbalance, dtype=float)
    if obi.ndim == 0:
        df["order_book_imbalance"] = float(obi)
    else:
        df["order_book_imbalance"] = obi.reshape(-1)

    if "close" not in df.columns:
        raise ValueError("features must include 'close' for RL env step()")
    return df


def prepare_from_training_result(
    features: pd.DataFrame,
    result: "TrainingResult",
    **kwargs,
) -> pd.DataFrame:
    """Обертка для ``TrainingResult`` оркестратора."""
    return prepare_df_for_rl_env(
        features,
        result.final_signals,
        result.meta_probabilities,
        result.regime_predictions,
        direction_prob=result.meta_probabilities,
        **kwargs,
    )
