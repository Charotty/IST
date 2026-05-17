"""
Сервисы интроспекции для GUI: ``explain`` (что система собирается сделать на
последнем баре) и ``regime-history`` (как менялся режим за период).

Обе функции работают поверх готового artifact bundle (после ``train_final_for_symbol``)
и не выполняют обучения.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .artifact_bundle import load_orchestrator_bundle, validate_bundle_feature_schema
from .glue import inference_stack_from_bundle
from .symbols import SymbolPaths, paths_for
from .symbol_pipeline import build_features


def _resolve_bundle(symbol: str, timeframe: str = "1h") -> Path:
    sp = paths_for(symbol, timeframe)
    bundle = sp.latest_bundle()
    if bundle is None:
        raise FileNotFoundError(
            f"No artifact bundle for {sp.slug}; run prepare-symbol first."
        )
    return bundle


def _ensure_features(symbol: str, timeframe: str, features: Optional[pd.DataFrame]) -> pd.DataFrame:
    if features is not None and not features.empty:
        return features
    sp = paths_for(symbol, timeframe)
    if not sp.parquet.is_file():
        raise FileNotFoundError(f"parquet not found: {sp.parquet}")
    return build_features(sp.parquet)


def explain_symbol(
    symbol: str,
    timeframe: str = "1h",
    *,
    features: Optional[pd.DataFrame] = None,
    window: int = 256,
) -> Dict[str, Any]:
    """
    «Карточка плана» для последнего бара символа: режим, активные веса, прогнозы
    каждой модели, итоговое решение и причина (если flat).
    """
    sp = paths_for(symbol, timeframe)
    bundle_dir = _resolve_bundle(symbol, timeframe)
    stack = inference_stack_from_bundle(bundle_dir)
    feat = _ensure_features(symbol, timeframe, features)
    cols = list(stack["feature_columns"])
    if not validate_bundle_feature_schema(bundle_dir, cols):
        # Колонки могли расшириться — возьмём ровно то, что в bundle.
        missing = [c for c in cols if c not in feat.columns]
        if missing:
            raise ValueError(f"Features missing columns required by bundle: {missing}")
    win = feat.iloc[-int(window):][cols + (["close"] if "close" in feat.columns else [])].copy()

    orch = stack["orchestrator"]
    orch.initialize(
        models=stack["models"],
        regime_detector=stack["regime_detector"],
        meta_weighting=stack["meta_weighting"],
    )
    result = orch.predict(win)

    # Причина «не торгуем»
    why_blocked: Optional[str] = None
    sig = int(result.signal)
    p = float(result.meta_probability)
    cfg = stack["config"]
    margin = float(getattr(cfg, "min_signal_margin", 0.0) or 0.0)
    thr = float(getattr(cfg, "direction_threshold", 0.5))
    if sig == 0:
        if abs(p - 0.5) < margin:
            why_blocked = "meta probability inside dead-zone (min_signal_margin)"
        elif abs(p - 0.5) < (thr - 0.5):
            why_blocked = "meta probability below direction_threshold"
        else:
            why_blocked = "decision pipeline / trade_mode filtered the signal"

    direction = "long" if sig > 0 else ("short" if sig < 0 else "flat")
    last_close = float(win["close"].iloc[-1]) if "close" in win.columns else None
    return {
        "symbol": sp.symbol,
        "timeframe": sp.timeframe,
        "bundle": str(bundle_dir),
        "as_of": str(win.index[-1]),
        "last_close": last_close,
        "regime": result.regime,
        "active_weights": result.meta_weights,
        "model_probs": result.model_predictions,
        "meta_probability": float(result.meta_probability),
        "confidence": float(result.confidence),
        "direction": direction,
        "signal": sig,
        "position_size_frac": float(result.position_size),
        "why_blocked": why_blocked,
        "active_models": list(result.active_models),
        "config": {
            "direction_threshold": thr,
            "min_signal_margin": margin,
            "trade_mode": getattr(cfg, "trade_mode", "both"),
            "ensemble_mode": getattr(cfg, "ensemble_mode", "regime_adaptive"),
        },
    }


def regime_history(
    symbol: str,
    timeframe: str = "1h",
    *,
    features: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    step: int = 1,
) -> List[Dict[str, Any]]:
    """
    Лента режимов: для каждого бара возвращает регим из ``RegimeDetector``.

    Без обучения, только inference. Подходит для оверлея на цене в GUI.
    """
    sp = paths_for(symbol, timeframe)
    bundle_dir = _resolve_bundle(symbol, timeframe)
    cfg, models, regime_detector, feat_cols, _thr, _hash = load_orchestrator_bundle(bundle_dir)
    feat = _ensure_features(symbol, timeframe, features)

    def _coerce(ts: str) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if isinstance(feat.index, pd.DatetimeIndex) and feat.index.tz is not None and t.tz is None:
            t = t.tz_localize(feat.index.tz)
        return t

    if start:
        feat = feat.loc[feat.index >= _coerce(start)]
    if end:
        feat = feat.loc[feat.index <= _coerce(end)]
    if step > 1:
        feat = feat.iloc[::step]

    info = regime_detector.get_regime_info(feat)
    regime_pred = np.asarray(info.get("regime_pred", []), dtype=int).reshape(-1)
    if regime_pred.shape[0] != len(feat):
        regime_pred = np.resize(regime_pred, len(feat))

    out: List[Dict[str, Any]] = []
    close_series = feat["close"] if "close" in feat.columns else None
    for i, ts in enumerate(feat.index):
        out.append({
            "t": str(ts),
            "regime_int": int(regime_pred[i]),
            "regime": "trend" if int(regime_pred[i]) == 1 else "range",
            "close": float(close_series.iloc[i]) if close_series is not None else None,
        })
    return out
