"""Explain, regime history, chart inference — wraps ``orchestration.introspect``."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from orchestration.introspect import explain_symbol, regime_history
from orchestration.glue import inference_stack_from_bundle
from orchestration.symbols import paths_for

from gui.api.types import ChartBar, ChartPayload, ExplainSnapshot, RegimeBar


class InferenceApi:
    def explain(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        window: int = 256,
    ) -> ExplainSnapshot:
        raw = explain_symbol(symbol, timeframe, window=window)
        regime = raw.get("regime", "range")
        if isinstance(regime, dict):
            regime_name = regime.get("market_regime", "range")
        else:
            regime_name = str(regime)
        weights = raw.get("active_weights") or {}
        if weights and isinstance(next(iter(weights.values())), (np.ndarray, list)):
            weights = {k: float(np.asarray(v).reshape(-1)[-1]) for k, v in weights.items()}
        probs = raw.get("model_probs") or raw.get("model_predictions") or {}
        if probs and isinstance(next(iter(probs.values())), (np.ndarray, list)):
            probs = {k: float(np.asarray(v).reshape(-1)[-1]) for k, v in probs.items()}

        sig = int(raw.get("signal", 0))
        regime_int = 1 if regime_name == "trend" else 0
        return ExplainSnapshot(
            symbol=raw["symbol"],
            timeframe=raw["timeframe"],
            bundle_dir=raw["bundle"],
            as_of=raw["as_of"],
            last_close=raw.get("last_close"),
            regime=regime_name,
            regime_int=regime_int,
            active_weights={k: float(v) for k, v in weights.items()},
            model_probs={k: float(v) for k, v in probs.items()},
            meta_probability=float(raw["meta_probability"]),
            confidence=float(raw.get("confidence", raw["meta_probability"])),
            direction=str(raw["direction"]),
            signal=sig,
            position_size_frac=float(raw.get("position_size_frac", 0.0)),
            why_blocked=raw.get("why_blocked"),
            active_models=list(raw.get("active_models") or probs.keys()),
            config=dict(raw.get("config") or {}),
        )

    def regime_series(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        step: int = 1,
    ) -> List[RegimeBar]:
        rows = regime_history(symbol, timeframe, start=start, end=end, step=step)
        return [
            RegimeBar(
                t=r["t"],
                regime_int=int(r["regime_int"]),
                regime=str(r["regime"]),
                close=r.get("close"),
            )
            for r in rows
        ]

    def chart_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_bars: int = 2000,
        include_signals: bool = True,
        window: int = 512,
    ) -> List[ChartBar]:
        """
        OHLCV from features parquet + optional last-bar signals via bundle inference.

        Full per-bar signals are expensive; default runs pipeline on trailing ``window``
        and maps signals only for that tail.
        """
        from orchestration.symbol_pipeline import build_features

        sp = paths_for(symbol, timeframe)
        feat = build_features(sp.parquet)
        if start:
            feat = feat.loc[feat.index >= pd.Timestamp(start)]
        if end:
            feat = feat.loc[feat.index <= pd.Timestamp(end)]
        if len(feat) > max_bars:
            feat = feat.iloc[-max_bars:]

        signals: dict = {}
        meta_probs: dict = {}
        if include_signals and sp.latest_bundle() is not None:
            try:
                tail = feat.iloc[-window:]
                stack = inference_stack_from_bundle(sp.latest_bundle())
                from orchestration.training_orchestrator import TrainingOrchestrator

                orch = TrainingOrchestrator(stack["config"])
                orch.initialize(
                    models=stack["models"],
                    regime_detector=stack["regime_detector"],
                    meta_weighting=stack["meta_weighting"],
                )
                result = orch.run_pipeline(tail)
                idx = tail.index
                fs = np.asarray(result.final_signals).reshape(-1)
                mp = np.asarray(result.meta_probabilities).reshape(-1)
                for i, ts in enumerate(idx):
                    signals[str(ts)] = int(fs[i])
                    meta_probs[str(ts)] = float(mp[i])
            except Exception:
                pass

        bars: List[ChartBar] = []
        for ts, row in feat.iterrows():
            key = str(ts)
            bars.append(
                ChartBar(
                    t=key,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in feat.columns else None,
                    signal=signals.get(key),
                    meta_probability=meta_probs.get(key),
                )
            )
        return bars

    def merge_regime_into_bars(
        self,
        bars: List[ChartBar],
        regime: List[RegimeBar],
    ) -> List[ChartBar]:
        rmap = {rb.t: rb.regime_int for rb in regime}
        merged: List[ChartBar] = []
        for b in bars:
            ri = rmap.get(b.t)
            merged.append(
                ChartBar(
                    t=b.t,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    signal=b.signal,
                    regime_int=ri if ri is not None else b.regime_int,
                    meta_probability=b.meta_probability,
                )
            )
        return merged

    @staticmethod
    def regime_segments_from_bars(bars: List[ChartBar]) -> List[tuple]:
        """Contiguous (x0, x1, regime_int) indices for chart background."""
        if not bars:
            return []
        segments: List[tuple] = []
        i0 = 0
        cur = bars[0].regime_int if bars[0].regime_int is not None else 0
        for i in range(1, len(bars)):
            ri = bars[i].regime_int if bars[i].regime_int is not None else cur
            if ri != cur:
                segments.append((i0, i - 1, cur))
                i0 = i
                cur = ri
        segments.append((i0, len(bars) - 1, cur))
        return segments

    def chart_payload(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        max_bars: int = 1200,
        window: int = 384,
        regime_step: int = 1,
        include_signals: bool = True,
    ) -> ChartPayload:
        bars = self.chart_bars(
            symbol,
            timeframe,
            max_bars=max_bars,
            window=window,
            include_signals=include_signals,
        )
        if not bars:
            return ChartPayload(bars=[], regime_segments=[])
        start = bars[0].t
        end = bars[-1].t
        regime = self.regime_series(
            symbol, timeframe, start=start, end=end, step=regime_step
        )
        bars = self.merge_regime_into_bars(bars, regime)
        segments = self.regime_segments_from_bars(bars)
        return ChartPayload(bars=bars, regime_segments=segments)
