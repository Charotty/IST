"""Typed payloads for GUI ↔ orchestration (PyQt6 models / signals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SymbolEntry:
    """One tradable slug (``BTC-USDT_1h``)."""

    slug: str
    symbol: str
    timeframe: str
    parquet_ohlcv: Optional[Path] = None
    parquet_features: Optional[Path] = None
    config_yaml: Optional[Path] = None
    latest_bundle_run_id: Optional[str] = None
    has_bundle: bool = False
    has_ohlcv: bool = False
    has_features: bool = False
    has_symbol_config: bool = False
    last_acceptance_passed: Optional[bool] = None


@dataclass
class DataHealth:
    """OHLCV / features freshness for a symbol."""

    slug: str
    ohlcv_path: Optional[Path]
    features_path: Optional[Path]
    ohlcv_rows: int = 0
    features_rows: int = 0
    ohlcv_last_ts: Optional[str] = None
    features_last_ts: Optional[str] = None
    ohlcv_exists: bool = False
    features_exists: bool = False


@dataclass
class BundleInfo:
    """Artifact bundle metadata from ``manifest.json``."""

    bundle_dir: Path
    run_id: str
    created_at: str
    feature_columns: List[str]
    feature_schema_hash: str
    train_meta_threshold: Optional[float]
    model_keys: List[str]
    orchestrator_config: Dict[str, Any] = field(default_factory=dict)
    schema_valid: Optional[bool] = None


@dataclass
class ExplainSnapshot:
    """Last-bar decision card (``orchestration.introspect.explain_symbol``)."""

    symbol: str
    timeframe: str
    bundle_dir: str
    as_of: str
    last_close: Optional[float]
    regime: str
    regime_int: int
    active_weights: Dict[str, float]
    model_probs: Dict[str, float]
    meta_probability: float
    confidence: float
    direction: str
    signal: int
    position_size_frac: float
    why_blocked: Optional[str]
    active_models: List[str]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeBar:
    """One point for regime overlay on chart."""

    t: str
    regime_int: int
    regime: str
    close: Optional[float]


@dataclass
class ChartBar:
    """OHLCV + optional overlays for candlestick view."""

    t: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    signal: Optional[int] = None
    regime_int: Optional[int] = None
    meta_probability: Optional[float] = None
    ts_ms: Optional[int] = None  # open time UTC ms (для live merge)


@dataclass
class BacktestRunSummary:
    """Row from ``docs/backtest_journal/runs.jsonl``."""

    run_id: str
    timestamp_utc: str
    label: str
    symbol: Optional[str]
    timeframe: Optional[str]
    stage: Optional[str]
    acceptance_passed: Optional[bool]
    target_passed: Optional[bool]
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobEvent:
    """Line from ``artifacts/<slug>/active/<task_id>.jsonl``."""

    task_id: str
    line_no: int
    raw: Dict[str, Any]


@dataclass
class PipelineStepStatus:
    """High-level prepare-symbol stages for Jobs panel."""

    name: str
    done: bool
    detail: str = ""


@dataclass
class FoldEquityPoint:
    """Cumulative OOS return built from per-fold ``Total Return (%)``."""

    fold: int
    oos_return_pct: float
    cumulative_pct: float


@dataclass
class ChartPayload:
    """Bars + aligned regime for chart view."""

    bars: List[ChartBar]
    regime_segments: List[tuple]  # (x0, x1, regime_int)


@dataclass
class OrderRecord:
    """Serializable order for GUI tables."""

    order_id: str
    symbol: str
    side: str
    status: str
    quantity: float
    filled_price: Optional[float]
    fees: float
    timestamp: Optional[float]
    error_message: Optional[str] = None


@dataclass
class PositionRecord:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class ExecutionAccountSnapshot:
    mode: str
    connected: bool
    balance: float
    available_balance: float
    initial_balance: float
    positions: List[PositionRecord]
    total_fees: float
    fill_rate: float


@dataclass
class ExecutionStepResult:
    """One paper/live inference + execution step."""

    symbol: str
    timeframe: str
    as_of: str
    price: float
    signal: int
    direction: str
    meta_probability: float
    position_fraction: float
    quantity: float
    order: Optional[OrderRecord]
    message: str
