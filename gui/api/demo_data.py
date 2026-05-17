"""Синтетические данные для демо-режима GUI (без parquet / bundle / ML)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gui.api.types import (
    BacktestRunSummary,
    BundleInfo,
    ChartBar,
    ChartPayload,
    DataHealth,
    ExecutionAccountSnapshot,
    ExecutionStepResult,
    ExplainSnapshot,
    FoldEquityPoint,
    JobEvent,
    OrderRecord,
    PipelineStepStatus,
    PositionRecord,
    SymbolEntry,
)

_DEMO_ROOT = Path("artifacts/demo/BTC-USDT_1h/20260517T120000Z_demo01")
_DEMO_RUN_IDS = [
    "20260517T100001Z_demo_wfo",
    "20260517T100002Z_demo_tune",
    "20260516T180000Z_demo_fail",
    "20260515T120000Z_demo_old",
    "20260514T090000Z_demo_smoke",
]

_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "atr_14",
    "sma_20",
    "sma_50",
    "regime_vol",
]


def demo_symbols() -> List[SymbolEntry]:
    return [
        SymbolEntry(
            slug="BTC-USDT_1h",
            symbol="BTC/USDT",
            timeframe="1h",
            parquet_ohlcv=Path("data/ohlcv/BTC-USDT_1h.parquet"),
            parquet_features=Path("data/features/BTC-USDT_1h.parquet"),
            config_yaml=Path("config/symbols/BTC-USDT_1h.yaml"),
            latest_bundle_run_id="20260517T120000Z_demo01",
            has_bundle=True,
        ),
        SymbolEntry(
            slug="ETH-USDT_1h",
            symbol="ETH/USDT",
            timeframe="1h",
            parquet_ohlcv=Path("data/ohlcv/ETH-USDT_1h.parquet"),
            parquet_features=Path("data/features/ETH-USDT_1h.parquet"),
            config_yaml=Path("config/symbols/ETH-USDT_1h.yaml"),
            latest_bundle_run_id="20260517T120001Z_demo02",
            has_bundle=True,
        ),
        SymbolEntry(
            slug="BTC-USDT_4h",
            symbol="BTC/USDT",
            timeframe="4h",
            parquet_ohlcv=Path("data/ohlcv/BTC-USDT_4h.parquet"),
            parquet_features=Path("data/features/BTC-USDT_4h.parquet"),
            config_yaml=Path("config/symbols/BTC-USDT_4h.yaml"),
            latest_bundle_run_id="20260517T120002Z_demo03",
            has_bundle=True,
        ),
    ]


def demo_explain(symbol: str, timeframe: str) -> ExplainSnapshot:
    base = 68_420.50 if "BTC" in symbol.upper() else 3_512.80
    return ExplainSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        bundle_dir=str(_DEMO_ROOT),
        as_of="2026-05-17 14:00:00+00:00",
        last_close=base,
        regime="trend",
        regime_int=1,
        active_weights={"lgb": 0.35, "gru": 0.25, "xgb": 0.22, "cnn": 0.18},
        model_probs={"lgb": 0.58, "gru": 0.61, "xgb": 0.55, "cnn": 0.57},
        meta_probability=0.578,
        confidence=0.578,
        direction="long",
        signal=1,
        position_size_frac=0.22,
        why_blocked=None,
        active_models=["lgb", "gru", "xgb", "cnn"],
        config={
            "direction_threshold": 0.52,
            "min_signal_margin": 0.02,
            "trade_mode": "both",
            "ensemble_mode": "regime_weighted",
        },
    )


def _demo_chart_bars(n: int = 480, base_price: float = 64_000.0) -> List[ChartBar]:
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    bars: List[ChartBar] = []
    price = base_price
    for i in range(n):
        ts = t0 + timedelta(hours=i)
        drift = 120 * math.sin(i / 36) + 40 * math.sin(i / 11)
        noise = 80 * math.sin(i * 0.73 + 1.2)
        close = base_price + drift + noise + i * 2.5
        open_ = price
        high = max(open_, close) + abs(30 * math.sin(i / 7))
        low = min(open_, close) - abs(25 * math.cos(i / 9))
        regime_int = 1 if (i // 72) % 2 == 0 else 0
        meta = 0.5 + 0.12 * math.sin(i / 24)
        sig: Optional[int] = None
        if meta > 0.56 and i % 17 == 0:
            sig = 1
        elif meta < 0.44 and i % 23 == 0:
            sig = -1
        bars.append(
            ChartBar(
                t=str(ts),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(800 + 400 * abs(math.sin(i / 15))),
                signal=sig,
                regime_int=regime_int,
                meta_probability=float(meta),
            )
        )
        price = close
    return bars


def demo_chart_payload(
    symbol: str,
    timeframe: str,
    *,
    max_bars: int = 1200,
) -> ChartPayload:
    base = 64_000.0 if "BTC" in symbol.upper() else 3_400.0
    bars = _demo_chart_bars(min(max_bars, 480), base_price=base)
    from gui.api.inference_api import InferenceApi

    segments = InferenceApi.regime_segments_from_bars(bars)
    return ChartPayload(bars=bars, regime_segments=segments)


def demo_bundle_info(symbol: str, timeframe: str) -> BundleInfo:
    return BundleInfo(
        bundle_dir=_DEMO_ROOT,
        run_id="20260517T120000Z_demo01",
        created_at="2026-05-17T12:00:00Z",
        feature_columns=list(_FEATURE_COLS),
        feature_schema_hash="demo_schema_a1b2c3",
        train_meta_threshold=0.52,
        model_keys=["lgb", "gru", "xgb", "cnn"],
        orchestrator_config={
            "ensemble_mode": "regime_weighted",
            "trend_weights": {"lgb": 0.30, "gru": 0.28, "xgb": 0.22, "cnn": 0.20},
            "range_weights": {"lgb": 0.40, "gru": 0.15, "xgb": 0.30, "cnn": 0.15},
            "model_keys": ["lgb", "gru", "xgb", "cnn"],
        },
        schema_valid=True,
    )


def demo_data_health(symbol: str, timeframe: str) -> DataHealth:
    slug = f"{symbol.replace('/', '-').replace('_', '-')}_{timeframe}"
    return DataHealth(
        slug=slug,
        ohlcv_path=Path(f"data/ohlcv/{slug}.parquet"),
        features_path=Path(f"data/features/{slug}.parquet"),
        ohlcv_rows=18_432,
        features_rows=18_408,
        ohlcv_last_ts="2026-05-17 14:00:00+00:00",
        features_last_ts="2026-05-17 14:00:00+00:00",
        ohlcv_exists=True,
        features_exists=True,
    )


def demo_pipeline_status(symbol: str, timeframe: str) -> List[PipelineStepStatus]:
    sp = demo_data_health(symbol, timeframe)
    return [
        PipelineStepStatus("ohlcv", True, str(sp.ohlcv_path)),
        PipelineStepStatus("features", True, str(sp.features_path)),
        PipelineStepStatus("symbol_config", True, f"config/symbols/{sp.slug}.yaml"),
        PipelineStepStatus("artifact_bundle", True, "20260517T120000Z_demo01"),
    ]


def demo_task_events(task_id: str) -> List[JobEvent]:
    lines = [
        {"ts": "2026-05-17T12:00:01Z", "stage": "download", "message": "OHLCV OK (18432 bars)"},
        {"ts": "2026-05-17T12:05:12Z", "stage": "features", "message": "Features built"},
        {"ts": "2026-05-17T12:18:44Z", "stage": "tune", "message": "WFO trial 12/30 — Sharpe 1.24"},
        {"ts": "2026-05-17T12:42:00Z", "stage": "train-final", "message": "Bundle saved"},
        {"ts": "2026-05-17T12:42:01Z", "stage": "done", "message": "prepare-symbol completed"},
    ]
    return [JobEvent(task_id=task_id, line_no=i + 1, raw=raw) for i, raw in enumerate(lines)]


def demo_merged_config(symbol: str, timeframe: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "demo_mode": True,
        "orchestration": {
            "direction_threshold": 0.52,
            "min_signal_margin": 0.02,
            "trade_mode": "both",
            "ensemble_mode": "regime_weighted",
            "trend_weights": {"lgb": 0.30, "gru": 0.28, "xgb": 0.22, "cnn": 0.20},
            "range_weights": {"lgb": 0.40, "gru": 0.15, "xgb": 0.30, "cnn": 0.15},
        },
        "risk": {
            "max_position_fraction": 0.35,
            "atr_stop_mult": 2.0,
            "trailing_stop_pct": 0.04,
        },
        "gui": {"refresh_interval_sec": 30, "chart_max_bars": 2000},
    }


def demo_backtest_runs(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> List[BacktestRunSummary]:
    norm = (symbol or "BTC-USDT").replace("/", "-")
    tf = timeframe or "1h"
    rows = [
        BacktestRunSummary(
            run_id=_DEMO_RUN_IDS[0],
            timestamp_utc="2026-05-17T10:00:01Z",
            label="canonical_4model",
            symbol=norm,
            timeframe=tf,
            stage="wfo_integrated",
            acceptance_passed=True,
            target_passed=True,
            summary={"mean_sharpe": 1.18, "mean_return_pct": 4.2, "max_dd_pct": -6.1},
        ),
        BacktestRunSummary(
            run_id=_DEMO_RUN_IDS[1],
            timestamp_utc="2026-05-17T08:30:00Z",
            label="tune_lgb_xgb",
            symbol=norm,
            timeframe=tf,
            stage="orchestration_tune",
            acceptance_passed=True,
            target_passed=False,
            summary={"mean_sharpe": 0.92, "mean_return_pct": 2.1, "max_dd_pct": -8.4},
        ),
        BacktestRunSummary(
            run_id=_DEMO_RUN_IDS[2],
            timestamp_utc="2026-05-16T18:00:00Z",
            label="smoke_fail",
            symbol=norm,
            timeframe=tf,
            stage="wfo_integrated",
            acceptance_passed=False,
            target_passed=False,
            summary={"mean_sharpe": 0.31, "mean_return_pct": -1.2, "max_dd_pct": -12.0},
        ),
        BacktestRunSummary(
            run_id=_DEMO_RUN_IDS[3],
            timestamp_utc="2026-05-15T12:00:00Z",
            label="baseline",
            symbol=norm,
            timeframe=tf,
            stage="report_real",
            acceptance_passed=True,
            target_passed=True,
            summary={"mean_sharpe": 0.85, "mean_return_pct": 3.0, "max_dd_pct": -7.2},
        ),
        BacktestRunSummary(
            run_id=_DEMO_RUN_IDS[4],
            timestamp_utc="2026-05-14T09:00:00Z",
            label="quick_smoke",
            symbol=norm,
            timeframe=tf,
            stage="smoke",
            acceptance_passed=True,
            target_passed=True,
            summary={"mean_sharpe": 1.05, "mean_return_pct": 1.8, "max_dd_pct": -4.5},
        ),
    ]
    return rows


def demo_run_detail(run_id: str) -> Dict[str, Any]:
    folds = [
        {"Fold": 1, "Total Return (%)": 1.2, "Sharpe": 1.1},
        {"Fold": 2, "Total Return (%)": 0.8, "Sharpe": 0.9},
        {"Fold": 3, "Total Return (%)": -0.4, "Sharpe": 0.2},
        {"Fold": 4, "Total Return (%)": 1.5, "Sharpe": 1.3},
        {"Fold": 5, "Total Return (%)": 0.9, "Sharpe": 1.0},
    ]
    passed = run_id != _DEMO_RUN_IDS[2]
    return {
        "run_id": run_id,
        "demo": True,
        "label": "demo_wfo",
        "symbol": "BTC-USDT",
        "timeframe": "1h",
        "acceptance_passed": passed,
        "target_passed": passed,
        "fold_metrics": folds,
        "criteria": {"min_sharpe": 0.5, "max_drawdown_pct": 15.0},
        "summary": {"mean_sharpe": 1.18 if passed else 0.31},
    }


def demo_fold_equity(run: Dict[str, Any]) -> List[FoldEquityPoint]:
    folds = run.get("fold_metrics") or []
    cum = 1.0
    out: List[FoldEquityPoint] = []
    for i, fold in enumerate(folds):
        if not isinstance(fold, dict):
            continue
        ret = float(fold.get("Total Return (%)", 0.0) or 0.0)
        cum *= 1.0 + ret / 100.0
        out.append(
            FoldEquityPoint(
                fold=int(fold.get("Fold", i + 1)),
                oos_return_pct=ret,
                cumulative_pct=(cum - 1.0) * 100.0,
            )
        )
    return out


class DemoPaperSession:
    """In-memory paper session для демо-вкладки «Исполнение»."""

    def __init__(self, *, initial_balance: float = 10_000.0):
        self.config = type("Cfg", (), {"initial_balance": initial_balance, "mode": "paper"})()
        self.max_position_fraction = 0.35
        self.connected = False
        self._balance = initial_balance
        self._initial = initial_balance
        self._fees = 12.45
        self._step_n = 0
        self._position = PositionRecord(
            symbol="BTC/USDT",
            quantity=0.012,
            entry_price=67_800.0,
            current_price=68_420.50,
            unrealized_pnl=7.45,
        )
        self._orders: List[OrderRecord] = [
            OrderRecord(
                order_id="demo_ord_001",
                symbol="BTC/USDT",
                side="buy",
                status="filled",
                quantity=0.012,
                filled_price=67_800.0,
                fees=4.07,
                timestamp=1715950800.0,
            ),
        ]

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def reset(self) -> None:
        self._balance = self._initial
        self._fees = 0.0
        self._orders.clear()
        self._position = PositionRecord(
            symbol="BTC/USDT", quantity=0.0, entry_price=0.0, current_price=68_420.50, unrealized_pnl=0.0
        )

    def account_snapshot(self) -> ExecutionAccountSnapshot:
        return ExecutionAccountSnapshot(
            mode="paper",
            connected=self.connected,
            balance=self._balance,
            available_balance=self._balance * 0.65,
            initial_balance=self._initial,
            positions=[self._position] if self._position.quantity > 0 else [],
            total_fees=self._fees,
            fill_rate=1.0,
        )

    def order_history(self, limit: int = 50) -> List[OrderRecord]:
        return list(reversed(self._orders[-limit:]))

    def run_step(self, symbol: str, timeframe: str, *, window: int = 256) -> ExecutionStepResult:
        self._step_n += 1
        snap = demo_explain(symbol, timeframe)
        qty = 0.0
        order: Optional[OrderRecord] = None
        msg = snap.why_blocked or "Демо: сигнал flat"
        if snap.signal == 1:
            qty = 0.008
            order = OrderRecord(
                order_id=f"demo_ord_{self._step_n:03d}",
                symbol=symbol.replace("-", "/") if "/" not in symbol else symbol,
                side="buy",
                status="filled",
                quantity=qty,
                filled_price=snap.last_close or 68_420.0,
                fees=0.41,
                timestamp=1715954400.0 + self._step_n,
            )
            self._orders.append(order)
            self._fees += order.fees
            msg = f"Демо: покупка {qty:.6f} @ {order.filled_price:.2f}"
        return ExecutionStepResult(
            symbol="BTC/USDT",
            timeframe=timeframe,
            as_of=snap.as_of,
            price=snap.last_close or 0.0,
            signal=snap.signal,
            direction=snap.direction,
            meta_probability=snap.meta_probability,
            position_fraction=snap.position_size_frac,
            quantity=qty,
            order=order,
            message=msg,
        )

    def emergency_stop(self) -> int:
        n = len(self._orders)
        self._orders.clear()
        return n


def demo_paper_compare(step: ExecutionStepResult) -> Any:
    from gui.api.reconcile_api import PaperBacktestCompare

    return PaperBacktestCompare(
        as_of=step.as_of,
        close=step.price,
        paper_signal=step.signal,
        paper_fraction=step.position_fraction,
        paper_quantity=step.quantity,
        paper_fill_price=step.order.filled_price if step.order else step.price,
        paper_fees=step.order.fees if step.order else 0.0,
        paper_message=f"[ДЕМО] {step.message}",
        bt_signal_bar="2026-05-17 13:00:00+00:00",
        bt_position_fraction=step.position_fraction,
        bt_strategy_return_pct=0.18,
        bt_costs_pct=0.06,
        bt_net_return_pct=0.12,
        bt_trade_on_bar=0.22,
        bt_same_bar_net_return_pct=0.15,
        paper_next_bar_return_pct=0.21,
        bt_next_bar_net_return_pct=0.19,
        notes=["Демо-режим: числа иллюстративные, не из реального бэктеста."],
    )
