"""
Paper replay, signal timeline и калибровка прогноза для GUI (без переобучения).

Использует замороженный artifact bundle: на каждом баре только inference + paper broker
или векторный Backtester с теми же комиссиями.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester
from execution.execution_manager import ExecutionConfig, ExecutionManager
from execution.brokers.base_broker import OrderStatus
from execution.brokers.paper_broker import PaperBroker
from .artifact_bundle import validate_bundle_feature_schema
from .glue import inference_stack_from_bundle
from .introspect import _resolve_bundle, _ensure_features
from .inference_orchestrator import InferenceOrchestrator
from .symbols import paths_for


def _broker_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("_", "-")
    if "/" in s:
        return s
    if "-" in s:
        return s.replace("-", "/", 1)
    return s


def _portfolio_equity(broker: PaperBroker, symbol: str, price: float) -> float:
    """Cash + mark-to-market позиций (для кривой equity, не только balance)."""
    broker._update_positions_pnl()
    acct = broker.get_account_info()
    equity = float(acct.balance)
    pos = acct.positions.get(symbol)
    if pos is not None and float(pos.quantity) > 0:
        equity += float(pos.quantity) * float(price)
    return equity


@dataclass
class BarDecision:
    as_of: str
    close: float
    meta_probability: float
    signal: int
    position_fraction: float
    regime: str
    why_blocked: Optional[str] = None


@dataclass
class ReplayTrade:
    as_of: str
    side: str
    quantity: float
    price: float
    fees: float
    balance_after: float


@dataclass
class PaperReplayResult:
    bundle_dir: str
    n_bars: int
    n_trades: int
    n_nonzero_signals: int
    initial_balance: float
    final_balance_paper: float
    total_return_paper_pct: float
    total_return_bt_pct: float
    equity_paper: List[float]
    equity_backtester: List[float]
    timestamps: List[str]
    trades: List[ReplayTrade] = field(default_factory=list)
    decisions: List[BarDecision] = field(default_factory=list)


@dataclass
class CalibrationBucket:
    label: str
    p_mid: float
    count: int
    mean_forward_net_pct: float
    mean_forward_market_pct: float


@dataclass
class CalibrationResult:
    n_bars: int
    n_buckets: int
    buckets: List[CalibrationBucket]
    bundle_dir: str


def _why_blocked(sig: int, p: float, cfg: Any) -> Optional[str]:
    margin = float(getattr(cfg, "min_signal_margin", 0.0) or 0.0)
    thr = float(getattr(cfg, "direction_threshold", 0.5))
    if sig != 0:
        return None
    if abs(p - 0.5) < margin:
        return "dead-zone (min_signal_margin)"
    if abs(p - 0.5) < (thr - 0.5):
        return "below direction_threshold"
    return "pipeline / trade_mode"


def _load_stack(symbol: str, timeframe: str) -> Tuple[Any, ...]:
    bundle_dir = _resolve_bundle(symbol, timeframe)
    stack = inference_stack_from_bundle(bundle_dir)
    feat = _ensure_features(symbol, timeframe, None)
    cols = list(stack["feature_columns"])
    if not validate_bundle_feature_schema(bundle_dir, cols):
        missing = [c for c in cols if c not in feat.columns]
        if missing:
            raise ValueError(f"Features missing columns required by bundle: {missing}")
    orch: InferenceOrchestrator = stack["orchestrator"]
    orch.initialize(
        models=stack["models"],
        regime_detector=stack["regime_detector"],
        meta_weighting=stack["meta_weighting"],
    )
    cfg = stack["config"]
    return bundle_dir, stack, feat, cols, orch, cfg


def _predict_bar(
    orch: InferenceOrchestrator,
    feat: pd.DataFrame,
    cols: List[str],
    bar_index: int,
    window: int,
    cfg: Any,
) -> BarDecision:
    w0 = max(0, bar_index - int(window) + 1)
    extra = ["close"] if "close" in feat.columns else []
    win = feat.iloc[w0 : bar_index + 1][cols + extra].copy()
    if len(win) < 2:
        close = float(feat["close"].iloc[bar_index]) if "close" in feat.columns else 0.0
        return BarDecision(
            as_of=str(feat.index[bar_index]),
            close=close,
            meta_probability=0.5,
            signal=0,
            position_fraction=0.0,
            regime="range",
            why_blocked="insufficient window",
        )
    result = orch.predict(win)
    sig = int(result.signal)
    p = float(result.meta_probability)
    regime = result.regime
    if isinstance(regime, dict):
        regime = str(regime.get("market_regime", "range"))
    else:
        regime = str(regime)
    close = float(win["close"].iloc[-1]) if "close" in win.columns else float(feat["close"].iloc[bar_index])
    return BarDecision(
        as_of=str(feat.index[bar_index]),
        close=close,
        meta_probability=p,
        signal=sig,
        position_fraction=float(result.position_size),
        regime=regime,
        why_blocked=_why_blocked(sig, p, cfg),
    )


def collect_bar_decisions(
    symbol: str,
    timeframe: str = "1h",
    *,
    n_bars: int = 300,
    window: int = 256,
) -> Tuple[str, List[BarDecision]]:
    """Inference-only timeline on the last ``n_bars`` bars (frozen bundle)."""
    bundle_dir, _stack, feat, cols, orch, cfg = _load_stack(symbol, timeframe)
    n = min(int(n_bars), len(feat))
    if n < 2:
        raise ValueError("Need at least 2 bars for paper evidence")
    i0 = len(feat) - n
    decisions: List[BarDecision] = []
    for i in range(i0, len(feat)):
        decisions.append(_predict_bar(orch, feat, cols, i, window, cfg))
    return str(bundle_dir), decisions


def run_paper_replay(
    symbol: str,
    timeframe: str = "1h",
    *,
    n_bars: int = 300,
    window: int = 256,
    initial_balance: float = 10_000.0,
    max_position_fraction: float = 0.35,
    commission_rate: float = 0.0006,
    slippage_rate: float = 0.0002,
) -> PaperReplayResult:
    bundle_dir, decisions = collect_bar_decisions(
        symbol, timeframe, n_bars=n_bars, window=window
    )
    sp = paths_for(symbol, timeframe)
    bsym = _broker_symbol(sp.symbol)

    exec_cfg = ExecutionConfig(
        mode="paper",
        initial_balance=float(initial_balance),
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    em = ExecutionManager(exec_cfg)
    if not em.connect():
        raise RuntimeError("Paper broker connect failed")

    pb: PaperBroker = em.broker  # type: ignore[assignment]
    equity_paper: List[float] = [float(initial_balance)]
    trades: List[ReplayTrade] = []
    n_trades = 0
    min_trade_qty = 1e-8

    for dec in decisions:
        price = dec.close
        if price <= 0:
            equity_paper.append(_portfolio_equity(pb, bsym, price))
            continue

        pb.update_price(bsym, price)
        equity = _portfolio_equity(pb, bsym, price)
        pos = pb.get_position(bsym)
        held_qty = float(pos.quantity) if pos else 0.0
        held_value = held_qty * price

        sig = int(dec.signal)
        frac = float(dec.position_fraction)
        if sig != 0 and frac <= 0:
            frac = min(max_position_fraction, 0.1)

        # Целевая доля капитала в активе (long-only; short → закрыть)
        if sig > 0:
            target_value = equity * min(frac, max_position_fraction)
        elif sig < 0:
            target_value = 0.0
        else:
            # signal=0: удерживаем позицию (как exposure до смены сигнала в shift(1)-бэктесте)
            target_value = held_value

        delta_value = target_value - held_value
        order = None

        if delta_value > price * min_trade_qty:
            qty = delta_value / price
            order = em.execute_signal(
                signal=1,
                symbol=bsym,
                position_size=qty,
                current_price=price,
                risk_multiplier=1.0,
            )
        elif delta_value < -price * min_trade_qty and held_qty > 0:
            qty = min(held_qty, (-delta_value) / price)
            order = em.execute_signal(
                signal=-1,
                symbol=bsym,
                position_size=qty,
                current_price=price,
                risk_multiplier=1.0,
            )

        if order is not None and order.status == OrderStatus.FILLED:
            n_trades += 1
            trades.append(
                ReplayTrade(
                    as_of=dec.as_of,
                    side=order.side.value,
                    quantity=float(order.filled_quantity),
                    price=float(order.filled_price or price),
                    fees=float(order.fees),
                    balance_after=_portfolio_equity(pb, bsym, price),
                )
            )

        equity_paper.append(_portfolio_equity(pb, bsym, price))

    em.disconnect()

    # Backtester on the same window (canonical shift(1))
    feat = _ensure_features(symbol, timeframe, None)
    n = len(decisions)
    i0 = len(feat) - n
    sl = feat.iloc[i0:].copy()
    sig_s = pd.Series(0.0, index=sl.index)
    pos_s = pd.Series(0.0, index=sl.index)
    for j, dec in enumerate(decisions):
        sig_s.iloc[j] = float(dec.signal)
        pos_s.iloc[j] = float(dec.position_fraction)

    bt = Backtester(commission=commission_rate, slippage=slippage_rate)
    bt_res = bt.run(sl, sig_s, pos_s)
    cum = bt_res["cum_strategy_returns"].astype(float)
    equity_bt = (cum * float(initial_balance)).tolist()

    # align lengths
    if len(equity_bt) != len(equity_paper):
        equity_bt = equity_bt[-len(equity_paper) :]

    final_paper = equity_paper[-1]
    final_bt = equity_bt[-1] if equity_bt else float(initial_balance)
    ret_p = 100.0 * (final_paper / float(initial_balance) - 1.0)
    ret_b = 100.0 * (final_bt / float(initial_balance) - 1.0)
    n_nz = sum(1 for d in decisions if d.signal != 0)

    return PaperReplayResult(
        bundle_dir=bundle_dir,
        n_bars=n,
        n_trades=n_trades,
        n_nonzero_signals=n_nz,
        initial_balance=float(initial_balance),
        final_balance_paper=final_paper,
        total_return_paper_pct=ret_p,
        total_return_bt_pct=ret_b,
        equity_paper=equity_paper,
        equity_backtester=equity_bt,
        timestamps=[d.as_of for d in decisions],
        trades=trades,
        decisions=decisions,
    )


def build_calibration(
    symbol: str,
    timeframe: str = "1h",
    *,
    n_bars: int = 500,
    window: int = 256,
    n_buckets: int = 8,
    commission_rate: float = 0.0006,
    slippage_rate: float = 0.0002,
) -> CalibrationResult:
    bundle_dir, decisions = collect_bar_decisions(
        symbol, timeframe, n_bars=n_bars, window=window
    )
    if len(decisions) < 3:
        raise ValueError("Need at least 3 bars for calibration")

    feat = _ensure_features(symbol, timeframe, None)
    n = len(decisions)
    i0 = len(feat) - n
    sl = feat.iloc[i0:].copy()
    sig_s = pd.Series(0.0, index=sl.index)
    pos_s = pd.Series(0.0, index=sl.index)
    p_s = pd.Series(0.5, index=sl.index)
    for j, dec in enumerate(decisions):
        sig_s.iloc[j] = float(dec.signal)
        pos_s.iloc[j] = float(dec.position_fraction)
        p_s.iloc[j] = float(dec.meta_probability)

    bt = Backtester(commission=commission_rate, slippage=slippage_rate)
    bt_res = bt.run(sl, sig_s, pos_s)
    # net_returns[i] — доходность на баре i от сигнала на i-1
    net = bt_res["net_returns"].astype(float)
    mkt = bt_res["market_returns"].astype(float)

    # пара (p[i-1], net[i]) для i >= 1
    ps: List[float] = []
    nets: List[float] = []
    mkts: List[float] = []
    for i in range(1, len(decisions)):
        ps.append(float(p_s.iloc[i - 1]))
        nets.append(float(net.iloc[i]))
        mkts.append(float(mkt.iloc[i]) if not np.isnan(mkt.iloc[i]) else 0.0)

    if not ps:
        raise ValueError("No calibration pairs")

    arr_p = np.array(ps, dtype=float)
    arr_n = np.array(nets, dtype=float)
    arr_m = np.array(mkts, dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_buckets) + 1)
    buckets: List[CalibrationBucket] = []
    for b in range(int(n_buckets)):
        lo, hi = edges[b], edges[b + 1]
        if b < int(n_buckets) - 1:
            mask = (arr_p >= lo) & (arr_p < hi)
        else:
            mask = (arr_p >= lo) & (arr_p <= hi)
        cnt = int(mask.sum())
        mid = (lo + hi) / 2.0
        label = f"{lo:.2f}–{hi:.2f}"
        if cnt == 0:
            buckets.append(
                CalibrationBucket(
                    label=label,
                    p_mid=mid,
                    count=0,
                    mean_forward_net_pct=0.0,
                    mean_forward_market_pct=0.0,
                )
            )
        else:
            buckets.append(
                CalibrationBucket(
                    label=label,
                    p_mid=mid,
                    count=cnt,
                    mean_forward_net_pct=float(arr_n[mask].mean() * 100.0),
                    mean_forward_market_pct=float(arr_m[mask].mean() * 100.0),
                )
            )

    return CalibrationResult(
        n_bars=n,
        n_buckets=int(n_buckets),
        buckets=buckets,
        bundle_dir=bundle_dir,
    )
