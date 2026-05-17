"""Paper / live execution sessions for GUI."""

from __future__ import annotations

from typing import List, Optional

from execution.brokers.base_broker import OrderSide, OrderStatus
from execution.execution_manager import ExecutionConfig, ExecutionManager
from orchestration.symbols import paths_for

from gui.api.types import (
    ExecutionAccountSnapshot,
    ExecutionStepResult,
    OrderRecord,
    PositionRecord,
)


def broker_symbol(symbol: str) -> str:
    """``BTC-USDT`` / ``BTC/USDT`` → ``BTC/USDT`` for brokers."""
    s = symbol.strip().upper().replace("_", "-")
    if "/" in s:
        return s
    if "-" in s:
        return s.replace("-", "/", 1)
    return s


class PaperExecutionSession:
    """In-memory paper trading session (one per GUI connect)."""

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        max_position_fraction: float = 0.35,
    ):
        self.config = ExecutionConfig(
            mode="paper",
            initial_balance=initial_balance,
            commission_rate=0.0006,
            slippage_rate=0.0002,
            max_order_size=None,
        )
        self.max_position_fraction = max_position_fraction
        self.manager = ExecutionManager(self.config)
        self.connected = False
        self._step_log: List[ExecutionStepResult] = []

    def connect(self) -> bool:
        self.connected = self.manager.connect()
        return self.connected

    def disconnect(self) -> None:
        self.manager.disconnect()
        self.connected = False

    def reset(self) -> None:
        if hasattr(self.manager.broker, "reset"):
            self.manager.broker.reset()
        self._step_log.clear()

    def account_snapshot(self) -> ExecutionAccountSnapshot:
        stats = self.manager.get_execution_statistics()
        acct = self.manager.get_account_info()
        positions = [
            PositionRecord(
                symbol=p.symbol,
                quantity=float(p.quantity),
                entry_price=float(p.entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pnl),
            )
            for p in acct.positions.values()
        ]
        return ExecutionAccountSnapshot(
            mode="paper",
            connected=self.connected,
            balance=float(acct.balance),
            available_balance=float(acct.available_balance),
            initial_balance=float(self.config.initial_balance),
            positions=positions,
            total_fees=float(stats["order_statistics"].get("total_fees", 0)),
            fill_rate=float(stats["order_statistics"].get("fill_rate", 0)),
        )

    def order_history(self, limit: int = 50) -> List[OrderRecord]:
        orders = self.manager.order_manager.get_order_history(limit=limit)
        return [_order_to_record(o) for o in reversed(orders)]

    def run_step(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        window: int = 256,
    ) -> ExecutionStepResult:
        if not self.connected:
            raise RuntimeError("Paper session not connected")

        sp = paths_for(symbol, timeframe)
        bundle = sp.latest_bundle()
        if bundle is None:
            raise FileNotFoundError(f"No bundle for {sp.slug}")

        from orchestration.symbol_pipeline import build_features

        feat = build_features(sp.parquet)
        win = feat.iloc[-window:].copy()
        price = float(win["close"].iloc[-1])
        as_of = str(win.index[-1])
        bsym = broker_symbol(sp.symbol)

        from gui.api import IstGuiClient

        explain = IstGuiClient().inference.explain(symbol, timeframe, window=window)
        signal = int(explain.signal)
        frac = float(explain.position_size_frac)
        if signal != 0 and frac <= 0:
            frac = min(self.max_position_fraction, 0.1)

        acct = self.manager.get_account_info()
        pos = self.manager.get_position(bsym)
        qty = 0.0
        message = ""

        if signal == 0:
            message = explain.why_blocked or "Signal flat — no order"
            result = ExecutionStepResult(
                symbol=bsym,
                timeframe=timeframe,
                as_of=as_of,
                price=price,
                signal=0,
                direction="flat",
                meta_probability=float(explain.meta_probability),
                position_fraction=frac,
                quantity=0.0,
                order=None,
                message=message,
            )
            self._step_log.append(result)
            return result

        if signal == 1:
            notional = acct.balance * min(frac, self.max_position_fraction)
            qty = notional / price if price > 0 else 0.0
            if qty <= 0:
                message = "Quantity zero (balance/price)"
        elif signal == -1:
            if pos and pos.quantity > 0:
                notional = acct.balance * min(frac, self.max_position_fraction)
                qty = min(float(pos.quantity), notional / price if price > 0 else 0.0)
                message = "Closing/reducing long (paper has no short selling)"
            else:
                message = "Short not supported in paper mode without position"
                result = ExecutionStepResult(
                    symbol=bsym,
                    timeframe=timeframe,
                    as_of=as_of,
                    price=price,
                    signal=signal,
                    direction="short",
                    meta_probability=float(explain.meta_probability),
                    position_fraction=frac,
                    quantity=0.0,
                    order=None,
                    message=message,
                )
                self._step_log.append(result)
                return result

        order = None
        if qty > 0:
            order_obj = self.manager.execute_signal(
                signal=signal,
                symbol=bsym,
                position_size=qty,
                current_price=price,
                risk_multiplier=1.0,
            )
            if order_obj is not None:
                order = _order_to_record(order_obj)
                message = f"Order {order.status}: {order.side} {order.quantity:.6f}"
            else:
                message = "execute_signal returned None"
        else:
            message = message or "No quantity to trade"

        result = ExecutionStepResult(
            symbol=bsym,
            timeframe=timeframe,
            as_of=as_of,
            price=price,
            signal=signal,
            direction=explain.direction,
            meta_probability=float(explain.meta_probability),
            position_fraction=frac,
            quantity=qty,
            order=order,
            message=message,
        )
        self._step_log.append(result)
        return result

    def emergency_stop(self) -> int:
        return self.manager.emergency_stop()

    def step_log(self) -> List[ExecutionStepResult]:
        return list(self._step_log)


class ExecutionApi:
    """Factory for execution sessions."""

    def create_paper_session(
        self,
        *,
        initial_balance: float = 10_000.0,
        max_position_fraction: float = 0.35,
    ) -> PaperExecutionSession:
        return PaperExecutionSession(
            initial_balance=initial_balance,
            max_position_fraction=max_position_fraction,
        )

    def live_available(self) -> bool:
        import os

        return bool(
            os.environ.get("OKX_API_KEY")
            and os.environ.get("OKX_SECRET_KEY")
            and os.environ.get("OKX_PASSPHRASE")
        )


def _order_to_record(order) -> OrderRecord:
    return OrderRecord(
        order_id=str(order.order_id),
        symbol=str(order.symbol),
        side=order.side.value if isinstance(order.side, OrderSide) else str(order.side),
        status=order.status.value if isinstance(order.status, OrderStatus) else str(order.status),
        quantity=float(order.quantity),
        filled_price=float(order.filled_price) if order.filled_price is not None else None,
        fees=float(order.fees or 0),
        timestamp=float(order.timestamp) if order.timestamp else None,
        error_message=order.error_message,
    )
