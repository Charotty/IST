"""Сверка paper-исполнения с ``Backtester`` на том же баре."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester
from orchestration.symbol_pipeline import build_features
from orchestration.symbols import paths_for

from gui.api.types import ExecutionStepResult


@dataclass
class PaperBacktestCompare:
    """Результат сравнения одного paper-шага с векторным бэктестом."""

    as_of: str
    close: float
    paper_signal: int
    paper_fraction: float
    paper_quantity: float
    paper_fill_price: Optional[float]
    paper_fees: float
    paper_message: str

    bt_signal_bar: str
    bt_position_fraction: float
    bt_strategy_return_pct: float
    bt_costs_pct: float
    bt_net_return_pct: float
    bt_trade_on_bar: float

    bt_same_bar_net_return_pct: float

    paper_next_bar_return_pct: Optional[float]
    bt_next_bar_net_return_pct: Optional[float]

    notes: List[str] = field(default_factory=list)

    def to_russian_text(self) -> str:
        lines = [
            "=== Сверка Paper vs Backtester ===",
            f"Бар: {self.as_of}  close={self.close:.2f}",
            "",
            "— Paper (исполнение на close бара) —",
            f"  Сигнал: {self.paper_signal}  доля: {self.paper_fraction:.4f}  qty: {self.paper_quantity:.6f}",
            f"  Цена: {self._fmt(self.paper_fill_price)}  комиссии: {self.paper_fees:.4f} USDT",
            f"  {self.paper_message}",
            "",
            "— Backtester (signal.shift(1): сигнал на пред. баре) —",
            f"  Сигнал на баре: {self.bt_signal_bar}",
            f"  Доля: {self.bt_position_fraction:.4f}",
            f"  На баре {self.as_of}: gross {self.bt_strategy_return_pct:+.4f}%  "
            f"издержки {self.bt_costs_pct:.4f}%  net {self.bt_net_return_pct:+.4f}%",
            f"  Изм. экспозиции: {self.bt_trade_on_bar:.4f}",
            "",
            "— Если сигнал на том же баре (не канон Backtester) —",
            f"  net return: {self.bt_same_bar_net_return_pct:+.4f}%",
        ]
        if self.paper_next_bar_return_pct is not None:
            lines.extend(
                [
                    "",
                    "— Следующий бар —",
                    f"  Paper (оценка): {self.paper_next_bar_return_pct:+.4f}%",
                    f"  Backtester net: {self._fmt_pct(self.bt_next_bar_net_return_pct)}",
                ]
            )
        if self.notes:
            lines.append("")
            lines.append("Примечания:")
            for n in self.notes:
                lines.append(f"  • {n}")
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "—"

    @staticmethod
    def _fmt_pct(v: Optional[float]) -> str:
        return f"{v:+.4f}%" if v is not None else "—"


class ReconcileApi:
    def compare_paper_step(
        self,
        step: ExecutionStepResult,
        symbol: str,
        timeframe: str = "1h",
        *,
        tail_bars: int = 8,
    ) -> PaperBacktestCompare:
        sp = paths_for(symbol, timeframe)
        feat = build_features(sp.parquet)
        ts = pd.Timestamp(step.as_of)
        if feat.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(feat.index.tz)
        elif feat.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)

        if ts not in feat.index:
            i = int(feat.index.get_indexer([ts], method="nearest")[0])
        else:
            loc = feat.index.get_loc(ts)
            if isinstance(loc, slice):
                i = loc.stop - 1
            elif isinstance(loc, np.ndarray):
                i = int(np.where(loc)[0][-1])
            else:
                i = int(loc)

        i0 = max(0, i - tail_bars + 1)
        sl = feat.iloc[i0 : i + 1].copy()
        if len(sl) < 2:
            raise ValueError("Нужно минимум 2 бара для сравнения с Backtester")

        frac = float(step.position_fraction)
        sig = float(step.signal)
        bt = Backtester(commission=0.0006, slippage=0.0002)

        signals = pd.Series(0.0, index=sl.index)
        pos = pd.Series(0.0, index=sl.index)
        signals.iloc[-2] = sig
        pos.iloc[-2] = frac
        res = bt.run(sl, signals, pos)
        last = res.iloc[-1]
        prev_ts = str(sl.index[-2])

        sig_same = pd.Series(0.0, index=sl.index)
        pos_same = pd.Series(0.0, index=sl.index)
        sig_same.iloc[-1] = sig
        pos_same.iloc[-1] = frac
        res_same = bt.run(sl, sig_same, pos_same)

        fill_price = (
            float(step.order.filled_price)
            if step.order and step.order.filled_price is not None
            else float(step.price)
        )
        paper_fees = float(step.order.fees) if step.order else 0.0

        notes = [
            "Backtester: сигнал со сдвигом 1 бар (доходность close[t-1]→close[t]).",
            "Paper: market на close[t] с комиссией и проскальзыванием.",
        ]
        if sig < 0:
            notes.append("Paper: short без открытой long-позиции не исполняется.")

        paper_next: Optional[float] = None
        bt_next: Optional[float] = None
        if i + 1 < len(feat) and step.quantity > 0 and fill_price > 0:
            next_close = float(feat["close"].iloc[i + 1])
            if sig > 0:
                pnl = step.quantity * (next_close - fill_price) - paper_fees
                paper_next = 100.0 * pnl / (step.quantity * fill_price)
            elif sig < 0:
                pnl = step.quantity * (fill_price - next_close) - paper_fees
                paper_next = 100.0 * pnl / (step.quantity * fill_price)

            ext = feat.iloc[i0 : i + 2]
            if len(ext) >= 2:
                sig_ext = pd.Series(0.0, index=ext.index)
                pos_ext = pd.Series(0.0, index=ext.index)
                sig_ext.iloc[-2] = sig
                pos_ext.iloc[-2] = frac
                r2 = bt.run(ext, sig_ext, pos_ext)
                bt_next = float(r2["net_returns"].iloc[-1] * 100)

        return PaperBacktestCompare(
            as_of=str(sl.index[-1]),
            close=float(step.price),
            paper_signal=int(step.signal),
            paper_fraction=frac,
            paper_quantity=float(step.quantity),
            paper_fill_price=fill_price if step.order else None,
            paper_fees=paper_fees,
            paper_message=step.message,
            bt_signal_bar=prev_ts,
            bt_position_fraction=frac,
            bt_strategy_return_pct=float(last["strategy_returns"] * 100),
            bt_costs_pct=float(last["costs"] * 100),
            bt_net_return_pct=float(last["net_returns"] * 100),
            bt_trade_on_bar=float(last["trades"]),
            bt_same_bar_net_return_pct=float(res_same["net_returns"].iloc[-1] * 100),
            paper_next_bar_return_pct=paper_next,
            bt_next_bar_net_return_pct=bt_next,
            notes=notes,
        )
