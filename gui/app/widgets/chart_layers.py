"""Overlays for pyqtgraph: regime bands, trade signals, meta probability."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from gui.api.types import ChartBar, ChartPayload

from gui.app.chart_bar_merge import LiveBarChange

_DEFAULT_VISIBLE_BARS = 150

try:
    import numpy as np
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor

    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_REGIME_BRUSH = {
    1: (46, 125, 50, 45),   # trend — green
    0: (158, 158, 158, 40),  # range — gray
}


def y_range_from_bars(bars: Sequence[ChartBar]) -> Tuple[float, float]:
    if not bars:
        return 0.0, 1.0
    lo = min(b.low for b in bars)
    hi = max(b.high for b in bars)
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def add_regime_background(
    plot: "pg.PlotWidget",
    segments: Sequence[Tuple[int, int, int]],
    y_min: float,
    y_max: float,
) -> List["pg.LinearRegionItem"]:
    """Paint contiguous regime segments as semi-transparent vertical bands."""
    if not _HAS_PG:
        return []
    items: List[pg.LinearRegionItem] = []
    for x0, x1, regime_int in segments:
        brush = _REGIME_BRUSH.get(int(regime_int), _REGIME_BRUSH[0])
        reg = pg.LinearRegionItem(
            values=[x0 - 0.5, x1 + 0.5],
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(*brush),
            pen=pg.mkPen(None),
        )
        reg.setZValue(-10)
        plot.addItem(reg)
        items.append(reg)
    return items


def add_signal_markers(
    plot: "pg.PlotWidget",
    bars: Sequence[ChartBar],
) -> List["pg.ScatterPlotItem"]:
    """BUY/SELL markers where ``ChartBar.signal`` is non-zero."""
    if not _HAS_PG or not bars:
        return []
    xs_buy, ys_buy = [], []
    xs_sell, ys_sell = [], []
    for i, b in enumerate(bars):
        if b.signal is None or b.signal == 0:
            continue
        y = float(b.low if b.signal > 0 else b.high)
        if b.signal > 0:
            xs_buy.append(i)
            ys_buy.append(y)
        else:
            xs_sell.append(i)
            ys_sell.append(y)
    items: List[pg.ScatterPlotItem] = []
    if xs_buy:
        buy = pg.ScatterPlotItem(
            xs_buy,
            ys_buy,
            symbol="t",
            size=12,
            brush=pg.mkBrush("#2e7d32"),
            pen=pg.mkPen("#1b5e20", width=1),
        )
        plot.addItem(buy)
        items.append(buy)
    if xs_sell:
        sell = pg.ScatterPlotItem(
            xs_sell,
            ys_sell,
            symbol="t1",
            size=12,
            brush=pg.mkBrush("#c62828"),
            pen=pg.mkPen("#b71c1c", width=1),
        )
        plot.addItem(sell)
        items.append(sell)
    return items


def plot_meta_probability(
    plot: "pg.PlotWidget",
    bars: Sequence[ChartBar],
) -> Optional["pg.PlotDataItem"]:
    """Second subplot: meta_probability when present on bars."""
    if not _HAS_PG:
        return None
    xs: List[int] = []
    ys: List[float] = []
    for i, b in enumerate(bars):
        if b.meta_probability is not None:
            xs.append(i)
            ys.append(float(b.meta_probability))
    if not xs:
        return None
    line = plot.plot(xs, ys, pen=pg.mkPen("#6a1b9a", width=1.5))
    plot.addLine(y=0.5, pen=pg.mkPen("#999", width=1, style=Qt.PenStyle.DashLine))
    plot.setLabel("left", "P(up)")
    plot.setYRange(0.0, 1.0)
    return line


def draw_chart_payload(
    price_plot: "pg.PlotWidget",
    meta_plot: Optional["pg.PlotWidget"],
    payload: ChartPayload,
    *,
    title: str = "",
) -> None:
    """Full redraw: candles, regime, signals, optional meta subplot."""
    if not _HAS_PG:
        return
    bars = payload.bars
    price_plot.clear()
    if meta_plot is not None:
        meta_plot.clear()

    if not bars:
        price_plot.setTitle(title or "Нет данных")
        return

    y_min, y_max = y_range_from_bars(bars)
    add_regime_background(price_plot, payload.regime_segments, y_min, y_max)

    data = [{"open": b.open, "high": b.high, "low": b.low, "close": b.close} for b in bars]
    from gui.app.widgets.candlestick import CandlestickItem

    price_plot.addItem(CandlestickItem(data))
    add_signal_markers(price_plot, bars)

    closes = np.array([b.close for b in bars], dtype=float)
    xs = np.arange(len(closes))
    price_plot.plot(xs, closes, pen=pg.mkPen("#1565c0", width=1, style=Qt.PenStyle.DotLine))

    price_plot.setTitle(title or f"{len(bars)} баров")
    price_plot.setLabel("left", "Цена")
    price_plot.setLabel("bottom", "Бар")

    if meta_plot is not None:
        plot_meta_probability(meta_plot, bars)


class ChartPlotController:
    """
    Управление слоями графика: полная отрисовка + live-обновление хвоста без сброса zoom/pan.
    """

    def __init__(self) -> None:
        self._price_plot: Optional["pg.PlotWidget"] = None
        self._meta_plot: Optional["pg.PlotWidget"] = None
        self._candles: Optional["CandlestickItem"] = None
        self._close_line: Optional["pg.PlotDataItem"] = None
        self._regime_items: List["pg.LinearRegionItem"] = []
        self._signal_items: List["pg.ScatterPlotItem"] = []
        self._meta_ref_line: Optional["pg.InfiniteLine"] = None
        self._bars: List[ChartBar] = []
        self._segments: List[tuple] = []
        self._saved_view: Optional[tuple] = None
        self._user_panned = False
        self._programmatic_view = False

    def attach(
        self,
        price_plot: "pg.PlotWidget",
        meta_plot: Optional["pg.PlotWidget"] = None,
    ) -> None:
        self._price_plot = price_plot
        self._meta_plot = meta_plot
        vb = price_plot.getViewBox()
        vb.sigRangeChanged.connect(self._on_view_range_changed)

    def _on_view_range_changed(self) -> None:
        if self._programmatic_view or not self._bars:
            return
        self._user_panned = True
        self._saved_view = self._price_plot.getViewBox().viewRange()

    def save_view(self) -> None:
        if self._price_plot is not None:
            self._saved_view = self._price_plot.getViewBox().viewRange()

    def restore_view(self) -> None:
        if not _HAS_PG or self._price_plot is None or self._saved_view is None:
            return
        if not self._user_panned:
            return
        xr, yr = self._saved_view
        self._set_view_range(xr, yr)

    def _set_view_range(self, xr, yr) -> None:
        vb = self._price_plot.getViewBox()
        self._programmatic_view = True
        try:
            vb.setRange(xRange=xr, yRange=yr, padding=0, update=True)
        finally:
            self._programmatic_view = False
        self._saved_view = vb.viewRange()

    def focus_recent(self, visible_bars: int = _DEFAULT_VISIBLE_BARS) -> None:
        """Показать правый хвост (после первой загрузки истории)."""
        if not _HAS_PG or self._price_plot is None or not self._bars:
            return
        n = len(self._bars)
        vis = min(visible_bars, n)
        y_min, y_max = y_range_from_bars(self._bars[max(0, n - vis) :])
        self._set_view_range(
            (max(0, n - vis) - 0.5, n - 0.5),
            (y_min, y_max),
        )
        self._user_panned = False

    def clear(self) -> None:
        if self._price_plot is not None:
            self._price_plot.clear()
        if self._meta_plot is not None:
            self._meta_plot.clear()
        self._candles = None
        self._close_line = None
        self._regime_items = []
        self._signal_items = []
        self._meta_ref_line = None
        self._bars = []
        self._segments = []
        self._user_panned = False
        self._saved_view = None

    def full_render(
        self,
        payload: ChartPayload,
        *,
        meta_plot: bool = False,
        title: str = "",
        focus_recent: bool = True,
    ) -> None:
        if not _HAS_PG or self._price_plot is None:
            return
        self.clear()
        self._bars = list(payload.bars)
        self._segments = list(payload.regime_segments)
        bars = self._bars
        if not bars:
            self._price_plot.setTitle(title or "Нет данных")
            return

        y_min, y_max = y_range_from_bars(bars)
        self._regime_items = add_regime_background(
            self._price_plot, self._segments, y_min, y_max
        )

        from gui.app.widgets.candlestick import CandlestickItem

        data = [
            {"open": b.open, "high": b.high, "low": b.low, "close": b.close}
            for b in bars
        ]
        self._candles = CandlestickItem(data)
        self._price_plot.addItem(self._candles)
        self._signal_items = add_signal_markers(self._price_plot, bars)

        closes = np.array([b.close for b in bars], dtype=float)
        xs = np.arange(len(closes))
        self._close_line = self._price_plot.plot(
            xs, closes, pen=pg.mkPen("#1565c0", width=1, style=Qt.PenStyle.DotLine)
        )

        self._price_plot.setTitle(title or f"{len(bars)} баров")
        self._price_plot.setLabel("left", "Цена")
        self._price_plot.setLabel("bottom", "Бар")

        if meta_plot and self._meta_plot is not None:
            self._meta_plot.clear()
            plot_meta_probability(self._meta_plot, bars)

        if focus_recent:
            self.focus_recent()

    def _at_live_edge(self) -> bool:
        if self._user_panned or not self._bars or self._price_plot is None:
            return False
        xr, _ = self._price_plot.getViewBox().viewRange()
        return float(xr[1]) >= len(self._bars) - 2.5

    def update_okx_live(
        self,
        bars: List[ChartBar],
        *,
        title: str = "",
        change: LiveBarChange = LiveBarChange.NONE,
    ) -> LiveBarChange:
        """
        Live без clear (как ``series.update`` в Lightweight Charts).

        Тот же open time → перерисовать последнюю свечу; больший → дописать справа.
        """
        if not _HAS_PG or self._price_plot is None:
            return False
        if not bars:
            return False

        old_n = len(self._bars)
        self._bars = bars
        new_n = len(bars)

        if self._candles is None or self._close_line is None:
            payload = ChartPayload(bars=bars, regime_segments=[])
            self.full_render(payload, title=title, focus_recent=old_n == 0)
            return change if change != LiveBarChange.NONE else LiveBarChange.NEW_BAR

        at_edge = self._at_live_edge()
        self.save_view()
        data = [
            {"open": b.open, "high": b.high, "low": b.low, "close": b.close}
            for b in bars
        ]
        self._candles.set_data(data)

        closes = np.array([b.close for b in bars], dtype=float)
        xs = np.arange(len(closes))
        self._close_line.setData(xs, closes)

        if change == LiveBarChange.NEW_BAR and at_edge:
            n = len(bars)
            vis = min(_DEFAULT_VISIBLE_BARS, n)
            y_min, y_max = y_range_from_bars(bars[max(0, n - vis) :])
            self._set_view_range((n - vis - 0.5, n - 0.5), (y_min, y_max))
            self._user_panned = False
        else:
            self.restore_view()

        if title:
            self._price_plot.setTitle(title)
        return change

    def full_render_from_bars(
        self,
        bars: List[ChartBar],
        *,
        regime_segments: Optional[List[tuple]] = None,
        meta_plot: bool = False,
        title: str = "",
        preserve_view: bool = False,
    ) -> None:
        payload = ChartPayload(
            bars=bars,
            regime_segments=regime_segments or [],
        )
        saved = self._saved_view
        panned = self._user_panned
        self.full_render(payload, meta_plot=meta_plot, title=title, focus_recent=not preserve_view)
        if preserve_view and saved is not None:
            self._saved_view = saved
            self._user_panned = panned
            self.restore_view()
