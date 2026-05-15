"""Multi-timeframe OHLCV alignment and MTF feature merge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from synchronization.config import SynchronizationConfig
from synchronization.gap_handler import align_to_base, ensure_datetime_index
from synchronization.mtf_features import build_features

if TYPE_CHECKING:
    from data_layer.loaders.okx_ohlcv_loader import OKXDataLoader


class MultiTimeframeEngine:
    """
    Load auxiliary OHLCV (optional), compute MTF features, resample to base TF.

    Reference: ``MultiTimeframeEngine`` in ``ist.py``.
    """

    def __init__(
        self,
        config: SynchronizationConfig | None = None,
        loader: OKXDataLoader | None = None,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        frames: dict[str, pd.DataFrame] | None = None,
    ):
        self.config = config or SynchronizationConfig()
        self.loader = loader
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self._frames: dict[str, pd.DataFrame] = {}
        if frames:
            self.set_frames(frames)

    def set_frames(self, frames: dict[str, pd.DataFrame]) -> "MultiTimeframeEngine":
        self._frames = {tf: ensure_datetime_index(df) for tf, df in frames.items()}
        return self

    def _need_loader(self) -> None:
        if self.loader is None or self.symbol is None:
            raise ValueError("loader and symbol are required to fetch auxiliary data")
        if self.start_date is None or self.end_date is None:
            raise ValueError("start_date and end_date are required to fetch auxiliary data")

    def get_multi_data(self, *, verbose: bool = True) -> "MultiTimeframeEngine":
        """Fetch auxiliary timeframes via ``OKXDataLoader`` (not the base series)."""
        self._need_loader()
        if verbose:
            print("--- Загрузка дополнительных таймфреймов ---")
        for tf in self.config.auxiliary_timeframes:
            assert self.loader is not None
            assert self.symbol is not None
            assert self.start_date is not None
            assert self.end_date is not None
            self._frames[tf] = self.loader.fetch_all_ohlcv(
                self.symbol,
                tf,
                self.start_date,
                self.end_date,
                verbose=verbose,
            )
        return self

    def compute_and_merge(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Compute MTF features, align to base index, join, optional ``dropna``."""
        df = ensure_datetime_index(base_df)
        missing = [tf for tf in self.config.auxiliary_timeframes if tf not in self._frames]
        if missing:
            raise ValueError(
                f"Missing auxiliary frames for: {missing}. "
                "Call get_multi_data() or set_frames() first."
            )

        for tf in self.config.auxiliary_timeframes:
            ohlcv = self._frames[tf]
            features = build_features(tf, ohlcv)
            aligned = align_to_base(features, self.config.rule, self.config.fill_method)
            df = df.join(aligned, how="left")

        if self.config.drop_na_after_merge:
            df = df.dropna()
        return df

    @classmethod
    def from_parquet(
        cls,
        base_path: str,
        auxiliary_paths: dict[str, str],
        config: SynchronizationConfig | None = None,
    ) -> tuple["MultiTimeframeEngine", pd.DataFrame]:
        """Load base + auxiliary OHLCV from parquet files."""
        base_df = pd.read_parquet(base_path)
        frames = {tf: pd.read_parquet(path) for tf, path in auxiliary_paths.items()}
        engine = cls(config=config, frames=frames)
        return engine, ensure_datetime_index(base_df)
