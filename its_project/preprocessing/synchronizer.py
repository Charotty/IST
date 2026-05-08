from __future__ import annotations

from typing import Mapping

import pandas as pd


class DataSynchronizer:
    """Timestamp alignment utilities for market-data streams."""

    def validate_timestamps(self, data: pd.DataFrame, timestamp_col: str = "timestamp") -> bool:
        if timestamp_col not in data.columns:
            return isinstance(data.index, pd.DatetimeIndex) and data.index.is_monotonic_increasing
        return pd.to_datetime(data[timestamp_col]).is_monotonic_increasing

    def align_timestamps(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        left_indexed = self._with_datetime_index(left)
        right_indexed = self._with_datetime_index(right)
        aligned = pd.merge_asof(
            left_indexed.sort_index(),
            right_indexed.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )
        return aligned.reset_index().rename(columns={"index": "timestamp"})

    def resample(self, data: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
        indexed = self._with_datetime_index(data)
        numeric = indexed.select_dtypes(include="number")
        return numeric.resample(freq).mean().ffill().reset_index().rename(columns={"index": "timestamp"})

    def fill_gaps(self, data: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
        if method in {"forward", "ffill"}:
            return data.ffill().bfill()
        if method in {"backward", "bfill"}:
            return data.bfill().ffill()
        if method == "zero":
            return data.fillna(0)
        raise ValueError(f"Unknown gap-fill method: {method}")

    def synchronize_streams(self, streams: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        renamed = []
        for stream_name, frame in streams.items():
            indexed = self._with_datetime_index(frame)
            value_columns = [column for column in indexed.columns if column != "timestamp"]
            renamed.append(indexed[value_columns].add_prefix(f"{stream_name}_"))

        if not renamed:
            return pd.DataFrame()

        synced = pd.concat(renamed, axis=1).sort_index()
        synced = synced.ffill().bfill()
        return synced.reset_index().rename(columns={"index": "timestamp"})

    def synchronize(self, streams: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        items = list(streams.items())
        if not items:
            return pd.DataFrame()

        _, base_frame = items[0]
        synced = self._with_datetime_index(base_frame).reset_index()
        for stream_name, frame in items[1:]:
            right = self._with_datetime_index(frame).reset_index()
            value_columns = [column for column in right.columns if column != "timestamp"]
            rename_map = {
                column: stream_name if len(value_columns) == 1 else f"{stream_name}_{column}"
                for column in value_columns
            }
            right = right.rename(columns=rename_map)
            synced = pd.merge_asof(
                synced.sort_values("timestamp"),
                right.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
            )
        return self.fill_gaps(synced, method="forward")

    @staticmethod
    def _with_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame = frame.set_index("timestamp")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Data must have a timestamp column or DatetimeIndex")
        return frame.sort_index()
