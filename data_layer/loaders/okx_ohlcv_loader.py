"""OKX OHLCV loader via ccxt REST (pagination, rate limit)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import ccxt
import pandas as pd

_MAX_FETCH_ERRORS = 8

from data_layer.config import DataLayerConfig
from data_layer.validators.ohlcv_validator import validate_ohlcv

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class OKXDataLoader:
    """Historical OHLCV from OKX; logic aligned with ``ist.py``."""

    def __init__(self, rate_limit: bool = True, exchange_factory: Callable[[], ccxt.Exchange] | None = None):
        self._rate_limit_enabled = rate_limit
        if exchange_factory is not None:
            self.exchange = exchange_factory()
        else:
            self.exchange = ccxt.okx({"enableRateLimit": rate_limit})

    def fetch_all_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_str: str,
        end_str: str,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Load OHLCV from ``start_str`` to ``end_str`` with pagination."""
        since = self.exchange.parse8601(start_str)
        end_ts = self.exchange.parse8601(end_str)
        all_ohlcv: list[list] = []

        if verbose:
            print(f"Начинаю загрузку {symbol} ({timeframe})...")

        errors = 0
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
                if not ohlcv:
                    break

                last_ts = ohlcv[-1][0]
                if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                    break

                all_ohlcv.extend(ohlcv)
                since = last_ts + 1
                errors = 0

                if verbose:
                    current_dt = datetime.fromtimestamp(last_ts / 1000)
                    print(f"Загружено до: {current_dt}", end="\r")

                if self._rate_limit_enabled:
                    time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                errors += 1
                if errors > _MAX_FETCH_ERRORS:
                    if verbose:
                        print(f"\nЗагрузка прервана после {_MAX_FETCH_ERRORS} ошибок: {e}")
                    break
                wait = min(60.0, 2.0**errors)
                if verbose:
                    print(f"\nОшибка ({errors}/{_MAX_FETCH_ERRORS}), пауза {wait:.0f}s: {e}")
                time.sleep(wait)

        if not all_ohlcv:
            empty = pd.DataFrame(columns=OHLCV_COLUMNS)
            empty["timestamp"] = pd.to_datetime([], utc=True)
            return empty.set_index("timestamp")

        df = pd.DataFrame(all_ohlcv, columns=OHLCV_COLUMNS)
        df = validate_ohlcv(df, end_str=end_str)

        if verbose:
            print(f"\nЗагрузка завершена. Всего строк: {len(df)}")

        return df

    def fetch_from_config(self, config: DataLayerConfig, timeframe: str | None = None) -> pd.DataFrame:
        """Load one timeframe using ``DataLayerConfig`` defaults."""
        tf = timeframe or config.timeframe
        return self.fetch_all_ohlcv(
            config.symbol,
            tf,
            config.start_date,
            config.end_date,
        )

    def fetch_all_timeframes(self, config: DataLayerConfig) -> dict[str, pd.DataFrame]:
        """Load base TF and optional MTF series; keys are timeframe strings."""
        result: dict[str, pd.DataFrame] = {}
        for tf in config.all_timeframes():
            result[tf] = self.fetch_from_config(config, tf)
        return result
