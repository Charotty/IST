from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class AppConfig:
    symbols_spot: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    symbols_futures: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    binance_ws_base_spot: str = "wss://stream.binance.com:9443/stream"
    binance_ws_base_futures: str = "wss://fstream.binance.com/stream"

    binance_rest_base_spot: str = "https://api.binance.com"
    binance_rest_base_futures: str = "https://fapi.binance.com"

    binance_spot_streams: list[str] = field(default_factory=lambda: ["trade", "depth@100ms"])
    binance_futures_streams: list[str] = field(default_factory=lambda: ["trade", "depth@250ms"])

    glassnode_api_key: str | None = None
    glassnode_poll_interval_s: float = 60.0

    twitter_bearer_token: str | None = None
    sentiment_poll_interval_s: float = 10.0

    queue_maxsize: int = 10_000

    # Storage
    timescale_dsn: str = "postgres://user:pass@localhost/tsdb"
    parquet_base_path: str = "data/parquet"

    @property
    def symbols(self) -> Sequence[str]:
        # Union preserving order
        seen: set[str] = set()
        out: list[str] = []
        for s in [*self.symbols_spot, *self.symbols_futures]:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out


def load_config(path: str | Path = "config.json") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()

    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = AppConfig()

    if isinstance(raw, dict):
        if isinstance(raw.get("symbols_spot"), list):
            cfg.symbols_spot = [str(x) for x in raw["symbols_spot"]]
        if isinstance(raw.get("symbols_futures"), list):
            cfg.symbols_futures = [str(x) for x in raw["symbols_futures"]]
        if isinstance(raw.get("queue_maxsize"), int):
            cfg.queue_maxsize = int(raw["queue_maxsize"])

        if isinstance(raw.get("binance_spot_streams"), list):
            cfg.binance_spot_streams = [str(x) for x in raw["binance_spot_streams"]]
        if isinstance(raw.get("binance_futures_streams"), list):
            cfg.binance_futures_streams = [str(x) for x in raw["binance_futures_streams"]]

        if isinstance(raw.get("binance_rest_base_spot"), str):
            cfg.binance_rest_base_spot = str(raw["binance_rest_base_spot"])
        if isinstance(raw.get("binance_rest_base_futures"), str):
            cfg.binance_rest_base_futures = str(raw["binance_rest_base_futures"])

        if isinstance(raw.get("glassnode_api_key"), str):
            cfg.glassnode_api_key = str(raw["glassnode_api_key"])
        if isinstance(raw.get("glassnode_poll_interval_s"), (int, float)):
            cfg.glassnode_poll_interval_s = float(raw["glassnode_poll_interval_s"])

        if isinstance(raw.get("twitter_bearer_token"), str):
            cfg.twitter_bearer_token = str(raw["twitter_bearer_token"])
        if isinstance(raw.get("sentiment_poll_interval_s"), (int, float)):
            cfg.sentiment_poll_interval_s = float(raw["sentiment_poll_interval_s"])

        if isinstance(raw.get("timescale_dsn"), str):
            cfg.timescale_dsn = str(raw["timescale_dsn"])
        if isinstance(raw.get("parquet_base_path"), str):
            cfg.parquet_base_path = str(raw["parquet_base_path"])

    return cfg
