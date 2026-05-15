"""CLI: compute features from synced or raw OHLCV parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_manager import FeatureManager
from feature_engineering.storage import save_features


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ML features from OHLCV (optionally MTF-synced) parquet.")
    p.add_argument("-i", "--input", type=Path, required=True, help="Input parquet (OHLCV or synced MTF)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output .parquet or .csv")
    p.add_argument(
        "--microstructure",
        choices=("simulated", "live", "off"),
        default="simulated",
        help="L2 feature mode (default: simulated)",
    )
    p.add_argument("--no-drop-na", action="store_true", help="Keep rows with NaN after indicators")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FeatureEngineeringConfig(
        drop_na=not args.no_drop_na,
        microstructure=config_micro(args.microstructure),
    )
    manager = FeatureManager(config)
    result = manager.from_parquet(args.input)
    if result.empty:
        print("Warning: empty feature matrix", file=sys.stderr)
        return 1
    out = save_features(result, args.output)
    if not args.quiet:
        mtf = [c for c in result.columns if "15m" in c or "4h" in c]
        print(f"Features: {result.shape[0]} rows x {result.shape[1]} cols -> {out}")
        if mtf:
            print(f"MTF columns preserved: {mtf}")
        if "order_book_imbalance" in result.columns:
            print("Microstructure: order_book_imbalance, bid_ask_spread")
    return 0


def config_micro(mode: str):
    from feature_engineering.config import MicrostructureConfig

    return MicrostructureConfig(mode=mode)


if __name__ == "__main__":
    raise SystemExit(main())
