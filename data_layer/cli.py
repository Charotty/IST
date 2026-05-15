"""CLI: download OKX OHLCV and save to a file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_layer.config import DataLayerConfig, MultiTimeframeConfig
from data_layer.loaders.okx_ohlcv_loader import OKXDataLoader
from data_layer.storage import default_output_path, save_ohlcv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download historical OHLCV from OKX (ccxt) and save to parquet/csv.",
    )
    p.add_argument("-s", "--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT")
    p.add_argument("-t", "--timeframe", default="1h", help="Candle interval, e.g. 1h, 15m, 4h")
    p.add_argument(
        "--from",
        dest="start_date",
        default="2020-01-01 00:00:00",
        metavar="START",
        help="Start datetime (ISO), e.g. 2024-01-01 or '2024-01-01 00:00:00'",
    )
    p.add_argument(
        "--to",
        dest="end_date",
        default="2026-01-01 00:00:00",
        metavar="END",
        help="End datetime (ISO, inclusive trim)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (.parquet or .csv). Default: data/ohlcv/<auto>",
    )
    p.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="File format when --output is omitted (default: parquet)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ohlcv"),
        help="Directory for auto-generated filenames (default: data/ohlcv)",
    )
    p.add_argument(
        "--mtf",
        action="store_true",
        help="Also download extra timeframes (15m, 4h by default; see --mtf-timeframes)",
    )
    p.add_argument(
        "--mtf-timeframes",
        nargs="+",
        default=["15m", "4h"],
        help="Extra timeframes with --mtf (default: 15m 4h)",
    )
    p.add_argument("--no-rate-limit", action="store_true", help="Disable ccxt rate limit sleep")
    p.add_argument("-q", "--quiet", action="store_true", help="Less progress output")
    return p


def normalize_date(s: str) -> str:
    """Accept 'YYYY-MM-DD' or full datetime."""
    s = s.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s} 00:00:00"
    return s


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = normalize_date(args.start_date)
    end = normalize_date(args.end_date)

    config = DataLayerConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
        multi_timeframe=MultiTimeframeConfig(
            enabled=args.mtf,
            timeframes=args.mtf_timeframes,
        ),
        rate_limit=not args.no_rate_limit,
    )
    loader = OKXDataLoader(rate_limit=config.rate_limit)
    verbose = not args.quiet
    saved: list[Path] = []

    timeframes = config.all_timeframes() if args.mtf else [config.timeframe]
    for tf in timeframes:
        df = loader.fetch_all_ohlcv(
            config.symbol,
            tf,
            config.start_date,
            config.end_date,
            verbose=verbose,
        )
        if df.empty:
            print(f"Warning: no data for {config.symbol} {tf}", file=sys.stderr)
            continue

        if args.output is not None and len(timeframes) == 1:
            out = args.output
        else:
            out = default_output_path(
                config.symbol,
                tf,
                config.start_date,
                config.end_date,
                output_dir=args.output_dir,
                fmt=args.format,
            )

        path = save_ohlcv(df, out, fmt=args.format)
        saved.append(path)
        if verbose:
            print(f"Saved {len(df)} rows -> {path}")

    if not saved:
        print("No files written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
