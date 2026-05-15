"""CLI: merge MTF features onto a base OHLCV parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synchronization.config import SynchronizationConfig
from synchronization.multi_timeframe_engine import MultiTimeframeEngine
from synchronization.storage import save_synced


def _parse_frame_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TF=path, e.g. 15m=data/ohlcv/foo.parquet")
    tf, path = value.split("=", 1)
    return tf.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synchronize auxiliary timeframes onto a base OHLCV file (MTF merge).",
    )
    p.add_argument("--base", type=Path, required=True, help="Base timeframe OHLCV (.parquet)")
    p.add_argument(
        "--frame",
        action="append",
        type=_parse_frame_arg,
        metavar="TF=PATH",
        required=True,
        help="Auxiliary TF and path, e.g. 15m=data/ohlcv/BTC_15m.parquet (repeatable)",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Output parquet path")
    p.add_argument("--base-timeframe", default="1h", help="Base TF label (for config, default 1h)")
    p.add_argument("--resample-rule", default=None, help="Pandas resample rule (default: base-timeframe)")
    p.add_argument("--fill", default="ffill", choices=("ffill", "bfill", "linear"))
    p.add_argument("--keep-na", action="store_true", help="Do not dropna after merge")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aux_paths = dict(args.frame)
    config = SynchronizationConfig(
        base_timeframe=args.base_timeframe,
        auxiliary_timeframes=list(aux_paths.keys()),
        resample_rule=args.resample_rule,
        fill_method=args.fill,
        drop_na_after_merge=not args.keep_na,
    )
    engine, base_df = MultiTimeframeEngine.from_parquet(
        str(args.base),
        {k: str(v) for k, v in aux_paths.items()},
        config=config,
    )
    merged = engine.compute_and_merge(base_df)
    out = save_synced(merged, args.output)
    if not args.quiet:
        print(f"Merged {len(merged)} rows, {len(merged.columns)} columns -> {out}")
        print(f"MTF columns: {[c for c in merged.columns if '15m' in c or '4h' in c]}")
    if merged.empty:
        print("Warning: merged dataframe is empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
