#!/usr/bin/env python3
"""
Пошаговый контур диплома: 4 модели, acceptance, BTC + ETH.

  python scripts/thesis_4model_steps.py --list
  python scripts/thesis_4model_steps.py --step 0
  python scripts/thesis_4model_steps.py --step all
  python scripts/thesis_4model_steps.py --step 3 --symbol BTC/USDT
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = """
Шаги дипломного контура (4 модели)

  0  check     Python, TF, GPU, parquet BTC/ETH
  1  pytest    Быстрые тесты (без integration)
  2  seed      report-real с config/symbols (семя acceptance)
  3  push      thesis_push_4model — узкий подбор → acceptance
  4  ablation  run_ablation.py на 8000 баров
  5  eth       push + report-real для ETH/USDT
  6  summary   Печать путей к JSON и журналу
"""


def _run(cmd: list[str], *, cwd: Path = ROOT) -> int:
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd))


def step_check(_args: argparse.Namespace) -> int:
    code = """
import sys
print("Python", sys.version)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow", tf.__version__, "GPU", gpus)
except Exception as e:
    print("TensorFlow:", e)
for m in ("lightgbm", "xgboost", "sklearn"):
    __import__(m)
    print(m, "OK")
"""
    subprocess.run([sys.executable, "-c", code], cwd=str(ROOT))
    for name in ("BTC-USDT_1h.parquet", "ETH-USDT_1h.parquet"):
        p = ROOT / "data" / "ohlcv" / name
        print(f"{'OK' if p.is_file() else 'MISSING'}: {p}")
    return 0


def step_pytest(_args: argparse.Namespace) -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_orchestration.py",
            "tests/test_backtesting_criteria.py",
            "tests/test_wfo_backtest.py",
            "-q",
            "--tb=line",
            "-m",
            "not integration",
        ]
    )


def step_seed(args: argparse.Namespace) -> int:
    sym = args.symbol or "BTC/USDT"
    tf = args.timeframe
    slug = sym.replace("/", "-")
    pq = ROOT / "data" / "ohlcv" / f"{slug}_{tf}.parquet"
    if not pq.is_file():
        print(f"Missing {pq}; run with --download or place parquet.", file=sys.stderr)
        return 2
    return _run(
        [
            sys.executable,
            "-m",
            "orchestration",
            "report-real",
            "--parquet",
            str(pq),
            "--max-rows",
            str(args.max_rows),
            "--use-tuning-best",
            "--symbol",
            sym,
            "--timeframe",
            tf,
            "--dl-epochs",
            str(args.dl_epochs),
            "--json-out",
            f"docs/thesis_seed_{slug}_{tf}.json",
        ]
    )


def step_push(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/thesis_push_4model.py",
        "--max-rows",
        str(args.max_rows),
        "--dl-epochs",
        str(args.dl_epochs),
        "--write-symbol-yaml",
    ]
    if args.symbol:
        cmd.extend(["--symbol", args.symbol, "--timeframe", args.timeframe])
    else:
        cmd.extend(["--parquet", str(ROOT / "data" / "ohlcv" / "BTC-USDT_1h.parquet")])
    return _run(cmd)


def step_ablation(args: argparse.Namespace) -> int:
    pq = ROOT / "data" / "ohlcv" / "BTC-USDT_1h.parquet"
    return _run(
        [
            sys.executable,
            "scripts/run_ablation.py",
            "--parquet",
            str(pq),
            "--max-rows",
            str(args.max_rows),
            "--dl-epochs",
            str(args.dl_epochs),
            "--out",
            "docs/reports/ablation_thesis_8k.json",
        ]
    )


def step_eth(args: argparse.Namespace) -> int:
    args.symbol = "ETH/USDT"
    r1 = step_push(args)
    r2 = step_seed(args)
    return r1 or r2


def step_summary(_args: argparse.Namespace) -> int:
    paths = [
        ROOT / "docs" / "THESIS_4MODEL_STEPS.md",
        ROOT / "docs" / "backtest_journal" / "INDEX.md",
        ROOT / "config" / "symbols" / "BTC-USDT_1h.yaml",
        ROOT / "config" / "symbols" / "ETH-USDT_1h.yaml",
    ]
    for p in paths:
        print(p, "exists" if p.is_file() else "missing")
    print("\nОткройте последние run_id в docs/backtest_journal/runs.jsonl (acceptance_passed: true).")
    return 0


HANDLERS = {
    "0": step_check,
    "check": step_check,
    "1": step_pytest,
    "pytest": step_pytest,
    "2": step_seed,
    "seed": step_seed,
    "3": step_push,
    "push": step_push,
    "4": step_ablation,
    "ablation": step_ablation,
    "5": step_eth,
    "eth": step_eth,
    "6": step_summary,
    "summary": step_summary,
}

ALL_SEQUENCE = ["0", "1", "2", "3", "4", "5", "6"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Thesis 4-model step runner")
    p.add_argument("--list", action="store_true", help="Show steps")
    p.add_argument("--step", default=None, help="Step id or 'all'")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--max-rows", type=int, default=8000)
    p.add_argument("--dl-epochs", type=int, default=3)
    args = p.parse_args(argv)

    if args.list:
        print(STEPS)
        return 0

    if not args.step:
        print(STEPS)
        print("Use: python scripts/thesis_4model_steps.py --step 0  |  --step all")
        return 0

    steps = ALL_SEQUENCE if args.step == "all" else [args.step]
    rc = 0
    for s in steps:
        fn = HANDLERS.get(s)
        if fn is None:
            print(f"Unknown step: {s}", file=sys.stderr)
            return 2
        print(f"\n========== STEP {s} ==========")
        if fn(args) != 0:
            rc = 1
            if args.step != "all":
                break
            print(f"Step {s} failed (continuing in --step all mode).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
