#!/usr/bin/env python3
"""Single entry point for the ITS research pipeline.

The project is intentionally kept as a pipeline-first package:

data -> features -> targets -> models -> decision -> backtesting/execution

UI launchers and demo dashboards were removed from this entry point so the
core system can be tested, refactored, and reasoned about as one flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("its_project")


CORE_LAYERS: dict[str, list[str]] = {
    "common": ["types.py", "config.py", "logging.py", "rate_limiter.py"],
    "data_layer": ["base.py", "ccxt_source.py", "okx_source.py", "real_market_data.py"],
    "storage": ["base.py", "parquet.py", "parquet_store.py", "timescale.py", "storage_manager.py"],
    "features": ["base.py", "pipeline.py", "technical.py", "economic_features.py", "microstructure_features.py"],
    "targets": ["economic_target.py", "returns_target.py"],
    "models": ["base.py", "boosting_model.py", "economic_boosting_model.py", "ensemble.py", "registry.py"],
    "metalearning": ["cv.py", "hyperopt.py", "selector.py", "weighted_ensemble.py"],
    "decision": ["decision.py", "signal_generator.py", "simple.py", "enhanced_decision.py", "economic_decision_maker.py"],
    "execution": ["base.py", "paper.py", "live.py", "manager.py", "kill_switch.py", "pnl_tracker.py"],
    "backtesting": ["base.py", "economic_backtester.py", "walkforward_validator.py", "baseline_strategies.py"],
    "system": ["config_system.py", "structured_logging.py"],
    "mlops": ["drift_detection.py", "experiment_tracking.py", "versioning.py"],
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def make_demo_ohlcv(n_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create deterministic OHLCV data for a smoke-testable pipeline run."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    returns = rng.normal(0.0001, 0.01, n_samples)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, n_samples))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, n_samples))
    volume = rng.lognormal(mean=8.0, sigma=0.35, size=n_samples)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def run_demo(args: argparse.Namespace) -> int:
    """Run the canonical local pipeline on synthetic data."""
    from its_project.features.economic_features import EconomicFeatures
    from its_project.features.microstructure_features import MicrostructureFeatures
    from its_project.models.economic_boosting_model import EconomicBoostingModel
    from its_project.targets.economic_target import EconomicTargetCalculator

    data = make_demo_ohlcv(args.samples)

    target_calculator = EconomicTargetCalculator(
        {
            "horizon": args.horizon,
            "threshold": args.threshold,
            "target_type": "direction",
        }
    )
    target, target_metadata = target_calculator.calculate_target(data)

    economic_features = EconomicFeatures(
        {
            "return_periods": [1, 5, 15],
            "volatility_windows": [5, 15],
            "use_risk_features": True,
        }
    ).calculate(data)
    microstructure_features = MicrostructureFeatures(
        {
            "use_order_book": False,
            "impact_window": 20,
            "efficiency_window": 50,
        }
    ).calculate(data)

    features = np.hstack([economic_features, microstructure_features])
    valid_mask = target.notna().to_numpy()
    X = features[valid_mask]
    y = target.to_numpy()[valid_mask].astype(int)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = EconomicBoostingModel(
        {
            "n_estimators": args.estimators,
            "learning_rate": 0.05,
            "max_depth": 3,
            "use_class_weights": True,
            "early_stopping_rounds": 5,
        }
    )
    model.fit(X_train, y_train)

    train_metrics = model.evaluate_economic_metrics(X_train, y_train)
    test_metrics = model.evaluate_economic_metrics(X_test, y_test)

    result: dict[str, Any] = {
        "samples": int(len(data)),
        "usable_samples": int(len(X)),
        "feature_shape": list(X.shape),
        "target_distribution": {
            str(k): int(v) for k, v in target_metadata["class_distribution"].items()
        },
        "train_accuracy": round(float(train_metrics["accuracy"]), 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "test_buy_signal_frequency": round(float(test_metrics["buy_signal_frequency"]), 4),
        "test_sell_signal_frequency": round(float(test_metrics["sell_signal_frequency"]), 4),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_check(_: argparse.Namespace) -> int:
    """Check that core layer directories and files are present."""
    missing: list[str] = []
    for layer, files in CORE_LAYERS.items():
        layer_dir = PROJECT_ROOT / layer
        if not layer_dir.exists():
            missing.append(f"{layer}/")
            continue
        for file_name in files:
            if not (layer_dir / file_name).exists():
                missing.append(f"{layer}/{file_name}")

    if missing:
        print("Missing core files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("Core pipeline structure is present.")
    return 0


def run_layers(_: argparse.Namespace) -> int:
    """Print the canonical core layer map."""
    print(json.dumps(CORE_LAYERS, indent=2, ensure_ascii=False))
    return 0


def run_tests(args: argparse.Namespace) -> int:
    """Run pytest with repository-local paths."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests"),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if args.collect_only:
        cmd.append("--collect-only")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ITS single pipeline launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate core pipeline structure")
    check_parser.set_defaults(func=run_check)

    layers_parser = subparsers.add_parser("layers", help="Print canonical layer map")
    layers_parser.set_defaults(func=run_layers)

    demo_parser = subparsers.add_parser("demo", help="Run synthetic end-to-end pipeline demo")
    demo_parser.add_argument("--samples", type=int, default=600)
    demo_parser.add_argument("--horizon", type=int, default=5)
    demo_parser.add_argument("--threshold", type=float, default=0.002)
    demo_parser.add_argument("--estimators", type=int, default=50)
    demo_parser.set_defaults(func=run_demo)

    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--collect-only", action="store_true")
    test_parser.set_defaults(func=run_tests)

    return parser


def main() -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
