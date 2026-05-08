#!/usr/bin/env python3
"""Run full OKX benchmark: ALL models x ALL timeframes with plots.

This script is designed for your request:
- a) all existing models in ModelRegistry
- b) all supported timeframes
- c) graphical comparison to analyze errors / potential overfitting

It will:
- load REAL OKX OHLCV once per (symbol,timeframe)
- build features/targets once per (symbol,timeframe)
- train/evaluate each model with a time-based split
- compute BOTH train and test metrics (for overfitting diagnosis)
- save a unified JSON + CSV

Then you can run okx_plot_report.py to generate figures.

Example:
  python okx_matrix_benchmark.py --symbols BTC/USDT ETH/USDT --n_samples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import models package to register everything available
import models  # noqa: F401

from models.registry import ModelRegistry
from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets
from evaluation.model_evaluator import ModelEvaluator


TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def load_okx_ohlcv(symbol: str, timeframe: str, limit: int) -> Optional[np.ndarray]:
    """Load OHLCV from OKX using backend.OKXRESTClient (CCXT) with pagination."""

    try:
        from backend.okx_rest_client import OKXRESTClient

        client = OKXRESTClient()
        if not client.connect():
            return None

        if hasattr(client, "get_historical_ohlcv"):
            max_limit = 300
            ohlcv_data: List[List[float]] = []
            remaining = limit

            import time as time_module

            timeframe_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }.get(timeframe, 60 * 60 * 1000)

            current_time = int(time_module.time() * 1000)
            since = current_time - (limit * timeframe_ms)

            while remaining > 0:
                current_limit = min(max_limit, remaining)
                chunk = client.get_historical_ohlcv(symbol, timeframe, since=since, limit=current_limit)
                if not chunk:
                    break

                existing_ts = set(c[0] for c in ohlcv_data)
                for c in chunk:
                    if c[0] not in existing_ts:
                        ohlcv_data.append(c)
                        existing_ts.add(c[0])

                remaining -= len(chunk)
                if len(chunk) < current_limit:
                    break

                since = int(chunk[-1][0]) + 1
                time_module.sleep(0.1)

            if not ohlcv_data:
                return None

            ohlcv_data.sort(key=lambda x: x[0])
            return np.array([[c[1], c[2], c[3], c[4], c[5]] for c in ohlcv_data], dtype=np.float64)

        raw = client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        raw.sort(key=lambda x: x[0])
        return np.array([[c[1], c[2], c[3], c[4], c[5]] for c in raw], dtype=np.float64)

    except Exception:
        return None


def time_split(arr: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    idx = int(len(arr) * train_ratio)
    return arr[:idx], arr[idx:]


def make_sequences(X2d: np.ndarray, y1d: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 1:
        return X2d.reshape(X2d.shape[0], 1, X2d.shape[1]), y1d
    if len(X2d) < seq_len:
        raise ValueError("not enough samples")

    X_seq = np.stack([X2d[i : i + seq_len] for i in range(len(X2d) - seq_len + 1)], axis=0)
    y_seq = y1d[seq_len - 1 :]
    return X_seq, y_seq


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
    ev = ModelEvaluator()
    return ev.evaluate_classification(y_true, y_pred, y_proba)


def regression_to_labels(y_pred: np.ndarray, thr: float = 0.001) -> np.ndarray:
    # Map continuous returns to {0,1,2}
    labels = np.full(len(y_pred), 1, dtype=int)
    labels[y_pred > thr] = 2
    labels[y_pred < -thr] = 0
    return labels


def train_and_eval_one(
    model_name: str,
    X: np.ndarray,
    y_returns: np.ndarray,
    y_labels: np.ndarray,
    lookback: int,
    prob_threshold: float,
) -> Dict[str, Any]:
    """Train/eval with train+test metrics."""

    from sklearn.preprocessing import StandardScaler

    # Split chronologically
    X_train, X_test = time_split(X, 0.8)
    yret_train, yret_test = time_split(y_returns, 0.8)
    y_train, y_test = time_split(y_labels, 0.8)

    sequence_models = {"gru", "lstm", "transformer"}
    seq_len = lookback if model_name in sequence_models else 1

    # Scale based on train only
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Prepare tensors
    if model_name in sequence_models:
        X_train_s, y_train_s = make_sequences(X_train_s, y_train, seq_len)
        X_test_s, y_test_s = make_sequences(X_test_s, y_test, seq_len)

        # align returns for trading
        _, yret_test_s = make_sequences(yret_test.reshape(-1, 1), yret_test, seq_len)
        yret_test_s = yret_test_s.astype(np.float64)
    else:
        y_train_s, y_test_s = y_train, y_test
        yret_test_s = yret_test.astype(np.float64)

    # Default configs
    configs: Dict[str, Dict[str, Any]] = {
        "boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "class_weight": "balanced", "random_state": 42},
        "ensemble": {"rf_estimators": 300, "rf_max_depth": 20, "gb_estimators": 300, "gb_max_depth": 5},
        "regression": {"model_type": "gradient_boosting", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
        "gru": {"input_size": 0, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "epochs": 15, "batch_size": 64, "learning_rate": 0.001, "patience": 5, "num_classes": 3},
        "lstm": {"input_size": 0, "hidden_size": 64, "num_layers": 2, "output_size": 3, "dropout": 0.2, "epochs": 15, "batch_size": 64, "num_classes": 3},
        "transformer": {"input_size": 0, "d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.2, "epochs": 15, "batch_size": 64, "num_classes": 3},
        # cnn_lob is present in registry, but requires orderbook-shaped inputs; we skip it here.
    }

    cfg = configs.get(model_name, {})

    # Reject CNN LOB in this OHLCV-only benchmark
    if model_name == "cnn_lob":
        return {"skipped": True, "reason": "cnn_lob requires orderbook features / 2D book tensors; this benchmark uses OHLCV-only features."}

    if model_name in sequence_models:
        cfg["input_size"] = int(X_train_s.shape[-1])

    model = ModelRegistry.get_model(model_name, cfg)

    # Train target for regression differs
    if model_name == "regression":
        # For regression we predict continuous next-step returns (y_returns)
        # X and y_returns are already aligned by FeatureEngineer.
        model.fit(X_train_s, yret_train)
        y_pred_train = model.predict(X_train_s)
        y_pred_test = model.predict(X_test_s)

        ev = ModelEvaluator()
        reg_train = ev.evaluate_regression(yret_train, y_pred_train)
        reg_test = ev.evaluate_regression(yret_test, y_pred_test)

        # Also compute classification-like metrics by thresholding regression outputs
        y_pred_train_cls = regression_to_labels(y_pred_train)
        y_pred_test_cls = regression_to_labels(y_pred_test)

        clf_train = compute_classification_metrics(y_train, y_pred_train_cls, None)
        clf_test = compute_classification_metrics(y_test, y_pred_test_cls, None)

        return {
            "skipped": False,
            "train": {"regression": reg_train, "classification": clf_train},
            "test": {"regression": reg_test, "classification": clf_test},
        }

    # Classification models
    model.fit(X_train_s, y_train_s)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    y_proba_train = None
    y_proba_test = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba_train = model.predict_proba(X_train_s)
            y_proba_test = model.predict_proba(X_test_s)
        except Exception:
            y_proba_train = None
            y_proba_test = None

    train_metrics = compute_classification_metrics(y_train_s, y_pred_train, y_proba_train)
    test_metrics = compute_classification_metrics(y_test_s, y_pred_test, y_proba_test)

    # Trading metrics on test (BUY only, confidence filter)
    trading = None
    if y_proba_test is not None:
        ev = ModelEvaluator()
        returns = np.zeros(len(yret_test_s), dtype=np.float64)
        trades = 0
        for i in range(len(yret_test_s)):
            max_prob = float(np.max(y_proba_test[i]))
            signal = int(np.argmax(y_proba_test[i]))
            if max_prob >= prob_threshold and signal == 2:
                returns[i] = float(yret_test_s[i])
                trades += 1
        trading = ev.evaluate_trading_performance(returns)
        trading["trades"] = trades
        trading["trade_rate"] = trades / max(1, len(yret_test_s))

    return {
        "skipped": False,
        "train": {"classification": train_metrics},
        "test": {"classification": test_metrics, "trading": trading},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OKX full matrix benchmark")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"], help="Symbols in GUI format")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_lags", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--prob_threshold", type=float, default=0.6)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "okx_reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_models = sorted(ModelRegistry.list_models())

    results: List[Dict[str, Any]] = []

    # Cache loaded+engineered data per (symbol,timeframe)
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for symbol in args.symbols:
        for timeframe in TIMEFRAMES:
            key = (symbol, timeframe)
            print(f"\n=== Loading OKX data {symbol} {timeframe} candles={args.n_samples} ===")

            ohlcv = load_okx_ohlcv(symbol, timeframe, args.n_samples)
            if ohlcv is None:
                print("  -> failed to load, skipping")
                for model_name in all_models:
                    results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "model": model_name,
                        "status": "skipped",
                        "reason": "failed_to_load_okx_ohlcv",
                    })
                continue

            fe = FeatureEngineer(
                n_lags=args.n_lags,
                volatility_window=10,
                include_volume=True,
                include_indicators=True,
            )
            X, y_returns = fe.build_features_from_ohlcv(ohlcv, lookback=args.lookback)
            y_labels = prepare_classification_targets(
                y_returns,
                use_three_classes=True,
                adaptive_threshold=True,
                volatility_window=20,
            )

            cache[key] = {
                "X": X,
                "y_returns": y_returns,
                "y_labels": y_labels,
                "feature_names": fe.get_feature_names(),
            }

            # Benchmark all models
            for model_name in all_models:
                print(f"  -> model={model_name}")
                try:
                    out = train_and_eval_one(
                        model_name=model_name,
                        X=X,
                        y_returns=y_returns,
                        y_labels=y_labels,
                        lookback=args.lookback,
                        prob_threshold=args.prob_threshold,
                    )

                    if out.get("skipped"):
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "model": model_name,
                            "status": "skipped",
                            "reason": out.get("reason", "unknown"),
                        })
                    else:
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "model": model_name,
                            "status": "ok",
                            "train": out.get("train"),
                            "test": out.get("test"),
                        })
                except Exception as e:
                    results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "model": model_name,
                        "status": "error",
                        "error": str(e),
                    })

    report = {
        "generated_at": stamp,
        "symbols": args.symbols,
        "timeframes": TIMEFRAMES,
        "models": all_models,
        "n_samples": args.n_samples,
        "n_lags": args.n_lags,
        "lookback": args.lookback,
        "prob_threshold": args.prob_threshold,
        "results": results,
    }

    json_path = out_dir / f"okx_matrix_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON report: {json_path}")

    # CSV export (flat)
    try:
        import pandas as pd

        rows = []
        for r in results:
            row = {
                "symbol": r.get("symbol"),
                "timeframe": r.get("timeframe"),
                "model": r.get("model"),
                "status": r.get("status"),
            }
            if r.get("status") == "ok":
                tr = (r.get("train") or {}).get("classification") or {}
                te = (r.get("test") or {}).get("classification") or {}

                row.update({
                    "train_accuracy": tr.get("accuracy"),
                    "train_f1_macro": tr.get("f1_macro"),
                    "test_accuracy": te.get("accuracy"),
                    "test_f1_macro": te.get("f1_macro"),
                })

                # optional trading
                trading = (r.get("test") or {}).get("trading")
                if isinstance(trading, dict):
                    row["trade_rate"] = trading.get("trade_rate")
                    row["sharpe_ratio"] = trading.get("sharpe_ratio")
                    row["max_drawdown"] = trading.get("max_drawdown")
                    row["cumulative_return"] = trading.get("cumulative_return")

            else:
                row["reason"] = r.get("reason") or r.get("error")

            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"okx_matrix_{stamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    except Exception as e:
        print(f"CSV export failed: {e}")


if __name__ == "__main__":
    main()
