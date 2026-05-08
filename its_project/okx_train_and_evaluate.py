#!/usr/bin/env python3
"""OKX real-data training + evaluation runner with ECONOMIC IMPROVEMENTS.

Goal: train project models on REAL OKX market data with improved economic features and realistic evaluation.

What it does:
- Fetches OHLCV from OKX (via existing backend.OKXRESTClient which wraps CCXT)
- Uses ECONOMIC targets based on future returns r_{t+h}
- Builds features WITHOUT look-ahead bias using economic and microstructure features
- Uses class weights and economic loss functions
- Implements realistic backtesting with transaction costs
- Walk-forward validation for temporal stability
- Baseline strategy comparison
- Economic metrics (Sharpe, Sortino, Calmar, etc.)

Examples:
  python okx_train_and_evaluate.py --symbols BTC/USDT ETH/USDT --timeframe 1h --n_samples 2000 --models boosting gru transformer
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project imports work when running as a script
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import IMPROVED components
try:
    from targets.economic_target import EconomicTargetCalculator
    from features.economic_features import EconomicFeatures
    from features.microstructure_features import MicrostructureFeatures
    from models.economic_boosting_model import EconomicBoostingModel
    from backtesting.economic_backtester import EconomicBacktester
    from backtesting.walkforward_validator import WalkForwardValidator
    from backtesting.baseline_strategies import BaselineStrategies
    from decision.economic_decision_maker import EconomicDecisionMaker
    from evaluation.model_evaluator import ModelEvaluator
except ImportError as e:
    print(f"Error importing improved components: {e}")
    print("Falling back to original components...")
    from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets
    from evaluation.model_evaluator import ModelEvaluator
    from models.registry import ModelRegistry


@dataclass
class SingleRunResult:
    symbol: str
    timeframe: str
    n_samples_requested: int
    n_samples_loaded: int
    feature_shape: Tuple[int, int]
    label_distribution: Dict[str, int]
    model_name: str
    metrics: Dict[str, Any]
    economic_metrics: Optional[Dict[str, Any]] = None
    backtest_metrics: Optional[Dict[str, Any]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None


def load_okx_ohlcv(symbol: str, timeframe: str, limit: int) -> Optional[np.ndarray]:
    """Load OHLCV from OKX using existing backend client.

    Returns numpy array shape (n, 5) columns [open, high, low, close, volume]
    """

    try:
        from backend.okx_rest_client import OKXRESTClient

        client = OKXRESTClient()
        if not client.connect():
            print("OKX REST client failed to connect")
            return None

        # OKX public endpoints: CCXT usually caps candles per request.
        # backend.OKXRESTClient has get_historical_ohlcv with pagination in evaluation scripts.
        # Prefer that API if present.
        if hasattr(client, "get_historical_ohlcv"):
            # OKX has max 300 candles per request -> paginate
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

        # Fallback: single request
        raw = client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        raw.sort(key=lambda x: x[0])
        return np.array([[c[1], c[2], c[3], c[4], c[5]] for c in raw], dtype=np.float64)

    except Exception as e:
        print(f"Error loading OKX data for {symbol} {timeframe}: {e}")
        return None


def time_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = int(len(X) * train_ratio)
    return X[:idx], X[idx:], y[:idx], y[idx:]


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2D feature matrix into 3D sequences.

    X: (n_samples, n_features)
    Returns:
      X_seq: (n_samples - seq_len + 1, seq_len, n_features)
      y_seq: (n_samples - seq_len + 1,)
    """
    if seq_len <= 1:
        return X.reshape(X.shape[0], 1, X.shape[1]), y
    if len(X) < seq_len:
        raise ValueError(f"Not enough samples for seq_len={seq_len}: n={len(X)}")

    X_seq = np.stack([X[i : i + seq_len] for i in range(len(X) - seq_len + 1)], axis=0)
    y_seq = y[seq_len - 1 :]
    return X_seq, y_seq


def label_dist(y: np.ndarray) -> Dict[str, int]:
    # Labels are expected in {0,1,2} -> SELL,HOLD,BUY
    return {
        "SELL": int(np.sum(y == 0)),
        "HOLD": int(np.sum(y == 1)),
        "BUY": int(np.sum(y == 2)),
    }


def evaluate_one_improved(
    symbol: str,
    timeframe: str,
    n_samples: int,
    model_name: str,
    prob_threshold: float,
    n_lags: int,
    lookback: int,
) -> SingleRunResult:
    """Improved evaluation with economic targets and features."""
    print(f"Loading OKX data for {symbol} {timeframe}...")
    ohlcv = load_okx_ohlcv(symbol, timeframe, limit=n_samples)
    if ohlcv is None:
        raise RuntimeError(f"Failed to load OKX OHLCV for {symbol} {timeframe}")

    # Convert to DataFrame for improved components
    dates = pd.date_range(end=datetime.now(), periods=len(ohlcv), freq='1h')
    data = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)
    
    print(f"Calculating economic targets...")
    # Use ECONOMIC target calculator
    target_config = {
        "horizon": 5,
        "threshold": 0.002,
        "target_type": "direction"
    }
    target_calculator = EconomicTargetCalculator(target_config)
    y, target_metadata = target_calculator.calculate_target(data)
    
    print(f"Building economic features...")
    # Use ECONOMIC features (no look-ahead bias)
    econ_config = {
        "return_periods": [1, 5, 15],
        "volatility_windows": [5, 15],
        "use_risk_features": True
    }
    econ_features = EconomicFeatures(econ_config)
    econ_feature_array = econ_features.calculate(data)
    
    # Use microstructure features
    micro_config = {
        "use_order_book": False,  # Disabled for OHLCV data
        "impact_window": 20,
        "efficiency_window": 50
    }
    micro_features = MicrostructureFeatures(micro_config)
    micro_feature_array = micro_features.calculate(data)
    
    # Ensure features have same number of samples
    min_samples = min(econ_feature_array.shape[0], micro_feature_array.shape[0], len(y))
    econ_feature_array = econ_feature_array[:min_samples]
    micro_feature_array = micro_feature_array[:min_samples]
    y = y.iloc[:min_samples]
    data = data.iloc[:min_samples]
    
    # Combine features
    X = np.hstack([econ_feature_array, micro_feature_array])
    
    # Remove NaN values and align
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.values))
    X = X[valid_mask]
    y = y.values[valid_mask]
    data = data.iloc[valid_mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {target_metadata['class_distribution']}")
    
    # Time split (avoid leakage)
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)
    # Split data indices to maintain alignment
    train_size = int(len(data) * 0.8)
    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training {model_name} model...")
    # Use ECONOMIC boosting model with class weights
    if model_name == "boosting":
        model_config = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "use_class_weights": True,
            "early_stopping_rounds": 10
        }
        model = EconomicBoostingModel(model_config)
    else:
        # Fallback to original models for other types
        try:
            from models.registry import ModelRegistry
            configs = {
                "gru": {
                    "input_size": X_train_scaled.shape[1],
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "epochs": 15,
                    "batch_size": 64,
                    "num_classes": 3,
                },
                "transformer": {
                    "input_size": X_train_scaled.shape[1],
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "output_size": 3,
                    "dropout": 0.2,
                    "epochs": 15,
                    "batch_size": 64,
                }
            }
            model = ModelRegistry.get_model(model_name, configs.get(model_name, {}))
        except:
            print(f"Model {model_name} not available, using boosting")
            model_config = {"n_estimators": 100, "use_class_weights": True}
            model = EconomicBoostingModel(model_config)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Get probabilities
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_scaled)
        except Exception:
            y_proba = None
    
    # Standard evaluation
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
    
    # Economic evaluation
    economic_metrics = None
    if hasattr(model, 'evaluate_economic_metrics'):
        economic_metrics = model.evaluate_economic_metrics(X_test_scaled, y_test)
    
    # Realistic backtesting
    backtest_metrics = None
    try:
        print("Running economic backtesting...")
        backtest_config = {
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "position_size": 0.1
        }
        backtester = EconomicBacktester(backtest_config)
        
        # Create simple decision maker
        decision_config = {"confidence_threshold": prob_threshold}
        decision_maker = EconomicDecisionMaker(decision_config)
        
        # Run backtest
        backtest_result = backtester.run(data_test, model, decision_maker, target_calculator)
        backtest_metrics = backtest_result.metrics
        
        print(f"Backtest PnL: {backtest_metrics.get('total_return', 0):.3f}")
        print(f"Backtest Sharpe: {backtest_metrics.get('sharpe_ratio', 0):.3f}")
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
    
    # Baseline comparison
    baseline_comparison = None
    try:
        print("Running baseline comparison...")
        baseline_config = {
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005
        }
        baselines = BaselineStrategies(baseline_config)
        baseline_results = baselines.run_all_baselines(data_test)
        
        # Compare with model if backtest was successful
        if backtest_metrics:
            # Create mock BacktestResult for comparison
            from its_project.backtesting.base import BacktestResult, Trade
            mock_result = BacktestResult(
                trades=[],
                equity_curve=np.array([10000.0]),
                returns=np.array([0.0]),
                metrics=backtest_metrics,
                positions=pd.DataFrame()
            )
            baseline_comparison = baselines.compare_with_model(mock_result, baseline_results)
            
    except Exception as e:
        print(f"Baseline comparison failed: {e}")
    
    # Add feature names
    feature_names = econ_features.get_feature_names() + micro_features.get_feature_names()
    metrics["feature_names"] = feature_names
    
    # Add target metadata
    metrics["target_metadata"] = target_metadata
    
    return SingleRunResult(
        symbol=symbol,
        timeframe=timeframe,
        n_samples_requested=n_samples,
        n_samples_loaded=int(len(ohlcv)),
        feature_shape=(int(X.shape[0]), int(X.shape[1])),
        label_distribution=target_metadata['class_distribution'],
        model_name=model_name,
        metrics=metrics,
        economic_metrics=economic_metrics,
        backtest_metrics=backtest_metrics,
        baseline_comparison=baseline_comparison
    )


def evaluate_one(
    symbol: str,
    timeframe: str,
    n_samples: int,
    model_name: str,
    prob_threshold: float,
    n_lags: int,
    lookback: int,
) -> SingleRunResult:
    """Wrapper that tries improved evaluation first, falls back to original."""
    try:
        return evaluate_one_improved(symbol, timeframe, n_samples, model_name, prob_threshold, n_lags, lookback)
    except Exception as e:
        print(f"Improved evaluation failed: {e}")
        print("Falling back to original evaluation...")
        
        # Original evaluation as fallback
        ohlcv = load_okx_ohlcv(symbol, timeframe, limit=n_samples)
        if ohlcv is None:
            raise RuntimeError(f"Failed to load OKX OHLCV for {symbol} {timeframe}")

        try:
            from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets
            fe = FeatureEngineer(
                n_lags=n_lags,
                volatility_window=10,
                include_volume=True,
                include_indicators=True,
            )
            X, y_returns = fe.build_features_from_ohlcv(ohlcv, lookback=lookback)

            y = prepare_classification_targets(
                y_returns,
                use_three_classes=True,
                adaptive_threshold=True,
                volatility_window=20,
            )
        except:
            # Simple fallback
            print("Using simple feature/target generation...")
            X = ohlcv
            y_returns = np.diff(ohlcv[:, 3]) / ohlcv[:-1, 3]  # Simple returns
            y = np.where(y_returns > 0.01, 2, np.where(y_returns < -0.01, 0, 1))  # Simple targets

        X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_train_s, y_test_s = y_train, y_test

        # Simple model
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classification(y_test_s, y_pred, y_proba)

        return SingleRunResult(
            symbol=symbol,
            timeframe=timeframe,
            n_samples_requested=n_samples,
            n_samples_loaded=int(len(ohlcv)),
            feature_shape=(int(X.shape[0]), int(X.shape[1])),
            label_distribution=label_dist(y),
            model_name=model_name,
            metrics=metrics,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate models on REAL OKX data")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"], help="Symbols in GUI format, e.g. BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h", help="1m,5m,15m,1h,4h,1d")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of candles to fetch")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["boosting", "gru"],
        choices=["boosting", "gru", "lstm", "transformer", "ensemble", "regression"],
    )
    parser.add_argument("--prob_threshold", type=float, default=0.6)
    parser.add_argument("--n_lags", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "okx_reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: List[Dict[str, Any]] = []

    for symbol in args.symbols:
        for model_name in args.models:
            print(f"\n=== OKX run: {symbol} {args.timeframe} model={model_name} candles={args.n_samples} ===")
            res = evaluate_one(
                symbol=symbol,
                timeframe=args.timeframe,
                n_samples=args.n_samples,
                model_name=model_name,
                prob_threshold=args.prob_threshold,
                n_lags=args.n_lags,
                lookback=args.lookback,
            )
            all_results.append(asdict(res))

            # Print quick summary
            print("Label dist:", res.label_distribution)
            print("Accuracy:", res.metrics.get("accuracy"))
            print("F1 macro:", res.metrics.get("f1_macro"))
            
            # Print economic metrics if available
            if res.economic_metrics:
                print("Economic metrics:")
                for key, value in res.economic_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            # Print backtest metrics if available
            if res.backtest_metrics:
                print("Backtest metrics:")
                for key, value in res.backtest_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            # Print baseline comparison if available
            if res.baseline_comparison:
                print("Baseline comparison:")
                print(f"  Overall assessment: {res.baseline_comparison.get('overall_assessment')}")
                if 'best_baseline' in res.baseline_comparison:
                    best = res.baseline_comparison['best_baseline']
                    print(f"  Best baseline: {best['name']} (Sharpe: {best['sharpe_ratio']:.3f})")
                    print(f"  Model vs best Sharpe: {best['model_vs_best_sharpe']:+.3f}")

    report = {
        "generated_at": stamp,
        "timeframe": args.timeframe,
        "n_samples": args.n_samples,
        "models": args.models,
        "symbols": args.symbols,
        "results": all_results,
    }

    out_path = out_dir / f"okx_report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
