"""
Model Evaluation Script

Run comprehensive evaluation of trained models with metrics and visualization.

Usage:
    python evaluation/evaluate_models.py --model regression --data_path data/ohlcv.csv
    python evaluation/evaluate_models.py --model ensemble --plot --save_dir results/
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.model_evaluator import ModelEvaluator
from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets
from models.registry import ModelRegistry


def load_ohlcv_data(data_path: str) -> Optional[np.ndarray]:
    """
    Load OHLCV data from CSV file.

    Expected format: timestamp, open, high, low, close, volume
    """
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        # Ensure columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain columns: {required_cols}")
            return None
        return df[required_cols].values
    except ImportError:
        print("Error: pandas not available. Install with: pip install pandas")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_okx_data(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1000) -> Optional[np.ndarray]:
    """
    Load real OHLCV data from OKX with pagination for large requests.

    Args:
        symbol: Trading pair
        timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch

    Returns:
        OHLCV array or None if error
    """
    try:
        import sys
        from pathlib import Path
        import time

        # Add project root to path (same as GUI does)
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend.okx_rest_client import OKXRESTClient

        client = OKXRESTClient()
        connected = client.connect()
        if not connected:
            print("OKX REST client failed to connect")
            return None

        # OKX has max 300 candles per request for public data
        # Need to use pagination with since parameter
        max_limit = 300
        ohlcv_data = []
        remaining = limit
        request_count = 0

        # Calculate start timestamp: current time - (limit * timeframe_ms)
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

        print(f"Requesting {limit} candles from OKX ({timeframe})...")
        print(f"  Start timestamp: {since} ({time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(since/1000))})")

        while remaining > 0:
            request_count += 1
            current_limit = min(max_limit, remaining)
            print(f"  Request {request_count}: limit={current_limit}, since={since}")

            chunk = client.get_historical_ohlcv(symbol, timeframe, since=since, limit=current_limit)
            print(f"  Received {len(chunk) if chunk else 0} candles")

            if not chunk:
                print("  No more data available")
                break

            # Deduplicate by timestamp
            existing_ts = set(c[0] for c in ohlcv_data)
            for c in chunk:
                if c[0] not in existing_ts:
                    ohlcv_data.append(c)
                    existing_ts.add(c[0])

            remaining -= len(chunk)
            print(f"  Total collected: {len(ohlcv_data)}, remaining: {remaining}")

            if len(chunk) < current_limit:
                # No more data available
                print("  Fewer candles returned than requested - end of data")
                break

            # Update since for next request (last timestamp + 1ms)
            since = int(chunk[-1][0]) + 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        if ohlcv_data:
            # Sort by timestamp
            ohlcv_data.sort(key=lambda x: x[0])

            # Convert to numpy array [ts, o, h, l, c, v] -> [o, h, l, c, v]
            data = np.array([[c[1], c[2], c[3], c[4], c[5]] for c in ohlcv_data])
            print(f"Loaded {len(data)} candles from OKX for {symbol} {timeframe}")
            return data
        else:
            print("Failed to load data from OKX")
            return None
    except ImportError as e:
        print(f"OKX client not available: {e}. Using synthetic data.")
        return None
    except Exception as e:
        print(f"Error loading OKX data: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    plot: bool = True,
    save_dir: Optional[str] = None,
    prob_threshold: float = 0.65
) -> None:
    """
    Evaluate a single model with comprehensive metrics.

    Args:
        model_name: Name of the model to evaluate
        model_config: Configuration for the model
        X_test: Test features
        y_test: Test targets (returns)
        plot: Whether to generate plots
        save_dir: Directory to save plots
        prob_threshold: Probability threshold for trading signals
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*60}")

    # Determine if regression or classification
    is_regression = model_name.lower() in {"regression"}

    # Prepare targets based on model type
    if is_regression:
        y_train = y_test
    else:
        # Convert returns to classification labels for classifiers
        y_train = prepare_classification_targets(y_test, use_three_classes=True)

    # Get model from registry
    try:
        model = ModelRegistry.get_model(model_name, model_config)
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # Use TimeSeriesSplit for time series validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Train on train split (use first 80% for training, last 20% for testing)
    split_idx = int(len(X_test) * 0.8)
    X_train_split, X_eval = X_test[:split_idx], X_test[split_idx:]
    y_train_split, y_eval = y_train[:split_idx], y_train[split_idx:]

    # Scale features for models that benefit from it (LogisticRegression, etc.)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_eval_scaled = scaler.transform(X_eval)

    print(f"Training on {len(X_train_split)} samples...")
    print(f"Validating on {len(X_eval)} samples...")

    try:
        model.fit(X_train_scaled, y_train_split)
        print("Training completed.")
    except Exception as e:
        print(f"Error training model: {e}")
        return

    # Make predictions
    try:
        y_pred = model.predict(X_eval_scaled)
    except Exception as e:
        print(f"Error predicting: {e}")
        return

    # Get probabilities if available
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_eval_scaled)
            print(f"Probabilities shape: {y_proba.shape}")
        except Exception as e:
            print(f"Error getting probabilities: {e}")
            import traceback
            traceback.print_exc()

    # Initialize evaluator
    evaluator = ModelEvaluator()

    if is_regression:
        # Regression evaluation
        metrics = evaluator.evaluate_regression(y_eval, y_pred)
        evaluator.print_metrics_report(metrics, model_name)

        if plot:
            fig = evaluator.plot_regression_metrics(y_eval, y_pred)
            if fig is not None:
                if save_dir:
                    save_path = Path(save_dir) / f"{model_name}_regression_metrics.png"
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"Saved plot to {save_path}")
                else:
                    import matplotlib.pyplot as plt
                    plt.show()
    else:
        # Classification evaluation
        metrics = evaluator.evaluate_classification(y_eval, y_pred, y_proba)
        evaluator.print_metrics_report(metrics, model_name)

        if plot:
            class_names = ["SELL", "HOLD", "BUY"]
            fig = evaluator.plot_classification_metrics(
                y_eval, y_pred, y_proba, class_names
            )
            if fig is not None:
                if save_dir:
                    save_path = Path(save_dir) / f"{model_name}_classification_metrics.png"
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"Saved plot to {save_path}")
                else:
                    import matplotlib.pyplot as plt
                    plt.show()

    # Trading performance evaluation with probability filter
    if not is_regression and y_proba is not None:
        # Use original returns for trading simulation
        split_idx = int(len(X_test) * 0.8)
        y_returns = y_test[split_idx:]  # Original returns for trading

        # Strategy with probability filter: trade only if prob > threshold
        returns = np.zeros(len(y_returns))
        trades = 0

        for i in range(1, len(y_returns)):
            max_prob = np.max(y_proba[i-1])
            signal = np.argmax(y_proba[i-1])

            # Only trade if confidence is high enough
            if max_prob >= prob_threshold:
                if signal == 2:  # BUY only (ignore SELL)
                    returns[i] = y_returns[i]
                    trades += 1
                else:  # SELL or HOLD
                    returns[i] = 0
            else:
                returns[i] = 0  # No trade

        print(f"\nTrading Strategy (prob threshold = {prob_threshold}):")
        print(f"  Trades executed: {trades}/{len(y_returns)} ({trades/len(y_returns)*100:.1f}%)")

        trading_metrics = evaluator.evaluate_trading_performance(returns)
        evaluator.print_metrics_report(trading_metrics, f"{model_name} (Trading, prob>={prob_threshold})")

        if plot:
            fig = evaluator.plot_equity_curve(returns)
            if fig is not None:
                if save_dir:
                    save_path = Path(save_dir) / f"{model_name}_equity_curve_prob{prob_threshold}.png"
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"Saved equity curve to {save_path}")
                else:
                    import matplotlib.pyplot as plt
                    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trading models")
    parser.add_argument(
        "--model",
        type=str,
        default="regression",
        choices=["regression", "ensemble", "boosting", "lstm", "transformer", "gru"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to OHLCV CSV file"
    )
    parser.add_argument(
        "--use_okx",
        action="store_true",
        help="Load real data from OKX API"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol for OKX data"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe for OKX data"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save plots"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to load (OKX candles or synthetic)"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.6,
        help="Probability threshold for trading signals"
    )
    parser.add_argument(
        "--n_lags",
        type=int,
        default=10,
        help="Number of price lags for features"
    )

    args = parser.parse_args()

    # Create save directory if specified
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if args.use_okx:
        print(f"Loading real data from OKX: {args.symbol} {args.timeframe}...")
        ohlcv = load_okx_data(args.symbol, args.timeframe, limit=args.n_samples)
        if ohlcv is None:
            print("Failed to load OKX data. Falling back to synthetic data.")
            args.use_okx = False
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        ohlcv = load_ohlcv_data(args.data_path)
        if ohlcv is None:
            print("Failed to load data. Exiting.")
            return
    else:
        print(f"Generating synthetic data ({args.n_samples} samples)...")
        args.use_okx = False

    if not args.use_okx and not args.data_path:
        # Generate synthetic OHLCV data
        np.random.seed(42)
        n = args.n_samples
        base_price = 42000.0
        prices = base_price * np.cumprod(1 + np.random.randn(n) * 0.001)
        ohlcv = np.zeros((n, 5))
        for i in range(n):
            c = prices[i]
            o = c * (1 + np.random.randn() * 0.0005)
            h = max(o, c) * (1 + abs(np.random.randn()) * 0.001)
            l = min(o, c) * (1 - abs(np.random.randn()) * 0.001)
            v = np.random.rand() * 1000
            ohlcv[i] = [o, h, l, c, v]

    # Build enhanced features and targets
    print("Building enhanced features...")
    feature_engineer = FeatureEngineer(
        n_lags=args.n_lags,
        volatility_window=10,
        include_volume=True,
        include_indicators=True
    )
    X, y = feature_engineer.build_features_from_ohlcv(ohlcv, lookback=20)
    print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
    print(f"Feature names: {feature_engineer.get_feature_names()[:10]}...")

    # Model configurations with class_weight for imbalance
    model_configs = {
        "regression": {
            "model_type": "gradient_boosting",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        },
        "ensemble": {
            "rf_estimators": 300,
            "rf_max_depth": 20,
            "gb_estimators": 300,
            "gb_max_depth": 10
        },
        "boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "class_weight": "balanced"  # Handle class imbalance
        },
        "lstm": {
            "input_size": X.shape[1],
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 3,
            "dropout": 0.2,
            "epochs": 10,
            "batch_size": 32
        },
        "transformer": {
            "input_size": X.shape[1],
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "output_size": 3,
            "dropout": 0.2,
            "epochs": 10,
            "batch_size": 32
        },
        "gru": {
            "input_size": X.shape[1],
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 3,
            "dropout": 0.2,
            "epochs": 10,
            "batch_size": 32
        }
    }

    # Evaluate selected model
    config = model_configs.get(args.model, {})
    evaluate_model(
        args.model,
        config,
        X,
        y,
        plot=args.plot,
        save_dir=args.save_dir,
        prob_threshold=args.prob_threshold
    )


if __name__ == "__main__":
    main()
