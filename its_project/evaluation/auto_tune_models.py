"""
Auto-Tune Models Script

Directly imports integrated models and tunes their parameters by modifying model files.
Tests with real OKX data and iteratively improves until target metrics are achieved.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.model_evaluator import ModelEvaluator
from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets


def load_okx_data(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 2000) -> Optional[np.ndarray]:
    """Load real OHLCV data from OKX with pagination."""
    try:
        import time as time_module
        from backend.okx_rest_client import OKXRESTClient

        client = OKXRESTClient()
        connected = client.connect()
        if not connected:
            print("OKX REST client failed to connect")
            return None

        max_limit = 300
        ohlcv_data = []
        remaining = limit
        request_count = 0

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

        while remaining > 0:
            request_count += 1
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

        if ohlcv_data:
            ohlcv_data.sort(key=lambda x: x[0])
            data = np.array([[c[1], c[2], c[3], c[4], c[5]] for c in ohlcv_data])
            print(f"Loaded {len(data)} candles from OKX for {symbol} {timeframe}")
            return data
        else:
            print("Failed to load data from OKX")
            return None
    except Exception as e:
        print(f"Error loading OKX data: {e}")
        return None


def modify_ensemble_params(params: Dict[str, Any]) -> bool:
    """Modify ensemble.py parameters directly."""
    file_path = project_root / "models" / "ensemble.py"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Modify RandomForest parameters
        if 'rf_estimators' in params:
            content = re.sub(
                r'n_estimators=config\.get\("rf_estimators", \d+\)',
                f'n_estimators=config.get("rf_estimators", {params["rf_estimators"]})',
                content
            )
        if 'rf_max_depth' in params:
            content = re.sub(
                r'max_depth=config\.get\("rf_max_depth", \d+\)',
                f'max_depth=config.get("rf_max_depth", {params["rf_max_depth"]})',
                content
            )

        # Modify GradientBoosting parameters
        if 'gb_estimators' in params:
            content = re.sub(
                r'n_estimators=config\.get\("gb_estimators", \d+\)',
                f'n_estimators=config.get("gb_estimators", {params["gb_estimators"]})',
                content
            )
        if 'gb_max_depth' in params:
            content = re.sub(
                r'max_depth=config\.get\("gb_max_depth", \d+\)',
                f'max_depth=config.get("gb_max_depth", {params["gb_max_depth"]})',
                content
            )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Modified ensemble.py with params: {params}")
        return True
    except Exception as e:
        print(f"Error modifying ensemble.py: {e}")
        return False


def modify_boosting_params(params: Dict[str, Any]) -> bool:
    """Modify boosting_model.py parameters directly."""
    file_path = project_root / "models" / "boosting_model.py"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Modify GradientBoosting parameters
        if 'n_estimators' in params:
            content = re.sub(
                r'n_estimators=config\.get\("n_estimators", \d+\)',
                f'n_estimators=config.get("n_estimators", {params["n_estimators"]})',
                content
            )
        if 'max_depth' in params:
            content = re.sub(
                r'max_depth=config\.get\("max_depth", \d+\)',
                f'max_depth=config.get("max_depth", {params["max_depth"]})',
                content
            )
        if 'learning_rate' in params:
            content = re.sub(
                r'learning_rate=config\.get\("learning_rate", [\d.]+\)',
                f'learning_rate=config.get("learning_rate", {params["learning_rate"]})',
                content
            )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Modified boosting_model.py with params: {params}")
        return True
    except Exception as e:
        print(f"Error modifying boosting_model.py: {e}")
        return False


def evaluate_ma_crossover_with_rsi(ohlcv: np.ndarray, fast_period: int = 10, slow_period: int = 20, rsi_period: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70) -> Dict[str, Any]:
    """Evaluate moving average crossover with RSI filter strategy."""
    try:
        closes = ohlcv[:, 3]
        
        # Calculate moving averages
        def calculate_ma(data, period):
            ma = np.zeros(len(data))
            for i in range(period - 1, len(data)):
                ma[i] = np.mean(data[i - period + 1:i + 1])
            return ma
        
        fast_ma = calculate_ma(closes, fast_period)
        slow_ma = calculate_ma(closes, slow_period)
        
        # Calculate RSI
        def calculate_rsi(data, period):
            rsi = np.zeros(len(data))
            for i in range(period, len(data)):
                gains = []
                losses = []
                for j in range(i - period + 1, i + 1):
                    change = data[j] - data[j - 1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(closes, rsi_period)
        
        # Generate signals with RSI filter
        signals = np.zeros(len(closes))
        for i in range(1, len(closes)):
            ma_signal = 0
            if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                ma_signal = 1  # BUY signal (crossover up)
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                ma_signal = -1  # SELL signal (crossover down)
            
            # RSI filter: only BUY if RSI < overbought, only SELL if RSI > oversold
            if ma_signal == 1 and rsi[i] < rsi_overbought:
                signals[i] = 1
            elif ma_signal == -1 and rsi[i] > rsi_oversold:
                signals[i] = -1
        
        # Calculate returns
        returns = np.zeros(len(closes) - 1)
        for i in range(len(closes) - 1):
            returns[i] = (closes[i + 1] - closes[i]) / closes[i]
        
        # Trading: follow the signals
        trading_returns = np.zeros(len(returns))
        trades = 0
        position = 0  # 0 = out, 1 = long, -1 = short
        
        for i in range(1, len(returns)):
            if signals[i-1] == 1 and position != 1:
                position = 1
                trades += 1
            elif signals[i-1] == -1 and position != -1:
                position = -1
                trades += 1
            
            if position == 1:
                trading_returns[i] = returns[i]
            elif position == -1:
                trading_returns[i] = -returns[i]
        
        # Evaluate trading
        evaluator = ModelEvaluator()
        trading_metrics = evaluator.evaluate_trading_performance(trading_returns)
        
        return {
            "trading": trading_metrics,
            "trades": trades,
            "total": len(returns)
        }
    except Exception as e:
        print(f"Error evaluating MA crossover with RSI: {e}")
        import traceback
        traceback.print_exc()
        return {}


def auto_tune_ma_crossover_with_rsi(ohlcv: np.ndarray, max_iterations: int = 10) -> Dict[str, Any]:
    """Auto-tune MA crossover with RSI by tuning parameters."""
    print("\n" + "="*60)
    print("Auto-tuning MA Crossover with RSI Strategy")
    print("="*60)

    best_sharpe = -float('inf')
    best_params = {}
    best_metrics = {}

    # Use best MA params from previous test: fast=15, slow=40
    fast_period = 15
    slow_period = 40

    # RSI parameter grid
    rsi_oversold_list = [25, 30, 35]
    rsi_overbought_list = [65, 70, 75]

    iteration = 0
    for oversold in rsi_oversold_list:
        for overbought in rsi_overbought_list:
            iteration += 1
            if iteration > max_iterations:
                break

            print(f"\nIteration {iteration}/{max_iterations}")
            print(f"  RSI oversold={oversold}, overbought={overbought}")

            metrics = evaluate_ma_crossover_with_rsi(
                ohlcv, 
                fast_period=fast_period, 
                slow_period=slow_period,
                rsi_period=14,
                rsi_oversold=oversold,
                rsi_overbought=overbought
            )

            if not metrics:
                continue

            sharpe = metrics["trading"].get("sharpe_ratio", -float('inf'))
            profit_factor = metrics["trading"].get("profit_factor", 0)
            win_rate = metrics["trading"].get("win_rate", 0)

            print(f"  Sharpe: {sharpe:.4f}, Profit Factor: {profit_factor:.4f}, Win Rate: {win_rate:.4f}")
            print(f"  Trades: {metrics['trades']}/{metrics['total']}")

            # Check if best
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {"fast": fast_period, "slow": slow_period, "rsi_oversold": oversold, "rsi_overbought": overbought}
                best_metrics = metrics.copy()
                print(f"  *** NEW BEST ***")

    print("\n" + "="*60)
    print("Best MA Crossover with RSI Configuration:")
    print("="*60)
    print(f"Parameters: {best_params}")
    print(f"Sharpe Ratio: {best_metrics['trading'].get('sharpe_ratio', 0):.4f}")
    print(f"Profit Factor: {best_metrics['trading'].get('profit_factor', 0):.4f}")
    print(f"Win Rate: {best_metrics['trading'].get('win_rate', 0):.4f}")
    print(f"Trades: {best_metrics['trades']}/{best_metrics['total']}")

    return best_metrics


def auto_tune_momentum(ohlcv: np.ndarray, max_iterations: int = 5) -> Dict[str, Any]:
    """Auto-tune momentum strategy by tuning momentum window."""
    print("\n" + "="*60)
    print("Auto-tuning Momentum Strategy")
    print("="*60)

    best_sharpe = -float('inf')
    best_window = 10
    best_metrics = {}

    # Momentum window grid
    window_list = [5, 10, 15, 20, 25]

    iteration = 0
    for window in window_list:
        iteration += 1
        if iteration > max_iterations:
            break

        print(f"\nIteration {iteration}/{max_iterations}")
        print(f"  momentum_window={window}")

        metrics = evaluate_momentum_strategy(ohlcv, momentum_window=window)

        if not metrics:
            continue

        sharpe = metrics["trading"].get("sharpe_ratio", -float('inf'))
        profit_factor = metrics["trading"].get("profit_factor", 0)
        win_rate = metrics["trading"].get("win_rate", 0)

        print(f"  Sharpe: {sharpe:.4f}, Profit Factor: {profit_factor:.4f}, Win Rate: {win_rate:.4f}")
        print(f"  Trades: {metrics['trades']}/{metrics['total']}")

        # Check if best
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_window = window
            best_metrics = metrics.copy()
            print(f"  *** NEW BEST ***")

    print("\n" + "="*60)
    print("Best Momentum Configuration:")
    print("="*60)
    print(f"Best Window: {best_window}")
    print(f"Sharpe Ratio: {best_metrics['trading'].get('sharpe_ratio', 0):.4f}")
    print(f"Profit Factor: {best_metrics['trading'].get('profit_factor', 0):.4f}")
    print(f"Win Rate: {best_metrics['trading'].get('win_rate', 0):.4f}")
    print(f"Trades: {best_metrics['trades']}/{best_metrics['total']}")

    return best_metrics


def auto_tune_regression(ohlcv: np.ndarray, max_iterations: int = 10) -> Dict[str, Any]:
    """Auto-tune regression model by tuning prediction threshold."""
    print("\n" + "="*60)
    print("Auto-tuning Regression Model (Prediction Threshold)")
    print("="*60)

    best_sharpe = -float('inf')
    best_threshold = 0.0
    best_metrics = {}

    # Threshold grid (trade only if |predicted_return| >= threshold)
    threshold_list = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]

    iteration = 0
    for threshold in threshold_list:
        iteration += 1
        if iteration > max_iterations:
            break

        print(f"\nIteration {iteration}/{max_iterations}")
        print(f"  min_pred_return={threshold}")

        metrics = evaluate_regression(ohlcv, min_pred_return=threshold)

        if not metrics:
            continue

        sharpe = metrics["trading"].get("sharpe_ratio", -float('inf'))
        profit_factor = metrics["trading"].get("profit_factor", 0)
        win_rate = metrics["trading"].get("win_rate", 0)

        print(f"  Sharpe: {sharpe:.4f}, Profit Factor: {profit_factor:.4f}, Win Rate: {win_rate:.4f}")
        print(f"  Trades: {metrics['trades']}/{metrics['total']}")

        # Check if best
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold
            best_metrics = metrics.copy()
            print(f"  *** NEW BEST ***")

    print("\n" + "="*60)
    print("Best Regression Configuration:")
    print("="*60)
    print(f"Best Threshold: {best_threshold}")
    print(f"Sharpe Ratio: {best_metrics['trading'].get('sharpe_ratio', 0):.4f}")
    print(f"Profit Factor: {best_metrics['trading'].get('profit_factor', 0):.4f}")
    print(f"Win Rate: {best_metrics['trading'].get('win_rate', 0):.4f}")
    print(f"Trades: {best_metrics['trades']}/{best_metrics['total']}")

    return best_metrics


def evaluate_lstm(ohlcv: np.ndarray, prob_threshold: float = 0.6) -> Dict[str, Any]:
    """Evaluate LSTM model with current parameters."""
    try:
        # Force reload of modified module
        if 'models.lstm' in sys.modules:
            del sys.modules['models.lstm']
        if 'models.base' in sys.modules:
            del sys.modules['models.base']

        from models.lstm import LSTMModel

        # Build features
        feature_engineer = FeatureEngineer(n_lags=10, volatility_window=10, include_volume=True, include_indicators=True)
        X, y = feature_engineer.build_features_from_ohlcv(ohlcv, lookback=20)

        # Prepare classification targets with enhanced logic
        y_labels = prepare_classification_targets(
            y, 
            threshold=0.001, 
            use_three_classes=True,
            adaptive_threshold=True,
            volatility_window=20
        )

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X[:split_idx], X[split_idx:]
        y_train, y_eval = y_labels[:split_idx], y_labels[split_idx:]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        # Reshape for LSTM (samples, timesteps, features)
        # Use sliding window of 10 timesteps
        timesteps = 10
        def reshape_for_lstm(data, timesteps):
            X_reshaped = []
            for i in range(timesteps, len(data)):
                X_reshaped.append(data[i-timesteps:i])
            return np.array(X_reshaped)

        X_train_lstm = reshape_for_lstm(X_train_scaled, timesteps)
        X_eval_lstm = reshape_for_lstm(X_eval_scaled, timesteps)
        y_train_lstm = y_train[timesteps:]
        y_eval_lstm = y_eval[timesteps:]

        # Create and train model
        config = {"input_size": X.shape[1], "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "epochs": 10, "batch_size": 32}
        model = LSTMModel(config)
        model.fit(X_train_lstm, y_train_lstm)

        # Predict
        y_pred = model.predict(X_eval_lstm)
        y_proba = model.predict_proba(X_eval_lstm)

        # Evaluate classification
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classification(y_eval_lstm, y_pred, y_proba)

        # Evaluate trading (BUY and SELL)
        y_returns = y[split_idx + timesteps:]
        returns = np.zeros(len(y_returns))
        trades = 0

        for i in range(1, len(y_returns)):
            max_prob = np.max(y_proba[i-1])
            signal = np.argmax(y_proba[i-1])
            if max_prob >= prob_threshold:
                if signal == 2:  # BUY
                    returns[i] = y_returns[i]
                    trades += 1
                elif signal == 0:  # SELL
                    returns[i] = -y_returns[i]
                    trades += 1

        trading_metrics = evaluator.evaluate_trading_performance(returns)

        return {
            "classification": metrics,
            "trading": trading_metrics,
            "trades": trades,
            "total": len(y_returns)
        }
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        import traceback
        traceback.print_exc()
        return {}


def auto_tune_lstm(ohlcv: np.ndarray, max_iterations: int = 5) -> Dict[str, Any]:
    """Auto-tune LSTM model by tuning probability threshold."""
    print("\n" + "="*60)
    print("Auto-tuning LSTM Model (Probability Threshold)")
    print("="*60)

    best_sharpe = -float('inf')
    best_threshold = 0.0
    best_metrics = {}

    # Threshold grid
    threshold_list = [0.5, 0.55, 0.6, 0.65, 0.7]

    iteration = 0
    for threshold in threshold_list:
        iteration += 1
        if iteration > max_iterations:
            break

        print(f"\nIteration {iteration}/{max_iterations}")
        print(f"  prob_threshold={threshold}")

        metrics = evaluate_lstm(ohlcv, prob_threshold=threshold)

        if not metrics:
            continue

        sharpe = metrics["trading"].get("sharpe_ratio", -float('inf'))
        profit_factor = metrics["trading"].get("profit_factor", 0)
        win_rate = metrics["trading"].get("win_rate", 0)

        print(f"  Sharpe: {sharpe:.4f}, Profit Factor: {profit_factor:.4f}, Win Rate: {win_rate:.4f}")
        print(f"  Trades: {metrics['trades']}/{metrics['total']}")

        # Check if best
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold
            best_metrics = metrics.copy()
            print(f"  *** NEW BEST ***")

    print("\n" + "="*60)
    print("Best LSTM Configuration:")
    print("="*60)
    print(f"Best Threshold: {best_threshold}")
    print(f"Sharpe Ratio: {best_metrics['trading'].get('sharpe_ratio', 0):.4f}")
    print(f"Profit Factor: {best_metrics['trading'].get('profit_factor', 0):.4f}")
    print(f"Win Rate: {best_metrics['trading'].get('win_rate', 0):.4f}")
    print(f"Trades: {best_metrics['trades']}/{best_metrics['total']}")

    return best_metrics


def evaluate_ensemble(ohlcv: np.ndarray, prob_threshold: float = 0.55) -> Dict[str, Any]:
    """Evaluate ensemble model with current parameters."""
    try:
        # Force reload of modified module
        if 'models.ensemble' in sys.modules:
            del sys.modules['models.ensemble']
        if 'models.base' in sys.modules:
            del sys.modules['models.base']

        from models.ensemble import EnsembleModel

        # Build features
        feature_engineer = FeatureEngineer(n_lags=10, volatility_window=10, include_volume=True, include_indicators=True)
        X, y = feature_engineer.build_features_from_ohlcv(ohlcv, lookback=20)

        # Prepare classification targets with enhanced logic
        y_labels = prepare_classification_targets(
            y, 
            threshold=0.001, 
            use_three_classes=True,
            adaptive_threshold=True,
            volatility_window=20
        )

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X[:split_idx], X[split_idx:]
        y_train, y_eval = y_labels[:split_idx], y_labels[split_idx:]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        # Create and train model
        config = {"rf_estimators": 100, "rf_max_depth": 10, "gb_estimators": 100, "gb_max_depth": 5}
        model = EnsembleModel(config)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_eval_scaled)
        y_proba = model.predict_proba(X_eval_scaled)

        # Evaluate classification
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classification(y_eval, y_pred, y_proba)

        # Evaluate trading (BUY only)
        y_returns = y[split_idx:]
        returns = np.zeros(len(y_returns))
        trades = 0

        for i in range(1, len(y_returns)):
            max_prob = np.max(y_proba[i-1])
            signal = np.argmax(y_proba[i-1])
            if max_prob >= prob_threshold:
                if signal == 2:  # BUY
                    returns[i] = y_returns[i]
                    trades += 1
                elif signal == 0:  # SELL
                    returns[i] = -y_returns[i]
                    trades += 1

        trading_metrics = evaluator.evaluate_trading_performance(returns)

        return {
            "classification": metrics,
            "trading": trading_metrics,
            "trades": trades,
            "total": len(y_returns)
        }
    except Exception as e:
        print(f"Error evaluating ensemble: {e}")
        import traceback
        traceback.print_exc()
        return {}


def auto_tune_ensemble(ohlcv: np.ndarray, max_iterations: int = 10) -> Dict[str, Any]:
    """Auto-tune ensemble model by modifying parameters."""
    print("\n" + "="*60)
    print("Auto-tuning Ensemble Model")
    print("="*60)

    best_sharpe = -float('inf')
    best_params = {}
    best_metrics = {}

    # Parameter grid
    rf_estimators_list = [100, 200, 300]
    rf_max_depth_list = [10, 15, 20]
    gb_estimators_list = [100, 200, 300]
    gb_max_depth_list = [5, 8, 10]

    iteration = 0
    for rf_est in rf_estimators_list:
        for rf_depth in rf_max_depth_list:
            for gb_est in gb_estimators_list:
                for gb_depth in gb_max_depth_list:
                    iteration += 1
                    if iteration > max_iterations:
                        break

                    print(f"\nIteration {iteration}/{max_iterations}")
                    print(f"  RF: estimators={rf_est}, depth={rf_depth}")
                    print(f"  GB: estimators={gb_est}, depth={gb_depth}")

                    # Modify parameters
                    params = {
                        "rf_estimators": rf_est,
                        "rf_max_depth": rf_depth,
                        "gb_estimators": gb_est,
                        "gb_max_depth": gb_depth
                    }
                    modify_ensemble_params(params)

                    # Evaluate
                    metrics = evaluate_ensemble(ohlcv, prob_threshold=0.55)

                    if not metrics:
                        continue

                    sharpe = metrics["trading"].get("sharpe_ratio", -float('inf'))
                    profit_factor = metrics["trading"].get("profit_factor", 0)
                    win_rate = metrics["trading"].get("win_rate", 0)

                    print(f"  Sharpe: {sharpe:.4f}, Profit Factor: {profit_factor:.4f}, Win Rate: {win_rate:.4f}")
                    print(f"  Trades: {metrics['trades']}/{metrics['total']}")

                    # Check if best
                    if sharpe > best_sharpe and profit_factor > 1.0:
                        best_sharpe = sharpe
                        best_params = params.copy()
                        best_metrics = metrics.copy()
                        print(f"  *** NEW BEST ***")

    print("\n" + "="*60)
    print("Best Ensemble Configuration:")
    print("="*60)
    print(f"Parameters: {best_params}")
    print(f"Sharpe Ratio: {best_metrics['trading'].get('sharpe_ratio', 0):.4f}")
    print(f"Profit Factor: {best_metrics['trading'].get('profit_factor', 0):.4f}")
    print(f"Win Rate: {best_metrics['trading'].get('win_rate', 0):.4f}")
    print(f"Trades: {best_metrics['trades']}/{best_metrics['total']}")

    # Apply best parameters
    if best_params:
        modify_ensemble_params(best_params)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Auto-tune integrated models")
    parser.add_argument("--model", type=str, default="ma_crossover_rsi", choices=["ensemble", "boosting", "regression", "lstm", "momentum", "ma_crossover", "ma_crossover_rsi"])
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--max_iterations", type=int, default=10)
    args = parser.parse_args()

    # Load data
    print(f"Loading real data from OKX: {args.symbol} {args.timeframe}...")
    ohlcv = load_okx_data(args.symbol, args.timeframe, args.n_samples)
    if ohlcv is None:
        print("Failed to load OKX data. Exiting.")
        return

    # Auto-tune
    if args.model == "ensemble":
        auto_tune_ensemble(ohlcv, args.max_iterations)
    elif args.model == "boosting":
        print("Boosting auto-tuning not implemented yet")
    elif args.model == "regression":
        auto_tune_regression(ohlcv, args.max_iterations)
    elif args.model == "lstm":
        auto_tune_lstm(ohlcv, args.max_iterations)
    elif args.model == "momentum":
        auto_tune_momentum(ohlcv, args.max_iterations)
    elif args.model == "ma_crossover":
        auto_tune_ma_crossover(ohlcv, args.max_iterations)
    elif args.model == "ma_crossover_rsi":
        auto_tune_ma_crossover_with_rsi(ohlcv, args.max_iterations)


if __name__ == "__main__":
    main()
