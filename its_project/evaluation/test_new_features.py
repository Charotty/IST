"""
Test script for new features (Order Book + On-chain + Microstructure)
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.feature_engineering import FeatureEngineer, prepare_classification_targets
from evaluation.model_evaluator import ModelEvaluator
from models.ensemble import EnsembleModel
from models.regression_model import RegressionModel
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def load_okx_data_with_orderbook(symbol: str = "BTC/USDT", timeframe: str = "15m", limit: int = 500) -> tuple:
    """Load OHLCV and orderbook data from OKX."""
    try:
        import time as time_module
        from backend.okx_rest_client import OKXRESTClient

        client = OKXRESTClient()
        connected = client.connect()
        if not connected:
            print("OKX REST client failed to connect")
            return None, None

        # Load OHLCV
        max_limit = 300
        ohlcv_data = []
        remaining = limit
        request_count = 0

        timeframe_ms = {
            '15m': 15 * 60 * 1000,
        }.get(timeframe, 15 * 60 * 1000)

        current_time = int(time_module.time() * 1000)
        since = current_time - (limit * timeframe_ms)

        print(f"Loading {limit} OHLCV candles from OKX...")

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
            ohlcv = np.array([[c[1], c[2], c[3], c[4], c[5]] for c in ohlcv_data])
            print(f"Loaded {len(ohlcv)} OHLCV candles")
        else:
            return None, None

        # Load orderbook for each candle (sample every 5th for better density)
        print(f"Loading orderbook data (sampling every 5th candle)...")
        orderbooks = []
        for i in range(0, len(ohlcv), 5):
            try:
                ob = client.get_orderbook(symbol, limit=10)
                orderbooks.append(ob)
                time_module.sleep(0.05)  # Rate limiting
            except Exception as e:
                print(f"Error loading orderbook at index {i}: {e}")
                orderbooks.append({})

        print(f"Loaded {len(orderbooks)} orderbook snapshots")

        return ohlcv, orderbooks

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_ensemble_with_new_features(ohlcv: np.ndarray, orderbooks: List[Dict]):
    """Test ensemble model with new features."""
    print("\n" + "="*60)
    print("Testing Ensemble Model with New Features")
    print("="*60)

    # Build features WITH orderbook
    feature_engineer = FeatureEngineer(
        n_lags=10,
        volatility_window=10,
        include_volume=True,
        include_indicators=True,
        include_orderbook=True
    )

    # Pad orderbooks to match OHLCV length
    padded_orderbooks = [{} for _ in range(len(ohlcv))]
    for i, ob in enumerate(orderbooks):
        padded_orderbooks[i * 5] = ob  # Place at sampled positions

    X, y = feature_engineer.build_features_with_extras(
        ohlcv,
        orderbooks=padded_orderbooks,
        onchain_data=None,
        lookback=20
    )

    print(f"Features shape: {X.shape}")
    print(f"Feature names: {feature_engineer.get_feature_names()[:10]}...")

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Apply SMOTE for class imbalance
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

    # Train ensemble on resampled data
    config = {"rf_estimators": 100, "rf_max_depth": 10, "gb_estimators": 100, "gb_max_depth": 5}
    model = EnsembleModel(config)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = model.predict(X_eval_scaled)
    y_proba = model.predict_proba(X_eval_scaled)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_eval, y_pred, y_proba)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    # Trading evaluation
    y_returns = y[split_idx:]
    returns = np.zeros(len(y_returns))
    trades = 0
    prob_threshold = 0.4  # Lower threshold

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

    print(f"\nTrading Metrics (threshold={prob_threshold}):")
    print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Profit Factor: {trading_metrics.get('profit_factor', 0):.4f}")
    print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.4f}")
    print(f"  Trades: {trades}/{len(y_returns)}")

    return metrics, trading_metrics


def test_regression_with_microstructure(ohlcv: np.ndarray):
    """Test regression model with microstructure features."""
    print("\n" + "="*60)
    print("Testing Regression Model with Microstructure Features")
    print("="*60)

    # Build features with microstructure (no orderbook)
    feature_engineer = FeatureEngineer(
        n_lags=10,
        volatility_window=10,
        include_volume=True,
        include_indicators=True,
        include_orderbook=False
    )

    X, y = feature_engineer.build_features_from_ohlcv(
        ohlcv,
        lookback=20
    )

    print(f"Features shape: {X.shape}")
    print(f"Feature names: {feature_engineer.get_feature_names()[:10]}...")

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_eval = X[:split_idx], X[split_idx:]
    y_train, y_eval = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Train regression
    config = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    model = RegressionModel(config)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_eval_scaled)

    # Evaluate regression
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression(y_eval, y_pred)

    print(f"\nRegression Metrics:")
    print(f"  R²: {metrics.get('r2', 0):.4f}")
    print(f"  MAE: {metrics.get('mae', 0):.6f}")
    print(f"  RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}")

    # Trading evaluation
    returns = np.zeros(len(y_eval))
    trades = 0
    min_pred_return = 0.001

    for i in range(1, len(y_eval)):
        if abs(y_pred[i-1]) >= min_pred_return:
            if y_pred[i-1] > 0:  # BUY
                returns[i] = y_eval[i]
                trades += 1
            elif y_pred[i-1] < 0:  # SELL
                returns[i] = -y_eval[i]
                trades += 1

    trading_metrics = evaluator.evaluate_trading_performance(returns)

    print(f"\nTrading Metrics (threshold={min_pred_return}):")
    print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Profit Factor: {trading_metrics.get('profit_factor', 0):.4f}")
    print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.4f}")
    print(f"  Trades: {trades}/{len(y_eval)}")

    return metrics, trading_metrics


def test_xgboost_with_smote(ohlcv: np.ndarray, orderbooks: List[Dict]):
    """Test XGBoost model with SMOTE and new features."""
    print("\n" + "="*60)
    print("Testing XGBoost Model with SMOTE")
    print("="*60)

    # Build features WITH orderbook
    feature_engineer = FeatureEngineer(
        n_lags=10,
        volatility_window=10,
        include_volume=True,
        include_indicators=True,
        include_orderbook=True
    )

    # Pad orderbooks to match OHLCV length
    padded_orderbooks = [{} for _ in range(len(ohlcv))]
    for i, ob in enumerate(orderbooks):
        padded_orderbooks[i * 5] = ob

    X, y = feature_engineer.build_features_with_extras(
        ohlcv,
        orderbooks=padded_orderbooks,
        onchain_data=None,
        lookback=20
    )

    print(f"Features shape: {X.shape}")

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Apply SMOTE
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = xgb_model.predict(X_eval_scaled)
    y_proba = xgb_model.predict_proba(X_eval_scaled)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_eval, y_pred, y_proba)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    # Feature importance
    feature_importance = xgb_model.feature_importances_
    feature_names = feature_engineer.get_feature_names()
    print(f"\nTop 10 Feature Importances:")
    for idx in np.argsort(feature_importance)[-10:][::-1]:
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

    # Trading evaluation
    y_returns = y[split_idx:]
    returns = np.zeros(len(y_returns))
    trades = 0
    prob_threshold = 0.35  # Lower threshold for XGBoost

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

    print(f"\nTrading Metrics (threshold={prob_threshold}):")
    print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Profit Factor: {trading_metrics.get('profit_factor', 0):.4f}")
    print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.4f}")
    print(f"  Trades: {trades}/{len(y_returns)}")

    return metrics, trading_metrics


def main():
    # Load data
    ohlcv, orderbooks = load_okx_data_with_orderbook("BTC/USDT", "15m", 500)

    if ohlcv is None:
        print("Failed to load data")
        return

    # Test XGBoost with SMOTE
    test_xgboost_with_smote(ohlcv, orderbooks)


if __name__ == "__main__":
    main()
