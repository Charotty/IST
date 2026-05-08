"""
Example usage of WeightedEnsemble with dynamic model selection.

Demonstrates Score = α·Sharpe + β·PnL - γ·DD formula
and real-time model switching based on performance degradation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from its_project.metalearning.weighted_ensemble import (
    WeightedEnsemble, 
    WeightedEnsembleConfig
)
from its_project.models.regression_model import RegressionModel
from its_project.models.ensemble import EnsembleModel
from its_project.evaluation.feature_engineering import FeatureEngineer


def create_sample_models() -> list:
    """Create sample models for ensemble."""
    models = [
        RegressionModel({
            "model_type": "linear",
            "random_state": 42
        }),
        RegressionModel({
            "model_type": "gradient_boosting", 
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        }),
        EnsembleModel({
            "rf_estimators": 50,
            "rf_max_depth": 5,
            "gb_estimators": 50,
            "gb_max_depth": 3,
            "random_state": 42
        })
    ]
    return models


def create_sample_data(n_samples: int = 1000) -> tuple:
    """Create sample trading data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    prices = 50000 * np.cumprod(1 + np.random.randn(n_samples) * 0.001)
    ohlcv = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        price = prices[i]
        volatility = 0.002  # 0.2% daily volatility
        
        open_price = price * (1 + np.random.randn() * volatility)
        high_price = price * (1 + abs(np.random.randn()) * volatility)
        low_price = price * (1 - abs(np.random.randn()) * volatility)
        close_price = price
        volume = np.random.rand() * 1000
        
        ohlcv[i] = [open_price, high_price, low_price, close_price, volume]
    
    # Build features and targets
    feature_engineer = FeatureEngineer(
        n_lags=5,
        volatility_window=10,
        include_volume=True,
        include_indicators=True
    )
    
    X, y = feature_engineer.build_features_from_ohlcv(ohlcv, lookback=20)
    
    return X, y, ohlcv


def demonstrate_weighted_ensemble():
    """Demonstrate WeightedEnsemble functionality."""
    print("🚀 WeightedEnsemble Demonstration")
    print("=" * 50)
    
    # Create configuration with custom weights
    config = WeightedEnsembleConfig(
        alpha=0.4,      # Weight for Sharpe ratio
        beta=0.4,        # Weight for PnL
        gamma=0.2,       # Weight for Drawdown (penalty)
        window_size=50,    # Sliding window for performance
        min_trades_for_switch=25,
        switch_threshold=0.05,
        score_smoothing=True,
        smoothing_window=10,
        ensemble_weights=[0.5, 0.3, 0.2]  # Custom ensemble weights
    )
    
    print(f"Configuration:")
    print(f"  Score formula: Score = {config.alpha}·Sharpe + {config.beta}·PnL - {config.gamma}·DD")
    print(f"  Window size: {config.window_size}")
    print(f"  Switch threshold: {config.switch_threshold}")
    print(f"  Ensemble weights: {config.ensemble_weights}")
    print()
    
    # Create models
    models = create_sample_models()
    print(f"Created {len(models)} models:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model.__class__.__name__}")
    print()
    
    # Create ensemble
    ensemble = WeightedEnsemble(models, config)
    
    # Generate sample data
    X, y, ohlcv = create_sample_data(1000)
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    # Fit ensemble
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)
    print("✅ Training complete")
    print()
    
    # Initial prediction
    print("Making initial predictions...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_proba.shape}")
    print()
    
    # Simulate trading performance updates
    print("Simulating trading performance...")
    
    # Calculate price returns from OHLCV
    price_returns = np.zeros(len(ohlcv) - 1)
    for i in range(len(ohlcv) - 1):
        price_returns[i] = (ohlcv[i+1, 3] - ohlcv[i, 3]) / ohlcv[i, 3]
    
    # Split returns corresponding to test set
    split_idx = len(y_train)
    test_returns = price_returns[split_idx:split_idx+len(y_test)]
    
    # Simulate multiple performance updates
    n_updates = 10
    batch_size = len(y_test) // n_updates
    
    for update in range(n_updates):
        start_idx = update * batch_size
        end_idx = min((update + 1) * batch_size, len(y_test))
        
        if start_idx >= len(y_test):
            break
            
        batch_y_true = y_test[start_idx:end_idx]
        batch_y_pred = y_pred[start_idx:end_idx]
        batch_returns = test_returns[start_idx:end_idx]
        
        # Update ensemble performance
        ensemble.update_performance(batch_y_true, batch_y_pred, batch_returns)
        
        current_model = ensemble.get_current_model()
        rankings = ensemble.get_model_rankings()
        
        print(f"Update {update+1}/{n_updates}:")
        print(f"  Current model: {current_model.__class__.__name__}")
        print(f"  Trades processed: {ensemble.total_trades}")
        print(f"  Top 3 models:")
        for i, (_, row) in enumerate(rankings.head(3).iterrows()):
            status = "🔥" if row['is_current'] else "  "
            print(f"    {status} {row['model']}: Score={row['window_score']:.4f}")
        print()
    
    # Final performance summary
    print("📊 Final Performance Summary")
    print("=" * 30)
    
    summary = ensemble.get_performance_summary()
    print(f"Final active model: {summary['current_model']}")
    print(f"Total trades processed: {summary['total_trades']}")
    print()
    
    print("Individual model performance:")
    for model_name, model_summary in summary['models'].items():
        print(f"  {model_name}:")
        print(f"    Current score: {model_summary['current_score']:.4f}")
        print(f"    Best score: {model_summary['best_score']:.4f}")
        print(f"    Window score: {model_summary['window_score']:.4f}")
        print(f"    Sharpe: {model_summary['window_sharpe']:.4f}")
        print(f"    PnL: {model_summary['window_pnl']:.4f}")
        print(f"    Drawdown: {model_summary['window_drawdown']:.4f}")
        print(f"    Trades: {model_summary['trades_count']}")
        print()
    
    # Demonstrate model switching
    print("🔄 Model Switching Demonstration")
    print("=" * 35)
    
    # Simulate performance degradation
    print("Simulating performance degradation for current model...")
    
    # Add poor performance for current model
    current_model_name = ensemble.current_model_name
    current_perf = ensemble.model_performances[current_model_name]
    
    for i in range(5):
        # Poor performance updates
        poor_sharpe = -0.5 + i * 0.1  # Deteriorating Sharpe
        poor_pnl = -0.02 - i * 0.005  # Deteriorating PnL
        poor_dd = 0.1 + i * 0.02  # Increasing drawdown
        
        batch_y_true = np.random.randint(0, 3, 20)
        batch_y_pred = np.random.randint(0, 3, 20)  # Random predictions
        batch_returns = np.random.randn(20) * 0.01
        
        ensemble.update_performance(batch_y_true, batch_y_pred, batch_returns)
        
        print(f"  Poor performance update {i+1}: Score={current_perf.current_score:.4f}")
    
    # Check if model switching occurs
    print("\nChecking for model switch...")
    old_model = ensemble.current_model_name
    ensemble._check_and_switch_model()
    new_model = ensemble.current_model_name
    
    if old_model != new_model:
        print(f"✅ Model switched: {old_model} → {new_model}")
    else:
        print("ℹ️  No model switch occurred")
    
    print()
    
    # Test different configurations
    print("🧪 Testing Different Configurations")
    print("=" * 40)
    
    configs = [
        {"name": "Sharpe-focused", "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        {"name": "PnL-focused", "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
        {"name": "Risk-averse", "alpha": 0.3, "beta": 0.3, "gamma": 0.4},
        {"name": "Balanced", "alpha": 0.33, "beta": 0.33, "gamma": 0.34},
    ]
    
    for config_dict in configs:
        print(f"\nTesting {config_dict['name']} configuration:")
        
        test_config = WeightedEnsembleConfig(
            alpha=config_dict["alpha"],
            beta=config_dict["beta"], 
            gamma=config_dict["gamma"],
            window_size=30,
            switch_threshold=0.1
        )
        
        test_ensemble = WeightedEnsemble(models[:2], test_config)  # Use 2 models for speed
        test_ensemble.fit(X_train, y_train)
        
        # Quick performance evaluation
        y_test_pred = test_ensemble.predict(X_test[:100])
        test_returns = test_returns[:100]
        test_ensemble.update_performance(y_test[:100], y_test_pred, test_returns)
        
        final_summary = test_ensemble.get_performance_summary()
        current_score = final_summary['models'][test_ensemble.current_model_name]['current_score']
        
        print(f"  Final score: {current_score:.4f}")
        print(f"  Active model: {test_ensemble.current_model_name}")
    
    # Demonstrate confidence thresholding
    print("\n🎯 Confidence Thresholding Demonstration")
    print("=" * 45)
    
    # Test different confidence strategies
    strategies = ["max_proba", "entropy", "margin"]
    threshold = 0.7
    
    print(f"Testing confidence strategies with threshold = {threshold}")
    print()
    
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        
        # Create ensemble with specific confidence strategy
        confidence_config = WeightedEnsembleConfig(
            confidence_threshold=threshold,
            confidence_strategy=strategy,
            use_confidence_filter=True
        )
        
        test_ensemble = WeightedEnsemble(models[:2], confidence_config)
        test_ensemble.fit(X_train, y_train)
        
        # Get confidence statistics
        stats = test_ensemble.get_confidence_stats(X_test[:50])
        
        print(f"  Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"  Above threshold: {stats['above_threshold']}/{50} ({stats['above_threshold']/50:.1%})")
        print(f"  Below threshold: {stats['below_threshold']}/{50} ({stats['below_threshold']/50:.1%})")
        print(f"  Threshold ratio: {stats['threshold_ratio']:.1%}")
        print()
    
    # Test confidence filtering impact
    print("🔍 Confidence Filtering Impact Analysis")
    print("=" * 40)
    
    # Test with and without confidence filtering
    config_with_filter = WeightedEnsembleConfig(
        confidence_threshold=0.6,
        use_confidence_filter=True,
        confidence_strategy="max_proba"
    )
    
    config_without_filter = WeightedEnsembleConfig(
        confidence_threshold=0.6,
        use_confidence_filter=False,
        confidence_strategy="max_proba"
    )
    
    ensemble_with_filter = WeightedEnsemble(models[:2], config_with_filter)
    ensemble_without_filter = WeightedEnsemble(models[:2], config_without_filter)
    
    # Fit both ensembles
    ensemble_with_filter.fit(X_train, y_train)
    ensemble_without_filter.fit(X_train, y_train)
    
    # Compare predictions
    X_sample = X_test[:100]
    
    pred_filtered, conf_filtered = ensemble_with_filter.predict_with_confidence(X_sample)
    pred_unfiltered, conf_unfiltered = ensemble_without_filter.predict_with_confidence(X_sample)
    
    # Calculate differences
    hold_ratio_filtered = np.mean(pred_filtered == 1)  # HOLD predictions
    hold_ratio_unfiltered = np.mean(pred_unfiltered == 1)
    
    print("Prediction comparison:")
    print(f"  With confidence filter:")
    print(f"    HOLD ratio: {hold_ratio_filtered:.1%}")
    print(f"    Mean confidence: {np.mean(conf_filtered):.4f}")
    print(f"    Below threshold: {np.sum(conf_filtered < 0.6)}/{len(conf_filtered)}")
    print()
    print(f"  Without confidence filter:")
    print(f"    HOLD ratio: {hold_ratio_unfiltered:.1%}")
    print(f"    Mean confidence: {np.mean(conf_unfiltered):.4f}")
    print(f"    Below threshold: {np.sum(conf_unfiltered < 0.6)}/{len(conf_unfiltered)}")
    print()
    
    # Test different thresholds
    print("📊 Threshold Sensitivity Analysis")
    print("=" * 35)
    
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    
    for thresh in thresholds:
        thresh_config = WeightedEnsembleConfig(
            confidence_threshold=thresh,
            use_confidence_filter=True,
            confidence_strategy="max_proba"
        )
        
        thresh_ensemble = WeightedEnsemble(models[:2], thresh_config)
        thresh_ensemble.fit(X_train, y_train)
        
        pred, conf = thresh_ensemble.predict_with_confidence(X_test[:50])
        hold_ratio = np.mean(pred == 1)
        
        print(f"  Threshold {thresh}: HOLD ratio {hold_ratio:.1%}, Mean confidence {np.mean(conf):.4f}")
    
    print("\n🎯 Demonstration Complete!")
    print("=" * 30)
    print("WeightedEnsemble successfully demonstrated:")
    print("✅ Dynamic model selection based on Score = α·Sharpe + β·PnL - γ·DD")
    print("✅ Real-time performance tracking with sliding windows")
    print("✅ Automatic model switching on performance degradation")
    print("✅ Configurable ensemble weights and parameters")
    print("✅ Confidence thresholding with multiple strategies")
    print("✅ p_hat based prediction filtering")
    print("✅ Configurable confidence thresholds")


if __name__ == "__main__":
    demonstrate_weighted_ensemble()
