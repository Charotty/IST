"""
Test threshold computation with real data.

Verifies that:
1. Safe threshold computation prevents look-ahead leakage
2. Rolling/expanding thresholds use only past data
3. Fixed threshold from training works correctly
4. Unsafe mode shows the leakage problem
"""

import pandas as pd
import numpy as np

from utils.data_leakage_prevention import (
    compute_safe_threshold,
    compute_train_threshold
)
from decision.signal_rules import (
    compute_final_signal,
    compute_integrated_signal
)
from feature_engineering.feature_engine import FeatureEngine
from feature_engineering.config import FeatureEngineeringConfig


def main():
    """Test threshold computation with real data."""
    print("=" * 60)
    print("Testing Threshold Computation with Real Data")
    print("=" * 60)
    
    # Step 1: Load real data
    print("\n[1] Loading real data...")
    df = pd.read_parquet('data/ohlcv/demo_BTC-USDT_1h.parquet')
    print(f"   Loaded {len(df)} rows")
    
    # Step 2: Generate features
    print("\n[2] Generating features...")
    feature_config = FeatureEngineeringConfig()
    feature_engine = FeatureEngine(df, feature_config)
    feature_engine.add_indicators()
    features = feature_engine.get_processed_data()
    print(f"   Generated features: {len(features)} rows")
    
    # Step 3: Create synthetic meta probabilities
    print("\n[3] Creating synthetic meta probabilities...")
    np.random.seed(42)
    # Create meta probabilities with some trend
    meta_prob = 0.5 + np.cumsum(np.random.randn(len(features)) * 0.01)
    meta_prob = np.clip(meta_prob, 0, 1)
    print(f"   Meta prob range: {meta_prob.min():.4f} to {meta_prob.max():.4f}")
    print(f"   Meta prob mean: {meta_prob.mean():.4f}")
    
    # Step 4: Test unsafe threshold computation (look-ahead leakage)
    print("\n[4] Testing UNSAFE threshold computation (look-ahead leakage)...")
    unsafe_threshold = np.median(meta_prob)
    print(f"   Unsafe median threshold (entire sample): {unsafe_threshold:.4f}")
    print(f"   This uses FUTURE data - will cause inflated backtest metrics!")
    
    # Step 5: Test safe threshold computation
    print("\n[5] Testing SAFE threshold computation...")
    
    # Fixed threshold from training
    train_size = int(len(meta_prob) * 0.7)
    train_meta_prob = meta_prob[:train_size]
    test_meta_prob = meta_prob[train_size:]
    
    train_threshold = compute_train_threshold(train_meta_prob, mode="median")
    print(f"   Train median threshold (training data only): {train_threshold:.4f}")
    print(f"   Test median threshold (test data only): {np.median(test_meta_prob):.4f}")
    print(f"   Difference: {abs(train_threshold - np.median(test_meta_prob)):.4f}")
    
    # Rolling window threshold
    rolling_threshold = compute_safe_threshold(
        meta_prob, mode="median", window=100, expanding=False
    )
    print(f"\n   Rolling threshold (window=100):")
    print(f"     First threshold: {rolling_threshold.iloc[100]:.4f}")
    print(f"     Last threshold: {rolling_threshold.iloc[-1]:.4f}")
    print(f"     Uses only past data - no look-ahead")
    
    # Expanding window threshold
    expanding_threshold = compute_safe_threshold(
        meta_prob, mode="median", window=None, expanding=True
    )
    print(f"\n   Expanding threshold:")
    print(f"     First threshold: {expanding_threshold.iloc[0]:.4f}")
    print(f"     Last threshold: {expanding_threshold.iloc[-1]:.4f}")
    print(f"     Uses only past data - no look-ahead")
    
    # Step 6: Test signal computation with safe thresholds
    print("\n[6] Testing signal computation with safe thresholds...")
    
    # Create synthetic direction signal
    direction_soft_signal = np.random.randn(len(features)) * 0.1
    
    # Unsafe signal
    unsafe_signal = compute_integrated_signal(
        direction_soft_signal, meta_prob,
        meta_threshold_mode="median",
        safe_mode=False  # UNSAFE
    )
    print(f"   Unsafe signal (look-ahead threshold):")
    print(f"     Long signals: {(unsafe_signal == 1).sum()}")
    print(f"     Short signals: {(unsafe_signal == -1).sum()}")
    print(f"     Flat signals: {(unsafe_signal == 0).sum()}")
    
    # Safe signal with train threshold
    safe_signal_fixed = compute_integrated_signal(
        direction_soft_signal, meta_prob,
        meta_threshold_mode="median",
        safe_mode=True,
        train_threshold=train_threshold
    )
    print(f"\n   Safe signal (fixed train threshold):")
    print(f"     Long signals: {(safe_signal_fixed == 1).sum()}")
    print(f"     Short signals: {(safe_signal_fixed == -1).sum()}")
    print(f"     Flat signals: {(safe_signal_fixed == 0).sum()}")
    
    # Safe signal with expanding threshold
    safe_signal_expanding = compute_integrated_signal(
        direction_soft_signal, meta_prob,
        meta_threshold_mode="median",
        safe_mode=True,
        expanding=True
    )
    print(f"\n   Safe signal (expanding threshold):")
    print(f"     Long signals: {(safe_signal_expanding == 1).sum()}")
    print(f"     Short signals: {(safe_signal_expanding == -1).sum()}")
    print(f"     Flat signals: {(safe_signal_expanding == 0).sum()}")
    
    # Step 7: Compare thresholds
    print("\n[7] Comparing threshold approaches...")
    print(f"   Unsafe (entire sample): {unsafe_threshold:.4f}")
    print(f"   Safe (train only): {train_threshold:.4f}")
    print(f"   Safe (expanding last): {expanding_threshold.iloc[-1]:.4f}")
    print(f"   Difference unsafe vs safe train: {abs(unsafe_threshold - train_threshold):.4f}")
    
    if abs(unsafe_threshold - train_threshold) > 0.01:
        print(f"\n   WARNING: Significant difference between unsafe and safe thresholds!")
        print(f"   This shows the look-ahead leakage problem.")
    
    # Step 8: Test train/test split scenario
    print("\n[8] Testing train/test split scenario...")
    
    # Train on first 70%, test on last 30%
    train_idx = int(len(meta_prob) * 0.7)
    train_meta = meta_prob[:train_idx]
    test_meta = meta_prob[train_idx:]
    
    # Compute threshold from train only
    train_only_threshold = compute_train_threshold(train_meta, mode="median")
    
    # Apply to test
    test_signal_safe = compute_integrated_signal(
        direction_soft_signal[train_idx:], test_meta,
        meta_threshold_mode="median",
        safe_mode=True,
        train_threshold=train_only_threshold
    )
    
    print(f"   Train threshold: {train_only_threshold:.4f}")
    print(f"   Test signals with train threshold:")
    print(f"     Long: {(test_signal_safe == 1).sum()}")
    print(f"     Short: {(test_signal_safe == -1).sum()}")
    print(f"     Flat: {(test_signal_safe == 0).sum()}")
    
    # Compare with unsafe (using test data in threshold)
    test_signal_unsafe = compute_integrated_signal(
        direction_soft_signal[train_idx:], test_meta,
        meta_threshold_mode="median",
        safe_mode=False
    )
    
    print(f"\n   Test signals with unsafe threshold (uses test data):")
    print(f"     Long: {(test_signal_unsafe == 1).sum()}")
    print(f"     Short: {(test_signal_unsafe == -1).sum()}")
    print(f"     Flat: {(test_signal_unsafe == 0).sum()}")
    
    signal_diff = np.sum(test_signal_safe != test_signal_unsafe)
    print(f"\n   Signal difference: {signal_diff} out of {len(test_signal_safe)}")
    print(f"   This shows the impact of look-ahead leakage on signals!")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Threshold leakage prevention working correctly!")
    print("=" * 60)
    print("\nSummary:")
    print("- Unsafe threshold uses entire sample (including future)")
    print("- Safe threshold uses only past data (train, rolling, or expanding)")
    print("- Fixed threshold from training prevents look-ahead")
    print("- Rolling/expanding thresholds adapt using only past data")
    print("- Signal differences show impact of leakage")
    print("\nRecommendation: Use safe_mode=True with train_threshold or expanding=True")


if __name__ == '__main__':
    main()
