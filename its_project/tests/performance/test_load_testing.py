"""
Load and Performance Testing for ITS Trading System

Tests system performance under various load conditions:
- High-frequency data processing
- Large model ensembles
- Concurrent operations
- Memory and CPU usage
- Latency measurements
"""

import pytest
import numpy as np
import pandas as pd
import time
import asyncio
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch

from its_project.metalearning.weighted_ensemble import WeightedEnsemble, WeightedEnsembleConfig
from its_project.decision.enhanced_decision import EnhancedDecisionMaker
from its_project.decision.sizing import PositionSizer, SizingConfig, SizingMethod
from its_project.decision.risk import RiskManager, RiskLimits


class TestLoadTesting:
    """Test system performance under load."""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for load testing."""
        np.random.seed(42)
        
        # Generate 100,000 data points
        n_points = 100000
        timestamps = pd.date_range(start='2020-01-01', periods=n_points, freq='1min')
        
        # Simulate realistic price movements
        returns = np.random.normal(0.00001, 0.02, n_points)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = []
        for i, timestamp in enumerate(timestamps):
            data.append({
                'timestamp_ms': int(timestamp.timestamp() * 1000),
                'open': prices[i],
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.001))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.001))),
                'close': prices[i],
                'volume': np.random.exponential(1000)
            })
        
        return data
    
    @pytest.fixture
    def system_config(self):
        """System configuration optimized for performance."""
        return {
            'weighted_ensemble': WeightedEnsembleConfig(
                alpha=0.4, beta=0.4, gamma=0.2,
                window_size=100, switch_threshold=0.1,
                confidence_threshold=0.7,
                confidence_strategy="max_proba",
                use_confidence_filter=True
            ),
            'decision_maker': {
                'confidence_threshold': 0.7,
                'min_price_move': 0.0005,
                'use_price_sign_logic': True,
                'cooldown_enabled': True,
                'cooldown_period': 60,  # Shorter for testing
                'volatility_filter_enabled': True,
                'volatility_window': 20,
                'max_volatility_threshold': 0.05
            },
            'position_sizer': SizingConfig(
                method=SizingMethod.FIXED_PERCENTAGE,
                fixed_percentage_min=0.01,
                fixed_percentage_max=0.02,
                fixed_percentage_default=0.015
            ),
            'risk_manager': RiskLimits(
                max_position_size=0.10,
                max_risk_per_trade=0.01,
                max_portfolio_risk=0.05
            )
        }
    
    def test_high_frequency_data_processing(self, large_dataset, system_config):
        """Test processing high-frequency data."""
        
        # Create ensemble with multiple models
        from its_project.models.base import BaseModel
        
        class FastModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
                self.trained = False
            
            def fit(self, X, y):
                self.trained = True
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                probs = np.random.dirichlet([1, 1, 1], size=len(X))
                return probs
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        models = [FastModel(f"model_{i}") for i in range(10)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Train with large dataset
        X_train = np.array([[d['close'] for d in large_dataset[:50000]]]).T
        y_train = np.random.choice([0, 1, 2], size=50000)
        
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert training_time < 60.0, f"Training took too long: {training_time}s"
        
        # Test high-frequency predictions
        X_test = np.array([[d['close'] for d in large_dataset[50000:60000]]]).T
        
        start_time = time.time()
        for i in range(1000):  # 1000 prediction cycles
            predictions = ensemble.predict(X_test)
            confidences = ensemble.predict_with_confidence(X_test)
        prediction_time = time.time() - start_time
        
        # Should be fast for high-frequency trading
        avg_prediction_time = prediction_time / 1000
        assert avg_prediction_time < 0.01, f"Average prediction too slow: {avg_prediction_time}s"
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Average prediction time: {avg_prediction_time:.4f}s")
    
    def test_large_ensemble_performance(self, system_config, large_dataset):
        """Test performance with large model ensembles."""
        
        from its_project.models.base import BaseModel
        
        class LightweightModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
                self.trained = False
            
            def fit(self, X, y):
                self.trained = True
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        # Test with different ensemble sizes
        ensemble_sizes = [5, 10, 20, 50, 100]
        training_times = []
        prediction_times = []
        
        X_train = np.array([[d['close'] for d in large_dataset[:10000]]]).T
        y_train = np.random.choice([0, 1, 2], size=10000)
        X_test = np.array([[d['close'] for d in large_dataset[10000:11000]]]).T
        
        for size in ensemble_sizes:
            models = [LightweightModel(f"model_{i}") for i in range(size)]
            ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
            
            # Measure training time
            start_time = time.time()
            ensemble.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Measure prediction time
            start_time = time.time()
            for _ in range(100):
                predictions = ensemble.predict(X_test)
            prediction_time = (time.time() - start_time) / 100
            prediction_times.append(prediction_time)
            
            print(f"Ensemble size {size}: Training {training_time:.2f}s, Prediction {prediction_time:.4f}s")
        
        # Check scalability
        # Training time should scale sub-linearly
        for i in range(1, len(training_times)):
            size_ratio = ensemble_sizes[i] / ensemble_sizes[0]
            time_ratio = training_times[i] / training_times[0]
            
            # Allow some overhead but should be reasonable
            assert time_ratio < size_ratio * 2, f"Poor scalability at size {ensemble_sizes[i]}"
        
        # Prediction time should remain fast
        for pred_time in prediction_times:
            assert pred_time < 0.1, f"Prediction too slow: {pred_time}s"
    
    def test_concurrent_operations(self, system_config):
        """Test concurrent system operations."""
        
        from its_project.models.base import BaseModel
        
        class ConcurrentModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
                self.trained = False
            
            def fit(self, X, y):
                time.sleep(0.01)  # Simulate training time
                self.trained = True
            
            def predict(self, X):
                time.sleep(0.001)  # Simulate prediction time
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        async def run_ensemble(model_id, data):
            """Run ensemble operation concurrently."""
            models = [ConcurrentModel(f"model_{i}_{model_id}") for i in range(5)]
            ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
            
            X_train = np.array([[d['close'] for d in data[:1000]]]).T
            y_train = np.random.choice([0, 1, 2], size=1000)
            
            ensemble.fit(X_train, y_train)
            
            X_test = np.array([[d['close'] for d in data[1000:1100]]]).T
            predictions = ensemble.predict(X_test)
            
            return {
                'model_id': model_id,
                'predictions': predictions,
                'performance': ensemble.get_performance_summary()
            }
        
        # Create test data for concurrent operations
        test_datasets = []
        for i in range(10):
            dataset = []
            for j in range(2000):
                dataset.append({
                    'timestamp_ms': int((time.time() + i * 1000 + j) * 1000),
                    'close': 50000 + np.random.normal(0, 100)
                })
            test_datasets.append(dataset)
        
        # Run concurrent operations
        async def run_concurrent():
            tasks = [run_ensemble(i, test_datasets[i]) for i in range(10)]
            results = await asyncio.gather(*tasks)
            return results
        
        # Measure concurrent execution time
        start_time = time.time()
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(run_concurrent())
        concurrent_time = time.time() - start_time
        
        # Verify all completed successfully
        assert len(results) == 10
        assert all(result['predictions'] is not None for result in results)
        assert all(result['performance'] is not None for result in results)
        
        # Should complete in reasonable time
        assert concurrent_time < 30.0, f"Concurrent operations too slow: {concurrent_time}s"
        
        print(f"Concurrent operations time: {concurrent_time:.2f}s")
    
    def test_memory_usage_under_load(self, system_config, large_dataset):
        """Test memory usage under various loads."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from its_project.models.base import BaseModel
        
        class MemoryTestModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
                self.large_data = np.random.randn(1000, 100)  # Large internal data
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        # Test with increasing model counts
        model_counts = [1, 5, 10, 20, 50]
        memory_usage = []
        
        for count in model_counts:
            # Clean up previous models
            import gc
            gc.collect()
            
            models = [MemoryTestModel(f"model_{i}") for i in range(count)]
            ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
            
            # Train with large dataset
            X_train = np.array([[d['close'] for d in large_dataset[:20000]]]).T
            y_train = np.random.choice([0, 1, 2], size=20000)
            
            ensemble.fit(X_train, y_train)
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            memory_usage.append(memory_increase)
            
            print(f"Model count {count}: Memory increase {memory_increase:.1f}MB")
        
        # Check memory efficiency
        # Memory should scale reasonably with model count
        for i in range(1, len(memory_usage)):
            count_ratio = model_counts[i] / model_counts[0]
            memory_ratio = memory_usage[i] / memory_usage[0]
            
            # Allow some overhead but should be reasonable
            assert memory_ratio < count_ratio * 3, f"Poor memory efficiency at {model_counts[i]} models"
        
        # Total memory should be reasonable
        assert memory_usage[-1] < 2000, f"Excessive memory usage: {memory_usage[-1]}MB"
    
    def test_cpu_usage_under_load(self, system_config):
        """Test CPU usage during high-load operations."""
        
        from its_project.models.base import BaseModel
        
        class CPUIntensiveModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
            
            def fit(self, X, y):
                # Simulate CPU-intensive training
                for _ in range(1000):
                    _ = np.dot(np.random.randn(100, 100), np.random.randn(100, 100))
            
            def predict(self, X):
                # Simulate CPU-intensive prediction
                for _ in range(100):
                    _ = np.dot(X, np.random.randn(X.shape[1], 100))
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        # Monitor CPU usage
        process = psutil.Process(os.getpid())
        
        models = [CPUIntensiveModel(f"model_{i}") for i in range(10)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Generate training data
        X_train = np.random.randn(10000, 50)
        y_train = np.random.choice([0, 1, 2], size=10000)
        
        # Measure CPU during training
        start_time = time.time()
        initial_cpu = process.cpu_percent()
        
        ensemble.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        final_cpu = process.cpu_percent()
        
        # Training should complete in reasonable time
        assert training_time < 120.0, f"CPU-intensive training too slow: {training_time}s"
        
        # CPU usage should be reasonable
        avg_cpu = (initial_cpu + final_cpu) / 2
        print(f"Average CPU usage during training: {avg_cpu:.1f}%")
        
        # Should not use excessive CPU (allow some overhead)
        assert avg_cpu < 95.0, f"Excessive CPU usage: {avg_cpu}%"
    
    def test_latency_measurements(self, system_config, large_dataset):
        """Measure system latency under various conditions."""
        
        from its_project.models.base import BaseModel
        
        class LatencyTestModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        models = [LatencyTestModel(f"model_{i}") for i in range(5)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Train ensemble
        X_train = np.array([[d['close'] for d in large_dataset[:5000]]]).T
        y_train = np.random.choice([0, 1, 2], size=5000)
        ensemble.fit(X_train, y_train)
        
        # Measure prediction latency
        X_test = np.array([[d['close'] for d in large_dataset[5000:6000]]]).T
        
        latencies = []
        for i in range(1000):
            start_time = time.perf_counter()
            predictions = ensemble.predict(X_test)
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        # Analyze latency statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        print(f"Latency stats:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  99th percentile: {p99_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        
        # Latency requirements
        assert avg_latency < 10.0, f"Average latency too high: {avg_latency}ms"
        assert p95_latency < 20.0, f"P95 latency too high: {p95_latency}ms"
        assert p99_latency < 50.0, f"P99 latency too high: {p99_latency}ms"
        assert max_latency < 100.0, f"Max latency too high: {max_latency}ms"


class TestStressTesting:
    """Stress testing for system limits."""
    
    def test_extreme_ensemble_size(self, system_config):
        """Test with extremely large ensembles."""
        
        from its_project.models.base import BaseModel
        
        class MinimalModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        # Test with extreme ensemble sizes
        extreme_sizes = [100, 200, 500, 1000]
        
        for size in extreme_sizes:
            print(f"Testing ensemble size: {size}")
            
            try:
                models = [MinimalModel(f"model_{i}") for i in range(size)]
                ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
                
                # Small training data
                X_train = np.random.randn(100, 10)
                y_train = np.random.choice([0, 1, 2], size=100)
                
                start_time = time.time()
                ensemble.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Test prediction
                X_test = np.random.randn(10, 10)
                start_time = time.time()
                predictions = ensemble.predict(X_test)
                prediction_time = time.time() - start_time
                
                print(f"  Size {size}: Training {training_time:.2f}s, Prediction {prediction_time:.4f}s")
                
                # Should still work (though slower)
                assert training_time < 300.0, f"Training failed for size {size}"
                assert prediction_time < 1.0, f"Prediction failed for size {size}"
                
            except Exception as e:
                # Should handle gracefully
                print(f"  Size {size}: Failed with {e}")
                if size <= 500:
                    raise  # Should work up to 500 models
                else:
                    print(f"  Expected failure for size {size}")
    
    def test_high_frequency_simulation(self, system_config):
        """Simulate high-frequency trading scenario."""
        
        from its_project.models.base import BaseModel
        
        class HFTModel(BaseModel):
            def __init__(self, model_id):
                self.model_id = model_id
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.choice([0, 1, 2], size=len(X))
            
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1, 1], size=len(X))
            
            def get_confidence(self, X):
                return np.random.uniform(0.6, 0.9, len(X))
        
        models = [HFTModel(f"model_{i}") for i in range(3)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Train ensemble
        X_train = np.random.randn(1000, 10)
        y_train = np.random.choice([0, 1, 2], size=1000)
        ensemble.fit(X_train, y_train)
        
        # Simulate high-frequency data stream
        decision_maker = EnhancedDecisionMaker(system_config['decision_maker'])
        sizer = PositionSizer(system_config['position_sizer'])
        
        # Process 10,000 data points rapidly
        total_time = 0
        decisions_made = 0
        
        for i in range(10000):
            start_time = time.perf_counter()
            
            # Generate market data
            X_new = np.random.randn(1, 10)
            predictions = ensemble.predict(X_new)
            confidences = ensemble.predict_with_confidence(X_new)
            
            # Make decision (simplified)
            if confidences[0] > 0.7:
                from its_project.decision.decision import Decision, Action
                
                decision = Decision(
                    action=Action.BUY if predictions[0] == 2 else Action.SELL,
                    symbol="BTCUSDT",
                    size=0.0,
                    price=50000.0,
                    timestamp=int(time.time() * 1000),
                    reason="HFT test"
                )
                
                # Size position
                sizing_result = sizer.calculate_position_size(
                    decision, 100000.0, {}, {"price": 50000.0}
                )
                
                if sizing_result.size > 0:
                    decisions_made += 1
            
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
        
        avg_processing_time = total_time / 10000 * 1000  # Convert to ms
        decision_rate = decisions_made / 10000
        
        print(f"HFT Simulation:")
        print(f"  Average processing time: {avg_processing_time:.3f}ms")
        print(f"  Decision rate: {decision_rate:.2%}")
        
        # Should be fast for HFT
        assert avg_processing_time < 1.0, f"Too slow for HFT: {avg_processing_time}ms"
        assert decision_rate > 0.01, f"Too few decisions: {decision_rate}"


if __name__ == "__main__":
    pytest.main([__file__])
