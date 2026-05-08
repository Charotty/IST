"""
Comprehensive System Integration Tests

Complete end-to-end testing of the entire ITS trading system:
- WeightedEnsemble with dynamic model selection
- Enhanced Decision Logic with confidence filtering
- Risk Management with SL/TP
- Position Sizing with fixed percentage
- Real-time market simulation
- Performance and stress testing
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import asyncio

# Import system components
from its_project.metalearning.weighted_ensemble import WeightedEnsemble, WeightedEnsembleConfig
from its_project.decision.enhanced_decision import EnhancedDecisionMaker
from its_project.decision.sizing import PositionSizer, SizingConfig, SizingMethod
from its_project.decision.risk import RiskManager, RiskLimits
from its_project.backtesting.engine import WalkForwardEngine
from its_project.models.base import BaseModel


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, model_id: str, accuracy: float = 0.6):
        self.model_id = model_id
        self.accuracy = accuracy
        self.trained = False
        self.predictions = []
        self.confidences = []
    
    def fit(self, X, y):
        """Mock training."""
        self.trained = True
        # Simulate training time
        import time
        time.sleep(0.01)
    
    def predict(self, X):
        """Mock prediction."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        # Simulate prediction based on model accuracy
        if len(X.shape) == 1:
            features = X[0]
        else:
            features = X[-1]
        
        # Simple rule-based prediction for testing
        if len(features) > 0 and features[-1] > 0.5:
            prediction = 2  # BUY
        elif len(features) > 0 and features[-1] < -0.5:
            prediction = 0  # SELL
        else:
            prediction = 1  # HOLD
        
        self.predictions.append(prediction)
        
        # Confidence based on model accuracy
        confidence = self.accuracy + np.random.normal(0, 0.1)
        confidence = np.clip(confidence, 0.0, 1.0)
        self.confidences.append(confidence)
        
        return np.array([prediction])
    
    def predict_proba(self, X):
        """Mock probability prediction."""
        pred = self.predict(X)
        if pred[0] == 0:  # SELL
            return np.array([[0.7, 0.2, 0.1]])  # High confidence in SELL
        elif pred[0] == 2:  # BUY
            return np.array([[0.1, 0.2, 0.7]])  # High confidence in BUY
        else:  # HOLD
            return np.array([[0.3, 0.4, 0.3]])  # Moderate confidence in HOLD


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        np.random.seed(42)
        
        # Generate 1000 bars of OHLCV data
        n_bars = 1000
        timestamps = pd.date_range(start='2023-01-01', periods=n_bars, freq='1H')
        
        # Simulate price movement
        returns = np.random.normal(0.0001, 0.02, n_bars)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            volume = np.random.exponential(1000)
            
            data.append({
                'timestamp_ms': int(timestamp.timestamp() * 1000),
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return data
    
    @pytest.fixture
    def system_config(self):
        """System configuration for testing."""
        return {
            'weighted_ensemble': WeightedEnsembleConfig(
                alpha=0.4, beta=0.4, gamma=0.2,
                window_size=50, switch_threshold=0.1,
                confidence_threshold=0.7,
                confidence_strategy="max_proba",
                use_confidence_filter=True
            ),
            'decision_maker': {
                'confidence_threshold': 0.7,
                'min_price_move': 0.0005,
                'use_price_sign_logic': True,
                'cooldown_enabled': True,
                'cooldown_period': 300,
                'volatility_filter_enabled': True,
                'volatility_window': 20,
                'max_volatility_threshold': 0.05,
                'position_size': 0.01
            },
            'position_sizer': SizingConfig(
                method=SizingMethod.FIXED_PERCENTAGE,
                fixed_percentage_min=0.01,
                fixed_percentage_max=0.02,
                fixed_percentage_default=0.015,
                max_risk_per_trade=0.01,
                max_portfolio_risk=0.05
            ),
            'risk_manager': RiskLimits(
                max_position_size=0.10,
                max_risk_per_trade=0.01,
                max_leverage=3.0,
                default_stop_loss_pct=0.02,
                default_take_profit_pct=0.04,
                trailing_stop_pct=0.01,
                max_portfolio_risk=0.05,
                max_drawdown=0.15,
                max_daily_loss=0.03,
                max_position_duration_hours=24.0,
                forced_close_hours=48.0,
                max_correlated_exposure=0.20,
                volatility_adjustment=True,
                regime_adjustment=True
            )
        }
    
    def test_weighted_ensemble_integration(self, system_config, sample_market_data):
        """Test WeightedEnsemble integration."""
        config = system_config['weighted_ensemble']
        
        # Create mock models with different performance
        models = [
            MockModel("model_1", accuracy=0.55),  # Poor performer
            MockModel("model_2", accuracy=0.65),  # Good performer
            MockModel("model_3", accuracy=0.75),  # Excellent performer
        ]
        
        # Initialize ensemble
        ensemble = WeightedEnsemble(models, config)
        
        # Prepare training data
        train_data = sample_market_data[:500]
        X_train = np.array([[d['close'] for d in train_data]])
        y_train = np.array([1 if i % 3 == 0 else 0 for i in range(len(train_data))])
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Test predictions
        test_data = sample_market_data[500:600]
        X_test = np.array([[d['close'] for d in test_data]])
        
        predictions = ensemble.predict(X_test)
        confidences = ensemble.predict_with_confidence(X_test)
        
        # Verify ensemble behavior
        assert len(predictions) == len(X_test)
        assert len(confidences) == 100
        
        # Test model switching
        initial_model = ensemble.current_model_name
        # Simulate performance degradation
        for i in range(10):
            ensemble.update_performance(
                y_true=np.random.choice([0, 1, 2], size=10),
                y_pred=np.random.choice([0, 1, 2], size=10),
                price_returns=np.random.normal(0, 0.01, size=10)
            )
        
        # Check if model switched
        final_model = ensemble.current_model_name
        assert final_model is not None
        
        # Get performance summary
        summary = ensemble.get_performance_summary()
        assert 'models' in summary
        assert 'current_model' in summary
        assert 'total_trades' in summary
    
    def test_enhanced_decision_integration(self, system_config, sample_market_data):
        """Test EnhancedDecisionMaker integration."""
        config = system_config['decision_maker']
        
        # Create decision maker
        decision_maker = EnhancedDecisionMaker(config)
        
        # Create mock signal
        from its_project.decision.decision import Signal, Action
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTCUSDT",
            metadata={
                "delta_p_hat": 0.015,  # 1.5% predicted move
                "p_hat": 0.8
            }
        )
        
        # Create market state
        market_state = {
            "price": 50000.0,
            "price_history": [49900, 49950, 50000, 50050],
            "volatility": 0.03
        }
        
        # Test decision making
        decision = decision_maker.decide(signal, market_state)
        
        # Verify decision
        assert decision is not None
        assert decision.action in [Action.BUY, Action.SELL, Action.HOLD]
        assert decision.size >= 0
        assert decision.symbol == "BTCUSDT"
        
        # Test confidence filtering
        low_confidence_signal = Signal(
            action=Action.BUY,
            confidence=0.4,  # Below threshold
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01, "p_hat": 0.4}
        )
        
        decision_low_conf = decision_maker.decide(low_confidence_signal, market_state)
        assert decision_low_conf.action == Action.HOLD
        assert "Low confidence" in decision_low_conf.reason
        
        # Test minimum movement filtering
        small_move_signal = Signal(
            action=Action.BUY,
            confidence=0.9,
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.0003, "p_hat": 0.9}  # Below min_move
        )
        
        decision_small_move = decision_maker.decide(small_move_signal, market_state)
        assert decision_small_move.action == Action.HOLD
        assert "Insufficient price move" in decision_small_move.reason
        
        # Test cooldown
        first_decision = decision_maker.decide(signal, market_state)
        second_decision = decision_maker.decide(signal, market_state)
        
        # Second signal should be blocked by cooldown
        if first_decision.action != Action.HOLD:
            assert second_decision.action == Action.HOLD
            assert "Cooldown active" in second_decision.reason
    
    def test_position_sizing_integration(self, system_config):
        """Test PositionSizer integration."""
        config = system_config['position_sizer']
        
        # Create position sizer
        sizer = PositionSizer(config)
        
        # Create mock decision
        from its_project.decision.decision import Decision, Action
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason="Test decision"
        )
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        # Test position sizing
        result = sizer.calculate_position_size(
            decision, portfolio_value, current_positions, market_data
        )
        
        # Verify sizing result
        assert result is not None
        assert result.size > 0
        assert result.size_fraction >= config.fixed_percentage_min
        assert result.size_fraction <= config.fixed_percentage_max
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert result.risk_amount > 0
        
        # Test different portfolio values
        small_portfolio = 10000.0
        result_small = sizer.calculate_position_size(
            decision, small_portfolio, current_positions, market_data
        )
        
        # Should scale with portfolio
        assert result_small.size_fraction == result.size_fraction  # Same percentage
        assert result_small.size < result.size  # Smaller absolute size
    
    def test_risk_manager_integration(self, system_config):
        """Test RiskManager integration."""
        config = system_config['risk_manager']
        
        # Create risk manager
        risk_manager = RiskManager(config)
        
        # Create mock decision
        from its_project.decision.decision import Decision, Action
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.02,  # 2% position
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason="Test decision"
        )
        
        portfolio_value = 100000.0
        current_positions = {}
        
        # Test risk checking
        risk_check = risk_manager.check_decision_risk(
            decision, current_positions, portfolio_value
        )
        
        # Verify risk assessment
        assert risk_check is not None
        assert isinstance(risk_check.approved, bool)
        assert risk_check.risk_level is not None
        
        # Test position limits
        large_decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.15,  # 15% position (above limit)
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason="Large position test"
        )
        
        risk_check_large = risk_manager.check_decision_risk(
            large_decision, current_positions, portfolio_value
        )
        
        # Should be rejected due to size limit
        assert not risk_check_large.approved
        assert risk_check_large.risk_level.value in ['medium', 'high', 'critical']
    
    def test_end_to_end_pipeline(self, system_config, sample_market_data):
        """Test complete end-to-end pipeline."""
        
        # Initialize all components
        ensemble_config = system_config['weighted_ensemble']
        decision_config = system_config['decision_maker']
        sizing_config = system_config['position_sizer']
        risk_config = system_config['risk_manager']
        
        # Create models
        models = [
            MockModel("model_1", accuracy=0.6),
            MockModel("model_2", accuracy=0.7),
            MockModel("model_3", accuracy=0.8)
        ]
        
        # Initialize components
        ensemble = WeightedEnsemble(models, ensemble_config)
        decision_maker = EnhancedDecisionMaker(decision_config)
        sizer = PositionSizer(sizing_config)
        risk_manager = RiskManager(risk_config)
        
        # Train ensemble
        train_data = sample_market_data[:500]
        X_train = np.array([[d['close'] for d in train_data]])
        y_train = np.random.choice([0, 1, 2], size=len(train_data))
        ensemble.fit(X_train, y_train)
        
        # Process new data point
        new_data = sample_market_data[500]
        X_new = np.array([[new_data['close']]])
        
        # Generate prediction
        prediction = ensemble.predict(X_new)
        confidence = ensemble.predict_with_confidence(X_new)
        
        # Create signal
        from its_project.decision.decision import Signal, Action
        
        signal = Signal(
            action=Action.BUY if prediction[0] == 2 else Action.SELL if prediction[0] == 0 else Action.HOLD,
            confidence=confidence[0],
            timestamp=new_data['timestamp_ms'],
            symbol="BTCUSDT",
            metadata={
                "delta_p_hat": 0.015,
                "p_hat": confidence[0]
            }
        )
        
        # Make decision
        market_state = {
            "price": new_data['close'],
            "price_history": [d['close'] for d in sample_market_data[450:500]],
            "volatility": 0.03
        }
        
        decision = decision_maker.decide(signal, market_state)
        
        # Check risk
        risk_check = risk_manager.check_decision_risk(
            decision, {}, 100000.0
        )
        
        # Size position
        if risk_check.approved:
            sizing_result = sizer.calculate_position_size(
                decision, 100000.0, {}, market_state
            )
            
            # Verify complete pipeline
            assert sizing_result is not None
            assert sizing_result.size > 0
            assert decision.action == sizing_result.method_used or decision.action in [Action.BUY, Action.SELL]
        
        # Test pipeline statistics
        ensemble_stats = ensemble.get_performance_summary()
        decision_stats = decision_maker.get_decision_stats()
        sizing_stats = sizer.get_statistics()
        risk_stats = risk_manager.get_statistics()
        
        # Verify all components have statistics
        assert ensemble_stats is not None
        assert decision_stats is not None
        assert sizing_stats is not None
        assert risk_stats is not None
    
    def test_performance_under_load(self, system_config, sample_market_data):
        """Test system performance under load."""
        
        # Initialize system
        models = [MockModel(f"model_{i}", accuracy=0.7) for i in range(5)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Train with larger dataset
        large_train_data = sample_market_data * 3  # 3000 data points
        X_train = np.array([[d['close'] for d in large_train_data]])
        y_train = np.random.choice([0, 1, 2], size=len(large_train_data))
        
        # Measure training time
        import time
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert training_time < 30.0  # 30 seconds max
        
        # Test prediction performance
        X_test = np.array([[d['close'] for d in sample_market_data[:100]]])
        
        start_time = time.time()
        for _ in range(100):  # 100 predictions
            predictions = ensemble.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Should be fast
        assert prediction_time < 1.0  # 1 second for 100 predictions
        assert len(predictions) == len(X_test)
    
    def test_error_handling_and_recovery(self, system_config):
        """Test error handling and system recovery."""
        
        # Test with invalid model
        with pytest.raises(ValueError):
            invalid_models = []
            WeightedEnsemble(invalid_models, system_config['weighted_ensemble'])
        
        # Test with invalid configuration
        with pytest.raises(ValueError):
            invalid_config = WeightedEnsembleConfig(alpha=-0.1)  # Invalid negative weight
            WeightedEnsemble([], invalid_config)
        
        # Test decision maker with invalid signal
        decision_maker = EnhancedDecisionMaker(system_config['decision_maker'])
        
        from its_project.decision.decision import Signal, Action
        
        invalid_signal = Signal(
            action=Action.BUY,
            confidence=1.5,  # Invalid confidence > 1.0
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTCUSDT",
            metadata={}
        )
        
        # Should handle gracefully
        decision = decision_maker.decide(invalid_signal, {})
        assert decision.action == Action.HOLD  # Fallback to HOLD
        assert "validation failed" in decision.reason.lower()
    
    def test_concurrent_operations(self, system_config, sample_market_data):
        """Test concurrent operations."""
        
        # Initialize multiple instances
        models = [MockModel(f"model_{i}", accuracy=0.7) for i in range(3)]
        
        ensembles = [
            WeightedEnsemble(models[:2], system_config['weighted_ensemble']),
            WeightedEnsemble(models[1:], system_config['weighted_ensemble'])
        ]
        
        # Train concurrently
        train_data = sample_market_data[:200]
        X_train = np.array([[d['close'] for d in train_data]])
        y_train = np.random.choice([0, 1, 2], size=len(train_data))
        
        async def train_ensemble(ensemble, X, y):
            ensemble.fit(X, y)
            return ensemble.get_performance_summary()
        
        # Run concurrent training
        async def run_concurrent():
            tasks = [train_ensemble(ensemble, X_train, y_train) for ensemble in ensembles]
            results = await asyncio.gather(*tasks)
            return results
        
        # Execute concurrent operations
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(run_concurrent())
        
        # Verify all completed successfully
        assert len(results) == 2
        assert all(result is not None for result in results)


class TestSystemPerformance:
    """Test system performance and optimization."""
    
    def test_memory_usage(self, system_config):
        """Test memory usage under different loads."""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large ensemble
        models = [MockModel(f"model_{i}", accuracy=0.7) for i in range(10)]
        ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
        
        # Train with large dataset
        large_data = []
        for i in range(1000):
            large_data.extend(np.random.randn(100).tolist())
        
        X_train = np.array(large_data)
        y_train = np.random.choice([0, 1, 2], size=len(large_data))
        
        ensemble.fit(X_train, y_train)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_scalability(self, system_config):
        """Test system scalability with different model counts."""
        
        model_counts = [1, 3, 5, 10, 20]
        training_times = []
        
        for count in model_counts:
            models = [MockModel(f"model_{i}", accuracy=0.7) for i in range(count)]
            ensemble = WeightedEnsemble(models, system_config['weighted_ensemble'])
            
            # Small dataset for speed testing
            X_train = np.random.randn(100, 10)
            y_train = np.random.choice([0, 1, 2], size=100)
            
            import time
            start_time = time.time()
            ensemble.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
        
        # Check scalability
        # Training time should scale reasonably
        for i in range(1, len(training_times)):
            ratio = training_times[i] / training_times[0]
            expected_ratio = (model_counts[i] / model_counts[0]) ** 1.5  # Expected scaling
            assert ratio < expected_ratio * 2  # Allow 2x overhead


if __name__ == "__main__":
    pytest.main([__file__])
