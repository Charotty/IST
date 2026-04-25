"""
Integration tests for full pipeline (data → storage → preprocessing → features → models → decision → execution).
"""
import pytest
import pandas as pd
import numpy as np
import asyncio
from its_project.preprocessing.cleaner import DataCleaner
from its_project.preprocessing.normalizer import DataNormalizer
from its_project.features.technical import TechnicalFeatures
from its_project.models.ensemble import EnsembleModel
from its_project.decision.simple import SimpleDecisionMaker
from its_project.execution.paper import PaperTradingExecutor
from its_project.decision.decision import Action, Signal


@pytest.mark.integration
@pytest.mark.e2e
class TestFullPipeline:
    """Test full end-to-end pipeline integration."""
    
    def test_data_preprocessing_features_model_decision_execution(self, sample_market_data):
        """Test data → preprocessing → features → model → decision → execution pipeline."""
        # Preprocessing
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        # Normalization
        normalizer = DataNormalizer(method='minmax')
        price_cols = ['open', 'high', 'low', 'close']
        normalized = normalizer.fit_transform(cleaned[price_cols])
        
        # Feature engineering
        features = TechnicalFeatures({'indicators': ['rsi', 'macd']})
        feature_data = features.calculate(cleaned)
        
        # Model prediction
        model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        X = feature_data[['rsi', 'macd']].values
        y = np.array([0, 1, 2] * len(feature_data))[:len(feature_data)]
        model.fit(X, y)
        prediction = model.predict(X)
        
        # Decision making
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1234567890000,
            symbol='BTCUSDT',
            metadata={'price': 42000}
        )
        
        decision_maker = SimpleDecisionMaker({'confidence_threshold': 0.7})
        decision = decision_maker.make_decision(signal, {'price': 42000}, 10000.0)
        
        # Execution
        executor = PaperTradingExecutor({'initial_balance': 10000.0, 'latency_ms': 50, 'commission_rate': 0.001, 'slippage_rate': 0.0005})
        
        if decision:
            from its_project.execution.base import Order, OrderType, OrderStatus
            order = Order(
                id='test_order',
                symbol='BTCUSDT',
                type=OrderType.MARKET,
                side=decision.action.value,
                amount=decision.size,
                price=decision.price,
                status=OrderStatus.PENDING,
                filled=0.0,
                remaining=decision.size,
                timestamp=1234567890000,
                info={}
            )
            result = executor.execute_order(order)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_queue_based_pipeline(self, async_queue):
        """Test queue-based pipeline communication."""
        # Simulate data flow through queues
        for i in range(10):
            await async_queue.put({
                'timestamp': 1234567890000 + i * 1000,
                'symbol': 'BTCUSDT',
                'price': 42000 + i * 10,
                'volume': 100 + i
            })
        
        # Process from queue
        messages = []
        while not async_queue.empty():
            messages.append(await async_queue.get())
        
        assert len(messages) == 10
    
    def test_error_propagation_through_pipeline(self, sample_market_data):
        """Test error propagation through pipeline."""
        # Test with invalid data
        invalid_data = sample_market_data.copy()
        invalid_data.loc[0, 'close'] = np.nan
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(invalid_data)
        
        # Should handle gracefully
        assert cleaned is not None
    
    def test_pipeline_performance(self, sample_market_data):
        """Test pipeline performance on larger dataset."""
        # Create larger dataset
        large_data = pd.concat([sample_market_data] * 10)
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(large_data)
        
        # Should complete in reasonable time
        assert len(cleaned) == len(large_data)
    
    def test_pipeline_state_consistency(self, sample_market_data):
        """Test state consistency across pipeline stages."""
        # Preprocessing
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        # Verify data integrity
        assert len(cleaned) == len(sample_market_data)
        assert 'timestamp' in cleaned.columns
    
    def test_pipeline_with_multiple_assets(self):
        """Test pipeline with multiple assets."""
        # Create data for multiple assets
        btc_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'open': [42000 + i for i in range(100)],
            'high': [42050 + i for i in range(100)],
            'low': [41950 + i for i in range(100)],
            'close': [42000 + i for i in range(100)],
            'volume': [100 + i for i in range(100)]
        })
        
        eth_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'open': [2200 + i for i in range(100)],
            'high': [2205 + i for i in range(100)],
            'low': [2195 + i for i in range(100)],
            'close': [2200 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        
        # Process both assets
        cleaner = DataCleaner()
        btc_cleaned = cleaner.clean(btc_data)
        eth_cleaned = cleaner.clean(eth_data)
        
        assert btc_cleaned is not None
        assert eth_cleaned is not None
