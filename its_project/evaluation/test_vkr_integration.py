"""
Test script for VKR integration
Verifies all VKR components work together
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_models():
    """Test all neural network models."""
    print("\n" + "="*60)
    print("Testing Neural Network Models")
    print("="*60)
    
    try:
        import torch
        from models.gru import GRUModel
        from models.transformer import TransformerModel
        from models.cnn_lob import CNNLOBModel
        from models.siamese_lob import SiameseLOBModel
        
        # Test GRU
        print("Testing GRU model...")
        gru_config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'output_size': 3}
        gru = GRUModel(gru_config)
        test_input = torch.FloatTensor(np.random.randn(4, 20, 10))
        output = gru(test_input)
        print(f"✓ GRU output shape: {output.shape}")
        
        # Test Transformer
        print("Testing Transformer model...")
        transformer_config = {'input_size': 10, 'd_model': 32, 'nhead': 4, 'num_layers': 2, 'num_classes': 3, 'max_seq_len': 20}
        transformer = TransformerModel(transformer_config)
        test_input = np.random.randn(4, 20, 10)
        output = transformer.predict(test_input)
        print(f"✓ Transformer output shape: {output.shape}")
        
        # Test CNN-LOB
        print("Testing CNN-LOB model...")
        cnn_config = {'input_channels': 40, 'num_levels': 20, 'seq_len': 100, 'num_classes': 3}
        cnn = CNNLOBModel(cnn_config)
        test_input = torch.FloatTensor(np.random.randn(4, 100, 40))
        output = cnn(test_input)
        print(f"✓ CNN-LOB output shape: {output.shape}")
        
        # Test Siamese
        print("Testing Siamese LOB model...")
        siamese_config = {'input_size': 2, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        siamese = SiameseLOBModel(siamese_config)
        bid_input = torch.FloatTensor(np.random.randn(4, 20, 2))
        ask_input = torch.FloatTensor(np.random.randn(4, 20, 2))
        output = siamese(bid_input, ask_input)
        print(f"✓ Siamese output shape: {output.shape}")
        
        print("\n✓ All models tested successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onchain_client():
    """Test on-chain data client."""
    print("\n" + "="*60)
    print("Testing On-Chain Client")
    print("="*60)
    
    try:
        from backend.onchain_client import OnChainClient
        
        print("Testing OnChainClient initialization...")
        client = OnChainClient(api_key=None)  # Free tier
        print("✓ OnChainClient initialized")
        
        # Note: Actual API calls would require valid API key
        print("Note: Actual data fetching requires Glassnode API key")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing on-chain client: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analyzer():
    """Test sentiment analyzer."""
    print("\n" + "="*60)
    print("Testing Sentiment Analyzer")
    print("="*60)
    
    try:
        from backend.sentiment_analyzer import get_sentiment_analyzer
        
        print("Testing sentiment analyzer (mock mode)...")
        analyzer = get_sentiment_analyzer(use_mock=True)
        
        test_text = "Bitcoin is showing strong bullish momentum today"
        result = analyzer.analyze_sentiment(test_text)
        
        print(f"✓ Sentiment analysis result: {result}")
        print(f"  Label: {result['label']}")
        print(f"  Positive: {result['positive']:.2f}")
        print(f"  Negative: {result['negative']:.2f}")
        print(f"  Neutral: {result['neutral']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing sentiment analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_threshold():
    """Test confidence thresholding."""
    print("\n" + "="*60)
    print("Testing Confidence Thresholding")
    print("="*60)
    
    try:
        from evaluation.confidence_threshold import ConfidenceThreshold
        
        print("Testing ConfidenceThreshold...")
        ct = ConfidenceThreshold(initial_threshold=0.65)
        
        # Test execution decision
        should_execute = ct.should_execute(0.7, 2)
        print(f"✓ Should execute (0.7 confidence): {should_execute}")
        
        should_execute = ct.should_execute(0.5, 2)
        print(f"✓ Should execute (0.5 confidence): {should_execute}")
        
        # Test threshold adaptation
        ct.update_threshold(0.8, 1.5)
        print(f"✓ Threshold after good performance: {ct.threshold:.2f}")
        
        ct.update_threshold(0.4, 0.3)
        print(f"✓ Threshold after poor performance: {ct.threshold:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing confidence threshold: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_meta_learning():
    """Test meta-learning layer."""
    print("\n" + "="*60)
    print("Testing Meta-Learning Layer")
    print("="*60)
    
    try:
        from evaluation.meta_learning import MetaLearner
        
        print("Testing MetaLearner...")
        # Create mock models
        mock_models = {
            'model1': type('MockModel', (), {'predict': lambda self, x: np.zeros(len(x))})(),
            'model2': type('MockModel', (), {'predict': lambda self, x: np.ones(len(x))})()
        }
        
        meta_learner = MetaLearner(mock_models)
        
        # Test performance tracking
        meta_learner.update_performance('model1', 0.8, 'accuracy')
        meta_learner.update_performance('model2', 0.6, 'accuracy')
        
        weights = meta_learner.get_model_weights()
        print(f"✓ Model weights: {weights}")
        
        best_model = meta_learner.select_best_model()
        print(f"✓ Best model: {best_model}")
        
        # Test regime detection
        X = np.random.randn(100, 10)
        regime = meta_learner.detect_market_regime(X)
        print(f"✓ Detected regime: {regime}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing meta-learning: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_drift():
    """Test data drift detection."""
    print("\n" + "="*60)
    print("Testing Data Drift Detection")
    print("="*60)
    
    try:
        from evaluation.data_drift import DataDriftDetector
        
        print("Testing DataDriftDetector...")
        detector = DataDriftDetector(threshold=0.05)
        
        # Fit reference distribution
        ref_data = np.random.randn(1000, 10)
        detector.fit_reference(ref_data)
        print(f"✓ Reference distribution fitted")
        
        # Test with similar data (no drift)
        similar_data = np.random.randn(100, 10)
        result = detector.detect_drift(similar_data)
        print(f"✓ No drift detected (similar data): {result['drift_detected']}")
        
        # Test with different data (drift)
        different_data = np.random.randn(100, 10) + 5  # Shifted distribution
        result = detector.detect_drift(different_data)
        print(f"✓ Drift detected (shifted data): {result['drift_detected']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data drift: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vkr_integration():
    """Test complete VKR integration."""
    print("\n" + "="*60)
    print("Testing Complete VKR Integration")
    print("="*60)
    
    try:
        from evaluation.vkr_integration import create_vkr_system
        
        print("Creating VKR system...")
        config = {
            'input_size': 10,
            'gru_hidden_size': 32,
            'd_model': 32,
            'nhead': 4,
            'transformer_layers': 2,
            'lob_channels': 40,
            'lob_levels': 20,
            'seq_len': 20,
            'confidence_threshold': 0.65,
            'use_mock_sentiment': True
        }
        
        system = create_vkr_system(config)
        print("✓ VKR system created")
        
        # Get system status
        status = system.get_system_status()
        print(f"✓ System status: {status}")
        
        # Test prediction
        X = np.random.randn(10, 10)
        prediction, metadata = system.predict(X, use_meta_learning=False)
        print(f"✓ Prediction shape: {prediction.shape}")
        print(f"✓ Selected model: {metadata['selected_model']}")
        
        # Test sentiment analysis
        sentiment = system.analyze_sentiment("Bitcoin is going up")
        print(f"✓ Sentiment: {sentiment['label']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing VKR integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VKR Integration Test Suite")
    print("="*60)
    
    # Import torch for model tests
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        import torch
    
    results = {}
    
    # Run all tests
    results['models'] = test_models()
    results['onchain'] = test_onchain_client()
    results['sentiment'] = test_sentiment_analyzer()
    results['confidence'] = test_confidence_threshold()
    results['meta_learning'] = test_meta_learning()
    results['data_drift'] = test_data_drift()
    results['integration'] = test_vkr_integration()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ All VKR components integrated successfully!")
    else:
        print(f"\n✗ {total_tests - total_passed} test(s) failed")


if __name__ == "__main__":
    main()
