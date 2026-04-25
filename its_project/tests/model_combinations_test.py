"""
Model Combinations and Load Testing Script

This script tests various model configurations and performs load testing
to ensure the training pipeline works correctly with different parameters.
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import concurrent.futures

# Backend URL
BACKEND_URL = "http://127.0.0.1:5050"

# Calculate date range for last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
DATE_START = start_date.strftime("%Y-%m-%d")
DATE_END = end_date.strftime("%Y-%m-%d")


class ModelTester:
    """Test model training with various configurations."""
    
    def __init__(self, backend_url: str = BACKEND_URL):
        self.backend_url = backend_url
        self.results: List[Dict] = []
    
    def load_data(self, symbol: str = "BTC/USDT", timeframe: str = "1h", 
                  start_date: str = DATE_START, end_date: str = DATE_END) -> bool:
        """Load historical data."""
        try:
            response = requests.post(
                f"{self.backend_url}/api/data/load",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_sources": ["candles"]
                },
                timeout=120
            )
            result = response.json()
            print(f"Data load: {result.get('message', 'Unknown')}")
            
            # Check if candles were actually loaded
            if result.get("success"):
                candle_count = result.get("data", {}).get("candle_count", 0)
                if candle_count == 0:
                    print("Warning: 0 candles loaded. Try different date range.")
                    return False
            
            return result.get("success", False)
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_features(self, technical_indicators: List[str] = None, 
                          orderbook_features: List[str] = None) -> bool:
        """Calculate features."""
        if technical_indicators is None:
            technical_indicators = ["sma", "ema", "rsi", "macd", "bollinger"]
        if orderbook_features is None:
            orderbook_features = ["imbalance", "spread", "depth"]
        
        try:
            response = requests.post(
                f"{self.backend_url}/api/features/calculate",
                json={
                    "technical_indicators": technical_indicators,
                    "orderbook_features": orderbook_features
                },
                timeout=60
            )
            result = response.json()
            print(f"Features calculated: {result.get('message', 'Unknown')}")
            return result.get("success", False)
        except Exception as e:
            print(f"Error calculating features: {e}")
            return False
    
    def train_model(self, config: Dict) -> Dict:
        """Train model with given configuration."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.backend_url}/api/models/train",
                json=config,
                timeout=300
            )
            elapsed_time = time.time() - start_time
            result = response.json()
            
            test_result = {
                "config": config,
                "success": result.get("success", False),
                "message": result.get("message", "Unknown"),
                "training_time": elapsed_time,
                "metrics": result.get("data", {})
            }
            
            print(f"Training completed in {elapsed_time:.2f}s: {result.get('message', 'Unknown')}")
            self.results.append(test_result)
            return test_result
        except Exception as e:
            print(f"Error training model: {e}")
            return {
                "config": config,
                "success": False,
                "message": str(e),
                "training_time": 0,
                "metrics": {}
            }


def test_model_combinations():
    """Test various model type and hyperparameter combinations."""
    tester = ModelTester()
    
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    if not tester.load_data():
        print("Failed to load data. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Calculating features...")
    print("=" * 60)
    if not tester.calculate_features():
        print("Failed to calculate features. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Testing model combinations...")
    print("=" * 60)
    
    # Model type combinations
    model_types = ["lstm", "gru", "transformer", "ensemble"]
    
    # Hyperparameter combinations
    sequence_lengths = [30, 60, 120]
    hidden_layers = [1, 2, 3]
    dropout_values = [0.1, 0.2, 0.3]
    learning_rates = [0.0001, 0.001, 0.01]
    epochs_values = [25, 50, 100]
    
    # Test a subset of combinations (full test would be too large)
    test_configs = [
        # Test different model types
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "gru", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "transformer", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "ensemble", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        
        # Test different sequence lengths
        {"model_type": "lstm", "sequence_length": 30, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "lstm", "sequence_length": 120, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        
        # Test different hidden layers
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 1, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 3, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        
        # Test different dropout values
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.1, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.3, "learning_rate": 0.001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        
        # Test different learning rates
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.0001, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.01, "epochs": 50, "batch_size": 32, "train_test_split": 0.8},
        
        # Test different epochs
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 25, "batch_size": 32, "train_test_split": 0.8},
        {"model_type": "lstm", "sequence_length": 60, "hidden_layers": 2, 
         "dropout": 0.2, "learning_rate": 0.001, "epochs": 100, "batch_size": 32, "train_test_split": 0.8},
    ]
    
    print(f"Testing {len(test_configs)} configurations...\n")
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] Testing config: {config['model_type']}, "
              f"seq={config['sequence_length']}, layers={config['hidden_layers']}, "
              f"dropout={config['dropout']}, lr={config['learning_rate']}, epochs={config['epochs']}")
        tester.train_model(config)
        time.sleep(1)  # Small delay between tests
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful = [r for r in tester.results if r["success"]]
    failed = [r for r in tester.results if not r["success"]]
    
    print(f"Total tests: {len(tester.results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful configurations:")
        for r in successful:
            metrics = r.get("metrics", {})
            print(f"  - {r['config']['model_type']}: "
                  f"accuracy={metrics.get('accuracy', 0):.3f}, "
                  f"f1={metrics.get('f1', 0):.3f}, "
                  f"time={r['training_time']:.2f}s")
    
    if failed:
        print("\nFailed configurations:")
        for r in failed:
            print(f"  - {r['config']['model_type']}: {r['message']}")
    
    # Save results to file
    with open("model_test_results.json", "w") as f:
        json.dump(tester.results, f, indent=2)
    print(f"\nResults saved to model_test_results.json")


def load_test_training():
    """Perform load testing on training endpoint."""
    tester = ModelTester()
    
    print("=" * 60)
    print("LOAD TESTING - Training Endpoint")
    print("=" * 60)
    
    # Load data and features once
    if not tester.load_data():
        print("Failed to load data. Exiting.")
        return
    
    if not tester.calculate_features():
        print("Failed to calculate features. Exiting.")
        return
    
    # Test configuration
    config = {
        "model_type": "lstm",
        "sequence_length": 60,
        "hidden_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "train_test_split": 0.8
    }
    
    # Test with different concurrency levels
    concurrency_levels = [1, 2, 5]
    
    for concurrency in concurrency_levels:
        print(f"\nTesting with {concurrency} concurrent requests...")
        
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(tester.train_model, config) for _ in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in concurrent request: {e}")
        
        elapsed_time = time.time() - start_time
        successful = sum(1 for r in results if r["success"])
        
        print(f"  Completed {len(results)} requests in {elapsed_time:.2f}s")
        print(f"  Successful: {successful}/{len(results)}")
        print(f"  Average time per request: {elapsed_time/len(results):.2f}s")
    
    print("\nLoad testing complete.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "combinations":
            test_model_combinations()
        elif mode == "load":
            load_test_training()
        else:
            print("Usage: python model_combinations_test.py [combinations|load]")
    else:
        print("Running model combinations test...")
        test_model_combinations()
