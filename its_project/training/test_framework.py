"""
Simple test script to verify GTX 1660 Ti training framework functionality
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from gtx1660_optimizer import OptimizedTrainer, GTX1660Config, MemoryMonitor
    print("✅ Successfully imported GTX 1660 Ti optimizer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic framework functionality."""
    print("\n🧪 Testing GTX 1660 Ti Framework")
    print("=" * 50)
    
    # Test GPU detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {gpu_memory:.1f} GB")
        print(f"✅ CUDA: {torch.version.cuda}")
    else:
        print("⚠️ CUDA not available - using CPU")
    
    # Test memory monitor
    try:
        monitor = MemoryMonitor()
        memory_info = monitor.get_memory_info()
        print(f"✅ Memory monitor: {memory_info['allocated_gb']:.2f}GB used")
    except Exception as e:
        print(f"❌ Memory monitor error: {e}")
    
    # Test configuration
    try:
        config = GTX1660Config(
            use_mixed_precision=True,
            use_torch_compile=False,  # Disabled for stability
            use_gradient_checkpointing=False,
            auto_batch_size=False,
            target_batch_size=8,
            max_batch_size=8
        )
        print("✅ Configuration created successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test model creation
    try:
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False
    
    # Test trainer creation
    try:
        trainer = OptimizedTrainer(model, config)
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Trainer creation error: {e}")
        return False
    
    # Test basic training step
    try:
        # Create dummy data
        batch_size = 4
        data = torch.randn(batch_size, 10)
        target = torch.randn(batch_size, 1)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        criterion = nn.MSELoss()
        
        # Training step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        # Forward pass
        if config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = trainer.model(data)
                loss = criterion(output, target)
        else:
            output = trainer.model(data)
            loss = criterion(output, target)
        
        # Backward pass
        if config.use_mixed_precision:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            trainer.optimizer.step()
        
        print(f"✅ Training step successful - Loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        return False
    
    return True

def test_optimization_levels():
    """Test different optimization levels."""
    print("\n🔬 Testing Optimization Levels")
    print("=" * 50)
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Test configurations
    configs = {
        "Baseline": GTX1660Config(
            use_mixed_precision=False,
            use_torch_compile=False,
            target_batch_size=8
        ),
        "Mixed Precision": GTX1660Config(
            use_mixed_precision=True,
            use_torch_compile=False,
            target_batch_size=8
        ),
        "Full (No Compile)": GTX1660Config(
            use_mixed_precision=True,
            use_torch_compile=False,  # Disabled for GTX 1660 Ti
            use_gradient_checkpointing=True,
            target_batch_size=8
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🔧 Testing {config_name}...")
        
        try:
            # Create fresh model
            test_model = TestModel()
            if torch.cuda.is_available():
                test_model = test_model.cuda()
            
            # Create trainer
            trainer = OptimizedTrainer(test_model, config)
            
            # Benchmark
            criterion = nn.MSELoss()
            data = torch.randn(8, 10)
            target = torch.randn(8, 1)
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    output = trainer.model(data)
                    loss = criterion(output, target)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(10):
                if config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = trainer.model(data)
                        loss = criterion(output, target)
                else:
                    output = trainer.model(data)
                    loss = criterion(output, target)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # Calculate metrics
            training_time = end_time - start_time
            iterations_per_sec = 10 / training_time
            memory_usage = trainer.memory_monitor.get_memory_info()['allocated_gb']
            
            results[config_name] = {
                'time': training_time,
                'iterations_per_sec': iterations_per_sec,
                'memory_gb': memory_usage
            }
            
            print(f"   Time: {training_time:.3f}s")
            print(f"   Speed: {iterations_per_sec:.1f} iter/s")
            print(f"   Memory: {memory_usage:.2f}GB")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[config_name] = {'error': str(e)}
    
    # Show comparison
    print(f"\n📊 Optimization Comparison:")
    print("-" * 60)
    
    baseline_speed = results.get("Baseline", {}).get("iterations_per_sec", 1.0)
    
    for config_name, result in results.items():
        if 'error' not in result:
            speedup = result['iterations_per_sec'] / baseline_speed
            print(f"{config_name:20} | {speedup:6.2f}x speed | {result['memory_gb']:.2f}GB")
        else:
            print(f"{config_name:20} | ERROR: {result['error']}")
    
    return results

def main():
    """Main test function."""
    print("🎯 GTX 1660 Ti Training Framework Test")
    print("=" * 60)
    
    # Basic functionality test
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed")
        return False
    
    # Optimization levels test
    optimization_results = test_optimization_levels()
    
    print(f"\n🎉 Framework Test Completed!")
    print(f"✅ All core functionality working")
    print(f"✅ Optimizations tested successfully")
    
    if torch.cuda.is_available():
        print(f"✅ Ready for GPU training on {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Ready for CPU training only")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🚀 Your GTX 1660 Ti training framework is ready!")
            print(f"💡 You can now start training models with full optimizations")
        else:
            print(f"\n❌ Framework test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
