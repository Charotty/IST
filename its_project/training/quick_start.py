"""
GTX 1660 Ti Quick Start Script

Quick start guide and testing script for GTX 1660 Ti training framework.
Perfect for first-time users to verify setup and see performance gains.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from gtx1660_optimizer import (
    OptimizedTrainer, GTX1660Config, MemoryMonitor,
    get_resnet_config, get_bert_config, get_llm_config
)
from benchmark import run_quick_benchmark


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)


def print_success(message):
    """Print success message."""
    print(f"✅ {message}")


def print_info(message):
    """Print info message."""
    print(f"ℹ️  {message}")


def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")


def check_environment():
    """Check if environment is properly set up."""
    print_header("Environment Check")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        
        print_success(f"CUDA available: {cuda_version}")
        print_success(f"GPU: {gpu_name}")
        print_success(f"VRAM: {gpu_memory:.1f} GB")
        
        if "1660" in gpu_name.upper():
            print_success("GTX 1660 Ti detected - Perfect!")
        else:
            print_warning(f"{gpu_name} detected - Framework optimized for GTX 1660 Ti")
        
        return True
    else:
        print_warning("CUDA not available - Training will be very slow on CPU")
        return False


def create_test_models():
    """Create test models for demonstration."""
    print_header("Creating Test Models")
    
    models = {}
    
    # Small CNN (ResNet-like)
    class SmallCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Simple Transformer
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, d_model=256, nhead=8):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.classifier(x)
            return x
    
    models['cnn'] = SmallCNN()
    models['transformer'] = SimpleTransformer()
    
    print_success(f"Created {len(models)} test models")
    return models


def demonstrate_optimizations():
    """Demonstrate different optimization levels."""
    print_header("Optimization Demonstration")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.layers(x)
            return x
    
    model = TestModel()
    
    # Test configurations
    configs = {
        "Baseline": GTX1660Config(
            use_mixed_precision=False,
            use_torch_compile=False,
            use_gradient_checkpointing=False,
            auto_batch_size=False,
            target_batch_size=16,
            max_batch_size=16
        ),
        "Mixed Precision": GTX1660Config(
            use_mixed_precision=True,
            use_torch_compile=False,
            use_gradient_checkpointing=False,
            auto_batch_size=False,
            target_batch_size=16,
            max_batch_size=16
        ),
        "torch.compile": GTX1660Config(
            use_mixed_precision=False,
            use_torch_compile=True,
            use_gradient_checkpointing=False,
            auto_batch_size=False,
            target_batch_size=16,
            max_batch_size=16
        ),
        "Full Optimization": GTX1660Config(
            use_mixed_precision=True,
            use_torch_compile=True,
            use_gradient_checkpointing=True,
            auto_batch_size=True,
            target_batch_size=32
        )
    }
    
    # Create dummy data
    batch_size = 16
    data = torch.randn(batch_size, 1, 28, 28)  # MNIST-like
    target = torch.randint(0, 10, (batch_size,))
    
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    print("Testing different optimization levels...")
    
    for config_name, config in configs.items():
        print(f"\n🔧 Testing {config_name}...")
        
        try:
            # Create fresh model copy
            test_model = TestModel()
            if torch.cuda.is_available():
                test_model = test_model.cuda()
            
            # Create trainer
            trainer = OptimizedTrainer(test_model, config)
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    output = trainer.model(data)
                    loss = criterion(output, target)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            for _ in range(10):  # 10 iterations
                if config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = trainer.model(data)
                        loss = criterion(output, target)
                        loss = loss / config.accumulation_steps
                else:
                    output = trainer.model(data)
                    loss = criterion(output, target)
                    loss = loss / config.accumulation_steps
                
                if config.use_mixed_precision:
                    trainer.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            
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
    print("\n📊 Optimization Comparison:")
    print("-" * 60)
    
    baseline_speed = results.get("Baseline", {}).get("iterations_per_sec", 1.0)
    baseline_memory = results.get("Baseline", {}).get("memory_gb", 1.0)
    
    for config_name, result in results.items():
        if 'error' not in result:
            speedup = result['iterations_per_sec'] / baseline_speed
            memory_savings = 1.0 - (result['memory_gb'] / baseline_memory)
            
            print(f"{config_name:20} | {speedup:6.2f}x speed | {memory_savings:6.1%} memory | {result['memory_gb']:.2f}GB")
        else:
            print(f"{config_name:20} | ERROR: {result['error']}")
    
    return results


def show_predefined_configs():
    """Show predefined configurations."""
    print_header("Predefined Configurations")
    
    configs = {
        "ResNet": get_resnet_config(),
        "BERT": get_bert_config(),
        "LLM": get_llm_config()
    }
    
    for model_type, config in configs.items():
        print(f"\n🎯 {model_type} Configuration:")
        print(f"   Target batch size: {config.target_batch_size}")
        print(f"   Mixed precision: {config.use_mixed_precision}")
        print(f"   Gradient checkpointing: {config.use_gradient_checkpointing}")
        print(f"   torch.compile: {config.use_torch_compile}")
        print(f"   8-bit optimizer: {config.use_8bit_optimizer}")
        print(f"   Max memory usage: {config.max_memory_usage}GB")


def run_quick_training_demo():
    """Run a quick training demonstration."""
    print_header("Quick Training Demo")
    
    # Create simple model
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = DemoModel()
    
    # Get optimized config
    config = get_resnet_config()
    config.target_batch_size = 32
    config.max_batch_size = 16
    config.accumulation_steps = 2
    
    print("🔧 Using optimized configuration")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    print(f"   torch.compile: {config.use_torch_compile}")
    print(f"   Effective batch size: {config.target_batch_size}")
    
    # Create trainer
    trainer = OptimizedTrainer(model, config)
    
    # Create dummy data
    def create_dummy_data_loader(batch_size, num_batches=20):
        for _ in range(num_batches):
            data = torch.randn(batch_size, 10)
            target = torch.randn(batch_size, 1)
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            yield data, target
    
    # Training loop
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0
    
    print("\n🏋️  Training for 20 batches...")
    
    model.train()
    trainer.optimizer.zero_grad()
    
    for i, (data, target) in enumerate(create_dummy_data_loader(config.max_batch_size)):
        # Forward pass
        loss = trainer._forward_pass(data, target, criterion)
        
        # Accumulate gradients
        trainer._accumulate_gradients(loss)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if i % 5 == 0:
            memory_info = trainer.memory_monitor.get_memory_info()
            print(f"   Batch {i+1:2d}/20 | Loss: {loss.item():.6f} | Memory: {memory_info['allocated_gb']:.2f}GB")
    
    avg_loss = total_loss / num_batches
    final_memory = trainer.memory_monitor.get_memory_info()
    
    print(f"\n📊 Training Results:")
    print(f"   Average loss: {avg_loss:.6f}")
    print(f"   Final memory usage: {final_memory['allocated_gb']:.2f}GB")
    print(f"   Peak memory usage: {final_memory['cached_gb']:.2f}GB")
    print(f"   GPU utilization: {torch.cuda.utilization():.1f}%" if torch.cuda.is_available() else "   CPU only")
    
    print_success("Quick training demo completed!")


def main():
    """Main quick start function."""
    print("🎯 GTX 1660 Ti Training Framework - Quick Start")
    print("This script will verify your setup and demonstrate optimizations")
    
    # Check environment
    has_cuda = check_environment()
    
    # Create test models
    models = create_test_models()
    
    # Show predefined configurations
    show_predefined_configs()
    
    # Demonstrate optimizations
    optimization_results = demonstrate_optimizations()
    
    # Run quick training demo
    run_quick_training_demo()
    
    # Offer to run benchmark
    print_header("Next Steps")
    
    response = input("\nWould you like to run a quick benchmark? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n🏁 Running quick benchmark...")
        run_quick_benchmark()
    
    # Final message
    print_header("Ready to Train!")
    
    print_success("Your GTX 1660 Ti training environment is ready!")
    print("\n📚 Example commands to get started:")
    print("   python training/examples/resnet_training.py --demo")
    print("   python training/examples/bert_training.py --demo")
    print("   python training/examples/llm_finetuning.py --demo")
    print("\n🔬 For full benchmark:")
    print("   python training/benchmark.py --full")
    
    print("\n💡 Tips for best performance:")
    print("   • Always use mixed precision (AMP)")
    print("   • Enable torch.compile for 15-30% speedup")
    print("   • Use gradient accumulation for larger effective batch sizes")
    print("   • Enable gradient checkpointing for large models")
    print("   • Monitor memory usage with built-in monitoring")
    
    if has_cuda:
        print_success("Your GTX 1660 Ti is ready for efficient training!")
    else:
        print_warning("CUDA not detected - training will be CPU only")
        print("   Please install CUDA drivers for GPU acceleration")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Quick start interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        print("Please check your installation and try again")
