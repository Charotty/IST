"""
GTX 1660 Ti Performance Benchmarking Tool

Comprehensive benchmarking suite to measure training performance
and optimization effectiveness on GTX 1660 Ti GPU.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))
from gtx1660_optimizer import (
    OptimizedTrainer, GTX1660Config, MemoryMonitor,
    get_resnet_config, get_bert_config, get_llm_config
)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    model_name: str
    optimization_level: str
    batch_size: int
    accumulation_steps: int
    memory_usage_gb: float
    peak_memory_gb: float
    training_time_sec: float
    samples_per_sec: float
    gpu_utilization: float
    speedup_vs_baseline: float
    memory_savings_vs_baseline: float
    error_message: str = ""


class GTX1660Benchmark:
    """GTX 1660 Ti specific benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.memory_monitor = MemoryMonitor(max_memory_usage=4.0)  # 4GB for GTX 1660 Ti
        
        # Test models
        self.models = {
            "resnet50": self._create_resnet50(),
            "bert-base": self._create_bert_base(),
            "small_llm": self._create_small_llm()
        }
        
        # Test configurations
        self.configs = {
            "baseline": self._get_baseline_config(),
            "amp_only": self._get_amp_only_config(),
            "accumulation": self._get_accumulation_config(),
            "checkpointing": self._get_checkpointing_config(),
            "compile": self._get_compile_config(),
            "full_optimized": self._get_full_config()
        }
    
    def _create_resnet50(self) -> nn.Module:
        """Create ResNet50 model for testing."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    
    def _create_bert_base(self) -> nn.Module:
        """Create BERT-base model for testing."""
        return nn.Sequential(
            nn.Embedding(1000, 768),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=768, nhead=12, batch_first=True
                ), num_layers=12
            ),
            nn.Linear(768, 1000)
        )
    
    def _create_small_llm(self) -> nn.Module:
        """Create small LLM for testing."""
        return nn.Sequential(
            nn.Embedding(1000, 512),
            nn.LSTM(512, 512, num_layers=6, batch_first=True),
            nn.Linear(512, 1000)
        )
    
    def run_benchmark(self, quick: bool = False) -> bool:
        """Run complete benchmark suite."""
        print("🚀 Running GTX 1660 Ti Benchmark")
        print("=" * 50)
        
        # System info
        self._print_system_info()
        
        # Test configurations
        configs_to_test = ["baseline", "full_optimized"] if quick else list(self.configs.keys())
        
        success = True
        for model_name, model in self.models.items():
            print(f"\n📊 Testing {model_name}")
            print("-" * 30)
            
            for config_name in configs_to_test:
                config = self.configs[config_name]
                print(f"  ⚡ {config_name}...", end=" ")
                
                try:
                    result = self._benchmark_model(model, model_name, config_name, config)
                    self.results.append(result)
                    
                    if result.error_message:
                        print(f"❌ {result.error_message}")
                        success = False
                    else:
                        print(f"✓ {result.samples_per_sec:.1f} samples/s")
                        
                except Exception as e:
                    error_result = BenchmarkResult(
                        model_name=model_name,
                        optimization_level=config_name,
                        batch_size=config.max_batch_size,
                        accumulation_steps=config.accumulation_steps,
                        memory_usage_gb=0.0,
                        peak_memory_gb=0.0,
                        training_time_sec=0.0,
                        samples_per_sec=0.0,
                        gpu_utilization=0.0,
                        speedup_vs_baseline=0.0,
                        memory_savings_vs_baseline=0.0,
                        error_message=str(e)
                    )
                    self.results.append(error_result)
                    print(f"❌ {str(e)}")
                    success = False
        
        # Generate reports
        if self.results:
            self._generate_reports()
        
        return success
    
    def _print_system_info(self):
        """Print system information."""
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("GPU: Not available")
    
    def _benchmark_model(self, model: nn.Module, model_name: str, 
                        optimization_level: str, config: GTX1660Config) -> BenchmarkResult:
        """Benchmark a single model with given configuration."""
        
        # Create model copy
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        
        # Create trainer
        trainer = OptimizedTrainer(model_copy, config)
        
        # Create dummy data
        if 'cnn' in model_name:
            sample_input = torch.randn(config.max_batch_size, 3, 224, 224)
            target = torch.randint(0, 10, (config.max_batch_size,))
        else:  # transformer
            sample_input = torch.randint(0, 1000, (config.max_batch_size, 128))
            target = torch.randint(0, 1000, (config.max_batch_size, 128))
        
        # Move to GPU
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
            target = target.cuda()
        
        # Benchmark training
        criterion = nn.CrossEntropyLoss()
        num_batches = 50  # Number of batches for benchmark
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                output = trainer.model(sample_input)
                loss = criterion(output, target)
        
        torch.cuda.synchronize()
        
        # Measure memory before
        initial_memory = self.memory_monitor.get_memory_info()['allocated_gb']
        peak_memory = initial_memory
        
        # Start timing
        start_time = time.time()
        total_samples = 0
        
        # Training loop
        for batch_idx in range(num_batches):
            if config.use_mixed_precision:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(device):
                    output = trainer.model(sample_input)
                    loss = criterion(output, target)
                    loss = loss / config.accumulation_steps
            else:
                output = trainer.model(sample_input)
                loss = criterion(output, target)
                loss = loss / config.accumulation_steps

            # Backward pass
            if config.use_mixed_precision:
                trainer.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % config.accumulation_steps == 0:
                if config.use_mixed_precision:
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                else:
                    trainer.optimizer.step()
                trainer.optimizer.zero_grad()

            total_samples += config.max_batch_size

            # Monitor peak memory
            current_memory = self.memory_monitor.get_memory_info()['allocated_gb']
            peak_memory = max(peak_memory, current_memory)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        training_time = end_time - start_time
        samples_per_sec = total_samples / training_time
        memory_usage = self.memory_monitor.get_memory_info()['allocated_gb']
        gpu_utilization = torch.cuda.utilization() if torch.cuda.is_available() else 0

        # Calculate speedup and memory savings vs baseline
        baseline_result = self._get_baseline_result(model_name)
        speedup = samples_per_sec / baseline_result.samples_per_sec if baseline_result.samples_per_sec > 0 else 1.0
        memory_savings = 1.0 - (memory_usage / baseline_result.memory_usage_gb) if baseline_result.memory_usage_gb > 0 else 0.0

        return BenchmarkResult(
            model_name=model_name,
            optimization_level=optimization_level,
            batch_size=config.max_batch_size,
            accumulation_steps=config.accumulation_steps,
            memory_usage_gb=memory_usage,
            peak_memory_gb=peak_memory,
            training_time_sec=training_time,
            samples_per_sec=samples_per_sec,
            gpu_utilization=gpu_utilization,
            speedup_vs_baseline=speedup,
            memory_savings_vs_baseline=memory_savings
        )

    def _get_baseline_result(self, model_name: str) -> BenchmarkResult:
        """Get baseline result for speedup calculation."""
        for result in self.results:
            if result.model_name == model_name and result.optimization_level == "baseline":
                return result

        # Return dummy result if no baseline found
        return BenchmarkResult(
            model_name=model_name,
            optimization_level="baseline",
            batch_size=1,
            accumulation_steps=1,
            memory_usage_gb=1.0,
            peak_memory_gb=1.0,
            training_time_sec=1.0,
            samples_per_sec=1.0,
            gpu_utilization=50.0,
            speedup_vs_baseline=1.0,
            memory_savings_vs_baseline=0.0
        )

    def _get_baseline_config(self) -> GTX1660Config:
        """Baseline configuration (no optimizations)."""
        return GTX1660Config(
            use_mixed_precision=False,
            use_gradient_checkpointing=False,
            use_torch_compile=False,
            use_8bit_optimizer=False,
            auto_batch_size=False,
            target_batch_size=8,
            max_batch_size=8,
            accumulation_steps=1
        )

    def _get_amp_only_config(self) -> GTX1660Config:
        """AMP only configuration."""
        return GTX1660Config(
            use_mixed_precision=True,
            use_gradient_checkpointing=False,
            use_torch_compile=False,
            use_8bit_optimizer=False,
            auto_batch_size=False,
            target_batch_size=8,
            max_batch_size=8,
            accumulation_steps=1
        )

    def _get_accumulation_config(self) -> GTX1660Config:
        """Gradient accumulation configuration."""
        return GTX1660Config(
            use_mixed_precision=False,
            use_gradient_checkpointing=False,
            use_torch_compile=False,
            use_8bit_optimizer=False,
            auto_batch_size=False,
            target_batch_size=32,
            max_batch_size=8,
            accumulation_steps=4
        )

    def _get_checkpointing_config(self) -> GTX1660Config:
        """Gradient checkpointing configuration."""
        return GTX1660Config(
            use_mixed_precision=False,
            use_gradient_checkpointing=True,
            use_torch_compile=False,
            use_8bit_optimizer=False,
            auto_batch_size=False,
            target_batch_size=8,
            max_batch_size=8,
            accumulation_steps=1
        )

    def _get_compile_config(self) -> GTX1660Config:
        """torch.compile configuration."""
        return GTX1660Config(
            use_mixed_precision=False,
            use_gradient_checkpointing=False,
            use_torch_compile=True,
            use_8bit_optimizer=False,
            auto_batch_size=False,
            target_batch_size=8,
            max_batch_size=8,
            accumulation_steps=1
        )

    def _get_full_config(self) -> GTX1660Config:
        """Full optimization configuration (GTX 1660 Ti compatible)."""
        return GTX1660Config(
            use_mixed_precision=True,
            use_gradient_checkpointing=True,
            use_torch_compile=False,  # Disabled for GTX 1660 Ti stability
            use_8bit_optimizer=False,
            auto_batch_size=True,
            target_batch_size=32,
            max_batch_size=16,
            accumulation_steps=2
        )

    def _generate_reports(self):
        """Generate benchmark reports and visualizations."""
        print("\n📈 Generating Benchmark Reports...")

        # Save raw results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            results_data = [asdict(result) for result in self.results]
            json.dump(results_data, f, indent=2)

        print(f"✓ Results saved to {results_file}")

        # Generate summary report
        self._generate_summary_report()

        # Generate visualizations
        self._generate_visualizations()

        print("✓ All reports generated successfully!")

    def _generate_summary_report(self):
        """Generate summary report."""
        summary_file = self.output_dir / "benchmark_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# GTX 1660 Ti Benchmark Summary\n\n")

            # System info
            f.write("## System Information\n\n")
            if torch.cuda.is_available():
                f.write(f"- **GPU**: {torch.cuda.get_device_name(0)}\n")
                f.write(f"- **VRAM**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
                f.write(f"- **CUDA**: {torch.version.cuda}\n")
                f.write(f"- **PyTorch**: {torch.__version__}\n")

            f.write("\n## Results Summary\n\n")

            # Group results by model
            models = set(result.model_name for result in self.results)

            for model_name in sorted(models):
                f.write(f"### {model_name}\n\n")
                f.write("| Optimization | Memory (GB) | Speed (samples/s) | Speedup | Memory Savings |\n")
                f.write("|--------------|-------------|-------------------|---------|----------------|\n")

                model_results = [r for r in self.results if r.model_name == model_name]
                model_results.sort(key=lambda x: x.speedup_vs_baseline, reverse=True)

                for result in model_results:
                    if result.error_message:
                        f.write(f"| {result.optimization_level} | ERROR | ERROR | ERROR | ERROR |\n")
                    else:
                        f.write(f"| {result.optimization_level} | {result.memory_usage_gb:.2f} | "
                               f"{result.samples_per_sec:.1f} | {result.speedup_vs_baseline:.2f}x | "
                               f"{result.memory_savings_vs_baseline:.1%} |\n")

                f.write("\n")

    def _generate_visualizations(self):
        """Generate performance visualizations."""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GTX 1660 Ti Benchmark Results', fontsize=16)

        # Group results by model
        models = set(result.model_name for result in self.results)
        
        for i, model_name in enumerate(sorted(models)):
            model_results = [r for r in self.results if r.model_name == model_name and not r.error_message]
            if not model_results:
                continue
                
            # Sort by optimization level
            model_results.sort(key=lambda x: x.optimization_level)
            
            opt_levels = [r.optimization_level for r in model_results]
            speeds = [r.samples_per_sec for r in model_results]
            memories = [r.memory_usage_gb for r in model_results]
            
            # Speed comparison
            ax = axes[0, 0]
            ax.bar([f"{m}\n{o}" for m, o in zip([model_name]*len(opt_levels), opt_levels)], 
                   speeds, alpha=0.7)
            ax.set_ylabel('Samples/sec')
            ax.set_title('Training Speed')
            ax.tick_params(axis='x', rotation=45)
            
            # Memory usage
            ax = axes[0, 1]
            ax.bar([f"{m}\n{o}" for m, o in zip([model_name]*len(opt_levels), opt_levels)], 
                   memories, alpha=0.7, color='orange')
            ax.set_ylabel('Memory Usage (GB)')
            ax.set_title('Memory Usage')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {plot_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GTX 1660 Ti Benchmark Tool")
    parser.add_argument("--quick", action="store_true", 
                      help="Run quick benchmark (baseline vs optimized only)")
    parser.add_argument("--output", default="benchmark_results",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = GTX1660Benchmark(output_dir=args.output)
    success = benchmark.run_benchmark(quick=args.quick)
    
    if success:
        print("\n✅ Benchmark completed successfully!")
        return 0
    else:
        print("\n❌ Benchmark completed with errors!")
        return 1


if __name__ == "__main__":
    exit(main())
