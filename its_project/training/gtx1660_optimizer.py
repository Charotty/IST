"""
GTX 1660 Ti Optimized Training Framework

Comprehensive training optimization framework for GTX 1660 Ti GPU with 6GB VRAM.
Implements mixed precision training, gradient accumulation, checkpointing,
and PyTorch 2.0 optimizations for maximum efficiency.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GTX1660Config:
    """Configuration for GTX 1660 Ti optimized training."""
    
    # Core training parameters
    target_batch_size: int = 32
    max_batch_size: int = 32
    min_batch_size: int = 1
    
    # Optimization flags
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_torch_compile: bool = True
    use_8bit_optimizer: bool = False
    
    # Memory management
    auto_batch_size: bool = True
    memory_monitoring: bool = True
    cleanup_frequency: int = 100
    
    # Compilation settings
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Gradient accumulation
    accumulation_steps: Optional[int] = None
    
    # Memory limits (in GB)
    max_memory_usage: float = 5.5  # Leave some margin


class MemoryMonitor:
    """VRAM monitoring and management for GTX 1660 Ti."""
    
    def __init__(self, max_memory_gb: float = 5.5):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1e9
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {'available': 0, 'used': 0, 'total': 0}
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        
        return {
            'total_gb': total_memory / 1e9,
            'allocated_gb': allocated_memory / 1e9,
            'cached_gb': cached_memory / 1e9,
            'free_gb': (total_memory - allocated_memory) / 1e9,
            'utilization': allocated_memory / total_memory
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        info = self.get_memory_info()
        return info['allocated_gb'] > self.max_memory_gb
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def log_memory_usage(self, step: int = 0):
        """Log current memory usage."""
        info = self.get_memory_info()
        logger.info(f"Step {step} - GPU Memory: {info['allocated_gb']:.2f}GB / "
                   f"{info['total_gb']:.2f}GB ({info['utilization']*100:.1f}%)")


class BatchSizeOptimizer:
    """Automatic batch size optimization for GTX 1660 Ti."""
    
    def __init__(self, model: nn.Module, sample_input: torch.Tensor, 
                 max_batch_size: int = 32, memory_monitor: MemoryMonitor = None):
        self.model = model
        self.sample_input = sample_input
        self.max_batch_size = max_batch_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
    def find_optimal_batch_size(self) -> int:
        """Find optimal batch size through binary search."""
        logger.info("Finding optimal batch size...")
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        optimal_size = 1
        
        for bs in batch_sizes:
            if bs > self.max_batch_size:
                break
                
            try:
                # Test batch with current size
                test_input = self.sample_input[:bs] if bs <= len(self.sample_input) else \
                           self.sample_input[:1].repeat(bs, *([1] * (len(self.sample_input.shape) - 1)))
                
                # Ensure correct dtype for embedding layers
                if test_input.dtype != self.sample_input.dtype:
                    test_input = test_input.to(self.sample_input.dtype)
                
                # Forward pass test
                with torch.no_grad():
                    output = self.model(test_input)
                
                # Check memory usage
                if self.memory_monitor.check_memory_limit():
                    logger.warning(f"Batch size {bs} exceeds memory limit")
                    break
                
                optimal_size = bs
                logger.info(f"Batch size {bs}: OK")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Batch size {bs}: Out of memory")
                    break
                else:
                    raise e
            finally:
                self.memory_monitor.cleanup_memory()
        
        logger.info(f"Optimal batch size: {optimal_size}")
        return optimal_size


class OptimizedTrainer:
    """GTX 1660 Ti optimized trainer with all optimizations."""
    
    def __init__(self, model: nn.Module, config: GTX1660Config):
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(config.max_memory_usage)
        if config.use_mixed_precision:
            if torch.cuda.is_available():
                self.scaler = GradScaler()
            else:
                self.scaler = GradScaler('cpu')
        else:
            self.scaler = None
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Training state
        self.current_step = 0
        self.accumulation_steps = 0
        
    def _apply_optimizations(self):
        """Apply all GTX 1660 Ti optimizations."""
        logger.info("Applying GTX 1660 Ti optimizations...")
        
        # 1. torch.compile optimization (Triton-safe for GTX 1660 Ti)
        if self.config.use_torch_compile and torch.__version__ >= "2.0":
            try:
                # Disable Triton for GTX 1660 Ti compatibility
                import os
                old_triton = os.environ.get('TORCHINDUCTOR_TRITON', '1')
                os.environ['TORCHINDUCTOR_TRITON'] = '0'
                
                self.model = torch.compile(
                    self.model,
                    mode=self.config.compile_mode,
                    fullgraph=self.config.compile_fullgraph
                )
                
                # Restore original setting
                if old_triton:
                    os.environ['TORCHINDUCTOR_TRITON'] = old_triton
                else:
                    os.environ.pop('TORCHINDUCTOR_TRITON', None)
                
                logger.info("✓ torch.compile optimization applied (Triton disabled for GTX 1660 Ti)")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                logger.info("💡 Try setting use_torch_compile=False in config")
        
        # 2. Find optimal batch size
        if self.config.auto_batch_size:
            # Create sample input for batch size testing based on model type
            if hasattr(self.model, 'embedding'):  # Check if model has embedding layer
                # For models with embedding layers (BERT, LSTM), use long tensors
                sample_input = torch.randint(0, 1000, (1, 128)).to(self.device).long()
            else:
                # For CNN models, use float tensors
                sample_input = torch.randn(1, 3, 224, 224).to(self.device).float()
            
            batch_optimizer = BatchSizeOptimizer(
                self.model, sample_input, self.config.max_batch_size, self.memory_monitor
            )
            optimal_batch_size = batch_optimizer.find_optimal_batch_size()
            
            # Update accumulation steps
            if self.config.accumulation_steps is None:
                self.config.accumulation_steps = max(1, self.config.target_batch_size // optimal_batch_size)
            
            logger.info(f"✓ Optimal batch size: {optimal_batch_size}")
            logger.info(f"✓ Accumulation steps: {self.config.accumulation_steps}")
        
        # 3. Create optimizer
        self.optimizer = self._create_optimizer()
        
        logger.info("✓ All optimizations applied")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create memory-efficient optimizer."""
        if self.config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                logger.info("Using 8-bit optimizer")
                return bnb.optim.Adam8bit(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                logger.warning("bitsandbytes not available, using standard optimizer")
        
        # Standard optimizer with memory optimization
        if self.config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
                foreach=True  # Memory optimization
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def train_epoch(self, dataloader, criterion, epoch: int = 0):
        """Train one epoch with all optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Reset accumulation
        self.optimizer.zero_grad()
        self.accumulation_steps = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with optimizations
            loss = self._forward_pass(data, target, criterion)
            
            # Accumulate gradients
            self._accumulate_gradients(loss)
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Memory monitoring and cleanup
            if self.config.memory_monitoring and batch_idx % self.config.cleanup_frequency == 0:
                self.memory_monitor.log_memory_usage(self.current_step)
                if self.memory_monitor.check_memory_limit():
                    logger.warning("Memory limit exceeded, cleaning up...")
                    self.memory_monitor.cleanup_memory()
            
            self.current_step += 1
        
        # Log epoch results
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def _forward_pass(self, data: torch.Tensor, target: torch.Tensor, 
                      criterion: nn.Module) -> torch.Tensor:
        """Optimized forward pass."""
        if self.config.use_mixed_precision:
            with autocast():
                output = self.model(data)
                loss = criterion(output, target)
                # Normalize loss for gradient accumulation
                loss = loss / self.config.accumulation_steps
        else:
            output = self.model(data)
            loss = criterion(output, target)
            loss = loss / self.config.accumulation_steps
        
        return loss
    
    def _accumulate_gradients(self, loss: torch.Tensor):
        """Accumulate gradients with mixed precision support."""
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accumulation_steps += 1
        
        # Update weights after accumulation
        if self.accumulation_steps >= self.config.accumulation_steps:
            self._update_weights()
    
    def _update_weights(self):
        """Update model weights with mixed precision support."""
        if self.config.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.accumulation_steps = 0
    
    def validate(self, dataloader, criterion) -> float:
        """Validate model with memory monitoring."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def train(self, train_loader, val_loader, criterion, epochs: int):
        """Complete training loop with all optimizations."""
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.config.use_mixed_precision}")
        logger.info(f"Gradient Checkpointing: {self.config.use_gradient_checkpointing}")
        logger.info(f"Torch Compile: {self.config.use_torch_compile}")
        
        # Log initial memory usage
        self.memory_monitor.log_memory_usage()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion, epoch)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader, criterion)
            val_losses.append(val_loss)
            
            # Memory cleanup
            if epoch % 10 == 0:
                self.memory_monitor.cleanup_memory()
        
        logger.info("Training completed!")
        return train_losses, val_losses


class CheckpointedModule(nn.Module):
    """Module wrapper with gradient checkpointing support."""
    
    def __init__(self, module: nn.Module, use_checkpointing: bool = True):
        super().__init__()
        self.module = module
        self.use_checkpointing = use_checkpointing
    
    def forward(self, *args, **kwargs):
        if self.use_checkpointing and self.training:
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)


def create_optimized_model(model: nn.Module, config: GTX1660Config) -> nn.Module:
    """Create optimized model with checkpointing support."""
    if config.use_gradient_checkpointing:
        # Wrap specific layers with checkpointing
        if hasattr(model, 'features'):  # CNN models
            model.features = CheckpointedModule(model.features, config.use_gradient_checkpointing)
        elif hasattr(model, 'encoder'):  # Transformer models
            model.encoder = CheckpointedModule(model.encoder, config.use_gradient_checkpointing)
    
    return model


# Predefined configurations for different model types
def get_resnet_config() -> GTX1660Config:
    """Optimized configuration for ResNet models."""
    return GTX1660Config(
        target_batch_size=32,
        max_batch_size=32,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        use_torch_compile=True,
        use_8bit_optimizer=False,
        auto_batch_size=True,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_memory_usage=4.5
    )


def get_bert_config() -> GTX1660Config:
    """Optimized configuration for BERT models."""
    return GTX1660Config(
        target_batch_size=16,
        max_batch_size=16,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        use_torch_compile=True,
        use_8bit_optimizer=True,
        auto_batch_size=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_memory_usage=5.2
    )


def get_llm_config() -> GTX1660Config:
    """Optimized configuration for LLM fine-tuning."""
    return GTX1660Config(
        target_batch_size=8,
        max_batch_size=8,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        use_torch_compile=True,
        use_8bit_optimizer=True,
        auto_batch_size=True,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_memory_usage=5.8
    )


# Example usage and training script
def example_training_script():
    """Example training script for GTX 1660 Ti."""
    
    # Import your model here
    # from your_model import YourModel
    
    # Create model
    # model = YourModel()
    
    # For demonstration, create a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN()
    
    # Get optimized configuration
    config = get_resnet_config()
    
    # Create optimized trainer
    trainer = OptimizedTrainer(model, config)
    
    # Create dummy data loaders (replace with your actual data)
    def create_dummy_dataloader(batch_size, num_samples=1000):
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        data = torch.randn(num_samples, 3, 224, 224)
        targets = torch.randint(0, 10, (num_samples,))
        
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_loader = create_dummy_dataloader(config.max_batch_size // config.accumulation_steps or 1)
    val_loader = create_dummy_dataloader(config.max_batch_size // config.accumulation_steps or 1, num_samples=200)
    
    # Define criterion
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    train_losses, val_losses = trainer.train(train_loader, val_loader, criterion, epochs=5)
    
    print("Training completed successfully!")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")


if __name__ == "__main__":
    example_training_script()
