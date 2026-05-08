"""
ResNet Training Example for GTX 1660 Ti

Optimized ResNet training script demonstrating all GTX 1660 Ti optimizations.
This example shows how to train image classification models efficiently on 6GB VRAM.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gtx1660_optimizer import (
    OptimizedTrainer, GTX1660Config, get_resnet_config,
    create_optimized_model, MemoryMonitor
)


class OptimizedResNetTrainer:
    """Specialized trainer for ResNet models on GTX 1660 Ti."""
    
    def __init__(self, model_name: str = "resnet50", num_classes: int = 1000):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Get GTX 1660 Ti optimized configuration
        self.config = get_resnet_config()
        
        # Create and optimize model
        self.model = self._create_model()
        
        # Create trainer
        self.trainer = OptimizedTrainer(self.model, self.config)
        
        # Setup data transforms
        self.transform = self._get_transforms()
        
        print(f"✓ Initialized {model_name} for GTX 1660 Ti")
        print(f"✓ Target batch size: {self.config.target_batch_size}")
        print(f"✓ Mixed precision: {self.config.use_mixed_precision}")
        print(f"✓ Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        print(f"✓ Torch compile: {self.config.use_torch_compile}")
    
    def _create_model(self) -> nn.Module:
        """Create and optimize ResNet model."""
        # Load pretrained ResNet
        if self.model_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == "resnet34":
            model = torchvision.models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == "resnet101":
            model = torchvision.models.resnet101(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Apply GTX 1660 Ti optimizations
        model = create_optimized_model(model, self.config)
        
        return model
    
    def _get_transforms(self):
        """Get data transforms for training."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def create_data_loaders(self, data_dir: str, batch_size: int = None):
        """Create data loaders for training."""
        if batch_size is None:
            # Use optimized batch size
            batch_size = self.config.max_batch_size // self.config.accumulation_steps or 1
        
        # Create datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=self.transform
        )
        
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=self.transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"✓ Created data loaders")
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Accumulation steps: {self.config.accumulation_steps}")
        print(f"✓ Effective batch size: {batch_size * self.config.accumulation_steps}")
        
        return train_loader, val_loader
    
    def train(self, data_dir: str, epochs: int = 10, save_dir: str = None):
        """Train the model with GTX 1660 Ti optimizations."""
        print(f"\n🚀 Starting {self.model_name} training on GTX 1660 Ti")
        print(f"📊 Epochs: {epochs}")
        print(f"📁 Data directory: {data_dir}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(data_dir)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        start_time = time.time()
        train_losses, val_losses = self.trainer.train(
            train_loader, val_loader, criterion, epochs
        )
        training_time = time.time() - start_time
        
        # Save model
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{self.model_name}_gtx1660ti.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'training_time': training_time
            }, model_path)
            print(f"✓ Model saved to {model_path}")
        
        # Final statistics
        print(f"\n📈 Training completed!")
        print(f"⏱️  Total time: {training_time:.2f} seconds")
        print(f"📊 Final train loss: {train_losses[-1]:.6f}")
        print(f"📊 Final val loss: {val_losses[-1]:.6f}")
        
        # Memory usage report
        memory_info = self.trainer.memory_monitor.get_memory_info()
        print(f"💾 Peak memory usage: {memory_info['allocated_gb']:.2f} GB")
        
        return train_losses, val_losses


def demo_training():
    """Demonstration of ResNet training with synthetic data."""
    print("🎯 Running ResNet training demo with synthetic data...")
    
    # Create trainer
    trainer = OptimizedResNetTrainer("resnet18", num_classes=10)
    
    # Create synthetic data loaders
    def create_synthetic_dataloader(batch_size, num_samples=1000):
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create synthetic image data (224x224x3)
        data = torch.randn(num_samples, 3, 224, 224)
        targets = torch.randint(0, 10, (num_samples,))
        
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate optimal batch size
    batch_size = trainer.config.max_batch_size // trainer.config.accumulation_steps or 1
    train_loader = create_synthetic_dataloader(batch_size, 1000)
    val_loader = create_synthetic_dataloader(batch_size, 200)
    
    # Define criterion
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print(f"🏋️  Training with batch size {batch_size}, "
          f"accumulation steps {trainer.config.accumulation_steps}")
    
    train_losses, val_losses = trainer.trainer.train(
        train_loader, val_loader, criterion, epochs=3
    )
    
    print("✅ Demo training completed successfully!")


def real_data_training(data_dir: str, model_name: str = "resnet18", epochs: int = 10):
    """Train ResNet on real image data."""
    print(f"🎯 Training {model_name} on real data from {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory {data_dir} does not exist")
        print("💡 Please provide a valid data directory with train/val subdirectories")
        return
    
    # Create trainer
    trainer = OptimizedResNetTrainer(model_name, num_classes=10)
    
    # Train model
    train_losses, val_losses = trainer.train(
        data_dir=data_dir,
        epochs=epochs,
        save_dir="./checkpoints"
    )
    
    return train_losses, val_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ResNet Training on GTX 1660 Ti")
    parser.add_argument("--data", type=str, help="Data directory path")
    parser.add_argument("--model", type=str, default="resnet18", 
                       choices=["resnet18", "resnet34", "resnet50", "resnet101"],
                       help="ResNet model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_training()
    elif args.data:
        real_data_training(args.data, args.model, args.epochs)
    else:
        print("Please provide --data path or use --demo for synthetic data training")
        print("Example: python resnet_training.py --data ./imagenet --model resnet50 --epochs 20")
        print("Example: python resnet_training.py --demo")
