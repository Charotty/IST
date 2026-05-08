"""
BERT Training Example for GTX 1660 Ti

Optimized BERT training script demonstrating GTX 1660 Ti optimizations
for NLP models. Shows how to fine-tune transformers efficiently on 6GB VRAM.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import numpy as np
import time
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gtx1660_optimizer import (
    OptimizedTrainer, GTX1660Config, get_bert_config,
    create_optimized_model, MemoryMonitor
)


class TextDataset(Dataset):
    """Custom dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class OptimizedBERTTrainer:
    """Specialized trainer for BERT models on GTX 1660 Ti."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Get GTX 1660 Ti optimized configuration for BERT
        self.config = get_bert_config()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create and optimize model
        self.model = self._create_model()
        
        # Create trainer
        self.trainer = OptimizedTrainer(self.model, self.config)
        
        print(f"✓ Initialized {model_name} for GTX 1660 Ti")
        print(f"✓ Target batch size: {self.config.target_batch_size}")
        print(f"✓ Mixed precision: {self.config.use_mixed_precision}")
        print(f"✓ Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        print(f"✓ 8-bit optimizer: {self.config.use_8bit_optimizer}")
        print(f"✓ Sequence length: 512")
    
    def _create_model(self) -> nn.Module:
        """Create and optimize BERT model."""
        # Load pretrained BERT
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        # Apply gradient checkpointing to BERT layers
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'bert'):
                model.bert.gradient_checkpointing_enable()
            elif hasattr(model, 'roberta'):
                model.roberta.gradient_checkpointing_enable()
        
        # Apply GTX 1660 Ti optimizations
        model = create_optimized_model(model, self.config)
        
        return model
    
    def create_data_loaders(self, texts, labels, batch_size: int = None, 
                           max_length: int = 512, validation_split: float = 0.2):
        """Create data loaders for training."""
        if batch_size is None:
            # Use optimized batch size
            batch_size = self.config.max_batch_size // self.config.accumulation_steps or 1
        
        # Split data
        total_samples = len(texts)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size
        
        # Shuffle indices
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"✓ Created data loaders")
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Accumulation steps: {self.config.accumulation_steps}")
        print(f"✓ Effective batch size: {batch_size * self.config.accumulation_steps}")
        print(f"✓ Max sequence length: {max_length}")
        
        return train_loader, val_loader
    
    def train(self, texts, labels, epochs: int = 3, max_length: int = 512, 
              save_dir: str = None):
        """Train the model with GTX 1660 Ti optimizations."""
        print(f"\n🚀 Starting {self.model_name} training on GTX 1660 Ti")
        print(f"📊 Epochs: {epochs}")
        print(f"📝 Samples: {len(texts)}")
        print(f"📏 Max sequence length: {max_length}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            texts, labels, max_length=max_length
        )
        
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
            model_path = os.path.join(save_dir, f"{self.model_name.replace('/', '_')}_gtx1660ti.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'training_time': training_time,
                'model_name': self.model_name
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


def create_synthetic_text_data(num_samples: int = 1000, num_classes: int = 2):
    """Create synthetic text data for demonstration."""
    texts = []
    labels = []
    
    # Generate synthetic text samples
    templates = [
        "This is a sample text about topic {}.",
        "The main subject here is {}.",
        "We are discussing {} in this example.",
        "This document contains information about {}.",
        "The content focuses on {} aspects."
    ]
    
    topics = [f"topic_{i}" for i in range(num_classes)]
    
    for i in range(num_samples):
        label = i % num_classes
        template = np.random.choice(templates)
        text = template.format(topics[label])
        
        # Add some random words to increase length
        extra_words = ["additional", "information", "details", "data", "analysis", 
                      "research", "study", "report", "summary", "conclusion"]
        text += " " + " ".join(np.random.choice(extra_words, 5))
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def demo_training():
    """Demonstration of BERT training with synthetic data."""
    print("🎯 Running BERT training demo with synthetic data...")
    
    # Create synthetic data
    texts, labels = create_synthetic_text_data(num_samples=500, num_classes=2)
    
    # Create trainer
    trainer = OptimizedBERTTrainer("bert-base-uncased", num_classes=2)
    
    # Train model
    print(f"🏋️  Training with 500 samples, 2 classes")
    
    train_losses, val_losses = trainer.train(
        texts=texts,
        labels=labels,
        epochs=2,
        max_length=256,  # Shorter sequences for demo
        save_dir="./checkpoints"
    )
    
    print("✅ Demo training completed successfully!")


def real_data_training(texts, labels, model_name: str = "bert-base-uncased", 
                      epochs: int = 3, max_length: int = 512):
    """Train BERT on real text data."""
    print(f"🎯 Training {model_name} on {len(texts)} samples")
    
    # Create trainer
    trainer = OptimizedBERTTrainer(model_name, num_classes=len(set(labels)))
    
    # Train model
    train_losses, val_losses = trainer.train(
        texts=texts,
        labels=labels,
        epochs=epochs,
        max_length=max_length,
        save_dir="./checkpoints"
    )
    
    return train_losses, val_losses


def load_text_data(file_path: str):
    """Load text data from file (CSV format expected)."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Convert labels to integers if needed
        label_map = {label: idx for idx, label in enumerate(set(labels))}
        labels = [label_map[label] for label in labels]
        
        print(f"✓ Loaded {len(texts)} samples from {file_path}")
        print(f"✓ Number of classes: {len(label_map)}")
        
        return texts, labels, label_map
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT Training on GTX 1660 Ti")
    parser.add_argument("--data", type=str, help="CSV file path with 'text' and 'label' columns")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="BERT model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_training()
    elif args.data:
        result = load_text_data(args.data)
        if result[0] is not None:
            texts, labels, label_map = result
            real_data_training(texts, labels, args.model, args.epochs, args.max_length)
    else:
        print("Please provide --data path or use --demo for synthetic data training")
        print("Example: python bert_training.py --data ./data/text_data.csv --model bert-base-uncased --epochs 5")
        print("Example: python bert_training.py --demo")
