"""
LLM Fine-tuning Example for GTX 1660 Ti

Optimized LLM fine-tuning script demonstrating GTX 1660 Ti optimizations
for large language models. Shows how to fine-tune 7B parameter models efficiently on 6GB VRAM.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import time
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gtx1660_optimizer import (
    OptimizedTrainer, GTX1660Config, get_llm_config,
    create_optimized_model, MemoryMonitor
)


class TextDataset(Dataset):
    """Custom dataset for LLM fine-tuning."""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
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
            'labels': encoding['input_ids'].flatten()  # For causal LM
        }


class OptimizedLLMTrainer:
    """Specialized trainer for LLM fine-tuning on GTX 1660 Ti."""
    
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        
        # Get GTX 1660 Ti optimized configuration for LLM
        self.config = get_llm_config()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create and optimize model
        self.model = self._create_model()
        
        # Create trainer
        self.trainer = OptimizedTrainer(self.model, self.config)
        
        print(f"✓ Initialized {model_name} for GTX 1660 Ti")
        print(f"✓ Target batch size: {self.config.target_batch_size}")
        print(f"✓ Mixed precision: {self.config.use_mixed_precision}")
        print(f"✓ Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        print(f"✓ 8-bit optimizer: {self.config.use_8bit_optimizer}")
        print(f"✓ Max sequence length: {max_length}")
        print(f"✓ Using LoRA fine-tuning for memory efficiency")
    
    def _create_model(self) -> nn.Module:
        """Create and optimize LLM with LoRA."""
        # Load pretrained model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32,
            device_map='auto'  # Automatic device mapping
        )
        
        # Apply gradient checkpointing
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        # Apply LoRA for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Low rank dimension
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention modules
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Apply GTX 1660 Ti optimizations
        model = create_optimized_model(model, self.config)
        
        return model
    
    def create_data_loaders(self, texts, batch_size: int = None):
        """Create data loaders for training."""
        if batch_size is None:
            # Use very small batch size for LLM
            batch_size = 1  # LLM requires minimal batch size on 6GB VRAM
        
        # Create dataset
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        
        # Create data loader with minimal workers to save memory
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,  # Minimal workers for memory efficiency
            pin_memory=True,
            drop_last=True
        )
        
        print(f"✓ Created data loader")
        print(f"✓ Training samples: {len(dataset)}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Accumulation steps: {self.config.accumulation_steps}")
        print(f"✓ Effective batch size: {batch_size * self.config.accumulation_steps}")
        
        return data_loader
    
    def train(self, texts, epochs: int = 1, save_dir: str = None):
        """Train the model with GTX 1660 Ti optimizations."""
        print(f"\n🚀 Starting {self.model_name} fine-tuning on GTX 1660 Ti")
        print(f"📊 Epochs: {epochs}")
        print(f"📝 Samples: {len(texts)}")
        print(f"📏 Max sequence length: {self.max_length}")
        
        # Create data loader
        data_loader = self.create_data_loaders(texts)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        start_time = time.time()
        
        # For LLM, we'll use a simpler training loop due to memory constraints
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            # Reset accumulation
            self.trainer.optimizer.zero_grad()
            self.trainer.accumulation_steps = 0
            
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.trainer.device)
                attention_mask = batch['attention_mask'].to(self.trainer.device)
                labels = batch['labels'].to(self.trainer.device)
                
                # Forward pass with optimizations
                loss = self._forward_pass(input_ids, attention_mask, labels, criterion)
                
                # Accumulate gradients
                self._accumulate_gradients(loss)
                
                # Update statistics
                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    memory_info = self.trainer.memory_monitor.get_memory_info()
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.6f}, "
                          f"Memory: {memory_info['allocated_gb']:.2f}GB")
                
                self.trainer.current_step += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Save model
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{self.model_name.replace('/', '_')}_lora_gtx1660ti.pth")
            
            # Save LoRA adapter
            self.model.save_pretrained(save_dir)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'training_time': training_time,
                'model_name': self.model_name
            }, model_path)
            print(f"✓ Model saved to {model_path}")
        
        # Final statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"\n📈 Fine-tuning completed!")
        print(f"⏱️  Total time: {training_time:.2f} seconds")
        print(f"📊 Final loss: {avg_loss:.6f}")
        
        # Memory usage report
        memory_info = self.trainer.memory_monitor.get_memory_info()
        print(f"💾 Peak memory usage: {memory_info['allocated_gb']:.2f} GB")
        
        return [avg_loss]
    
    def _forward_pass(self, input_ids, attention_mask, labels, criterion):
        """Optimized forward pass for LLM."""
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                # Normalize loss for gradient accumulation
                loss = loss / self.config.accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss = loss / self.config.accumulation_steps
        
        return loss
    
    def _accumulate_gradients(self, loss):
        """Accumulate gradients with mixed precision support."""
        if self.config.use_mixed_precision:
            self.trainer.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.trainer.accumulation_steps += 1
        
        # Update weights after accumulation
        if self.trainer.accumulation_steps >= self.config.accumulation_steps:
            self._update_weights()
    
    def _update_weights(self):
        """Update model weights with mixed precision support."""
        if self.config.use_mixed_precision:
            self.trainer.scaler.step(self.trainer.optimizer)
            self.trainer.scaler.update()
        else:
            self.trainer.optimizer.step()
        
        self.trainer.optimizer.zero_grad()
        self.trainer.accumulation_steps = 0


def create_synthetic_text_data(num_samples: int = 100):
    """Create synthetic text data for LLM fine-tuning."""
    texts = []
    
    # Generate synthetic text samples
    templates = [
        "The weather today is {} and the temperature is {} degrees.",
        "In the news today, there was a report about {}.",
        "The main character in the story is named {}.",
        "The company announced that they will {} next quarter.",
        "Scientists have discovered a new {} that could change everything.",
        "The restaurant serves the best {} in town.",
        "Students are learning about {} in their science class.",
        "The movie features a {} who saves the day.",
        "Technology has advanced to allow {} to become possible.",
        "The book explains the concept of {} in detail."
    ]
    
    for i in range(num_samples):
        template = np.random.choice(templates)
        
        # Fill in template with random words
        words = ["sunny", "cloudy", "rainy", "snowy", "windy"]
        temps = ["15", "20", "25", "30", "35"]
        subjects = ["economy", "technology", "health", "environment", "politics"]
        characters = ["hero", "scientist", "detective", "teacher", "doctor"]
        actions = ["expand", "launch", "release", "develop", "improve"]
        discoveries = ["species", "element", "planet", "disease", "treatment"]
        foods = ["pizza", "pasta", "sushi", "tacos", "burgers"]
        topics = ["physics", "chemistry", "biology", "astronomy", "geology"]
        roles = ["protagonist", "villain", "mentor", "sidekick", "narrator"]
        innovations = ["AI", "space travel", "renewable energy", "quantum computing", "gene editing"]
        concepts = ["gravity", "evolution", "relativity", "democracy", "capitalism"]
        
        # Randomly fill template
        text = template.format(
            np.random.choice(words),
            np.random.choice(temps),
            np.random.choice(subjects),
            np.random.choice(characters),
            np.random.choice(actions),
            np.random.choice(discoveries),
            np.random.choice(foods),
            np.random.choice(topics),
            np.random.choice(roles),
            np.random.choice(innovations),
            np.random.choice(concepts)
        )
        
        texts.append(text)
    
    return texts


def demo_training():
    """Demonstration of LLM fine-tuning with synthetic data."""
    print("🎯 Running LLM fine-tuning demo with synthetic data...")
    
    # Create synthetic data
    texts = create_synthetic_text_data(num_samples=100)
    
    # Create trainer with smaller model for demo
    trainer = OptimizedLLMTrainer("EleutherAI/gpt-neo-125M", max_length=128)
    
    # Train model
    print(f"🏋️  Fine-tuning with 100 samples")
    
    losses = trainer.train(
        texts=texts,
        epochs=1,
        save_dir="./checkpoints"
    )
    
    print("✅ Demo fine-tuning completed successfully!")
    
    # Test generation
    print("\n🔍 Testing generation...")
    test_prompt = "The weather today is"
    inputs = trainer.tokenizer(test_prompt, return_tensors="pt").to(trainer.trainer.device)
    
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=trainer.tokenizer.eos_token_id
        )
    
    generated_text = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


def real_data_finetuning(texts, model_name: str = "EleutherAI/gpt-neo-125M", 
                          epochs: int = 1, max_length: int = 256):
    """Fine-tune LLM on real text data."""
    print(f"🎯 Fine-tuning {model_name} on {len(texts)} samples")
    
    # Create trainer
    trainer = OptimizedLLMTrainer(model_name, max_length=max_length)
    
    # Train model
    losses = trainer.train(
        texts=texts,
        epochs=epochs,
        max_length=max_length,
        save_dir="./checkpoints"
    )
    
    return losses


def load_text_data(file_path: str):
    """Load text data from file (one text per line)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Loaded {len(texts)} samples from {file_path}")
        return texts
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Fine-tuning on GTX 1660 Ti")
    parser.add_argument("--data", type=str, help="Text file path (one text per line)")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125M",
                       help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_training()
    elif args.data:
        texts = load_text_data(args.data)
        if texts is not None:
            real_data_finetuning(texts, args.model, args.epochs, args.max_length)
    else:
        print("Please provide --data path or use --demo for synthetic data fine-tuning")
        print("Example: python llm_finetuning.py --data ./data/text_samples.txt --model EleutherAI/gpt-neo-125M --epochs 2")
        print("Example: python llm_finetuning.py --demo")
        print("\n📝 Note: For LLM fine-tuning on GTX 1660 Ti:")
        print("   - Use smaller models (125M-350M parameters)")
        print("   - Keep sequence length short (128-256 tokens)")
        print("   - Use batch size 1 with gradient accumulation")
        print("   - Enable all memory optimizations")
