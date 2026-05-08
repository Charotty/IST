"""
GTX 1660 Ti Checkpoint Manager

Advanced checkpointing and resumption system for training on GTX 1660 Ti.
Supports model saving, loading, automatic resumption, and memory-efficient checkpointing.

Author: AI Assistant
Created: 2024
"""

import torch
import torch.nn as nn
import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import pickle
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata structure."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    model_name: str
    optimizer_type: str
    batch_size: int
    accumulation_steps: int
    best_metric: float
    training_time_hours: float
    config_hash: str
    pytorch_version: str
    cuda_version: str
    gpu_name: str
    memory_usage_gb: float


class CheckpointManager:
    """Advanced checkpoint management for GTX 1660 Ti training."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 max_checkpoints: int = 5, save_interval: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        
        # Current checkpoint info
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.training_start_time = time.time()
        
        # Checkpoint history
        self.checkpoint_history: List[CheckpointMetadata] = []
        
        logger.info(f"✓ Checkpoint manager initialized")
        logger.info(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"💾 Max checkpoints: {max_checkpoints}")
        logger.info(f"⏰ Save interval: {save_interval} steps")
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], loss: float, epoch: int, step: int,
                       config: Dict[str, Any], additional_data: Optional[Dict] = None,
                       is_best: bool = False) -> str:
        """Save comprehensive checkpoint with metadata."""
        
        # Update current state
        self.current_epoch = epoch
        self.current_step = step
        
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Create metadata
        metadata = CheckpointMetadata(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            model_name=config.get('model_name', 'unknown'),
            optimizer_type=type(optimizer).__name__,
            batch_size=config.get('batch_size', 0),
            accumulation_steps=config.get('accumulation_steps', 0),
            best_metric=self.best_metric,
            training_time_hours=(time.time() - self.training_start_time) / 3600,
            config_hash=self._hash_config(config),
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else "N/A",
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            memory_usage_gb=self._get_memory_usage()
        )
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': self._prepare_model_state(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metadata': asdict(metadata),
            'config': config,
            'additional_data': additional_data or {}
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save metadata separately for quick inspection
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Update history
            self.checkpoint_history.append(metadata)
            
            # Save best checkpoint
            if is_best or loss < self.best_metric:
                self.best_metric = loss
                best_path = self.checkpoint_dir / "best_checkpoint.pth"
                shutil.copy2(checkpoint_path, best_path)
                
                # Also save metadata
                best_meta_path = best_path.with_suffix('.json')
                shutil.copy2(metadata_path, best_meta_path)
                
                logger.info(f"✓ New best checkpoint saved (loss: {loss:.6f})")
            
            # Manage checkpoint count
            self._manage_checkpoint_count()
            
            logger.info(f"✓ Checkpoint saved: {checkpoint_name}")
            logger.info(f"📊 Loss: {loss:.6f}, Epoch: {epoch}, Step: {step}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None,
                       device: Optional[torch.device] = None) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """Load checkpoint and restore training state."""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            
            # Restore model state
            model_state = checkpoint_data['model_state_dict']
            self._restore_model_state(model, model_state)
            
            # Restore optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Restore scheduler state
            if scheduler and checkpoint_data['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Extract metadata
            metadata = CheckpointMetadata(**checkpoint_data['metadata'])
            
            # Update current state
            self.current_epoch = metadata.epoch
            self.current_step = metadata.step
            self.best_metric = metadata.best_metric
            
            logger.info(f"✓ Checkpoint loaded: {checkpoint_path.name}")
            logger.info(f"📊 Epoch: {metadata.epoch}, Step: {metadata.step}, Loss: {metadata.loss:.6f}")
            logger.info(f"⏰ Training time: {metadata.training_time_hours:.2f} hours")
            
            return checkpoint_data, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def auto_resume(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: Optional[Any] = None, device: Optional[torch.device] = None) -> bool:
        """Automatically resume from latest checkpoint."""
        
        # Find latest checkpoint
        latest_checkpoint = self._find_latest_checkpoint()
        
        if latest_checkpoint:
            try:
                logger.info(f"🔄 Auto-resuming from {latest_checkpoint.name}")
                self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler, device)
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-resume: {e}")
                return False
        else:
            logger.info("ℹ️ No checkpoint found, starting fresh")
            return False
    
    def should_save(self, step: int, epoch: int, loss: float, 
                   force_save: bool = False) -> bool:
        """Determine if checkpoint should be saved."""
        if force_save:
            return True
        
        # Save at regular intervals
        if step % self.save_interval == 0:
            return True
        
        # Save at epoch boundaries
        if step == 0:  # End of epoch
            return True
        
        # Save for significant loss improvement
        if loss < self.best_metric * 0.95:  # 5% improvement
            return True
        
        return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint information."""
        checkpoints = self._list_checkpoints()
        
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(checkpoints),
            'latest_checkpoint': checkpoints[0].name if checkpoints else None,
            'best_checkpoint': "best_checkpoint.pth" if (self.checkpoint_dir / "best_checkpoint.pth").exists() else None,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_metric': self.best_metric,
            'training_time_hours': (time.time() - self.training_start_time) / 3600,
            'checkpoint_history': [asdict(meta) for meta in self.checkpoint_history[-5:]]  # Last 5
        }
    
    def cleanup_old_checkpoints(self, keep_last_n: int = None):
        """Clean up old checkpoints, keeping only the most recent ones."""
        keep_count = keep_last_n or self.max_checkpoints
        checkpoints = self._list_checkpoints()
        
        if len(checkpoints) > keep_count:
            # Remove oldest checkpoints
            for checkpoint in checkpoints[keep_count:]:
                try:
                    checkpoint.unlink()
                    metadata_path = checkpoint.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    logger.info(f"🗑️ Removed old checkpoint: {checkpoint.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {checkpoint.name}: {e}")
    
    def export_checkpoint(self, checkpoint_path: str, export_path: str, 
                          include_optimizer: bool = False):
        """Export checkpoint in a portable format."""
        checkpoint_data, metadata = self.load_checkpoint(checkpoint_path, None, None)
        
        # Prepare export data
        export_data = {
            'model_state_dict': checkpoint_data['model_state_dict'],
            'metadata': asdict(metadata),
            'config': checkpoint_data['config']
        }
        
        if include_optimizer:
            export_data['optimizer_state_dict'] = checkpoint_data['optimizer_state_dict']
        
        # Save export
        torch.save(export_data, export_path)
        logger.info(f"✓ Checkpoint exported to {export_path}")
    
    def _prepare_model_state(self, model: nn.Module) -> Dict[str, Any]:
        """Prepare model state for saving, handling special cases."""
        state_dict = model.state_dict()
        
        # Handle special cases (e.g., LoRA adapters, distributed models)
        if hasattr(model, 'peft_config') and hasattr(model, 'save_pretrained'):
            # PEFT model (LoRA)
            return {
                'peft_state_dict': state_dict,
                'is_peft_model': True
            }
        else:
            return {
                'state_dict': state_dict,
                'is_peft_model': False
            }
    
    def _restore_model_state(self, model: nn.Module, model_state: Dict[str, Any]):
        """Restore model state from saved checkpoint."""
        if model_state.get('is_peft_model', False):
            # PEFT model
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(model_state['peft_state_dict'])
        else:
            # Regular model
            model.load_state_dict(model_state['state_dict'])
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoint files, sorted by modification time."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = self._list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def _manage_checkpoint_count(self):
        """Manage checkpoint count by removing old ones."""
        checkpoints = self._list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            # Keep the latest N checkpoints
            checkpoints_to_remove = checkpoints[self.max_checkpoints:]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    metadata_path = checkpoint.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {checkpoint.name}: {e}")


class TrainingState:
    """Training state management for resumption."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.start_epoch = 0
        self.start_step = 0
        self.best_metric = float('inf')
        self.metrics_history = []
    
    def initialize(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  scheduler: Optional[Any] = None, device: Optional[torch.device] = None):
        """Initialize training state from checkpoint."""
        if self.checkpoint_manager.auto_resume(model, optimizer, scheduler, device):
            # Successfully resumed
            self.start_epoch = self.checkpoint_manager.current_epoch
            self.start_step = self.checkpoint_manager.current_step
            self.best_metric = self.checkpoint_manager.best_metric
            logger.info(f"✅ Training resumed from epoch {self.start_epoch}, step {self.start_step}")
        else:
            # Starting fresh
            self.start_epoch = 0
            self.start_step = 0
            self.best_metric = float('inf')
            logger.info("🆕 Starting fresh training")
    
    def update_metrics(self, loss: float, epoch: int, step: int):
        """Update training metrics."""
        self.metrics_history.append({
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get training progress information."""
        return {
            'start_epoch': self.start_epoch,
            'start_step': self.start_step,
            'current_epoch': self.checkpoint_manager.current_epoch,
            'current_step': self.checkpoint_manager.current_step,
            'best_metric': self.best_metric,
            'total_steps': len(self.metrics_history),
            'training_time_hours': (time.time() - self.checkpoint_manager.training_start_time) / 3600
        }


# Integration with OptimizedTrainer
class EnhancedOptimizedTrainer:
    """Enhanced trainer with checkpoint management."""
    
    def __init__(self, model: nn.Module, config, checkpoint_dir: str = "./checkpoints"):
        # Import here to avoid circular imports
        from gtx1660_optimizer import OptimizedTrainer
        
        # Create base trainer
        self.trainer = OptimizedTrainer(model, config)
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Create training state
        self.training_state = TrainingState(self.checkpoint_manager)
        
        # Initialize training state
        self.training_state.initialize(
            model, self.trainer.optimizer, device=self.trainer.device
        )
    
    def train_with_checkpoints(self, train_loader, val_loader, criterion, epochs: int,
                             save_interval: int = 1000, resume: bool = True):
        """Train with automatic checkpointing."""
        
        logger.info(f"🚀 Starting training with checkpoints")
        logger.info(f"📁 Checkpoint directory: {self.checkpoint_manager.checkpoint_dir}")
        logger.info(f"💾 Save interval: {save_interval} steps")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.training_state.start_epoch, epochs):
            # Training
            train_loss = self._train_epoch_with_checkpoints(
                train_loader, criterion, epoch, save_interval
            )
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.trainer.validate(val_loader, criterion)
            val_losses.append(val_loss)
            
            # Update metrics
            self.training_state.update_metrics(val_loss, epoch, self.trainer.current_step)
            
            # Log progress
            progress = self.training_state.get_progress()
            logger.info(f"📊 Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(f"⏰ Training time: {progress['training_time_hours']:.2f} hours")
        
        return train_losses, val_losses
    
    def _train_epoch_with_checkpoints(self, dataloader, criterion, epoch: int, 
                                     save_interval: int):
        """Train one epoch with checkpointing."""
        self.trainer.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Reset accumulation
        self.trainer.optimizer.zero_grad()
        self.trainer.accumulation_steps = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Skip if we're resuming and haven't reached the starting step
            if epoch == self.training_state.start_epoch and batch_idx < self.training_state.start_step:
                continue
            
            data, target = data.to(self.trainer.device), target.to(self.trainer.device)
            
            # Forward pass
            loss = self.trainer._forward_pass(data, target, criterion)
            
            # Accumulate gradients
            self.trainer._accumulate_gradients(loss)
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Checkpoint saving logic
            current_step = epoch * len(dataloader) + batch_idx
            if self.checkpoint_manager.should_save(current_step, epoch, loss.item()):
                self._save_training_checkpoint(loss.item(), epoch, batch_idx)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def _save_training_checkpoint(self, loss: float, epoch: int, step: int):
        """Save training checkpoint."""
        try:
            self.checkpoint_manager.save_checkpoint(
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                scheduler=None,  # Add scheduler if available
                loss=loss,
                epoch=epoch,
                step=step,
                config={},  # Add actual config
                is_best=(loss < self.training_state.best_metric)
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save checkpoint: {e}")


if __name__ == "__main__":
    # Demo checkpoint functionality
    print("🎯 Checkpoint Manager Demo")
    
    # Create a simple model for testing
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager("./demo_checkpoints", max_checkpoints=3)
    
    # Save a checkpoint
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss=0.5,
        epoch=1,
        step=100,
        config={'model_name': 'test_model', 'batch_size': 32}
    )
    
    # Get checkpoint info
    info = checkpoint_manager.get_checkpoint_info()
    print(f"📊 Checkpoint info: {info}")
    
    print("✅ Checkpoint manager demo completed!")
