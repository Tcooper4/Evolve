"""
Training Loop - Batch 17
Enhanced training loop with gradient clipping for training stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0  # Gradient clipping norm
    clip_gradients: bool = True
    early_stopping_patience: int = 10
    validation_interval: int = 1
    save_interval: int = 5
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    training_time: float = 0.0
    samples_per_second: float = 0.0

class TrainingLoop:
    """
    Enhanced training loop with gradient clipping and stability features.
    
    Features:
    - Gradient clipping before optimizer step
    - Mixed precision training support
    - Gradient accumulation
    - Early stopping
    - Comprehensive metrics tracking
    - Model checkpointing
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize training loop.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = scheduler
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history: List[TrainingMetrics] = []
        
        # Create output directory
        self.output_dir = Path("checkpoints")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Training loop initialized on {config.device}")
        logger.info(f"Gradient clipping: {config.clip_gradients}, max_norm: {config.max_grad_norm}")
    
    def train_epoch(self) -> TrainingMetrics:
        """
        Train for one epoch with gradient clipping.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_start_time = time.time()
        
        total_loss = 0.0
        total_samples = 0
        total_grad_norm = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping before optimizer step
                if self.config.clip_gradients:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.max_grad_norm
                    )
                    total_grad_norm += grad_norm.item()
                else:
                    # Calculate gradient norm without clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=float('inf')
                    )
                    total_grad_norm += grad_norm.item()
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size * self.config.gradient_accumulation_steps
            total_samples += batch_size
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = batch_size / batch_time
                
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}, Grad Norm: {grad_norm.item():.6f}, "
                    f"Samples/sec: {samples_per_sec:.2f}"
                )
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_samples
        avg_grad_norm = total_grad_norm / num_batches
        samples_per_second = total_samples / epoch_time
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            train_loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            gradient_norm=avg_grad_norm,
            training_time=epoch_time,
            samples_per_second=samples_per_second
        )
        
        logger.info(
            f"Epoch {self.current_epoch} completed: "
            f"Train Loss: {avg_loss:.6f}, "
            f"Avg Grad Norm: {avg_grad_norm:.6f}, "
            f"Time: {epoch_time:.2f}s, "
            f"Samples/sec: {samples_per_second:.2f}"
        )
        
        return metrics
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                if self.config.mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        val_loss = total_loss / total_samples
        logger.info(f"Validation Loss: {val_loss:.6f}")
        
        return val_loss
    
    def train(self) -> List[TrainingMetrics]:
        """
        Complete training loop with early stopping.
        
        Returns:
            List of training metrics for all epochs
        """
        logger.info("Starting training loop")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch()
            
            # Validation
            if epoch % self.config.validation_interval == 0:
                val_loss = self.validate()
                metrics.val_loss = val_loss
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pth")
                    logger.info(f"New best validation loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"Validation loss did not improve. Patience: {self.patience_counter}")
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics.val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Store metrics
            self.training_history.append(metrics)
        
        logger.info("Training completed")
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.output_dir / filename
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_history:
            return {}
        
        losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history if m.val_loss > 0]
        grad_norms = [m.gradient_norm for m in self.training_history]
        
        summary = {
            'total_epochs': len(self.training_history),
            'best_train_loss': min(losses),
            'final_train_loss': losses[-1],
            'best_val_loss': min(val_losses) if val_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'avg_gradient_norm': np.mean(grad_norms),
            'max_gradient_norm': max(grad_norms),
            'total_training_time': sum(m.training_time for m in self.training_history),
            'avg_samples_per_second': np.mean([m.samples_per_second for m in self.training_history])
        }
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        try:
            import matplotlib.pyplot as plt
            
            epochs = [m.epoch for m in self.training_history]
            train_losses = [m.train_loss for m in self.training_history]
            val_losses = [m.val_loss for m in self.training_history if m.val_loss > 0]
            grad_norms = [m.gradient_norm for m in self.training_history]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Training loss
            axes[0, 0].plot(epochs, train_losses, label='Train Loss')
            if val_losses:
                val_epochs = [m.epoch for m in self.training_history if m.val_loss > 0]
                axes[0, 0].plot(val_epochs, val_losses, label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Gradient norm
            axes[0, 1].plot(epochs, grad_norms, label='Gradient Norm')
            axes[0, 1].axhline(y=self.config.max_grad_norm, color='r', linestyle='--', label='Clip Threshold')
            axes[0, 1].set_title('Gradient Norm')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate
            lrs = [m.learning_rate for m in self.training_history]
            axes[1, 0].plot(epochs, lrs, label='Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Samples per second
            samples_per_sec = [m.samples_per_second for m in self.training_history]
            axes[1, 1].plot(epochs, samples_per_sec, label='Samples/sec')
            axes[1, 1].set_title('Training Speed')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Samples/sec')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Training curves saved: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")

def create_training_loop(model: nn.Module,
                        config: TrainingConfig,
                        train_loader: DataLoader,
                        val_loader: Optional[DataLoader] = None) -> TrainingLoop:
    """Factory function to create a training loop."""
    return TrainingLoop(model, config, train_loader, val_loader) 