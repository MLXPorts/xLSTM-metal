
"""
Unified training script for xLSTM models (MLX and PyTorch)
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model config
    backend: str = "pytorch"  # "pytorch" or "mlx"
    vocab_size: int = 50257
    num_layers: int = 12
    signature: tuple = (7, 1)  # (num_mLSTM, num_sLSTM)
    inp_dim: int = 768
    head_dim: int = 96
    head_num: int = 8
    dropout: float = 0.1
    
    # Training config
    batch_size: int = 8
    seq_length: int = 512
    learning_rate: float = 3e-4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Data config
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    
    # Output config
    output_dir: str = "./checkpoints"
    experiment_name: str = "xlstm_training"
    
    # Device config
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"
    mixed_precision: bool = True
    
    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class xLSTMTrainer:
    """Unified trainer for xLSTM models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.backend = config.backend
        
        # Setup device
        self._setup_device()
        
        # Create model
        self.model = self._create_model()
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_optimization()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(self.output_dir / "config.json")
    
    def _setup_device(self):
        """Setup compute device"""
        if self.config.device == "auto":
            if self.backend == "pytorch":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:  # mlx
                self.device = "gpu"  # MLX handles device automatically
        else:
            self.device = self.config.device
        
        print(f"Using device: {self.device} with {self.backend} backend")
    
    def _create_model(self):
        """Create model based on backend"""
        if self.backend == "pytorch":
            from xlstm_pytorch import create_xlstm_model
            from xlstm_pytorch_inference import xLSTMInference
            
            base_model = create_xlstm_model(
                vocab_size=self.config.vocab_size,
                num_layers=self.config.num_layers,
                signature=self.config.signature,
                inp_dim=self.config.inp_dim,
                head_dim=self.config.head_dim,
                head_num=self.config.head_num,
                dropout=self.config.dropout,
                device=self.device
            )
            
            # Wrap with inference capabilities
            model = xLSTMInference(base_model)
            
            # Setup mixed precision if requested
            if self.config.mixed_precision and self.device == "cuda":
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
                
        else:  # mlx
            from xlstm_mlx import create_xlstm_model
            
            model = create_xlstm_model(
                vocab_size=self.config.vocab_size,
                num_layers=self.config.num_layers,
                signature=self.config.signature,
                inp_dim=self.config.inp_dim,
                head_dim=self.config.head_dim,
                head_num=self.config.head_num,
                dropout=self.config.dropout
            )
            self.scaler = None
        
        # Count parameters
        param_count = self._count_parameters(model)
        print(f"Model created with {param_count:,} parameters")
        
        return model
    
    def _count_parameters(self, model):
        """Count model parameters"""
        if self.backend == "pytorch":
            return sum(p.numel() for p in model.parameters())
        else:  # mlx
            import mlx.core as mx
            
            def count_mlx_params(params):
                count = 0
                for p in params.values():
                    if isinstance(p, mx.array):
                        count += p.size
                    elif isinstance(p, dict):
                        count += count_mlx_params(p)
                return count
            
            return count_mlx_params(model.parameters())
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        if self.backend == "pytorch":
            import torch
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.1
            )
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs * 1000,  # Approximate steps
                eta_min=self.config.learning_rate * 0.1
            )
            
        else:  # mlx
            import mlx.optimizers as optim
            
            self.optimizer = optim.AdamW(
                learning_rate=self.config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.1
            )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        if self.backend == "pytorch":
            return self._train_step_pytorch(batch)
        else:
            return self._train_step_mlx(batch)
    
    def _train_step_pytorch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """PyTorch training step"""
        import torch
        import torch.nn.functional as F
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = self.model.forward_sequence(input_ids, use_cache=False)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
        else:
            logits, _ = self.model.forward_sequence(input_ids, use_cache=False)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'perplexity': torch.exp(loss).item()
        }
    
    def _train_step_mlx(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """MLX training step"""
        import mlx.core as mx
        import mlx.nn as nn
        
        input_ids = mx.array(batch['input_ids'])
        labels = mx.array(batch['labels'])
        
        def loss_fn(model, input_ids, labels):
            logits = model(input_ids)
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction='mean'
            )
        
        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model, input_ids, labels)
        
        # Update weights
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        
        return {
            'loss': float(loss),
            'learning_rate': self.config.learning_rate,
            'perplexity': float(mx.exp(loss))
        }
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model on validation data"""
        if self.backend == "pytorch":
            import torch
            self.model.eval()
            
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad() if self.backend == "pytorch" else nullcontext():
            for batch in eval_dataloader:
                if self.backend == "pytorch":
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits, _ = self.model.forward_sequence(input_ids, use_cache=False)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_tokens += labels.numel()
                else:  # mlx
                    import mlx.core as mx
                    import mlx.nn as nn
                    
                    input_ids = mx.array(batch['input_ids'])
                    labels = mx.array(batch['labels'])
                    
                    logits = self.model(input_ids)
                    loss = nn.losses.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        reduction='sum'
                    )
                    
                    total_loss += float(loss)
                    total_tokens += labels.size
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        if self.backend == "pytorch":
            self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "pytorch":
            import torch
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step': step,
                'metrics': metrics,
                'config': self.config.__dict__
            }, checkpoint_dir / "checkpoint.pt")
            
        else:  # mlx
            import mlx.core as mx
            
            # Save model weights
            mx.save(checkpoint_dir / "model.npz", self.model.parameters())
            
            # Save training state
            with open(checkpoint_dir / "training_state.json", 'w') as f:
                json.dump({
                    'global_step': step,
                    'metrics': metrics,
                    'config': self.config.__dict__
                }, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            epoch_start_time = time.time()
            
            for step, batch in enumerate(train_dataloader):
                # Training step
                metrics = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    print(f"Step {self.global_step}: "
                          f"Loss={metrics['loss']:.4f}, "
                          f"PPL={metrics['perplexity']:.2f}, "
                          f"LR={metrics['learning_rate']:.2e}")
                
                # Evaluation
                if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    print(f"Eval at step {self.global_step}: "
                          f"Loss={eval_metrics['eval_loss']:.4f}, "
                          f"PPL={eval_metrics['eval_perplexity']:.2f}")
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(self.global_step, eval_metrics)
                
                # Regular checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(self.global_step, metrics)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        print("\nTraining completed!")
        return self.model


# Dummy data generator for testing
def create_dummy_dataloader(batch_size, seq_length, vocab_size, num_batches=100):
    """Create dummy data for testing"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_length))
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Format batches
    formatted_batches = []
    for input_batch, label_batch in dataloader:
        formatted_batches.append({
            'input_ids': input_batch,
            'labels': label_batch
        })
    
    return formatted_batches


from contextlib import nullcontext

def main():
    parser = argparse.ArgumentParser(description="Train xLSTM model")
    parser.add_argument("--backend", choices=["pytorch", "mlx"], default="pytorch")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig(
            backend=args.backend,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_layers=2,  # Small model for testing
            inp_dim=128,
            vocab_size=1000
        )
    
    # Create trainer
    trainer = xLSTMTrainer(config)
    
    # Create dummy data
    print("Creating dummy training data...")
    train_data = create_dummy_dataloader(
        config.batch_size,
        config.seq_length,
        config.vocab_size,
        num_batches=50
    )
    
    eval_data = create_dummy_dataloader(
        config.batch_size,
        config.seq_length,
        config.vocab_size,
        num_batches=10
    )
    
    # Train model
    model = trainer.train(train_data, eval_data)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()