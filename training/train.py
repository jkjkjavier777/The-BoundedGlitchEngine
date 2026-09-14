"""
BoundedGlitchGPT: Training Script
==================================

This script handles:
1. Loading training data
2. Tokenizing text
3. Creating data batches
4. Training the model
5. Saving checkpoints
6. Monitoring loss and performance

Usage:
    python training/train.py --data_path data/train.txt --output checkpoints/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os

# Add parent directory to path so we can import model.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import GPT, GPTConfig, SimpleTokenizer


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """
    A simple dataset that loads text and converts it to token sequences.
    
    This is a minimal implementation. In production, you'd use:
    - Streaming datasets (for large files that don't fit in memory)
    - Pre-tokenized files (for faster loading)
    - Multi-GPU data loading with proper shuffling
    """
    
    def __init__(self, file_path: str, tokenizer: SimpleTokenizer, 
                 max_context_length: int, split_ratio: float = 1.0):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to convert text to tokens
            max_context_length: Maximum sequence length
            split_ratio: Fraction of data to use (for quick testing with split_ratio=0.1)
        """
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        
        print(f"Loading data from {file_path}...")
        
        # Read the entire file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use only a fraction if specified (for quick testing)
        if split_ratio < 1.0:
            text = text[:int(len(text) * split_ratio)]
        
        print(f"Text length: {len(text):,} characters")
        
        # Tokenize the entire text
        print("Tokenizing...")
        self.tokens = self.tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens):,}")
    
    def __len__(self) -> int:
        """Number of sequences we can create from the tokens."""
        return len(self.tokens) - self.max_context_length
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training example: input sequence and target sequence.
        
        The target is the input shifted by one position (predicting the next token).
        
        Example:
            tokens: [1, 2, 3, 4, 5]
            idx: 0
            max_context_length: 3
            
            input:  [1, 2, 3]     (positions 0-2)
            target: [2, 3, 4]     (positions 1-3) - shifted by 1
        """
        # Input: tokens at positions idx to idx+max_context_length
        input_tokens = self.tokens[idx:idx + self.max_context_length]
        
        # Target: next token after each input token (for next-token prediction)
        target_tokens = self.tokens[idx + 1:idx + self.max_context_length + 1]
        
        return torch.tensor(input_tokens, dtype=torch.long), \
               torch.tensor(target_tokens, dtype=torch.long)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Handles the training loop, checkpointing, and monitoring.
    """
    
    def __init__(self, model: GPT, config: GPTConfig, device: str, 
                 output_dir: str = "checkpoints"):
        """
        Initialize the trainer.
        
        Args:
            model: GPT model instance
            config: GPTConfig with training hyperparameters
            device: 'cuda' or 'cpu'
            output_dir: Where to save checkpoints
        """
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Optimizer: Adam is standard for transformers
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler: decay LR over time
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Training history for monitoring
        self.history = {
            'epoch': [],
            'batch': [],
            'loss': [],
            'learning_rate': []
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint of the model, optimizer, and training state.
        
        Args:
            epoch: Current epoch number
            is_best: If True, also save as "best_model.pt"
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'vocab_size': self.config.vocab_size,
                'max_context_length': self.config.max_context_length,
                'embed_dim': self.config.embed_dim,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'ffn_hidden_dim': self.config.ffn_hidden_dim,
            },
            'history': self.history,
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets=target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping: prevent exploding gradients
            # This is important for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimization step
            self.optimizer.step()
            
            # Track statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Log to history
            self.history['batch'].append(batch_idx)
            self.history['loss'].append(batch_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss:.4f})
            
            # Periodic logging
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                      f"Loss = {batch_loss:.4f}")
        
        # Average loss for the epoch
        avg_loss = total_loss / num_batches
        
        # Learning rate scheduling
        self.scheduler.step()
        
        self.history['epoch'].append(epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            _, loss = self.model(input_ids, targets=target_ids)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader=None, 
              resume_from: str = None):
        """
        Main training loop.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print("=" * 70 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Training Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"  Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch)
            else:
                self.save_checkpoint(epoch)
            
            # Save training history
            self.save_history()
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Checkpoints saved to: {self.output_dir}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main entry point for training."""
    
    parser = argparse.ArgumentParser(description="Train BoundedGlitchGPT")
    parser.add_argument('--data_path', type=str, default='data/train.txt',
                        help='Path to training data file')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data file (optional)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension (smaller for testing)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer blocks (smaller for testing)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--split_ratio', type=float, default=1.0,
                        help='Fraction of data to use (for quick testing)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        print("\nTo test training, create a simple text file:")
        print("  echo 'The quick brown fox jumps over the lazy dog.' > data/train.txt")
        sys.exit(1)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create config with command-line overrides
    config = GPTConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.embed_dim = args.embed_dim
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    
    # Validate config
    if config.embed_dim % config.num_heads != 0:
        raise ValueError(f"embed_dim ({config.embed_dim}) must be divisible by "
                        f"num_heads ({config.num_heads})")
    
    print(f"\nModel Config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embedding dim: {config.embed_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    # Create datasets
    train_dataset = TextDataset(
        args.data_path,
        tokenizer,
        config.max_context_length,
        split_ratio=args.split_ratio
    )
    
    val_dataset = None
    if args.val_data_path:
        val_dataset = TextDataset(
            args.val_data_path,
            tokenizer,
            config.max_context_length
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    print(f"\nDataset Config:")
    print(f"  Training examples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation examples: {len(val_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    model = GPT(config).to(device)
    
    # Initialize trainer
    trainer = Trainer(model, config, device, output_dir=args.output_dir)
    
    # Train
    trainer.train(
        train_loader,
        val_loader=val_loader,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
