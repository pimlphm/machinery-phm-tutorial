import torch
import torch.optim as optim
# Adam optimizer from PyTorch for efficient gradient-based optimization
import os
# Used for file operations and path handling for model saving

# (1) Setup optimizer and loss function for RUL prediction task
optimizer = optim.Adam(model.parameters(), lr=1e-3)    # Use Adam optimizer with learning rate 1e-3
criterion = torch.nn.MSELoss()                         # Mean Squared Error loss for regression (RUL prediction)
# For RUL prediction, we use MSE loss since we're predicting continuous values (remaining useful life)

# === Integrated Callbacks for PyTorch training ===
class TrainingCallbacks:
    """
    Integrated callbacks for PyTorch training including ModelCheckpoint and EarlyStopping
    """
    def __init__(self, checkpoint_filepath="best_model.pth", monitor='val_loss', 
                 patience=10, save_best_only=True, restore_best_weights=True, verbose=1):
        # ModelCheckpoint settings
        self.checkpoint_filepath = checkpoint_filepath
        self.save_best_only = save_best_only
        
        # EarlyStopping settings
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        # Common settings
        self.monitor = monitor
        self.verbose = verbose
        
        # State tracking
        self.best_score = float('inf') if 'loss' in monitor else float('-inf')
        self.patience_counter = 0
        self.best_weights = None
        
    def __call__(self, current_score, model, epoch=None):
        """
        Main callback function to be called during training
        
        Args:
            current_score: Current validation score (loss or metric)
            model: PyTorch model
            epoch: Current epoch number (optional, for logging)
            
        Returns:
            bool: True if training should stop (early stopping triggered), False otherwise
        """
        is_better = (current_score < self.best_score) if 'loss' in self.monitor else (current_score > self.best_score)
        
        # Handle model checkpointing
        if not self.save_best_only or is_better:
            if is_better:
                self.best_score = current_score
                self.patience_counter = 0
                # Save best weights for potential restoration
                if self.restore_best_weights:
                    self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            
            # Save model checkpoint
            torch.save(model.state_dict(), self.checkpoint_filepath)
            if self.verbose:
                epoch_str = f" (epoch {epoch})" if epoch is not None else ""
                print(f"Model saved to {self.checkpoint_filepath} - {self.monitor}: {current_score:.4f}{epoch_str}")
        else:
            # No improvement
            self.patience_counter += 1
            
        # Handle early stopping
        if self.patience_counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered - {self.monitor} hasn't improved for {self.patience} epochs")
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
            
            return True  # Signal to stop training
        
        return False  # Continue training

# (2) Initialize integrated callbacks for RUL prediction training
callbacks = TrainingCallbacks(
    checkpoint_filepath="best_rul_model.pth",    # File to save PyTorch model state dict
    monitor='val_loss',                          # Monitor validation loss (MSE) for RUL task
    patience=10,                                 # Wait 10 epochs before stopping if no improvement
    save_best_only=True,                         # Only save if it's the best so far
    restore_best_weights=True,                   # Go back to best model weights after stopping
    verbose=1                                    # Print messages when model is saved or early stopping is triggered
)
