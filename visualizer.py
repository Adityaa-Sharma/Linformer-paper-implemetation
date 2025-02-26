import matplotlib.pyplot as plt
import os
import torch
import numpy as np

class Visualizer:
    def __init__(self,plot_dir):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def to_numpy(self, tensor):
        """Safely convert tensor to numpy array regardless of device"""
        if isinstance(tensor, (list, np.ndarray)):
            return np.array(tensor)
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def plot_training_metrics(self, train_losses, val_losses):
        """Plot training metrics using PyTorch operations"""
        # Convert inputs to numpy arrays safely
        train_losses_np = self.to_numpy(train_losses)
        val_losses_np = self.to_numpy(val_losses)
        
        plt.figure(figsize=(12, 8))
        
        # 1. Standard training curve
        plt.subplot(2, 2, 1)
        plt.plot(train_losses_np, label='Training Loss', alpha=0.7)
        plt.plot(val_losses_np, label='Validation Loss', alpha=0.7)
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)

        # 2. Moving average curve
        plt.subplot(2, 2, 2)
        window_size = 50
        
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        train_ma = moving_average(train_losses_np, window_size)
        val_ma = moving_average(val_losses_np, window_size)
        
        plt.plot(train_ma, label=f'Train ({window_size}-step MA)', alpha=0.7)
        plt.plot(val_ma, label=f'Val ({window_size}-step MA)', alpha=0.7)
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss (Moving Average)')
        plt.title(f'Moving Average ({window_size} steps)')
        plt.legend()
        plt.grid(True)

        # 3. Loss difference
        plt.subplot(2, 2, 3)
        loss_diff = val_losses_np - train_losses_np
        plt.plot(loss_diff, label='Val - Train', color='purple', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss Difference')
        plt.title('Validation - Training Loss')
        plt.legend()
        plt.grid(True)

        # 4. Loss distribution
        plt.subplot(2, 2, 4)
        plt.hist(train_losses_np, bins=50, alpha=0.5, label='Training', density=True)
        plt.hist(val_losses_np, bins=50, alpha=0.5, label='Validation', density=True)
        plt.xlabel('Loss Value')
        plt.ylabel('Density')
        plt.title('Loss Distribution')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_analysis.png'), dpi=300)
        plt.close()

    def plot_learning_curve(self, epoch_train_losses, epoch_val_losses):
        """Plot per-epoch learning curve"""
        plt.figure(figsize=(10, 6))
        
        # Convert inputs to numpy arrays safely
        epoch_train_losses = self.to_numpy(epoch_train_losses)
        epoch_val_losses = self.to_numpy(epoch_val_losses)
        epochs = np.arange(1, len(epoch_train_losses) + 1)
        
        plt.plot(epochs, epoch_train_losses, 'bo-', label='Training', alpha=0.7)
        plt.plot(epochs, epoch_val_losses, 'ro-', label='Validation', alpha=0.7)
        
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Learning Curve (per epoch)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.plot_dir, 'learning_curve.png'), dpi=300)
        plt.close()
