from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    
    # Training parameters
    batch_size: int = 32
    block_size: int = 256
    max_iters: int = 150000
    eval_interval: int = 5000
    eval_iter: int = 50
    learning_rate: float = 1e-5 # gpt paper
    n_epochs: int = 1
    
    # Model architecture
    n_embed: int = 480
    n_layer: int = 12 
    n_head: int = 12 
    dropout: float = 0.1
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
