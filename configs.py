from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    
    # Training parameters
    batch_size: int = 8
    block_size: int = 512
    max_iters: int = 240000
    eval_interval: int = 5000
    eval_iter: int = 50
    learning_rate: float = 1e-5
    n_epochs: int = 1
    
    # Model architecture
    n_embed: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.1
    
    # Linformer parameters
    k: int = 128
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
