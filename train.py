from Linformer import LinearAttentionModel
from configs import ModelConfig
from visualizer import Visualizer
import matplotlib as plt
import torch



device = ModelConfig.device
torch.manual_seed(1337)

class Trainer:
    def __init__(self, model, optimizer, tokenizer, train_data, val_data):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        # Ensure data is in the correct format (Long/Int)
        self.train_data = train_data.long()  # Convert to Long
        self.val_data = val_data.long()      # Convert to Long
        self.train_losses = []
        self.val_losses = []
        self.epoch_train_losses = []  # Average loss per epoch
        self.epoch_val_losses = []    # Average loss per epoch
        self.visualizer = Visualizer()
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(ModelConfig.eval_iter, device=device)
            for k in range(ModelConfig.eval_iter):
                x, y = BatchGenerator(self.train_data,self.val_data).get_batch(split)
                # Ensure input tensors are in long format
                x, y = x.long().to(device), y.long().to(device)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_epoch(self, epoch):
        iters_per_epoch = ModelConfig.max_iters // ModelConfig.n_epochs
        epoch_train_losses = []
        epoch_val_losses = []
        
        for iter in range(iters_per_epoch):
            if iter % ModelConfig.eval_interval == 0:
        
                losses = self.estimate_loss()
                print(f'Epoch {epoch}, Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')
                self.train_losses.append(losses["train"])
                self.val_losses.append(losses["val"])
                epoch_train_losses.append(losses["train"])
                epoch_val_losses.append(losses["val"])
            
            xb, yb = BatchGenerator(self.train_data,self.val_data).get_batch('train')
            # Ensure input tensors are in long format
            xb, yb = xb.long().to(device), yb.long().to(device)
            logits, loss = self.model(xb, yb)
        
        # Store average losses for this epoch
        self.epoch_train_losses.append(torch.tensor(epoch_train_losses).mean().item())
        self.epoch_val_losses.append(torch.tensor(epoch_val_losses).mean().item())
        
        # Add visualization at the end of each epoch
        self.visualizer.plot_training_metrics(self.train_losses, self.val_losses)
        self.visualizer.plot_learning_curve(self.epoch_train_losses, self.epoch_val_losses)

