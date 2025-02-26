from Linformer import LinearAttentionModel
from configs import ModelConfig
from visualizer import Visualizer
import matplotlib as plt
import torch


device = ModelConfig.device
torch.manual_seed(1337)

class BatchGenerator:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        # print("printing data length: ", len(data))
        ix = torch.randint(0, len(data)-ModelConfig.block_size, (ModelConfig.batch_size,))
        # print("printing ix: ", ix)
        
        x = torch.stack([data[i:i+ModelConfig.block_size] for i in ix])
        y = torch.stack([data[i+1:i+ModelConfig.block_size+1] for i in ix])
        x, y = x.to(ModelConfig.device), y.to(ModelConfig.device)
        return x, y



class Trainer:
    def __init__(self, model, optimizer, tokenizer, train_data, val_data):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_data = train_data.long()
        self.val_data = val_data.long()
        self.train_losses = []
        self.val_losses = []
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=ModelConfig.max_iters,
            eta_min=1e-5
        )
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        
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
        for iter in range(iters_per_epoch):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Get batch and compute loss
            xb, yb = BatchGenerator(self.train_data, self.val_data).get_batch('train')
            logits, loss = self.model(xb, yb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            if iter % ModelConfig.eval_interval == 0:
                losses = self.estimate_loss()
                print(f'Epoch {epoch}, Iter {iter}, '
                      f'Train loss: {losses["train"]:.4f}, '
                      f'Val loss: {losses["val"]:.4f}, '
                      f'LR: {self.scheduler.get_last_lr()[0]:.2e}')
                self.train_losses.append(losses["train"])
                self.val_losses.append(losses["val"])
                self.epoch_train_losses.append(losses["train"])
                self.epoch_val_losses.append(losses["val"])
            
            xb, yb = BatchGenerator(self.train_data,self.val_data).get_batch('train')
            # Ensure input tensors are in long format
            xb, yb = xb.long().to(device), yb.long().to(device)
            logits, loss = self.model(xb, yb)
        
        # Store average losses for this epoch
        self.epoch_train_losses.append(torch.tensor(self.epoch_train_losses).mean().item())
        self.epoch_val_losses.append(torch.tensor(self.epoch_val_losses).mean().item())



    
    