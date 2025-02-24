#Requireed modules
# E,F projection matrix
# Block ->MultiheadAttention->linformer attention->FeedForward->LayerNorm->Dropout
#Activation function
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import ModelConfig



#E,F
def get_EF_matrix(n_embed,k,type='learnable',head_dim=None,bias=True): #n_embed=embedding size, k=reduced dimension
    if type=='convolution':
        conv=nn.Conv1d(n_embed,k,kernel_size=int(n_embed/k),stride=int(n_embed/k),bias=bias)
        return conv
    elif type=='no-params':# notrmal distribution N(0,1/K)
        mat=torch.zeros(n_embed,k)
        nn.init.normal_(mat,mean=0,std=1/k)
        return mat
    
    #by default xavier initialization, i.e ~N(0,2/fan_in+fan_out) where fan_in=n_embed, fan_out=k
    mat=nn.Linear(n_embed,k,bias=bias)
    nn.init.xavier_normal_(mat.weight)
    return mat
        


#LinformrHead

class LinformerHead(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.value=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.query=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.dropout=nn.Dropout(ModelConfig.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(ModelConfig.block_size, ModelConfig.block_size)))
        
    def forward(self, x):
        B,T,C=x.shape #Batch size, Sequence length, Embedding size
        key=self.key(x)
        