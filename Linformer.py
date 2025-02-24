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
        conv=nn.Conv1d(head_dim,head_dim,kernel_size=int(n_embed/k),stride=int(n_embed/k),bias=bias)
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
    def __init__(self,head_size,E,F):
        super().__init__()
        self.head_size = head_size
        self.key=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.value=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.query=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.dropout=nn.Dropout(ModelConfig.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(ModelConfig.block_size, ModelConfig.block_size)))
        self.E=E
        self.F=F
        
    def forward(self, x):
        B,T,C=x.shape #Batch size, Sequence length, Embedding size
        
        key=self.key(x) #K # B,T,head_size
        value=self.value(x) #V
        query=self.query(x)#Q
        
        # projecting the key and value to the reduced dimension
        
        #clculating the attention matrix
        # K,Q->T,C, E->C,K
        #k_Proj->Q^T,E
        k_proj=torch.matmul(key, self.E)
        weight=torch.matmul(query,k_proj.transpose(1,2))/ModelConfig.k**0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight=torch.softmax(weight,dim=-1)
        weight=self.dropout(weight)
        
        #calculating the output
        v_proj=torch.matmul(value,self.F)
        out=torch.matmul(weight,v_proj)
        return out
        

class MultiHeadLinearAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([
        LinformerHead(head_size, get_EF_matrix(ModelConfig.n_embed, ModelConfig.k), get_EF_matrix(ModelConfig.n_embed, ModelConfig.k))
        for _ in range(num_heads)])
        self.proj=nn.Linear(ModelConfig.n_embed,ModelConfig.n_embed)
        self.dropout=nn.Dropout(ModelConfig.dropout)
        
    def forward(self,x):
        out=torch.cat([head(x) for head in self.heads],dim=-1)#concatenating along the head size --> (B,T,C)
        out=self.proj(out)
        return out

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed,n_embed)
        )
            
    def forward(self,x):
        return self.net(x)
    
    
class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.MultiHeadLinearAttention=MultiHeadLinearAttention(n_head,head_size)
        self.FeedForward=FeedForward(n_embed)
        self.LayerNorm=nn.LayerNorm(n_embed)
        self.LayerNorm2=nn.LayerNorm(n_embed)
        
    def forward(self,x):
        x=self.LayerNorm(x+self.MultiHeadLinearAttention(x))
        x=self.LayerNorm2(x+self.FeedForward(x))
        return x
        
class LinearAttentionModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embeddings_table=nn.Rmbeddings(vocab_size,ModelConfig.n_embed)
        self.positional_embeddings=nn.Embeddings(ModelConfig.block_size,ModelConfig.n_embed)
        self.blocks=nn.Sequential(
            Block(ModelConfig.n_embed,ModelConfig.n_head) for _ in range(ModelConfig.n_layer)
        )
        self.layer_norm=nn.LayerNorm(ModelConfig.n_embed)
        self.head=nn.Linear(ModelConfig.n_embed,vocab_size)
        