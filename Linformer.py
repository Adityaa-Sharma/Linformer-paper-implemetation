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
    if type=='no-params':# notrmal distribution N(0,1/K)
        mat=torch.zeros(n_embed,k)
        nn.init.normal_(mat,mean=0,std=1/k)
        return mat
    
    #by default xavier initialization, i.e ~N(0,2/fan_in+fan_out) where fan_in=n_embed, fan_out=k
    # mat=nn.Linear(ModelConfig.batch_size,n_embed,k,bias=bias).weight
    mat = torch.empty(ModelConfig.batch_size, ModelConfig.block_size, k)
    torch.nn.init.xavier_normal_(mat)
    # print("size of E,F matrix: ", mat.size()) 
    return mat.to(device=ModelConfig.device)


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
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        B,T,C=x.shape #Batch size, Sequence length, Embedding size
        
        k=self.key(x) #K # B,T,head_size
        v=self.value(x) #V
        q=self.query(x)#Q
        # print("shape of k,v,q: ", k.shape, v.shape, q.shape)
        
        # projecting the key and value to the reduced dimension
        
        #clculating the attention matrix
        # K,Q->T,C, E->C,K
        #k_Proj->Q^T,E
        # k_proj = self.E(k) #change
        k_proj=torch.matmul( self.E.transpose(1,2),k)
        
        # print("shape of k_proj: ", k_proj.shape)
        weight=torch.matmul(q,k_proj.transpose(1,2))/ModelConfig.k**0.5
        # print("shape of weight: ", weight.shape)
        weight = weight.masked_fill(self.tril[:T, :ModelConfig.k] == 0, float('-inf'))
        weight=torch.softmax(weight,dim=-1)
        weight=self.dropout(weight)
        
        #calculating the output
        # v_proj = self.F(v) #change
        v_proj=torch.matmul(self.E.transpose(1,2),v)
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
        self.embeddings_table=nn.Embedding(vocab_size,ModelConfig.n_embed)
        self.positional_embeddings=nn.Embedding(ModelConfig.block_size,ModelConfig.n_embed)
        self.blocks = nn.Sequential(
            Block(ModelConfig.n_embed, ModelConfig.n_head),
            Block(ModelConfig.n_embed, ModelConfig.n_head),
            Block(ModelConfig.n_embed, ModelConfig.n_head)
        )
        self.layer_norm=nn.LayerNorm(ModelConfig.n_embed)
        self.lm_head=nn.Linear(ModelConfig.n_embed,vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        
    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_embeddings=self.embeddings_table(idx)
        pos_embeddings=self.positional_embeddings(torch.arange(T).to(idx.device))
        x=tok_embeddings+pos_embeddings
        x=self.blocks(x)
        x=self.layer_norm(x)
        logits=self.lm_head(x)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)  #B,T,C->B*T,C
            targets=targets.view(B*T) #B,T->B*T
            loss=F.cross_entropy(logits,targets)
        return logits,loss
        
    def generate(self,idx,max_len=1200):
        for _ in range(max_len):
            idx=idx[:,-ModelConfig.block_size:]
            logits,loss=self(idx)
            logits=logits[:, -1, :]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat([idx,idx_next],dim=-1)
            
        return idx





