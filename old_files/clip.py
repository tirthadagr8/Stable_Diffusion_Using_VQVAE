import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab,n_embed,n_tokens):
        super().__init__()
        self.token_embedding=nn.Embedding(n_vocab,n_embed)
        self.position_embedding=nn.Parameter(torch.zeros(n_tokens,n_embed))
        
    def forward(self,tokens):
        # (batch,seq_len) -> (batch,seq_len,Dim)
        x=self.token_embedding(tokens)
        x+=self.position_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self,n_head,n_embed):
        super().__init__()
        
        self.layer_norm1=nn.LayerNorm(n_embed)
        self.attn=nn.MultiheadAttention(
            embed_dim=n_embed,
            num_heads=n_head,
            batch_first=True,
            dropout=True,
            add_bias_kv=True
        )
        self.layer_norm2=nn.LayerNorm(n_embed)
        self.linear1=nn.Linear(n_embed,4*n_embed)
        self.linear2=nn.Linear(4*n_embed,n_embed)
        
    def forward(self,x):
        # batch,seq_len,dim
        residue=x
        # self attention
        x=self.layer_norm1(x)
        x,_=self.attn(x,x,x)
        
        x+=residue
        
        # feedforward layer
        residue=x
        x=self.layer_norm2(x)
        x=self.linear1(x)
        x=x*torch.sigmoid(1.702*x) # quickGELU actv func
        x=self.linear2(x)
        x+=residue
        return x
        


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=CLIPEmbedding(49408,768,77)
        
        self.layers=nn.ModuleList([
            CLIPLayer(12,768) for i in range(12)
        ])
        
        self.layer_norm=nn.LayerNorm(768)
    
    def forward(self,tokens):
        # tokens (batch,seq_len)
        tokens=tokens.type(torch.long)
        # (batch,seq_len) -> (batch,seq_len,embed_dim/768)
        state=self.embedding(tokens)
        
        for layer in self.layers:
            state=layer(state)
        output=self.layer_norm(state)
        
        return output