import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.parameters import hp


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,temb_dim,layers,self_attention=False,cross_attention=False,context_dim=None):
        super(ResidualBlock, self).__init__()
        self.layers=layers
        self.temb_dim=temb_dim
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.context_dim=context_dim
        
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
            nn.GroupNorm(hp.norm_channels, in_channels if i==0 else out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            for i in range(layers)
        ])
        
        self.temb_layers=nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim,out_channels)
            )
            for _ in range(layers)
        ])
        
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
            nn.GroupNorm(hp.norm_channels, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )    
            for _ in range(layers)
        ])
        
        self.attn_norms=nn.ModuleList([
            nn.GroupNorm(hp.norm_channels,out_channels) for _ in range(layers)    
        ])
        self.attn=nn.ModuleList([
            nn.MultiheadAttention(out_channels,hp.num_heads,batch_first=True) for _ in range(layers)    
        ])
        
        self.cross_attn_norms=nn.ModuleList([
            nn.GroupNorm(hp.norm_channels,out_channels) for _ in range(layers)    
        ])
        self.cross_attn=nn.ModuleList([
            nn.MultiheadAttention(out_channels,hp.num_heads,batch_first=True) for _ in range(layers)    
        ])
        
        if self.context_dim is not None:
            self.context_proj=nn.ModuleList([
                nn.Linear(self.context_dim,out_channels) for _ in range(self.layers)    
            ])


        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1, stride=1) for i in range(self.layers)  
        ])
      
    def forward(self, x, t_emb, context=None):
        # out : (B,C,H,W)
        # t_emb: (B,temb_dim)
        
        out=x
        for i in range(self.layers):
            resnet_input=out
            out=self.resnet_conv_first[i](out)
            out=out+self.temb_layers[i](t_emb).unsqueeze(-1).unsqueeze(-1)
            out=self.resnet_conv_second[i](out)
            out=out+self.residual_input_conv[i](resnet_input)
            
            if self.self_attention:
                B,C,H,W=out.shape
                in_attn=out.reshape(B,C,H*W)
                in_attn=self.attn_norms[i](in_attn)
                in_attn=in_attn.transpose(1,2)
                out_attn,_=self.attn[i](in_attn,in_attn,in_attn)
                out_attn=out_attn.transpose(1,2).reshape(B,C,H,W)
                out=out+out_attn
            
            if self.cross_attention:
                B,C,H,W=out.shape
                in_attn=out.reshape(B,C,H*W)
                in_attn=self.cross_attn_norms[i](in_attn)
                in_attn=in_attn.transpose(1,2)
                context_proj=self.context_proj[i](context)
                out_attn,_=self.cross_attn[i](in_attn,context_proj,context_proj)
                out_attn=out_attn.transpose(1,2).reshape(B,C,H,W)
                out=out+out_attn
                
        return out
    

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in=nn.Conv2d(hp.in_channels,hp.down_blocks[0],kernel_size=3,stride=1,padding=1)
        self.t_proj=nn.Sequential(
            nn.Linear(hp.temb_dim,hp.temb_dim),
            nn.SiLU(),
            nn.Linear(hp.temb_dim,hp.temb_dim)
        )

        self.downblock=nn.ModuleList([])
        for i in range(len(hp.down_blocks)-1):
            self.downblock.append(
                ResidualBlock(hp.down_blocks[i],hp.down_blocks[i+1],hp.temb_dim,hp.num_layers,True,True,hp.text_embed_dim)
            )
            self.downblock.append(nn.Conv2d(hp.down_blocks[i+1],hp.down_blocks[i+1],kernel_size=3,stride=2,padding=1))
            
        self.midblock=nn.ModuleList([])
        for i in range(len(hp.mid_blocks)-1):
            self.midblock.append(
                ResidualBlock(hp.mid_blocks[i],hp.mid_blocks[i+1],hp.temb_dim,hp.num_layers,True,True,hp.text_embed_dim)
            )
        
        # 2 1 0
        self.upblock=nn.ModuleList([])
        for i in reversed(range(len(hp.down_blocks) - 1)):
            self.upblock.append(nn.ConvTranspose2d(hp.down_blocks[i],hp.down_blocks[i],kernel_size=4,stride=2,padding=1))
            self.upblock.append(
                ResidualBlock(hp.down_blocks[i]*2,hp.down_blocks[i-1] if i!=0 else hp.conv_out_channels,hp.temb_dim,hp.num_layers,True,True,hp.text_embed_dim)
            )
            
            
        self.norm_out=nn.GroupNorm(hp.norm_channels,hp.conv_out_channels)
        self.silu=nn.SiLU()
        self.conv_out=nn.Conv2d(hp.conv_out_channels,3,kernel_size=3,stride=1,padding=1)
        

    def forward(self,x,t_emb,context):
        x=self.conv_in(x)
        t_emb=self.t_proj(t_emb)
        down_outs = []
        for layer in self.downblock:
            if isinstance(layer,ResidualBlock):
                down_outs.append(x)
                x=layer(x,t_emb,context)
            else:
                x=layer(x)
                
        for layer in self.midblock:
            if isinstance(layer,ResidualBlock):
                x=layer(x,t_emb,context)
            else:
                x=layer(x)
               
        for layer in self.upblock:
            if isinstance(layer,ResidualBlock):
                x=layer(x,t_emb,context)
            else:
                x=layer(x)
                out_down=down_outs.pop()
                x = torch.cat([x, out_down], dim=1)
            
            
        x=self.silu(self.norm_out(x))
        x=self.conv_out(x)
        return x
        
diffusion=UNET()
        

