import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.group_norm=nn.GroupNorm(32,channels)
        self.self_attn=nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=4,
            dropout=True,
            batch_first=True,
            add_bias_kv=True
            )
        
    def forward(self,x):
        residue=x
        n,c,h,w=x.shape
        
        # x (batch,channels,height,width) -> (batch, channels, height*width)
        x=x.view(n,c,h*w)
        # x (batch, channels, height*width) -> (batch,height*width, channels)
        x=x.transpose(-1,-2)
        # all this is done because multiheadattention works on last dim,
        # or rather actually embeddings of each pixel, so shape is N,H*W,C
        # shows that for each pixel in H*W, the embedding size is 'channels'
        # and we know only about channels, so lets just bring channels at 
        # the last dim. The Multiheadattn initialized above works with 'channels'
        # For reference, check the text to voice transformer
        x,_=self.self_attn(x,x,x) 
        # returning back to original shape
        x=x.transpose(-1,-2)
        x=x.view(n,c,h,w)
        
        return x+residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.group_norm1=nn.GroupNorm(32,in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        
        self.group_norm2=nn.GroupNorm(32,out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels==out_channels:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    
    def forward(self,x):
        # x: (batch,in_channel,height,width)
        residue=x
        x=self.group_norm1(x)
        x=F.silu(x)
        x=self.conv1(x)
        x=self.group_norm2(x)
        x=F.silu(x)
        x=self.conv2(x)
        # skip connection is like connecting with original input
        # now we need to return x+residual, but in cases where in_channels != out_channels
        # then sum will not be possible, since output from conv2 may not match with original x
        # so we pass x/residual through residual layer to match the channels
        return x+self.residual_layer(residue)
    

class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.module=nn.Sequential(
            nn.Conv2d(4,4,kernel_size=1,padding=0),# input
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),

            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),

            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),

            VAE_ResidualBlock(64,64),
            VAE_ResidualBlock(64,64),
            VAE_ResidualBlock(64,64),
            
            nn.GroupNorm(32,64),
            
            nn.SiLU(),

            nn.Conv2d(64,3,kernel_size=3,padding=1)
         
        )
      
    def forward(self,x):
        # the scaling we did in encoder, we nullify it here
        x/=0.18215
        return self.module(x)