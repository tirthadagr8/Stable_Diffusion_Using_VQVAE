from statistics import variance
import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input=nn.Conv2d(3,128,kernel_size=3,padding=1)
        # size remains same in residual block
        self.residual_block1=VAE_ResidualBlock(128,128)
        self.residual_block2=VAE_ResidualBlock(128,128)
        
        self.conv1=nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1)
        
        self.residual_block3=VAE_ResidualBlock(128,256)
        self.residual_block4=VAE_ResidualBlock(256,256)

        self.conv2=nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1)

        self.residual_block5=VAE_ResidualBlock(256,512)
        self.residual_block6=VAE_ResidualBlock(512,512)
            
        self.conv3=nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1)

        self.residual_block7=VAE_ResidualBlock(512,512)
        self.residual_block8=VAE_ResidualBlock(512,512)
        self.residual_block9=VAE_ResidualBlock(512,512)

        self.attn=VAE_AttentionBlock(512)

        self.residual_block10=VAE_ResidualBlock(512,512)
        
        self.group_norm=nn.GroupNorm(num_groups=32,num_channels=512)
        
        self.silu=nn.SiLU()
        
        self.conv4=nn.Conv2d(512,8,kernel_size=3,padding=1)
        self.out=nn.Conv2d(8,8,kernel_size=1)
        
    def forward(self,x,noise):
        # x-> (batch,channel,height,width)
        # noise is supposed to have shave exactly same as ouput of encoder
        # noise -> (batch,out_channel,height/8,width/8)
        x=self.input(x)
        x=self.residual_block1(x)
        x=self.residual_block2(x)
        x=self.conv1(x)
        x=self.residual_block3(x)
        x=self.residual_block4(x)
        x=self.conv2(x)
        x=self.residual_block5(x)
        x=self.residual_block6(x)
        x=self.conv3(x)
        x=self.residual_block7(x)
        x=self.residual_block8(x)
        x=self.residual_block9(x)
        x=self.attn(x)
        x=self.residual_block10(x)
        x=self.group_norm(x)
        x=self.silu(x)
        x=self.conv4(x)
        x=self.out(x)

        # now for mean and log_var, we divide the x into 2 chunks of data along provided dim, obviously channels
        # x-> (batch,8,height/8,width/8) -> two tensor of size (batch,4,height/8,width/8)
        mean,log_var=torch.chunk(x,2,dim=1)

        # now we need to clamp the log_var within a range in order to avoid being too big or small
        log_var=torch.clamp(log_var,min=-30,max=20)
        variance=log_var.exp()
        # now we find the std variance=sqrt(variance)
        std=torch.sqrt(variance)
        # after all these operations, the shape remained the same, i.e., (batch,4,height/8,width/8)
        x=mean+std*noise
        x*=0.18215  # scaled, dunno why this constant
        return x







