import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, GroupNorm, Sequential, MultiheadAttention, SiLU, Embedding

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.residual_input_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.down_sample_conv = Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        residual = self.residual_input_conv(x)
        x = self.resnet_conv_first(x)
        x = self.resnet_conv_second(x)
        x = F.silu(x + residual)
        x = self.down_sample_conv(x)
        return x

class MidBlock(nn.Module):
    def __init__(self, channels):
        super(MidBlock, self).__init__()
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention_norms = nn.GroupNorm(32, channels)
        self.attentions = nn.MultiheadAttention(embed_dim=channels, num_heads=2)
        self.residual_input_conv = Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.residual_input_conv(x)
        x = self.resnet_conv_first(x)
        x = self.resnet_conv_second(x)
        x = F.silu(x + residual)
        B,C,H,W=x.shape
        residual=x
        x=x.reshape(B,C,H*W)
        x = self.attention_norms(x)
        x=x.transpose(1,2)
        x, _ = self.attentions(x, x, x)
        x=x.transpose(1,2).reshape(B,C,H,W)
        return x+residual

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.residual_input_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.up_sample_conv = ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        residual = self.residual_input_conv(x)
        x = self.resnet_conv_first(x)
        x = self.resnet_conv_second(x)
        x = F.silu(x + residual)
        x = self.up_sample_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels=3, codebook_size=8192):
        super(VQVAE, self).__init__()
        self.encoder_conv_in = Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_layers = nn.ModuleList([
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 256)
        ])
        self.encoder_mids = nn.ModuleList([
            MidBlock(256)
        ])
        self.encoder_norm_out = GroupNorm(32, 256)
        
        self.encoder_conv_out = Conv2d(256, input_channels, kernel_size=3, stride=1, padding=1)
        
        self.mean_fc=nn.Sequential(
                nn.Conv2d(input_channels,4,kernel_size=3,stride=1,padding=1)
            )
        self.log_var_fc=nn.Sequential(
                nn.Conv2d(input_channels,4,kernel_size=3,stride=1,padding=1)
            )
        
        self.decoder_conv_in = Conv2d(4, 256, kernel_size=3, stride=1, padding=1)
        self.decoder_mids = nn.ModuleList([
            MidBlock(256)
        ])
        self.decoder_layers = nn.ModuleList([
            UpBlock(256, 256),
            UpBlock(256, 128),
            UpBlock(128, 64)
        ])
        self.decoder_norm_out = GroupNorm(32, 64)
        self.decoder_conv_out = Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1)

    def encode(self,x):
        x = self.encoder_conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.encoder_mids:
            x = layer(x)
        x = self.encoder_norm_out(x)
        x = self.encoder_conv_out(x)
        mean=self.mean_fc(x)
        log_var=self.log_var_fc(x)
        return mean,log_var
    
    def sampling(self,mean,log_var):# this will recieve the mean and log variance from encoder
                
        std=torch.exp(0.5*log_var)
        sample=torch.rand_like(std)
        sample=sample*std+mean
        return sample
    
    
    def decode(self,x):
        x = self.decoder_conv_in(x)
        for layer in self.decoder_mids:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder_norm_out(x)
        x = self.decoder_conv_out(x)
        return x
    
    def forward(self, x):
        # Encoder
        mean,log_var=self.encode(x)
        # print('Encoding Completed',x.shape)
        sample=self.sampling(mean,log_var)
        # Decoder
        output=self.decode(sample)
        
        return output,mean,log_var
