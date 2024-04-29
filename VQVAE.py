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
        x=x.transpose(1,2).reshape(B,C,W,H)
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

class VQVAE(nn.Module):
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
        self.pre_quant_conv = Conv2d(input_channels, input_channels, kernel_size=1, stride=1)
        self.embedding = Embedding(codebook_size, input_channels)#this embedding_dim should match pre_quant_conv channels
        self.post_quant_conv = Conv2d(input_channels, 16, kernel_size=1, stride=1)
        self.decoder_conv_in = Conv2d(16, 256, kernel_size=3, stride=1, padding=1)
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

        
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
#         print(x.shape)
        x = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
#         print(x.shape,self.embedding.weight.shape)
        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()
        
        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices
    
    
    def encode(self,x):
        x = self.encoder_conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.encoder_mids:
            x = layer(x)
        x = self.encoder_norm_out(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        out, quant_losses, _ = self.quantize(x)
        return out,quant_losses
    
    
    def decode(self,x):
        x=self.post_quant_conv(x)
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
        z,quant_losses=self.encode(x)
        # print('Encoding Completed',x.shape)
        
        # Decoder
        output=self.decode(z)
        
        return output,z,quant_losses
