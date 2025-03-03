# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:33:32 2025

@author: SQamar
"""

import torch
import torch.nn as nn
import torchvision.ops as ops

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        # Simplified Swin Transformer implementation
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.mlp(x)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

class CATM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.swin_block = SwinTransformerBlock(dim, num_heads=4)
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=1)
        
    def forward(self, x_skip, x_decoder):
        qkv = self.swin_block(x_decoder)
        x = torch.cat([x_skip, qkv], dim=1)
        return self.conv(x)

class AdaptiveFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.swin = SwinTransformerBlock(in_channels, num_heads=4)
        self.deform_conv = ops.DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.offset = nn.Conv2d(in_channels, 2*3*3, kernel_size=3, padding=1)
        
    def forward(self, x):
        swin_feat = self.swin(x)
        offset = self.offset(x)
        deform_feat = self.deform_conv(x, offset)
        return x + torch.cat([swin_feat, deform_feat], dim=1)

class ScaleFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                SwinTransformerBlock(64, 4)
            ])
        # Decoder
        self.decoder = nn.ModuleList([
            AdaptiveFusionBlock(64)
        ])
        self.catm = CATM(64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)
        
        # Decoder path
        for i, layer in enumerate(self.decoder):
            x = nn.Upsample(scale_factor=2)(x)
            x = self.catm(skips[-(i+1)], x)
            x = layer(x)
            
        return torch.sigmoid(self.final_conv(x))