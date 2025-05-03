# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:33:39 2025

@author: drsaq
"""



import torch
import torch.nn as nn
import torchvision.ops as ops

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        # Apply LayerNorm across channel dimension
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x

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
        x = x.flatten(2).permute(2, 0, 1)  # HW, B, C
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.mlp(x)
        x = x.permute(1, 2, 0).view(B, C, H, W)  # B, C, H, W
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
        self.fusion_conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        
    def forward(self, x):
        swin_feat = self.swin(x)
        offset = self.offset(x)
        deform_feat = self.deform_conv(x, offset)
        # Combine the features
        combined = torch.cat([x, swin_feat, deform_feat], dim=1)
        return self.fusion_conv(combined)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ScaleFusionNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_levels=3):
        super().__init__()
        
        # Encoder components
        self.patch_embed = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        
        # Start with initial patch embedding
        self.initial_embed = PatchEmbedding(in_channels, base_channels)
        
        # Create encoder levels
        current_channels = base_channels
        for i in range(num_levels):
            # For each level, we have patch embedding followed by Swin Transformer block
            self.patch_embed.append(PatchEmbedding(current_channels, current_channels*2))
            self.swin_blocks.append(SwinTransformerBlock(current_channels*2, num_heads=4))
            self.down_sample.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels *= 2
        
        # Decoder components
        self.catm_blocks = nn.ModuleList()
        self.adaptive_fusion = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        # Create decoder levels
        for i in range(num_levels):
            self.catm_blocks.append(CATM(current_channels))
            self.adaptive_fusion.append(AdaptiveFusionBlock(current_channels))
            self.upconv_blocks.append(UpConv(current_channels, current_channels//2))
            current_channels //= 2
        
        # Final segmentation layer
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Initial embedding
        x = self.initial_embed(x)
        
        # Encoder path - store skip connections
        skip_connections = []
        skip_connections.append(x)
        
        for i in range(len(self.swin_blocks)):
            x = self.patch_embed[i](x)
            x = self.swin_blocks[i](x)
            skip_connections.append(x)
            x = self.down_sample[i](x)
        
        # Decoder path
        for i in range(len(self.catm_blocks)):
            # Get the corresponding skip connection (in reverse order)
            skip = skip_connections[-(i+1)]
            
            # Apply CATM to combine skip connection with current features
            x = self.catm_blocks[i](skip, x)
            
            # Apply AdaptiveFusion block
            x = self.adaptive_fusion[i](x)
            
            # Upsample
            x = self.upconv_blocks[i](x)
        
        # Generate segmentation mask
        mask = torch.sigmoid(self.final_conv(x))
        
        return mask

# Usage example
if __name__ == "__main__":
    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 256, 256)
    
    # Create model with 3 levels (matching the diagram)
    model = ScaleFusionNet(in_channels=3, base_channels=64, num_levels=3)
    
    # Forward pass
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")