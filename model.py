# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:33:39 2025

@author: drsaq
"""



from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from timm import create_model
from thop import profile
from torchsummary import summary
import math

# -----------------------------------------------------------------------------
#  Basic building blocks
# -----------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Patch Embedding: projects input image to lower-res token feature map."""
    def __init__(self, in_ch: int, out_ch: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H', W')
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x
    
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.window_size = window_size  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(x_flat)
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        return x

# -----------------------------------------------------------------------------
#  Dynamic Swin Transformer (handles arbitrary input sizes)
# -----------------------------------------------------------------------------

class DynamicPatchMerging(nn.Module):
    """Dynamic patch merging that works with any spatial resolution."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Handle very small spatial dimensions - use 1x1 conv instead of patch merging
        if H <= 1 or W <= 1:
            # Use 1x1 conv for dimension reduction when spatial size is too small
            conv = nn.Conv2d(C, C*4, kernel_size=1, bias=False).to(x.device)
            x = conv(x)  # (B, 4*C, H, W)
            x = x.flatten(2).transpose(1, 2)  # (B, HW, 4*C)
            x = self.norm(x)
            x = self.reduction(x)  # (B, HW, output_dim)
            x = x.transpose(1, 2).view(B, -1, H, W)  # (B, output_dim, H, W)
            return x
        
        # Ensure even dimensions for patch merging
        pad_h = pad_w = 0
        if H % 2 == 1:
            pad_h = 1
        if W % 2 == 1:
            pad_w = 1
            
        if pad_h > 0 or pad_w > 0:
            # Only pad if we have enough spatial dimension
            if H >= pad_h and W >= pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                H += pad_h
                W += pad_w
            else:
                # Fallback to 1x1 conv for very small tensors
                conv = nn.Conv2d(C, C*4, kernel_size=1, bias=False).to(x.device)
                x = conv(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = self.reduction(x)
                x = x.transpose(1, 2).view(B, -1, H, W)
                return x
        
        # Reshape to patches of 2x2
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H//2, W//2, C, 2, 2)
        x = x.view(B, H//2, W//2, 4*C)  # (B, H//2, W//2, 4*C)
        
        # Apply norm and reduction
        x = self.norm(x)
        x = self.reduction(x)  # (B, H//2, W//2, output_dim)
        
        # Convert back to (B, C, H, W) format
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, output_dim, H//2, W//2)
        return x

class DynamicSwinStage(nn.Module):
    """A stage of Swin Transformer with dynamic patch merging."""
    def __init__(self, input_dim: int, output_dim: int, depth: int = 2, num_heads: int = 3):
        super().__init__()
        # Only create patch merging if dimensions are different
        if input_dim != output_dim:
            self.patch_merge = DynamicPatchMerging(input_dim, output_dim)
            self.use_patch_merge = True
        else:
            self.patch_merge = None
            self.use_patch_merge = False
        
        # Swin transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(output_dim, num_heads) for _ in range(depth)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch merging (downsampling) only if needed
        if self.use_patch_merge and self.patch_merge is not None:
            x = self.patch_merge(x)
        
        # Apply Swin blocks
        for block in self.blocks:
            x = block(x)
            
        return x

class DynamicSwinTransformer(nn.Module):
    """Dynamic Swin Transformer that handles arbitrary input sizes."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        
        # Initial patch embedding (dynamic)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4 stages like Swin-T
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]  # [96, 192, 384, 768]
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            input_dim = embed_dim if i == 0 else dims[i-1]
            output_dim = dims[i]
            self.stages.append(
                DynamicSwinStage(input_dim, output_dim, depths[i], num_heads[i])
            )
            
        self.final_norm = nn.LayerNorm(dims[-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/4, W/4)
        B, C, H, W = x.shape
        
        # Apply layer norm
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Apply stages
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        # Final normalization on the last feature
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.final_norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x  # Return final stage output

# -----------------------------------------------------------------------------
#  Cross‑Attention Transformer Module (CATM)
# -----------------------------------------------------------------------------

class CATM(nn.Module):
    """Cross‑Attention Transformer Module (CATM).
    Implements Algorithm 1 in the manuscript.
    1.  A Swin‑Transformer block extracts Q/K/V from *decoder* features.
    2.  Encoder skip features **query** the decoder features via cross‑attention.
    3.  Shared SA (implemented with a 1×1 conv + ReLU) refines the fusion.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.swin = SwinTransformerBlock(dim, num_heads)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Shared self‑attention (very lightweight ‑ 1×1 conv + BN + ReLU)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_skip: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_skip.shape

        # 1) Swin block on the decoder side ➜ provides richer K/V features
        dec_feat = self.swin(x_decoder)  # (B,C,H,W)

        # 2) Flatten to sequences for Multi‑Head Attention
        skip_seq = x_skip.flatten(2).transpose(1, 2)  # (B, HW, C)
        dec_seq = dec_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        q = self.q_proj(skip_seq)
        k = self.k_proj(dec_seq)
        v = self.v_proj(dec_seq)

        attn_out, _ = self.cross_attn(q, k, v)  # (B, HW, C)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)  # ➜ (B,C,H,W)

        # 3) Shared self‑attention refinement & residual connection
        fused = self.fuse(attn_out + x_skip)
        return fused
    
# -----------------------------------------------------------------------------
# Adaptive Fusion Block 
# -----------------------------------------------------------------------------

class AdaptiveFusionBlock(nn.Module):
    """Given an input feature map `X`, it computes:
      1. `X_swin`   = DynamicSwinTransformer(X)  # Now dynamic!
      2. `Offset`   = Conv2D_3x3(X)         → offset map for deform conv
      3. `X_deform` = DeformConv(X, Offset)
      4. Concatenate [X, X_swin, X_deform]
      5. Fuse with 1x1 conv to produce final output
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Dynamic Swin Transformer (no need for RGB conversion)
        self.swin = DynamicSwinTransformer(in_channels, embed_dim=in_channels)
        
        # Project Swin output back to input channels if needed
        # Since our dynamic Swin outputs 8*embed_dim, we need projection
        swin_out_dim = in_channels * 8  # Final stage output
        self.proj_back = nn.Conv2d(swin_out_dim, in_channels, kernel_size=1)

        # Offset + deformable convolution (18 = 2 * 3 * 3 for offset + mask)
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = ops.DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Fusion layer
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Dynamic Swin Transformer path
        original_size = x.shape[-2:]
        swin_feat = self.swin(x)  # Dynamic Swin handles any size
        swin_feat = self.proj_back(swin_feat)
        
        # Resize to match input size if needed
        if swin_feat.shape[-2:] != original_size:
            swin_feat = F.interpolate(swin_feat, size=original_size, mode="bilinear", align_corners=False)

        # Step 2–3: Offset + DeformConv path
        offset = self.offset(x)
        deform_feat = self.deform_conv(x, offset)

        # Step 4–5: Concatenate + 1×1 conv fusion
        combined = torch.cat([x, swin_feat, deform_feat], dim=1)
        out = self.act(self.norm(self.fuse(combined)))
        return out
    
class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))
    
# -----------------------------------------------------------------------------
#  ScaleFusionNet 
# -----------------------------------------------------------------------------
      
class ScaleFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,   # Match Swin-T base
        num_levels: int = 4,
    ):
        super().__init__()

        # --- Encoder: 4× PatchEmbedding + AdaptiveFusionBlock ---
        self.patch_embed = nn.ModuleList()
        self.adaptive_blocks = nn.ModuleList()

        embed_dims = [base_channels * (2 ** i) for i in range(num_levels)]
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else embed_dims[i - 1]
            out_ch = embed_dims[i]
            self.patch_embed.append(PatchEmbedding(in_ch, out_ch, patch_size=4 if i == 0 else 2))
            self.adaptive_blocks.append(AdaptiveFusionBlock(out_ch))

        # --- Decoder: CATM at 3 levels (not deepest), UpConv + AFB ---
        self.upconvs = nn.ModuleList()
        self.catms = nn.ModuleList()
        self.afbs = nn.ModuleList()

        # Build decoder from deepest to shallowest (3 levels)
        for i in reversed(range(1, num_levels)):
            self.upconvs.append(nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], 2, 2))
            self.catms.append(CATM(embed_dims[i - 1]))
            self.afbs.append(AdaptiveFusionBlock(embed_dims[i - 1]))

        # Final segmentation head
        self.seg_head = nn.Conv2d(embed_dims[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        
        # Store original input size for final interpolation
        original_size = x.shape[-2:]

        # Encoder: PatchEmbedding + AdaptiveFusionBlock at each level
        for pe, afb in zip(self.patch_embed, self.adaptive_blocks):
            x = pe(x)     # Patch embedding (downsampling)
            x = afb(x)    # Adaptive Fusion Block
            skips.append(x)

        # Decoder: Start from deepest feature, go up 3 levels
        x = skips[-1]  # Start with deepest features
        
        for i in range(3):
            x = self.upconvs[i](x)                    # Upsample
            skip_idx = -(i + 2)                       # Get corresponding skip connection
            x = self.catms[i](skips[skip_idx], x)     # CATM(skip, decoder)
            x = self.afbs[i](x)                       # Adaptive Fusion Block

        # Final segmentation prediction
        x = self.seg_head(x)
        
        # Interpolate to original input size
        x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
        return x


if __name__ == '__main__':
    model = ScaleFusionNet()
    dummy_input = torch.randn(1, 3, 256, 256)

    print("\nModel Summary:")
    summary(model, input_size=(3, 256, 256), device="cpu")

    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    
