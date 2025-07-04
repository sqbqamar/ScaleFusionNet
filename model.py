# -*- coding: utf-8 -*-
"""
@author: drsaq
"""

import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchsummary import summary

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

class DynamicPatchMerging(nn.Module):
    """Dynamic patch merging that works with any spatial resolution."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if H <= 1 or W <= 1:
            conv = nn.Conv2d(C, C*4, kernel_size=1, bias=False).to(x.device)
            x = conv(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = self.reduction(x)
            x = x.transpose(1, 2).view(B, -1, H, W)
            return x
        
        pad_h = pad_w = 0
        if H % 2 == 1:
            pad_h = 1
        if W % 2 == 1:
            pad_w = 1
            
        if pad_h > 0 or pad_w > 0:
            if H >= pad_h and W >= pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                H += pad_h
                W += pad_w
            else:
                conv = nn.Conv2d(C, C*4, kernel_size=1, bias=False).to(x.device)
                x = conv(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = self.reduction(x)
                x = x.transpose(1, 2).view(B, -1, H, W)
                return x
        
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, H//2, W//2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class DynamicSwinStage(nn.Module):
    """A stage of Swin Transformer with dynamic patch merging."""
    def __init__(self, input_dim: int, output_dim: int, depth: int = 2, num_heads: int = 3):
        super().__init__()
        if input_dim != output_dim:
            self.patch_merge = DynamicPatchMerging(input_dim, output_dim)
            self.use_patch_merge = True
        else:
            self.patch_merge = None
            self.use_patch_merge = False
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(output_dim, num_heads) for _ in range(depth)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_patch_merge and self.patch_merge is not None:
            x = self.patch_merge(x)
        
        for block in self.blocks:
            x = block(x)
            
        return x

class DynamicSwinTransformer(nn.Module):
    """Dynamic Swin Transformer that handles arbitrary input sizes."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)
        self.norm = nn.LayerNorm(embed_dim)
        
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
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
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class CATM(nn.Module):
    """Cross-Attention Transformer Module with corrected SharedSA (AvgPool + MaxPool)."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.swin = SwinTransformerBlock(dim, num_heads)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Corrected SharedSA implementation with AvgPool + MaxPool
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),  # Concatenates [X, AvgPool(X), MaxPool(X)]
            nn.Sigmoid()  # Spatial attention gating
        )

    def forward_shared_sa(self, x: torch.Tensor) -> torch.Tensor:
        """Implements SharedSA with AvgPool + MaxPool as per manuscript."""
        avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1)).expand_as(x)
        max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1))[0].expand_as(x)
        pooled = torch.cat([x, avg_pool, max_pool], dim=1)
        attention = self.fuse(pooled)
        return attention * x

    def forward(self, x_skip: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_skip.shape
        dec_feat = self.swin(x_decoder)
        skip_seq = x_skip.flatten(2).transpose(1, 2)
        dec_seq = dec_feat.flatten(2).transpose(1, 2)

        q = self.q_proj(skip_seq)
        k = self.k_proj(dec_seq)
        v = self.v_proj(dec_seq)

        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        
        # Use corrected SharedSA
        fused = self.forward_shared_sa(attn_out + x_skip)
        return fused

class AdaptiveFusionBlock(nn.Module):
    """Given an input feature map `X`, it computes:
      1. `X_swin`   = Selective Swin stages based on resolution
      2. `Offset`   = Conv2D_3x3(X) → offset map for deform conv
      3. `X_deform` = DeformConv(X, Offset)
      4. Concatenate [X, X_swin, X_deform]
      5. Fuse with 1x1 conv to produce final output
    """

    def __init__(self, in_channels: int, encoder_level: int = 0):
        super().__init__()
        
        self.encoder_level = encoder_level
        self.in_channels = in_channels
        
        # Calculate expected resolution at this encoder level
        # Level 0: H/4, W/4 (after first patch embedding)
        # Level 1: H/8, W/8 
        # Level 2: H/16, W/16
        # Level 3: H/32, W/32
        self.expected_reduction = 4 * (2 ** encoder_level)
        
        # Adaptive Swin stages based on encoder level and expected resolution
        if encoder_level == 0:
            # Early stage: Use first 2 stages only
            self.num_swin_stages = 2
            self.swin_embed_dim = min(in_channels // 4, 96)  # Conservative embedding
        elif encoder_level == 1:
            # Mid stage: Use first 3 stages
            self.num_swin_stages = 3
            self.swin_embed_dim = min(in_channels // 2, 96)
        elif encoder_level == 2:
            # Late stage: Use all 4 stages but with smaller embedding
            self.num_swin_stages = 4
            self.swin_embed_dim = min(in_channels // 2, 64)  # Smaller for higher resolution
        else:
            # Final stage (encoder_level == 3): Skip Swin entirely for very small features
            self.num_swin_stages = 0
            self.swin_embed_dim = 32
            
        # Create selective Swin Transformer
        if self.num_swin_stages > 0:
            self.swin = SelectiveSwinTransformer(
                in_channels=in_channels, 
                embed_dim=self.swin_embed_dim,
                num_stages=self.num_swin_stages
            )
            # Project back to input channels
            swin_out_dim = self.swin_embed_dim * (2 ** (self.num_swin_stages - 1))
            self.proj_back = nn.Conv2d(swin_out_dim, in_channels, kernel_size=1)
        else:
            # For very deep levels, use simple conv instead of Swin
            self.swin = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.proj_back = nn.Identity()

        # Offset + deformable convolution
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = ops.DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Fusion layer
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        
        # Step 1: Adaptive Swin Transformer path
        if self.num_swin_stages > 0:
            swin_feat = self.swin(x)
            swin_feat = self.proj_back(swin_feat)
            
            # Resize to match input size if needed
            if swin_feat.shape[-2:] != original_size:
                swin_feat = F.interpolate(swin_feat, size=original_size, mode="bilinear", align_corners=False)
        else:
            # For deepest level, use simple processing
            swin_feat = self.swin(x)

        # Step 2-3: Offset + DeformConv path
        offset = self.offset(x)
        deform_feat = self.deform_conv(x, offset)

        # Step 4-5: Concatenate + 1×1 conv fusion
        combined = torch.cat([x, swin_feat, deform_feat], dim=1)
        out = self.act(self.norm(self.fuse(combined)))
        return out


class SelectiveSwinTransformer(nn.Module):
    """Selective Swin Transformer that only uses specified number of stages."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 96, num_stages: int = 4):
        super().__init__()
        
        self.num_stages = min(num_stages, 4)  # Max 4 stages
        
        # Initial patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Only create the requested number of stages
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        depths = [2, 2, 6, 2][:self.num_stages]
        num_heads = [3, 6, 12, 24][:self.num_stages]
        
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            input_dim = embed_dim if i == 0 else dims[i-1]
            output_dim = dims[i]
            self.stages.append(
                DynamicSwinStage(input_dim, output_dim, depths[i], num_heads[i])
            )
            
        # Final norm only for the last stage we actually use
        self.final_norm = nn.LayerNorm(dims[-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is too small for patch embedding
        if x.shape[-1] < 4 or x.shape[-2] < 4:
            # Use 1x1 conv instead
            conv = nn.Conv2d(x.shape[1], self.embed_dim, kernel_size=1).to(x.device)
            x = conv(x)
        else:
            # Normal patch embedding
            x = self.patch_embed(x)
        
        B, C, H, W = x.shape
        
        # Apply layer norm
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Apply only the requested number of stages
        for stage in self.stages:
            x = stage(x)
            
        # Final normalization
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x

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
    
    
# Updated ScaleFusionNet with encoder level information
class ScaleFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
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
            # Pass encoder level to AdaptiveFusionBlock
            self.adaptive_blocks.append(AdaptiveFusionBlock(out_ch, encoder_level=i))

        # --- Decoder: CATM at 3 levels (not deepest), UpConv + AFB ---
        self.upconvs = nn.ModuleList()
        self.catms = nn.ModuleList()
        self.afbs = nn.ModuleList()

        # Build decoder from deepest to shallowest (3 levels)
        for i in reversed(range(1, num_levels)):
            self.upconvs.append(nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], 2, 2))
            self.catms.append(CATM(embed_dims[i - 1]))
            # Decoder AFBs use encoder level information too
            decoder_level = num_levels - 1 - i  # Convert to decoder level
            self.afbs.append(AdaptiveFusionBlock(embed_dims[i - 1], encoder_level=decoder_level))

        # Final segmentation head
        self.seg_head = nn.Conv2d(embed_dims[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        original_size = x.shape[-2:]

        # Encoder
        for pe, afb in zip(self.patch_embed, self.adaptive_blocks):
            x = pe(x)
            x = afb(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for i in range(3):
            x = self.upconvs[i](x)
            skip_idx = -(i + 2)
            x = self.catms[i](skips[skip_idx], x)
            x = self.afbs[i](x)

        # Final prediction
        x = self.seg_head(x)
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



