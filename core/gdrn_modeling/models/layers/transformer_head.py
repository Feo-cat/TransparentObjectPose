from .attention import AttentionRope, FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossOnlyBlockRope
from .ffn import Mlp
import math
import torch
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        attn_class=FlashAttentionRope,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class can be FlashAttentionRope or AttentionRope for stability.
                attn_class=attn_class,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d(nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)


class DecoderHead(nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat


class FeatureFusionBlock(nn.Module):
    """最省显存、最安全的 RefineNet 融合块"""
    def __init__(self, features):
        super().__init__()
        self.res_conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.res_conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x_deep, x_skip):
        if x_deep is not None:
            # 🌟 Nearest 插值：0 显存缓存负担，绝对不报错 🌟
            x_up = F.interpolate(x_deep, size=x_skip.shape[-2:], mode="nearest")
            out = x_up + x_skip
        else:
            out = x_skip
            
        res = self.relu(out)
        res = self.res_conv1(res)
        res = self.relu(res)
        res = self.res_conv2(res)
        return out + res


class DepthHead(nn.Module):
    """全卷积 DPT Head（Nearest 插值 + 极致显存优化版）"""
    def __init__(self, in_channels, hidden_channels=256, mask_activation="none"):
        super().__init__()
        self.mask_activation = str(mask_activation).lower()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU()
        )
        
        # 🌟 显存优化 1：用 Nearest + Conv 取代所有耗显存的转置卷积
        self.resize_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode='nearest'),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
            ),
            nn.Identity(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=True)
        ])
        
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(hidden_channels) for _ in range(4)
        ])
        
        # 🌟 显存优化 2：全图上采样前，将通道数暴降至 32 维！
        head_features = 32  
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, head_features, kernel_size=3, padding=1, bias=True),
            nn.GELU()
        )
        
        self.depth_head = nn.Conv2d(head_features, 1, kernel_size=3, padding=1, bias=True)
        self.mask_head = nn.Conv2d(head_features, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x, out_size):
        x = self.proj(x)
        
        resized_features = [resize(x) for resize in self.resize_layers]
        
        f = self.fusion_blocks[3](None, resized_features[3])        
        f = self.fusion_blocks[2](f, resized_features[2])           
        f = self.fusion_blocks[1](f, resized_features[1])           
        f = self.fusion_blocks[0](f, resized_features[0])           
        
        # 降维到 32 通道
        f = self.output_conv(f)
        
        # 用 nearest 放大到 640x480，因为通道数仅为 32，显存占用极低
        f = F.interpolate(f, size=out_size, mode="nearest")
        
        # 在最高分辨率上只做最后一次 3x3 卷积，消除锯齿
        depth = self.depth_head(f)
        mask = self.mask_head(f)
        
        if self.mask_activation == "sigmoid":
            mask = torch.sigmoid(mask)
            
        return depth, mask




class ContextOnlyTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        prenorm=False,
        use_checkpoint=True,
    ):
        super().__init__()

        if prenorm:
            self.pre_norm = nn.LayerNorm(in_dim)
        else:
            self.pre_norm = None

        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            CrossOnlyBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        if self.pre_norm is not None:
            hidden = self.pre_norm(hidden)
            context = self.pre_norm(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, context, xpos=xpos, ypos=ypos, use_reentrant=False)
            else:
                hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)
        return out