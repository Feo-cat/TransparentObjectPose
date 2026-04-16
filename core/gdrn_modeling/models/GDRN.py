import logging
import math
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils import quaternion_lf, lie_algebra
from core.utils.solver_utils import build_optimizer_with_params

from ..losses.l2_loss import L2Loss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from ..losses.loss_ops import (
    apply_mtl_uncertainty,
    build_pm_loss,
    compute_bind_loss,
    compute_centroid_loss,
    compute_mask_loss,
    compute_region_loss,
    compute_rotation_loss,
    compute_translation_losses,
    compute_xyz_losses,
    compute_z_loss,
)
from ..view_interaction_utils import (
    analyze_target_indices,
    build_context_pose_confidence,
    build_context_view_geom_features,
)
from ..losses.relative_geo_sym_loss import RelativeGeoSymLoss
from .cdpn_rot_head_region import RotWithRegionHead
from .cdpn_trans_head import TransHeadNet
from .context_geo_prior import ContextGeometricPriorInputs, ContextGeometricPriorModule

# pnp net variants
from .conv_pnp_net import ConvPnPNet
from .model_utils import (
    clean_mask_with_temporal_prior,
    compute_mean_re_te,
    get_mask_prob,
    masks_to_roi_geometry,
)
from .point_pnp_net import PointPnPNet, SimplePointPnPNet
from .pose_from_pred import decode_pose_from_raw
from .resnet_backbone import ResNetBackboneNet, resnet_spec

# attention blocks
from .layers.block import BlockRope
from .layers.attention import AttentionRope, FlashAttentionRope
from .layers.ffn import Mlp
from .layers.pos_embed import PositionGetter, RoPE2D
from .layers.transformer_head import TransformerDecoder, DecoderHead, DepthHead

import sys
sys.path.append("/mnt/afs/TransparentObjectPose/dinov3")
from dinov3.hub.backbones import dinov3_vits16


logger = logging.getLogger(__name__)


def _env_flag(name, default=False):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# View Interaction Module (Proposal V1)
# ---------------------------------------------------------------------------

def _backproject_depth_to_xyz(depth, mask, R, t, K):
    """Differentiable backprojection: pixel depth → object-frame XYZ.

    Args:
        depth:  (B, 1, H, W)  predicted depth (camera-frame metric depth).
        mask:   (B, 1, H, W)  binary-ish mask (>0 marks valid object pixels).
        R:      (B, 3, 3)     rotation  cam_R_m2c  (object → camera).
        t:      (B, 3)        translation cam_t_m2c.
        K:      (B, 3, 3)     camera intrinsics.

    Returns:
        xyz_obj: (B, 3, H, W) per-pixel object-frame coordinates, zero outside mask.
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # pixel grid (B, 3, H*W):  [u, v, 1]
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    pixel_coords = torch.stack([grid_x, grid_y, ones], dim=0)  # (3, H, W)
    pixel_coords = pixel_coords.reshape(3, H * W).unsqueeze(0).expand(B, -1, -1)  # (B, 3, HW)

    # unproject to camera frame
    K_inv = torch.inverse(K)                           # (B, 3, 3)
    cam_pts = torch.bmm(K_inv, pixel_coords)           # (B, 3, HW)
    depth_flat = depth.reshape(B, 1, H * W)            # (B, 1, HW)
    cam_pts = cam_pts * depth_flat                     # (B, 3, HW)

    # camera → object:  X_obj = R^T (X_cam - t)
    cam_pts = cam_pts - t.unsqueeze(-1)                # (B, 3, HW)
    R_t = R.transpose(1, 2)                            # (B, 3, 3)
    obj_pts = torch.bmm(R_t, cam_pts)                  # (B, 3, HW)
    obj_pts = obj_pts.reshape(B, 3, H, W)

    # zero out outside mask
    valid = (mask > 0.5).float()                       # (B, 1, H, W)
    xyz_obj = obj_pts * valid
    return xyz_obj


class ViewInteractionModule(nn.Module):
    """Cross-view context info injection for transparent object pose estimation (V1).

    Builds context-memory tokens (roi_feat + xyz_cue + rgb + coord2d + mask)
    and target-query tokens (roi_feat + rgb + coord2d + mask), performs
    cross-attention, and produces a residual feature that enriches the target
    roi_feat with context-view information before the shared rot_head_net +
    pnp_net second pass.

    Designed for the context-then-target AR schedule.
    """

    def __init__(
        self,
        roi_feat_dim=512,
        token_dim=256,
        num_heads=4,
        num_attn_layers=2,
        roi_grid_size=8,
        roi_input_res=256,
        max_views=12,
        rgb_embed_dim=32,
    ):
        super().__init__()
        self.roi_feat_dim = roi_feat_dim
        self.token_dim = token_dim
        self.roi_grid_size = roi_grid_size
        self.roi_input_res = roi_input_res

        # --- rgb embed: pixel_unshuffle + 1x1 conv ---
        ps_factor = roi_input_res // roi_grid_size  # 256 // 8 = 32
        self.ps_factor = ps_factor
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(3 * ps_factor * ps_factor, rgb_embed_dim, 1, bias=True),
            nn.GELU(),
        )

        # --- context MLP (roi_feat + xyz3 + coord2d + mask1 + rgb) → token_dim ---
        ctx_in = roi_feat_dim + 3 + 2 + 1 + rgb_embed_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # --- target MLP (roi_feat + coord2d + mask1 + rgb) → token_dim ---
        tgt_in = roi_feat_dim + 2 + 1 + rgb_embed_dim
        self.tgt_proj = nn.Sequential(
            nn.Linear(tgt_in, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # --- role & time embeddings ---
        self.role_embed = nn.Embedding(2, token_dim)   # 0=context, 1=target
        self.time_embed = nn.Embedding(max_views, token_dim)

        # --- cross-attention layers ---
        attn_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(attn_layer, num_layers=num_attn_layers)

        # --- output projection back to roi_feat_dim ---
        self.out_proj = nn.Conv2d(token_dim, roi_feat_dim, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        # zero-init out_proj so interaction starts as identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ----- helpers -----

    def _make_rgb_embed(self, roi_rgb):
        """roi_rgb: (N, 3, 256, 256) → (N, rgb_dim, 8, 8)"""
        x = F.pixel_unshuffle(roi_rgb, self.ps_factor)  # (N, 3*32*32, 8, 8)
        return self.rgb_proj(x)                          # (N, rgb_dim, 8, 8)

    def _downsample_to_grid(self, x):
        """Downsample (N, C, 256, 256) → (N, C, 8, 8) via area pooling."""
        if x.shape[-1] != self.roi_grid_size:
            x = F.adaptive_avg_pool2d(x, self.roi_grid_size)
        return x

    def forward(
        self,
        target_roi_feat,          # (B*T, 512, 8, 8)   abs-head target features
        context_roi_feat,         # (B*C, 512, 8, 8)   context features
        context_xyz_cue,          # (B*C, 3, 8, 8)     object-frame xyz
        target_roi_rgb,           # (B*T, 3, 256, 256)  target RoI crops
        context_roi_rgb,          # (B*C, 3, 256, 256)  context RoI crops
        target_roi_coord2d,       # (B*T, 2, G, G)      normalized 2D coords in RoI
        context_roi_coord2d,      # (B*C, 2, G, G)
        target_roi_mask,          # (B*T, 1, G, G)      mask/confidence
        context_roi_mask,         # (B*C, 1, G, G)
        batch_size,               # B
        target_num,               # T
        context_num,              # C
        target_time_ids,          # list of length T, view indices in window
        context_time_ids,         # list of length C, view indices in window
    ):
        """
        Returns:
            fused_feat: (B*T, 512, 8, 8) — target features enriched with context info.
        """
        G = self.roi_grid_size  # 8
        BT = target_roi_feat.shape[0]
        BC = context_roi_feat.shape[0]
        B = batch_size
        T = target_num
        C = context_num
        device = target_roi_feat.device
        dtype = target_roi_feat.dtype

        # --- RGB embeds ---
        tgt_rgb = self._make_rgb_embed(target_roi_rgb)    # (B*T, rgb_dim, G, G)
        ctx_rgb = self._make_rgb_embed(context_roi_rgb)    # (B*C, rgb_dim, G, G)

        # --- Downsample coord2d and mask to token grid ---
        tgt_coord = self._downsample_to_grid(target_roi_coord2d)   # (B*T, 2, G, G)
        ctx_coord = self._downsample_to_grid(context_roi_coord2d)  # (B*C, 2, G, G)
        tgt_mask = self._downsample_to_grid(target_roi_mask)       # (B*T, 1, G, G)
        ctx_mask = self._downsample_to_grid(context_roi_mask)      # (B*C, 1, G, G)

        # --- Assemble per-token features and flatten spatial dims ---
        # target: (B*T, roi_feat_dim + 2 + 1 + rgb_dim, G, G) → (B*T, G*G, feat_in)
        tgt_cat = torch.cat([target_roi_feat, tgt_coord, tgt_mask, tgt_rgb], dim=1)
        tgt_cat = tgt_cat.flatten(2).transpose(1, 2)  # (B*T, G*G, feat_in)

        # context: (B*C, roi_feat_dim + 3 + 2 + 1 + rgb_dim, G, G) → (B*C, G*G, feat_in)
        ctx_cat = torch.cat([context_roi_feat, context_xyz_cue, ctx_coord, ctx_mask, ctx_rgb], dim=1)
        ctx_cat = ctx_cat.flatten(2).transpose(1, 2)  # (B*C, G*G, feat_in)

        # --- Project to token_dim ---
        tgt_tokens = self.tgt_proj(tgt_cat)  # (B*T, G*G, token_dim)
        ctx_tokens = self.ctx_proj(ctx_cat)  # (B*C, G*G, token_dim)

        # --- Add role embeddings ---
        tgt_tokens = tgt_tokens + self.role_embed(torch.ones(1, dtype=torch.long, device=device))   # target=1
        ctx_tokens = ctx_tokens + self.role_embed(torch.zeros(1, dtype=torch.long, device=device))   # context=0

        # --- Add time embeddings ---
        # target_time_ids / context_time_ids are view-level indices in the window
        tgt_time = torch.tensor(target_time_ids, device=device, dtype=torch.long)  # (T,)
        ctx_time = torch.tensor(context_time_ids, device=device, dtype=torch.long)  # (C,)

        # Expand time embedding per batch element: (B*T, G*G, token_dim)
        tgt_time_emb = self.time_embed(tgt_time)  # (T, token_dim)
        tgt_time_emb = tgt_time_emb.repeat(B, 1)  # (B*T, token_dim)
        tgt_tokens = tgt_tokens + tgt_time_emb.unsqueeze(1)

        ctx_time_emb = self.time_embed(ctx_time)  # (C, token_dim)
        ctx_time_emb = ctx_time_emb.repeat(B, 1)  # (B*C, token_dim)
        ctx_tokens = ctx_tokens + ctx_time_emb.unsqueeze(1)

        # --- Reshape to (B, V*G*G, token_dim) for batched cross-attention ---
        tgt_tokens = tgt_tokens.reshape(B, T * G * G, -1)  # (B, T*G*G, d)
        ctx_tokens = ctx_tokens.reshape(B, C * G * G, -1)  # (B, C*G*G, d)

        # --- Cross attention: target queries attend to context memory ---
        ctx_enriched = self.cross_attn(tgt_tokens, ctx_tokens)  # (B, T*G*G, d)

        # --- Reshape back to spatial feature map ---
        ctx_enriched = ctx_enriched.reshape(B * T, G * G, -1)  # (B*T, G*G, d)
        ctx_enriched = ctx_enriched.transpose(1, 2).reshape(B * T, -1, G, G)  # (B*T, d, G, G)

        # --- Residual fusion ---
        fused_feat = target_roi_feat + self.out_proj(ctx_enriched)  # (B*T, 512, 8, 8)
        return fused_feat




class GDRN(nn.Module):
    def __init__(self, cfg, backbone, rot_head_net, trans_head_net=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.CDPN.NAME == "GDRN", cfg.MODEL.CDPN.NAME
        
        # here backbone can be deleted because we use DINO encoder + attention layers instead
        self.backbone = backbone
        if not cfg.MODEL.CDPN.RESNET_BACKBONE:
            del self.backbone

        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net

        self.trans_head_net = trans_head_net

        self.cfg = cfg
        self._hotspot_profile_enabled = str(os.environ.get("GDRN_PROFILE_HOTSPOTS", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._hotspot_profile_sync = str(os.environ.get("GDRN_PROFILE_SYNC", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._hotspot_profile_print_every = max(int(os.environ.get("GDRN_PROFILE_PRINT_EVERY", "1")), 1)
        self._hotspot_profile_forward_idx = 0
        self._hotspot_profile_stats = None
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT # be set to False by default
        self.r_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)
        self.rot_param_dim = self._get_rot_param_dim(cfg.MODEL.CDPN.PNP_NET.ROT_TYPE)
        self.dino_num_blocks = 0
        self.dino_tune_last_n_blocks = 0
        self._dino_trainable_block_start = 0
        self._dino_trainable_block_count = 0
        self._image_encoder_trainable = False
        self._image_encoder_use_no_grad = True
        
        # ---- DINO encoder ----
        if cfg.MODEL.CDPN.VGGT_BACKBONE:
            encoder_path = "/mnt/afs/TransparentObjectPose/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
            self.image_encoder = dinov3_vits16(pretrained=False, weights=encoder_path)
            self.image_encoder.load_state_dict(ckpt, strict=True)
            # self.image_encoder = torch.hub.load("/mnt/afs/TransparentObjectPose/dinov3", 'dinov3_vits16', source='local', weights="/mnt/afs/TransparentObjectPose/dinov3/weights/    dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

            dino_total_blocks = len(self.image_encoder.blocks)
            self.dino_num_blocks, self.attn_decoder_depth, self.roi_decoder_depth = (
                resolve_dino_and_decoder_depths(cfg.MODEL.CDPN, dino_total_blocks=dino_total_blocks)
            )
            self.dino_tune_last_n_blocks = resolve_dino_tune_last_n_blocks(
                cfg.MODEL.CDPN,
                dino_num_blocks=self.dino_num_blocks,
            )
            self.roi_feat_input_res = resolve_roi_feat_input_res(cfg.MODEL.CDPN)
            self.roi_decoder_dim, self.roi_head_out_dim = resolve_roi_decoder_feature_dims(
                cfg.MODEL.CDPN, dec_num_heads=16
            )
            if self.dino_num_blocks < dino_total_blocks:
                self.image_encoder.blocks = nn.ModuleList(list(self.image_encoder.blocks)[: self.dino_num_blocks])
                self.image_encoder.n_blocks = self.dino_num_blocks
            self._configure_dino_trainability()
            
        self.patch_size = 16

        # ---- attention layers ----
        if cfg.MODEL.CDPN.VGGT_BACKBONE:
            attn_impl = str(cfg.MODEL.CDPN.get("ATTN_IMPL", "flash")).lower()
            if attn_impl in ["flash", "flash_rope"]:
                attn_cls = FlashAttentionRope
            elif attn_impl in ["torch", "vanilla", "safe"]:
                attn_cls = AttentionRope
            else:
                raise ValueError(f"Unknown MODEL.CDPN.ATTN_IMPL: {attn_impl}")

            # (tiny setting)
            dec_embed_dim = 384
            # dec_num_heads = 4
            dec_num_heads = 2
            mlp_ratio = 4
            dec_depth = self.attn_decoder_depth
            
            self.pos_type = 'rope100'
            if self.pos_type.startswith('rope'): # eg rope100 
                if RoPE2D is None: 
                    raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
                freq = float(self.pos_type[len('rope'):])
                self.rope = RoPE2D(freq=freq)
                self.position_getter = PositionGetter()
            else:
                raise NotImplementedError
            
            self.decoder = nn.ModuleList([
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
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=attn_cls,
                    rope=self.rope
                ) for _ in range(dec_depth)])
        
            # ---- transformer decoder + head ----
            self.roi_decoder = TransformerDecoder(
                in_dim=2*dec_embed_dim, 
                dec_embed_dim=self.roi_decoder_dim,
                # dec_num_heads=8,
                # out_dim=256,
                dec_num_heads=8,
                out_dim=self.roi_decoder_dim,
                depth=self.roi_decoder_depth,
                rope=self.rope,
                attn_class=attn_cls,
            )
            self.roi_head = DecoderHead(
                patch_size=self.patch_size,
                dec_embed_dim=self.roi_decoder_dim,
                output_dim=self.roi_head_out_dim,
            )
            # self.roi_head = DecoderHead(patch_size=self.patch_size, dec_embed_dim=512, output_dim=64)
            # self.roi_head = DecoderHead(patch_size=self.patch_size, dec_embed_dim=256, output_dim=64)
            self.roi_feat_encoder = build_roi_feat_encoder(
                self.roi_feat_input_res, out_res=8, in_channels=self.roi_head_out_dim
            )

        # 假设提取的 4 层特征通道数相同
        feat_dim = 2 * dec_embed_dim

        # ---- View Interaction Module (Proposal V1) ----
        vi_cfg = cfg.MODEL.CDPN.get("VIEW_INTERACTION", {})
        self.vi_cfg = vi_cfg
        self.view_interaction_enabled = vi_cfg.get("ENABLED", False)
        if self.view_interaction_enabled:
            self.view_interaction = ViewInteractionModule(
                roi_feat_dim=512,                     # roi_feat_encoder output channels
                token_dim=vi_cfg.get("TOKEN_DIM", 256),
                num_heads=vi_cfg.get("NUM_HEADS", 4),
                num_attn_layers=vi_cfg.get("NUM_ATTN_LAYERS", 2),
                roi_grid_size=8,                      # roi_feat_encoder spatial output
                roi_input_res=256,                    # crop_resize_features output_size
                max_views=12,
                rgb_embed_dim=vi_cfg.get("RGB_EMBED_DIM", 32),
            )
        gp_cfg = vi_cfg.get("GEOMETRIC_PRIOR", {})
        self.geo_prior_enabled = bool(self.view_interaction_enabled and gp_cfg.get("ENABLED", False))
        self.geo_prior_cfg = gp_cfg
        if self.geo_prior_enabled:
            validate_geo_prior_resolution(gp_cfg, output_res=cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES)
            self.context_geo_prior = ContextGeometricPriorModule(
                roi_feat_dim=512,
                token_dim=vi_cfg.get("TOKEN_DIM", 256),
                retrieval_res=gp_cfg.get("RETRIEVAL_RES", 8),
                prior_res=gp_cfg.get("PRIOR_RES", 64),
                rgb_embed_dim=vi_cfg.get("RGB_EMBED_DIM", 32),
            )
        rel_geo_cfg = vi_cfg.get("RELATIVE_GEOMETRY_LOSS", {})
        self.rel_geo_loss_enabled = bool(rel_geo_cfg.get("ENABLED", False))
        self.relative_geo_sym_loss = RelativeGeoSymLoss()
        
        self.depth_head = DepthHead(
            in_channels=feat_dim,
            hidden_channels=256,
            mask_activation=cfg.MODEL.CDPN.get("DEPTH_HEAD_MASK_ACT", "none"),
        )

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        if cfg.MODEL.CDPN.USE_MTL:
            self.loss_names = [
                "mask",
                "coor_x",
                "coor_y",
                "coor_z",
                "coor_x_bin",
                "coor_y_bin",
                "coor_z_bin",
                "region",
                "PM_R",
                "PM_xy",
                "PM_z",
                "PM_xy_noP",
                "PM_z_noP",
                "PM_T",
                "PM_T_noP",
                "centroid",
                "z",
                "trans_xy",
                "trans_z",
                "trans_LPnP",
                "rot",
                "bind",
                "obj_mask",
                "dp_reg",
                "dp_gd",
                "dp_3d",
                "pose_flow",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )

    def _configure_dino_trainability(self):
        if not hasattr(self, "image_encoder"):
            self._dino_trainable_block_start = 0
            self._dino_trainable_block_count = 0
            self._image_encoder_trainable = False
            self._image_encoder_use_no_grad = True
            return

        freeze_encoder = bool(self.cfg.MODEL.CDPN.get("FREEZE_IMAGE_ENCODER", False))
        if freeze_encoder and self.dino_tune_last_n_blocks > 0:
            logger.info(
                "MODEL.CDPN.FREEZE_IMAGE_ENCODER=True, ignoring DINO_TUNE_LAST_N_BLOCKS=%d.",
                self.dino_tune_last_n_blocks,
            )

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        effective_tune_blocks = 0
        if not freeze_encoder and self.dino_tune_last_n_blocks > 0:
            effective_tune_blocks = min(int(self.dino_tune_last_n_blocks), len(self.image_encoder.blocks))
            if effective_tune_blocks > 0:
                for block in list(self.image_encoder.blocks)[-effective_tune_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
                if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
                    for param in self.image_encoder.norm.parameters():
                        param.requires_grad = True

        trainable_params = sum(param.numel() for param in self.image_encoder.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in self.image_encoder.parameters())
        self._dino_trainable_block_count = int(effective_tune_blocks)
        self._dino_trainable_block_start = max(len(self.image_encoder.blocks) - int(effective_tune_blocks), 0)
        self._image_encoder_trainable = trainable_params > 0
        self._image_encoder_use_no_grad = not self._image_encoder_trainable
        logger.info(
            "DINO trainability: kept_blocks=%d tune_last_n=%d effective_trainable_blocks=%d "
            "trainable_block_start=%d trainable_params=%d/%d use_no_grad=%s",
            int(self.dino_num_blocks),
            int(self.dino_tune_last_n_blocks),
            int(effective_tune_blocks),
            int(self._dino_trainable_block_start),
            int(trainable_params),
            int(total_params),
            str(self._image_encoder_use_no_grad),
        )

    def _run_image_encoder(self, x):
        if not self.training:
            with torch.no_grad():
                return self.image_encoder(x, is_training=True)

        if self._image_encoder_use_no_grad:
            with torch.no_grad():
                return self.image_encoder(x, is_training=True)

        # Full-train or no clean split available.
        if self._dino_trainable_block_start <= 0:
            return self.image_encoder(x, is_training=True)

        # Prefix frozen blocks under no_grad; only tail blocks participate in autograd.
        x_tokens, (h_tok, w_tok) = self.image_encoder.prepare_tokens_with_masks(x)
        total_blocks = len(self.image_encoder.blocks)
        split = min(max(int(self._dino_trainable_block_start), 0), total_blocks)

        with torch.no_grad():
            for i in range(split):
                rope_sincos = (
                    self.image_encoder.rope_embed(H=h_tok, W=w_tok)
                    if self.image_encoder.rope_embed is not None
                    else None
                )
                x_tokens = self.image_encoder.blocks[i](x_tokens, rope_sincos)

        for i in range(split, total_blocks):
            rope_sincos = (
                self.image_encoder.rope_embed(H=h_tok, W=w_tok)
                if self.image_encoder.rope_embed is not None
                else None
            )
            x_tokens = self.image_encoder.blocks[i](x_tokens, rope_sincos)

        n_storage_tokens = int(self.image_encoder.n_storage_tokens)
        if self.image_encoder.untie_cls_and_patch_norms or self.image_encoder.untie_global_and_local_cls_norm:
            if self.image_encoder.untie_cls_and_patch_norms:
                x_norm_cls_reg = self.image_encoder.cls_norm(x_tokens[:, : n_storage_tokens + 1])
            else:
                x_norm_cls_reg = self.image_encoder.norm(x_tokens[:, : n_storage_tokens + 1])
            x_norm_patch = self.image_encoder.norm(x_tokens[:, n_storage_tokens + 1 :])
        else:
            x_norm = self.image_encoder.norm(x_tokens)
            x_norm_cls_reg = x_norm[:, : n_storage_tokens + 1]
            x_norm_patch = x_norm[:, n_storage_tokens + 1 :]

        return {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": x_tokens,
            "masks": None,
        }

    def _hotspot_profile_active(self):
        return self._hotspot_profile_enabled and self.training

    def _hotspot_profile_is_main_process(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _hotspot_profile_start(self):
        if not self._hotspot_profile_active() or not self._hotspot_profile_is_main_process():
            return None
        if self._hotspot_profile_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _hotspot_profile_stop(self, label, start_t, **meta):
        if start_t is None or not self._hotspot_profile_active() or not self._hotspot_profile_is_main_process():
            return
        if self._hotspot_profile_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - start_t
        if self._hotspot_profile_stats is None:
            self._hotspot_profile_stats = {}
        stat = self._hotspot_profile_stats.setdefault(label, {"time": 0.0, "count": 0})
        stat["time"] += dt
        stat["count"] += 1
        for key, value in meta.items():
            if isinstance(value, (int, float, bool)):
                stat[key] = stat.get(key, 0.0) + float(value)

    def _hotspot_profile_begin_forward(self):
        if not self._hotspot_profile_active() or not self._hotspot_profile_is_main_process():
            return None
        self._hotspot_profile_forward_idx += 1
        self._hotspot_profile_stats = {}
        return self._hotspot_profile_start()

    def _hotspot_profile_end_forward(self, start_t, **meta):
        if start_t is None or not self._hotspot_profile_active() or not self._hotspot_profile_is_main_process():
            return
        self._hotspot_profile_stop("forward_total", start_t, **meta)
        if self._hotspot_profile_forward_idx % self._hotspot_profile_print_every != 0:
            return
        stats = self._hotspot_profile_stats or {}
        logger.info(
            "[GDRN_PROFILE] forward=%d summary_start entries=%d",
            self._hotspot_profile_forward_idx,
            len(stats),
        )
        for label, stat in sorted(stats.items(), key=lambda kv: kv[1].get("time", 0.0), reverse=True):
            count = max(int(stat.get("count", 0)), 1)
            extras = []
            for extra_key in ("pose_n", "batch_size", "target_view_num", "point_n", "target_num", "view_num"):
                if extra_key in stat:
                    extras.append(f"{extra_key}_avg={stat[extra_key] / count:.1f}")
            logger.info(
                "[GDRN_PROFILE] %s total=%.4fs count=%d avg=%.6fs %s",
                label,
                stat.get("time", 0.0),
                count,
                stat.get("time", 0.0) / count,
                " ".join(extras),
            )
        logger.info("[GDRN_PROFILE] forward=%d summary_end", self._hotspot_profile_forward_idx)

    @staticmethod
    def _get_rot_param_dim(rot_type):
        if rot_type in ["allo_quat", "ego_quat"]:
            return 4
        if rot_type in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            return 3
        if rot_type in ["allo_rot6d", "ego_rot6d"]:
            return 6
        raise ValueError(f"Unknown ROT_TYPE: {rot_type}")

    @staticmethod
    def _rot_repr_to_mat(pred_rot_raw, rot_type):
        if rot_type in ["ego_quat", "allo_quat"]:
            return quat2mat_torch(pred_rot_raw)
        if rot_type in ["ego_log_quat", "allo_log_quat"]:
            return quat2mat_torch(quaternion_lf.qexp(pred_rot_raw))
        if rot_type in ["ego_lie_vec", "allo_lie_vec"]:
            return lie_algebra.lie_vec_to_rot(pred_rot_raw)
        if rot_type in ["ego_rot6d", "allo_rot6d"]:
            return ortho6d_to_mat_batch(pred_rot_raw)
        raise RuntimeError(f"Wrong pred_rot dim: {pred_rot_raw.shape}")

    @staticmethod
    def _get_context_pose_from_full_gt(full_rot, full_trans, batch_size, context_last_idx, dtype, device):
        has_full_gt = (
            torch.is_tensor(full_rot)
            and torch.is_tensor(full_trans)
            and full_rot.dim() >= 4
            and full_trans.dim() >= 3
            and full_rot.shape[0] == batch_size
            and full_trans.shape[0] == batch_size
            and full_rot.shape[1] > context_last_idx
            and full_trans.shape[1] > context_last_idx
        )
        if has_full_gt:
            ctx_rot = full_rot[:, context_last_idx].to(dtype=dtype, device=device).contiguous()
            ctx_trans = full_trans[:, context_last_idx].to(dtype=dtype, device=device).contiguous()
        else:
            ctx_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            ctx_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return ctx_rot, ctx_trans

    def _encode_view_tokens_to_roi_features(self, view_hidden, roi_centers, scales, H, W):
        """Convert decoded per-view tokens into ROI-aligned feature maps for the pose heads."""
        if view_hidden.dim() == 4:
            batch_size, view_num, hw, feat_dim = view_hidden.shape
            view_hidden_flat = view_hidden.reshape(batch_size * view_num, hw, feat_dim).contiguous()
            roi_centers_flat = roi_centers.reshape(batch_size * view_num, 2).contiguous()
            scales_flat = scales.reshape(batch_size * view_num, 1).contiguous()
        elif view_hidden.dim() == 3:
            batch_size, hw, feat_dim = view_hidden.shape
            view_hidden_flat = view_hidden.contiguous()
            roi_centers_flat = roi_centers.reshape(batch_size, 2).contiguous()
            scales_flat = scales.reshape(batch_size, 1).contiguous()
        else:
            raise ValueError(f"Unsupported view_hidden shape: {view_hidden.shape}")

        h_p, w_p = H // self.patch_size, W // self.patch_size
        pos = self.position_getter(view_hidden_flat.shape[0], h_p, w_p, view_hidden_flat.device)
        dense_feat = self.roi_decoder(view_hidden_flat, xpos=pos)
        dense_feat = self.roi_head([dense_feat], (H, W))
        dense_feat = crop_resize_features(dense_feat, roi_centers_flat, scales_flat, output_size=self.roi_feat_input_res)
        dense_feat = self.roi_feat_encoder(dense_feat)
        return dense_feat

    def _encode_target_and_context_roi_features(
        self,
        target_hidden,
        target_roi_centers,
        target_scales,
        context_hidden,
        context_roi_centers,
        context_scales,
        H,
        W,
    ):
        """Jointly encode target/context view tokens into RoI features and split."""
        if target_hidden.dim() == 3:
            target_hidden = target_hidden.unsqueeze(1).contiguous()
        elif target_hidden.dim() != 4:
            raise ValueError(f"Unsupported target_hidden shape: {target_hidden.shape}")

        if context_hidden.dim() == 3:
            context_hidden = context_hidden.unsqueeze(1).contiguous()
        elif context_hidden.dim() != 4:
            raise ValueError(f"Unsupported context_hidden shape: {context_hidden.shape}")

        if target_roi_centers.dim() == 2:
            target_roi_centers = target_roi_centers.unsqueeze(1).contiguous()
        if target_scales.dim() == 2:
            target_scales = target_scales.unsqueeze(1).contiguous()
        if context_roi_centers.dim() == 2:
            context_roi_centers = context_roi_centers.unsqueeze(1).contiguous()
        if context_scales.dim() == 2:
            context_scales = context_scales.unsqueeze(1).contiguous()

        batch_size = target_hidden.shape[0]
        target_num = target_hidden.shape[1]
        context_num = context_hidden.shape[1]

        if context_num <= 0:
            target_feat = self._encode_view_tokens_to_roi_features(
                target_hidden, target_roi_centers, target_scales, H, W
            )
            return target_feat, None

        joint_hidden = torch.cat([target_hidden, context_hidden], dim=1).contiguous()
        joint_centers = torch.cat([target_roi_centers, context_roi_centers], dim=1).contiguous()
        joint_scales = torch.cat([target_scales, context_scales], dim=1).contiguous()
        joint_feat = self._encode_view_tokens_to_roi_features(joint_hidden, joint_centers, joint_scales, H, W)

        feat_shape = joint_feat.shape[1:]
        joint_feat = joint_feat.reshape(batch_size, target_num + context_num, *feat_shape)
        target_feat = joint_feat[:, :target_num].reshape(batch_size * target_num, *feat_shape).contiguous()
        context_feat = joint_feat[:, target_num:].reshape(batch_size * context_num, *feat_shape).contiguous()
        return target_feat, context_feat

    @staticmethod
    def _add_pose_noise_for_ctx_info(R, t, vi_cfg):
        """Add small random noise to context poses during training.

        This narrows the train/infer gap: at inference the context xyz cue is
        built from AR-predicted poses (which carry prediction error), while at
        training time it would otherwise use clean GT poses.

        Args:
            R: (N, 3, 3) rotation matrices.
            t: (N, 3)    translations.
            vi_cfg: VIEW_INTERACTION config dict (carries noise magnitude).

        Returns:
            R_noisy: (N, 3, 3), t_noisy: (N, 3).
        """
        rot_noise_deg = float(vi_cfg.get("CTX_POSE_NOISE_ROT_DEG", 5.0))
        trans_noise_ratio = float(vi_cfg.get("CTX_POSE_NOISE_TRANS_RATIO", 0.02))

        if rot_noise_deg <= 0 and trans_noise_ratio <= 0:
            return R, t

        N = R.shape[0]
        device = R.device
        dtype = R.dtype

        # Rotation noise: random axis-angle with magnitude ~ N(0, sigma_rad)
        if rot_noise_deg > 0:
            sigma_rad = rot_noise_deg * (math.pi / 180.0)
            axis = torch.randn(N, 3, device=device, dtype=dtype)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-8)
            angle = torch.randn(N, 1, device=device, dtype=dtype) * sigma_rad  # (N, 1)
            # Rodrigues: R_noise = I + sin(a)*K + (1-cos(a))*K^2
            K = torch.zeros(N, 3, 3, device=device, dtype=dtype)
            K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
            K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
            K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
            sin_a = torch.sin(angle).unsqueeze(-1)   # (N, 1, 1)
            cos_a = torch.cos(angle).unsqueeze(-1)   # (N, 1, 1)
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            R_noise = eye + sin_a * K + (1 - cos_a) * (K @ K)
            R = R_noise @ R

        # Translation noise: additive Gaussian proportional to ||t||
        if trans_noise_ratio > 0:
            t_norm = t.norm(dim=1, keepdim=True).clamp(min=1e-6)
            t = t + torch.randn_like(t) * trans_noise_ratio * t_norm

        return R, t

    def _build_context_supervise_pose_tensors(
        self,
        full_gt_ego_rot,
        full_gt_trans,
        context_gt_rot_bt,
        context_gt_trans_bt,
        context_pose_hypothesis_rot_bt,
        context_pose_hypothesis_trans_bt,
        model_infos,
        context_last_idx,
        batch_size,
        dtype,
        device,
    ):
        if context_gt_rot_bt is None or context_gt_trans_bt is None:
            context_gt_rot_bt, context_gt_trans_bt = self._get_context_pose_from_full_gt(
                full_gt_ego_rot, full_gt_trans, batch_size, context_last_idx, dtype=dtype, device=device
            )
        flat_infos = self._flatten_model_infos_for_pose_count(model_infos, batch_size)
        if flat_infos is not None:
            context_gt_rot_bt, context_gt_trans_bt = self._canonicalize_continuous_abs_pose_camera_facing(
                context_gt_rot_bt, context_gt_trans_bt, flat_infos
            )
            context_pose_hypothesis_rot_bt, context_pose_hypothesis_trans_bt = (
                self._canonicalize_continuous_abs_pose_camera_facing(
                    context_pose_hypothesis_rot_bt, context_pose_hypothesis_trans_bt, flat_infos
                )
            )
        return (
            context_gt_rot_bt,
            context_gt_trans_bt,
            context_pose_hypothesis_rot_bt,
            context_pose_hypothesis_trans_bt,
        )

    @staticmethod
    def _safe_get_discrete_sym_transforms(model_info, device, dtype):
        if not isinstance(model_info, dict) or "symmetries_discrete" not in model_info:
            return None, None
        sym_raw = model_info["symmetries_discrete"]
        if sym_raw is None or len(sym_raw) == 0:
            return None, None
        sym_arr = torch.as_tensor(sym_raw, device=device, dtype=dtype)
        if sym_arr.ndim == 2 and sym_arr.shape[-1] == 16:
            sym_arr = sym_arr.reshape(-1, 4, 4)
        elif sym_arr.ndim == 2 and sym_arr.shape[-1] == 9:
            sym_arr = sym_arr.reshape(-1, 3, 3)
        elif sym_arr.ndim == 3 and sym_arr.shape[-2:] in ((4, 4), (3, 3)):
            pass
        else:
            return None, None

        if sym_arr.shape[-2:] == (4, 4):
            sym_rot = sym_arr[:, :3, :3].contiguous()
            sym_trans = sym_arr[:, :3, 3].contiguous()
        else:
            sym_rot = sym_arr.contiguous()
            sym_trans = torch.zeros(sym_rot.shape[0], 3, device=device, dtype=dtype)

        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        zero = torch.zeros(1, 3, device=device, dtype=dtype)
        rot_is_identity = torch.isclose(
            sym_rot, eye.expand_as(sym_rot), atol=1e-6, rtol=1e-6
        ).reshape(sym_rot.shape[0], -1).all(dim=1)
        trans_is_zero = torch.isclose(
            sym_trans, zero.expand_as(sym_trans), atol=1e-6, rtol=1e-6
        ).reshape(sym_trans.shape[0], -1).all(dim=1)
        has_identity = bool((rot_is_identity & trans_is_zero).any().item())
        if not has_identity:
            sym_rot = torch.cat([eye, sym_rot], dim=0)
            sym_trans = torch.cat([zero, sym_trans], dim=0)
        return sym_rot, sym_trans

    @staticmethod
    def _safe_get_discrete_sym_rots(model_info, device, dtype):
        sym_rot, _ = GDRN._safe_get_discrete_sym_transforms(model_info, device, dtype)
        return sym_rot

    @staticmethod
    def _safe_get_continuous_sym_axis_offset(model_info, device, dtype):
        if not isinstance(model_info, dict) or "symmetries_continuous" not in model_info:
            return None, None
        sym_cont = model_info["symmetries_continuous"]
        if not isinstance(sym_cont, (list, tuple)) or len(sym_cont) == 0:
            return None, None
        axis_info = sym_cont[0]
        if not isinstance(axis_info, dict):
            return None, None
        axis = axis_info.get("axis", None) if isinstance(axis_info, dict) else None
        if axis is None:
            return None, None
        axis = torch.as_tensor(axis, device=device, dtype=dtype).reshape(-1)
        if axis.numel() != 3:
            return None, None
        axis = axis / axis.norm(p=2).clamp(min=1e-8)
        offset = axis_info.get("offset", [0.0, 0.0, 0.0])
        offset = torch.as_tensor(offset, device=device, dtype=dtype).reshape(-1)
        if offset.numel() != 3:
            offset = torch.zeros(3, device=device, dtype=dtype)
        return axis, offset

    @staticmethod
    def _safe_get_continuous_sym_axis(model_info, device, dtype):
        axis, _ = GDRN._safe_get_continuous_sym_axis_offset(model_info, device, dtype)
        return axis

    @staticmethod
    def _has_continuous_symmetry(model_info):
        if not isinstance(model_info, dict):
            return False
        sym_cont = model_info.get("symmetries_continuous", None)
        return isinstance(sym_cont, (list, tuple)) and len(sym_cont) > 0

    @staticmethod
    def _flatten_model_infos_for_pose_count(model_infos_batch, expected_pose_n):
        if model_infos_batch is None or not isinstance(model_infos_batch, (list, tuple)):
            return None
        if len(model_infos_batch) == expected_pose_n:
            return list(model_infos_batch)

        flat_model_infos = []
        for item in model_infos_batch:
            if isinstance(item, (list, tuple)):
                flat_model_infos.extend(list(item))
            else:
                flat_model_infos.append(item)
        if len(flat_model_infos) == expected_pose_n:
            return flat_model_infos

        if expected_pose_n > 0 and len(model_infos_batch) > 0 and expected_pose_n % len(model_infos_batch) == 0:
            repeat_factor = expected_pose_n // len(model_infos_batch)
            repeated = []
            for item in model_infos_batch:
                repeated.extend([item] * repeat_factor)
            if len(repeated) == expected_pose_n:
                return repeated
        return None

    @staticmethod
    def _safe_get_region_fps_points(model_info, num_regions, device, dtype):
        if not isinstance(model_info, dict):
            return None
        num_regions = int(num_regions)
        if num_regions <= 1:
            return None

        fps_points = model_info.get("fps_points", None)
        if fps_points is None:
            key = f"fps{num_regions}"
            key_center = f"fps{num_regions}_and_center"
            if key in model_info:
                fps_points = model_info[key]
            elif key_center in model_info:
                fps_points = model_info[key_center]
        if fps_points is None:
            return None

        fps_points = torch.as_tensor(fps_points, device=device, dtype=dtype)
        if fps_points.dim() != 2 or fps_points.shape[1] != 3:
            return None
        # Some datasets store "..._and_center". Region targets only need NUM_REGIONS anchors.
        if fps_points.shape[0] == num_regions + 1:
            fps_points = fps_points[:-1]
        if fps_points.shape[0] != num_regions:
            return None
        if not torch.isfinite(fps_points).all():
            return None
        return fps_points.contiguous()

    def _canonicalize_continuous_abs_pose_with_anchor(
        self,
        gt_abs_rot,
        gt_abs_trans,
        gt_rot_anchor,
        model_infos_flat,
    ):
        expected_pose_n = int(gt_abs_rot.shape[0]) if torch.is_tensor(gt_abs_rot) else -1
        if (
            expected_pose_n <= 0
            or (not torch.is_tensor(gt_rot_anchor))
            or gt_rot_anchor.shape[0] != expected_pose_n
            or model_infos_flat is None
            or len(model_infos_flat) != expected_pose_n
        ):
            return gt_abs_rot, gt_abs_trans, None, None

        gt_rot_supervise = gt_abs_rot.clone()
        gt_trans_supervise = (
            gt_abs_trans.clone()
            if torch.is_tensor(gt_abs_trans) and gt_abs_trans.shape[0] == expected_pose_n
            else gt_abs_trans
        )
        cont_mask = torch.zeros(expected_pose_n, dtype=torch.bool, device=gt_abs_rot.device)
        cont_axis = torch.zeros(expected_pose_n, 3, dtype=gt_abs_rot.dtype, device=gt_abs_rot.device)
        cont_offset = torch.zeros(expected_pose_n, 3, dtype=gt_abs_rot.dtype, device=gt_abs_rot.device)

        for i, item in enumerate(model_infos_flat):
            axis, offset = self._safe_get_continuous_sym_axis_offset(item, gt_abs_rot.device, gt_abs_rot.dtype)
            if axis is None:
                continue
            cont_mask[i] = True
            cont_axis[i] = axis
            cont_offset[i] = offset

        if not cont_mask.any():
            return gt_rot_supervise, gt_trans_supervise, cont_mask, cont_axis

        anchor_rots_cont = gt_rot_anchor[cont_mask]
        gt_abs_rot_cont = gt_abs_rot[cont_mask]
        axis_cont_local = cont_axis[cont_mask]
        offset_cont_local = cont_offset[cont_mask]

        # raw_rel_obj = R_anchor^T * R_target. We then pick the equivalent
        # symmetry clone whose axis direction is closest in this anchor frame.
        raw_rel_obj = torch.matmul(anchor_rots_cont.transpose(-1, -2), gt_abs_rot_cont)
        gt_axis_obj = torch.matmul(raw_rel_obj, axis_cont_local.unsqueeze(-1)).squeeze(-1)
        rel_obj_canon = self._minimal_rotation_between_unit_vectors(axis_cont_local, gt_axis_obj)

        # Right-multiply the raw target pose by the object-space symmetry transform.
        sym_rot_obj = torch.matmul(raw_rel_obj.transpose(-1, -2), rel_obj_canon)
        gt_rot_supervise[cont_mask] = torch.matmul(gt_abs_rot_cont, sym_rot_obj)

        if torch.is_tensor(gt_trans_supervise):
            sym_trans_obj = offset_cont_local - torch.matmul(sym_rot_obj, offset_cont_local.unsqueeze(-1)).squeeze(-1)
            gt_trans_supervise[cont_mask] = gt_abs_trans[cont_mask] + torch.matmul(
                gt_abs_rot_cont,
                sym_trans_obj.unsqueeze(-1),
            ).squeeze(-1)

        return gt_rot_supervise, gt_trans_supervise, cont_mask, cont_axis

    def _compute_mean_re_te_symmetry_aware(self, pred_transes, pred_rots, gt_transes, gt_rots, model_infos_batch=None):
        pred_rots_det = pred_rots.detach()
        gt_rots_det = gt_rots.detach()
        pred_trans_det = pred_transes.detach()
        gt_trans_det = gt_transes.detach()

        bs = pred_rots_det.shape[0]
        flat_model_infos = self._flatten_model_infos_for_pose_count(model_infos_batch, bs)
        R_errs = []
        T_errs = []

        for i in range(bs):
            pred_rot_i = pred_rots_det[i : i + 1]
            gt_rot_i = gt_rots_det[i : i + 1]
            pred_trans_i = pred_trans_det[i : i + 1]
            gt_trans_i = gt_trans_det[i : i + 1]
            model_info = flat_model_infos[i] if flat_model_infos is not None else None

            rot_err_i = None
            trans_err_i = torch.linalg.norm(pred_trans_i - gt_trans_i, dim=-1)[0]
            if model_info is not None:
                axis = self._safe_get_continuous_sym_axis(model_info, pred_rot_i.device, pred_rot_i.dtype)
                if axis is not None:
                    gt_rot_eval, gt_trans_eval, cont_mask, _ = self._canonicalize_continuous_abs_pose_with_anchor(
                        gt_rot_i,
                        gt_trans_i,
                        pred_rot_i,
                        [model_info],
                    )
                    if cont_mask is not None and cont_mask.any():
                        gt_rot_i = gt_rot_eval
                        if torch.is_tensor(gt_trans_eval):
                            gt_trans_i = gt_trans_eval
                    axis_col = axis.view(1, 3, 1)
                    pred_axis = torch.matmul(pred_rot_i, axis_col).squeeze(-1)
                    gt_axis = torch.matmul(gt_rot_i, axis_col).squeeze(-1)
                    rot_err_i = self._vector_angle_deg(pred_axis, gt_axis)[0]
                    trans_err_i = torch.linalg.norm(pred_trans_i - gt_trans_i, dim=-1)[0]
                else:
                    cand_abs_rot, cand_abs_trans = self._build_discrete_abs_pose_candidates(
                        gt_rot_i,
                        gt_trans_i,
                        model_info,
                        pred_rot_i.device,
                        pred_rot_i.dtype,
                    )
                    if cand_abs_rot is not None:
                        re_deg = self._rot_batch_angular_distance_deg(pred_rot_i.expand_as(cand_abs_rot), cand_abs_rot)
                        best_idx = torch.argmin(re_deg)
                        rot_err_i = re_deg[best_idx]
                        if cand_abs_trans is not None:
                            trans_err_i = torch.linalg.norm(pred_trans_i - cand_abs_trans[best_idx : best_idx + 1], dim=-1)[0]

            if rot_err_i is None:
                rot_err_i = self._rot_batch_angular_distance_deg(pred_rot_i, gt_rot_i)[0]
            R_errs.append(rot_err_i)
            T_errs.append(trans_err_i)

        return torch.stack(R_errs).mean().item(), torch.stack(T_errs).mean().item()

    @staticmethod
    def _compute_mean_t_xy_z_cm(pred_transes, gt_transes):
        pred_trans_det = pred_transes.detach()
        gt_trans_det = gt_transes.detach()
        xy_err_cm = torch.linalg.norm(pred_trans_det[:, :2] - gt_trans_det[:, :2], dim=-1).mean().item() * 100.0
        z_err_cm = torch.abs(pred_trans_det[:, 2] - gt_trans_det[:, 2]).mean().item() * 100.0
        return xy_err_cm, z_err_cm

    def _canonicalize_abs_pose_targets(
        self,
        cfg,
        gt_rot,
        gt_trans=None,
        model_infos=None,
        pred_rots_for_canon=None,
    ):
        """Canonicalize absolute GT pose targets.

        Discrete symmetry: hard-coded closest_pose — always use pred_rots_for_canon as anchor.
        Continuous symmetry: read CONT_SYM_GT_MODE config.
            "face_camera" -> camera-facing canonicalization (frame-local gauge)
            "closest_pose" -> prediction-anchored (legacy fallback)
        """
        expected_pose_n = int(gt_rot.shape[0]) if torch.is_tensor(gt_rot) else 0
        profile_t0 = self._hotspot_profile_start()
        try:
            if gt_rot is None or (not torch.is_tensor(gt_rot)):
                return gt_rot, gt_trans

            if expected_pose_n <= 0:
                return gt_rot, gt_trans

            model_infos_flat = self._flatten_model_infos_for_pose_count(model_infos, expected_pose_n)
            if model_infos_flat is None or len(model_infos_flat) != expected_pose_n:
                return gt_rot, gt_trans

            pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
            cont_sym_gt_mode = self._get_cont_sym_gt_mode(pnp_net_cfg)

            gt_rot_supervise = gt_rot.clone()
            gt_trans_supervise = (
                gt_trans.clone()
                if torch.is_tensor(gt_trans) and gt_trans.shape[0] == expected_pose_n
                else gt_trans
            )

            has_pred_anchor = (
                torch.is_tensor(pred_rots_for_canon)
                and pred_rots_for_canon.shape[0] == expected_pose_n
            )

            # --- Discrete symmetry: hard-coded closest_pose ---
            if has_pred_anchor:
                for i, item in enumerate(model_infos_flat):
                    if self._has_continuous_symmetry(item):
                        continue
                    cand_rot, cand_trans = self._build_discrete_abs_pose_candidates(
                        gt_rot[i : i + 1],
                        gt_trans[i : i + 1]
                        if torch.is_tensor(gt_trans) and gt_trans.shape[0] == expected_pose_n
                        else None,
                        item,
                        gt_rot.device,
                        gt_rot.dtype,
                    )
                    if cand_rot is None:
                        continue
                    ref_rot_i = pred_rots_for_canon[i : i + 1]
                    re_deg = self._rot_batch_angular_distance_deg(ref_rot_i.expand_as(cand_rot), cand_rot)
                    best_idx = torch.argmin(re_deg)
                    gt_rot_supervise[i] = cand_rot[best_idx]
                    if torch.is_tensor(gt_trans_supervise) and cand_trans is not None:
                        gt_trans_supervise[i] = cand_trans[best_idx]

            # --- Continuous symmetry ---
            if cont_sym_gt_mode == "face_camera":
                degenerate_thresh = float(
                    pnp_net_cfg.get("CONT_SYM_CANON_DEGENERATE_THRESH", 0.1)
                )
                gt_rot_supervise, gt_trans_supervise = self._canonicalize_continuous_abs_pose_camera_facing(
                    gt_rot_supervise,
                    gt_trans_supervise,
                    model_infos_flat,
                    degenerate_thresh=degenerate_thresh,
                )
            else:
                # "closest_pose" fallback: anchor on prediction
                if has_pred_anchor:
                    gt_rot_canon, gt_trans_canon, cont_mask_tensor, _ = (
                        self._canonicalize_continuous_abs_pose_with_anchor(
                            gt_rot,
                            gt_trans
                            if torch.is_tensor(gt_trans) and gt_trans.shape[0] == expected_pose_n
                            else None,
                            pred_rots_for_canon,
                            model_infos_flat,
                        )
                    )
                    if cont_mask_tensor is not None and cont_mask_tensor.any():
                        gt_rot_supervise[cont_mask_tensor] = gt_rot_canon[cont_mask_tensor]
                        if torch.is_tensor(gt_trans_supervise) and torch.is_tensor(gt_trans_canon):
                            gt_trans_supervise[cont_mask_tensor] = gt_trans_canon[cont_mask_tensor]

            return gt_rot_supervise, gt_trans_supervise
        finally:
            self._hotspot_profile_stop("_canonicalize_abs_pose_targets", profile_t0, pose_n=expected_pose_n)

    def _prepare_pm_supervision_targets(
        self,
        pred_rots,
        gt_rots,
        gt_transes=None,
        model_infos_batch=None,
        fallback_sym_infos=None,
        cont_sym_gt_mode="face_camera",
        degenerate_thresh=0.1,
    ):
        """Build PM supervision targets for symmetric objects.

        Discrete symmetry: hard-coded closest_pose — delegate to PM loss.
        Continuous symmetry: use cont_sym_gt_mode ("face_camera" or "closest_pose").
        """
        expected_pose_n = int(pred_rots.shape[0]) if torch.is_tensor(pred_rots) else -1
        if expected_pose_n <= 0 or (not torch.is_tensor(gt_rots)) or gt_rots.shape[0] != expected_pose_n:
            return gt_rots, gt_transes, fallback_sym_infos

        flat_model_infos = self._flatten_model_infos_for_pose_count(model_infos_batch, expected_pose_n)
        if flat_model_infos is None or len(flat_model_infos) != expected_pose_n:
            return gt_rots, gt_transes, fallback_sym_infos

        gt_rot_pm = gt_rots.clone()
        gt_trans_pm = (
            gt_transes.clone()
            if torch.is_tensor(gt_transes) and gt_transes.shape[0] == expected_pose_n
            else gt_transes
        )

        # --- Continuous symmetry ---
        if cont_sym_gt_mode == "face_camera":
            gt_rot_pm, gt_trans_pm = self._canonicalize_continuous_abs_pose_camera_facing(
                gt_rot_pm,
                gt_trans_pm,
                flat_model_infos,
                degenerate_thresh=degenerate_thresh,
            )
        else:
            # closest_pose: anchor on prediction
            gt_rot_pm, gt_trans_pm, _, _ = self._canonicalize_continuous_abs_pose_with_anchor(
                gt_rot_pm,
                gt_trans_pm if torch.is_tensor(gt_trans_pm) and gt_trans_pm.shape[0] == expected_pose_n else None,
                pred_rots.detach(),
                flat_model_infos,
            )

        # --- Discrete symmetry: delegate selection to the PM loss (closest_pose) ---
        pm_sym_infos = []
        for item in flat_model_infos:
            pm_sym_infos.append(self._safe_get_discrete_sym_rots(item, pred_rots.device, pred_rots.dtype))

        return gt_rot_pm, gt_trans_pm, pm_sym_infos

    def _build_discrete_abs_pose_candidates(self, gt_rot, gt_trans, model_info, device, dtype):
        sym_rot, sym_trans = self._safe_get_discrete_sym_transforms(model_info, device, dtype)
        if sym_rot is None:
            return None, None

        if gt_rot.dim() == 2:
            gt_rot = gt_rot.unsqueeze(0)
        cand_rot = torch.matmul(gt_rot.unsqueeze(1), sym_rot.unsqueeze(0)).squeeze(0)

        cand_trans = None
        if gt_trans is not None:
            if gt_trans.dim() == 1:
                gt_trans = gt_trans.unsqueeze(0)
            cand_trans = torch.matmul(gt_rot.unsqueeze(1), sym_trans.unsqueeze(-1)).squeeze(-1) + gt_trans
        return cand_rot, cand_trans

    @staticmethod
    def _rot_batch_angular_distance_deg(pred_rot, gt_rot):
        rel = torch.matmul(pred_rot, gt_rot.transpose(-1, -2))
        cos = ((torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.rad2deg(torch.acos(cos))

    @staticmethod
    def _wrap_angle_rad(angle_rad):
        return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))

    @staticmethod
    def _get_cont_sym_gt_mode(pnp_net_cfg):
        mode = pnp_net_cfg.get("CONT_SYM_GT_MODE", "face_camera")
        mode = str(mode).strip().lower()
        if mode not in {"face_camera", "closest_pose"}:
            raise ValueError(
                f"Unsupported CONT_SYM_GT_MODE={mode!r}. Expected 'face_camera' or 'closest_pose'."
            )
        return mode

    @staticmethod
    def _get_continuous_sym_ref_dir(model_info, axis, device, dtype):
        """Get reference direction b (in object frame, perp to sym axis a).

        Reads 'reference_dir' from symmetries_continuous config if present;
        otherwise auto-selects from object-frame basis vectors.
        """
        # Try explicit config
        sym_cfg = None
        if model_info is not None and hasattr(model_info, "get"):
            sym_cfg_list = model_info.get("symmetries_continuous", [])
            if sym_cfg_list:
                sym_cfg = sym_cfg_list[0] if isinstance(sym_cfg_list, (list, tuple)) else sym_cfg_list
        if sym_cfg is not None and hasattr(sym_cfg, "get"):
            ref_dir_raw = sym_cfg.get("reference_dir", None)
            if ref_dir_raw is not None:
                b = torch.tensor(ref_dir_raw, device=device, dtype=dtype)
                b = b - (b @ axis) * axis
                b_norm = b.norm()
                if b_norm > 1e-6:
                    return b / b_norm

        # Auto-select from object-frame basis
        candidates = [
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        ]
        best_b = None
        best_perp = -1.0
        for c in candidates:
            perp = (1.0 - abs(float((c @ axis).item())))
            if perp > best_perp:
                best_perp = perp
                best_b = c
        b = best_b - (best_b @ axis) * axis
        b_norm = b.norm()
        if b_norm < 1e-6:
            # fallback: return any vector perp to axis using cross product
            fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            if abs(float((fallback @ axis).item())) > 0.9:
                fallback = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            b = torch.cross(axis, fallback, dim=0)
            b_norm = b.norm()
        return b / b_norm.clamp(min=1e-8)

    def _canonicalize_continuous_abs_pose_camera_facing(
        self, gt_abs_rot, gt_abs_trans, model_infos_flat, degenerate_thresh=0.1
    ):
        """Camera-facing canonicalization for continuous-symmetry objects.

        For each continuous-sym object, selects the spin-equivalent pose where
        the reference direction b (perp to sym axis a) faces toward the camera.
        Non-continuous-sym objects are left unchanged.

        Args:
            gt_abs_rot:      (N, 3, 3) GT rotation matrices
            gt_abs_trans:    (N, 3) GT translations (camera frame)
            model_infos_flat: list of N model_info dicts
            degenerate_thresh: |u_perp| < thresh -> fallback (use raw pose)

        Returns:
            (gt_rot_canon, gt_trans_canon) — same shape, continuous-sym objects
            canonicalized; all others unchanged.
        """
        if (
            not torch.is_tensor(gt_abs_rot)
            or model_infos_flat is None
            or len(model_infos_flat) != gt_abs_rot.shape[0]
        ):
            return gt_abs_rot, gt_abs_trans

        gt_rot_out = gt_abs_rot.clone()
        gt_trans_out = (
            gt_abs_trans.clone()
            if torch.is_tensor(gt_abs_trans) and gt_abs_trans.shape[0] == gt_abs_rot.shape[0]
            else gt_abs_trans
        )

        for i, model_info in enumerate(model_infos_flat):
            if not self._has_continuous_symmetry(model_info):
                continue

            axis, offset = self._safe_get_continuous_sym_axis_offset(
                model_info, gt_abs_rot.device, gt_abs_rot.dtype
            )
            if axis is None:
                continue

            R = gt_abs_rot[i]   # (3, 3)
            t = gt_abs_trans[i] if torch.is_tensor(gt_abs_trans) else None  # (3,)

            if t is None:
                continue

            # Camera in object frame: c_obj = -R^T t
            c_obj = -R.T @ t                         # (3,)
            if offset is not None:
                u = c_obj - offset
            else:
                u = c_obj
            # Project onto plane perp to sym axis
            u_perp = u - (u @ axis) * axis           # (3,)
            u_perp_norm = u_perp.norm()

            if u_perp_norm < degenerate_thresh:
                # Degenerate: camera nearly on-axis, skip canonicalization
                continue

            d = u_perp / u_perp_norm                 # target direction

            b = self._get_continuous_sym_ref_dir(model_info, axis, gt_abs_rot.device, gt_abs_rot.dtype)

            # Signed angle from b to d around axis a
            cos_theta = (b @ d).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            cross_bd = torch.cross(b, d, dim=0)
            sin_theta = (cross_bd @ axis).clamp(-1.0, 1.0)
            theta = torch.atan2(sin_theta, cos_theta)

            # Rodrigues: S_cf = I + sin(theta) K + (1-cos(theta)) K^2
            K = torch.zeros(3, 3, device=gt_abs_rot.device, dtype=gt_abs_rot.dtype)
            K[0, 1] = -axis[2]; K[0, 2] = axis[1]
            K[1, 0] = axis[2];  K[1, 2] = -axis[0]
            K[2, 0] = -axis[1]; K[2, 1] = axis[0]

            S_cf = (
                torch.eye(3, device=gt_abs_rot.device, dtype=gt_abs_rot.dtype)
                + sin_theta * K
                + (1.0 - cos_theta) * (K @ K)
            )

            # R_cf = R @ S_cf
            gt_rot_out[i] = R @ S_cf

            # t_cf = t + R @ (offset - S_cf @ offset); offset=0 -> t_cf = t
            if offset is not None and torch.is_tensor(gt_trans_out):
                t_correction = R @ (offset - S_cf @ offset)
                gt_trans_out[i] = t + t_correction
            # else: offset=0, t unchanged

        return gt_rot_out, gt_trans_out

    @staticmethod
    def _align_xyz_targets_to_pose_supervision(
        gt_xyz,
        gt_xyz_bin,
        gt_mask_xyz,
        gt_rot_raw,
        gt_rot_supervise,
        model_infos_batch,
        extents,
        xyz_bin_num=None,
    ):
        """Align xyz supervision with the selected pose supervision branch.

        For continuous-symmetry objects, pose supervision may be canonicalized
        (e.g., face_camera) from R_raw to R_supervise = R_raw @ S.
        The dense xyz target must then be transformed by the inverse symmetry:
            X_supervise = S^T (X_raw - o) + o
        where o is the symmetry-axis offset in object coordinates.
        """
        if not (torch.is_tensor(gt_rot_raw) and torch.is_tensor(gt_rot_supervise)):
            return gt_xyz, gt_xyz_bin
        if not torch.is_tensor(gt_mask_xyz):
            return gt_xyz, gt_xyz_bin

        n = int(gt_rot_supervise.shape[0]) if gt_rot_supervise.dim() == 3 else -1
        if n <= 0 or gt_rot_raw.shape[0] != n:
            return gt_xyz, gt_xyz_bin

        model_infos_flat = GDRN._flatten_model_infos_for_pose_count(model_infos_batch, n)
        if model_infos_flat is None:
            return gt_xyz, gt_xyz_bin

        if not torch.is_tensor(extents):
            return gt_xyz, gt_xyz_bin
        extents_flat = extents
        if extents_flat.dim() == 3:
            extents_flat = extents_flat.flatten(0, 1).contiguous()
        if extents_flat.dim() != 2 or extents_flat.shape[0] != n or extents_flat.shape[1] != 3:
            return gt_xyz, gt_xyz_bin

        mask = gt_mask_xyz
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.dim() != 3 or mask.shape[0] != n:
            return gt_xyz, gt_xyz_bin

        device = gt_rot_supervise.device
        dtype = gt_rot_supervise.dtype
        extents_flat = extents_flat.to(device=device, dtype=dtype).clamp(min=1e-6)
        mask = mask.to(device=device, dtype=dtype)

        xyz_float = None
        if torch.is_tensor(gt_xyz):
            if gt_xyz.dim() == 4 and gt_xyz.shape[0] == n and gt_xyz.shape[1] == 3:
                xyz_float = gt_xyz.to(device=device, dtype=dtype).clone()
        if xyz_float is None and torch.is_tensor(gt_xyz_bin):
            if gt_xyz_bin.dim() == 4 and gt_xyz_bin.shape[0] == n and gt_xyz_bin.shape[1] == 3:
                if xyz_bin_num is None:
                    return gt_xyz, gt_xyz_bin
                xyz_bin_num = int(xyz_bin_num)
                if xyz_bin_num <= 0:
                    return gt_xyz, gt_xyz_bin
                xyz_float = (gt_xyz_bin.to(device=device, dtype=dtype) + 0.5) / float(xyz_bin_num)
        if xyz_float is None:
            return gt_xyz, gt_xyz_bin

        xyz_out = xyz_float.clone()
        for i in range(n):
            model_info_i = model_infos_flat[i]
            if not GDRN._has_continuous_symmetry(model_info_i):
                continue
            axis, offset = GDRN._safe_get_continuous_sym_axis_offset(model_info_i, device, dtype)
            if axis is None:
                continue

            valid = mask[i] > 0.5
            if not bool(valid.any().item()):
                continue

            r_raw_i = gt_rot_raw[i].to(device=device, dtype=dtype)
            r_sup_i = gt_rot_supervise[i].to(device=device, dtype=dtype)
            s_obj = r_raw_i.transpose(0, 1) @ r_sup_i  # R_sup = R_raw @ S
            s_inv = s_obj.transpose(0, 1)

            ext_i = extents_flat[i].view(3, 1, 1)
            xyz_metric_i = (xyz_out[i] - 0.5) * ext_i
            xyz_metric_hw3 = xyz_metric_i.permute(1, 2, 0).contiguous()
            pts = xyz_metric_hw3[valid]  # (K, 3)
            pts_new = torch.matmul(s_inv, (pts - offset).transpose(0, 1)).transpose(0, 1) + offset
            xyz_metric_hw3[valid] = pts_new
            xyz_out[i] = (xyz_metric_hw3.permute(2, 0, 1).contiguous() / ext_i) + 0.5

        gt_xyz_out = gt_xyz
        if torch.is_tensor(gt_xyz) and gt_xyz.dim() == 4 and gt_xyz.shape == xyz_out.shape:
            gt_xyz_out = xyz_out.to(device=gt_xyz.device, dtype=gt_xyz.dtype)

        gt_xyz_bin_out = gt_xyz_bin
        if torch.is_tensor(gt_xyz_bin):
            if xyz_bin_num is None:
                return gt_xyz_out, gt_xyz_bin_out
            xyz_bin_num = int(xyz_bin_num)
            if xyz_bin_num <= 0:
                return gt_xyz_out, gt_xyz_bin_out
            xyz_norm = xyz_out.clamp(0.0, 0.999999)
            xyz_bin_new = (xyz_norm * float(xyz_bin_num)).to(dtype=torch.long)
            bg = ~(mask > 0.5).unsqueeze(1).expand_as(xyz_bin_new)
            xyz_bin_new[bg] = xyz_bin_num
            gt_xyz_bin_out = xyz_bin_new.to(device=gt_xyz_bin.device, dtype=gt_xyz_bin.dtype)

        return gt_xyz_out, gt_xyz_bin_out

    @staticmethod
    def _recompute_region_targets_from_aligned_xyz(
        gt_xyz_supervise,
        gt_region,
        gt_mask_region,
        model_infos_batch,
        extents,
        num_regions,
    ):
        """Recompute region labels from pose-aligned xyz supervision.

        This keeps region supervision on the same canonicalized branch as xyz/pose.
        Falls back to the original gt_region when required metadata is unavailable.
        """
        num_regions = int(num_regions)
        if num_regions <= 1:
            return gt_region
        if not (torch.is_tensor(gt_xyz_supervise) and torch.is_tensor(gt_mask_region)):
            return gt_region
        if gt_xyz_supervise.dim() != 4 or gt_xyz_supervise.shape[1] != 3:
            return gt_region

        n, _, h, w = gt_xyz_supervise.shape
        model_infos_flat = GDRN._flatten_model_infos_for_pose_count(model_infos_batch, n)
        if model_infos_flat is None or len(model_infos_flat) != n:
            return gt_region
        if not torch.is_tensor(extents):
            return gt_region
        extents_flat = extents
        if extents_flat.dim() == 3:
            extents_flat = extents_flat.flatten(0, 1).contiguous()
        if extents_flat.dim() != 2 or extents_flat.shape[0] != n or extents_flat.shape[1] != 3:
            return gt_region

        mask = gt_mask_region
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.dim() != 3 or mask.shape[0] != n or mask.shape[1] != h or mask.shape[2] != w:
            return gt_region

        device = gt_xyz_supervise.device
        dtype = gt_xyz_supervise.dtype
        xyz_norm = gt_xyz_supervise.to(device=device, dtype=dtype)
        extents_flat = extents_flat.to(device=device, dtype=dtype).clamp(min=1e-6)
        mask = mask.to(device=device, dtype=dtype)

        if torch.is_tensor(gt_region) and gt_region.shape[0] == n and gt_region.shape[-2:] == (h, w):
            region_out = gt_region.clone()
        else:
            region_out = torch.zeros((n, h, w), device=device, dtype=torch.long)

        recompute_success = False
        for i in range(n):
            fps_points = GDRN._safe_get_region_fps_points(
                model_infos_flat[i], num_regions=num_regions, device=device, dtype=dtype
            )
            if fps_points is None:
                continue

            valid = mask[i] > 0.5
            if not bool(valid.any().item()):
                region_out[i] = torch.zeros((h, w), device=region_out.device, dtype=region_out.dtype)
                recompute_success = True
                continue

            ext_i = extents_flat[i].view(3, 1, 1)
            xyz_metric_hw3 = ((xyz_norm[i] - 0.5) * ext_i).permute(1, 2, 0).contiguous()
            pts_valid = xyz_metric_hw3[valid]  # (K, 3)
            dists = torch.cdist(pts_valid, fps_points)  # (K, num_regions)
            region_valid = torch.argmin(dists, dim=1).to(dtype=torch.long) + 1

            region_i = torch.zeros((h, w), device=device, dtype=torch.long)
            region_i[valid] = region_valid
            region_out[i] = region_i.to(device=region_out.device, dtype=region_out.dtype)
            recompute_success = True

        if not recompute_success:
            return gt_region
        return region_out

    @staticmethod
    def _skew_batch(v):
        K = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
        K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
        return K

    @staticmethod
    def _minimal_rotation_between_unit_vectors(src, dst, eps=1e-6):
        src = src / src.norm(dim=-1, keepdim=True).clamp(min=eps)
        dst = dst / dst.norm(dim=-1, keepdim=True).clamp(min=eps)
        batch = src.shape[0]
        eye = torch.eye(3, device=src.device, dtype=src.dtype).unsqueeze(0).repeat(batch, 1, 1)

        cross = torch.cross(src, dst, dim=-1)
        cross_norm = cross.norm(dim=-1, keepdim=True)
        dot = (src * dst).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        rot = eye.clone()
        general_mask = (cross_norm.squeeze(-1) > eps)
        if general_mask.any():
            axis = cross[general_mask] / cross_norm[general_mask].clamp(min=eps)
            theta = torch.atan2(cross_norm[general_mask], dot[general_mask])
            K = GDRN._skew_batch(axis)
            sin_t = torch.sin(theta).view(-1, 1, 1)
            cos_t = torch.cos(theta).view(-1, 1, 1)
            eye_g = eye[general_mask]
            rot[general_mask] = eye_g + sin_t * K + (1.0 - cos_t) * torch.matmul(K, K)

        opposite_mask = (~general_mask) & (dot.squeeze(-1) < 0.0)
        if opposite_mask.any():
            src_opp = src[opposite_mask]
            helper_idx = src_opp.abs().argmin(dim=-1)
            helper = F.one_hot(helper_idx, num_classes=3).to(dtype=src.dtype, device=src.device)
            axis_opp = torch.cross(src_opp, helper, dim=-1)
            axis_opp = axis_opp / axis_opp.norm(dim=-1, keepdim=True).clamp(min=eps)
            K = GDRN._skew_batch(axis_opp)
            eye_o = eye[opposite_mask]
            rot[opposite_mask] = eye_o + 2.0 * torch.matmul(K, K)

        return rot

    @staticmethod
    def _vector_angle_deg(vec_a, vec_b, eps=1e-8):
        vec_a = vec_a / vec_a.norm(dim=-1, keepdim=True).clamp(min=eps)
        vec_b = vec_b / vec_b.norm(dim=-1, keepdim=True).clamp(min=eps)
        cos = (vec_a * vec_b).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.rad2deg(torch.acos(cos))

    def forward(
        self,
        x,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_region=None,
        gt_allo_quat=None,
        gt_ego_quat=None,
        gt_allo_rot6d=None,
        gt_ego_rot6d=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        model_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None,
        roi_centers=None,
        scales=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        input_images=None,
        input_depths=None,
        input_obj_masks=None,
        external_target_masks=None,
        noisy_obj_masks=None,
        target_idx=None, # here idx can be a list of indices
        do_loss=False,
        model_points=None,
    ):
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        profile_forward_t0 = self._hotspot_profile_begin_forward() if do_loss else None

        B, N, C, H, W = input_images.shape
        h_p, w_p = H // 16, W // 16
        target_view_depth = None
        target_view_mask = None

        # print("gt_xyz: ", gt_xyz.shape if gt_xyz is not None else None)
        # print("gt_xyz_bin: ", gt_xyz_bin.shape if gt_xyz_bin is not None else None)
        # print("gt_mask_trunc: ", gt_mask_trunc.shape if gt_mask_trunc is not None else None)
        # print("gt_mask_visib: ", gt_mask_visib.shape if gt_mask_visib is not None else None)
        # print("gt_mask_obj: ", gt_mask_obj.shape if gt_mask_obj is not None else None)
        # print("gt_region: ", gt_region.shape if gt_region is not None else None)
        # print("gt_allo_quat: ", gt_allo_quat.shape if gt_allo_quat is not None else None)
        # print("gt_ego_quat: ", gt_ego_quat.shape if gt_ego_quat is not None else None)
        # print("gt_allo_rot6d: ", gt_allo_rot6d.shape if gt_allo_rot6d is not None else None)
        # print("gt_ego_rot6d: ", gt_ego_rot6d.shape if gt_ego_rot6d is not None else None)
        # print("gt_ego_rot: ", gt_ego_rot.shape if gt_ego_rot is not None else None)
        # print("gt_points: ", gt_points.shape if gt_points is not None else None)
        # # print("sym_infos: ", sym_infos)
        # print("gt_trans: ", gt_trans.shape if gt_trans is not None else None)
        # print("gt_trans_ratio: ", gt_trans_ratio.shape if gt_trans_ratio is not None else None)
        # print("roi_classes: ", roi_classes.shape if roi_classes is not None else None)
        # print("roi_coord_2d: ", roi_coord_2d.shape if roi_coord_2d is not None else None)
        # print("roi_cams: ", roi_cams.shape if roi_cams is not None else None)
        # print("roi_centers: ", roi_centers.shape if roi_centers is not None else None)
        # print("roi_whs: ", roi_whs.shape if roi_whs is not None else None)
        # print("roi_extents: ", roi_extents.shape if roi_extents is not None else None)
        # print("resize_ratios: ", resize_ratios.shape if resize_ratios is not None else None)
        # print("input_images: ", input_images.shape if input_images is not None else None)
        # print("roi_patch_masks: ", roi_patch_masks.shape if roi_patch_masks is not None else None)
        full_gt_ego_rot = gt_ego_rot
        full_gt_trans = gt_trans
        full_roi_coord_2d = roi_coord_2d
        full_roi_cams = roi_cams
        full_roi_centers = roi_centers
        full_scales = scales
        full_roi_whs = roi_whs
        full_roi_extents = roi_extents
        full_resize_ratios = resize_ratios
        full_input_images = input_images
        full_input_depths = input_depths      # (B, N, H, W) before target slicing
        full_input_obj_masks = input_obj_masks  # (B, N, H, W) before target slicing

        target_idx_list, has_context_for_target, is_all_target = analyze_target_indices(target_idx, N)

        gt_xyz = gt_xyz[:, target_idx] if gt_xyz is not None else None
        gt_xyz_bin = gt_xyz_bin[:, target_idx] if gt_xyz_bin is not None else None
        gt_mask_trunc = gt_mask_trunc[:, target_idx] if gt_mask_trunc is not None else None
        gt_mask_visib = gt_mask_visib[:, target_idx] if gt_mask_visib is not None else None
        gt_mask_obj = gt_mask_obj[:, target_idx] if gt_mask_obj is not None else None
        gt_region = gt_region[:, target_idx] if gt_region is not None else None
        gt_allo_quat = gt_allo_quat[:, target_idx] if gt_allo_quat is not None else None
        gt_ego_quat = gt_ego_quat[:, target_idx] if gt_ego_quat is not None else None
        gt_allo_rot6d = gt_allo_rot6d[:, target_idx] if gt_allo_rot6d is not None else None
        gt_ego_rot6d = gt_ego_rot6d[:, target_idx] if gt_ego_rot6d is not None else None
        gt_ego_rot = gt_ego_rot[:, target_idx] if gt_ego_rot is not None else None
        gt_points = gt_points[:, target_idx] if gt_points is not None else None
        if sym_infos is not None:
            if isinstance(target_idx, (list, tuple)):
                sym_infos = [[sym_info[i] for i in target_idx] for sym_info in sym_infos]
            elif isinstance(target_idx, np.ndarray) and target_idx.ndim > 0:
                idx_list = target_idx.tolist()
                sym_infos = [[sym_info[i] for i in idx_list] for sym_info in sym_infos]
            elif torch.is_tensor(target_idx) and target_idx.dim() > 0 and target_idx.numel() > 1:
                idx_list = target_idx.tolist()
                sym_infos = [[sym_info[i] for i in idx_list] for sym_info in sym_infos]
            else:
                sym_infos = [sym_info[target_idx] for sym_info in sym_infos]
        else:
            sym_infos = None
        gt_trans = gt_trans[:, target_idx] if gt_trans is not None else None
        gt_trans_ratio = gt_trans_ratio[:, target_idx] if gt_trans_ratio is not None else None
        roi_coord_2d = roi_coord_2d[:, target_idx] if roi_coord_2d is not None else None
        roi_cams = roi_cams[:, target_idx] if roi_cams is not None else None
        roi_centers = roi_centers[:, target_idx] if roi_centers is not None else None
        scales = scales[:, target_idx] if scales is not None else None
        roi_whs = roi_whs[:, target_idx] if roi_whs is not None else None
        roi_extents = roi_extents[:, target_idx] if roi_extents is not None else None
        resize_ratios = resize_ratios[:, target_idx] if resize_ratios is not None else None
        input_depths = input_depths[:, target_idx] if input_depths is not None else None
        input_obj_masks = input_obj_masks[:, target_idx] if input_obj_masks is not None else None
        external_target_masks = external_target_masks[:, target_idx] if external_target_masks is not None else None
        noisy_obj_masks = noisy_obj_masks[:, target_idx] if noisy_obj_masks is not None else None
        if input_depths is not None and input_depths.dim() == 3:
            input_depths = input_depths.unsqueeze(1)
        if input_obj_masks is not None and input_obj_masks.dim() == 3:
            input_obj_masks = input_obj_masks.unsqueeze(1)
        if external_target_masks is not None and external_target_masks.dim() == 4:
            external_target_masks = external_target_masks.unsqueeze(2)
        if noisy_obj_masks is not None and noisy_obj_masks.dim() == 4:
            noisy_obj_masks = noisy_obj_masks.unsqueeze(2)
        is_multiview_target = (
            isinstance(target_idx, (list, tuple))
            or (isinstance(target_idx, np.ndarray) and target_idx.ndim > 0 and target_idx.size > 1)
            or (torch.is_tensor(target_idx) and target_idx.numel() > 1)
        )
        if isinstance(target_idx, (list, tuple)):
            target_num = len(target_idx)
        elif isinstance(target_idx, np.ndarray):
            target_num = int(target_idx.size) if target_idx.ndim > 0 else 1
        elif torch.is_tensor(target_idx):
            target_num = int(target_idx.numel()) if target_idx.numel() > 1 else 1
        else:
            target_num = 1

        context_last_idx = max(min(target_idx_list) - 1, 0)

        if cfg.MODEL.CDPN.VGGT_BACKBONE:
            # encode by dinov3, here we only encode the target view for single view implementation test
            input_images = input_images.reshape(B*N, C, H, W)

            profile_dino_encode_t0 = self._hotspot_profile_start()
            hidden = self._run_image_encoder(input_images)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            self._hotspot_profile_stop("dino_encode", profile_dino_encode_t0, batch_size=B, view_num=N)

            # # FOR DEBUGGING VISUALIZATION
            # vis_img = input_images.reshape(B, N, C, H, W)
            # vis_img = vis_img[0, -1]
            # mean = torch.tensor([0.485, 0.456, 0.406], dtype=vis_img.dtype).to(vis_img.device)
            # std = torch.tensor([0.229, 0.224, 0.225], dtype=vis_img.dtype).to(vis_img.device)
            # mean = mean.unsqueeze(-1).unsqueeze(-1)
            # std = std.unsqueeze(-1).unsqueeze(-1)
            # vis_img = (vis_img * std + mean) * 255.0
            # vis_img = vis_img.permute(1, 2, 0).cpu().numpy()
            # vis_img = vis_img.astype(np.uint8)
            # from PIL import Image
            # vis_target_roi_patch_masks = target_roi_patch_masks.repeat_interleave(
            #        self.patch_size, dim=-1
            #     ).repeat_interleave(
            #        self.patch_size, dim=-2
            #     )
            # vis_target_roi_patch_masks = vis_target_roi_patch_masks[0].cpu().numpy()
            # vis_img[vis_target_roi_patch_masks] = vis_img[vis_target_roi_patch_masks] + 20.
            # Image.fromarray(vis_img).save(f"/mnt/afs/TransparentObjectPose/debug/vis_img.png")
            # import pdb; pdb.set_trace()
            
            # attention layers
            profile_decode_t0 = self._hotspot_profile_start()
            hidden, pos = self.decode(hidden, N, H, W) # (B*N, hw, C)
            self._hotspot_profile_stop("decode_tokens", profile_decode_t0, batch_size=B, view_num=N)
            hidden_by_view = hidden.reshape(B, N, h_p * w_p, -1).contiguous()
            target_hidden = hidden_by_view[:, target_idx].contiguous()

            # depth head to output the depth and mask
            profile_depth_head_t0 = self._hotspot_profile_start()
            target_hidden_2d = target_hidden.reshape(B * target_num, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
            target_depth_2d, target_mask_2d = self.depth_head(target_hidden_2d, out_size=(H, W))
            target_view_depth = target_depth_2d.reshape(B, target_num, 1, H, W).contiguous()
            target_view_mask = target_mask_2d.reshape(B, target_num, 1, H, W).contiguous()
            self._hotspot_profile_stop("depth_head_forward", profile_depth_head_t0, batch_size=B, target_num=target_num)

            # RoI processing + BN for target views.
            # In VI context-then-target training path, defer target-only ROI encoding
            # so target/context can be jointly encoded in one pass.
            defer_target_roi_encode_for_vi = (
                self.training
                and bool(do_loss)
                and self.view_interaction_enabled
                and has_context_for_target
                and (not is_all_target)
            )
            if defer_target_roi_encode_for_vi:
                features = None
            else:
                profile_roi_target_t0 = self._hotspot_profile_start()
                features = self._encode_view_tokens_to_roi_features(target_hidden, roi_centers, scales, H, W)
                self._hotspot_profile_stop("roi_encode_target", profile_roi_target_t0, batch_size=B, target_num=target_num)

            
        # x.shape [bs, 3, 256, 256]
        if self.concat:
            # features, x_f64, x_f32, x_f16 = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # # joints.shape [bs, 1152, 64, 64]
            # mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features, x_f64, x_f32, x_f16)
            raise NotImplementedError("Concat is not supported yet in DINO-GDRN!")
        else:
            # features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            x_target = x[:, target_idx]
            if is_multiview_target and x_target.dim() == 5:
                x_target = x_target.reshape(B * target_num, x_target.shape[2], x_target.shape[3], x_target.shape[4])
            if cfg.MODEL.CDPN.VGGT_BACKBONE and cfg.MODEL.CDPN.RESNET_BACKBONE:
                roi_features = self.backbone(x_target)
                features = features + roi_features
            elif cfg.MODEL.CDPN.RESNET_BACKBONE and not cfg.MODEL.CDPN.VGGT_BACKBONE:
                features = self.backbone(x_target)
            elif cfg.MODEL.CDPN.VGGT_BACKBONE and not cfg.MODEL.CDPN.RESNET_BACKBONE:
                pass # features = features, is already in the hidden
            else:
                raise NotImplementedError("Backbone is not found!")

            # ================================================================
            # View Interaction: inject context-view info into target features
            # (must run BEFORE rot_head_net so features are enhanced first)
            # ================================================================
            vi_cfg = cfg.MODEL.CDPN.get("VIEW_INTERACTION", {})
            gp_cfg = vi_cfg.get("GEOMETRIC_PRIOR", {})
            geo_prior_out = None
            geo_prior_out_flat = None
            geo_prior_active = False
            prior_was_dropped = False
            run_view_interaction = (
                self.view_interaction_enabled
                and (has_context_for_target or (is_all_target and N >= 2))
                and cfg.MODEL.CDPN.VGGT_BACKBONE
            )
            if run_view_interaction:
                profile_view_interaction_t0 = self._hotspot_profile_start()
                has_context_pose = full_gt_ego_rot is not None and full_gt_trans is not None
                has_context_geom = full_input_depths is not None and full_input_obj_masks is not None
                # Training VI relies on GT context poses to build the xyz cue.
                # In generic eval/test callers (e.g. gdrn_evaluator / single-image
                # scripts), those poses are often unavailable. In that case, fall
                # back to the baseline path instead of crashing the whole eval.
                if not (has_context_pose and has_context_geom):
                    if do_loss:
                        raise AssertionError(
                            "View Interaction requires context poses and geometry inputs "
                            "(gt_ego_rot / gt_trans / input_depths / input_obj_masks) "
                            "during training, but some are None."
                        )
                    run_view_interaction = False

                if run_view_interaction and is_all_target:
                    # Leave-one-view-out: for each target view i, context = all other views.
                    # Run VI once per target and collect enhanced features.
                    # Important: do NOT recompute context roi_feat via
                    # _encode_view_tokens_to_roi_features() inside the loop.
                    # In all-target mode, `features` already contains the ROI features
                    # for all views; recomputing the heavy roi_decoder -> roi_head path
                    # per target causes a massive O(T * (T-1)) memory blow-up.
                    features_list = []
                    features_by_view = features.reshape(B, target_num, *features.shape[1:]).contiguous()
                    target_local_pos = {int(view_id): local_i for local_i, view_id in enumerate(target_idx_list)}
                    for i, tgt_i in enumerate(target_idx_list):
                        profile_vi_alltarget_geom_t0 = self._hotspot_profile_start()
                        ctx_idx_i = [j for j in range(N) if j != tgt_i]
                        ctx_num_i = len(ctx_idx_i)
                        tgt_local_i = target_local_pos[int(tgt_i)]
                        ctx_local_idx_i = [target_local_pos[int(j)] for j in ctx_idx_i]

                        # Single target's feature slice: (B, 512, 8, 8)
                        feat_i = features_by_view[:, tgt_local_i].contiguous()  # (B, 512, 8, 8)

                        # Context roi_feat: reuse already-computed all-view roi features.
                        ctx_roi_feat_i = features_by_view[:, ctx_local_idx_i].reshape(
                            B * ctx_num_i, *features.shape[1:]
                        ).contiguous()  # (B*C, 512, 8, 8)
                        ctx_roi_centers_i = full_roi_centers[:, ctx_idx_i]
                        ctx_scales_i = full_scales[:, ctx_idx_i]

                        # Context pose (GT, face_camera canonicalized)
                        ctx_R_i = full_gt_ego_rot[:, ctx_idx_i].reshape(B * ctx_num_i, 3, 3)
                        ctx_t_i = full_gt_trans[:, ctx_idx_i].reshape(B * ctx_num_i, 3)
                        if model_infos is not None:
                            ctx_model_infos_i = self._flatten_model_infos_for_pose_count(
                                model_infos, B * ctx_num_i
                            )
                            if ctx_model_infos_i is not None:
                                ctx_R_i, ctx_t_i = self._canonicalize_continuous_abs_pose_camera_facing(
                                    ctx_R_i, ctx_t_i, ctx_model_infos_i
                                )
                        if do_loss and self.training:
                            ctx_R_i, ctx_t_i = self._add_pose_noise_for_ctx_info(ctx_R_i, ctx_t_i, vi_cfg)
                        ctx_K_i = full_roi_cams[:, ctx_idx_i].reshape(B * ctx_num_i, 3, 3)

                        assert full_input_depths is not None, (
                            "View Interaction (all-target) requires input_depths but got None."
                        )
                        assert full_input_obj_masks is not None, (
                            "View Interaction (all-target) requires input_obj_masks but got None."
                        )
                        ctx_depth_i = full_input_depths[:, ctx_idx_i].reshape(
                            B * ctx_num_i, 1, H, W
                        ).to(device=ctx_R_i.device, dtype=ctx_R_i.dtype)
                        ctx_mask_i = full_input_obj_masks[:, ctx_idx_i].reshape(
                            B * ctx_num_i, 1, H, W
                        ).to(device=ctx_R_i.device, dtype=ctx_R_i.dtype).clamp(0.0, 1.0)

                        ctx_xyz_full_i = _backproject_depth_to_xyz(
                            ctx_depth_i, ctx_mask_i, ctx_R_i, ctx_t_i, ctx_K_i
                        )
                        ctx_xyz_roi_i = crop_resize_features(
                            ctx_xyz_full_i,
                            ctx_roi_centers_i.reshape(B * ctx_num_i, 2),
                            ctx_scales_i.reshape(B * ctx_num_i, 1),
                            output_size=8,
                        )
                        if roi_extents is not None:
                            ctx_ext_i = full_roi_extents[:, ctx_idx_i].reshape(B * ctx_num_i, 3)
                            ctx_xyz_roi_i = (
                                ctx_xyz_roi_i / ctx_ext_i.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6) + 0.5
                            )
                        self._hotspot_profile_stop(
                            "vi_alltarget_geom", profile_vi_alltarget_geom_t0, batch_size=B, target_num=target_num
                        )

                        # Zero-xyz-cue dropout: simulate inference first-chunk (no cached poses).
                        # Applied only during training to expose the model to the zero-xyz condition.
                        _xyz_drop_prob = float(vi_cfg.get("XYZ_CUE_DROPOUT_PROB", 0.0))
                        if self.training and _xyz_drop_prob > 0 and torch.rand(1).item() < _xyz_drop_prob:
                            ctx_xyz_roi_i = torch.zeros_like(ctx_xyz_roi_i)

                        profile_vi_alltarget_prep_t0 = self._hotspot_profile_start()
                        # RGB crops
                        C_img = full_input_images.shape[2]
                        tgt_img_i = crop_resize_features(
                            full_input_images[:, tgt_i],  # (B, C, H, W)
                            full_roi_centers[:, tgt_i],
                            full_scales[:, tgt_i],
                            output_size=256,
                        )  # (B, 3, 256, 256)
                        ctx_imgs_i = crop_resize_features(
                            full_input_images[:, ctx_idx_i].reshape(B * ctx_num_i, C_img, H, W),
                            ctx_roi_centers_i.reshape(B * ctx_num_i, 2),
                            ctx_scales_i.reshape(B * ctx_num_i, 1),
                            output_size=256,
                        )

                        # coord_2d and mask (target)
                        if roi_coord_2d.dim() == 5:
                            tgt_coord_i = F.adaptive_avg_pool2d(roi_coord_2d[:, i].contiguous(), 8)
                        else:
                            # flat (B*T, 2, H, W) — pick rows belonging to this target
                            tgt_coord_i = F.adaptive_avg_pool2d(
                                roi_coord_2d.reshape(B, target_num, 2, roi_coord_2d.shape[-2], roi_coord_2d.shape[-1])[:, i], 8
                            )

                        tgt_mask_i = F.adaptive_avg_pool2d(
                            torch.sigmoid(target_view_mask[:, i, :1]), 8
                        )  # (B, 1, 8, 8)

                        # context coord_2d
                        _dev = features.device
                        xs_vi = torch.linspace(0, 1, W, device=_dev, dtype=features.dtype)
                        ys_vi = torch.linspace(0, 1, H, device=_dev, dtype=features.dtype)
                        yy_vi, xx_vi = torch.meshgrid(ys_vi, xs_vi, indexing="ij")
                        fc_vi = torch.stack([xx_vi, yy_vi], dim=0).unsqueeze(0).expand(B * ctx_num_i, -1, -1, -1)
                        ctx_coord_i = crop_resize_features(
                            fc_vi,
                            ctx_roi_centers_i.reshape(B * ctx_num_i, 2),
                            ctx_scales_i.reshape(B * ctx_num_i, 1),
                            output_size=8,
                        )
                        ctx_mask_vi_i = F.adaptive_avg_pool2d(ctx_mask_i, 8)
                        self._hotspot_profile_stop(
                            "vi_alltarget_token_prep", profile_vi_alltarget_prep_t0, batch_size=B, target_num=target_num
                        )

                        profile_vi_alltarget_module_t0 = self._hotspot_profile_start()
                        feat_i_enhanced = self.view_interaction(
                            target_roi_feat=feat_i,
                            context_roi_feat=ctx_roi_feat_i,
                            context_xyz_cue=ctx_xyz_roi_i,
                            target_roi_rgb=tgt_img_i,
                            context_roi_rgb=ctx_imgs_i,
                            target_roi_coord2d=tgt_coord_i,
                            context_roi_coord2d=ctx_coord_i,
                            target_roi_mask=tgt_mask_i,
                            context_roi_mask=ctx_mask_vi_i,
                            batch_size=B,
                            target_num=1,
                            context_num=ctx_num_i,
                            target_time_ids=[tgt_i],
                            context_time_ids=ctx_idx_i,
                        )  # (B, 512, 8, 8)
                        self._hotspot_profile_stop(
                            "vi_alltarget_module", profile_vi_alltarget_module_t0, batch_size=B, target_num=target_num
                        )
                        features_list.append(feat_i_enhanced)

                    # Reassemble: stack as (B, T, 512, 8, 8) → (B*T, 512, 8, 8)
                    features = torch.stack(features_list, dim=1).flatten(0, 1).contiguous()

                elif run_view_interaction:
                    # context-then-target: fixed context set, all targets share the same context
                    context_idx_list = [i for i in range(N) if i not in target_idx_list]
                    context_num = len(context_idx_list)

                    if context_num > 0:
                        context_hidden = hidden_by_view[:, context_idx_list].contiguous()
                        context_roi_centers = full_roi_centers[:, context_idx_list]
                        context_scales = full_scales[:, context_idx_list]
                        if defer_target_roi_encode_for_vi:
                            # Joint ROI encode: one heavy pass for target + context, then split.
                            profile_roi_joint_t0 = self._hotspot_profile_start()
                            features, context_roi_feat = self._encode_target_and_context_roi_features(
                                target_hidden=target_hidden,
                                target_roi_centers=roi_centers,
                                target_scales=scales,
                                context_hidden=context_hidden,
                                context_roi_centers=context_roi_centers,
                                context_scales=context_scales,
                                H=H,
                                W=W,
                            )
                            self._hotspot_profile_stop(
                                "roi_encode_target_context_joint",
                                profile_roi_joint_t0,
                                batch_size=B,
                                target_num=target_num,
                            )
                        else:
                            # --- 1. Extract context roi_feat ---
                            profile_roi_context_t0 = self._hotspot_profile_start()
                            context_roi_feat = self._encode_view_tokens_to_roi_features(
                                context_hidden, context_roi_centers, context_scales, H, W
                            )  # (B*C, 512, 8, 8)
                            self._hotspot_profile_stop(
                                "roi_encode_context", profile_roi_context_t0, batch_size=B, target_num=target_num
                            )

                    profile_vi_ctx_geom_t0 = self._hotspot_profile_start()
                    # --- 2. Build context xyz cue from GT depth + GT mask + context pose ---
                    # Use GT depth and GT mask (not depth_head predictions) so that:
                    # (a) the xyz cue is well-defined from the first training step, and
                    # (b) context depth is not required to be supervised separately.
                    # At inference the cached predicted pose takes the place of GT pose.
                    ctx_R = full_gt_ego_rot[:, context_idx_list].reshape(B * context_num, 3, 3)
                    ctx_t = full_gt_trans[:, context_idx_list].reshape(B * context_num, 3)
                    # Canonicalize context poses to face each context view's own camera
                    # (face_camera for continuous symmetry). This matches inference, where
                    # context poses are AR predictions that are already canonicalized.
                    if model_infos is not None:
                        ctx_model_infos = self._flatten_model_infos_for_pose_count(
                            model_infos, B * context_num
                        )
                        if ctx_model_infos is not None:
                            ctx_R, ctx_t = self._canonicalize_continuous_abs_pose_camera_facing(
                                ctx_R, ctx_t, ctx_model_infos
                            )
                    if do_loss and self.training:
                        ctx_R, ctx_t = self._add_pose_noise_for_ctx_info(ctx_R, ctx_t, vi_cfg)
                    ctx_K = full_roi_cams[:, context_idx_list].reshape(B * context_num, 3, 3)

                    # GT depth: (B, N, H, W) → select context views → (B*C, 1, H, W)
                    # Fail fast: silently falling back to zeros/ones would feed semantically
                    # corrupt geometry memory to VI without any visible error signal.
                    assert full_input_depths is not None, (
                        "View Interaction requires input_depths (GT depth maps) but got None. "
                        "Ensure the dataloader provides depth for all views when VI is enabled."
                    )
                    assert full_input_obj_masks is not None, (
                        "View Interaction requires input_obj_masks (GT visible masks) but got None. "
                        "Ensure the dataloader provides mask_visib for all views when VI is enabled."
                    )
                    ctx_depth_gt = full_input_depths[:, context_idx_list].reshape(B * context_num, 1, H, W).to(
                        device=ctx_R.device, dtype=ctx_R.dtype
                    )
                    ctx_mask_gt = full_input_obj_masks[:, context_idx_list].reshape(B * context_num, 1, H, W).to(
                        device=ctx_R.device, dtype=ctx_R.dtype
                    ).clamp(0.0, 1.0)

                    ctx_xyz_full = _backproject_depth_to_xyz(
                        ctx_depth_gt, ctx_mask_gt, ctx_R, ctx_t, ctx_K
                    )  # (B*C, 3, H, W)

                    ctx_xyz_roi = crop_resize_features(
                        ctx_xyz_full,
                        context_roi_centers.reshape(B * context_num, 2),
                        context_scales.reshape(B * context_num, 1),
                        output_size=8,
                    )  # (B*C, 3, 8, 8)

                    if roi_extents is not None:
                        ctx_extents = full_roi_extents[:, context_idx_list].reshape(B * context_num, 3)
                        ext = ctx_extents.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
                        ctx_xyz_roi = ctx_xyz_roi / ext + 0.5  # normalize to ~[0,1], matching roi_xyz supervision
                    self._hotspot_profile_stop(
                        "vi_ctx_geom", profile_vi_ctx_geom_t0, batch_size=B, target_num=target_num
                    )

                    profile_vi_ctx_token_prep_t0 = self._hotspot_profile_start()
                    # --- 4. Prepare token inputs ---
                    # RGB: crop from full_input_images (ImageNet-normalized), matching inference
                    C_img = full_input_images.shape[2]
                    target_imgs_vi = crop_resize_features(
                        full_input_images[:, target_idx_list].reshape(B * target_num, C_img, H, W),
                        roi_centers.reshape(B * target_num, 2),
                        scales.reshape(B * target_num, 1),
                        output_size=256,
                    )
                    context_imgs_vi = crop_resize_features(
                        full_input_images[:, context_idx_list].reshape(B * context_num, C_img, H, W),
                        context_roi_centers.reshape(B * context_num, 2),
                        context_scales.reshape(B * context_num, 1),
                        output_size=256,
                    )

                    # target coord_2d at 8x8
                    _roi_coord_flat = roi_coord_2d.flatten(0, 1) if roi_coord_2d.dim() == 5 else roi_coord_2d
                    tgt_coord_vi = F.adaptive_avg_pool2d(_roi_coord_flat, 8)  # (B*T, 2, 8, 8)

                    # target mask at 8x8: from depth_head (same source as inference)
                    tgt_mask_vi = F.adaptive_avg_pool2d(
                        torch.sigmoid(target_view_mask.flatten(0, 1)[:, :1]), 8
                    )  # (B*T, 1, 8, 8)

                    # context coord_2d at 8x8
                    _dev = features.device
                    xs_vi = torch.linspace(0, 1, W, device=_dev, dtype=features.dtype)
                    ys_vi = torch.linspace(0, 1, H, device=_dev, dtype=features.dtype)
                    yy_vi, xx_vi = torch.meshgrid(ys_vi, xs_vi, indexing="ij")
                    full_coord_vi = torch.stack([xx_vi, yy_vi], dim=0).unsqueeze(0).expand(B * context_num, -1, -1, -1)
                    ctx_coord_vi = crop_resize_features(
                        full_coord_vi,
                        context_roi_centers.reshape(B * context_num, 2),
                        context_scales.reshape(B * context_num, 1),
                        output_size=8,
                    )  # (B*C, 2, 8, 8)

                    # context mask at 8x8: use GT mask (already clamped to [0,1], no sigmoid needed)
                    ctx_mask_vi = F.adaptive_avg_pool2d(ctx_mask_gt, 8)
                    ctx_view_dir_obj, ctx_cam_center_obj = build_context_view_geom_features(ctx_R, ctx_t, out_hw=8)
                    self._hotspot_profile_stop(
                        "vi_ctx_token_prep", profile_vi_ctx_token_prep_t0, batch_size=B, target_num=target_num
                    )

                    # --- 5. Run view interaction (enhances features in-place) ---
                    profile_vi_ctx_module_t0 = self._hotspot_profile_start()
                    features = self.view_interaction(
                        target_roi_feat=features,
                        context_roi_feat=context_roi_feat,
                        context_xyz_cue=ctx_xyz_roi,
                        target_roi_rgb=target_imgs_vi,
                        context_roi_rgb=context_imgs_vi,
                        target_roi_coord2d=tgt_coord_vi,
                        context_roi_coord2d=ctx_coord_vi,
                        target_roi_mask=tgt_mask_vi,
                        context_roi_mask=ctx_mask_vi,
                        batch_size=B,
                        target_num=target_num,
                        context_num=context_num,
                        target_time_ids=target_idx_list,
                        context_time_ids=context_idx_list,
                    )  # (B*T, 512, 8, 8)
                    self._hotspot_profile_stop(
                        "vi_ctx_module", profile_vi_ctx_module_t0, batch_size=B, target_num=target_num
                    )

                    geo_prior_active = self.geo_prior_enabled and (not is_all_target) and (context_num > 0)
                    if geo_prior_active:
                        profile_geo_prior_forward_t0 = self._hotspot_profile_start()
                        target_roi_feat_by_view = features.reshape(B, target_num, *features.shape[1:])
                        context_roi_feat_by_view = context_roi_feat.reshape(B, context_num, *context_roi_feat.shape[1:])
                        ctx_xyz_roi_by_view = ctx_xyz_roi.reshape(B, context_num, *ctx_xyz_roi.shape[1:])
                        ctx_view_dir_obj_by_view = ctx_view_dir_obj.reshape(B, context_num, *ctx_view_dir_obj.shape[1:])
                        ctx_cam_center_obj_by_view = ctx_cam_center_obj.reshape(B, context_num, *ctx_cam_center_obj.shape[1:])
                        context_imgs_vi_by_view = context_imgs_vi.reshape(B, context_num, *context_imgs_vi.shape[1:])
                        ctx_coord_vi_by_view = ctx_coord_vi.reshape(B, context_num, *ctx_coord_vi.shape[1:])
                        ctx_mask_vi_by_view = ctx_mask_vi.reshape(B, context_num, *ctx_mask_vi.shape[1:])
                        target_imgs_vi_by_view = target_imgs_vi.reshape(B, target_num, *target_imgs_vi.shape[1:])
                        tgt_coord_vi_by_view = tgt_coord_vi.reshape(B, target_num, *tgt_coord_vi.shape[1:])
                        tgt_mask_vi_by_view = tgt_mask_vi.reshape(B, target_num, *tgt_mask_vi.shape[1:])

                        geo_inputs = ContextGeometricPriorInputs(
                            target_roi_feat=target_roi_feat_by_view,
                            context_roi_feat=context_roi_feat_by_view,
                            context_xyz=ctx_xyz_roi_by_view,
                            context_view_dir_obj=ctx_view_dir_obj_by_view,
                            context_cam_center_obj=ctx_cam_center_obj_by_view,
                            target_rgb=target_imgs_vi_by_view,
                            context_rgb=context_imgs_vi_by_view,
                            target_coord=tgt_coord_vi_by_view,
                            context_coord=ctx_coord_vi_by_view,
                            target_mask=tgt_mask_vi_by_view,
                            context_mask=ctx_mask_vi_by_view,
                            target_valid=torch.ones(B, target_num, dtype=torch.bool, device=features.device),
                            context_valid=torch.ones(B, context_num, dtype=torch.bool, device=features.device),
                        )
                        geo_prior_out = self.context_geo_prior(geo_inputs)
                        self._hotspot_profile_stop(
                            "geo_prior_forward", profile_geo_prior_forward_t0, batch_size=B, target_num=target_num
                        )

                        if self.training:
                            prior_dropout_prob = float(gp_cfg.get("PRIOR_DROPOUT_PROB", 0.15))
                            # DDP safety: synchronize the dropout decision across all ranks.
                            # torch.rand() is independent per rank; broadcasting from rank 0
                            # ensures every rank makes the same drop/keep choice.
                            _drop_flag = torch.zeros(1, device=features.device)
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                if torch.distributed.get_rank() == 0:
                                    _drop_flag.fill_(1.0 if torch.rand(1).item() < prior_dropout_prob else 0.0)
                                torch.distributed.broadcast(_drop_flag, src=0)
                            else:
                                _drop_flag.fill_(1.0 if torch.rand(1).item() < prior_dropout_prob else 0.0)
                            if bool(_drop_flag.item()):
                                # Out-of-place zeros to avoid in-place ops on graph tensors.
                                geo_prior_out["xyz_prior_64"] = torch.zeros_like(geo_prior_out["xyz_prior_64"])
                                geo_prior_out["prior_conf_64"] = torch.zeros_like(geo_prior_out["prior_conf_64"])
                                geo_prior_out["prior_ambiguity_64"] = torch.zeros_like(geo_prior_out["prior_ambiguity_64"])
                                geo_prior_out["retrieved_feat_8"] = torch.zeros_like(geo_prior_out["retrieved_feat_8"])
                                prior_was_dropped = True

                        if not prior_was_dropped:
                            profile_geo_prior_fuse_t0 = self._hotspot_profile_start()
                            features = features + geo_prior_out["retrieved_feat_8"].flatten(0, 1)
                            self._hotspot_profile_stop(
                                "geo_prior_fuse", profile_geo_prior_fuse_t0, batch_size=B, target_num=target_num
                            )
                        geo_prior_out_flat = {
                            "xyz_prior_64": geo_prior_out["xyz_prior_64"].flatten(0, 1),
                            "prior_conf_64": geo_prior_out["prior_conf_64"].flatten(0, 1),
                            "prior_ambiguity_64": geo_prior_out["prior_ambiguity_64"].flatten(0, 1),
                        }
                self._hotspot_profile_stop(
                    "view_interaction_total", profile_view_interaction_t0, batch_size=B, target_num=target_num
                )
            # ================================================================

            if features is None:
                profile_roi_target_t0 = self._hotspot_profile_start()
                features = self._encode_view_tokens_to_roi_features(target_hidden, roi_centers, scales, H, W)
                self._hotspot_profile_stop("roi_encode_target", profile_roi_target_t0, batch_size=B, target_num=target_num)

            # joints.shape [bs, 1152, 64, 64]
            profile_rot_head_t0 = self._hotspot_profile_start()
            mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features)
            self._hotspot_profile_stop("rot_head_forward", profile_rot_head_t0, batch_size=B, target_num=target_num)

        # TODO: remove this trans_head_net
        # trans = self.trans_head_net(features)

        device = x.device
        bs = features.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES

        def _flatten_batch_view_tensor(tensor):
            if tensor is None or (not torch.is_tensor(tensor)) or (not is_multiview_target):
                return tensor
            if tensor.dim() >= 2 and tensor.shape[0] == B and tensor.shape[1] == target_num:
                return tensor.flatten(0, 1).contiguous()
            return tensor

        roi_coord_2d_pnp = _flatten_batch_view_tensor(roi_coord_2d)
        roi_extents_pnp = _flatten_batch_view_tensor(roi_extents)
        roi_cams_pnp = _flatten_batch_view_tensor(roi_cams)
        roi_centers_pnp = _flatten_batch_view_tensor(roi_centers)
        resize_ratios_pnp = _flatten_batch_view_tensor(resize_ratios)
        roi_whs_pnp = _flatten_batch_view_tensor(roi_whs)
        gt_ego_rot_eval = _flatten_batch_view_tensor(gt_ego_rot)
        gt_trans_eval = _flatten_batch_view_tensor(gt_trans)
        gt_trans_ratio_eval = _flatten_batch_view_tensor(gt_trans_ratio)
        gt_xyz_eval = _flatten_batch_view_tensor(gt_xyz)
        gt_mask_visib_eval = _flatten_batch_view_tensor(gt_mask_visib)
        gt_points_eval = _flatten_batch_view_tensor(gt_points)

        roi_classes_head = roi_classes
        if roi_classes_head is not None and roi_classes_head.shape[0] != bs:
            if is_multiview_target and roi_classes_head.shape[0] == B:
                roi_classes_head = roi_classes_head.repeat_interleave(target_num, dim=0)
            else:
                raise ValueError(
                    f"roi_classes shape mismatch: roi_classes={roi_classes_head.shape[0]}, bs={bs}"
                )

        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        profile_head_post_t0 = self._hotspot_profile_start()
        if r_head_cfg.ROT_CLASS_AWARE:
            assert roi_classes_head is not None
            coor_x = coor_x.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes_head]
            coor_y = coor_y.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes_head]
            coor_z = coor_z.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes_head]

        if r_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes_head is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes_head]

        if r_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes_head is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes_head]
        self._hotspot_profile_stop("head_postprocess", profile_head_post_t0, batch_size=B, target_num=target_num)

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        profile_pnp_input_t0 = self._hotspot_profile_start()
        save_pnp_internals = bool(cfg.TEST.get("SAVE_PNP_INTERNALS", False))

        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
            xyz_single = decode_xyz_expectation_from_logits(coor_x[:, :-1, :, :], coor_y[:, :-1, :, :], coor_z[:, :-1, :, :])
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
            xyz_single = decode_xyz_expectation_from_logits(coor_x, coor_y, coor_z)

        if self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True) and geo_prior_out_flat is None:
            geo_prior_out = {
                "xyz_prior_64": torch.zeros(B, target_num, 3, out_res, out_res, device=features.device, dtype=features.dtype),
                "prior_conf_64": torch.zeros(B, target_num, 1, out_res, out_res, device=features.device, dtype=features.dtype),
                "prior_ambiguity_64": torch.zeros(B, target_num, 1, out_res, out_res, device=features.device, dtype=features.dtype),
            }
            geo_prior_out_flat = {
                "xyz_prior_64": geo_prior_out["xyz_prior_64"].flatten(0, 1),
                "prior_conf_64": geo_prior_out["prior_conf_64"].flatten(0, 1),
                "prior_ambiguity_64": geo_prior_out["prior_ambiguity_64"].flatten(0, 1),
            }
            # Inactive path uses zero prior + zero confidence as a pure placeholder.

        # Keep zero-filled prior channels when prior dropout is active so PnP
        # learns to remain stable without geometric prior support.
        coor_feat_base = coor_feat
        xyz_single_for_pnp = xyz_single
        prior_xyz_for_pnp = None
        prior_conf_for_pnp = None
        prior_residual_for_pnp = None
        if self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True) and geo_prior_out_flat is not None:
            prior_xyz_for_pnp = geo_prior_out_flat["xyz_prior_64"]
            if gp_cfg.get("APPEND_PRIOR_CONF", True):
                prior_conf_for_pnp = geo_prior_out_flat["prior_conf_64"]
            if gp_cfg.get("APPEND_PRIOR_RESIDUAL", True):
                prior_residual_for_pnp = xyz_single_for_pnp - prior_xyz_for_pnp

        coor_feat = assemble_pnp_inputs_with_geo_prior(
            coor_feat=coor_feat,
            roi_coord_2d_pnp=roi_coord_2d_pnp,
            xyz_single=xyz_single,
            geo_prior=geo_prior_out_flat if (self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True)) else None,
            with_2d_coord=pnp_net_cfg.WITH_2D_COORD,
            append_prior_conf=gp_cfg.get("APPEND_PRIOR_CONF", True),
            append_prior_residual=gp_cfg.get("APPEND_PRIOR_RESIDUAL", True),
        )

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        mask_atten = None
        profile_mask_attn_t0 = self._hotspot_profile_start()
        if pnp_net_cfg.MASK_ATTENTION != "none":
            if do_loss and gt_mask_obj is not None:
                _use_noisy = (
                    noisy_obj_masks is not None
                    and roi_centers is not None
                    and scales is not None
                )
                if _use_noisy:
                    _noisy_flat = noisy_obj_masks.to(device=coor_feat.device, dtype=coor_feat.dtype)
                    if _noisy_flat.dim() == 5:
                        _noisy_flat = _noisy_flat.flatten(0, 1)
                    if _noisy_flat.dim() == 3:
                        _noisy_flat = _noisy_flat.unsqueeze(1)
                    with torch.no_grad():
                        _ma = crop_resize_features(
                            _noisy_flat.detach(),
                            roi_centers.reshape(-1, 2).detach(),
                            scales.reshape(-1, 1).detach(),
                            output_size=out_res,
                        )
                    mask_atten = (_ma > 0.5).to(dtype=coor_feat.dtype)
                else:
                    _ma = gt_mask_obj
                    if _ma.dim() == 4 and is_multiview_target:
                        _ma = _ma.flatten(0, 1)
                    if _ma.dim() == 2:
                        _ma = _ma.unsqueeze(0)
                    if _ma.dim() == 3:
                        _ma = _ma.unsqueeze(1)
                    _ma = _ma.to(device=coor_feat.device, dtype=coor_feat.dtype)
                    if _ma.shape[-2:] != (out_res, out_res):
                        _ma = F.interpolate(_ma, size=(out_res, out_res), mode="nearest")
                    mask_atten = _ma.clamp(0.0, 1.0)
            else:
                mask_atten = get_mask_prob(cfg, mask)
        self._hotspot_profile_stop("mask_attention_prep", profile_mask_attn_t0, batch_size=B, target_num=target_num)

        region_atten = None
        if pnp_net_cfg.REGION_ATTENTION:
            region_atten = region_softmax
        self._hotspot_profile_stop("pnp_input_prep", profile_pnp_input_t0, batch_size=B, target_num=target_num)

        profile_pnp_net_t0 = self._hotspot_profile_start()
        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat, region=region_atten, extents=roi_extents_pnp, mask_attention=mask_atten
        )
        if pnp_net_cfg.R_ONLY:  # override trans pred
            pred_t_ = self.trans_head_net(features)
        self._hotspot_profile_stop("pnp_net_forward", profile_pnp_net_t0, batch_size=B, target_num=target_num)

        # convert pred_rot to rot mat -------------------------
        profile_pose_decode_t0 = self._hotspot_profile_start()
        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = self._rot_repr_to_mat(pred_rot_, rot_type)
        pred_ego_rot, pred_trans = decode_pose_from_raw(
            pnp_net_cfg,
            pred_rot_m,
            pred_t_,
            roi_cams=roi_cams_pnp,
            roi_centers=roi_centers_pnp,
            resize_ratios=resize_ratios_pnp,
            roi_whs=roi_whs_pnp,
            eps=1e-4,
            is_train=do_loss,
        )
        self._hotspot_profile_stop("pose_decode", profile_pose_decode_t0, batch_size=B, target_num=target_num)

        # Training must output B * target_num poses when multiview (one per batch item per target view).
        expected_pose_count = (B * target_num) if is_multiview_target else B
        if do_loss and pred_ego_rot.shape[0] != expected_pose_count:
            roi_cams_shape = roi_cams_pnp.shape if (roi_cams_pnp is not None and torch.is_tensor(roi_cams_pnp)) else None
            raise AssertionError(
                f"Pose count mismatch: pred_ego_rot.shape[0]={pred_ego_rot.shape[0]}, expected B*target_num={expected_pose_count} (B={B}, target_num={target_num}). "
                f"is_multiview_target={is_multiview_target}, features.shape={features.shape}, "
                f"roi_cams_pnp.shape={roi_cams_shape}."
            )
        # Context GT pose — used only for vis/debug metrics (error_R_context_self).
        context_gt_rot_bt = None
        context_gt_trans_bt = None
        if has_context_for_target:
            context_gt_rot_bt, context_gt_trans_bt = self._get_context_pose_from_full_gt(
                full_gt_ego_rot,
                full_gt_trans,
                batch_size=B,
                context_last_idx=context_last_idx,
                dtype=pred_rot_.dtype,
                device=pred_rot_.device,
            )
        context_rot_flat = context_gt_rot_bt.repeat_interleave(target_num, dim=0).contiguous() if torch.is_tensor(context_gt_rot_bt) else None
        context_trans_flat = context_gt_trans_bt.repeat_interleave(target_num, dim=0).contiguous() if torch.is_tensor(context_gt_trans_bt) else None


        if not do_loss:  # test
            out_dict = {
                "rot": pred_ego_rot,
                "trans": pred_trans,
                "abs_rot": pred_ego_rot,
                "abs_trans": pred_trans,
            }
            if context_rot_flat is not None:
                out_dict["context_rot"] = context_rot_flat
                out_dict["context_trans"] = context_trans_flat
            # TODO: move the pnp/ransac inside forward
            out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region})
        else:
            out_dict = {
                "rot": pred_ego_rot,
                "trans": pred_trans,
                "abs_rot": pred_ego_rot,
                "abs_trans": pred_trans,
                "gt_rot": gt_ego_rot,
                "gt_trans": gt_trans,
            }
            if context_rot_flat is not None:
                out_dict["context_rot"] = context_rot_flat
                out_dict["context_trans"] = context_trans_flat
            if cfg.TRAIN.VIS_IMG:
                out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region})
        if geo_prior_out_flat is not None:
            out_dict["xyz_prior_64"] = geo_prior_out_flat["xyz_prior_64"]
            out_dict["prior_conf_64"] = geo_prior_out_flat["prior_conf_64"]
            out_dict["prior_ambiguity_64"] = geo_prior_out_flat["prior_ambiguity_64"]
        if target_view_depth is not None and target_view_mask is not None:
            out_dict["target_view_depth"] = target_view_depth
            out_dict["target_view_mask"] = target_view_mask
            out_dict["target_view_mask_clean"] = target_view_mask
            out_dict["target_view_mask_raw"] = target_view_mask

        if do_loss:
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            gt_rot_supervise_eval = gt_ego_rot_eval
            gt_trans_supervise_eval = gt_trans_eval
            if torch.is_tensor(pred_ego_rot) and torch.is_tensor(gt_ego_rot_eval):
                gt_rot_supervise_eval, gt_trans_supervise_eval = self._canonicalize_abs_pose_targets(
                    cfg=cfg,
                    gt_rot=gt_ego_rot_eval,
                    gt_trans=gt_trans_eval,
                    model_infos=model_infos,
                    pred_rots_for_canon=pred_ego_rot.detach(),
                )
            geo_prior_gt_xyz_pose_aligned = gt_xyz_eval
            if (
                torch.is_tensor(gt_xyz_eval)
                and torch.is_tensor(gt_mask_visib_eval)
                and torch.is_tensor(gt_ego_rot_eval)
                and torch.is_tensor(gt_rot_supervise_eval)
                and torch.is_tensor(roi_extents_pnp)
            ):
                geo_prior_gt_xyz_pose_aligned, _ = self._align_xyz_targets_to_pose_supervision(
                    gt_xyz=gt_xyz_eval,
                    gt_xyz_bin=None,
                    gt_mask_xyz=gt_mask_visib_eval,
                    gt_rot_raw=gt_ego_rot_eval,
                    gt_rot_supervise=gt_rot_supervise_eval,
                    model_infos_batch=model_infos,
                    extents=roi_extents_pnp,
                    xyz_bin_num=None,
                )

            # print(f"------------{torch.linalg.det(pred_ego_rot)}------------------{torch.linalg.det(gt_ego_rot)}")
            # print(f"------------{pred_trans}------------------{gt_trans}")
            # print(f"------------{pred_ego_rot[0]}------------------{gt_ego_rot[0]}------------{pred_trans[0]}------------------{gt_trans[0]}")
            # print(f"------------{torch.linalg.norm(pred_ego_rot - gt_ego_rot)}------------------{torch.linalg.norm(pred_trans - gt_trans)}")
            storage = get_event_storage()
            vis_metrics_interval = max(int(cfg.TRAIN.get("VIS_METRICS_INTERVAL", 1)), 1)
            compute_vis_metrics_debug = ((int(storage.iter) + 1) % vis_metrics_interval) == 0
            vis_dict = {}
            geo_prior_gt_xyz_aligned = None
            geo_prior_gt_mask_aligned = None
            need_geo_prior_supervision = (
                geo_prior_active
                and (not prior_was_dropped)
                and (not is_all_target)
                and geo_prior_out_flat is not None
                and gt_xyz_eval is not None
                and gt_mask_visib_eval is not None
            )
            if need_geo_prior_supervision:
                geo_prior_gt_xyz_aligned, geo_prior_gt_mask_aligned = align_geo_prior_debug_targets(
                    xyz_prior=geo_prior_out_flat["xyz_prior_64"],
                    gt_xyz=geo_prior_gt_xyz_pose_aligned,
                    gt_mask_visib=gt_mask_visib_eval,
                )
                if geo_prior_gt_xyz_aligned is None or geo_prior_gt_mask_aligned is None:
                    raise AssertionError(
                        "Cannot align geo-prior supervision targets to xyz_prior shape: "
                        f"xyz_prior={tuple(geo_prior_out_flat['xyz_prior_64'].shape)}, "
                        f"gt_xyz={tuple(gt_xyz_eval.shape) if torch.is_tensor(gt_xyz_eval) else gt_xyz_eval}, "
                        f"gt_mask_visib={tuple(gt_mask_visib_eval.shape) if torch.is_tensor(gt_mask_visib_eval) else gt_mask_visib_eval}."
                    )
            if compute_vis_metrics_debug:
                profile_vis_metrics_t0 = self._hotspot_profile_start()
                mean_re, mean_te = compute_mean_re_te(pred_trans, pred_ego_rot, gt_trans_eval, gt_ego_rot_eval)
                mean_re_sym, mean_te_sym = self._compute_mean_re_te_symmetry_aware(
                    pred_trans,
                    pred_ego_rot,
                    gt_trans_eval,
                    gt_ego_rot_eval,
                    model_infos,
                )
                mean_txy_cm, mean_tz_cm = self._compute_mean_t_xy_z_cm(pred_trans, gt_trans_eval)
                vis_dict.update(
                    {
                        "vis/error_R": mean_re,
                        "vis/error_R_sym": mean_re_sym,
                        "vis/error_t": mean_te * 100,  # cm
                        "vis/error_t_sym": mean_te_sym * 100,  # cm
                        "vis/error_t_xy": mean_txy_cm,
                        "vis/error_t_z": mean_tz_cm,
                        "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans_eval[0, 0].detach().item()) * 100,  # cm
                        "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans_eval[0, 1].detach().item()) * 100,  # cm
                        "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans_eval[0, 2].detach().item()) * 100,  # cm
                        "vis/tx_pred": pred_trans[0, 0].detach().item(),
                        "vis/ty_pred": pred_trans[0, 1].detach().item(),
                        "vis/tz_pred": pred_trans[0, 2].detach().item(),
                        "vis/tx_net": pred_t_[0, 0].detach().item(),
                        "vis/ty_net": pred_t_[0, 1].detach().item(),
                        "vis/tz_net": pred_t_[0, 2].detach().item(),
                        "vis/tx_gt": gt_trans_eval[0, 0].detach().item(),
                        "vis/ty_gt": gt_trans_eval[0, 1].detach().item(),
                        "vis/tz_gt": gt_trans_eval[0, 2].detach().item(),
                        "vis/tx_ratio_gt": gt_trans_ratio_eval[0, 0].detach().item(),
                        "vis/ty_ratio_gt": gt_trans_ratio_eval[0, 1].detach().item(),
                        "vis/tz_ratio_gt": gt_trans_ratio_eval[0, 2].detach().item(),
                        "vis/cont_sample_num": 0.0,
                        "vis/cont_sample_ratio": 0.0,
                    }
                )
                vis_dict.update(
                    compute_geo_prior_debug_metrics(
                        geo_prior_out=geo_prior_out,
                        geo_prior_out_flat=geo_prior_out_flat,
                        geo_prior_active=geo_prior_active,
                        prior_was_dropped=prior_was_dropped,
                        is_all_target=is_all_target,
                        gt_xyz=geo_prior_gt_xyz_aligned,
                        gt_mask_visib=geo_prior_gt_mask_aligned,
                    )
                )
                self._hotspot_profile_stop("vis_metrics_debug", profile_vis_metrics_t0, batch_size=B, target_num=target_num)
            vis_dict["vis/geo_prior_inject_to_pnp"] = float(
                self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True)
            )
            loss_fn = self.gdrn_loss_multiview if is_multiview_target else self.gdrn_loss
            loss_dict = loss_fn(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                model_infos=model_infos,
                extents=roi_extents,
                target_idx_list=target_idx_list,
                full_gt_rot=full_gt_ego_rot,
                full_gt_trans=full_gt_trans,
                # abs head has no context input; use closest_pose (prediction-anchored)
                # symmetry branch selection so the same image always maps to the same
                # branch and the head can converge stably.
                gt_rot_anchor_for_sym=None,
                # roi_classes=roi_classes,
            )

            if target_view_depth is not None and target_view_mask is not None and input_depths is not None:
                profile_depth_mask_loss_t0 = self._hotspot_profile_start()
                depth_mask_loss_dict = self.depth_mask_loss(
                    cfg=cfg,
                    depth_pred=target_view_depth,
                    mask_pred=target_view_mask,
                    depth_gt=input_depths,
                    mask_gt=input_obj_masks,
                    roi_cams=roi_cams,
                    pred_rot=pred_ego_rot,
                    pred_trans=pred_trans,
                    gt_rot=gt_rot_supervise_eval,
                    gt_trans=gt_trans_supervise_eval,
                )
                if cfg.MODEL.CDPN.USE_MTL:
                    for _k in list(depth_mask_loss_dict.keys()):
                        _name = _k.replace("loss_", "log_var_")
                        if hasattr(self, _name):
                            cur_log_var = getattr(self, _name)
                            depth_mask_loss_dict[_k] = (
                                depth_mask_loss_dict[_k] * torch.exp(-cur_log_var)
                                + torch.log(1 + torch.exp(cur_log_var))
                            )
                loss_dict.update(depth_mask_loss_dict)
                self._hotspot_profile_stop(
                    "depth_mask_loss_forward", profile_depth_mask_loss_t0, batch_size=B, target_num=target_num
                )

            vi_cfg_train = cfg.MODEL.CDPN.get("VIEW_INTERACTION", {})
            if (
                geo_prior_active
                and (not prior_was_dropped)
                and (not is_all_target)
                and geo_prior_out_flat is not None
                and geo_prior_gt_xyz_aligned is not None
                and geo_prior_gt_mask_aligned is not None
            ):
                profile_geo_prior_train_loss_t0 = self._hotspot_profile_start()
                geo_losses = compute_geo_prior_training_losses(
                    xyz_prior=geo_prior_out_flat["xyz_prior_64"],
                    prior_conf=geo_prior_out_flat["prior_conf_64"],
                    prior_ambiguity=geo_prior_out_flat["prior_ambiguity_64"],
                    roi_xyz=geo_prior_gt_xyz_aligned,
                    roi_mask_visib=geo_prior_gt_mask_aligned,
                    cfg_loss=get_geo_prior_loss_cfg(vi_cfg_train),
                )
                loss_dict.update(geo_losses)
                self._hotspot_profile_stop(
                    "geo_prior_train_loss", profile_geo_prior_train_loss_t0, batch_size=B, target_num=target_num
                )

            rel_geo_cfg = vi_cfg_train.get("RELATIVE_GEOMETRY_LOSS", {})
            vis_dict["vis/rel_geo_enabled"] = float(rel_geo_cfg.get("ENABLED", False))
            vis_dict["vis/loss_rel_geo_sym_raw"] = 0.0
            vis_dict["vis/loss_rel_trans_raw"] = 0.0
            if (
                rel_geo_cfg.get("ENABLED", False)
                and has_context_for_target
                and (not is_all_target)
                and gt_points_eval is not None
                and gt_ego_rot_eval is not None
                and gt_trans_eval is not None
            ):
                profile_rel_geo_loss_t0 = self._hotspot_profile_start()
                context_gt_rot_bt, context_gt_trans_bt = self._get_context_pose_from_full_gt(
                    full_gt_ego_rot, full_gt_trans, B, context_last_idx, dtype=pred_rot_.dtype, device=pred_rot_.device
                )
                context_pose_hypothesis_rot_bt = context_gt_rot_bt.clone()
                context_pose_hypothesis_trans_bt = context_gt_trans_bt.clone()
                if get_geo_prior_hyp_source(vi_cfg_train) == "noisy_gt":
                    context_pose_hypothesis_rot_bt, context_pose_hypothesis_trans_bt = self._add_pose_noise_for_ctx_info(
                        context_pose_hypothesis_rot_bt,
                        context_pose_hypothesis_trans_bt,
                        vi_cfg_train,
                    )

                context_gt_rot_bt, context_gt_trans_bt, context_hyp_rot_bt, context_hyp_trans_bt = (
                    self._build_context_supervise_pose_tensors(
                        full_gt_ego_rot=full_gt_ego_rot,
                        full_gt_trans=full_gt_trans,
                        context_gt_rot_bt=context_gt_rot_bt,
                        context_gt_trans_bt=context_gt_trans_bt,
                        context_pose_hypothesis_rot_bt=context_pose_hypothesis_rot_bt,
                        context_pose_hypothesis_trans_bt=context_pose_hypothesis_trans_bt,
                        model_infos=model_infos,
                        context_last_idx=context_last_idx,
                        batch_size=B,
                        dtype=pred_rot_.dtype,
                        device=pred_rot_.device,
                    )
                )
                context_gt_rot_flat = context_gt_rot_bt.repeat_interleave(target_num, dim=0).contiguous()
                context_gt_trans_flat = context_gt_trans_bt.repeat_interleave(target_num, dim=0).contiguous()
                context_hyp_rot_flat = context_hyp_rot_bt.repeat_interleave(target_num, dim=0).contiguous()
                context_hyp_trans_flat = context_hyp_trans_bt.repeat_interleave(target_num, dim=0).contiguous()

                gt_rot_supervise, gt_trans_supervise = gt_rot_supervise_eval, gt_trans_supervise_eval

                rel_losses = self.relative_geo_sym_loss(
                    pred_rot=pred_ego_rot,
                    pred_trans=pred_trans,
                    ctx_hyp_rot=context_hyp_rot_flat,
                    ctx_hyp_trans=context_hyp_trans_flat,
                    gt_rot=gt_rot_supervise,
                    gt_trans=gt_trans_supervise,
                    ctx_gt_rot=context_gt_rot_flat,
                    ctx_gt_trans=context_gt_trans_flat,
                    model_points=gt_points_eval,
                    max_points=int(rel_geo_cfg.get("NUM_POINTS", 1024)),
                )
                vis_dict["vis/loss_rel_geo_sym_raw"] = float(rel_losses["loss_rel_geo_sym"].detach().item())
                vis_dict["vis/loss_rel_trans_raw"] = float(rel_losses["loss_rel_trans"].detach().item())
                loss_dict["loss_rel_geo_sym"] = rel_losses["loss_rel_geo_sym"] * float(rel_geo_cfg.get("LW", 0.0))
                loss_dict["loss_rel_trans"] = rel_losses["loss_rel_trans"] * float(rel_geo_cfg.get("TRANS_LW", 0.0))
                self._hotspot_profile_stop(
                    "rel_geo_loss_forward", profile_rel_geo_loss_t0, batch_size=B, target_num=target_num
                )

            if self.geo_prior_enabled and hasattr(self, "context_geo_prior"):
                profile_ddp_anchor_t0 = self._hotspot_profile_start()
                # DDP anchor: keep geo-prior parameters in the graph every iteration
                # even when all-target / prior-dropout branches skip geo-prior supervision.
                loss_dict["loss_geo_prior_ddp_anchor"] = build_module_ddp_dummy_anchor_loss(
                    self.context_geo_prior, reference_tensor=pred_t_
                )
                self._hotspot_profile_stop(
                    "geo_prior_ddp_anchor", profile_ddp_anchor_t0, batch_size=B, target_num=target_num
                )

            profile_storage_t0 = self._hotspot_profile_start()
            if cfg.MODEL.CDPN.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage.put_scalars(**vis_dict)
            self._hotspot_profile_stop("event_storage_write", profile_storage_t0, batch_size=B, target_num=target_num)

            self._hotspot_profile_end_forward(
                profile_forward_t0,
                batch_size=B,
                view_num=N,
                target_num=len(target_idx) if isinstance(target_idx, (list, tuple)) else 1,
            )
            return out_dict, loss_dict
        self._hotspot_profile_end_forward(profile_forward_t0, batch_size=B, view_num=N)
        return out_dict

    def forward_infer(
        self,
        input_images,
        roi_classes=None,
        roi_cams=None,
        roi_extents=None,
        model_infos=None,
        gt_ego_rot=None,
        gt_trans=None,
        gt_pose_valid=None,   # (B, N) bool tensor: True if pose was actually predicted (not placeholder)
        seed_roi_centers=None,
        seed_scales=None,
        target_idx=-1,
        mask_thr=0.5,
        min_mask_pixels=16,
        model_points=None,
        external_target_masks=None,
        prev_target_mask=None,
        mask_postproc="none",
        mask_prev_dilate_kernel=11,
        mask_prev_gate=True,
        mask_post_open_kernel=0,
        mask_post_dilate_kernel=3,
        mask_post_close_kernel=3,
        mask_fallback_to_prev=False,
        external_mask_pad_scale=None,
    ):
        """Inference-only forward that derives RoI from predicted mask."""
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        if not cfg.MODEL.CDPN.VGGT_BACKBONE:
            raise NotImplementedError("forward_infer requires MODEL.CDPN.VGGT_BACKBONE=True")
        if roi_cams is None or roi_extents is None:
            raise ValueError("forward_infer requires roi_cams and roi_extents.")

        B, N, C, H, W = input_images.shape
        h_p, w_p = H // 16, W // 16

        target_ids, has_context_for_target, is_all_target = analyze_target_indices(target_idx, N)
        target_num = len(target_ids)
        context_last_idx = max(min(target_ids) - 1, 0)

        def _flatten_batch_view_tensor(tensor):
            if tensor is None:
                return None
            if torch.is_tensor(tensor) and tensor.dim() >= 2 and tensor.shape[0] == B and tensor.shape[1] == target_num:
                return tensor.flatten(0, 1).contiguous()
            return tensor

        # 1) Encode full image tokens and predict target-view depth/mask.
        input_images_flat = input_images.reshape(B * N, C, H, W)
        hidden = self._run_image_encoder(input_images_flat)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, _ = self.decode(hidden, N, H, W)  # (B*N, h_p*w_p, C)
        hidden_by_view = hidden.reshape(B, N, h_p * w_p, -1).contiguous()
        target_hidden = hidden_by_view[:, target_ids].contiguous()  # (B, T, hw, C)

        target_hidden_2d = target_hidden.reshape(B * target_num, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
        target_depth_2d, target_mask_2d = self.depth_head(target_hidden_2d, out_size=(H, W))
        target_view_depth = target_depth_2d.reshape(B, target_num, 1, H, W).contiguous()
        target_view_mask = target_mask_2d.reshape(B, target_num, 1, H, W).contiguous()

        external_target_masks_target = None
        if external_target_masks is not None:
            if external_target_masks.dim() == 5:
                external_target_masks_target = external_target_masks[:, target_ids]
            elif external_target_masks.dim() == 4:
                external_target_masks_target = external_target_masks[:, target_ids].unsqueeze(2)
            else:
                raise ValueError(f"external_target_masks must have 4 or 5 dims, got {external_target_masks.dim()}")
            external_target_masks_target = external_target_masks_target.to(device=input_images.device, dtype=input_images.dtype)
            if external_target_masks_target.shape[-2:] != (H, W):
                external_target_masks_target = F.interpolate(
                    external_target_masks_target.flatten(0, 1),
                    size=(H, W),
                    mode="nearest",
                ).reshape(B, target_num, 1, H, W)
            external_target_masks_target = (external_target_masks_target > 0.5).to(dtype=input_images.dtype)

        # 2) Convert mask to RoI geometry.
        # If external masks are provided, use them as the mask source for all downstream
        # ROI/bbox estimation instead of the model-predicted mask.
        mask_act = str(cfg.MODEL.CDPN.get("DEPTH_HEAD_MASK_ACT", "none")).lower()
        if mask_act == "sigmoid":
            mask_prob = target_view_mask.clamp(0.0, 1.0)
        else:
            mask_prob = torch.sigmoid(target_view_mask)
        pred_raw_mask_bin = mask_prob[:, :, 0] > float(mask_thr)  # (B, T, H, W)
        if external_target_masks_target is not None:
            raw_mask_bin = external_target_masks_target[:, :, 0] > 0.5
        else:
            raw_mask_bin = pred_raw_mask_bin

        if external_target_masks_target is not None:
            mask_bin = raw_mask_bin
            target_view_mask_clean = raw_mask_bin.to(dtype=input_images.dtype).unsqueeze(2)
        else:
            if prev_target_mask is not None:
                prev_target_mask_np = prev_target_mask.detach().cpu().numpy()
            else:
                prev_target_mask_np = None

            if str(mask_postproc).lower() == "none":
                mask_bin = raw_mask_bin
                target_view_mask_clean = raw_mask_bin.to(dtype=input_images.dtype).unsqueeze(2)
            else:
                raw_mask_bin_np = raw_mask_bin.detach().cpu().numpy().astype(np.uint8)
                clean_mask_np = np.zeros_like(raw_mask_bin_np, dtype=np.uint8)
                for b in range(B):
                    prev_mask_b = None
                    if prev_target_mask_np is not None:
                        prev_mask_b = np.asarray(prev_target_mask_np[b, 0], dtype=np.uint8)
                    for t in range(target_num):
                        clean_mask_np[b, t] = clean_mask_with_temporal_prior(
                            cur_mask_u8=raw_mask_bin_np[b, t],
                            prev_mask_u8=prev_mask_b,
                            mode=mask_postproc,
                            prev_dilate_kernel=mask_prev_dilate_kernel,
                            prev_gate=mask_prev_gate,
                            post_open_kernel=mask_post_open_kernel,
                            post_dilate_kernel=mask_post_dilate_kernel,
                            post_close_kernel=mask_post_close_kernel,
                            min_mask_pixels=min_mask_pixels,
                            fallback_to_prev=mask_fallback_to_prev,
                        )
                        prev_mask_b = clean_mask_np[b, t]
                mask_bin = torch.from_numpy(clean_mask_np > 0).to(device=input_images.device, dtype=torch.bool)
                target_view_mask_clean = torch.from_numpy(clean_mask_np[:, :, None].astype(np.float32)).to(
                    device=input_images.device, dtype=input_images.dtype
                )

        observed_target_mask = target_view_mask_clean

        base_pad_scale = float(cfg.INPUT.DZI_PAD_SCALE)
        if external_mask_pad_scale is not None:
            external_mask_pad_scale = float(external_mask_pad_scale)
            if external_mask_pad_scale <= 0:
                raise ValueError(f"external_mask_pad_scale must be > 0, got {external_mask_pad_scale}")
        patch_size = int(getattr(self, "patch_size", 16))
        patch_align = bool(getattr(cfg.INPUT, "DZI_PATCH_GRID_ADSORPTION", False))
        roi_centers, scales, roi_whs = masks_to_roi_geometry(
            mask_bin=mask_bin,
            output_dtype=input_images.dtype,
            output_device=input_images.device,
            image_hw=(H, W),
            base_pad_scale=base_pad_scale,
            patch_size=patch_size,
            patch_align=patch_align,
            min_mask_pixels=min_mask_pixels,
            external_mask_pad_scale=external_mask_pad_scale,
            use_external_masks=external_target_masks_target is not None,
        )

        resize_ratios = out_res / scales[..., 0].clamp(min=1e-6)  # (B, T)

        # 3) Build roi_coord_2d by cropping full-image 2D coordinate maps.
        xs = torch.linspace(0, 1, W, device=input_images.device, dtype=input_images.dtype)
        ys = torch.linspace(0, 1, H, device=input_images.device, dtype=input_images.dtype)
        yy, xx = torch.meshgrid(ys, xs)
        full_coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B * target_num, -1, -1, -1).contiguous()
        roi_coord_2d = crop_resize_features(
            full_coord,
            roi_centers.reshape(B * target_num, 2),
            scales.reshape(B * target_num, 1),
            output_size=out_res,
        ).reshape(B, target_num, 2, out_res, out_res).contiguous()

        # 4) RoI feature decoding + optional ResNet fusion.
        features = self._encode_view_tokens_to_roi_features(target_hidden, roi_centers, scales, H, W)

        if cfg.MODEL.CDPN.RESNET_BACKBONE:
            input_res = int(cfg.MODEL.CDPN.BACKBONE.INPUT_RES)
            target_imgs = input_images[:, target_ids].reshape(B * target_num, C, H, W).contiguous()
            roi_imgs = crop_resize_features(
                target_imgs,
                roi_centers.reshape(B * target_num, 2),
                scales.reshape(B * target_num, 1),
                output_size=input_res,
            )
            roi_features = self.backbone(roi_imgs)
            if cfg.MODEL.CDPN.VGGT_BACKBONE:
                features = features + roi_features
            else:
                features = roi_features

        # ================================================================
        # View Interaction: inject context-view info (inference path)
        # ================================================================
        # Derive per-view pose validity.  gt_pose_valid (B, N) bool tensor
        # tells us which views have real predicted poses vs. I/0 placeholders.
        # Fall back to a heuristic check when not explicitly provided.
        if gt_pose_valid is not None:
            _pose_valid_np = gt_pose_valid[0].cpu().numpy().astype(bool)  # (N,)
        elif gt_ego_rot is not None and gt_trans is not None:
            _eye = torch.eye(3, device=gt_ego_rot.device, dtype=gt_ego_rot.dtype)
            _is_identity = (gt_ego_rot[0] - _eye.unsqueeze(0)).abs().max(dim=-1).values.max(dim=-1).values < 1e-6  # (N,)
            _is_zero_t = gt_trans[0].abs().max(dim=-1).values < 1e-6  # (N,)
            _pose_valid_np = ~(_is_identity & _is_zero_t).cpu().numpy()  # (N,)
        else:
            _pose_valid_np = np.zeros(N, dtype=bool)

        run_vi_infer = (
            self.view_interaction_enabled
            and (has_context_for_target or (is_all_target and N >= 2))
            and gt_ego_rot is not None
            and gt_trans is not None
            and (is_all_target or (seed_roi_centers is not None and seed_scales is not None))
        )
        gp_cfg = self.geo_prior_cfg if hasattr(self, "geo_prior_cfg") else {}
        geo_prior_out = None
        geo_prior_out_flat = None
        if run_vi_infer:
            if is_all_target:
                # Leave-one-view-out: each target view attends to all other N-1 views as context.
                # Context xyz cue is pose-dependent (backprojection); for views whose poses are
                # not yet predicted (first AR chunk), we zero the xyz cue so the model still
                # sees cross-view texture/shape information via rgb_embed + mask, just without
                # geometry.  Training exercises the same zero-xyz path via XYZ_CUE_DROPOUT_PROB.
                features_list = []
                features_by_view = features.reshape(B, target_num, *features.shape[1:]).contiguous()
                target_local_pos = {int(view_id): local_i for local_i, view_id in enumerate(target_ids)}
                for i, tgt_i in enumerate(target_ids):
                    # All other views are context (no pose validity filter here — we zero xyz instead)
                    ctx_idx_i = [j for j in range(N) if j != tgt_i]
                    ctx_num_i = len(ctx_idx_i)
                    tgt_local_i = target_local_pos[int(tgt_i)]
                    ctx_local_idx_i = [target_local_pos[int(j)] for j in ctx_idx_i]

                    # Single target feature (B, 512, 8, 8)
                    feat_i = features_by_view[:, tgt_local_i].contiguous()

                    # Reuse current all-view target ROI features as context memory.
                    # In all-target mode every view in the window is already a target view in
                    # this same forward, so these current mask-derived RoIs are the most
                    # self-consistent context representation and avoid re-running roi decoding.
                    ctx_roi_feat_i = features_by_view[:, ctx_local_idx_i].reshape(
                        B * ctx_num_i, *features.shape[1:]
                    ).contiguous()
                    ctx_roi_centers_i = roi_centers[:, ctx_idx_i]
                    ctx_scales_i = scales[:, ctx_idx_i]

                    ctx_hidden_i = hidden_by_view[:, ctx_idx_i].contiguous()

                    ctx_hidden_2d_i = ctx_hidden_i.reshape(
                        B * ctx_num_i, h_p, w_p, -1
                    ).permute(0, 3, 1, 2).contiguous()
                    ctx_depth_i, ctx_mask_i = self.depth_head(ctx_hidden_2d_i, out_size=(H, W))

                    # Build xyz cue: backproject for valid-pose views, zero for invalid.
                    # This makes first-chunk all-target consistent with subsequent chunks
                    # (cross-view attention always runs; only the xyz geometry is absent).
                    ctx_xyz_roi_i = torch.zeros(B * ctx_num_i, 3, 8, 8, device=features.device, dtype=features.dtype)
                    valid_local = [li for li, gj in enumerate(ctx_idx_i) if _pose_valid_np[gj]]
                    if valid_local:
                        vl_idx = valid_local  # local indices within ctx_idx_i
                        ctx_R_v = gt_ego_rot[:, [ctx_idx_i[k] for k in vl_idx]].reshape(len(vl_idx) * B, 3, 3).to(
                            device=features.device, dtype=features.dtype)
                        ctx_t_v = gt_trans[:, [ctx_idx_i[k] for k in vl_idx]].reshape(len(vl_idx) * B, 3).to(
                            device=features.device, dtype=features.dtype)
                        ctx_K_v = roi_cams[:, [ctx_idx_i[k] for k in vl_idx]].reshape(len(vl_idx) * B, 3, 3).to(
                            device=features.device, dtype=features.dtype)
                        ctx_depth_v = ctx_depth_i.reshape(B, ctx_num_i, 1, H, W)[:, vl_idx].reshape(len(vl_idx) * B, 1, H, W)
                        ctx_mask_v  = ctx_mask_i.reshape(B, ctx_num_i, 1, H, W)[:, vl_idx].reshape(len(vl_idx) * B, 1, H, W)
                        ctx_xyz_full_v = _backproject_depth_to_xyz(
                            ctx_depth_v, torch.sigmoid(ctx_mask_v), ctx_R_v, ctx_t_v, ctx_K_v
                        )
                        ctx_roi_centers_v = roi_centers[:, [ctx_idx_i[k] for k in vl_idx]]
                        ctx_scales_v = scales[:, [ctx_idx_i[k] for k in vl_idx]]
                        ctx_xyz_roi_v = crop_resize_features(
                            ctx_xyz_full_v,
                            ctx_roi_centers_v.reshape(len(vl_idx) * B, 2),
                            ctx_scales_v.reshape(len(vl_idx) * B, 1),
                            output_size=8,
                        )
                        if roi_extents is not None:
                            ctx_ext_v = roi_extents[:, [ctx_idx_i[k] for k in vl_idx]].reshape(len(vl_idx) * B, 3)
                            ctx_xyz_roi_v = ctx_xyz_roi_v / ctx_ext_v.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6) + 0.5
                        # Write valid rows back into the full ctx_xyz_roi_i tensor
                        # ctx_xyz_roi_i is (B*C, 3, 8, 8) with B=1 in inference, so row k maps to vl_idx[k]
                        for out_k, in_k in enumerate(vl_idx):
                            ctx_xyz_roi_i[in_k] = ctx_xyz_roi_v[out_k]

                    # Target RGB: use current mask-derived roi_centers/scales (not seed_*)
                    # so that rgb_embed aligns with roi_feat and roi_coord_2d.
                    tgt_img_i = crop_resize_features(
                        input_images[:, tgt_i],  # (B, C, H, W)
                        roi_centers[:, i],        # (B, 2) — current mask-derived RoI
                        scales[:, i],             # (B, 1)
                        output_size=256,
                    )
                    ctx_imgs_i = crop_resize_features(
                        input_images[:, ctx_idx_i].reshape(B * ctx_num_i, C, H, W),
                        ctx_roi_centers_i.reshape(B * ctx_num_i, 2),
                        ctx_scales_i.reshape(B * ctx_num_i, 1),
                        output_size=256,
                    )

                    tgt_coord_i = F.adaptive_avg_pool2d(
                        roi_coord_2d.reshape(B, target_num, 2, roi_coord_2d.shape[-2], roi_coord_2d.shape[-1])[:, i], 8
                    )
                    tgt_mask_vi_i = F.adaptive_avg_pool2d(torch.sigmoid(target_view_mask[:, i, :1]), 8)

                    _dev = features.device
                    xs_vi = torch.linspace(0, 1, W, device=_dev, dtype=features.dtype)
                    ys_vi = torch.linspace(0, 1, H, device=_dev, dtype=features.dtype)
                    yy_vi, xx_vi = torch.meshgrid(ys_vi, xs_vi, indexing="ij")
                    fc_vi = torch.stack([xx_vi, yy_vi], dim=0).unsqueeze(0).expand(B * ctx_num_i, -1, -1, -1)
                    ctx_coord_i = crop_resize_features(
                        fc_vi, ctx_roi_centers_i.reshape(B * ctx_num_i, 2),
                        ctx_scales_i.reshape(B * ctx_num_i, 1), output_size=8,
                    )
                    ctx_mask_vi_i = F.adaptive_avg_pool2d(torch.sigmoid(ctx_mask_i), 8)

                    feat_i_enhanced = self.view_interaction(
                        target_roi_feat=feat_i,
                        context_roi_feat=ctx_roi_feat_i,
                        context_xyz_cue=ctx_xyz_roi_i,
                        target_roi_rgb=tgt_img_i,
                        context_roi_rgb=ctx_imgs_i,
                        target_roi_coord2d=tgt_coord_i,
                        context_roi_coord2d=ctx_coord_i,
                        target_roi_mask=tgt_mask_vi_i,
                        context_roi_mask=ctx_mask_vi_i,
                        batch_size=B,
                        target_num=1,
                        context_num=ctx_num_i,
                        target_time_ids=[tgt_i],
                        context_time_ids=ctx_idx_i,
                    )  # (B, 512, 8, 8)
                    features_list.append(feat_i_enhanced)

                # Reassemble: (B, T, 512, 8, 8) → (B*T, 512, 8, 8)
                features = torch.stack(features_list, dim=1).flatten(0, 1).contiguous()

            else:
                # context-then-target: fixed context set
                context_idx_list = [i for i in range(N) if i not in target_ids]
                context_num = len(context_idx_list)

                # Filter to valid context views only
                context_idx_list = [j for j in context_idx_list if _pose_valid_np[j]]
                context_num = len(context_idx_list)

                if context_num > 0:

                    # Context ROI geometry from cached seeds
                    ctx_roi_centers = seed_roi_centers[:, context_idx_list]  # (B, C, 2)
                    ctx_scales = seed_scales[:, context_idx_list]  # (B, C, 1)

                    # Context hidden → roi features
                    context_hidden = hidden_by_view[:, context_idx_list].contiguous()
                    context_roi_feat = self._encode_view_tokens_to_roi_features(
                        context_hidden, ctx_roi_centers, ctx_scales, H, W
                    )  # (B*C, 512, 8, 8)

                    # Depth/mask for context views
                    ctx_hidden_2d = context_hidden.reshape(
                        B * context_num, h_p, w_p, -1
                    ).permute(0, 3, 1, 2).contiguous()
                    ctx_depth_infer, ctx_mask_infer = self.depth_head(ctx_hidden_2d, out_size=(H, W))
                    ctx_mask_for_geom = torch.sigmoid(ctx_mask_infer)
                    # If external masks are available, prefer them for context geometry/mask tokens.
                    # This reduces train/infer gap and avoids injecting depth-head mask noise into
                    # context bank construction when mask observations are provided.
                    if external_target_masks is not None:
                        ctx_external_masks = external_target_masks[:, context_idx_list].reshape(B * context_num, 1, H, W).to(
                            device=features.device, dtype=features.dtype
                        )
                        ctx_mask_for_geom = ctx_external_masks.clamp(0.0, 1.0)

                    # xyz cue from cached context pose
                    ctx_R = gt_ego_rot[:, context_idx_list].reshape(B * context_num, 3, 3).to(
                        device=features.device, dtype=features.dtype)
                    ctx_t = gt_trans[:, context_idx_list].reshape(B * context_num, 3).to(
                        device=features.device, dtype=features.dtype)
                    ctx_K = roi_cams[:, context_idx_list].reshape(B * context_num, 3, 3).to(
                        device=features.device, dtype=features.dtype)

                    ctx_xyz_full = _backproject_depth_to_xyz(
                        ctx_depth_infer, ctx_mask_for_geom, ctx_R, ctx_t, ctx_K
                    )
                    ctx_xyz_roi = crop_resize_features(
                        ctx_xyz_full,
                        ctx_roi_centers.reshape(B * context_num, 2),
                        ctx_scales.reshape(B * context_num, 1),
                        output_size=8,
                    )
                    if roi_extents is not None:
                        ctx_ext = roi_extents[:, context_idx_list].reshape(B * context_num, 3)
                        ctx_xyz_roi = ctx_xyz_roi / ctx_ext.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6) + 0.5

                    # RoI RGB crops via crop_resize from full images
                    target_imgs_vi = crop_resize_features(
                        input_images[:, target_ids].reshape(B * target_num, C, H, W),
                        roi_centers.reshape(B * target_num, 2),
                        scales.reshape(B * target_num, 1),
                        output_size=256,
                    )
                    context_imgs_vi = crop_resize_features(
                        input_images[:, context_idx_list].reshape(B * context_num, C, H, W),
                        ctx_roi_centers.reshape(B * context_num, 2),
                        ctx_scales.reshape(B * context_num, 1),
                        output_size=256,
                    )

                    # coord_2d and mask at 8x8
                    tgt_coord_vi = F.adaptive_avg_pool2d(
                        _flatten_batch_view_tensor(roi_coord_2d), 8
                    )
                    tgt_mask_vi = F.adaptive_avg_pool2d(
                        torch.sigmoid(target_view_mask.flatten(0, 1)[:, :1]), 8
                    )  # (B*T, 1, 8, 8) from depth head

                    _dev = features.device
                    xs_vi = torch.linspace(0, 1, W, device=_dev, dtype=features.dtype)
                    ys_vi = torch.linspace(0, 1, H, device=_dev, dtype=features.dtype)
                    yy_vi, xx_vi = torch.meshgrid(ys_vi, xs_vi, indexing="ij")
                    fc_vi = torch.stack([xx_vi, yy_vi], dim=0).unsqueeze(0).expand(B * context_num, -1, -1, -1)
                    ctx_coord_vi = crop_resize_features(
                        fc_vi, ctx_roi_centers.reshape(B * context_num, 2),
                        ctx_scales.reshape(B * context_num, 1), output_size=8,
                    )
                    # Keep VI/geo-prior context mask source consistent with xyz backprojection above.
                    ctx_mask_vi = F.adaptive_avg_pool2d(ctx_mask_for_geom, 8)
                    ctx_view_dir_obj, ctx_cam_center_obj = build_context_view_geom_features(ctx_R, ctx_t, out_hw=8)

                    features = self.view_interaction(
                        target_roi_feat=features,
                        context_roi_feat=context_roi_feat,
                        context_xyz_cue=ctx_xyz_roi,
                        target_roi_rgb=target_imgs_vi,
                        context_roi_rgb=context_imgs_vi,
                        target_roi_coord2d=tgt_coord_vi,
                        context_roi_coord2d=ctx_coord_vi,
                        target_roi_mask=tgt_mask_vi,
                        context_roi_mask=ctx_mask_vi,
                        batch_size=B,
                        target_num=target_num,
                        context_num=context_num,
                        target_time_ids=target_ids,
                        context_time_ids=context_idx_list,
                    )

                    if self.geo_prior_enabled and context_num > 0 and not is_all_target:
                        ctx_pose_conf_bt = build_context_pose_confidence(gt_ego_rot, gt_trans)
                        ctx_mask_conf_by_view = ctx_mask_vi.reshape(B, context_num, 1, 8, 8)
                        ctx_conf_vi_by_view = ctx_mask_conf_by_view * ctx_pose_conf_bt[:, context_idx_list].reshape(
                            B, context_num, 1, 1, 1
                        )

                        target_roi_feat_by_view = features.reshape(B, target_num, *features.shape[1:])
                        context_roi_feat_by_view = context_roi_feat.reshape(B, context_num, *context_roi_feat.shape[1:])
                        ctx_xyz_roi_by_view = ctx_xyz_roi.reshape(B, context_num, *ctx_xyz_roi.shape[1:])
                        ctx_view_dir_obj_by_view = ctx_view_dir_obj.reshape(B, context_num, *ctx_view_dir_obj.shape[1:])
                        ctx_cam_center_obj_by_view = ctx_cam_center_obj.reshape(B, context_num, *ctx_cam_center_obj.shape[1:])
                        context_imgs_vi_by_view = context_imgs_vi.reshape(B, context_num, *context_imgs_vi.shape[1:])
                        ctx_coord_vi_by_view = ctx_coord_vi.reshape(B, context_num, *ctx_coord_vi.shape[1:])
                        target_imgs_vi_by_view = target_imgs_vi.reshape(B, target_num, *target_imgs_vi.shape[1:])
                        tgt_coord_vi_by_view = tgt_coord_vi.reshape(B, target_num, *tgt_coord_vi.shape[1:])
                        tgt_mask_vi_by_view = tgt_mask_vi.reshape(B, target_num, *tgt_mask_vi.shape[1:])

                        geo_inputs = ContextGeometricPriorInputs(
                            target_roi_feat=target_roi_feat_by_view,
                            context_roi_feat=context_roi_feat_by_view,
                            context_xyz=ctx_xyz_roi_by_view,
                            context_view_dir_obj=ctx_view_dir_obj_by_view,
                            context_cam_center_obj=ctx_cam_center_obj_by_view,
                            target_rgb=target_imgs_vi_by_view,
                            context_rgb=context_imgs_vi_by_view,
                            target_coord=tgt_coord_vi_by_view,
                            context_coord=ctx_coord_vi_by_view,
                            target_mask=tgt_mask_vi_by_view,
                            context_mask=ctx_conf_vi_by_view,
                            target_valid=torch.ones(B, target_num, dtype=torch.bool, device=features.device),
                            context_valid=(ctx_pose_conf_bt[:, context_idx_list] > 0),
                        )
                        geo_prior_out = self.context_geo_prior(geo_inputs)
                        if should_inject_geo_prior_features(geo_prior_out):
                            retrieved_feat_8 = geo_prior_out["retrieved_feat_8"]
                            # Inference-time robustification: gate retrieved context feature
                            # by prior confidence to avoid injecting noisy memory when
                            # geometry prior is uncertain.
                            retrieved_gate_8 = torch.ones(
                                B, target_num, 1, retrieved_feat_8.shape[-2], retrieved_feat_8.shape[-1],
                                device=retrieved_feat_8.device,
                                dtype=retrieved_feat_8.dtype,
                            )
                            prior_conf_64 = geo_prior_out.get("prior_conf_64", None)
                            if prior_conf_64 is not None:
                                retrieved_gate_8 = F.adaptive_avg_pool2d(
                                    prior_conf_64.flatten(0, 1),
                                    output_size=retrieved_feat_8.shape[-1],
                                ).reshape(B, target_num, 1, retrieved_feat_8.shape[-2], retrieved_feat_8.shape[-1])
                            features = features + (retrieved_feat_8 * retrieved_gate_8).flatten(0, 1)
                            geo_prior_out["retrieved_gate_8"] = retrieved_gate_8
                        geo_prior_out_flat = {
                            "xyz_prior_64": geo_prior_out["xyz_prior_64"].flatten(0, 1),
                            "prior_conf_64": geo_prior_out["prior_conf_64"].flatten(0, 1),
                            "prior_ambiguity_64": geo_prior_out["prior_ambiguity_64"].flatten(0, 1),
                        }
        # ================================================================

        mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features)
        bs = features.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES
        device = features.device

        roi_cams_target = roi_cams[:, target_ids] if roi_cams.dim() >= 4 else roi_cams
        roi_extents_target = roi_extents[:, target_ids] if roi_extents.dim() >= 3 else roi_extents

        roi_coord_2d_pnp = _flatten_batch_view_tensor(roi_coord_2d)
        roi_extents_pnp = _flatten_batch_view_tensor(roi_extents_target)
        roi_cams_pnp = _flatten_batch_view_tensor(roi_cams_target)
        roi_centers_pnp = _flatten_batch_view_tensor(roi_centers)
        resize_ratios_pnp = _flatten_batch_view_tensor(resize_ratios)
        roi_whs_pnp = _flatten_batch_view_tensor(roi_whs)

        roi_classes_head = roi_classes
        if roi_classes_head is not None and roi_classes_head.shape[0] != bs:
            if roi_classes_head.shape[0] == B:
                roi_classes_head = roi_classes_head.repeat_interleave(target_num, dim=0)
            else:
                raise ValueError(f"roi_classes shape mismatch: roi_classes={roi_classes_head.shape[0]}, bs={bs}")

        if r_head_cfg.ROT_CLASS_AWARE:
            assert roi_classes_head is not None
            coor_x = coor_x.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs, device=device), roi_classes_head]
            coor_y = coor_y.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs, device=device), roi_classes_head]
            coor_z = coor_z.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs, device=device), roi_classes_head]

        if r_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes_head is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs, device=device), roi_classes_head]

        if r_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes_head is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs, device=device), roi_classes_head]

        save_pnp_internals = bool(cfg.TEST.get("SAVE_PNP_INTERNALS", False))

        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
            xyz_single = decode_xyz_expectation_from_logits(coor_x[:, :-1, :, :], coor_y[:, :-1, :, :], coor_z[:, :-1, :, :])
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)
            xyz_single = decode_xyz_expectation_from_logits(coor_x, coor_y, coor_z)

        if self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True) and geo_prior_out_flat is None:
            geo_prior_out = {
                "xyz_prior_64": torch.zeros(B, target_num, 3, out_res, out_res, device=features.device, dtype=features.dtype),
                "prior_conf_64": torch.zeros(B, target_num, 1, out_res, out_res, device=features.device, dtype=features.dtype),
                "prior_ambiguity_64": torch.zeros(B, target_num, 1, out_res, out_res, device=features.device, dtype=features.dtype),
            }
            geo_prior_out_flat = {
                "xyz_prior_64": geo_prior_out["xyz_prior_64"].flatten(0, 1),
                "prior_conf_64": geo_prior_out["prior_conf_64"].flatten(0, 1),
                "prior_ambiguity_64": geo_prior_out["prior_ambiguity_64"].flatten(0, 1),
            }

        coor_feat_base = coor_feat
        xyz_single_for_pnp = xyz_single
        prior_xyz_for_pnp = None
        prior_conf_for_pnp = None
        prior_residual_for_pnp = None
        if self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True) and geo_prior_out_flat is not None:
            prior_xyz_for_pnp = geo_prior_out_flat["xyz_prior_64"]
            if gp_cfg.get("APPEND_PRIOR_CONF", True):
                prior_conf_for_pnp = geo_prior_out_flat["prior_conf_64"]
            if gp_cfg.get("APPEND_PRIOR_RESIDUAL", True):
                prior_residual_for_pnp = xyz_single_for_pnp - prior_xyz_for_pnp

        coor_feat = assemble_pnp_inputs_with_geo_prior(
            coor_feat=coor_feat,
            roi_coord_2d_pnp=roi_coord_2d_pnp,
            xyz_single=xyz_single,
            geo_prior=geo_prior_out_flat if (self.geo_prior_enabled and gp_cfg.get("INJECT_TO_PNP", True)) else None,
            with_2d_coord=pnp_net_cfg.WITH_2D_COORD,
            append_prior_conf=gp_cfg.get("APPEND_PRIOR_CONF", True),
            append_prior_residual=gp_cfg.get("APPEND_PRIOR_RESIDUAL", True),
        )

        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)
        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            if external_target_masks_target is not None:
                ext_mask_roi = crop_resize_features(
                    observed_target_mask.flatten(0, 1),
                    roi_centers.reshape(B * target_num, 2),
                    scales.reshape(B * target_num, 1),
                    output_size=out_res,
                )
                mask_atten = ext_mask_roi.clamp(0.0, 1.0)
            else:
                mask_atten = get_mask_prob(cfg, mask)
        region_atten = region_softmax if pnp_net_cfg.REGION_ATTENTION else None
        pred_rot_, pred_t_ = self.pnp_net(coor_feat, region=region_atten, extents=roi_extents_pnp, mask_attention=mask_atten)
        if pnp_net_cfg.R_ONLY:
            pred_t_ = self.trans_head_net(features)

        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = self._rot_repr_to_mat(pred_rot_, rot_type)

        pred_ego_rot, pred_trans = decode_pose_from_raw(
            pnp_net_cfg,
            pred_rot_m,
            pred_t_,
            roi_cams=roi_cams_pnp,
            roi_centers=roi_centers_pnp,
            resize_ratios=resize_ratios_pnp,
            roi_whs=roi_whs_pnp,
            eps=1e-4,
            is_train=False,
        )

        # Use context-last anchor when available; otherwise keep no-context fallback.
        if has_context_for_target:
            context_rot_bt, context_trans_bt = self._get_context_pose_from_full_gt(
                gt_ego_rot,
                gt_trans,
                batch_size=B,
                context_last_idx=context_last_idx,
                dtype=pred_rot_.dtype,
                device=pred_rot_.device,
            )
        else:
            context_rot_bt = torch.eye(3, device=pred_rot_.device, dtype=pred_rot_.dtype).unsqueeze(0).repeat(B, 1, 1)
            context_trans_bt = torch.zeros(B, 3, device=pred_rot_.device, dtype=pred_rot_.dtype)
        context_rot_flat = context_rot_bt.repeat_interleave(target_num, dim=0).contiguous()
        context_trans_flat = context_trans_bt.repeat_interleave(target_num, dim=0).contiguous()
        out_dict = {
            "rot": pred_ego_rot,
            "trans": pred_trans,
            "abs_rot": pred_ego_rot,
            "abs_trans": pred_trans,
            "pnp_pred_rot_raw": pred_rot_,
            "pnp_pred_t_raw": pred_t_,
            "context_rot": context_rot_flat,
            "context_trans": context_trans_flat,
            "target_view_depth": target_view_depth,
            "target_view_mask": observed_target_mask,
            "target_view_mask_clean": observed_target_mask,
            "target_view_mask_raw": target_view_mask,
            "infer_roi_cams": roi_cams,
            "infer_roi_centers": roi_centers,
            "infer_scales": scales,
            "infer_roi_whs": roi_whs,
            "infer_resize_ratios": resize_ratios,
        }
        if geo_prior_out_flat is not None:
            out_dict["xyz_prior_64"] = geo_prior_out_flat["xyz_prior_64"]
            out_dict["prior_conf_64"] = geo_prior_out_flat["prior_conf_64"]
            out_dict["prior_ambiguity_64"] = geo_prior_out_flat["prior_ambiguity_64"]
            out_dict["prior_conf_64_mean"] = geo_prior_out_flat["prior_conf_64"].mean()
            out_dict["prior_ambiguity_64_mean"] = geo_prior_out_flat["prior_ambiguity_64"].mean()
            out_dict["geo_prior_all_target_fallback"] = bool(is_all_target)
            if geo_prior_out is not None:
                bank_conf = geo_prior_out.get("bank_conf", None)
                if bank_conf is not None:
                    out_dict["geo_prior_bank_valid_ratio"] = (bank_conf > 0).to(dtype=bank_conf.dtype).mean()
                    out_dict["geo_prior_bank_conf_mean"] = bank_conf.mean()
                bank_weights = geo_prior_out.get("bank_weights_8", None)
                if bank_weights is not None:
                    out_dict["geo_prior_bank_weight_max_mean"] = bank_weights.max(dim=-1).values.mean()
                retrieved_gate = geo_prior_out.get("retrieved_gate_8", None)
                if retrieved_gate is not None:
                    out_dict["geo_prior_retrieved_gate_mean"] = retrieved_gate.mean()
        if cfg.TEST.USE_PNP:
            out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region})
        if save_pnp_internals:
            out_dict["dbg_pnp_coor_feat_base"] = coor_feat_base
            out_dict["dbg_pnp_coor_feat_final"] = coor_feat
            out_dict["dbg_pnp_xyz_single"] = xyz_single_for_pnp
            if pnp_net_cfg.WITH_2D_COORD:
                out_dict["dbg_pnp_roi_coord_2d"] = roi_coord_2d_pnp
            out_dict["dbg_pnp_region_softmax"] = region_softmax
            if region_atten is not None:
                out_dict["dbg_pnp_region_atten"] = region_atten
            if mask_atten is not None:
                out_dict["dbg_pnp_mask_atten"] = mask_atten
            if prior_xyz_for_pnp is not None:
                out_dict["dbg_pnp_prior_xyz"] = prior_xyz_for_pnp
            if prior_conf_for_pnp is not None:
                out_dict["dbg_pnp_prior_conf"] = prior_conf_for_pnp
            if prior_residual_for_pnp is not None:
                out_dict["dbg_pnp_prior_residual"] = prior_residual_for_pnp
        return out_dict

    def depth_mask_loss_original(self, cfg, depth_pred, mask_pred, depth_gt, mask_gt=None):
        """Depth/mask losses on target views with safe masked reduction."""
        eps = 1e-8
        loss_cfg = cfg.MODEL.CDPN.get("DEPTH_HEAD_LOSS", {})
        lambda_bce = float(loss_cfg.get("LAMBDA_BCE", 1.0))
        lambda_dice = float(loss_cfg.get("LAMBDA_DICE", 1.0))
        lambda_dp_reg = float(loss_cfg.get("LAMBDA_DP_REG", 1.0))
        lambda_dp_gd = float(loss_cfg.get("LAMBDA_DP_GD", 1.0))

        if depth_gt.dim() == 4:
            depth_gt = depth_gt.unsqueeze(2)
        if mask_gt is not None and mask_gt.dim() == 4:
            mask_gt = mask_gt.unsqueeze(2)

        depth_gt = depth_gt.to(device=depth_pred.device, dtype=depth_pred.dtype)
        if mask_gt is None:
            mask_gt = (depth_gt > 0).to(dtype=depth_pred.dtype)
        else:
            mask_gt = mask_gt.to(device=depth_pred.device, dtype=depth_pred.dtype)

        if depth_pred.shape != depth_gt.shape:
            raise ValueError(f"depth_pred/depth_gt shape mismatch: {depth_pred.shape} vs {depth_gt.shape}")

        B, T, _, H, W = depth_pred.shape
        depth_pred_flat = depth_pred.reshape(B * T, 1, H, W).contiguous()
        depth_gt_flat = depth_gt.reshape(B * T, 1, H, W).contiguous()
        mask_pred_flat = mask_pred.reshape(B * T, 1, H, W).contiguous()
        mask_gt_flat = mask_gt.reshape(B * T, 1, H, W).contiguous().clamp(0.0, 1.0)
        mask_act = str(cfg.MODEL.CDPN.get("DEPTH_HEAD_MASK_ACT", "none")).lower()

        # A. object mask loss = BCE + Dice
        if mask_act == "sigmoid":
            mask_prob_flat = mask_pred_flat.clamp(min=eps, max=1.0 - eps)
            bce_loss = F.binary_cross_entropy(mask_prob_flat, mask_gt_flat, reduction="mean")
        else:
            bce_loss = F.binary_cross_entropy_with_logits(mask_pred_flat, mask_gt_flat, reduction="mean")
            mask_prob_flat = torch.sigmoid(mask_pred_flat)
        inter = (mask_prob_flat * mask_gt_flat).sum(dim=(1, 2, 3))
        denom = mask_prob_flat.sum(dim=(1, 2, 3)) + mask_gt_flat.sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()
        
        # 统一把 lambda 乘法放到最后 return 时做，代码更整洁
        loss_obj_mask = lambda_bce * bce_loss + lambda_dice * dice_loss

        # B1. masked Smooth L1 depth regression loss (保持不变)
        valid_mask = (mask_gt_flat > 0.5).to(dtype=depth_pred_flat.dtype)
        depth_reg = F.smooth_l1_loss(depth_pred_flat, depth_gt_flat, reduction="none")
        loss_dp_reg = (depth_reg * valid_mask).sum() / (valid_mask.sum() + eps)

        # -----------------------------------------------------------------
        # B2. masked gradient loss 使用安全的张量切片 (Finite Differences)
        # 彻底弃用 F.conv2d 和 F.max_pool2d，规避 cuDNN 底层崩溃
        # -----------------------------------------------------------------
        
        # 1. 计算 X 方向的梯度 (右边像素减去当前像素)
        pred_dx = depth_pred_flat[:, :, :, 1:] - depth_pred_flat[:, :, :, :-1]
        gt_dx = depth_gt_flat[:, :, :, 1:] - depth_gt_flat[:, :, :, :-1]
        grad_diff_x = torch.abs(pred_dx - gt_dx)

        # 2. 计算 Y 方向的梯度 (下方像素减去当前像素)
        pred_dy = depth_pred_flat[:, :, 1:, :] - depth_pred_flat[:, :, :-1, :]
        gt_dy = depth_gt_flat[:, :, 1:, :] - depth_gt_flat[:, :, :-1, :]
        grad_diff_y = torch.abs(pred_dy - gt_dy)

        # 3. 计算 X 和 Y 方向的有效 Mask
        # 只有当相邻的两个像素都在 Mask 内部时，这个梯度计算才是有效的（天然的边缘腐蚀效果）
        valid_mask_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        valid_mask_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]

        # 4. 安全求均值
        loss_dx = (grad_diff_x * valid_mask_x).sum() / (valid_mask_x.sum() + eps)
        loss_dy = (grad_diff_y * valid_mask_y).sum() / (valid_mask_y.sum() + eps)
        loss_dp_gd = loss_dx + loss_dy

        return {
            "loss_obj_mask": loss_obj_mask,
            "loss_dp_reg": loss_dp_reg * lambda_dp_reg,
            "loss_dp_gd": loss_dp_gd * lambda_dp_gd,
        }


    def depth_mask_loss(
        self,
        cfg,
        depth_pred,
        mask_pred,
        depth_gt,
        mask_gt=None,
        roi_cams=None,
        pred_rot=None,
        pred_trans=None,
        gt_rot=None,
        gt_trans=None,
    ):
        """融合了 3D 物理距离 (监监督深度) 和 Dense Flow 重投影 (监督位姿) 的终极 Loss"""
        eps = 1e-6 
        loss_cfg = cfg.MODEL.CDPN.get("DEPTH_HEAD_LOSS", {})
        lambda_bce = float(loss_cfg.get("LAMBDA_BCE", 1.0))
        lambda_dice = float(loss_cfg.get("LAMBDA_DICE", 1.0))
        lambda_dp_reg = float(loss_cfg.get("LAMBDA_DP_REG", 1.0))
        lambda_dp_gd = float(loss_cfg.get("LAMBDA_DP_GD", 1.0))
        lambda_dp_3d = float(loss_cfg.get("LAMBDA_DP_3D", 1.0))       # 3D 点云深度监督权重
        lambda_pose_flow = float(loss_cfg.get("LAMBDA_REPROJ", 0.0))  # 稠密光流位姿监督权重

        if depth_gt.dim() == 4:
            depth_gt = depth_gt.unsqueeze(2)
        if mask_gt is not None and mask_gt.dim() == 4:
            mask_gt = mask_gt.unsqueeze(2)

        depth_gt = depth_gt.to(device=depth_pred.device, dtype=depth_pred.dtype)
        if mask_gt is None:
            mask_gt = (depth_gt > 0).to(dtype=depth_pred.dtype)
        else:
            mask_gt = mask_gt.to(device=depth_pred.device, dtype=depth_pred.dtype)

        B, T, _, H, W = depth_pred.shape
        depth_pred_flat = depth_pred.reshape(B * T, 1, H, W).contiguous()
        depth_gt_flat = depth_gt.reshape(B * T, 1, H, W).contiguous()
        mask_pred_flat = mask_pred.reshape(B * T, 1, H, W).contiguous()
        mask_gt_flat = mask_gt.reshape(B * T, 1, H, W).contiguous().clamp(0.0, 1.0)
        mask_act = str(cfg.MODEL.CDPN.get("DEPTH_HEAD_MASK_ACT", "none")).lower()

        # ==========================================
        # A. Object Mask Loss (BCE + Dice)
        # ==========================================
        if mask_act == "sigmoid":
            mask_prob_flat = mask_pred_flat.clamp(min=eps, max=1.0 - eps)
            bce_loss = F.binary_cross_entropy(mask_prob_flat, mask_gt_flat, reduction="mean")
        else:
            bce_loss = F.binary_cross_entropy_with_logits(mask_pred_flat, mask_gt_flat, reduction="mean")
            mask_prob_flat = torch.sigmoid(mask_pred_flat)
            
        inter = (mask_prob_flat * mask_gt_flat).sum(dim=(1, 2, 3))
        denom = mask_prob_flat.sum(dim=(1, 2, 3)) + mask_gt_flat.sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()
        loss_obj_mask = lambda_bce * bce_loss + lambda_dice * dice_loss
        del mask_prob_flat 

        valid_mask_bool = mask_gt_flat > 0.5 
        valid_mask_float = valid_mask_bool.to(depth_pred_flat.dtype)

        # ==========================================
        # B1 & B2. 2D Depth Regression & Gradient
        # ==========================================
        depth_reg = F.smooth_l1_loss(depth_pred_flat, depth_gt_flat, reduction="none")
        loss_dp_reg = (depth_reg * valid_mask_float).sum() / (valid_mask_float.sum() + eps)

        valid_mask_x_float = (valid_mask_float[:, :, :, 1:] * valid_mask_float[:, :, :, :-1])
        pred_dx = depth_pred_flat[:, :, :, 1:] - depth_pred_flat[:, :, :, :-1]
        gt_dx = depth_gt_flat[:, :, :, 1:] - depth_gt_flat[:, :, :, :-1]
        loss_dx = (torch.abs(pred_dx - gt_dx) * valid_mask_x_float).sum() / (valid_mask_x_float.sum() + eps)
            
        valid_mask_y_float = (valid_mask_float[:, :, 1:, :] * valid_mask_float[:, :, :-1, :])
        pred_dy = depth_pred_flat[:, :, 1:, :] - depth_pred_flat[:, :, :-1, :]
        gt_dy = depth_gt_flat[:, :, 1:, :] - depth_gt_flat[:, :, :-1, :]
        loss_dy = (torch.abs(pred_dy - gt_dy) * valid_mask_y_float).sum() / (valid_mask_y_float.sum() + eps)
        loss_dp_gd = loss_dx + loss_dy

        # ==========================================
        # 准备相机内参与扁平化辅助函数
        # ==========================================
        def _flatten_bt_tensor(tensor):
            if tensor is None or not torch.is_tensor(tensor):
                return None
            if tensor.dim() >= 3 and tensor.shape[0] == B and tensor.shape[1] == T:
                return tensor.reshape(B * T, *tensor.shape[2:]).contiguous()
            return tensor

        roi_cams_flat = _flatten_bt_tensor(roi_cams)
        loss_dp_3d = depth_pred_flat.sum() * 0.0
        loss_pose_flow = depth_pred_flat.sum() * 0.0

        # ==========================================
        # C1. 3D Point Cloud Loss (专职监督 Depth)
        # 利用全局网格并行计算，极速算出 3D 物理误差
        # ==========================================
        if lambda_dp_3d > 0 and roi_cams_flat is not None and roi_cams_flat.shape[0] == B * T:
            ys, xs = torch.meshgrid(
                torch.arange(H, device=depth_pred_flat.device, dtype=depth_pred_flat.dtype),
                torch.arange(W, device=depth_pred_flat.device, dtype=depth_pred_flat.dtype),
                indexing="ij"
            )
            u = xs.view(1, 1, H, W).expand(B * T, -1, -1, -1)
            v = ys.view(1, 1, H, W).expand(B * T, -1, -1, -1)

            fx = roi_cams_flat[:, 0, 0].view(-1, 1, 1, 1)
            fy = roi_cams_flat[:, 1, 1].view(-1, 1, 1, 1)
            cx = roi_cams_flat[:, 0, 2].view(-1, 1, 1, 1)
            cy = roi_cams_flat[:, 1, 2].view(-1, 1, 1, 1)

            x_pred = (u - cx) * depth_pred_flat / fx
            y_pred = (v - cy) * depth_pred_flat / fy
            pts_3d_pred = torch.cat([x_pred, y_pred, depth_pred_flat], dim=1)
            
            x_gt = (u - cx) * depth_gt_flat / fx
            y_gt = (v - cy) * depth_gt_flat / fy
            pts_3d_gt = torch.cat([x_gt, y_gt, depth_gt_flat], dim=1)

            pts_diff = torch.abs(pts_3d_pred - pts_3d_gt)
            valid_sum = valid_mask_float.sum()
            if valid_sum > 0:
                loss_dp_3d = (pts_diff * valid_mask_float).sum() / (valid_sum * 3 + eps)

        # ==========================================
        # C2. Dense Flow Reprojection Loss (专职监督 Pose)
        # 你的高效采样循环，我加入了 .detach() 截断梯度
        # ==========================================
        pred_rot_flat = _flatten_bt_tensor(pred_rot)
        pred_trans_flat = _flatten_bt_tensor(pred_trans)
        gt_rot_flat = _flatten_bt_tensor(gt_rot)
        gt_trans_flat = _flatten_bt_tensor(gt_trans)

        has_pose = (pred_rot_flat is not None and pred_trans_flat is not None and 
                    gt_rot_flat is not None and gt_trans_flat is not None)

        if (
            lambda_pose_flow > 0
            and roi_cams_flat is not None
            and roi_cams_flat.shape[0] == B * T
            and has_pose
            and pred_rot_flat.shape[0] == B * T
            and pred_trans_flat.shape[0] == B * T
            and gt_rot_flat.shape[0] == B * T
            and gt_trans_flat.shape[0] == B * T
        ):
            n = depth_pred_flat.shape[0]
            max_points = int(loss_cfg.get("REPROJ_MAX_POINTS", 4096))

            ys, xs = torch.meshgrid(
                torch.arange(H, device=depth_pred_flat.device, dtype=depth_pred_flat.dtype),
                torch.arange(W, device=depth_pred_flat.device, dtype=depth_pred_flat.dtype),
                indexing="ij",
            )
            uv_grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
            reproj_loss_list = []

            for i in range(n):
                valid_i = valid_mask_bool[i, 0].reshape(-1) & (depth_gt_flat[i, 0].reshape(-1) > eps)
                valid_ids = torch.nonzero(valid_i, as_tuple=False).squeeze(1)
                
                if valid_ids.numel() == 0:
                    continue
                if max_points > 0 and valid_ids.numel() > max_points:
                    sel = torch.randperm(valid_ids.numel(), device=valid_ids.device)[:max_points]
                    valid_ids = valid_ids[sel]

                K = roi_cams_flat[i].to(dtype=depth_pred_flat.dtype)
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                uv_src = uv_grid[valid_ids]
                
                # 🌟 关键防爆点：这里必须 detach，让深度梯度留给 C1 优化，光流专心优化姿态！
                z_src = depth_pred_flat[i, 0].reshape(-1)[valid_ids].detach() 
                
                x_src = (uv_src[:, 0] - cx) * z_src / fx.clamp(min=eps)
                y_src = (uv_src[:, 1] - cy) * z_src / fy.clamp(min=eps)
                p_cam_src = torch.stack([x_src, y_src, z_src], dim=-1)

                R_pred, t_pred = pred_rot_flat[i], pred_trans_flat[i]
                R_gt, t_gt = gt_rot_flat[i], gt_trans_flat[i]

                # 相对位姿变换: p_tgt = R_gt * R_pred^T * (p_src - t_pred) + t_gt
                A = torch.matmul(R_gt, R_pred.transpose(0, 1))
                b = t_gt - torch.matmul(A, t_pred)
                p_cam_tgt = torch.matmul(p_cam_src, A.transpose(0, 1)) + b[None, :]

                z_tgt = p_cam_tgt[:, 2].clamp(min=0.1)
                uv_tgt_x = fx * p_cam_tgt[:, 0] / z_tgt + cx
                uv_tgt_y = fy * p_cam_tgt[:, 1] / z_tgt + cy
                uv_tgt = torch.stack([uv_tgt_x, uv_tgt_y], dim=-1)

                # 剔除那些被错误姿态甩到画面几万像素之外的离谱点
                # 假设画面尺寸扩大 2 倍作为合理容忍边界
                valid_uv_mask = (
                    (uv_tgt[:, 0] >= -W) & (uv_tgt[:, 0] <= 2 * W) &
                    (uv_tgt[:, 1] >= -H) & (uv_tgt[:, 1] <= 2 * H)
                )

                # 如果所有点都飞出去了，这步直接跳过，算 0 梯度
                if not valid_uv_mask.any():
                    continue
                    
                uv_tgt_safe = uv_tgt[valid_uv_mask]
                uv_src_safe = uv_src[valid_uv_mask]

                # 在算 Loss 时直接套用 Huber 惩罚机制 (Smooth L1 的本质)
                # 显式设定 beta (1.0 即可)，当像素误差大于 1 时，梯度恒定为 1，拒绝梯度爆炸
                safe_loss = F.smooth_l1_loss(uv_tgt_safe, uv_src_safe, reduction="mean", beta=1.0)
                reproj_loss_list.append(safe_loss)

            if len(reproj_loss_list) > 0:
                loss_pose_flow = torch.stack(reproj_loss_list, dim=0).mean()

        return {
            "loss_obj_mask": loss_obj_mask,
            "loss_dp_reg": loss_dp_reg * lambda_dp_reg,
            "loss_dp_gd": loss_dp_gd * lambda_dp_gd,
            "loss_dp_3d": loss_dp_3d * lambda_dp_3d,          # 深度获得了物理尺度
            "loss_pose_flow": loss_pose_flow * lambda_pose_flow, # 姿态获得了稠密对齐
        }



    def gdrn_loss(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        model_infos=None,
        extents=None,
        target_idx_list=None,
        full_gt_rot=None,
        full_gt_trans=None,
        gt_rot_anchor_for_sym=None,
    ):
        profile_total_t0 = self._hotspot_profile_start()
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        loss_dict = {}

        gt_rot_supervise = gt_rot
        gt_trans_supervise = gt_trans
        expected_pose_n = int(out_rot.shape[0]) if torch.is_tensor(out_rot) else 0
        abs_sym_loop_t0 = self._hotspot_profile_start()
        if torch.is_tensor(out_rot) and torch.is_tensor(gt_rot):
            gt_trans_flat = gt_trans.reshape(-1, 3).contiguous() if torch.is_tensor(gt_trans) and gt_trans.dim() == 3 else gt_trans
            gt_rot_supervise, gt_trans_supervise = self._canonicalize_abs_pose_targets(
                cfg=cfg,
                gt_rot=gt_rot,
                gt_trans=gt_trans_flat,
                model_infos=model_infos,
                pred_rots_for_canon=out_rot.detach(),
            )
        self._hotspot_profile_stop("gdrn_loss_abs_sym_loop", abs_sym_loop_t0, pose_n=expected_pose_n)

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}
        gt_mask_xyz = gt_masks.get(r_head_cfg.XYZ_LOSS_MASK_GT, None)
        xyz_bin_num = int(r_head_cfg.XYZ_BIN) if "CE" in str(r_head_cfg.XYZ_LOSS_TYPE) else None
        gt_xyz_supervise, gt_xyz_bin_supervise = self._align_xyz_targets_to_pose_supervision(
            gt_xyz=gt_xyz,
            gt_xyz_bin=gt_xyz_bin,
            gt_mask_xyz=gt_mask_xyz,
            gt_rot_raw=gt_rot,
            gt_rot_supervise=gt_rot_supervise,
            model_infos_batch=model_infos,
            extents=extents,
            xyz_bin_num=xyz_bin_num,
        )
        gt_region_supervise = self._recompute_region_targets_from_aligned_xyz(
            gt_xyz_supervise=gt_xyz_supervise,
            gt_region=gt_region,
            gt_mask_region=gt_masks.get(r_head_cfg.REGION_LOSS_MASK_GT, None),
            model_infos_batch=model_infos,
            extents=extents,
            num_regions=int(r_head_cfg.NUM_REGIONS),
        )
        loss_dict.update(compute_xyz_losses(r_head_cfg, out_x, out_y, out_z, gt_xyz_supervise, gt_xyz_bin_supervise, gt_masks))
        loss_dict.update(compute_mask_loss(r_head_cfg, out_mask, gt_masks))
        loss_dict.update(compute_region_loss(r_head_cfg, out_region, gt_region_supervise, gt_masks))

        # point matching loss ---------------
        if pnp_net_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            pm_prep_t0 = self._hotspot_profile_start()
            pm_gt_rot, pm_gt_trans, pm_sym_infos = self._prepare_pm_supervision_targets(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                gt_transes=gt_trans,
                model_infos_batch=model_infos,
                fallback_sym_infos=sym_infos,
            )
            self._hotspot_profile_stop("gdrn_loss_pm_prep", pm_prep_t0, pose_n=expected_pose_n)
            loss_func = build_pm_loss(pnp_net_cfg)
            pm_loss_t0 = self._hotspot_profile_start()
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=pm_gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=pm_gt_trans,
                extents=extents,
                sym_infos=pm_sym_infos,
            )
            point_n = int(gt_points.shape[1]) if torch.is_tensor(gt_points) and gt_points.dim() >= 2 else 0
            self._hotspot_profile_stop("gdrn_loss_pm_loss", pm_loss_t0, pose_n=expected_pose_n, point_n=point_n)
            loss_dict.update(loss_pm_dict)

        if pnp_net_cfg.ROT_LW > 0:
            loss_dict["loss_rot"] = compute_rotation_loss(pnp_net_cfg, out_rot, gt_rot_supervise)

        if pnp_net_cfg.CENTROID_LW > 0 or (pnp_net_cfg.Z_LW > 0 and str(pnp_net_cfg.Z_TYPE).upper() == "REL"):
            # Centroid (gt_trans_ratio[:, :2]) and REL-mode Z (gt_trans_ratio[:, 2]) are
            # pre-computed from raw gt_trans and cannot be re-derived here without camera/ROI
            # params.  They are therefore NOT canonicalized when symmetry changes translation.
            # This is exact only when the continuous-symmetry axis passes through the object
            # origin (offset == 0).  Raise an error if a non-zero offset is detected.
            _flat_mi = self._flatten_model_infos_for_pose_count(model_infos, int(out_rot.shape[0])) if torch.is_tensor(out_rot) else None
            if _flat_mi is not None:
                for _mi in _flat_mi:
                    _, _offset = self._safe_get_continuous_sym_axis_offset(_mi, out_rot.device, out_rot.dtype)
                    if _offset is not None and float(_offset.norm()) > 1e-6:
                        raise NotImplementedError(
                            "loss_centroid / loss_z (REL) are not canonicalized for continuous symmetry "
                            "with non-zero offset. Either set CENTROID_LW=0 and use Z_TYPE='ABS', or "
                            "ensure all symmetry axes pass through the object origin (offset=0)."
                        )

        if pnp_net_cfg.CENTROID_LW > 0:
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"
            # gt_trans_ratio is pre-computed from raw gt_trans; recomputing it from
            # gt_trans_supervise would require camera/ROI parameters not available here.
            # This is consistent when symmetry offset=0 (the common case).
            loss_dict["loss_centroid"] = compute_centroid_loss(pnp_net_cfg, out_centroid, gt_trans_ratio)

        if pnp_net_cfg.Z_LW > 0:
            # When Z_TYPE="ABS", gt_trans_supervise[:, 2] is used (canonicalized).
            # When Z_TYPE="REL", compute_z_loss reads gt_trans_ratio[:, 2] and ignores
            # gt_trans, so canonicalization is not applied (no camera params available here).
            loss_dict["loss_z"] = compute_z_loss(pnp_net_cfg, out_trans_z, gt_trans_supervise, gt_trans_ratio)

        if pnp_net_cfg.TRANS_LW > 0:
            loss_dict.update(compute_translation_losses(pnp_net_cfg, out_trans, gt_trans_supervise))

        if pnp_net_cfg.get("BIND_LW", 0.0) > 0.0:
            loss_dict["loss_bind"] = compute_bind_loss(pnp_net_cfg, out_rot, out_trans, gt_rot_supervise, gt_trans_supervise)

        if cfg.MODEL.CDPN.USE_MTL:
            apply_mtl_uncertainty(loss_dict, self)
        self._hotspot_profile_stop("gdrn_loss_total", profile_total_t0, pose_n=expected_pose_n)
        return loss_dict

    def _compute_pose_losses_only(
        self,
        cfg,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        out_trans_for_direct_loss=None,
        gt_trans=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
        model_infos=None,
    ):
        """Pose-only supervision for the abs-head."""
        profile_t0 = self._hotspot_profile_start()
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        loss_dict = {}
        expected_pose_n = int(out_rot.shape[0]) if torch.is_tensor(out_rot) else -1

        if out_rot is None or gt_rot is None or out_trans is None or gt_trans is None:
            return loss_dict

        def _flatten_gt_batch_view(tensor):
            if tensor is None or (not torch.is_tensor(tensor)):
                return tensor
            # Keep already-aligned tensors as-is.
            if tensor.shape[0] == expected_pose_n:
                return tensor.contiguous()
            # Flatten [B, T, ...] tensors to [B*T, ...] when needed.
            if tensor.dim() >= 2 and tensor.shape[0] * tensor.shape[1] == expected_pose_n:
                return tensor.flatten(0, 1).contiguous()
            return tensor

        def _flatten_sym_infos(sym_infos_batch):
            if sym_infos_batch is None:
                return None
            flat_sym_infos = []
            for item in sym_infos_batch:
                if isinstance(item, (list, tuple)):
                    flat_sym_infos.extend(list(item))
                elif isinstance(item, np.ndarray) and item.dtype == object:
                    flat_sym_infos.extend(item.tolist())
                elif isinstance(item, np.ndarray) and item.ndim == 1:
                    flat_sym_infos.extend(item.tolist())
                else:
                    flat_sym_infos.append(item)
            return flat_sym_infos

        def _flatten_model_infos(model_infos_batch):
            if model_infos_batch is None:
                return None
            if not isinstance(model_infos_batch, (list, tuple)):
                return None
            if len(model_infos_batch) == expected_pose_n:
                return list(model_infos_batch)
            flat_model_infos = []
            for item in model_infos_batch:
                if isinstance(item, (list, tuple)):
                    flat_model_infos.extend(list(item))
                else:
                    flat_model_infos.append(item)
            if len(flat_model_infos) == expected_pose_n:
                return flat_model_infos
            # Common multiview case: one model_info per batch item, repeated across target views.
            if expected_pose_n > 0 and len(model_infos_batch) > 0 and expected_pose_n % len(model_infos_batch) == 0:
                repeat_factor = expected_pose_n // len(model_infos_batch)
                repeated = []
                for item in model_infos_batch:
                    repeated.extend([item] * repeat_factor)
                if len(repeated) == expected_pose_n:
                    return repeated
            return None

        gt_rot_flat = _flatten_gt_batch_view(gt_rot)
        gt_trans_flat = _flatten_gt_batch_view(gt_trans)
        out_trans_direct_flat = _flatten_gt_batch_view(out_trans_for_direct_loss)
        gt_points_flat = _flatten_gt_batch_view(gt_points)
        extents_flat = _flatten_gt_batch_view(extents)
        sym_infos_flat = _flatten_sym_infos(sym_infos)
        model_infos_flat = _flatten_model_infos(model_infos)
        if (not torch.is_tensor(out_trans_direct_flat)) or out_trans_direct_flat.shape[0] != expected_pose_n:
            out_trans_direct_flat = out_trans

        gt_rot_supervise, gt_trans_supervise = self._canonicalize_abs_pose_targets(
            cfg=cfg,
            gt_rot=gt_rot_flat,
            gt_trans=gt_trans_flat,
            model_infos=model_infos_flat,
            pred_rots_for_canon=out_rot,
        )

        if pnp_net_cfg.PM_LW > 0 and gt_points_flat is not None:
            pm_prep_t0 = self._hotspot_profile_start()
            pm_gt_rot, pm_gt_trans, pm_sym_infos = self._prepare_pm_supervision_targets(
                pred_rots=out_rot,
                gt_rots=gt_rot_flat,
                gt_transes=gt_trans_flat,
                model_infos_batch=model_infos_flat,
                fallback_sym_infos=sym_infos_flat,
            )
            self._hotspot_profile_stop("_compute_pose_losses_only_pm_prep", pm_prep_t0, pose_n=expected_pose_n)
            loss_func = build_pm_loss(pnp_net_cfg)
            pm_loss_t0 = self._hotspot_profile_start()
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=pm_gt_rot,
                points=gt_points_flat,
                pred_transes=out_trans,
                gt_transes=pm_gt_trans,
                extents=extents_flat,
                sym_infos=pm_sym_infos,
            )
            point_n = int(gt_points_flat.shape[1]) if torch.is_tensor(gt_points_flat) and gt_points_flat.dim() >= 2 else 0
            self._hotspot_profile_stop(
                "_compute_pose_losses_only_pm_loss",
                pm_loss_t0,
                pose_n=expected_pose_n,
                point_n=point_n,
            )
            loss_dict.update(loss_pm_dict)

        if pnp_net_cfg.ROT_LW > 0:
            loss_dict["loss_rot"] = compute_rotation_loss(pnp_net_cfg, out_rot, gt_rot_supervise)

        if pnp_net_cfg.TRANS_LW > 0:
            loss_dict.update(compute_translation_losses(pnp_net_cfg, out_trans_direct_flat, gt_trans_supervise))

        if pnp_net_cfg.get("BIND_LW", 0.0) > 0.0:
            loss_dict["loss_bind"] = compute_bind_loss(
                pnp_net_cfg,
                out_rot,
                out_trans,
                gt_rot_supervise,
                gt_trans_supervise,
            )

        if cfg.MODEL.CDPN.USE_MTL:
            apply_mtl_uncertainty(loss_dict, self)

        self._hotspot_profile_stop("_compute_pose_losses_only", profile_t0, pose_n=expected_pose_n)
        return loss_dict

    def gdrn_loss_multiview(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        model_infos=None,
        extents=None,
        target_idx_list=None,
        full_gt_rot=None,
        full_gt_trans=None,
        gt_rot_anchor_for_sym=None,
    ):
        """Compute losses for multiview inputs by flattening GT [B, V, ...] to [B*V, ...]."""
        profile_t0 = self._hotspot_profile_start()

        view_num = None
        for cand in [gt_xyz, gt_xyz_bin, gt_mask_trunc, gt_mask_visib, gt_mask_obj, gt_region, gt_trans]:
            if torch.is_tensor(cand) and cand.dim() >= 3:
                view_num = cand.shape[1]
                break

        def _flatten_gt_batch_view(tensor):
            if tensor is None or (not torch.is_tensor(tensor)):
                return tensor
            if view_num is None:
                return tensor
            # For GT tensors, the multiview layout is [B, V, ...].
            # Flatten only this pattern; keep already-flattened predictions untouched.
            if tensor.dim() >= 3 and tensor.shape[1] == view_num:
                return tensor.flatten(0, 1).contiguous()
            return tensor

        def _flatten_sym_infos(sym_infos_batch):
            if sym_infos_batch is None:
                return None
            flat_sym_infos = []
            for item in sym_infos_batch:
                if isinstance(item, (list, tuple)):
                    flat_sym_infos.extend(list(item))
                elif isinstance(item, np.ndarray) and item.dtype == object:
                    flat_sym_infos.extend(item.tolist())
                elif isinstance(item, np.ndarray) and item.ndim == 1:
                    flat_sym_infos.extend(item.tolist())
                else:
                    flat_sym_infos.append(item)
            return flat_sym_infos

        loss_dict = self.gdrn_loss(
            cfg=cfg,
            out_mask=out_mask,
            gt_mask_trunc=_flatten_gt_batch_view(gt_mask_trunc),
            gt_mask_visib=_flatten_gt_batch_view(gt_mask_visib),
            gt_mask_obj=_flatten_gt_batch_view(gt_mask_obj),
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            gt_xyz=_flatten_gt_batch_view(gt_xyz),
            gt_xyz_bin=_flatten_gt_batch_view(gt_xyz_bin),
            out_region=out_region,
            gt_region=_flatten_gt_batch_view(gt_region),
            out_rot=out_rot,
            gt_rot=_flatten_gt_batch_view(gt_rot),
            out_trans=out_trans,
            gt_trans=_flatten_gt_batch_view(gt_trans),
            out_centroid=out_centroid,
            out_trans_z=out_trans_z,
            gt_trans_ratio=_flatten_gt_batch_view(gt_trans_ratio),
            gt_points=_flatten_gt_batch_view(gt_points),
            sym_infos=_flatten_sym_infos(sym_infos),
            model_infos=model_infos,
            extents=_flatten_gt_batch_view(extents),
            target_idx_list=target_idx_list,
            full_gt_rot=full_gt_rot,
            full_gt_trans=full_gt_trans,
            gt_rot_anchor_for_sym=gt_rot_anchor_for_sym,
        )
        self._hotspot_profile_stop("gdrn_loss_multiview_total", profile_t0, view_num=view_num or 0)
        return loss_dict
    
    
    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        # Concatenate special tokens with patch tokens
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)


def crop_resize_features(features, centers, scales, output_size=256):
    """Crop and resize RoI regions from full-image feature maps (differentiable).

    Similar to `crop_resize_by_warp_affine` in data_utils.py, but operates on
    GPU tensors using `F.affine_grid` + `F.grid_sample` so that gradients can
    flow back through the cropped features.

    Args:
        features:    (B, C, H, W) feature tensor at full image resolution.
        centers:     (B, 2) RoI bbox centers in pixel coordinates (cx, cy).
        scales:      (B, 1) square bbox size in pixels (same value used as
                     both width and height, matching the data-loader convention).
        output_size: int, spatial size of the output (default 256 → B×C×256×256).

    Returns:
        roi_features: (B, C, output_size, output_size)
    """
    B, C, H, W = features.shape
    device = features.device
    dtype = features.dtype

    cx = centers[:, 0]  # (B,)
    cy = centers[:, 1]  # (B,)
    s = scales.to(dtype).squeeze(-1)  # (B, 1) → (B,)

    # Build a 2×3 affine matrix per sample that maps the output grid (in
    # normalised coordinates [-1, 1]) to the corresponding source window in
    # the input feature map.
    #
    # With align_corners=True the grid value -1 corresponds to pixel 0 and +1
    # to pixel (W-1) (or (H-1) for the y axis).
    #
    # We want:
    #   output -1  →  input pixel  cx - s/2   →  normalised  2*(cx - s/2)/(W-1) - 1
    #   output +1  →  input pixel  cx + s/2   →  normalised  2*(cx + s/2)/(W-1) - 1
    #
    # So the linear mapping  g_in = a * g_out + b  has:
    #   a_x = s / (W - 1),   b_x = 2*cx / (W - 1) - 1
    #   a_y = s / (H - 1),   b_y = 2*cy / (H - 1) - 1

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = s / (W - 1)                  # x scale
    theta[:, 1, 1] = s / (H - 1)                  # y scale
    theta[:, 0, 2] = 2.0 * cx / (W - 1) - 1.0    # x translation
    theta[:, 1, 2] = 2.0 * cy / (H - 1) - 1.0    # y translation

    grid = F.affine_grid(theta, (B, C, output_size, output_size), align_corners=True)
    roi_features = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return roi_features


def crop_resize_features_xywh(features, centers, whs, output_size=256):
    """Crop and resize rectangular RoI regions from full-image tensors.

    This matches the renderer convention where the ROI is parameterized by
    an image-space center plus width/height, instead of the square `scale` used
    by the pose head crop.
    """
    B, C, H, W = features.shape
    device = features.device
    dtype = features.dtype

    if isinstance(output_size, int):
        out_h = output_size
        out_w = output_size
    else:
        out_h, out_w = output_size

    cx = centers[:, 0].to(dtype)
    cy = centers[:, 1].to(dtype)
    roi_w = whs[:, 0].to(dtype).clamp(min=1.0)
    roi_h = whs[:, 1].to(dtype).clamp(min=1.0)

    denom_w = max(W - 1, 1)
    denom_h = max(H - 1, 1)

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = roi_w / denom_w
    theta[:, 1, 1] = roi_h / denom_h
    theta[:, 0, 2] = 2.0 * cx / denom_w - 1.0
    theta[:, 1, 2] = 2.0 * cy / denom_h - 1.0

    grid = F.affine_grid(theta, (B, C, out_h, out_w), align_corners=True)
    roi_features = F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return roi_features


def resolve_roi_feat_input_res(cdpn_cfg):
    roi_feat_input_res = int(cdpn_cfg.get("ROI_FEAT_INPUT_RES", 256))
    if roi_feat_input_res < 16:
        raise AssertionError(
            f"MODEL.CDPN.ROI_FEAT_INPUT_RES must be >= 16, got {roi_feat_input_res}"
        )
    if roi_feat_input_res % 8 != 0:
        raise AssertionError(
            f"MODEL.CDPN.ROI_FEAT_INPUT_RES must be divisible by 8, got {roi_feat_input_res}"
        )
    down_ratio = roi_feat_input_res // 8
    if down_ratio & (down_ratio - 1):
        raise AssertionError(
            "MODEL.CDPN.ROI_FEAT_INPUT_RES / 8 must be a power of two, "
            f"got ratio={down_ratio} for ROI_FEAT_INPUT_RES={roi_feat_input_res}"
        )
    return roi_feat_input_res


def resolve_roi_decoder_feature_dims(cdpn_cfg, dec_num_heads):
    roi_decoder_dim = int(cdpn_cfg.get("ROI_DECODER_DIM", 512))
    roi_head_out_dim = int(cdpn_cfg.get("ROI_HEAD_OUT_DIM", 128))
    if roi_decoder_dim <= 0:
        raise AssertionError(f"MODEL.CDPN.ROI_DECODER_DIM must be positive, got {roi_decoder_dim}")
    if roi_decoder_dim % int(dec_num_heads) != 0:
        raise AssertionError(
            "MODEL.CDPN.ROI_DECODER_DIM must be divisible by roi decoder heads "
            f"({dec_num_heads}), got {roi_decoder_dim}"
        )
    if roi_head_out_dim <= 0:
        raise AssertionError(f"MODEL.CDPN.ROI_HEAD_OUT_DIM must be positive, got {roi_head_out_dim}")
    return roi_decoder_dim, roi_head_out_dim


def build_roi_feat_encoder(input_res, out_res=8, in_channels=128, final_channels=512):
    input_res = int(input_res)
    out_res = int(out_res)
    if input_res < out_res:
        raise AssertionError(f"input_res must be >= out_res, got {input_res} < {out_res}")
    if input_res % out_res != 0:
        raise AssertionError(f"input_res must be divisible by out_res, got {input_res} / {out_res}")
    ratio = input_res // out_res
    if ratio & (ratio - 1):
        raise AssertionError(
            f"input_res/out_res must be power-of-two ratio, got input_res={input_res}, out_res={out_res}"
        )

    num_down = int(math.log2(ratio))
    layers = []
    cur_channels = int(in_channels)
    for i in range(num_down):
        if i < max(num_down - 2, 0):
            next_channels = int(in_channels)
        elif i == num_down - 2:
            next_channels = 256
        else:
            next_channels = int(final_channels)
        layers.append(nn.Conv2d(cur_channels, next_channels, kernel_size=3, stride=2, padding=1, bias=True))
        if i != num_down - 1:
            layers.append(nn.GELU())
        cur_channels = next_channels

    if num_down == 0:
        layers.append(nn.Conv2d(cur_channels, final_channels, kernel_size=1, stride=1, padding=0, bias=True))

    return nn.Sequential(*layers)


def resolve_dino_and_decoder_depths(cdpn_cfg, dino_total_blocks):
    dino_total_blocks = int(dino_total_blocks)
    if dino_total_blocks <= 0:
        raise AssertionError(f"dino_total_blocks must be positive, got {dino_total_blocks}")

    dino_num_blocks = int(cdpn_cfg.get("DINO_NUM_BLOCKS", dino_total_blocks))
    attn_decoder_depth = int(cdpn_cfg.get("ATTN_DECODER_DEPTH", 6))
    roi_decoder_depth = int(cdpn_cfg.get("ROI_DECODER_DEPTH", 5))

    if not (1 <= dino_num_blocks <= dino_total_blocks):
        raise AssertionError(
            f"MODEL.CDPN.DINO_NUM_BLOCKS must be in [1, {dino_total_blocks}], got {dino_num_blocks}"
        )
    if attn_decoder_depth < 2:
        raise AssertionError(
            f"MODEL.CDPN.ATTN_DECODER_DEPTH must be >= 2, got {attn_decoder_depth}"
        )
    if roi_decoder_depth < 1:
        raise AssertionError(
            f"MODEL.CDPN.ROI_DECODER_DEPTH must be >= 1, got {roi_decoder_depth}"
        )

    return dino_num_blocks, attn_decoder_depth, roi_decoder_depth


def resolve_dino_tune_last_n_blocks(cdpn_cfg, dino_num_blocks):
    dino_num_blocks = int(dino_num_blocks)
    if dino_num_blocks <= 0:
        raise AssertionError(f"dino_num_blocks must be positive, got {dino_num_blocks}")

    tune_last_n = int(cdpn_cfg.get("DINO_TUNE_LAST_N_BLOCKS", dino_num_blocks))
    if not (0 <= tune_last_n <= dino_num_blocks):
        raise AssertionError(
            f"MODEL.CDPN.DINO_TUNE_LAST_N_BLOCKS must be in [0, {dino_num_blocks}], got {tune_last_n}"
        )
    return tune_last_n


def get_xyz_mask_region_out_dim(cfg):
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        r_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 3 * (r_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    if mask_loss_type in ["L1", "BCE"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    region_out_dim = r_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    assert region_out_dim > 2, region_out_dim

    return r_out_dim, mask_out_dim, region_out_dim


def get_geo_prior_extra_pnp_channels(gp_cfg):
    if not gp_cfg or not gp_cfg.get("ENABLED", False):
        return 0
    if not gp_cfg.get("INJECT_TO_PNP", True):
        return 0

    extra = 3  # xyz_prior
    if gp_cfg.get("APPEND_PRIOR_CONF", True):
        extra += 1
    if gp_cfg.get("APPEND_PRIOR_RESIDUAL", True):
        extra += 3
    return extra


def get_geo_prior_hyp_source(vi_cfg):
    gp_hyp_cfg = vi_cfg.get("GEOMETRIC_PRIOR", {})
    return gp_hyp_cfg.get("HYP_SOURCE", "noisy_gt")


def get_geo_prior_loss_cfg(vi_cfg):
    return vi_cfg.get("GEOMETRIC_PRIOR_LOSS", {})


def validate_geo_prior_resolution(gp_cfg, output_res):
    if not gp_cfg or not gp_cfg.get("ENABLED", False):
        return
    prior_res = int(gp_cfg.get("PRIOR_RES", output_res))
    if prior_res != int(output_res):
        raise AssertionError(
            f"GEOMETRIC_PRIOR.PRIOR_RES ({prior_res}) must match BACKBONE.OUTPUT_RES ({int(output_res)})"
        )


def get_pnp_net_input_channels(cdpn_cfg):
    r_head_cfg = cdpn_cfg.ROT_HEAD
    pnp_net_cfg = cdpn_cfg.PNP_NET
    gp_cfg = cdpn_cfg.get("VIEW_INTERACTION", {}).get("GEOMETRIC_PRIOR", {})

    if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
        # coor_feat uses softmax logits without bg bin: 3 * XYZ_BIN
        pnp_in = 3 * int(r_head_cfg.XYZ_BIN)
    else:
        # regression heads output one channel per axis
        pnp_in = 3

    if pnp_net_cfg.WITH_2D_COORD:
        pnp_in += 2
    if pnp_net_cfg.REGION_ATTENTION:
        pnp_in += r_head_cfg.NUM_REGIONS
    if pnp_net_cfg.MASK_ATTENTION in ["concat"]:
        pnp_in += 1
    pnp_in += get_geo_prior_extra_pnp_channels(gp_cfg)
    return pnp_in


def decode_xyz_expectation_from_logits(coor_x, coor_y, coor_z):
    if coor_x.shape[1] == 1:
        return torch.cat([coor_x, coor_y, coor_z], dim=1)

    num_bins = coor_x.shape[1]
    bins = torch.linspace(0.0, 1.0, num_bins, device=coor_x.device, dtype=coor_x.dtype).view(1, num_bins, 1, 1)
    x = (F.softmax(coor_x, dim=1) * bins).sum(dim=1, keepdim=True)
    y = (F.softmax(coor_y, dim=1) * bins).sum(dim=1, keepdim=True)
    z = (F.softmax(coor_z, dim=1) * bins).sum(dim=1, keepdim=True)
    return torch.cat([x, y, z], dim=1)


def assemble_pnp_inputs_with_geo_prior(
    coor_feat,
    roi_coord_2d_pnp,
    xyz_single,
    geo_prior,
    with_2d_coord,
    append_prior_conf,
    append_prior_residual,
):
    parts = [coor_feat]
    if with_2d_coord:
        parts.append(roi_coord_2d_pnp)

    if geo_prior is not None:
        xyz_prior = geo_prior["xyz_prior_64"]
        parts.append(xyz_prior)
        if append_prior_conf:
            parts.append(geo_prior["prior_conf_64"])
        if append_prior_residual:
            parts.append(xyz_single - xyz_prior)

    return torch.cat(parts, dim=1)


def should_inject_geo_prior_features(geo_prior_out):
    if geo_prior_out is None:
        return False
    bank_conf = geo_prior_out.get("bank_conf", None)
    if bank_conf is None:
        return True
    return bool(bank_conf.numel() > 0)


def build_module_ddp_dummy_anchor_loss(module, reference_tensor):
    # Build a zero-valued scalar connected to all trainable module params.
    # This avoids DDP "unused parameter" reduction errors on conditional branches.
    anchor = reference_tensor.sum() * 0.0
    if module is None:
        return anchor
    for param in module.parameters():
        if param.requires_grad:
            anchor = anchor + param.sum() * 0.0
    return anchor


def compute_geo_prior_debug_metrics(
    geo_prior_out,
    geo_prior_out_flat,
    geo_prior_active,
    prior_was_dropped,
    is_all_target,
    gt_xyz=None,
    gt_mask_visib=None,
):
    metrics = {
        "vis/geo_prior_active": float(geo_prior_active),
        "vis/geo_prior_dropped": float(prior_was_dropped),
        "vis/geo_prior_all_target": float(is_all_target),
        "vis/geo_prior_has_bank": 0.0,
        "vis/geo_prior_bank_valid_ratio": 0.0,
        "vis/prior_conf_mean": 0.0,
        "vis/prior_conf_obj_mean": 0.0,
        "vis/prior_ambiguity_mean": 0.0,
        "vis/prior_xyz_abs_err": 0.0,
    }
    if geo_prior_out is not None:
        bank_conf = geo_prior_out.get("bank_conf", None)
        if torch.is_tensor(bank_conf):
            bank_tokens_valid = float(bank_conf.numel())
            metrics["vis/geo_prior_has_bank"] = float(bank_tokens_valid > 0.0)
            bank_weights = geo_prior_out.get("bank_weights_8", None)
            if torch.is_tensor(bank_weights) and bank_weights.ndim == 3:
                total_tokens = float(bank_weights.shape[0] * bank_weights.shape[2])
                if total_tokens > 0:
                    metrics["vis/geo_prior_bank_valid_ratio"] = bank_tokens_valid / total_tokens

    if geo_prior_out_flat is None:
        return metrics

    prior_conf = geo_prior_out_flat.get("prior_conf_64", None)
    prior_ambiguity = geo_prior_out_flat.get("prior_ambiguity_64", None)
    xyz_prior = geo_prior_out_flat.get("xyz_prior_64", None)
    if torch.is_tensor(prior_conf):
        metrics["vis/prior_conf_mean"] = float(prior_conf.mean().detach().item())
    if torch.is_tensor(prior_ambiguity):
        metrics["vis/prior_ambiguity_mean"] = float(prior_ambiguity.mean().detach().item())
    if torch.is_tensor(xyz_prior) and torch.is_tensor(gt_xyz) and torch.is_tensor(gt_mask_visib):
        gt_xyz_aligned, gt_mask_aligned = align_geo_prior_debug_targets(
            xyz_prior=xyz_prior, gt_xyz=gt_xyz, gt_mask_visib=gt_mask_visib
        )
        if gt_xyz_aligned is not None and gt_mask_aligned is not None:
            valid = (gt_mask_aligned > 0.5).to(dtype=xyz_prior.dtype)
            if torch.is_tensor(prior_conf):
                conf_obj_mean = (prior_conf * valid).sum() / (valid.sum() + 1e-6)
                metrics["vis/prior_conf_obj_mean"] = float(conf_obj_mean.detach().item())
            xyz_abs_err = (xyz_prior - gt_xyz_aligned).abs() * valid
            err = xyz_abs_err.sum() / (valid.sum() * 3.0 + 1e-6)
            metrics["vis/prior_xyz_abs_err"] = float(err.detach().item())

    return metrics


def align_geo_prior_debug_targets(xyz_prior, gt_xyz, gt_mask_visib):
    gt_xyz_aligned = gt_xyz
    gt_mask_aligned = gt_mask_visib

    # Common multiview layout: (B, T, C, H, W) -> (B*T, C, H, W)
    if gt_xyz_aligned.dim() == 5 and gt_xyz_aligned.shape[2] == xyz_prior.shape[1]:
        gt_xyz_aligned = gt_xyz_aligned.flatten(0, 1).contiguous()
    if gt_mask_aligned.dim() == 5 and gt_mask_aligned.shape[2] == 1:
        gt_mask_aligned = gt_mask_aligned.flatten(0, 1).contiguous()

    # Common alternative layout: (N, H, W, C) -> (N, C, H, W)
    if gt_xyz_aligned.dim() == 4 and gt_xyz_aligned.shape[-1] == xyz_prior.shape[1] and gt_xyz_aligned.shape[1] != xyz_prior.shape[1]:
        gt_xyz_aligned = gt_xyz_aligned.permute(0, 3, 1, 2).contiguous()
    if gt_mask_aligned.dim() == 4 and gt_mask_aligned.shape[-1] == 1 and gt_mask_aligned.shape[1] != 1:
        gt_mask_aligned = gt_mask_aligned.permute(0, 3, 1, 2).contiguous()

    # Seen in some paths: (C, N, H, W) / (1, N, H, W) -> (N, C, H, W) / (N, 1, H, W)
    if gt_xyz_aligned.dim() == 4 and gt_xyz_aligned.shape[0] == xyz_prior.shape[1] and gt_xyz_aligned.shape[1] == xyz_prior.shape[0]:
        gt_xyz_aligned = gt_xyz_aligned.permute(1, 0, 2, 3).contiguous()
    if gt_mask_aligned.dim() == 4 and gt_mask_aligned.shape[0] == 1 and gt_mask_aligned.shape[1] == xyz_prior.shape[0]:
        gt_mask_aligned = gt_mask_aligned.permute(1, 0, 2, 3).contiguous()

    if gt_mask_aligned.dim() == 3:
        gt_mask_aligned = gt_mask_aligned.unsqueeze(1).contiguous()

    if (
        gt_xyz_aligned.dim() != 4
        or gt_mask_aligned.dim() != 4
        or gt_xyz_aligned.shape != xyz_prior.shape
        or gt_mask_aligned.shape[0] != xyz_prior.shape[0]
        or gt_mask_aligned.shape[-2:] != xyz_prior.shape[-2:]
    ):
        return None, None
    if gt_mask_aligned.shape[1] != 1:
        gt_mask_aligned = gt_mask_aligned[:, :1].contiguous()
    return gt_xyz_aligned, gt_mask_aligned


def compute_geo_prior_training_losses(xyz_prior, prior_conf, prior_ambiguity, roi_xyz, roi_mask_visib, cfg_loss):
    valid = (roi_mask_visib > 0.5).to(dtype=xyz_prior.dtype)
    xyz_lw = float(cfg_loss.get("XYZ_LW", 0.0))
    conf_lw = float(cfg_loss.get("CONF_LW", 0.0))
    conf_thr = float(cfg_loss.get("CONF_ERR_THR", 0.05))

    xyz_err = torch.abs(xyz_prior - roi_xyz) * valid
    xyz_loss = xyz_err.sum() / (valid.sum() * 3.0 + 1e-6)

    xyz_err_norm = torch.norm((xyz_prior - roi_xyz) * valid, dim=1, keepdim=True)
    target_conf = ((roi_mask_visib > 0.5) & (xyz_err_norm < conf_thr)).to(dtype=prior_conf.dtype)
    conf_loss = F.binary_cross_entropy(prior_conf.clamp(1e-4, 1.0 - 1e-4), target_conf)
    ambiguity_loss = F.smooth_l1_loss(prior_ambiguity, 1.0 - target_conf, reduction="mean")

    return {
        "loss_xyz_prior": xyz_loss * xyz_lw,
        "loss_prior_conf": conf_loss * conf_lw,
        "loss_prior_ambiguity": ambiguity_loss * (0.5 * conf_lw),
    }


def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

    if "resnet" in backbone_cfg.ARCH:
        params_lr_list = []
        
        # backbone net, but here backbone is deleted because we use DINO encoder + attention layers instead
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        if cfg.MODEL.CDPN.RESNET_BACKBONE:
            backbone_net = ResNetBackboneNet(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
            )
            if backbone_cfg.FREEZE:
                for param in backbone_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR),
                    }
                )
        else:
            backbone_net = None

        # rotation head net -----------------------------------------------------
        r_out_dim, mask_out_dim, region_out_dim = get_xyz_mask_region_out_dim(cfg)
        rot_head_net = RotWithRegionHead(
            cfg,
            channels[-1],
            r_head_cfg.NUM_LAYERS,
            r_head_cfg.NUM_FILTERS,
            r_head_cfg.CONV_KERNEL_SIZE,
            r_head_cfg.OUT_CONV_KERNEL_SIZE,
            rot_output_dim=r_out_dim,
            mask_output_dim=mask_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
        )
        if r_head_cfg.FREEZE:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # translation head net --------------------------------------------------------
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],  # the channels of backbone output layer
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
                    }
                )

        # -----------------------------------------------
        pnp_net_in_channel = get_pnp_net_input_channels(cfg.MODEL.CDPN)

        if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
            rot_dim = 4
        elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            rot_dim = 3
        elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
            rot_dim = 6
        else:
            raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

        pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG
        pnp_head_type = pnp_head_cfg.pop("type")
        if pnp_head_type == "ConvPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            )
            pnp_net = ConvPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "PointPnPNet":
            pnp_head_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
            pnp_net = PointPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "SimplePointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                # num_regions=r_head_cfg.NUM_REGIONS,
            )
            pnp_net = SimplePointPnPNet(**pnp_head_cfg)
        else:
            raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

        if pnp_net_cfg.FREEZE:
            for param in pnp_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
                }
            )
        # ================================================

        # CDPN (Coordinates-based Disentangled Pose Network)
        model = GDRN(cfg, backbone_net, rot_head_net, trans_head_net=trans_head_net, pnp_net=pnp_net)
        if cfg.MODEL.CDPN.USE_MTL:
            params_lr_list.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                    ),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )
            
        # HACK（rcw）: if has newly added module, e.g. image_encoder, then we need to add it to params_lr_list
        # newly_added_modules = [model.image_encoder, model.decoder, model.roi_decoder, model.encoder_post_mlp, model.pose_encoder]
        # newly_added_modules = [model.decoder, model.roi_decoder, model.encoder_post_mlp]
        # newly_added_modules = [model.decoder, model.roi_decoder]
        # newly_added_modules = [model.decoder, model.roi_decoder, model.encoder_post_mlp, model.image_encoder]
        newly_added_modules = []
        if cfg.MODEL.CDPN.VGGT_BACKBONE:
            newly_added_modules.append(model.image_encoder)
            newly_added_modules.append(model.decoder)
            newly_added_modules.append(model.roi_decoder)
            newly_added_modules.append(model.roi_head)
            newly_added_modules.append(model.depth_head)
        if model.view_interaction_enabled:
            newly_added_modules.append(model.view_interaction)
        if getattr(model, "geo_prior_enabled", False):
            newly_added_modules.append(model.context_geo_prior)
        for module in newly_added_modules:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, module.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )
        # # HACK（rcw）: freeze the image encoder
        if cfg.MODEL.CDPN.FREEZE_IMAGE_ENCODER:
            no_grad_modules = [model.image_encoder]
            for module in no_grad_modules:
                for param in module.parameters():
                    param.requires_grad = False

        # get optimizer
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = cfg.MODEL.CDPN.BACKBONE.get("PRETRAINED", "")
        if hasattr(model, "backbone") and model.backbone is not None:
            # if backbone is not deleted, then skip this initialization
            if backbone_pretrained == "":
                logger.warning("Randomly initialize weights for backbone!")
            else:
                # initialize backbone with official ImageNet weights
                logger.info(f"load backbone weights from: {backbone_pretrained}")
                load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
