from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ContextGeometricPriorInputs:
    target_roi_feat: torch.Tensor
    context_roi_feat: torch.Tensor
    context_xyz: torch.Tensor
    context_view_dir_obj: torch.Tensor
    context_cam_center_obj: torch.Tensor
    target_rgb: torch.Tensor
    context_rgb: torch.Tensor
    target_coord: torch.Tensor
    context_coord: torch.Tensor
    target_mask: torch.Tensor
    context_mask: torch.Tensor
    target_valid: torch.Tensor
    context_valid: torch.Tensor


class ContextGeometricPriorModule(nn.Module):
    def __init__(self, roi_feat_dim, token_dim, retrieval_res, prior_res, rgb_embed_dim):
        super().__init__()
        self.retrieval_res = retrieval_res
        self.prior_res = prior_res
        # prior_in = [target token, retrieved token, xyz_retrieved(3), conf(1), ambiguity(1)]
        self.prior_in_dim = 2 * token_dim + 5
        self.rgb_proj = nn.Conv2d(3, rgb_embed_dim, kernel_size=1, bias=True)
        self.tgt_proj = nn.Conv2d(roi_feat_dim + rgb_embed_dim + 3, token_dim, kernel_size=1, bias=True)
        self.ctx_proj = nn.Conv2d(roi_feat_dim + rgb_embed_dim + 12, token_dim, kernel_size=1, bias=True)
        self.retrieve_out = nn.Conv2d(token_dim, roi_feat_dim, kernel_size=1, bias=True)
        self.prior_head = nn.Sequential(
            nn.Conv2d(self.prior_in_dim, token_dim, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(prior_res, prior_res), mode="bilinear", align_corners=False),
            nn.Conv2d(token_dim, 5, 1, bias=True),
        )

    def _flatten_context_bank(self, ctx_tokens, context_xyz, context_mask, context_valid):
        b, c_ctx, d, h, w = ctx_tokens.shape
        feat = ctx_tokens.flatten(3).permute(0, 1, 3, 2).reshape(b, c_ctx * h * w, d)
        xyz = context_xyz.flatten(3).permute(0, 1, 3, 2).reshape(b, c_ctx * h * w, 3)
        if context_mask.shape[2] != 1:
            raise AssertionError(f"context_mask must be single-channel, got shape={tuple(context_mask.shape)}")
        conf = context_mask[:, :, 0].flatten(2).reshape(b, c_ctx * h * w)
        valid = context_valid[:, :, None].expand(b, c_ctx, h * w).reshape(b, c_ctx * h * w)
        return feat, xyz, conf, valid

    def forward(self, inputs: ContextGeometricPriorInputs) -> Dict[str, torch.Tensor]:
        b, t, _, h, w = inputs.target_roi_feat.shape
        _, c_ctx, _, _, _ = inputs.context_roi_feat.shape
        if inputs.context_mask.shape[2] != 1:
            raise AssertionError(
                f"context_mask must be single-channel, got shape={tuple(inputs.context_mask.shape)}"
            )
        if h != self.retrieval_res or w != self.retrieval_res:
            raise AssertionError(
                f"target_roi_feat spatial size must match retrieval_res: got ({h}, {w}) vs retrieval_res={self.retrieval_res}"
            )
        if (
            inputs.context_roi_feat.shape[-2] != self.retrieval_res
            or inputs.context_roi_feat.shape[-1] != self.retrieval_res
        ):
            raise AssertionError(
                "context_roi_feat spatial size must match retrieval_res: "
                f"got {inputs.context_roi_feat.shape[-2:]} vs retrieval_res={self.retrieval_res}"
            )

        tgt_rgb = F.adaptive_avg_pool2d(self.rgb_proj(inputs.target_rgb.flatten(0, 1)), self.retrieval_res).reshape(
            b, t, -1, h, w
        )
        ctx_rgb = F.adaptive_avg_pool2d(self.rgb_proj(inputs.context_rgb.flatten(0, 1)), self.retrieval_res).reshape(
            b, c_ctx, -1, h, w
        )

        tgt_in = torch.cat([inputs.target_roi_feat, tgt_rgb, inputs.target_coord, inputs.target_mask], dim=2)
        ctx_in = torch.cat(
            [
                inputs.context_roi_feat,
                ctx_rgb,
                inputs.context_xyz,
                inputs.context_view_dir_obj,
                inputs.context_cam_center_obj,
                inputs.context_coord,
                inputs.context_mask,
            ],
            dim=2,
        )

        tgt_tok = self.tgt_proj(tgt_in.flatten(0, 1)).reshape(b, t, -1, h, w)
        ctx_tok = self.ctx_proj(ctx_in.flatten(0, 1)).reshape(b, c_ctx, -1, h, w)
        bank_feat, bank_xyz, bank_conf, bank_valid = self._flatten_context_bank(
            ctx_tok, inputs.context_xyz, inputs.context_mask, inputs.context_valid
        )

        # DDP safety: never branch on per-rank tensor values.
        # Instead, always run the full forward pass and use masked operations.
        # bank_valid gates the attention scores; nan_to_num handles the all-invalid case.
        _, _, d, _, _ = tgt_tok.shape
        q = tgt_tok.flatten(3).permute(0, 1, 3, 2).reshape(b, t * h * w, d)
        score = torch.einsum("bqd,bkd->bqk", q, bank_feat) / (d ** 0.5)
        score = score + bank_conf.clamp(min=1e-4).log().unsqueeze(1)
        # Mask invalid bank slots. When all entries are -inf, softmax → NaN → nan_to_num(0).
        score = score.masked_fill(~bank_valid.unsqueeze(1), float("-inf"))
        weights = torch.softmax(score, dim=-1)
        weights = weights.nan_to_num(0.0)

        xyz_expect = torch.einsum("bqk,bkc->bqc", weights, bank_xyz)
        xyz_second = torch.einsum("bqk,bkc->bqc", weights, bank_xyz * bank_xyz)
        xyz_var = (xyz_second - xyz_expect * xyz_expect).clamp(min=0.0).mean(dim=-1, keepdim=True)

        xyz_retrieved = xyz_expect.transpose(1, 2).reshape(b, t, 3, h, w)
        ambiguity_8 = torch.sqrt(xyz_var + 1e-6).transpose(1, 2).reshape(b, t, 1, h, w)
        conf_8 = torch.exp(-ambiguity_8)

        feat_retrieved = torch.einsum("bqk,bkd->bqd", weights, bank_feat).transpose(1, 2).reshape(b, t, -1, h, w)
        retrieved = self.retrieve_out((tgt_tok + feat_retrieved).flatten(0, 1)).reshape(b, t, -1, h, w)

        prior_in = torch.cat([tgt_tok, feat_retrieved, xyz_retrieved, conf_8, ambiguity_8], dim=2).flatten(0, 1)
        prior_raw = self.prior_head(prior_in).reshape(b, t, 5, self.prior_res, self.prior_res)
        ambiguity_seed = F.interpolate(
            ambiguity_8.flatten(0, 1), size=(self.prior_res, self.prior_res), mode="bilinear", align_corners=False
        ).reshape(b, t, 1, self.prior_res, self.prior_res)
        xyz_prior = prior_raw[:, :, :3]
        ambiguity = torch.sigmoid(prior_raw[:, :, 3:4]) * ambiguity_seed
        conf = torch.sigmoid(prior_raw[:, :, 4:5]) * torch.exp(-ambiguity)

        return {
            # bank_xyz / bank_conf are debug-only (not used for gradients).
            # Keep the full (b, K, 3) shape rather than boolean-indexing to
            # avoid rank-local variable-length outputs in DDP.
            "bank_xyz": bank_xyz,
            "bank_conf": bank_conf,
            "retrieved_feat_8": retrieved,
            "xyz_prior_64": xyz_prior,
            "prior_conf_64": conf.clamp(0.0, 1.0),
            "prior_ambiguity_64": ambiguity.clamp(0.0, 1.0),
            "bank_weights_8": weights,
        }
