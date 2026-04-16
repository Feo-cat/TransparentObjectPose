import torch
from torch import nn


class RelativeGeoSymLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _relative_transform(self, rot_a, trans_a, rot_b, trans_b):
        rel_rot = torch.matmul(rot_a, rot_b.transpose(1, 2))
        rel_trans = trans_a - torch.bmm(rel_rot, trans_b.unsqueeze(-1)).squeeze(-1)
        return rel_rot, rel_trans

    def forward(
        self,
        pred_rot,
        pred_trans,
        ctx_hyp_rot,
        ctx_hyp_trans,
        gt_rot,
        gt_trans,
        ctx_gt_rot,
        ctx_gt_trans,
        model_points,
        max_points=None,
    ):
        if max_points is not None:
            max_points = int(max_points)
            if max_points > 0 and model_points.shape[1] > max_points:
                # Deterministic truncation keeps behavior reproducible across ranks.
                model_points = model_points[:, :max_points, :]

        rel_pred_rot, rel_pred_trans = self._relative_transform(pred_rot, pred_trans, ctx_hyp_rot, ctx_hyp_trans)
        rel_gt_rot, rel_gt_trans = self._relative_transform(gt_rot, gt_trans, ctx_gt_rot, ctx_gt_trans)

        pts_pred = torch.bmm(model_points, rel_pred_rot.transpose(1, 2)) + rel_pred_trans.unsqueeze(1)
        pts_gt = torch.bmm(model_points, rel_gt_rot.transpose(1, 2)) + rel_gt_trans.unsqueeze(1)

        # Use cdist to avoid materializing a (B, P, P, 3) tensor.
        dist = torch.cdist(pts_pred, pts_gt, p=2)
        loss_geo = dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)
        loss_trans = torch.abs(rel_pred_trans - rel_gt_trans).mean()
        return {
            "loss_rel_geo_sym": loss_geo.mean(),
            "loss_rel_trans": loss_trans,
        }
