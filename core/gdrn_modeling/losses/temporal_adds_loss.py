import torch
import torch.nn as nn


class TemporalADDSLoss(nn.Module):
    """Temporal ADD-S (symmetric Chamfer) smoothness loss.

    This loss penalizes frame-to-frame jitter in 3D physical space and is robust
    to rotational symmetries because it compares transformed point sets with
    nearest-neighbor distance (ADD-S / bidirectional Chamfer distance).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, R_t, t_t, R_prev, t_prev, model_points):
        """Compute mean temporal ADD-S loss over a batch.

        Args:
            R_t:          Current-frame rotation matrices, shape (B, 3, 3).
            t_t:          Current-frame translations, shape (B, 3).
            R_prev:       Previous-frame rotation matrices, shape (B, 3, 3).
            t_prev:       Previous-frame translations, shape (B, 3).
            model_points: Object CAD points, shape (B, P, 3) or (P, 3).

        Returns:
            Scalar tensor: batch-mean bidirectional Chamfer distance.
        """
        if model_points.dim() == 2:
            model_points = model_points.unsqueeze(0).expand(R_t.shape[0], -1, -1)

        if model_points.shape[0] != R_t.shape[0]:
            raise ValueError(
                f"Batch mismatch: model_points={model_points.shape[0]} vs R_t={R_t.shape[0]}"
            )

        model_points = model_points.to(device=R_t.device, dtype=R_t.dtype)
        t_t = t_t.to(dtype=R_t.dtype)
        R_prev = R_prev.to(dtype=R_t.dtype)
        t_prev = t_prev.to(dtype=R_t.dtype)

        # (B, P, 3): batched rigid transforms in camera space.
        # 这里的 R.transpose(1,2) 处理得非常完美！
        pts_t = torch.bmm(model_points, R_t.transpose(1, 2)) + t_t.unsqueeze(1)
        pts_prev = torch.bmm(model_points, R_prev.transpose(1, 2)) + t_prev.unsqueeze(1)

        # ---------------- 改造开始：安全的距离计算 ----------------
        # 1. 计算两组点云的差异向量: (B, P, 1, 3) - (B, 1, P, 3) -> (B, P, P, 3)
        diff = pts_t.unsqueeze(2) - pts_prev.unsqueeze(1)
        
        # 2. 计算平方距离 (B, P, P)
        dist_sq = diff.pow(2).sum(dim=-1)
        
        # 3. 在开根号"内部"加上极小值 eps，绝对杜绝梯度 NaN
        pairwise_dist = torch.sqrt(dist_sq + self.eps)
        # ---------------- 改造结束 ----------------

        min_t_to_prev = pairwise_dist.min(dim=2).values
        min_prev_to_t = pairwise_dist.min(dim=1).values

        loss_per_sample = min_t_to_prev.mean(dim=1) + min_prev_to_t.mean(dim=1)
        
        # 直接返回均值即可，不需要在外面减去 eps 了
        return loss_per_sample.mean()
