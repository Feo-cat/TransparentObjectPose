import torch
import torch.nn as nn

class TemporalSmoothLoss(nn.Module):
    """
    Computes 1st-order (velocity) and 2nd-order (acceleration) smoothing loss
    for a sequence of poses, using 3D model points (ADD metric).
    
    This is calculated entirely on the predicted poses to penalize jitter and
    enforce constant-velocity (inertial) motion.
    """
    def __init__(self, eps: float = 1e-6, loss_type="SmoothL1"):
        super().__init__()
        self.eps = eps
        self.loss_type = loss_type

    def forward(self, R_t, t_t, R_prev, t_prev, R_prev2=None, t_prev2=None, model_points=None):
        """
        Args:
            R_t:          Current-frame rotation matrices, shape (B, 3, 3).
            t_t:          Current-frame translations, shape (B, 3).
            R_prev:       Previous-frame rotation matrices, shape (B, 3, 3).
            t_prev:       Previous-frame translations, shape (B, 3).
            R_prev2:      Frame t-2 rotation matrices, shape (B, 3, 3). Optional.
            t_prev2:      Frame t-2 translations, shape (B, 3). Optional.
            model_points: Object CAD points, shape (B, P, 3) or (P, 3).
            
        Returns:
            loss_1st: scalar tensor, velocity penalty.
            loss_2nd: scalar tensor, acceleration penalty (0 if R_prev2 is None).
        """
        if model_points.dim() == 2:
            model_points = model_points.unsqueeze(0).expand(R_t.shape[0], -1, -1)
            
        model_points = model_points.to(device=R_t.device, dtype=R_t.dtype)
        t_t = t_t.to(dtype=R_t.dtype)
        R_prev = R_prev.to(dtype=R_t.dtype)
        t_prev = t_prev.to(dtype=R_t.dtype)

        # Transform points to camera space
        pts_t = torch.bmm(model_points, R_t.transpose(1, 2)) + t_t.unsqueeze(1)
        pts_prev = torch.bmm(model_points, R_prev.transpose(1, 2)) + t_prev.unsqueeze(1)
        
        # 1st-order velocity (L1 or SmoothL1 is more robust than L2 for outliers)
        v_t = pts_t - pts_prev
        
        if self.loss_type == "L1":
            loss_1st = v_t.abs().mean()
        elif self.loss_type == "SmoothL1":
            loss_1st = torch.nn.functional.smooth_l1_loss(pts_t, pts_prev, reduction="mean")
        else:
            loss_1st = torch.sqrt(v_t.pow(2).sum(dim=-1) + self.eps).mean()

        loss_2nd = R_t.new_tensor(0.0)
        if R_prev2 is not None and t_prev2 is not None:
            R_prev2 = R_prev2.to(dtype=R_t.dtype)
            t_prev2 = t_prev2.to(dtype=R_t.dtype)
            pts_prev2 = torch.bmm(model_points, R_prev2.transpose(1, 2)) + t_prev2.unsqueeze(1)
            v_prev = pts_prev - pts_prev2
            a_t = v_t - v_prev  # acceleration
            
            if self.loss_type == "L1":
                loss_2nd = a_t.abs().mean()
            elif self.loss_type == "SmoothL1":
                loss_2nd = torch.nn.functional.smooth_l1_loss(v_t, v_prev, reduction="mean")
            else:
                loss_2nd = torch.sqrt(a_t.pow(2).sum(dim=-1) + self.eps).mean()
                
        return loss_1st, loss_2nd
