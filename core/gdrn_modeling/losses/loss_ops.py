import torch
import torch.nn as nn
import torch.nn.functional as F

from .coor_cross_entropy import CrossEntropyHeatmapLoss
from .l2_loss import L2Loss
from .pm_loss import PyPMLoss
from .rot_loss import angular_distance, rot_l2_loss


def compute_xyz_losses(r_head_cfg, out_x, out_y, out_z, gt_xyz, gt_xyz_bin, gt_masks):
    loss_dict = {}
    if r_head_cfg.FREEZE:
        return loss_dict

    xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
    gt_mask_xyz = gt_masks[r_head_cfg.XYZ_LOSS_MASK_GT]
    denom = gt_mask_xyz.sum().float().clamp(min=1.0)
    if xyz_loss_type == "L1":
        loss_func = nn.L1Loss(reduction="sum")
        loss_dict["loss_coor_x"] = loss_func(
            out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
        ) / denom
        loss_dict["loss_coor_y"] = loss_func(
            out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
        ) / denom
        loss_dict["loss_coor_z"] = loss_func(
            out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
        ) / denom
    elif xyz_loss_type == "CE_coor":
        gt_xyz_bin = gt_xyz_bin.long()
        loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)
        loss_dict["loss_coor_x"] = loss_func(
            out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
        ) / denom
        loss_dict["loss_coor_y"] = loss_func(
            out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
        ) / denom
        loss_dict["loss_coor_z"] = loss_func(
            out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
        ) / denom
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    for key in ("loss_coor_x", "loss_coor_y", "loss_coor_z"):
        loss_dict[key] *= r_head_cfg.XYZ_LW
    return loss_dict


def compute_mask_loss(r_head_cfg, out_mask, gt_masks):
    loss_dict = {}
    if r_head_cfg.FREEZE:
        return loss_dict

    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
    gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT]
    if mask_loss_type == "L1":
        loss = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
    elif mask_loss_type == "BCE":
        loss = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
    elif mask_loss_type == "CE":
        loss = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    loss_dict["loss_mask"] = loss * r_head_cfg.MASK_LW
    return loss_dict


def compute_region_loss(r_head_cfg, out_region, gt_region, gt_masks):
    loss_dict = {}
    if r_head_cfg.FREEZE:
        return loss_dict

    region_loss_type = r_head_cfg.REGION_LOSS_TYPE
    gt_mask_region = gt_masks[r_head_cfg.REGION_LOSS_MASK_GT]
    if region_loss_type == "CE":
        gt_region = gt_region.long()
        loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)
        loss = loss_func(
            out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
        ) / gt_mask_region.sum().float().clamp(min=1.0)
    else:
        raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
    loss_dict["loss_region"] = loss * r_head_cfg.REGION_LW
    return loss_dict


def build_pm_loss(pnp_net_cfg):
    return PyPMLoss(
        loss_type=pnp_net_cfg.PM_LOSS_TYPE,
        beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
        reduction="mean",
        loss_weight=pnp_net_cfg.PM_LW,
        norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
        symmetric=pnp_net_cfg.PM_LOSS_SYM,
        disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
        disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
        t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
        r_only=pnp_net_cfg.PM_R_ONLY,
    )


def compute_rotation_loss(pnp_net_cfg, out_rot, gt_rot):
    if pnp_net_cfg.ROT_LOSS_TYPE == "angular":
        loss = angular_distance(out_rot, gt_rot)
    elif pnp_net_cfg.ROT_LOSS_TYPE == "L2":
        loss = rot_l2_loss(out_rot, gt_rot)
    else:
        raise ValueError(f"Unknown rot loss type: {pnp_net_cfg.ROT_LOSS_TYPE}")
    return loss * pnp_net_cfg.ROT_LW


def compute_centroid_loss(pnp_net_cfg, out_centroid, gt_trans_ratio):
    if pnp_net_cfg.CENTROID_LOSS_TYPE == "L1":
        loss = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
    elif pnp_net_cfg.CENTROID_LOSS_TYPE == "L2":
        loss = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
    elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
        loss = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
    else:
        raise ValueError(f"Unknown centroid loss type: {pnp_net_cfg.CENTROID_LOSS_TYPE}")
    return loss * pnp_net_cfg.CENTROID_LW


def compute_z_loss(pnp_net_cfg, out_trans_z, gt_trans, gt_trans_ratio):
    if pnp_net_cfg.Z_TYPE == "REL":
        gt_z = gt_trans_ratio[:, 2]
    elif pnp_net_cfg.Z_TYPE == "ABS":
        gt_z = gt_trans[:, 2]
    else:
        raise NotImplementedError

    if pnp_net_cfg.Z_LOSS_TYPE == "L1":
        loss = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
    elif pnp_net_cfg.Z_LOSS_TYPE == "L2":
        loss = L2Loss(reduction="mean")(out_trans_z, gt_z)
    elif pnp_net_cfg.Z_LOSS_TYPE == "MSE":
        loss = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
    else:
        raise ValueError(f"Unknown z loss type: {pnp_net_cfg.Z_LOSS_TYPE}")
    return loss * pnp_net_cfg.Z_LW


def compute_translation_losses(pnp_net_cfg, out_trans, gt_trans):
    loss_dict = {}
    if pnp_net_cfg.TRANS_LOSS_DISENTANGLE:
        if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
            loss_xy = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
            loss_z = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
            loss_xy = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
            loss_z = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
            loss_xy = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
            loss_z = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        else:
            raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
        loss_dict["loss_trans_xy"] = loss_xy * pnp_net_cfg.TRANS_LW
        loss_dict["loss_trans_z"] = loss_z * pnp_net_cfg.TRANS_LW
        return loss_dict

    if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
        loss = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
    elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
        loss = L2Loss(reduction="mean")(out_trans, gt_trans)
    elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
        loss = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
    else:
        raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
    loss_dict["loss_trans_LPnP"] = loss * pnp_net_cfg.TRANS_LW
    return loss_dict


def compute_bind_loss(pnp_net_cfg, out_rot, out_trans, gt_rot, gt_trans):
    pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
    gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
    if pnp_net_cfg.BIND_LOSS_TYPE == "L1":
        loss = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
    elif pnp_net_cfg.BIND_LOSS_TYPE == "L2":
        loss = L2Loss(reduction="mean")(pred_bind, gt_bind)
    elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
        loss = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
    else:
        raise ValueError(f"Unknown bind loss (R^T@t) type: {pnp_net_cfg.BIND_LOSS_TYPE}")
    return loss * pnp_net_cfg.BIND_LW


def compute_zero_target_regression_loss(loss_type, pred, target, smooth_l1_beta=0.01):
    if loss_type == "L1":
        return nn.L1Loss(reduction="mean")(pred, target)
    if loss_type == "L2":
        return L2Loss(reduction="mean")(pred, target)
    if loss_type == "MSE":
        return nn.MSELoss(reduction="mean")(pred, target)
    if loss_type == "SmoothL1":
        return F.smooth_l1_loss(pred, target, reduction="mean", beta=smooth_l1_beta)
    raise ValueError(f"Unknown regression loss type: {loss_type}")


def apply_mtl_uncertainty(loss_dict, loss_owner):
    for key in list(loss_dict.keys()):
        log_var_name = key.replace("loss_", "log_var_")
        if hasattr(loss_owner, log_var_name):
            cur_log_var = getattr(loss_owner, log_var_name)
            loss_dict[key] = loss_dict[key] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
    return loss_dict
