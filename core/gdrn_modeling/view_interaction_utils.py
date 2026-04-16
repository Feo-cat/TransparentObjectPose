import numpy as np
import torch


VI_SOURCE_TO_MODEL_KEY = {
    "ego_rot": "gt_ego_rot",
    "trans": "gt_trans",
    "input_depths": "input_depths",
    "input_obj_masks": "input_obj_masks",
    "model_info": "model_infos",
}


def normalize_target_indices(target_idx, num_views):
    """Normalize target indices to a positive python int list."""
    if isinstance(target_idx, (list, tuple)):
        target_ids = [int(i) for i in target_idx]
    elif isinstance(target_idx, np.ndarray):
        target_ids = (
            [int(i) for i in target_idx.reshape(-1).tolist()]
            if target_idx.ndim > 0
            else [int(target_idx.item())]
        )
    elif torch.is_tensor(target_idx):
        target_ids = (
            [int(i) for i in target_idx.reshape(-1).tolist()]
            if target_idx.numel() > 1
            else [int(target_idx.item())]
        )
    else:
        target_ids = [int(target_idx)]
    return [idx + num_views if idx < 0 else idx for idx in target_ids]


def analyze_target_indices(target_idx, num_views):
    target_ids = normalize_target_indices(target_idx, num_views)
    has_context_for_target = len(target_ids) > 0 and min(target_ids) > 0
    is_all_target = set(target_ids) == set(range(num_views))
    return target_ids, has_context_for_target, is_all_target


def collect_view_interaction_model_kwargs(data, device=None):
    """Collect the shared VI-related model kwargs from a train/eval batch dict."""
    kwargs = {}
    missing = []
    for source_key, model_key in VI_SOURCE_TO_MODEL_KEY.items():
        value = data.get(source_key, None)
        if torch.is_tensor(value) and device is not None:
            value = value.to(device)
        kwargs[model_key] = value
        if source_key != "model_info" and value is None:
            missing.append(source_key)
    return kwargs, missing


def build_context_pose_confidence(gt_ego_rot, gt_trans, eps=1e-6):
    eye = torch.eye(3, device=gt_ego_rot.device, dtype=gt_ego_rot.dtype).view(1, 1, 3, 3)
    is_eye = (gt_ego_rot - eye).abs().amax(dim=(-1, -2)) < eps
    is_zero_t = gt_trans.abs().amax(dim=-1) < eps
    valid = ~(is_eye & is_zero_t)
    return valid.to(dtype=gt_trans.dtype)


def build_context_view_geom_features(ctx_rot, ctx_trans, out_hw=8):
    cam_forward_cam = torch.tensor([0.0, 0.0, 1.0], device=ctx_rot.device, dtype=ctx_rot.dtype).view(1, 3, 1)
    view_dir_obj = torch.bmm(ctx_rot.transpose(1, 2), cam_forward_cam.expand(ctx_rot.shape[0], -1, -1))
    view_dir_obj = view_dir_obj.squeeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out_hw, out_hw)

    cam_center_obj = -torch.bmm(ctx_rot.transpose(1, 2), ctx_trans.unsqueeze(-1))
    cam_center_obj = cam_center_obj.squeeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out_hw, out_hw)
    return view_dir_obj, cam_center_obj
