import cv2
import numpy as np
import torch
from lib.pysixd.pose_error import re, te


def get_2d_coord(bs, width, height, dtype=torch.float32, device="cuda"):
    """
    Args:
        bs: batch size
        width:
        height:
    """
    # coords values are in [-1, 1]
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    xy = np.meshgrid(x, y)
    coord = np.stack([xy for _ in range(bs)])
    coord_tensor = torch.tensor(coord, dtype=dtype, device=device)
    coord_tensor = coord_tensor.view(bs, 2, height, width)

    return coord_tensor  # [bs, 2, h, w]


def get_mask_prob(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.CDPN.ROT_HEAD.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type == "BCE":
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return mask_prob


def normalize_kernel_size(kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 0
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def morphology_kernel(kernel_size):
    kernel_size = normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return None
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def clean_mask_with_temporal_prior(
    cur_mask_u8,
    prev_mask_u8=None,
    mode="none",
    prev_dilate_kernel=11,
    prev_gate=True,
    post_open_kernel=0,
    post_dilate_kernel=3,
    post_close_kernel=3,
    min_mask_pixels=16,
    fallback_to_prev=False,
):
    """Clean a binary mask using connected components and optional temporal prior."""
    cur_mask_u8 = (np.asarray(cur_mask_u8) > 0).astype(np.uint8)
    prev_mask_u8 = None if prev_mask_u8 is None else (np.asarray(prev_mask_u8) > 0).astype(np.uint8)
    mode = str(mode).lower()
    overlap_keep_ratio = 0.20
    strict_prev_dilate_kernel = 3
    strict_keep_ratio = 0.45

    selected_mask = cur_mask_u8.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cur_mask_u8, connectivity=8)
    prev_guide = None
    prev_guide_strict = None
    if prev_mask_u8 is not None and int(prev_mask_u8.sum()) > 0:
        prev_kernel_strict = morphology_kernel(strict_prev_dilate_kernel)
        prev_guide_strict = prev_mask_u8.copy()
        if prev_kernel_strict is not None:
            prev_guide_strict = cv2.dilate(prev_guide_strict, prev_kernel_strict, iterations=1)
        prev_kernel = morphology_kernel(prev_dilate_kernel)
        prev_guide = prev_mask_u8.copy()
        if prev_kernel is not None:
            prev_guide = cv2.dilate(prev_guide, prev_kernel, iterations=1)

    if prev_gate and prev_guide is not None:
        prev_area = int(prev_mask_u8.sum()) if prev_mask_u8 is not None else 0
        hard_overlap_mask = None
        if prev_mask_u8 is not None:
            hard_overlap_mask = np.logical_and(cur_mask_u8 > 0, prev_mask_u8 > 0).astype(np.uint8)
        strict_gated_mask = None
        if prev_guide_strict is not None:
            strict_gated_mask = np.logical_and(cur_mask_u8 > 0, prev_guide_strict > 0).astype(np.uint8)
        broad_gated_mask = np.logical_and(cur_mask_u8 > 0, prev_guide > 0).astype(np.uint8)

        overlap_min_pixels = max(int(min_mask_pixels), int(round(prev_area * overlap_keep_ratio)))
        strict_min_pixels = max(int(min_mask_pixels), int(round(prev_area * strict_keep_ratio)))
        if hard_overlap_mask is not None and int(hard_overlap_mask.sum()) >= overlap_min_pixels:
            selected_mask = hard_overlap_mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(selected_mask, connectivity=8)
        elif strict_gated_mask is not None and int(strict_gated_mask.sum()) >= strict_min_pixels:
            selected_mask = strict_gated_mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(selected_mask, connectivity=8)
        elif int(broad_gated_mask.sum()) >= int(min_mask_pixels):
            selected_mask = broad_gated_mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(selected_mask, connectivity=8)

    if num_labels > 1 and mode in ["largest_cc", "overlap_cc"]:
        best_label = 0
        best_key = None
        for label_i in range(1, num_labels):
            comp_area = int(stats[label_i, cv2.CC_STAT_AREA])
            if prev_guide is not None:
                overlap_area = int(np.logical_and(labels == label_i, prev_guide > 0).sum())
                key = (int(overlap_area > 0), overlap_area, comp_area)
            else:
                key = (comp_area,)
            if best_key is None or key > best_key:
                best_key = key
                best_label = label_i

        if best_label > 0:
            selected_mask = (labels == best_label).astype(np.uint8)
            if (
                prev_guide is not None
                and best_key is not None
                and best_key[0] == 0
                and fallback_to_prev
                and prev_mask_u8 is not None
                and int(prev_mask_u8.sum()) >= int(min_mask_pixels)
            ):
                selected_mask = prev_mask_u8.copy()

    open_kernel = morphology_kernel(post_open_kernel)
    if open_kernel is not None:
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_OPEN, open_kernel)

    close_kernel = morphology_kernel(post_close_kernel)
    if close_kernel is not None:
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = morphology_kernel(post_dilate_kernel)
    if dilate_kernel is not None:
        selected_mask = cv2.dilate(selected_mask, dilate_kernel, iterations=1)

    if (
        int(selected_mask.sum()) < int(min_mask_pixels)
        and fallback_to_prev
        and prev_mask_u8 is not None
        and int(prev_mask_u8.sum()) >= int(min_mask_pixels)
    ):
        selected_mask = prev_mask_u8.copy()

    return (selected_mask > 0).astype(np.uint8)


def masks_to_roi_geometry(
    mask_bin,
    output_dtype,
    output_device,
    image_hw,
    base_pad_scale,
    patch_size=16,
    patch_align=False,
    min_mask_pixels=16,
    external_mask_pad_scale=None,
    use_external_masks=False,
):
    """Convert binary masks `(B, T, H, W)` to roi centers/scales/whs."""
    B, target_num, H, W = mask_bin.shape
    roi_centers = torch.zeros(B, target_num, 2, device=output_device, dtype=output_dtype)
    scales = torch.zeros(B, target_num, 1, device=output_device, dtype=output_dtype)
    roi_whs = torch.zeros(B, target_num, 2, device=output_device, dtype=output_dtype)

    max_side = float(max(image_hw))
    fallback_scale = max_side
    fallback_center = torch.tensor(
        [(W - 1) * 0.5, (H - 1) * 0.5],
        device=output_device,
        dtype=output_dtype,
    )

    for b in range(B):
        for t in range(target_num):
            cur_mask = mask_bin[b, t]
            if int(cur_mask.sum().item()) < int(min_mask_pixels):
                cx, cy = fallback_center[0], fallback_center[1]
                bw = torch.tensor(float(W), device=output_device, dtype=output_dtype)
                bh = torch.tensor(float(H), device=output_device, dtype=output_dtype)
                scale = torch.tensor(fallback_scale, device=output_device, dtype=output_dtype)
            else:
                ys, xs = torch.where(cur_mask)
                x1, x2 = xs.min().to(output_dtype), xs.max().to(output_dtype)
                y1, y2 = ys.min().to(output_dtype), ys.max().to(output_dtype)
                bw = (x2 - x1 + 1.0).clamp(min=1.0)
                bh = (y2 - y1 + 1.0).clamp(min=1.0)
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                cur_pad_scale = (
                    float(external_mask_pad_scale)
                    if use_external_masks and external_mask_pad_scale is not None
                    else float(base_pad_scale)
                )
                scale = torch.maximum(bw, bh) * cur_pad_scale
                scale = torch.clamp(scale, min=1.0, max=max_side)

            if patch_align:
                xmin = cx - scale * 0.5
                ymin = cy - scale * 0.5
                xmax = cx + scale * 0.5
                ymax = cy + scale * 0.5
                px_min = torch.floor(xmin / patch_size)
                py_min = torch.floor(ymin / patch_size)
                px_max = torch.ceil(xmax / patch_size)
                py_max = torch.ceil(ymax / patch_size)
                p = torch.maximum(px_max - px_min, py_max - py_min)
                if int(p.item()) % 2 == 1:
                    p = p + 1.0
                cx_p = (px_min + px_max) * 0.5
                cy_p = (py_min + py_max) * 0.5
                px_min = cx_p - p * 0.5
                px_max = cx_p + p * 0.5
                py_min = cy_p - p * 0.5
                py_max = cy_p + p * 0.5
                xmin = px_min * patch_size
                xmax = px_max * patch_size
                ymin = py_min * patch_size
                ymax = py_max * patch_size
                cx = (xmin + xmax) * 0.5
                cy = (ymin + ymax) * 0.5
                scale = (xmax - xmin).clamp(min=1.0, max=max_side)
                bw = scale
                bh = scale

            roi_centers[b, t, 0] = cx
            roi_centers[b, t, 1] = cy
            scales[b, t, 0] = scale
            roi_whs[b, t, 0] = bw
            roi_whs[b, t, 1] = bh

    return roi_centers, scales, roi_whs


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
