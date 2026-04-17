import numpy as np
import torch


def get_num_spatial_views(cfg, default=1):
    """Read configured spatial-view count (>=1)."""
    try:
        v = int(cfg.MODEL.CDPN.PNP_NET.get("TRAIN_NUM_SPATIAL_VIEWS", default))
    except Exception:
        v = int(default)
    return max(v, 1)


def infer_vt_layout_from_length(total_length, num_spatial_views):
    """Infer (V, T) from flattened sequence length."""
    v = max(int(num_spatial_views), 1)
    n = int(total_length)
    if n <= 0:
        return 1, 0
    if n % v != 0:
        return 1, n
    return v, n // v


def maybe_reshape_nt_to_vt_tensor(tensor, num_spatial_views):
    """Reshape [N, ...] -> [V, T, ...] when divisible; else keep [1, N, ...]."""
    if (not torch.is_tensor(tensor)) or tensor.dim() < 1:
        return tensor
    n = int(tensor.shape[0])
    v, t = infer_vt_layout_from_length(n, num_spatial_views)
    return tensor.view(v, t, *tensor.shape[1:]).contiguous()


def flatten_bvt_tensor(tensor):
    """Flatten [B,V,T,...] -> [B,N,...], keep [B,N,...] unchanged."""
    if (not torch.is_tensor(tensor)) or tensor.dim() < 3:
        return tensor
    if tensor.dim() >= 4:
        b, v, t = tensor.shape[:3]
        return tensor.view(b, v * t, *tensor.shape[3:]).contiguous()
    # [B,V,T] scalar-like case
    b, v, t = tensor.shape
    return tensor.view(b, v * t).contiguous()


def flatten_vt_tensor(tensor):
    """Flatten [V,T,...] -> [N,...], keep [N,...] unchanged."""
    if (not torch.is_tensor(tensor)) or tensor.dim() < 2:
        return tensor
    if tensor.dim() >= 3:
        v, t = tensor.shape[:2]
        return tensor.view(v * t, *tensor.shape[2:]).contiguous()
    # [V,T]
    v, t = tensor.shape
    return tensor.view(v * t).contiguous()


def flatten_batch_spatiotemporal(batch):
    """Create a flattened [B,N,...] copy from [B,V,T,...] tensors in a dict."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() >= 3:
            out[key] = flatten_bvt_tensor(value)
        else:
            out[key] = value
    return out


def expand_time_indices_to_flat(time_indices, num_views, num_times):
    """Expand time-local target indices to flattened indices over views.

    Example:
      time_indices=[3,4,5], V=2, T=6 -> [3,4,5, 9,10,11]
    """
    if isinstance(time_indices, np.ndarray):
        t_ids = [int(v) for v in time_indices.reshape(-1).tolist()]
    elif torch.is_tensor(time_indices):
        t_ids = [int(v) for v in time_indices.reshape(-1).tolist()]
    elif isinstance(time_indices, (list, tuple)):
        t_ids = [int(v) for v in time_indices]
    else:
        t_ids = [int(time_indices)]
    if len(t_ids) == 0:
        return []
    v = max(int(num_views), 1)
    t = max(int(num_times), 1)
    norm_t_ids = []
    for tid in t_ids:
        tid_n = int(tid)
        if tid_n < 0:
            tid_n += t
        if tid_n < 0 or tid_n >= t:
            raise ValueError(f"time index out of range: tid={tid}, T={t}")
        norm_t_ids.append(tid_n)
    flat = []
    for view_i in range(v):
        base = view_i * t
        for tid_n in norm_t_ids:
            flat.append(base + tid_n)
    return flat


def maybe_expand_target_idx_for_bvt(target_idx, num_views, num_times):
    """If target_idx looks like time-local ids, expand to flattened view-time ids.

    If any index is >= num_times, treat input as already flattened.
    """
    if isinstance(target_idx, np.ndarray):
        ids = [int(v) for v in target_idx.reshape(-1).tolist()]
    elif torch.is_tensor(target_idx):
        ids = [int(v) for v in target_idx.reshape(-1).tolist()]
    elif isinstance(target_idx, (list, tuple)):
        ids = [int(v) for v in target_idx]
    else:
        ids = [int(target_idx)]
    if len(ids) == 0:
        return ids
    if max(ids) < int(num_times):
        return expand_time_indices_to_flat(ids, num_views=num_views, num_times=num_times)
    return ids


def reshape_flat_target_to_bvt(tensor_flat, batch_size, num_views, num_target_times):
    """Reshape [B*V*Tt,...] -> [B,V,Tt,...]."""
    if tensor_flat is None or (not torch.is_tensor(tensor_flat)):
        return tensor_flat
    b = int(batch_size)
    v = int(num_views)
    tt = int(num_target_times)
    if b <= 0 or v <= 0 or tt <= 0:
        return tensor_flat
    expected = b * v * tt
    if tensor_flat.shape[0] != expected:
        return tensor_flat
    return tensor_flat.view(b, v, tt, *tensor_flat.shape[1:]).contiguous()
