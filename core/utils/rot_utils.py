import torch
import math
import numpy as np

def rotation_matrix_to_axis_angle_safe(R):
    """
    Safe version: handles theta ~ 0 and theta ~ pi.
    R: [...,3,3]
    returns:
        axis: [...,3]
        angle: [...]
    """
    eps = 1e-6

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) * 0.5
    cos_theta = torch.clamp(cos_theta, -1, 1)

    theta = torch.acos(cos_theta)

    axis = torch.zeros((*R.shape[:-2], 3), device=R.device, dtype=R.dtype)

    # general case
    mask = (theta > eps) & (torch.abs(theta - math.pi) > eps)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    v = torch.stack([rx, ry, rz], dim=-1)

    axis[mask] = v[mask] / (2 * torch.sin(theta[mask])[..., None])

    # theta ~ pi case
    mask_pi = torch.abs(theta - math.pi) <= eps
    if mask_pi.any():
        RpI = R[mask_pi] + torch.eye(3, device=R.device, dtype=R.dtype)
        axis_pi = torch.zeros_like(RpI[..., 0])
        for i in range(3):
            col = RpI[..., :, i]
            norm = col.norm(dim=-1)
            valid = norm > eps
            axis_pi[valid] = col[valid] / norm[valid, None]
            if valid.any():
                break
        axis[mask_pi] = axis_pi

    axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    return axis, theta



def extract_axis_from_rotation_matrix(T):
    """
    T: [4,4] or [3,3]
    return: [3]
    """
    if T.shape[-1] == 4:
        R = T[:3, :3]
    else:
        R = T

    axis, _ = rotation_matrix_to_axis_angle_safe(R[None])
    return axis[0]



def find_symmetry_order(T, max_order=12):
    """
    找最小 n 使 R^n ≈ I
    """
    if T.shape[-1] == 4:
        R = T[:3, :3]
    else:
        R = T

    I = torch.eye(3, device=R.device, dtype=R.dtype)

    Rn = torch.eye(3, device=R.device, dtype=R.dtype)
    for n in range(1, max_order + 1):
        Rn = Rn @ R
        if torch.allclose(Rn, I, atol=1e-3):
            return n

    raise ValueError("Symmetry order not found, increase max_order.")




def extract_theta_along_axis(R, axis):
    """
    R: [B,3,3]
    axis: [3]
    return:
        theta: [B]
    """
    axis = axis / axis.norm()
    axis = axis[None]

    K = torch.zeros((3,3), device=R.device, dtype=R.dtype)
    K[0,1] = -axis[0,2]
    K[0,2] =  axis[0,1]
    K[1,0] =  axis[0,2]
    K[1,2] = -axis[0,0]
    K[2,0] = -axis[0,1]
    K[2,1] =  axis[0,0]

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)

    sin_theta = 0.5 * (
        R[..., 2,1] - R[..., 1,2]
    ) * axis[0,0] + \
    0.5 * (
        R[..., 0,2] - R[..., 2,0]
    ) * axis[0,1] + \
    0.5 * (
        R[..., 1,0] - R[..., 0,1]
    ) * axis[0,2]

    theta = torch.atan2(sin_theta, cos_theta)
    return theta


def rotmat_to_6d(R):
    """
    R: [..., 3, 3]
    return: [..., 6]
    """
    return R[..., :2].reshape(*R.shape[:-2], 6)


def build_discrete_sym_code(R, sym_mats_4x4):
    """
    R: [B,3,3]
    sym_mats_4x4: [K,4,4]
    return:
        sym_code: [B,5]
    """

    B = R.shape[0]
    device = R.device
    dtype = R.dtype

    sym_mats = sym_mats_4x4.to(device=device, dtype=dtype)
    sym_rots = sym_mats[:, :3, :3]  # [K,3,3]

    # find one non-identity rotation
    I = torch.eye(3, device=device, dtype=dtype)
    diffs = (sym_rots - I).abs().sum(dim=(1,2))
    idx = torch.argmax(diffs)
    R0 = sym_rots[idx]

    axis = extract_axis_from_rotation_matrix(R0)
    axis = axis / axis.norm()

    n = sym_rots.shape[0]

    theta = extract_theta_along_axis(R, axis)
    theta_mod = torch.remainder(theta, 2 * math.pi / n)

    sc = torch.stack([torch.sin(theta_mod), torch.cos(theta_mod)], dim=-1)
    axis = axis.expand(B, 3)

    return torch.cat([axis, sc], dim=-1)



def build_symmetry_pose_code(R, sym_type=None, sym_info=None):
    """
    R: [B,3,3]
    sym_type: "continuous" or "discrete"
    sym_info:
        continuous: {"axis": [3], "offset": [3]}
        discrete: {"mat": [4,4]}
    return:
        pose_code: [B,12】
        pose_code = [ sym5 | nonsym6 | flag ], flag = 0 for nonsymmetry, 1 for continuous symmetry, 2 for discrete symmetry
    """

    B = R.shape[0]
    device = R.device
    dtype = R.dtype
    
    if sym_type is None or sym_info is None:
        nonsym_code = rotmat_to_6d(R)
        sym_code = torch.zeros((B, 5), device=device, dtype=dtype)
        flag = torch.zeros((B, 1), device=device, dtype=dtype)
        return torch.cat([sym_code, nonsym_code, flag], dim=-1)

    if sym_type == "continuous":
        axis = torch.tensor(sym_info["axis"], device=device, dtype=dtype)
        axis = axis / axis.norm()
        axis = axis.expand(B, 3)

        sc = torch.zeros((B, 2), device=device, dtype=dtype)
        sym_code = torch.cat([axis, sc], dim=-1)
        nonsym_code = torch.zeros((B, 6), device=device, dtype=dtype)
        flag = torch.ones((B, 1), device=device, dtype=dtype)
        return torch.cat([sym_code, nonsym_code, flag], dim=-1)

    elif sym_type == "discrete":
        sym_code = build_discrete_sym_code(R, sym_info)
        
        nonsym_code = torch.zeros((B, 6), device=device, dtype=dtype)
        flag = torch.ones((B, 1), device=device, dtype=dtype) * 2
        return torch.cat([sym_code, nonsym_code, flag], dim=-1)

    else:
        raise ValueError(f"Unknown symmetry type {sym_type}")
    
    
if __name__ == "__main__":
    sym_discrete = torch.tensor([[
        [-1,0,0,0],
        [0,-1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ]], dtype=torch.float32)
    
    sym_continuous = {
        "axis": [0,0,1],
        "offset": [0,0,0]
    }
    
    # discrete symmetry
    R = torch.eye(3)[None].cuda()   # [1,3,3]
    print(sym_discrete.shape)

    code = build_symmetry_pose_code(
        R,
        sym_type="continuous",
        sym_info=sym_continuous
    )
    
    print(code)
    # [0,0,1,0,0]
    
    # continuous symmetry
    code = build_symmetry_pose_code(
        R,
        sym_type="discrete",
        sym_info=sym_discrete
    )
    print(code)
    
    # no symmetry
    code = build_symmetry_pose_code(
        R,
        sym_type=None,
        sym_info=None
    )
    print(code)