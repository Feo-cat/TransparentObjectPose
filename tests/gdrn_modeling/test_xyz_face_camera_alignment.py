import numpy as np
import torch

from core.gdrn_modeling.models.GDRN import GDRN


def _rotz(deg: float) -> torch.Tensor:
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def test_align_xyz_targets_rotates_with_face_camera_pose_no_offset():
    # one valid pixel at X_obj=(1,0,0), extent=(2,2,2) => normalized=(1.0,0.5,0.5)
    gt_xyz = torch.tensor([[[[1.0]], [[0.5]], [[0.5]]]], dtype=torch.float32)
    gt_xyz_bin = torch.tensor([[[[7]], [[4]], [[4]]]], dtype=torch.uint8)  # XYZ_BIN=8
    gt_mask_xyz = torch.ones((1, 1, 1), dtype=torch.float32)

    gt_rot_raw = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    gt_rot_supervise = _rotz(90.0).unsqueeze(0)  # R_sup = R_raw @ S, S=Rz(+90)

    model_infos = [{"symmetries_continuous": [{"axis": [0.0, 0.0, 1.0], "offset": [0.0, 0.0, 0.0]}]}]
    extents = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)

    xyz_new, xyz_bin_new = GDRN._align_xyz_targets_to_pose_supervision(
        gt_xyz=gt_xyz,
        gt_xyz_bin=gt_xyz_bin,
        gt_mask_xyz=gt_mask_xyz,
        gt_rot_raw=gt_rot_raw,
        gt_rot_supervise=gt_rot_supervise,
        model_infos_batch=model_infos,
        extents=extents,
        xyz_bin_num=8,
    )

    # X' = S^T X = Rz(-90) * (1,0,0) = (0,-1,0) => normalized=(0.5,0.0,0.5)
    expected = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32)
    assert torch.allclose(xyz_new[0, :, 0, 0], expected, atol=1e-5)

    # binized with clip to [0, 1): [4,0,4]
    assert int(xyz_bin_new[0, 0, 0, 0]) == 4
    assert int(xyz_bin_new[0, 1, 0, 0]) == 0
    assert int(xyz_bin_new[0, 2, 0, 0]) == 4


def test_align_xyz_targets_respects_symmetry_offset():
    # one valid pixel at X_obj=(2,0,0), offset=(1,0,0), extent=(4,4,4) => normalized=(1.0,0.5,0.5)
    gt_xyz = torch.tensor([[[[1.0]], [[0.5]], [[0.5]]]], dtype=torch.float32)
    gt_mask_xyz = torch.ones((1, 1, 1), dtype=torch.float32)

    gt_rot_raw = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    gt_rot_supervise = _rotz(90.0).unsqueeze(0)

    model_infos = [{"symmetries_continuous": [{"axis": [0.0, 0.0, 1.0], "offset": [1.0, 0.0, 0.0]}]}]
    extents = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float32)

    xyz_new, _ = GDRN._align_xyz_targets_to_pose_supervision(
        gt_xyz=gt_xyz,
        gt_xyz_bin=None,
        gt_mask_xyz=gt_mask_xyz,
        gt_rot_raw=gt_rot_raw,
        gt_rot_supervise=gt_rot_supervise,
        model_infos_batch=model_infos,
        extents=extents,
        xyz_bin_num=None,
    )

    # X' = S^T (X-o) + o = Rz(-90)*(1,0,0)+(1,0,0)=(1,-1,0)
    # normalized with extent=4: (0.75,0.25,0.5)
    expected = torch.tensor([0.75, 0.25, 0.5], dtype=torch.float32)
    assert torch.allclose(xyz_new[0, :, 0, 0], expected, atol=1e-5)


def test_region_supervision_recomputed_from_pose_aligned_xyz():
    # Aligned xyz points to metric (0, -1, 0), which should map to fps point #2.
    gt_xyz_supervise = torch.tensor([[[[0.5]], [[0.0]], [[0.5]]]], dtype=torch.float32)
    gt_region_raw = torch.tensor([[[1]]], dtype=torch.long)  # intentionally wrong
    gt_mask_region = torch.ones((1, 1, 1), dtype=torch.float32)
    extents = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    model_infos = [{"fps_points": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]}]

    gt_region_new = GDRN._recompute_region_targets_from_aligned_xyz(
        gt_xyz_supervise=gt_xyz_supervise,
        gt_region=gt_region_raw,
        gt_mask_region=gt_mask_region,
        model_infos_batch=model_infos,
        extents=extents,
        num_regions=2,
    )

    assert int(gt_region_new[0, 0, 0]) == 2
