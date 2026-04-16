from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GDRN_PATH = REPO_ROOT / "core" / "gdrn_modeling" / "models" / "GDRN.py"


def test_depth_mask_loss_uses_canonicalized_pose_targets():
    src = GDRN_PATH.read_text()
    # depth_mask_loss call should use canonicalized eval pose vars, not raw gt_ego_rot/gt_trans
    assert "depth_mask_loss(" in src
    assert "gt_rot=gt_rot_supervise_eval" in src
    assert "gt_trans=gt_trans_supervise_eval" in src


def test_geo_prior_losses_use_pose_aligned_xyz_targets():
    src = GDRN_PATH.read_text()
    # geo-prior debug/loss path should consume pose-aligned xyz targets
    assert "geo_prior_gt_xyz_pose_aligned" in src
    assert "gt_xyz=geo_prior_gt_xyz_pose_aligned" in src
    assert "roi_xyz=geo_prior_gt_xyz_aligned" in src


def test_region_loss_uses_pose_aligned_xyz_recomputed_targets():
    src = GDRN_PATH.read_text()
    assert "gt_region_supervise = self._recompute_region_targets_from_aligned_xyz(" in src
    assert "compute_region_loss(r_head_cfg, out_region, gt_region_supervise, gt_masks)" in src
