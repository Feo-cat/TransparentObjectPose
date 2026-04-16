import torch
from torch import nn

from core.gdrn_modeling.models.GDRN import (
    align_geo_prior_debug_targets,
    assemble_pnp_inputs_with_geo_prior,
    build_module_ddp_dummy_anchor_loss,
    compute_geo_prior_debug_metrics,
    compute_geo_prior_training_losses,
    should_inject_geo_prior_features,
)
from core.gdrn_modeling.view_interaction_utils import build_context_pose_confidence


def test_assemble_pnp_inputs_appends_prior_channels():
    coor_feat = torch.randn(2, 5, 64, 64)
    roi_coord = torch.randn(2, 2, 64, 64)
    prior = {
        "xyz_prior_64": torch.randn(2, 3, 64, 64),
        "prior_conf_64": torch.rand(2, 1, 64, 64),
    }
    out = assemble_pnp_inputs_with_geo_prior(
        coor_feat=coor_feat,
        roi_coord_2d_pnp=roi_coord,
        xyz_single=torch.randn(2, 3, 64, 64),
        geo_prior=prior,
        with_2d_coord=True,
        append_prior_conf=True,
        append_prior_residual=True,
    )
    assert out.shape == (2, 14, 64, 64)


def test_compute_geo_prior_training_losses_returns_expected_keys():
    losses = compute_geo_prior_training_losses(
        xyz_prior=torch.randn(2, 3, 64, 64),
        prior_conf=torch.rand(2, 1, 64, 64),
        prior_ambiguity=torch.rand(2, 1, 64, 64),
        roi_xyz=torch.randn(2, 3, 64, 64),
        roi_mask_visib=torch.ones(2, 1, 64, 64),
        cfg_loss=dict(XYZ_LW=1.0, CONF_LW=0.2, CONF_ERR_THR=0.05),
    )
    assert "loss_xyz_prior" in losses
    assert "loss_prior_conf" in losses
    assert "loss_prior_ambiguity" in losses


def test_build_context_pose_confidence_marks_identity_zero_pose_low_conf():
    rot = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
    trans = torch.zeros(1, 3, 3)
    conf = build_context_pose_confidence(rot, trans)
    assert conf.shape == (1, 3)
    assert torch.all(conf == 0.0)


def test_should_inject_geo_prior_features_uses_bank_conf_presence():
    assert should_inject_geo_prior_features(None) is False
    assert should_inject_geo_prior_features({"bank_conf": torch.zeros(0)}) is False
    assert should_inject_geo_prior_features({"bank_conf": torch.ones(1)}) is True


def test_compute_geo_prior_debug_metrics_reports_bank_and_confidence():
    geo_prior_out = {
        "bank_conf": torch.ones(6),
        "bank_weights_8": torch.zeros(2, 4, 4),  # B=2, K=4 => total=8
    }
    geo_prior_out_flat = {
        "xyz_prior_64": torch.ones(2, 3, 2, 2),
        "prior_conf_64": torch.full((2, 1, 2, 2), 0.25),
        "prior_ambiguity_64": torch.full((2, 1, 2, 2), 0.75),
    }
    gt_xyz = torch.zeros(2, 3, 2, 2)
    gt_mask = torch.ones(2, 1, 2, 2)

    out = compute_geo_prior_debug_metrics(
        geo_prior_out=geo_prior_out,
        geo_prior_out_flat=geo_prior_out_flat,
        geo_prior_active=True,
        prior_was_dropped=False,
        is_all_target=False,
        gt_xyz=gt_xyz,
        gt_mask_visib=gt_mask,
    )
    assert out["vis/geo_prior_active"] == 1.0
    assert out["vis/geo_prior_has_bank"] == 1.0
    assert abs(out["vis/geo_prior_bank_valid_ratio"] - 0.75) < 1e-6
    assert abs(out["vis/prior_conf_mean"] - 0.25) < 1e-6
    assert abs(out["vis/prior_conf_obj_mean"] - 0.25) < 1e-6
    assert abs(out["vis/prior_ambiguity_mean"] - 0.75) < 1e-6
    assert abs(out["vis/prior_xyz_abs_err"] - 1.0) < 1e-6


def test_align_geo_prior_debug_targets_handles_channel_first_swapped_layout():
    xyz_prior = torch.ones(36, 3, 2, 2)
    gt_xyz = torch.zeros(3, 36, 2, 2)
    gt_mask = torch.ones(1, 36, 2, 2)
    gt_xyz_aligned, gt_mask_aligned = align_geo_prior_debug_targets(
        xyz_prior=xyz_prior, gt_xyz=gt_xyz, gt_mask_visib=gt_mask
    )
    assert gt_xyz_aligned is not None
    assert gt_mask_aligned is not None
    assert gt_xyz_aligned.shape == xyz_prior.shape
    assert gt_mask_aligned.shape == (36, 1, 2, 2)


def test_align_geo_prior_debug_targets_returns_none_for_unrecoverable_layout():
    xyz_prior = torch.ones(4, 3, 2, 2)
    gt_xyz = torch.zeros(5, 3, 2, 2)
    gt_mask = torch.ones(5, 1, 2, 2)
    gt_xyz_aligned, gt_mask_aligned = align_geo_prior_debug_targets(
        xyz_prior=xyz_prior, gt_xyz=gt_xyz, gt_mask_visib=gt_mask
    )
    assert gt_xyz_aligned is None
    assert gt_mask_aligned is None


def test_build_module_ddp_dummy_anchor_loss_touches_all_trainable_params():
    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    ref = torch.randn(2, 3, requires_grad=True)
    loss = build_module_ddp_dummy_anchor_loss(module, reference_tensor=ref)
    assert float(loss.detach().item()) == 0.0
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None
