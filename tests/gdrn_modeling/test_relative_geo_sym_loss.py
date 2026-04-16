import math

import torch

from core.gdrn_modeling.losses.relative_geo_sym_loss import RelativeGeoSymLoss


def test_relative_geo_sym_loss_ignores_spin_around_symmetry_axis():
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=513, dtype=torch.float32)[:-1]
    ring_low = torch.stack([torch.cos(angles), torch.sin(angles), -torch.ones_like(angles)], dim=1)
    ring_high = torch.stack([torch.cos(angles), torch.sin(angles), torch.ones_like(angles)], dim=1)
    points = torch.cat([ring_low, ring_high], dim=0).unsqueeze(0)

    loss_fn = RelativeGeoSymLoss()
    pred_rot = torch.eye(3).unsqueeze(0)
    ctx_rot = torch.eye(3).unsqueeze(0)
    theta = math.radians(37.0)
    gt_rot = torch.tensor(
        [[
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    pred_t = torch.zeros(1, 3)
    ctx_t = torch.zeros(1, 3)
    gt_t = torch.zeros(1, 3)

    loss = loss_fn(
        pred_rot=pred_rot,
        pred_trans=pred_t,
        ctx_hyp_rot=ctx_rot,
        ctx_hyp_trans=ctx_t,
        gt_rot=gt_rot,
        gt_trans=gt_t,
        ctx_gt_rot=ctx_rot,
        ctx_gt_trans=ctx_t,
        model_points=points,
    )

    assert float(loss["loss_rel_geo_sym"]) < 2e-2


def test_relative_geo_sym_loss_penalizes_wrong_relative_translation():
    points = torch.randn(1, 128, 3)
    loss_fn = RelativeGeoSymLoss()

    out = loss_fn(
        pred_rot=torch.eye(3).unsqueeze(0),
        pred_trans=torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32),
        ctx_hyp_rot=torch.eye(3).unsqueeze(0),
        ctx_hyp_trans=torch.zeros(1, 3),
        gt_rot=torch.eye(3).unsqueeze(0),
        gt_trans=torch.zeros(1, 3),
        ctx_gt_rot=torch.eye(3).unsqueeze(0),
        ctx_gt_trans=torch.zeros(1, 3),
        model_points=points,
    )

    assert float(out["loss_rel_geo_sym"]) > 0.01
    assert float(out["loss_rel_trans"]) > 0.01


def test_relative_geo_sym_loss_respects_max_points_truncation():
    torch.manual_seed(0)
    points = torch.randn(2, 32, 3)
    loss_fn = RelativeGeoSymLoss()

    pred_rot = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    pred_trans = torch.randn(2, 3)
    ctx_hyp_rot = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    ctx_hyp_trans = torch.randn(2, 3)
    gt_rot = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    gt_trans = torch.randn(2, 3)
    ctx_gt_rot = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    ctx_gt_trans = torch.randn(2, 3)

    out_trunc = loss_fn(
        pred_rot=pred_rot,
        pred_trans=pred_trans,
        ctx_hyp_rot=ctx_hyp_rot,
        ctx_hyp_trans=ctx_hyp_trans,
        gt_rot=gt_rot,
        gt_trans=gt_trans,
        ctx_gt_rot=ctx_gt_rot,
        ctx_gt_trans=ctx_gt_trans,
        model_points=points[:, :8, :],
    )
    out_max_points = loss_fn(
        pred_rot=pred_rot,
        pred_trans=pred_trans,
        ctx_hyp_rot=ctx_hyp_rot,
        ctx_hyp_trans=ctx_hyp_trans,
        gt_rot=gt_rot,
        gt_trans=gt_trans,
        ctx_gt_rot=ctx_gt_rot,
        ctx_gt_trans=ctx_gt_trans,
        model_points=points,
        max_points=8,
    )

    assert torch.allclose(out_trunc["loss_rel_geo_sym"], out_max_points["loss_rel_geo_sym"], atol=1e-6)
    assert torch.allclose(out_trunc["loss_rel_trans"], out_max_points["loss_rel_trans"], atol=1e-6)
