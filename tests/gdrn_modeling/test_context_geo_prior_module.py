import torch

from core.gdrn_modeling.models.context_geo_prior import ContextGeometricPriorInputs, ContextGeometricPriorModule


def _make_inputs(b=2, t=3, c=3):
    return ContextGeometricPriorInputs(
        target_roi_feat=torch.randn(b, t, 512, 8, 8),
        context_roi_feat=torch.randn(b, c, 512, 8, 8),
        context_xyz=torch.randn(b, c, 3, 8, 8),
        context_view_dir_obj=torch.randn(b, c, 3, 8, 8),
        context_cam_center_obj=torch.randn(b, c, 3, 8, 8),
        target_rgb=torch.randn(b, t, 3, 256, 256),
        context_rgb=torch.randn(b, c, 3, 256, 256),
        target_coord=torch.randn(b, t, 2, 8, 8),
        context_coord=torch.randn(b, c, 2, 8, 8),
        target_mask=torch.rand(b, t, 1, 8, 8),
        context_mask=torch.rand(b, c, 1, 8, 8),
        target_valid=torch.ones(b, t, dtype=torch.bool),
        context_valid=torch.ones(b, c, dtype=torch.bool),
    )


def test_context_geo_prior_returns_dual_scale_outputs():
    module = ContextGeometricPriorModule(
        roi_feat_dim=512,
        token_dim=128,
        retrieval_res=8,
        prior_res=64,
        rgb_embed_dim=16,
    )
    inputs = _make_inputs(b=2, t=3, c=3)
    out = module(inputs)

    assert out["retrieved_feat_8"].shape == (2, 3, 512, 8, 8)
    assert out["xyz_prior_64"].shape == (2, 3, 3, 64, 64)
    assert out["prior_conf_64"].shape == (2, 3, 1, 64, 64)
    assert out["prior_ambiguity_64"].shape == (2, 3, 1, 64, 64)


def test_context_geo_prior_invalid_context_zero_bank_weights():
    # With all-invalid context the module must still complete a full forward pass
    # (DDP safety: no per-rank branching).  The bank attention weights must be
    # zero (softmax of all-(-inf) → nan_to_num → 0), and prior_conf/xyz_prior
    # may be non-zero because the prior_head runs unconditionally on target
    # tokens — that is the intended DDP-safe behaviour.
    module = ContextGeometricPriorModule(
        roi_feat_dim=512,
        token_dim=128,
        retrieval_res=8,
        prior_res=64,
        rgb_embed_dim=16,
    )
    inputs = ContextGeometricPriorInputs(
        target_roi_feat=torch.randn(1, 2, 512, 8, 8),
        context_roi_feat=torch.randn(1, 3, 512, 8, 8),
        context_xyz=torch.randn(1, 3, 3, 8, 8),
        context_view_dir_obj=torch.randn(1, 3, 3, 8, 8),
        context_cam_center_obj=torch.randn(1, 3, 3, 8, 8),
        target_rgb=torch.randn(1, 2, 3, 256, 256),
        context_rgb=torch.randn(1, 3, 3, 256, 256),
        target_coord=torch.randn(1, 2, 2, 8, 8),
        context_coord=torch.randn(1, 3, 2, 8, 8),
        target_mask=torch.rand(1, 2, 1, 8, 8),
        context_mask=torch.zeros(1, 3, 1, 8, 8),
        target_valid=torch.ones(1, 2, dtype=torch.bool),
        context_valid=torch.zeros(1, 3, dtype=torch.bool),
    )
    out = module(inputs)
    # Bank weights must be all-zero when every context slot is invalid.
    assert torch.allclose(out["bank_weights_8"], torch.zeros_like(out["bank_weights_8"]), atol=1e-6)
    # Module must still produce properly shaped outputs (no NaN/Inf).
    assert not torch.isnan(out["prior_conf_64"]).any()
    assert not torch.isnan(out["xyz_prior_64"]).any()


def test_context_geo_prior_exposes_normalized_bank_weights():
    module = ContextGeometricPriorModule(
        roi_feat_dim=512,
        token_dim=64,
        retrieval_res=8,
        prior_res=64,
        rgb_embed_dim=8,
    )
    inputs = _make_inputs(b=1, t=1, c=3)
    out = module(inputs)

    weights = out["bank_weights_8"]
    assert weights.ndim == 3
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4)


def test_context_geo_prior_asserts_retrieval_res_matches_roi_spatial():
    module = ContextGeometricPriorModule(
        roi_feat_dim=512,
        token_dim=64,
        retrieval_res=4,
        prior_res=64,
        rgb_embed_dim=8,
    )
    inputs = _make_inputs(b=1, t=1, c=1)
    try:
        module(inputs)
        assert False, "Expected retrieval_res mismatch assertion"
    except AssertionError as exc:
        assert "retrieval_res" in str(exc)


def test_context_geo_prior_asserts_single_channel_context_mask():
    module = ContextGeometricPriorModule(
        roi_feat_dim=512,
        token_dim=64,
        retrieval_res=8,
        prior_res=64,
        rgb_embed_dim=8,
    )
    inputs = _make_inputs(b=1, t=1, c=1)
    inputs.context_mask = torch.rand(1, 1, 2, 8, 8)
    try:
        module(inputs)
        assert False, "Expected context_mask channel assertion"
    except AssertionError as exc:
        assert "context_mask" in str(exc)
