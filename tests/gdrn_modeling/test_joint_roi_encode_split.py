import types

import torch

from core.gdrn_modeling.models.GDRN import GDRN


def _fake_encode(self, view_hidden, roi_centers, scales, H, W):
    del roi_centers, scales, H, W
    if view_hidden.dim() == 4:
        batch_size, view_num = view_hidden.shape[:2]
    elif view_hidden.dim() == 3:
        batch_size, view_num = view_hidden.shape[0], 1
    else:
        raise ValueError(f"Unsupported dim: {view_hidden.dim()}")
    feat = torch.arange(
        batch_size * view_num * 2 * 2 * 2,
        dtype=view_hidden.dtype,
        device=view_hidden.device,
    )
    return feat.reshape(batch_size * view_num, 2, 2, 2)


def test_joint_roi_encode_split_matches_joint_tensor_split():
    model = GDRN.__new__(GDRN)
    model._encode_view_tokens_to_roi_features = types.MethodType(_fake_encode, model)

    B, T, C, HW, D = 2, 3, 2, 4, 5
    target_hidden = torch.randn(B, T, HW, D)
    context_hidden = torch.randn(B, C, HW, D)
    target_centers = torch.randn(B, T, 2)
    target_scales = torch.ones(B, T, 1)
    context_centers = torch.randn(B, C, 2)
    context_scales = torch.ones(B, C, 1)

    target_feat, context_feat = model._encode_target_and_context_roi_features(
        target_hidden=target_hidden,
        target_roi_centers=target_centers,
        target_scales=target_scales,
        context_hidden=context_hidden,
        context_roi_centers=context_centers,
        context_scales=context_scales,
        H=64,
        W=64,
    )

    joint_hidden = torch.cat([target_hidden, context_hidden], dim=1)
    joint_centers = torch.cat([target_centers, context_centers], dim=1)
    joint_scales = torch.cat([target_scales, context_scales], dim=1)
    joint_feat = _fake_encode(model, joint_hidden, joint_centers, joint_scales, 64, 64)
    joint_feat = joint_feat.reshape(B, T + C, *joint_feat.shape[1:])

    expected_target = joint_feat[:, :T].reshape(B * T, *joint_feat.shape[2:]).contiguous()
    expected_context = joint_feat[:, T:].reshape(B * C, *joint_feat.shape[2:]).contiguous()

    assert torch.equal(target_feat, expected_target)
    assert torch.equal(context_feat, expected_context)


def test_joint_roi_encode_split_supports_single_target_view_tensor():
    model = GDRN.__new__(GDRN)
    model._encode_view_tokens_to_roi_features = types.MethodType(_fake_encode, model)

    B, C, HW, D = 2, 2, 4, 5
    target_hidden = torch.randn(B, HW, D)
    context_hidden = torch.randn(B, C, HW, D)
    target_centers = torch.randn(B, 2)
    target_scales = torch.ones(B, 1)
    context_centers = torch.randn(B, C, 2)
    context_scales = torch.ones(B, C, 1)

    target_feat, context_feat = model._encode_target_and_context_roi_features(
        target_hidden=target_hidden,
        target_roi_centers=target_centers,
        target_scales=target_scales,
        context_hidden=context_hidden,
        context_roi_centers=context_centers,
        context_scales=context_scales,
        H=64,
        W=64,
    )

    assert target_feat.shape[0] == B
    assert context_feat.shape[0] == B * C

