from pathlib import Path
import runpy

import pytest
import torch

from core.gdrn_modeling.models.GDRN import (
    build_roi_feat_encoder,
    resolve_dino_and_decoder_depths,
    resolve_roi_decoder_feature_dims,
    resolve_roi_feat_input_res,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CFG_PATH = REPO_ROOT / "configs" / "_base_" / "gdrn_base.py"


def test_base_cfg_has_dino_decoder_depth_defaults():
    cfg_dict = runpy.run_path(str(BASE_CFG_PATH))
    cdpn_cfg = cfg_dict["MODEL"]["CDPN"]

    assert int(cdpn_cfg["DINO_NUM_BLOCKS"]) == 12
    assert int(cdpn_cfg["ATTN_DECODER_DEPTH"]) == 6
    assert int(cdpn_cfg["ROI_DECODER_DEPTH"]) == 5
    assert int(cdpn_cfg["ROI_FEAT_INPUT_RES"]) == 256
    assert int(cdpn_cfg["ROI_DECODER_DIM"]) == 512
    assert int(cdpn_cfg["ROI_HEAD_OUT_DIM"]) == 128


def test_resolve_dino_and_decoder_depths_accepts_valid_overrides():
    cdpn_cfg = dict(
        DINO_NUM_BLOCKS=8,
        ATTN_DECODER_DEPTH=4,
        ROI_DECODER_DEPTH=3,
    )
    dino_num_blocks, attn_decoder_depth, roi_decoder_depth = resolve_dino_and_decoder_depths(
        cdpn_cfg, dino_total_blocks=12
    )

    assert dino_num_blocks == 8
    assert attn_decoder_depth == 4
    assert roi_decoder_depth == 3


def test_resolve_dino_and_decoder_depths_enforces_decoder_depth_constraints():
    with pytest.raises(AssertionError, match="ATTN_DECODER_DEPTH"):
        resolve_dino_and_decoder_depths(dict(ATTN_DECODER_DEPTH=1), dino_total_blocks=12)

    with pytest.raises(AssertionError, match="ROI_DECODER_DEPTH"):
        resolve_dino_and_decoder_depths(dict(ROI_DECODER_DEPTH=0), dino_total_blocks=12)


def test_resolve_dino_and_decoder_depths_enforces_dino_block_range():
    with pytest.raises(AssertionError, match="DINO_NUM_BLOCKS"):
        resolve_dino_and_decoder_depths(dict(DINO_NUM_BLOCKS=0), dino_total_blocks=12)

    with pytest.raises(AssertionError, match="DINO_NUM_BLOCKS"):
        resolve_dino_and_decoder_depths(dict(DINO_NUM_BLOCKS=13), dino_total_blocks=12)


def test_resolve_roi_feat_input_res_accepts_valid_values():
    assert resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=256)) == 256
    assert resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=128)) == 128
    assert resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=64)) == 64


def test_resolve_roi_feat_input_res_rejects_invalid_values():
    with pytest.raises(AssertionError, match=">= 16"):
        resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=8))
    with pytest.raises(AssertionError, match="divisible by 8"):
        resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=130))
    with pytest.raises(AssertionError, match="power of two"):
        resolve_roi_feat_input_res(dict(ROI_FEAT_INPUT_RES=24))


def test_resolve_roi_decoder_feature_dims_accepts_valid_values():
    dec_dim, head_out_dim = resolve_roi_decoder_feature_dims(
        dict(ROI_DECODER_DIM=384, ROI_HEAD_OUT_DIM=96),
        dec_num_heads=16,
    )
    assert dec_dim == 384
    assert head_out_dim == 96


def test_resolve_roi_decoder_feature_dims_rejects_invalid_values():
    with pytest.raises(AssertionError, match="ROI_DECODER_DIM"):
        resolve_roi_decoder_feature_dims(dict(ROI_DECODER_DIM=0), dec_num_heads=16)
    with pytest.raises(AssertionError, match="divisible"):
        resolve_roi_decoder_feature_dims(dict(ROI_DECODER_DIM=390), dec_num_heads=16)
    with pytest.raises(AssertionError, match="ROI_HEAD_OUT_DIM"):
        resolve_roi_decoder_feature_dims(dict(ROI_HEAD_OUT_DIM=0), dec_num_heads=16)


@pytest.mark.parametrize("input_res", [64, 128, 256])
def test_build_roi_feat_encoder_preserves_8x8_512_shape(input_res):
    encoder = build_roi_feat_encoder(input_res=input_res, out_res=8, in_channels=128, final_channels=512)
    x = torch.randn(2, 128, input_res, input_res)
    y = encoder(x)
    assert y.shape == (2, 512, 8, 8)
