from pathlib import Path

from mmcv import Config

from core.gdrn_modeling.models.GDRN import (
    get_geo_prior_extra_pnp_channels,
    get_geo_prior_hyp_source,
    get_geo_prior_loss_cfg,
    get_pnp_net_input_channels,
    validate_geo_prior_resolution,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CFG_PATH = REPO_ROOT / "configs" / "_base_" / "gdrn_base.py"
EXP_CFG_PATH = REPO_ROOT / "configs" / "gdrn" / "labsim" / "a6_cPnP_lm13_ctx_prior.py"


def test_geo_prior_defaults_exist():
    cfg = Config.fromfile(str(BASE_CFG_PATH))
    vi_cfg = cfg.MODEL.CDPN.VIEW_INTERACTION
    gp_cfg = vi_cfg.GEOMETRIC_PRIOR
    rel_cfg = vi_cfg.RELATIVE_GEOMETRY_LOSS

    assert gp_cfg.ENABLED is False
    assert gp_cfg.RETRIEVAL_RES == 8
    assert gp_cfg.PRIOR_RES == 64
    assert gp_cfg.WITH_8X8_RETRIEVAL is True
    assert gp_cfg.WITH_64X64_PRIOR is True
    assert gp_cfg.INJECT_TO_PNP is True
    assert rel_cfg.ENABLED is False
    assert rel_cfg.NUM_POINTS == 1024


def test_geo_prior_extra_pnp_channels():
    cfg = Config.fromfile(str(BASE_CFG_PATH))
    gp_cfg = cfg.MODEL.CDPN.VIEW_INTERACTION.GEOMETRIC_PRIOR

    assert get_geo_prior_extra_pnp_channels(gp_cfg) == 0

    gp_cfg.ENABLED = True
    gp_cfg.INJECT_TO_PNP = True
    gp_cfg.APPEND_PRIOR_CONF = True
    gp_cfg.APPEND_PRIOR_RESIDUAL = True

    assert get_geo_prior_extra_pnp_channels(gp_cfg) == 7


def test_pnp_input_channels_match_manual_formula_for_experiment_config():
    cfg = Config.fromfile(str(EXP_CFG_PATH))
    cdpn = cfg.MODEL.CDPN
    r_head_cfg = cdpn.ROT_HEAD
    pnp_cfg = cdpn.PNP_NET

    if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
        expected = 3 * r_head_cfg.XYZ_BIN
    else:
        expected = 3
    if pnp_cfg.WITH_2D_COORD:
        expected += 2
    if pnp_cfg.REGION_ATTENTION:
        expected += r_head_cfg.NUM_REGIONS
    if pnp_cfg.MASK_ATTENTION in ["concat"]:
        expected += 1
    expected += 7

    assert get_pnp_net_input_channels(cdpn) == expected


def test_get_geo_prior_hyp_source_defaults_when_geo_prior_missing():
    vi_cfg = dict(RELATIVE_GEOMETRY_LOSS=dict(ENABLED=True))
    assert get_geo_prior_hyp_source(vi_cfg) == "noisy_gt"


def test_get_geo_prior_loss_cfg_defaults_when_missing():
    vi_cfg = dict(RELATIVE_GEOMETRY_LOSS=dict(ENABLED=True))
    assert get_geo_prior_loss_cfg(vi_cfg) == {}


def test_validate_geo_prior_resolution_raises_when_mismatch():
    gp_cfg = dict(ENABLED=True, PRIOR_RES=32)
    try:
        validate_geo_prior_resolution(gp_cfg, output_res=64)
        assert False, "Expected PRIOR_RES/OUTPUT_RES mismatch assertion"
    except AssertionError as exc:
        assert "PRIOR_RES" in str(exc)


def test_pnp_input_channels_works_without_view_interaction_cfg():
    cfg = Config.fromfile(str(EXP_CFG_PATH))
    cdpn = cfg.MODEL.CDPN
    if "VIEW_INTERACTION" in cdpn:
        del cdpn["VIEW_INTERACTION"]

    r_head_cfg = cdpn.ROT_HEAD
    pnp_cfg = cdpn.PNP_NET
    if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
        expected = 3 * r_head_cfg.XYZ_BIN
    else:
        expected = 3
    if pnp_cfg.WITH_2D_COORD:
        expected += 2
    if pnp_cfg.REGION_ATTENTION:
        expected += r_head_cfg.NUM_REGIONS
    if pnp_cfg.MASK_ATTENTION in ["concat"]:
        expected += 1

    assert get_pnp_net_input_channels(cdpn) == expected


def test_experiment_config_sets_view_interaction_enabled():
    cfg = Config.fromfile(str(EXP_CFG_PATH))
    assert bool(cfg.MODEL.CDPN.VIEW_INTERACTION.ENABLED) is True


def test_geo_prior_loss_conf_err_thr_default_and_experiment_override():
    base_cfg = Config.fromfile(str(BASE_CFG_PATH))
    exp_cfg = Config.fromfile(str(EXP_CFG_PATH))

    assert float(base_cfg.MODEL.CDPN.VIEW_INTERACTION.GEOMETRIC_PRIOR_LOSS.CONF_ERR_THR) == 0.05
    assert float(exp_cfg.MODEL.CDPN.VIEW_INTERACTION.GEOMETRIC_PRIOR_LOSS.CONF_ERR_THR) == 0.08
