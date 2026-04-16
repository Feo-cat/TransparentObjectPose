from pathlib import Path

from mmcv import Config

from core.gdrn_modeling.models.GDRN import get_pnp_net_input_channels
import inference_batch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_CFG_PATH = REPO_ROOT / "configs" / "gdrn" / "labsim" / "a6_cPnP_lm13_ctx_prior.py"


def test_inference_batch_pnp_channels_match_gdrn_builder():
    cfg = Config.fromfile(str(EXP_CFG_PATH))
    expected = get_pnp_net_input_channels(cfg.MODEL.CDPN)
    actual = inference_batch.get_inference_pnp_net_input_channels(cfg)
    assert actual == expected
