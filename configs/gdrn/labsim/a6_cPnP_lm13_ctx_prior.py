_base_ = "./a6_cPnP_lm13.py"


MODEL = dict(
    CDPN=dict(
        VIEW_INTERACTION=dict(
            ENABLED=True,
            GEOMETRIC_PRIOR=dict(
                ENABLED=True,
                RETRIEVAL_RES=8,
                PRIOR_RES=64,
                WITH_8X8_RETRIEVAL=True,
                WITH_64X64_PRIOR=True,
                INJECT_TO_PNP=True,
                APPEND_PRIOR_CONF=True,
                APPEND_PRIOR_RESIDUAL=True,
            ),
            GEOMETRIC_PRIOR_LOSS=dict(
                XYZ_LW=1.0,
                CONF_LW=0.2,
                CONF_ERR_THR=0.08,
            ),
            RELATIVE_GEOMETRY_LOSS=dict(
                ENABLED=True,
                LW=0.25,
                TRANS_LW=0.05,
                NUM_POINTS=1024,
            ),
        ),
    ),
)

TRAIN = dict(
    BAD_CASE_STATS_ENABLED=False,
)
