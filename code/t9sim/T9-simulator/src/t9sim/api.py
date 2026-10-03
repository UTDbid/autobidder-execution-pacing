"""Public API — the three-step benchmark loop in three calls.

    import t9sim.api as t9

    t9.generate("golden", seed=90213)              # 1. generate a master (100K)
    results = t9.evaluate("golden")                # 2. train + score reference
                                                   #    models under C1-C4 vs truth
    df_c1 = t9.view(master_df, "C1")               # 3. (or) censor a master to one
                                                   #    view to train YOUR method

The paper's operative configuration (schema V10, "Private-Rival Market") is
``V10_EDGES`` with ``rho=0.8`` — the externally anchored privateness point:

    t9.generate("test", seed=90213, rho=0.8, bn_edges=t9.V10_EDGES)

Conditions (censored views over ONE master):
    C1 = DSP only          C2 = DSP+MMP (funnel labels on all rows)
    C3 = DSP+SSP (clearing prices / rival count)   C4 = all layers.

Everything here delegates to the underlying modules (`simulate`, `censor`,
`pipeline`) — scripted pipelines can keep using those directly.
"""
from __future__ import annotations

from . import censor as _censor
from . import pipeline as _pipeline
from . import simulate as _simulate

# Schema V10 = v7 funnel BN edges (#1-#5) + the private-rival market layer.
# (B1 `min_bid_to_win` and B2 `explore_traffic` are inert flags from an
#  abandoned line of work: off, unreported, and not part of the schema.)
V10_EDGES = {
    "pairing": True, "os_spend": True, "payer_timing": True,
    "hour": True, "exposure": True,
    "rival_pool": True,
}

#: Anchored privateness (value-of-lost-price-information band; see docs/v10_Anchor_Bands.md)
RHO_STAR = 0.8

CONDITIONS = ("C1", "C2", "C3", "C4")


def generate(profile="golden", seed=None, rho=None, bn_edges=None, out_name=None):
    """Generate a master dataset (funnel outcomes on ALL rows + latents kept).

    profile: golden=100K · test=1M · scale10m=10M (see config/profiles.yaml)
    seed:    RNG seed; same (profile, seed, rho, edges) -> byte-identical data.
    rho:     rival-bid privateness in [0,1]; the paper's anchored point is 0.8.
    bn_edges: dependency-layer flags; use ``V10_EDGES`` for the paper schema.
    out_name: output subdir under output/ (defaults to the profile name).

    Returns True when all generation-time direction checks pass.
    """
    kw = {}
    if rho is not None:
        kw["rho"] = rho
    return _simulate.run(profile, seed=seed, bn_edges=bn_edges,
                         out_name=out_name, **kw)


def view(master_df, condition, keep_latents=False):
    """Censor a master DataFrame to one condition's visible view (a mask,
    never a re-simulation). C1/C3 blank funnel labels on lost rows; C1/C2
    blank clearing prices. Latents/estimands are dropped unless
    ``keep_latents`` (oracle diagnostics only)."""
    return _censor.view(master_df, condition, keep_latents=keep_latents)


def evaluate(profile="golden", do_shap=False, hist_clearing=True, **kw):
    """Train the reference two-tier XGBoost stack under each condition and
    score against the retained ground truth (all/won/lost slices, DeLong CIs,
    calibration, CRPS/RMSE, two-bidder economics). Returns the results dict
    (also what `scripts/` runners persist as JSON)."""
    return _pipeline.run_profile(profile, do_shap=do_shap,
                                 hist_clearing=hist_clearing, **kw)
