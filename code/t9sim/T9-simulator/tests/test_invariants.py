"""v6 invariant + direction tests on a small fixed-seed in-memory run.

NOTE the anti-tautology rule (v6 verification finding): expectations about the
censoring map, attribution columns and forbidden latents are HARDCODED LITERALS
here, never read from censor.py / schema.py - so a transposed condition map or a
shortened hide-list fails loudly instead of adapting.
"""
import re

import numpy as np
import pandas as pd

from t9sim import censor, user_profiles, validate
from t9sim.auctions import AuctionEngine
from t9sim.catalogue import build_apps, build_campaigns
from t9sim.config_loader import Config

# ── HARDCODED expectations (do not import from censor.py / schema.py) ──
ATTRIBUTION = ["click", "click_timestamp", "install", "install_timestamp",
               "is_payer", "ltv_value", "ltv_7d", "ltv_30d"]
# condition -> (attribution NaN on lost rows?, clearing visible?)
EXPECT = {
    "C1": (True, False),     # DSP only: won-row postbacks, no prices
    "C2": (False, False),    # +MMP: cross-network attribution, no prices
    "C3": (True, True),      # +SSP: won-row postbacks, prices
    "C4": (False, True),     # all layers
}
FORBIDDEN_RE = re.compile(r"^(lu|la|lc)\d")
FORBIDDEN_COLS = {"p_click", "p_install", "p_payer", "e_ltv", "ev_truth",
                  "user_id"}


def _views(df):
    return {c: censor.view(df, c) for c in EXPECT}


# ── master invariants ──────────────────────────────────────────────
def test_funnel_on_all_rows(master):
    _, df = master
    for col in ATTRIBUTION:
        assert df[col].notna().all(), f"{col} has NaN in the master"


def test_funnel_chain_monotone(master):
    _, df = master
    assert (df.loc[df["install"] == 1, "click"] == 1).all()
    assert (df.loc[df["is_payer"] == 1, "install"] == 1).all()
    assert (df.loc[df["is_payer"] == 0, "ltv_value"] == 0).all()
    assert (df.loc[df["is_payer"] == 1, "ltv_value"] > 0).all()


def test_structural_nonpayers(master):
    """fixed:0 LTV archetypes must be structural non-payers (the payer draw
    can never fire), preserving is_payer=1 <=> ltv_value>0."""
    cfg, df = master
    users = user_profiles.generate_users(cfg)
    inactive = users[users["lu1_archetype"] == "inactive"]
    assert len(inactive) > 0
    assert (inactive["lu4_payer_prob"] == 0).all()
    assert (inactive["lu5_ltv_mult"] == 0).all()
    assert (df.loc[df["lu1_archetype"] == "inactive", "is_payer"] == 0).all()


def test_ltv_subsets(master):
    _, df = master
    np.testing.assert_allclose(df["ltv_7d"], df["ltv_value"] * 0.40)
    np.testing.assert_allclose(df["ltv_30d"], df["ltv_value"] * 0.70)


def test_h4_piecewise(master):
    _, df = master
    h2, lu7 = df["bid_price"], df["lu7_competing_bid"]
    h1, h4 = df["floor_price"], df["clearing_price"]
    won = df["won"] == 1
    unsold = np.maximum(h2, lu7) < h1
    sold_lost = ~won & ~unsold

    assert (h4[won] == h2[won]).all()
    assert (h4[sold_lost] == lu7[sold_lost]).all()
    assert (lu7[sold_lost] >= h1[sold_lost]).all()
    assert (h4[~unsold] >= h1[~unsold]).all()      # clearing >= floor on sold
    assert h4[unsold].isna().all()
    assert h4[~unsold].notna().all()
    assert ((h2 >= np.maximum(lu7, h1)) == won).all()


def test_timestamps(master):
    # equality E2 == D1 is expected ~3% of clicks (floor of Exp(30s) delay
    # at 1s resolution) - >= is deliberate, not an oversight
    _, df = master
    clicked = df["click"] == 1
    assert (df.loc[clicked, "click_timestamp"]
            >= df.loc[clicked, "timestamp"]).all()
    assert (df.loc[~clicked, "click_timestamp"] == -1).all()
    installed = df["install"] == 1
    assert (df.loc[installed, "install_timestamp"]
            >= df.loc[installed, "click_timestamp"]).all()
    assert (df.loc[~installed, "install_timestamp"] == -1).all()


def test_win_rate_pinned(master):
    _, df = master
    assert 0.25 <= df["won"].mean() <= 0.35


def test_determinism(master):
    """Same profile/seed -> byte-identical pools and chunks."""
    cfg1, cfg2 = Config("golden"), Config("golden")
    u1, u2 = (user_profiles.generate_users(c) for c in (cfg1, cfg2))
    pd.testing.assert_frame_equal(u1, u2)
    e1 = AuctionEngine(cfg1, u1, build_apps(cfg1), build_campaigns(cfg1))
    e2 = AuctionEngine(cfg2, u2, build_apps(cfg2), build_campaigns(cfg2))
    assert e1.calibrate() == e2.calibrate()
    pd.testing.assert_frame_equal(e1.chunk(5000, cfg1.rng("det")),
                                  e2.chunk(5000, cfg2.rng("det")))


# ── censoring invariants (against HARDCODED expectations) ──────────
def test_censor_constants_match_expectations():
    assert censor.ATTRIBUTION_COLS == ATTRIBUTION
    assert set(censor.CONDITIONS) == set(EXPECT)
    for cond, (lost_nan, clearing) in EXPECT.items():
        spec = censor.CONDITIONS[cond]
        assert spec["attribution_all_rows"] == (not lost_nan), cond
        assert spec["clearing"] == clearing, cond


def test_views_censoring(master):
    _, df = master
    lost = (df["won"] == 0).to_numpy()
    unsold = (np.maximum(df["bid_price"], df["lu7_competing_bid"])
              < df["floor_price"]).to_numpy()
    for cond, (lost_nan, clearing) in EXPECT.items():
        v = censor.view(df, cond)
        assert len(v) == len(df), cond
        for col in ATTRIBUTION:
            if lost_nan:
                assert v.loc[lost, col].isna().all(), f"{cond}.{col} lost"
                assert v.loc[~lost, col].notna().all(), f"{cond}.{col} won"
            else:
                assert v[col].notna().all(), f"{cond}.{col}"
        if clearing:
            np.testing.assert_allclose(v["clearing_price"],
                                       df["clearing_price"], equal_nan=True)
            assert v.loc[~unsold, "clearing_price"].notna().all(), cond
        else:
            assert v["clearing_price"].isna().all(), cond


def test_views_leak_nothing(master):
    """Forbidden set derived INDEPENDENTLY of censor.py's hide-list."""
    _, df = master
    forbidden = {c for c in df.columns if FORBIDDEN_RE.match(c)} \
        | (FORBIDDEN_COLS & set(df.columns))
    assert forbidden, "fixture lost its latent columns?"
    for cond, v in _views(df).items():
        leaked = forbidden & set(v.columns)
        assert not leaked, f"{cond} leaks {leaked}"


def test_views_value_fidelity(master):
    """Visible cells must EQUAL the master, not merely be non-null."""
    _, df = master
    won = df["won"] == 1
    for cond, v in _views(df).items():
        for col in ATTRIBUTION:
            np.testing.assert_allclose(
                v.loc[won, col], df.loc[won, col].astype("float64"),
                err_msg=f"{cond}.{col} won-row values differ from master")
        for col in ("timestamp", "bid_price", "floor_price", "won"):
            np.testing.assert_allclose(v[col], df[col],
                                       err_msg=f"{cond}.{col}")


def test_view_does_not_mutate_master(master):
    _, df = master
    snapshot = df.copy()
    for cond in EXPECT:
        censor.view(df, cond)
    pd.testing.assert_frame_equal(df, snapshot)


# ── directions ─────────────────────────────────────────────────────
def test_eltv_ordering_by_archetype(master):
    """Deterministic stand-in for the realized-LTV ordering (too few payers
    below ~1M rows): mean ground-truth E[ltv] monotone in archetype value."""
    _, df = master
    m = df.groupby("lu1_archetype")["e_ltv"].mean()
    assert m["whale"] > m["engaged_spender"] > m["casual"] \
        > m["time_filler"] >= m["inactive"]


def test_directions(master):
    _, df = master
    failures = {k: d for k, (ok, d) in validate.directions(df).items()
                if not ok}
    assert not failures, failures
