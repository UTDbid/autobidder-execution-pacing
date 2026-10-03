"""Pipeline unit tests (the v6 pipeline had ZERO). Pure-logic, fast: split
boundaries, leakage guard, the shared population + AFT-interval helpers, and
win-curve monotonicity - the bits most likely to break silently in a rewrite.
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from t9sim import pipeline as P
from t9sim import schema


def test_temporal_split_boundaries():
    day = 86400
    t0 = 1_700_000_000
    ts = t0 + np.array([0, 19 * day, 20 * day, 23 * day, 24 * day, 27 * day])
    sp = P.temporal_split(pd.DataFrame({"timestamp": ts})).tolist()
    assert sp == ["train", "train", "val", "val", "test", "test"]


def test_assert_no_leak():
    P._assert_no_leak(schema.TIER1_FEATURES)                   # clean -> ok
    with pytest.raises(AssertionError):
        P._assert_no_leak(schema.TIER1_FEATURES + ["ev_truth"])      # estimand
    with pytest.raises(AssertionError):
        P._assert_no_leak(schema.TIER1_FEATURES + ["is_payer"])      # label
    with pytest.raises(AssertionError):
        P._assert_no_leak(schema.TIER1_FEATURES + ["lu5_ltv_mult"])  # latent
    # latents permitted only in the oracle diagnostic
    P._assert_no_leak(schema.TIER1_FEATURES + ["lu5_ltv_mult"],
                      allow_latents=True)


def _toy():
    return pd.DataFrame({
        "won":      [1, 1, 0, 0, 1, 0, 1, 0],
        "click":    [1, 0, 1, 0, 1, 1, 0, 0],
        "install":  [1, 0, 0, 0, 1, 0, 0, 0],
        "is_payer": [1, 0, 0, 0, 0, 0, 0, 0],
    })


def test_funnel_population():
    df = _toy()
    # C2 / C4 = all rows
    assert len(P.funnel_population(df, "C2", "click")) == 8
    assert len(P.funnel_population(df, "C2", "install")) == int((df["click"] == 1).sum())
    assert len(P.funnel_population(df, "C2", "payer")) == int((df["install"] == 1).sum())
    assert len(P.funnel_population(df, "C2", "spend")) == int(
        ((df["install"] == 1) & (df["is_payer"] == 1)).sum())
    # C1 / C3 = won rows only
    won = df[df["won"] == 1]
    assert len(P.funnel_population(df, "C1", "click")) == len(won)
    assert len(P.funnel_population(df, "C1", "install")) == int((won["click"] == 1).sum())
    with pytest.raises(ValueError):
        P.funnel_population(df, "C1", "bogus")


def test_aft_interval_labels():
    df = pd.DataFrame({"bid_price": [2.0, 3.0, 5.0],
                       "won": [1, 0, 0],
                       "clearing_price": [2.0, np.nan, 4.0]})
    lo, hi = P.aft_interval_labels(df, "C1")          # no clearing visible
    assert hi[0] == 2.0 and lo[0] <= 1e-6 + 1e-12     # won -> (0, bid]
    assert lo[1] == 3.0 and np.isinf(hi[1])           # lost -> [bid, inf)
    assert lo[2] == 5.0 and np.isinf(hi[2])
    lo2, hi2 = P.aft_interval_labels(df, "C3")        # clearing visible
    assert lo2[2] == 4.0 and hi2[2] == 4.0            # sold-lost -> exact H4
    assert lo2[1] == 3.0 and np.isinf(hi2[1])         # unsold-lost -> [bid, inf)


class _StubBooster:
    def predict(self, dmat):
        return np.full(dmat.num_row(), 10.0)          # fixed eCPM prediction


def test_pwin_monotone_in_bid():
    n = 5
    df = pd.DataFrame({
        f: (["a"] * n if f in schema.CAT_FEATURES else np.arange(n, dtype=float))
        for f in schema.TIER2_FEATURES})
    for c in schema.CAT_FEATURES:
        df[c] = df[c].astype("category")
    pw = P.pwin_aft(_StubBooster(), df, P.BID_GRID)   # (n_rows, n_bids)
    assert pw.shape == (n, len(P.BID_GRID))
    assert np.all(np.diff(pw, axis=1) >= -1e-9)       # nondecreasing in bid
    assert (pw >= 0).all() and (pw <= 1).all()
