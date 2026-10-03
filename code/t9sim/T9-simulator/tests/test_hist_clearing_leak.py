"""v8 hist_clearing (H6/H7) LEAK + CENSORING-CONSISTENCY tests.

Three guards, weakest->strongest (Blueprint section 4 / section 6 / section 8):

  (1) ROW-POISON unit test (this file, fast, no XGBoost): an eval row's OWN
      target can never enter its OOF/train->eval encoding; a train row's OWN
      target can never enter its LOO encoding. WITH a positive control proving
      the harness actually detects a real (in-fold) leak.

  (2) CENSORING-CONSISTENCY gate (this file, fast): the per-condition Tier-2
      feature list NEVER contains a V8_SSP_ONLY column for a non-clearing
      condition (C1/C2) -- a FEATURE-LIST assert, not NaN-masking.

  (3) The generator-OFF end-to-end negative control is THE leak gate and lives
      in scripts/neg_control_generator_off.py (it generates data + runs
      the pipeline, too heavy for the unit suite). See that file's docstring.

WHY (1) ALONE IS INSUFFICIENT (Blueprint section 8, constraint 4): the
row-poison test only proves the *mechanics* (self-exclusion) are wired. It says
NOTHING about whether the OOF cell is, for a high-cardinality key, a near-copy
of a single neighbour. With ~thousands of {app_id x slot_format x ad_exchange x
daypart} cells, a sparse cell can hold a row + ONE same-cell mate; LOO then
returns (mate_clearing + k*parent)/(1+k) -- which, with k small, is ~the mate's
clearing, i.e. essentially that mate's TARGET. No single row sees its OWN
target (so test (1) passes), yet the feature is a laundered copy of a
neighbour's label and hist_clearing_ssp manufactures a FAKE C3-C1 lift. Only an
end-to-end run on a generator with NO real SSP structure (test (3)) can show
that the *aggregate* C3-C1 auc_win is ~0; the unit test cannot. EB shrinkage to
the parent (HCP_EB_K) is the mitigation; test (3) is what proves it worked.
"""
import numpy as np
import pandas as pd
import pytest

from t9sim import censor, schema

# Skip the whole module cleanly until Step 3b wires these into pipeline.py.
P = pytest.importorskip("t9sim.pipeline")
pytestmark = pytest.mark.skipif(
    not hasattr(P, "_hcp_loo"),
    reason="Step 3b (hist_clearing OOF/LOO) not yet built in pipeline.py")

TOL = 1e-9


# ── synthetic frame with the bucket key + clearing/bid/won ────────────
def _frame(n, seed):
    rng = np.random.default_rng(seed)
    # deliberately FEW cells so sparse-cell collapse is exercised
    apps = [f"a{i}" for i in range(4)]
    df = pd.DataFrame({
        "app_id":      rng.choice(apps, n),
        "slot_format": rng.choice([1, 2, 3], n),
        "ad_exchange": rng.choice(["X", "Y"], n),
        "hour_of_day": rng.integers(0, 24, n),
        "won":         rng.integers(0, 2, n).astype("int8"),
        "bid_price":   rng.uniform(0.5, 5.0, n),
    })
    # H4: on won rows clearing == bid (first-price); some lost rows are "sold"
    h4 = np.where(df["won"] == 1, df["bid_price"].to_numpy(), np.nan)
    sold_lost = (df["won"] == 0) & (rng.random(n) < 0.5)
    h4[sold_lost.to_numpy()] = rng.uniform(0.3, 4.0, int(sold_lost.sum()))
    df["clearing_price"] = h4
    for c in ("app_id", "ad_exchange"):
        df[c] = df[c].astype("category")
    return df


# ── (1a) EVAL: train->eval encoding cannot see an eval row's own target
def test_eval_row_target_excluded_from_oof():
    train, ev = _frame(800, 1), _frame(200, 2)
    src_tr = train["clearing_price"].notna()
    cell, par, gm = P._hcp_cell_stats(train, src_tr)
    base = P._hcp_apply(ev.copy(), cell, par, gm)

    # poison an eval row's OWN clearing (a row that HAS a clearing observed)
    j = int(np.flatnonzero(ev["clearing_price"].notna().to_numpy())[0])
    pois = ev.copy()
    pois.loc[pois.index[j], "clearing_price"] = 1e6
    after = P._hcp_apply(pois, cell, par, gm)

    # train->eval stats are fit on TRAIN ONLY -> eval target NEVER enters
    assert abs(after[j] - base[j]) < TOL, (
        "LEAK: eval row's OWN clearing changed its train->eval encoding")
    np.testing.assert_allclose(after, base, atol=TOL)   # whole vector stable


# ── (1b) TRAIN: LOO encoding cannot see a train row's own target
def test_train_row_target_excluded_from_loo():
    train = _frame(800, 3)
    src = train["clearing_price"].notna()
    base = P._hcp_loo(train, src)

    i = int(np.flatnonzero(src.to_numpy())[0])     # a row that contributes
    pois = train.copy()
    pois.loc[pois.index[i], "clearing_price"] = 1e6
    after = P._hcp_loo(pois, pois["clearing_price"].notna())

    # row i's OWN LOO value excludes itself -> unchanged by poisoning itself
    assert abs(after[i] - base[i]) < TOL, (
        "LEAK: train row's OWN clearing entered its LOO encoding")
    # same-cell MATES legitimately change (they see the poisoned mate); that is
    # NOT a leak -- only self-inclusion is. So we only pin row i.


# ── (1c) POSITIVE CONTROL: the naive in-fold one-liner DOES leak ──────
# Proves the assertions above are not vacuous: a real leak is detected.
def test_positive_control_infold_transform_leaks():
    train, ev = _frame(800, 4), _frame(200, 5)
    KEYS = P.HCP_KEYS if hasattr(P, "HCP_KEYS") else \
        ["app_id", "slot_format", "ad_exchange", "_daypart"]
    for d in (train, ev):
        d["_daypart"] = P._daypart(d["hour_of_day"].to_numpy())
    pool = pd.concat([train, ev], ignore_index=True)
    naive = pool.groupby(KEYS, observed=True)["clearing_price"].transform("mean")
    j_global = len(train) + int(
        np.flatnonzero(ev["clearing_price"].notna().to_numpy())[0])
    base_j = float(naive.iloc[j_global])
    pool2 = pool.copy()
    pool2.loc[j_global, "clearing_price"] = 1e6
    naive2 = pool2.groupby(KEYS, observed=True)["clearing_price"] \
        .transform("mean")
    # the FAKE feature DOES move when its own target is poisoned -> a real leak
    assert abs(float(naive2.iloc[j_global]) - base_j) > 1.0, (
        "positive control failed -> the row-poison harness cannot detect a "
        "leak and tests (1a)/(1b) are vacuous")


# ── (2) CENSORING-CONSISTENCY: feature-list gate, not NaN-masking ─────
def test_t2_feats_no_ssp_only_in_dsp_conditions():
    """A V8_SSP_ONLY column (bid_density, hist_clearing_ssp, min_bid_to_win)
    must NEVER appear in the Tier-2 feature list of a non-clearing condition
    (C1/C2). HARDCODED expectation (anti-tautology): C1/C2 forbidden, C3/C4
    may carry it. (min_bid_to_win added 10 Jul 2026 — v11 B1 arm, H10.)"""
    SSP_ONLY = {"bid_density", "hist_clearing_ssp",
                "min_bid_to_win"}                          # do NOT import schema
    assert set(schema.V8_SSP_ONLY) == SSP_ONLY, \
        "schema.V8_SSP_ONLY drifted from the hardcoded contract"
    for cond in ("C1", "C2", "C3", "C4"):
        feats = set(P.t2_feats(cond, hist_clearing=True))
        clearing = censor.CONDITIONS[cond]["clearing"]
        leaked = feats & SSP_ONLY
        if not clearing:                                  # C1/C2
            assert not leaked, f"{cond} Tier-2 list leaks SSP-only {leaked}"
        # H7 dsp baseline is ALL conditions
        assert "hist_clearing_dsp" in feats, f"{cond} missing H7 dsp"
    # hist_clearing OFF -> exactly the v7 contract
    for cond in ("C1", "C2", "C3", "C4"):
        assert P.t2_feats(cond, hist_clearing=False) == list(
            schema.TIER2_FEATURES), f"{cond} OFF != v7 TIER2_FEATURES"


def test_evaluate_condition_asserts_feature_gate():
    """The SSP guarantee is asserted, not merely NaN-masked: evaluate_condition
    (and train_tier2_aft) must REFUSE a feats list that puts an SSP-only col in
    a C1/C2 run. Skipped if the guard isn't wired yet."""
    if not hasattr(P, "_assert_ssp_gate"):
        pytest.skip("Step 3b SSP feature-list assert (_assert_ssp_gate) "
                    "not yet wired")
    P._assert_ssp_gate(["hist_clearing_dsp"], "C1")              # ok
    with pytest.raises(AssertionError):
        P._assert_ssp_gate(["hist_clearing_dsp", "bid_density"], "C1")
    with pytest.raises(AssertionError):
        P._assert_ssp_gate(["hist_clearing_ssp"], "C2")
    P._assert_ssp_gate(["hist_clearing_dsp", "bid_density",
                        "hist_clearing_ssp"], "C3")             # ok
