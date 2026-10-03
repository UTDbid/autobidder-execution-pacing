"""T9 training + evaluation pipeline.

For one generated master (a profile's auctions.parquet):
  1. temporal split (by auction day, BEFORE censoring): 1-20 train / 21-24 val / 25-28 test
  2. Tier-1: 4 XGBoost heads (hurdle) per condition - P(click), P(install|click),
     P(payer|install), E[spend|payer]; EV = product. Trained on the condition's
     visible population (won-rows-only in C1/C3, all rows in C2/C4).
  3. Tier-2: XGBoost AFT censored regression on the competing bid per condition
     (win/loss bounds everywhere; exact LU7 = H4 on lost-sold rows in C3/C4) -> P(win|b).
  4. profit-max bidder (EV_hat x P(win|b)) per condition vs ORACLE (true EV+LU7) and
     H2 incumbent -> ROAS + profit on the test set; per-tier metrics; SHAP on Tier-1.

Leakage guards: latents/estimands/labels never enter features (asserted). Prices
handled per-impression (eCPM / 1000) so EV (per-impression $) and bid are commensurate.

Clean-room rewrite changes vs v6:
  - run config threaded via an immutable PipelineConfig dataclass (no module-global
    mutation of _T1_FEATS/_ORACLE).
  - ONE funnel_population() and ONE aft_interval_labels() shared by the in-memory
    and streaming paths (v6 duplicated both).
  - feature/label/leakage sets come from schema.py (single source of truth).

Usage:  python -m t9sim.pipeline --profile golden|test|scale10m [--seed N]
        python -m t9sim.pipeline --profile scale50m --chunk-rows 5000000
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import censor, schema
from .paths import OUTPUT_DIR

warnings.filterwarnings("ignore")
import xgboost as xgb                                              # noqa: E402
from sklearn.metrics import (mean_squared_error, roc_auc_score,    # noqa: E402
                             log_loss)


# ── run configuration (immutable; replaces v6 module globals) ─────────
@dataclass(frozen=True)
class PipelineConfig:
    """Per-run feature configuration. oracle=True adds the latent value-drivers
    to Tier-1 (the upper-bound diagnostic a perfect BN layer could reach)."""
    oracle: bool = False

    @property
    def t1_feats(self):
        extra = (schema.ORACLE_NUM + schema.ORACLE_CAT) if self.oracle else []
        return list(schema.TIER1_FEATURES) + extra

    @property
    def t1_cat(self):
        return list(schema.CAT_FEATURES) + (
            schema.ORACLE_CAT if self.oracle else [])


XGB_CLF = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
               tree_method="hist", enable_categorical=True,
               eval_metric="logloss", early_stopping_rounds=20)
XGB_REG = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
               tree_method="hist", enable_categorical=True,
               eval_metric="rmse", early_stopping_rounds=20)
AFT_SCALE = 1.0          # aft_loss_distribution_scale (sigma), normal dist
BID_GRID = np.geomspace(1e-4, 0.12, 48)          # candidate bids, $ per impression

# ── v8 hist_clearing (H6 ssp / H7 dsp) OOF/LOO encoder ────────────────
# Post-hoc placement-clearing features the SSP/DSP would compute from logs.
# H7 dsp: WON-only mean of bid_price (uncensored in ALL conds; winner's-curse biased).
# H6 ssp: ALL-clears mean of clearing_price (won H2 + sold-lost H4; C3/C4 only; unbiased).
# C3-C1 = value of de-biasing. Leak-safe: LOO on train, train->eval on val/test,
# EB-shrink cell -> (app_id-free, dense) parent -> global. THE leak gate is the
# generator-OFF negative control (scripts/neg_control_generator_off.py), not row-poison.
HCP_EB_K   = float(os.environ.get("T9_HCP_EB_K", "20"))     # env-tunable (leak-vs-debias diag)
HCP_KEYS   = ["app_id", "slot_format", "ad_exchange", "_daypart"]
HCP_PARENT = ["slot_format", "ad_exchange", "_daypart"]      # backoff drops app_id (dense)


def _daypart(hour):
    """4 dayparts: 0 night[0-5] 1 morning[6-11] 2 afternoon[12-17] 3 evening[18-23].
    Transient groupby key, never a model feature."""
    return pd.cut(np.asarray(hour), bins=[-1, 5, 11, 17, 23],
                  labels=False).astype("int16")


def _ensure_daypart(df):
    """Return df guaranteed to carry the transient _daypart key. No-op in the
    production path (attach_hist_clearing sets it once on the big frames); a cheap
    copy for direct helper callers (the unit tests)."""
    if "_daypart" in df.columns:
        return df
    return df.assign(_daypart=_daypart(df["hour_of_day"].to_numpy()))


def _assert_ssp_gate(feats, cond):
    """CENSORING gate: an SSP-only column (bid_density, hist_clearing_ssp) may appear
    in a Tier-2 feature list ONLY when the condition un-censors clearing (C3/C4).
    The eval frame is uncensored, so the FEATURE LIST is the guarantee, not NaN-masking."""
    if not censor.CONDITIONS[cond]["clearing"]:
        leaked = [c for c in schema.V8_SSP_ONLY if c in feats]
        assert not leaked, f"CENSORING: SSP-only {leaked} in {cond} Tier-2 feats"


def _hcp_cell_stats(train, src_mask, target_col="clearing_price"):
    """Cell/parent sum+count of the clearing target on the TRAIN split only
    (leak-safe for val/test). src_mask (caller-owned) selects contributing rows:
    H7 -> won==1, target bid_price; H6 -> clearing_price.notna(), target clearing_price.
    Returns (cell_df, par_df, gmean)."""
    base = _ensure_daypart(train)[src_mask]
    gmean = float(base[target_col].mean())
    cell = base.groupby(HCP_KEYS,   observed=True)[target_col].agg(["sum", "count"])
    par  = base.groupby(HCP_PARENT, observed=True)[target_col].agg(["sum", "count"])
    return cell, par, gmean


def _hcp_apply(df_eval, cell, par, gmean, k=HCP_EB_K):
    """EB-shrunk group mean for val/test (train->eval): cell -> parent -> global.
    Sparse cells never NaN."""
    df_eval = _ensure_daypart(df_eval)
    cd, pdd = cell.to_dict("index"), par.to_dict("index")
    ci = list(df_eval[HCP_KEYS].itertuples(index=False, name=None))
    pi = list(df_eval[HCP_PARENT].itertuples(index=False, name=None))
    out = np.empty(len(df_eval), dtype="float64")
    for i, (c, p) in enumerate(zip(ci, pi)):
        pr = pdd.get(p)
        parent = (pr["sum"] / pr["count"]) if pr else gmean
        r = cd.get(c)
        out[i] = (r["sum"] + k * parent) / (r["count"] + k) if r else parent
    return out


def _hcp_loo(train, src_mask, k=HCP_EB_K, target_col="clearing_price"):
    """Leave-one-out EB mean for TRAIN rows' OWN encodings: subtract each row's own
    (target,1) from its cell/parent sum+count before shrinking, so a train row never
    sees its own target. (groupby.transform('mean') in-fold would LEAK.) Rows outside
    src_mask contribute (0,0) -> they get the full train-group EB mean (not NaN)."""
    train = _ensure_daypart(train)
    valid = src_mask.to_numpy() if hasattr(src_mask, "to_numpy") else np.asarray(src_mask)
    c = np.where(valid, train[target_col].to_numpy(), 0.0).astype("float64")
    n = valid.astype("float64")
    s = train.assign(_c=c, _n=n)
    cs = s.groupby(HCP_KEYS,   observed=True)[["_c", "_n"]].transform("sum")
    ps = s.groupby(HCP_PARENT, observed=True)[["_c", "_n"]].transform("sum")
    gmean = float(train.loc[valid, target_col].mean())
    loo_c, loo_n = cs["_c"].to_numpy() - c, cs["_n"].to_numpy() - n
    par_c, par_n = ps["_c"].to_numpy() - c, ps["_n"].to_numpy() - n
    parent = np.where(par_n > 0, par_c / np.maximum(par_n, 1e-9), gmean)
    return (loo_c + k * parent) / (loo_n + k)


def attach_hist_clearing(train, val, test, want_ssp):
    """Attach hist_clearing_dsp (H7, all conds) and, iff want_ssp, hist_clearing_ssp
    (H6, C3/C4). Mutates the three frames in place. LOO within train; train->eval for
    val/test. _daypart is transient."""
    for df in (train, val, test):
        df["_daypart"] = _daypart(df["hour_of_day"].to_numpy())
    # H7 dsp: won-only bid_price (NOT clearing_price: H4 is NaN in C1/C2)
    src = (train["won"] == 1)
    cell, par, gm = _hcp_cell_stats(train, src, target_col="bid_price")
    train["hist_clearing_dsp"] = _hcp_loo(train, src, target_col="bid_price")
    val["hist_clearing_dsp"]   = _hcp_apply(val,  cell, par, gm)
    test["hist_clearing_dsp"]  = _hcp_apply(test, cell, par, gm)
    if want_ssp:
        if "min_bid_to_win" in train.columns and train["min_bid_to_win"].notna().any():
            # v11 B1: with H10 the SSP observes the TRUE top-rival price on wins too,
            # so the ssp cell mean becomes an all-outcome min-to-win mean
            # (won -> H10, lost-sold -> H4=LU7) instead of (won -> own bid, lost -> LU7).
            # Same column name; activates only when the generator flag emitted H10.
            train["_t2clear"] = np.where(train["won"].to_numpy() == 1,
                                         train["min_bid_to_win"].to_numpy(),
                                         train["clearing_price"].to_numpy())
            src = train["_t2clear"].notna()
            cell, par, gm = _hcp_cell_stats(train, src, target_col="_t2clear")
            train["hist_clearing_ssp"] = _hcp_loo(train, src, target_col="_t2clear")
        else:
            src = train["clearing_price"].notna()
            cell, par, gm = _hcp_cell_stats(train, src, target_col="clearing_price")
            train["hist_clearing_ssp"] = _hcp_loo(train, src, target_col="clearing_price")
        val["hist_clearing_ssp"]   = _hcp_apply(val,  cell, par, gm)
        test["hist_clearing_ssp"]  = _hcp_apply(test, cell, par, gm)
    for df in (train, val, test):
        df.drop(columns=["_daypart", "_t2clear"], inplace=True, errors="ignore")


def t2_feats(cond, hist_clearing, view_cols=None):
    """Per-condition Tier-2 feature list. OFF -> exactly list(schema.TIER2_FEATURES)
    (v7-identical). ON -> + hist_clearing_dsp (all conds); + hist_clearing_ssp +
    bid_density iff CONDITIONS[cond]['clearing'] (C3/C4). view_cols gates bid_density
    (a generator column present only if competition was ON)."""
    f = list(schema.TIER2_FEATURES)
    if hist_clearing:
        f.append("hist_clearing_dsp")
        if censor.CONDITIONS[cond]["clearing"]:
            f.append("hist_clearing_ssp")
            if view_cols is not None and "bid_density" in view_cols:
                f.append("bid_density")
    _assert_ssp_gate(f, cond)
    _assert_no_leak(f)
    return f


# ── shared helpers (used by BOTH in-memory and streaming paths) ───────
def temporal_split(df):
    """Assign split by auction day-of-window (BEFORE censoring).
    days 1-20 train / 21-24 val / 25-28 test, derived from timestamp."""
    t0 = df["timestamp"].min()
    day = ((df["timestamp"] - t0) // 86400).astype(int)      # 0-based day
    split = np.where(day < 20, "train", np.where(day < 24, "val", "test"))
    return pd.Series(split, index=df.index)


def funnel_population(df, cond, stage):
    """Visible training population for a funnel stage under a condition.
    C1/C3 see outcomes on won rows only; C2/C4 on all rows. Used identically
    on the in-memory censored view and on each streamed master batch."""
    all_rows = censor.CONDITIONS[cond]["attribution_all_rows"]
    base = df if all_rows else df[df["won"] == 1]
    if stage == "click":
        return base
    if stage == "install":
        return base[base["click"] == 1]
    if stage == "payer":
        return base[base["install"] == 1]
    if stage == "spend":
        return base[(base["install"] == 1) & (base["is_payer"] == 1)]
    raise ValueError(stage)


def aft_interval_labels(c, cond):
    """Per-row AFT interval (eCPM, per-mille) for the competing-bid regression:
       won            -> (0, bid]        upper-bounded
       lost           -> [bid, +inf)     lower-bounded
       C3/C4 lost-sold -> [H4, H4]        exact (SSP un-censors).
       C3/C4 won, H10 present (v11 B1) -> H10>floor: [H10, H10] exact LU7;
                                          H10==floor: (0, floor] (LU7 below floor).
    Returns (lower, upper) with lower clamped to 1e-6."""
    bid = c["bid_price"].to_numpy()
    won = c["won"].to_numpy() == 1
    lo = np.where(won, 0.0, bid).astype("float64")
    hi = np.where(won, bid, np.inf).astype("float64")
    if censor.CONDITIONS[cond]["clearing"]:
        h4 = c["clearing_price"].to_numpy()
        sold_lost = (~won) & ~np.isnan(h4)
        lo[sold_lost] = h4[sold_lost]
        hi[sold_lost] = h4[sold_lost]
        if "min_bid_to_win" in c.columns:          # v11 B1: win-side reveal
            mbw = c["min_bid_to_win"].to_numpy()
            flr = c["floor_price"].to_numpy()
            seen = won & ~np.isnan(mbw)
            pt = seen & (mbw > flr * (1.0 + 1e-9))  # H10 > floor -> LU7 point-observed
            lo[pt] = mbw[pt]
            hi[pt] = mbw[pt]
            ub = seen & ~pt                          # H10 == floor -> LU7 <= floor
            hi[ub] = np.minimum(hi[ub], mbw[ub])
    return np.maximum(lo, 1e-6), hi


def _assert_no_leak(cols, allow_latents=False):
    # labels and estimands (the answers) are ALWAYS forbidden; latents are
    # forbidden except in the oracle-features diagnostic.
    bad = [c for c in cols if c in schema.LABELS or c in schema.ESTIMANDS
           or (c.startswith(schema.LATENT_PREFIXES) and not allow_latents)]
    assert not bad, f"LEAKAGE: forbidden columns in feature set: {bad}"


def _as_cat(df, cols):
    """Select feature cols. Categorical dtype is set ONCE globally (run_profile)
    so train/val/test share one code-space and test-only categories (e.g. an
    app first seen on a later day) route to the tree's default branch instead
    of erroring."""
    return df[cols]


# ── chunked-loading helpers (scale50m / scale100m) ────────────────────
class _BoosterWrapper:
    """Raw XGBoost booster wrapped to look like XGBClassifier/XGBRegressor so
    predict_ev / evaluate_condition / shap_tier1 are path-agnostic."""

    def __init__(self, booster, is_clf):
        self._b = booster
        self._is_clf = is_clf

    def get_booster(self):
        return self._b

    def predict_proba(self, X):
        p = self._b.predict(xgb.DMatrix(X, enable_categorical=True))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self._b.predict(xgb.DMatrix(X, enable_categorical=True))


class _PopIter(xgb.DataIter):
    """Streams train rows for one Tier-1 funnel population from parquet,
    reusing funnel_population() so it cannot drift from the in-memory path."""

    def __init__(self, path, needed, cat_levels, t0,
                 features, label_col, cond, stage, chunk_rows):
        self._path = path
        self._needed = needed
        self._cat_levels = cat_levels
        self._t0 = t0
        self._features = features
        self._label_col = label_col
        self._cond = cond
        self._stage = stage
        self._chunk_rows = chunk_rows
        self._gen = None
        super().__init__()

    def _make_gen(self):
        for batch in pq.ParquetFile(self._path).iter_batches(
                batch_size=self._chunk_rows, columns=self._needed):
            c = batch.to_pandas()
            day = ((c["timestamp"] - self._t0) // 86400).astype(int)
            c = c[day < 20]
            c = funnel_population(c, self._cond, self._stage)
            if len(c) == 0:
                continue
            for col, cats in self._cat_levels.items():
                if col in c.columns:
                    c[col] = pd.Categorical(c[col], categories=cats)
            yield c

    def next(self, input_data):
        if self._gen is None:
            self._gen = self._make_gen()
        try:
            c = next(self._gen)
            input_data(data=c[self._features], label=c[self._label_col])
            return 1
        except StopIteration:
            return 0

    def reset(self):
        self._gen = self._make_gen()


class _AftIter(xgb.DataIter):
    """Streams train rows for Tier-2 AFT, reusing aft_interval_labels()."""

    def __init__(self, path, needed, cat_levels, t0, features, cond, chunk_rows):
        self._path = path
        self._needed = needed
        self._cat_levels = cat_levels
        self._t0 = t0
        self._features = features
        self._cond = cond
        self._chunk_rows = chunk_rows
        self._gen = None
        super().__init__()

    def _make_gen(self):
        for batch in pq.ParquetFile(self._path).iter_batches(
                batch_size=self._chunk_rows, columns=self._needed):
            c = batch.to_pandas()
            day = ((c["timestamp"] - self._t0) // 86400).astype(int)
            c = c[day < 20]
            if len(c) == 0:
                continue
            for col, cats in self._cat_levels.items():
                if col in c.columns:
                    c[col] = pd.Categorical(c[col], categories=cats)
            yield c

    def next(self, input_data):
        if self._gen is None:
            self._gen = self._make_gen()
        try:
            c = next(self._gen)
            lo, hi = aft_interval_labels(c, self._cond)
            input_data(data=c[self._features],
                       label_lower_bound=lo, label_upper_bound=hi)
            return 1
        except StopIteration:
            return 0

    def reset(self):
        self._gen = self._make_gen()


# ── Tier 1 ──────────────────────────────────────────────────────────
def train_tier1(pc, train, val, cond, chunk_cfg=None):
    """Train 4 Tier-1 heads.  chunk_cfg enables streaming QuantileDMatrix."""
    _assert_no_leak(pc.t1_feats, allow_latents=pc.oracle)
    feats = pc.t1_feats
    models = {}
    for stage, target, is_clf in (
            ("click", "click", True), ("install", "install", True),
            ("payer", "is_payer", True), ("spend", "ltv_value", False)):
        va = funnel_population(val, cond, stage)
        Xva, yva = _as_cat(va, feats), va[target]
        params = XGB_CLF if is_clf else XGB_REG
        obj = "binary:logistic" if is_clf else "reg:squarederror"

        if chunk_cfg:
            it = _PopIter(chunk_cfg["path"], chunk_cfg["needed"],
                          chunk_cfg["cat_levels"], chunk_cfg["t0"],
                          feats, target, cond, stage,
                          chunk_cfg["chunk_rows"])
            dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)
            bp = {"max_depth": params["max_depth"],
                  "learning_rate": params["learning_rate"],
                  "tree_method": "hist",
                  "eval_metric": params["eval_metric"],
                  "objective": obj, "verbosity": 0}
            evals, cbs = [], []
            if len(va) > 0:
                dval = xgb.DMatrix(Xva, label=yva, enable_categorical=True)
                evals = [(dval, "val")]
                cbs = [xgb.callback.EarlyStopping(
                    rounds=params["early_stopping_rounds"])]
            booster = xgb.train(bp, dtrain,
                                num_boost_round=params["n_estimators"],
                                evals=evals, callbacks=cbs)
            m = _BoosterWrapper(booster, is_clf)
        else:
            tr = funnel_population(train, cond, stage)
            Xtr = _as_cat(tr, feats)
            cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
            if len(va) > 0:
                m = cls(**params)
                m.fit(Xtr, tr[target], eval_set=[(Xva, yva)], verbose=False)
            else:
                p2 = {k: v for k, v in params.items()
                      if k != "early_stopping_rounds"}
                m = cls(**p2)
                m.fit(Xtr, tr[target], verbose=False)
        models[stage] = m
    return models


def predict_ev(pc, models, df):
    X = _as_cat(df, pc.t1_feats)
    p_click = models["click"].predict_proba(X)[:, 1]
    p_inst = models["install"].predict_proba(X)[:, 1]
    p_payer = models["payer"].predict_proba(X)[:, 1]
    e_spend = np.clip(models["spend"].predict(X), 0, None)
    return (p_click * p_inst * p_payer * e_spend,
            p_click, p_inst, p_payer, e_spend)


# ── Tier 2 (AFT censored regression on competing bid) ───────────────
def train_tier2_aft(train, cond, chunk_cfg=None, feats=None):
    """AFT on LU7 with per-row interval labels (see aft_interval_labels).
    feats defaults to schema.TIER2_FEATURES (OFF + streaming = v7-identical)."""
    feats = list(schema.TIER2_FEATURES) if feats is None else feats
    _assert_no_leak(feats)
    _assert_ssp_gate(feats, cond)
    params = dict(objective="survival:aft", eval_metric="aft-nloglik",
                  aft_loss_distribution="normal",
                  aft_loss_distribution_scale=AFT_SCALE,
                  tree_method="hist", max_depth=6, learning_rate=0.1,
                  verbosity=0)
    if chunk_cfg:
        if feats != list(schema.TIER2_FEATURES):
            raise NotImplementedError(
                "hist_clearing Tier-2 features unsupported on the streaming "
                "(_AftIter) path (needs 2-pass cell stats); use the in-memory "
                "path (golden/1M/10M) for the v8 headline.")
        it = _AftIter(chunk_cfg["path"], chunk_cfg["needed"],
                      chunk_cfg["cat_levels"], chunk_cfg["t0"],
                      feats, cond, chunk_cfg["chunk_rows"])
        d = xgb.QuantileDMatrix(it, enable_categorical=True)
    else:
        lo, hi = aft_interval_labels(train, cond)
        X = _as_cat(train, feats)
        d = xgb.DMatrix(X, enable_categorical=True)
        d.set_float_info("label_lower_bound", lo)
        d.set_float_info("label_upper_bound", hi)
    booster = xgb.train(params, d, num_boost_round=250)
    return booster


def pwin_aft(booster, df, bids_per_imp, feats=None):
    """P(win|b) = P(LU7 <= b) = Phi((log b - log pred_LU7)/sigma).
    pred is eCPM (per-mille); bids_per_imp are per-impression -> x1000.
    feats MUST match training (defaults to schema.TIER2_FEATURES)."""
    from scipy.stats import norm
    feats = list(schema.TIER2_FEATURES) if feats is None else feats
    X = _as_cat(df, feats)
    pred = booster.predict(xgb.DMatrix(X, enable_categorical=True))  # eCPM
    pred = np.maximum(pred, 1e-6)
    b_ecpm = np.asarray(bids_per_imp) * 1000.0
    z = (np.log(b_ecpm)[None, :] - np.log(pred)[:, None]) / AFT_SCALE
    return norm.cdf(z)


def pwin_clf(clf_bst, clf_feats, df, bids_per_imp):
    """P(win|b) from the well-specified won-classifier: re-predict with the
    bid_price column set to each grid bid (eCPM). The economics analogue of the
    AUC shrink test - bidder metrics that do NOT ride the misspecified AFT."""
    base = _as_cat(df, clf_feats).copy()
    out = np.empty((len(df), len(bids_per_imp)))
    for j, b in enumerate(bids_per_imp):
        base["bid_price"] = float(b) * 1000.0            # $ per imp -> eCPM
        out[:, j] = clf_bst.predict(xgb.DMatrix(base, enable_categorical=True))
    return out


# ── evaluation: bidder / oracle / H2 ────────────────────────────────
def _roas(revenue, cost):
    return float(revenue / cost) if cost > 0 else float("nan")


def _eval_pop(test, stage):
    """COMMON test population for fair cross-condition scoring - all rows with
    only the natural funnel-stage conditioning; NEVER the won-only restriction
    (every condition is judged on the same yardstick = the bidding-relevant
    all-rows distribution, using ground-truth labels available in the master)."""
    if stage == "click":
        return test
    if stage == "install":
        return test[test["click"] == 1]
    if stage == "payer":
        return test[test["install"] == 1]
    return test[(test["install"] == 1) & (test["is_payer"] == 1)]   # spend


def _resolve_bidder(ev_hat, pw, test):
    """Profit-max bid over BID_GRID given per-impression EV ($) and the
    (n, n_bids) win-prob grid; resolve wins against the realised clearing
    price and return the economic metrics. SHARED by evaluate_condition
    (learned models) and oracle_bidder_ev (true values) so both use the
    identical bidding/auction-resolution code."""
    profit = (ev_hat[:, None] - BID_GRID[None, :]) * pw    # expected profit grid
    best = profit.argmax(axis=1)
    best_profit = profit[np.arange(len(test)), best]
    bid_star = np.where(best_profit > 0, BID_GRID[best], 0.0)  # don't bid if <=0
    lu7 = test["lu7_competing_bid"].to_numpy() / 1000.0
    floor = test["floor_price"].to_numpy() / 1000.0
    realised_ltv = test["ltv_value"].to_numpy()
    need = np.maximum(lu7, floor)                          # min first-price bid to win
    won_mask = (bid_star > 0) & (bid_star >= need)
    rev = realised_ltv[won_mask].sum()
    cost = bid_star[won_mask].sum()
    nw = int(won_mask.sum())
    # overpayment = paid own bid minus the minimum needed to clear (first-price shade slack)
    overpay = float((bid_star[won_mask] - need[won_mask]).sum())
    return {"roas": round(_roas(rev, cost), 4),
            "profit": round(float(rev - cost), 2),
            "n_won": nw,
            "revenue": round(float(rev), 2),
            "spend": round(float(cost), 2),
            "overpay": round(overpay, 2),
            "overpay_per_won": round(overpay / max(nw, 1), 5),
            "surplus_per_won": round(float(rev - cost) / max(nw, 1), 5)}


def _ece(pw, won, n_bins=15):
    """Expected / max calibration error, equal-width bins (Klimis #3 calibration)."""
    pw = np.clip(np.asarray(pw, float), 0.0, 1.0)
    won = np.asarray(won, float)
    ids = np.minimum((pw * n_bins).astype(int), n_bins - 1)
    ece, mce, N = 0.0, 0.0, len(pw)
    for b in range(n_bins):
        m = ids == b
        if m.any():
            gap = abs(pw[m].mean() - won[m].mean())
            ece += m.sum() / N * gap
            mce = max(mce, gap)
    return round(float(ece), 4), round(float(mce), 4)


def _delong(y, p_a, p_b):
    """Paired DeLong test for AUC(p_a) - AUC(p_b) on the SAME test set (midrank
    fast version). Returns diff, se, 95% CI, z, p - the inference layer every
    layer-contrast needs (point estimates alone aren't Klimis-grade)."""
    from scipy.stats import rankdata, norm as _norm
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    V10, V01, aucs = np.empty((2, m)), np.empty((2, n)), np.empty(2)
    for i, p in enumerate((np.asarray(p_a, float), np.asarray(p_b, float))):
        pos, neg = p[y], p[~y]
        r_all = rankdata(np.concatenate([pos, neg]))
        V10[i] = (r_all[:m] - rankdata(pos)) / n
        V01[i] = 1.0 - (r_all[m:] - rankdata(neg)) / m
        aucs[i] = V10[i].mean()
    S10, S01 = np.cov(V10), np.cov(V01)
    var = ((S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m
           + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n)
    se = float(np.sqrt(max(var, 1e-24)))
    d = float(aucs[0] - aucs[1])
    z = d / se
    return {"diff": round(d, 4), "se": round(se, 5),
            "ci95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
            "z": round(z, 2), "p": round(float(2 * _norm.sf(abs(z))), 6),
            "auc_a": round(float(aucs[0]), 4), "auc_b": round(float(aucs[1]), 4)}


def _gain_top(booster, k=8):
    """Top-k gain importance - the Tier-2 'why' (which features carry the model)."""
    g = booster.get_score(importance_type="gain")
    top = sorted(g.items(), key=lambda kv: -kv[1])[:k]
    return {f: round(v, 1) for f, v in top}


def train_tier2_clf(train, val, cond, feats=None):
    """Well-specified Tier-2 companion: won-classifier P(win|bid,ctx). Serves as
    (a) the per-run feature-Bayes estimate and (b) the standing shrink test - the
    AFT contrast is label-efficiency iff it vanishes here (v8: +0.0044 -> +0.0002).
    Same leak/censoring gates as the AFT; bid+floor added (P(win) conditions on bid)."""
    feats = (list(schema.TIER2_FEATURES) if feats is None else list(feats)) \
        + ["bid_price", "floor_price"]
    _assert_no_leak(feats)
    _assert_ssp_gate(feats, cond)
    d_tr = xgb.DMatrix(_as_cat(train, feats), label=train["won"],
                       enable_categorical=True)
    d_va = xgb.DMatrix(_as_cat(val, feats), label=val["won"],
                       enable_categorical=True)
    bst = xgb.train({"objective": "binary:logistic", "eval_metric": "auc",
                     "tree_method": "hist", "max_depth": 8, "learning_rate": 0.1,
                     "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0},
                    d_tr, num_boost_round=400, evals=[(d_va, "val")],
                    early_stopping_rounds=30, verbose_eval=False)
    return bst, feats


def evaluate_condition(pc, models, booster, test, cond, feats=None, clf=None):
    from scipy.stats import norm, spearmanr
    feats = list(schema.TIER2_FEATURES) if feats is None else feats
    _assert_ssp_gate(feats, cond)            # te is uncensored -> feature list is the gate
    res = {}
    # ---- per-tier metrics on the COMMON all-rows population (fair) ----
    for stage, target in (("click", "click"), ("install", "install"),
                          ("payer", "is_payer")):
        pop = _eval_pop(test, stage)
        if pop[target].nunique() > 1:
            X = _as_cat(pop, pc.t1_feats)
            p = models[stage].predict_proba(X)[:, 1]
            res[f"auc_{stage}"] = round(roc_auc_score(pop[target], p), 4)
            res[f"logloss_{stage}"] = round(log_loss(pop[target], p), 4)
    pop = _eval_pop(test, "spend")             # E[spend|payer] on payer rows
    if len(pop) > 5:
        X = _as_cat(pop, pc.t1_feats)
        pred = np.clip(models["spend"].predict(X), 0, None)
        res["rmse_spend"] = round(mean_squared_error(pop["ltv_value"], pred)
                                  ** 0.5, 4)

    # ---- DIAGNOSTIC 1: EV-bias (EV_hat vs ev_truth, all rows) ----
    ev_hat, _, _, _, _ = predict_ev(pc, models, test)
    ev_true = test["ev_truth"].to_numpy()
    res["ev_pred_mean"] = round(float(ev_hat.mean()), 6)
    res["ev_true_mean"] = round(float(ev_true.mean()), 6)
    res["ev_ratio"] = round(float(ev_hat.mean() / max(ev_true.mean(), 1e-12)), 4)
    res["ev_rmse"] = round(float(np.sqrt(((ev_hat - ev_true) ** 2).mean())), 6)
    res["ev_spearman"] = round(float(spearmanr(ev_hat, ev_true).correlation), 4)

    # ---- Tier-2 win-curve: P(win|own bid) vs realised won (all rows) ----
    X2 = _as_cat(test, feats)
    predlu7 = np.maximum(booster.predict(
        xgb.DMatrix(X2, enable_categorical=True)), 1e-6)
    pw_own = norm.cdf((np.log(test["bid_price"].to_numpy())
                       - np.log(predlu7)) / AFT_SCALE)
    if test["won"].nunique() > 1:
        res["auc_win"] = round(roc_auc_score(test["won"], pw_own), 4)
        res["logloss_win"] = round(log_loss(test["won"],
                                            np.clip(pw_own, 1e-6, 1 - 1e-6)), 4)
        # ---- DIAGNOSTIC 2: win-curve reliability (calibration) ----
        rel = pd.DataFrame({"pred": pw_own, "won": test["won"].to_numpy()})
        rel["dec"] = pd.qcut(rel["pred"], 10, labels=False, duplicates="drop")
        g = rel.groupby("dec").agg(pred=("pred", "mean"), act=("won", "mean"))
        res["wincurve_calib_err"] = round(float((g["pred"] - g["act"])
                                                 .abs().mean()), 4)
        res["wincurve_reliability"] = {int(k): [round(r.pred, 3),
                                                 round(r.act, 3)]
                                       for k, r in g.iterrows()}
        res["ece_win"], res["mce_win"] = _ece(pw_own, test["won"].to_numpy())
    # ---- price accuracy (Klimis #3 / anchoring metric): AFT pred vs true LU7 ----
    lu7_true = test["lu7_competing_bid"].to_numpy()
    err = predlu7 - lu7_true
    wm = test["won"].to_numpy() == 1
    res["price_rmse_all"] = round(float(np.sqrt(np.mean(err ** 2))), 4)
    res["price_rmse_won"] = round(float(np.sqrt(np.mean(err[wm] ** 2))), 4)
    res["price_rmse_lost"] = round(float(np.sqrt(np.mean(err[~wm] ** 2))), 4)
    # CRPS of the AFT's log-normal price distribution (closed form, Klimis #3)
    zc = (np.log(np.maximum(lu7_true, 1e-6)) - np.log(predlu7)) / AFT_SCALE
    crps = AFT_SCALE * (zc * (2 * norm.cdf(zc) - 1)
                        + 2 * norm.pdf(zc) - 1.0 / np.sqrt(np.pi))
    res["price_crps_all"] = round(float(np.mean(crps)), 4)
    res["price_crps_won"] = round(float(np.mean(crps[wm])), 4)
    res["price_crps_lost"] = round(float(np.mean(crps[~wm])), 4)
    res["t2_gain_top"] = _gain_top(booster)          # Tier-2 'why' (AFT)
    res["_pw"] = {"aft": pw_own}                     # for paired DeLong (popped later)

    # ---- well-specified companion (won-classifier): per-run feature-Bayes
    #      estimate + standing shrink test vs the AFT contrast ----
    if clf is not None:
        clf_bst, clf_feats = clf
        pw_clf = clf_bst.predict(xgb.DMatrix(_as_cat(test, clf_feats),
                                             enable_categorical=True))
        if test["won"].nunique() > 1:
            res["auc_win_clf"] = round(roc_auc_score(test["won"], pw_clf), 4)
            res["logloss_win_clf"] = round(log_loss(
                test["won"], np.clip(pw_clf, 1e-6, 1 - 1e-6)), 4)
            res["ece_win_clf"], res["mce_win_clf"] = _ece(
                pw_clf, test["won"].to_numpy())
        res["t2_gain_top_clf"] = _gain_top(clf_bst)
        res["_pw"]["clf"] = pw_clf

    # ---- profit-max bidder (shared with the oracle via _resolve_bidder) ----
    ev_hat, _, _, _, _ = predict_ev(pc, models, test)     # per-impression $
    pw = pwin_aft(booster, test, BID_GRID, feats=feats)   # (n, n_bids)
    res.update(_resolve_bidder(ev_hat, pw, test))
    # CLF-bidder: same EV, win curve from the well-specified classifier - the
    # economics that do NOT ride the AFT (model-robustness for overpay/profit)
    if clf is not None:
        pw2 = pwin_clf(clf[0], clf[1], test, BID_GRID)
        res["clf_bidder"] = _resolve_bidder(ev_hat, pw2, test)
    return res


def h2_incumbent_baseline(test):
    """H2 incumbent baseline = the recorded/logged bids (won set = rows with
    won==1, pay own bid). A reference for what the logged bidder achieved -
    NOT an oracle. The ceiling is oracle_bidder_ev (the pure value oracle)."""
    ltv = test["ltv_value"].to_numpy()
    h2 = test["bid_price"].to_numpy() / 1000.0
    hw = test["won"].to_numpy() == 1
    h_rev, h_cost = ltv[hw].sum(), h2[hw].sum()
    return {"roas": round(_roas(h_rev, h_cost), 4),
            "profit": round(float(h_rev - h_cost), 2), "n_won": int(hw.sum())}


def oracle_bidder_ev(pc, test):
    """PURE TRUE-VALUE ORACLE (the ceiling) = the SAME profit-max bidder as
    C1-C4, but fed the simulator's TRUE per-impression EV (ev_truth) and a
    win curve centred on the TRUE competing bid (lu7). No learned models →
    the apples-to-apples ceiling C1-C4 are measured against. Returns the same
    economic metric dict + the ceiling diagnostics (ev_ratio = 1.0; per-head
    AUC/logloss = Bayes-optimal from the saved TRUE probabilities, when
    those columns are loaded)."""
    from scipy.stats import norm
    ev_hat = test["ev_truth"].to_numpy()                              # TRUE EV ($)
    lu7_ecpm = np.maximum(test["lu7_competing_bid"].to_numpy(), 1e-6)  # TRUE competing bid (eCPM)
    b_ecpm = BID_GRID * 1000.0
    pw = norm.cdf((np.log(b_ecpm)[None, :]
                   - np.log(lu7_ecpm)[:, None]) / AFT_SCALE)          # TRUE win curve
    res = _resolve_bidder(ev_hat, pw, test)
    # EV diagnostics are exact by construction (the oracle IS the truth)
    res["ev_true_mean"] = round(float(ev_hat.mean()), 6)
    res["ev_pred_mean"] = res["ev_true_mean"]
    res["ev_ratio"] = 1.0
    res["ev_rmse"] = 0.0
    res["ev_spearman"] = 1.0
    # Bayes-optimal per-head AUC/logloss/rmse from the saved TRUE probabilities
    for stage, col, target in (("click", "p_click", "click"),
                               ("install", "p_install", "install"),
                               ("payer", "p_payer", "is_payer")):
        if col not in test.columns:
            continue
        pop = _eval_pop(test, stage)
        if pop[target].nunique() > 1:
            p = np.clip(pop[col].to_numpy(), 1e-6, 1 - 1e-6)
            res[f"auc_{stage}"] = round(roc_auc_score(pop[target], p), 4)
            res[f"logloss_{stage}"] = round(log_loss(pop[target], p), 4)
    if "e_ltv" in test.columns:
        pop = _eval_pop(test, "spend")
        if len(pop) > 5:
            res["rmse_spend"] = round(
                mean_squared_error(pop["ltv_value"], pop["e_ltv"]) ** 0.5, 4)
    return res


def shap_tier1(pc, models, test, n=3000):
    """Aggregated mean|SHAP| for the P(click) model - where each layer's signal sits."""
    import shap
    samp = test.sample(min(n, len(test)), random_state=0)
    X = _as_cat(samp, pc.t1_feats)
    expl = shap.TreeExplainer(models["click"])
    sv = expl.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:8]
    feats = pc.t1_feats
    return {feats[i]: round(float(imp[i]), 5) for i in order}


# ── orchestration ───────────────────────────────────────────────────
def run_profile(profile, do_shap=True, max_train=None, seed=0,
                oracle_features=False, chunk_rows=None, hist_clearing=False):
    t0_wall = time.time()
    if chunk_rows and hist_clearing:
        raise NotImplementedError(
            "hist_clearing is in-memory only (needs 2-pass cell stats); "
            "run without --chunk-rows for the v8 headline (golden/1M/10M).")
    pc = PipelineConfig(oracle=oracle_features)
    run = OUTPUT_DIR / profile
    path = run / "auctions.parquet"

    needed_set = schema.NEEDED | (set(schema.ORACLE_NUM + schema.ORACLE_CAT)
                                  if oracle_features else set())
    pf_cols = set(pq.ParquetFile(path).schema_arrow.names)
    needed = [c for c in needed_set if c in pf_cols]

    if chunk_rows:
        # ── streaming path (scale50m / scale100m) ────────────────────
        scan_cols = ["timestamp"] + [c for c in pc.t1_cat if c in pf_cols]
        t0_ts = float("inf")
        cat_uniques = {c: set() for c in pc.t1_cat if c in pf_cols}
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=chunk_rows, columns=scan_cols):
            chunk = batch.to_pandas()
            t0_ts = min(t0_ts, chunk["timestamp"].min())
            for c in cat_uniques:
                cat_uniques[c].update(chunk[c].dropna().unique())
        cat_levels = {c: sorted(v for v in vals if pd.notna(v))
                      for c, vals in cat_uniques.items()}
        print(f"  scan done: t0={t0_ts}, categories resolved")

        val_parts, test_parts, n_total = [], [], 0
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=chunk_rows, columns=needed):
            chunk = batch.to_pandas()
            for c in pc.t1_cat:
                if c in chunk.columns and c in cat_levels:
                    chunk[c] = pd.Categorical(chunk[c], categories=cat_levels[c])
            day = ((chunk["timestamp"] - t0_ts) // 86400).astype(int)
            val_parts.append(chunk[(day >= 20) & (day < 24)])
            test_parts.append(chunk[day >= 24])
            n_total += len(chunk)
        base_val = pd.concat(val_parts, ignore_index=True)
        base_test = pd.concat(test_parts, ignore_index=True)
        del val_parts, test_parts
        n_train = n_total - len(base_val) - len(base_test)
        print(f"  train: {n_train:,} (streamed)  "
              f"val: {len(base_val):,}  test: {len(base_test):,}")

        chunk_cfg = {"path": str(path), "needed": needed,
                     "cat_levels": cat_levels, "t0": t0_ts,
                     "chunk_rows": chunk_rows}
        out = {"profile": profile, "n_rows": n_total,
               "split_counts": {"train": n_train, "val": len(base_val),
                                "test": len(base_test)},
               "chunked": True, "chunk_rows": chunk_rows,
               "conditions": {}}
        base_train = None

    else:
        # ── standard in-memory path ───────────────────────────────────
        df = pd.read_parquet(path, columns=needed)
        for c in pc.t1_cat:                 # one global category code-space
            if c in df.columns:
                df[c] = df[c].astype("category")
        sp = temporal_split(df)
        df = df.assign(_split=sp.values)
        train_m, val_m, test_m = (df["_split"] == s
                                  for s in ("train", "val", "test"))
        out = {"profile": profile, "n_rows": len(df),
               "split_counts": {s: int((df["_split"] == s).sum())
                                for s in ("train", "val", "test")},
               "conditions": {}}
        base_train, base_val, base_test = df[train_m], df[val_m], df[test_m]
        if max_train and len(base_train) > max_train:
            base_train = base_train.sample(max_train, random_state=seed)
            out["train_subsampled_to"] = int(max_train)
        chunk_cfg = None

    out["h2_incumbent"] = h2_incumbent_baseline(base_test)
    out["oracle_features"] = oracle_features
    out["hist_clearing"] = hist_clearing

    for cond in ("C1", "C2", "C3", "C4"):
        try:
            va = censor.view(base_val, cond, keep_latents=oracle_features)
            if chunk_cfg:
                models = train_tier1(pc, None, va, cond, chunk_cfg=chunk_cfg)
                booster = train_tier2_aft(None, cond, chunk_cfg=chunk_cfg)
                res = evaluate_condition(pc, models, booster, base_test, cond)
            else:
                tr = censor.view(base_train, cond, keep_latents=oracle_features)
                te, feats = base_test, None
                if hist_clearing:           # v8: attach OOF/LOO clearing features
                    te = base_test.copy()   # local encodings per condition
                    attach_hist_clearing(tr, va, te,
                                         censor.CONDITIONS[cond]["clearing"])
                    feats = t2_feats(cond, hist_clearing, view_cols=tr.columns)
                models = train_tier1(pc, tr, va, cond)
                booster = train_tier2_aft(tr, cond, feats=feats)
                clf = train_tier2_clf(tr, va, cond, feats=feats)
                # evaluate on (uncensored) master test = fair cross-condition yardstick
                res = evaluate_condition(pc, models, booster, te, cond,
                                         feats=feats, clf=clf)
            if do_shap:
                try:
                    res["shap_click_top"] = shap_tier1(pc, models, base_test)
                except Exception as e:
                    res["shap_error"] = str(e)[:200]
            out["conditions"][cond] = res
            print(f"  {cond}: ROAS {res.get('roas')} profit {res.get('profit')} "
                  f"AUC_click {res.get('auc_click')} won {res.get('n_won')}")
        except Exception as e:
            import traceback
            out["conditions"][cond] = {"ERROR": str(e)[:300],
                                       "trace": traceback.format_exc()[-800:]}
            print(f"  {cond}: ERROR {e}")

    # ---- pure true-value ORACLE column (the ceiling): true EV + true LU7
    #      through the SAME profit-max bidder as C1-C4 (apples-to-apples) ----
    try:
        out["conditions"]["oracle"] = oracle_bidder_ev(pc, base_test)
        o = out["conditions"]["oracle"]
        print(f"  oracle: ROAS {o.get('roas')} profit {o.get('profit')} "
              f"AUC_click {o.get('auc_click')} won {o.get('n_won')}")
    except Exception as e:
        import traceback
        out["conditions"]["oracle"] = {"ERROR": str(e)[:300],
                                       "trace": traceback.format_exc()[-800:]}
        print(f"  oracle: ERROR {e}")

    # ---- paired DeLong CIs on the layer contrasts (AFT + CLF companion) ----
    won_te = base_test["won"].to_numpy()
    contrasts = {}
    for name, ca, cb in (("ssp_C3_C1", "C3", "C1"), ("ssp_C4_C2", "C4", "C2"),
                         ("mmp_C2_C1", "C2", "C1")):
        pa = out["conditions"].get(ca, {}).get("_pw", {})
        pb = out["conditions"].get(cb, {}).get("_pw", {})
        for mdl in ("aft", "clf"):
            if mdl in pa and mdl in pb:
                try:
                    contrasts[f"{name}_{mdl}"] = _delong(won_te, pa[mdl], pb[mdl])
                except Exception as e:
                    contrasts[f"{name}_{mdl}"] = {"ERROR": str(e)[:120]}
    if contrasts:
        out["contrasts"] = contrasts
    for c in out["conditions"].values():
        if isinstance(c, dict):
            c.pop("_pw", None)

    # ---- normalized economics (Klimis #3): capture vs oracle + SSP reductions ----
    ov0 = out["conditions"].get("oracle", {})
    econ = {}
    if all("profit" in out["conditions"].get(c, {}) for c in ("C1", "C2", "C3", "C4")):
        if ov0.get("profit", 0) and ov0["profit"] > 0:
            econ["profit_capture_vs_oracle"] = {
                c: round(out["conditions"][c]["profit"] / ov0["profit"], 4)
                for c in ("C1", "C2", "C3", "C4")}
        for lab, ca, cb in (("C3_vs_C1", "C3", "C1"), ("C4_vs_C2", "C4", "C2")):
            opa = out["conditions"][ca].get("overpay_per_won")
            opb = out["conditions"][cb].get("overpay_per_won")
            spa = out["conditions"][ca].get("surplus_per_won")
            spb = out["conditions"][cb].get("surplus_per_won")
            if opa is not None and opb:
                econ.setdefault("overpay_reduction_pct", {})[lab] = round(
                    100.0 * (1.0 - opa / opb), 2)
            if spa is not None and spb:
                econ.setdefault("surplus_uplift_pct", {})[lab] = round(
                    100.0 * (spa / spb - 1.0), 2)
    if econ:
        out["economics"] = econ

    # lifts vs C1 and vs oracle
    if all(c in out["conditions"] and "roas" in out["conditions"][c]
           for c in ("C1", "C2", "C3", "C4")):
        r = {c: out["conditions"][c]["roas"] for c in ("C1", "C2", "C3", "C4")}
        out["roas_lift_vs_C1"] = {c: round(r[c] - r["C1"], 4)
                                  for c in ("C2", "C3", "C4")}
    # C1-C4 vs the value-oracle ceiling (regret = oracle profit - theirs)
    ov = out["conditions"].get("oracle", {})
    have_cxx = all("roas" in out["conditions"].get(c, {})
                   for c in ("C1", "C2", "C3", "C4"))
    if have_cxx and ov.get("roas") and not np.isnan(ov["roas"]):
        out["pct_of_oracle"] = {
            c: round(100 * out["conditions"][c]["roas"] / ov["roas"], 1)
            for c in ("C1", "C2", "C3", "C4")}
    if have_cxx and "profit" in ov:
        out["profit_regret_vs_oracle"] = {
            c: round(ov["profit"] - out["conditions"][c]["profit"], 2)
            for c in ("C1", "C2", "C3", "C4")}
    out["elapsed_s"] = round(time.time() - t0_wall, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--oracle-features", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=None,
                    help="Enable streaming mode (scale50m/scale100m). "
                         "Batch size in rows, e.g. 5000000.")
    ap.add_argument("--hist-clearing", action="store_true",
                    help="v8: attach OOF/LOO hist_clearing_ssp/dsp Tier-2 features.")
    args = ap.parse_args()
    out = run_profile(args.profile, do_shap=not args.no_shap,
                      max_train=args.max_train,
                      oracle_features=args.oracle_features,
                      chunk_rows=args.chunk_rows,
                      hist_clearing=args.hist_clearing)
    d = OUTPUT_DIR / args.profile / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    fname = ("results_histclearing.json" if args.hist_clearing
             else "results_oracle.json" if args.oracle_features else "results.json")
    (d / fname).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "conditions"},
                     indent=2))
    print("wrote", d / fname)


if __name__ == "__main__":
    main()
