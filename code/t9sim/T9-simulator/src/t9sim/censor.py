"""Censoring layer - the C1-C4 condition views (v6, corrected map).

One master table; four views; identical rows everywhere. Only label
VISIBILITY differs (column- AND row-conditional):

  C1  DSP only        E/F/G on won rows only          H4 hidden
  C2  DSP + MMP       E/F/G on ALL rows               H4 hidden
  C3  DSP + SSP       E/F/G on won rows only          H4 visible
  C4  all layers      E/F/G on ALL rows               H4 visible

Won-row attribution is standard MMP postbacks (every DSP gets them); all-rows
attribution is MMP OWNERSHIP (cross-network SDK visibility). Latents and
estimand columns are dropped from every view.

The censorable column list and the hidden latent/estimand sets come from
schema.py (the single source of truth). CONDITIONS stays local because the
invariant tests cross-check it against their own hardcoded expectations.
"""
from __future__ import annotations

import numpy as np

from . import schema

CONDITIONS = {
    "C1": {"attribution_all_rows": False, "clearing": False},
    "C2": {"attribution_all_rows": True, "clearing": False},
    "C3": {"attribution_all_rows": False, "clearing": True},
    "C4": {"attribution_all_rows": True, "clearing": True},
}

# re-exported from schema so consumers/tests can read censor.ATTRIBUTION_COLS
ATTRIBUTION_COLS = schema.ATTRIBUTION_COLS


def view(master, condition, keep_latents=False):
    """Return the condition view of a master table (a censored copy;
    at training scale apply the same masks lazily instead).

    Censored cells are NaN - distinct from -1 (observed no-event) and from
    0 labels. "Hidden" columns (H4 in C1/C2) are all-NaN, not dropped.

    keep_latents=True retains the LU/LA/LC latent columns (for the
    oracle-features diagnostic only); estimands (p_*, ev_truth) and user_id
    are ALWAYS dropped, and the label/H4 censoring is unchanged.
    """
    spec = CONDITIONS[condition]
    drop = [c for c in master.columns
            if (c.startswith(schema.LATENT_PREFIXES) and not keep_latents)
            or c in schema.ESTIMANDS]
    df = master.drop(columns=drop).copy()
    # float64 up front: keeps dtypes identical across conditions after masking
    # (presence-filtered: v11 G5 n_transactions exists only when its flag is ON)
    attr = [c for c in ATTRIBUTION_COLS if c in df.columns]
    df[attr] = df[attr].astype("float64")

    if not spec["attribution_all_rows"]:
        lost = df["won"] == 0
        df.loc[lost, attr] = np.nan

    if not spec["clearing"]:
        df["clearing_price"] = np.nan
        # v8 SSP-only features (H5 bid_density, H6 hist_clearing_ssp) follow H4:
        # visible only to C3/C4. NaN the whole column in C1/C2 (when present).
        for col in schema.V8_SSP_ONLY:
            if col in df.columns:
                df[col] = np.nan
    return df
