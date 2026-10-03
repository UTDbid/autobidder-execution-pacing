"""Single source of truth for the master-table column / feature contract.

The v6 codebase duplicated these lists across auctions.py (producer), censor.py
and pipeline.py (consumers), and the tests - and they drifted independently.
Here they live once; consumers import them. The ONE deliberate exception is the
test suite, which keeps its OWN hardcoded copy of the immutable contract so a
typo here fails loudly (see tests/test_schema.py) - the anti-tautology rule.

IMMUTABLE OUTPUT-COLUMN CONTRACT (renaming silently breaks censor, pipeline,
tests, diagrams, notebooks). Do not rename without updating every consumer.
"""
from __future__ import annotations

# ── observable model-input features ────────────────────────────────────
CAT_FEATURES = [
    "region", "city", "os", "os_version", "device_type",
    "app_id", "app_category", "ad_exchange", "slot_format",
    "advertiser_id", "advertiser_scale", "campaign_id", "ad_genre",
]
NUM_FEATURES = ["slot_width", "slot_height", "hour_of_day", "day_of_week"]
TIER1_FEATURES = CAT_FEATURES + NUM_FEATURES
TIER2_FEATURES = CAT_FEATURES + NUM_FEATURES            # AFT context (LU7 drivers)

# ── v8 SSP/price-layer feature columns (PROPOSED) ──────────────────────
# Emitted ONLY when the relevant `bn.edges` flag is ON; with all v8 edges OFF
# the master is byte-identical v7 (no new columns -> fingerprint unchanged).
# NOT yet folded into TIER1/TIER2_FEATURES — the pipeline wires them in
# per-condition when present (H5/H6 SSP-only = C3/C4; H7 = all conditions).
# See `Simulator_Schema - June 10 (v8).md` §Price/Competition Dependency Layer.
V8_SSP_NUM = ["bid_density", "hist_clearing_ssp", "hist_clearing_dsp"]  # H5/H6/H7
# C3/C4 only (follow H4); H7 = all conds. v11 adds H10 min_bid_to_win (B1 arm:
# = max(LU7, H1) on WON rows, NaN else — the real first-price feedback field).
V8_SSP_ONLY = ["bid_density", "hist_clearing_ssp", "min_bid_to_win"]

# ── oracle-only LATENT features (--oracle-features upper bound) ─────────
# the generative drivers, NOT the computed estimands (p_*/ev_truth) which
# would be answer-leakage.
ORACLE_NUM = [
    "lu2_click_prop", "lu3_install_prop", "lu4_payer_prob", "lu5_ltv_mult",
    "lu6_casual", "lu6_strategy", "lu6_rpg", "lu6_hypercasual",
    "la1_app_quality", "lc1_creative_appeal", "lc2_game_quality",
]
ORACLE_CAT = ["lu1_archetype"]

# ── attribution (censorable outcome) columns: E/F/G ────────────────────
ATTRIBUTION_COLS = [
    "click", "click_timestamp", "install", "install_timestamp",
    "is_payer", "ltv_value", "ltv_7d", "ltv_30d",
]

# ── labels / targets the model must NEVER receive as a feature ─────────
LABELS = {
    "click", "click_timestamp", "install", "install_timestamp",
    "is_payer", "ltv_value", "ltv_7d", "ltv_30d", "won", "clearing_price",
    "min_bid_to_win",   # v11 H10: post-auction feedback — training label only, never a feature
}

# ── hidden generative drivers (latents) + estimands ────────────────────
LATENT_PREFIXES = (
    "lu1_", "lu2_", "lu3_", "lu4_", "lu5_", "lu6_", "lu7_",
    "la1_", "la2_", "lc1_", "lc2_",
)
# estimands (computed probabilities) + master-lineage id; dropped from every
# censor view and forbidden as features.
ESTIMANDS = {"p_click", "p_install", "p_payer", "e_ltv", "ev_truth", "user_id"}

# ── minimal parquet read set for the pipeline (saves ~40% RAM) ─────────
NEEDED = set(CAT_FEATURES + NUM_FEATURES + [
    "timestamp", "won",
    "click", "click_timestamp", "install", "install_timestamp",
    "is_payer", "ltv_value", "ltv_7d", "ltv_30d",
    "bid_price", "floor_price", "clearing_price",
    "lu7_competing_bid", "ev_truth",
    "p_click", "p_install", "p_payer", "e_ltv",   # true per-head probs → oracle Bayes-optimal AUC ceilings (small RAM cost; drop for 100M if tight)
    "bid_density",          # v8 H5 (present only when competition ON; pf-presence filtered)
    "min_bid_to_win",       # v11 H10 (present only when flag ON; pf-presence filtered)
])
