"""Anti-tautology cross-check: an INDEPENDENT hardcoded copy of the immutable
column/feature contract. A typo in schema.py fails loudly here. Do NOT import
the expected values from schema - that would defeat the purpose.
"""
from t9sim import schema

CAT = ["region", "city", "os", "os_version", "device_type",
       "app_id", "app_category", "ad_exchange", "slot_format",
       "advertiser_id", "advertiser_scale", "campaign_id", "ad_genre"]
NUM = ["slot_width", "slot_height", "hour_of_day", "day_of_week"]
ATTRIBUTION = ["click", "click_timestamp", "install", "install_timestamp",
               "is_payer", "ltv_value", "ltv_7d", "ltv_30d"]
ORACLE_NUM = ["lu2_click_prop", "lu3_install_prop", "lu4_payer_prob",
              "lu5_ltv_mult", "lu6_casual", "lu6_strategy", "lu6_rpg",
              "lu6_hypercasual", "la1_app_quality", "lc1_creative_appeal",
              "lc2_game_quality"]
ORACLE_CAT = ["lu1_archetype"]
LATENT_PREFIXES = ("lu1_", "lu2_", "lu3_", "lu4_", "lu5_", "lu6_", "lu7_",
                   "la1_", "la2_", "lc1_", "lc2_")
ESTIMANDS = {"p_click", "p_install", "p_payer", "e_ltv", "ev_truth", "user_id"}
LABELS = {"click", "click_timestamp", "install", "install_timestamp",
          "is_payer", "ltv_value", "ltv_7d", "ltv_30d", "won",
          "clearing_price", "min_bid_to_win"}   # v11 B1: H10 in labels


def test_feature_lists():
    assert schema.CAT_FEATURES == CAT
    assert schema.NUM_FEATURES == NUM
    assert schema.TIER1_FEATURES == CAT + NUM
    assert schema.TIER2_FEATURES == CAT + NUM


def test_oracle_lists():
    assert schema.ORACLE_NUM == ORACLE_NUM
    assert schema.ORACLE_CAT == ORACLE_CAT


def test_attribution_and_labels():
    assert schema.ATTRIBUTION_COLS == ATTRIBUTION
    assert schema.LABELS == LABELS


def test_hidden_sets():
    assert schema.LATENT_PREFIXES == LATENT_PREFIXES
    assert schema.ESTIMANDS == ESTIMANDS


def test_needed_covers_features_and_labels():
    # the pipeline must read every feature + every attribution/price column
    assert set(CAT + NUM) <= schema.NEEDED
    for col in ("timestamp", "won", "bid_price", "floor_price",
                "clearing_price", "lu7_competing_bid", "ev_truth"):
        assert col in schema.NEEDED, col
    for col in ATTRIBUTION:
        assert col in schema.NEEDED, col


def test_no_feature_is_forbidden():
    # a feature must never be a label / estimand / latent
    for f in schema.TIER1_FEATURES + schema.TIER2_FEATURES:
        assert f not in schema.LABELS
        assert f not in schema.ESTIMANDS
        assert not f.startswith(schema.LATENT_PREFIXES)
