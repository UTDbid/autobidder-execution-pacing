"""App pool (build_apps) and campaign pool (build_campaigns) - v6.

Two pools carrying the entity latents the v6 auction engine needs:

  Apps      - app_id, app_category, la1_app_quality
  Campaigns - campaign_id, advertiser_id, advertiser_scale, ad_genre,
              lc1_creative_appeal, lc2_game_quality, sample_weight

sample_weight is proportional to the campaign's budget-tier share
(benchmarks.advertiser_scale.*.campaign_share) divided by the number of
campaigns in that tier, normalised to sum to 1; the auction engine uses it as
the campaign-sampling probability so major-advertiser campaigns win a
disproportionate share of auctions consistent with market concentration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_apps(cfg) -> pd.DataFrame:
    """Fixed app pool: category mix from benchmarks, LA1 quality latent."""
    n = int(cfg.profile["n_apps"])
    cats = cfg.benchmarks["app_categories"]
    sigma_app = cfg.benchmarks["entity_latents"]["sigma_app"]
    rng = cfg.rng("app_pool")

    app_ids, app_cats = [], []
    i = 0
    for cat, spec in cats.items():
        n_cat = max(1, round(spec["share"] * n))
        for _ in range(n_cat):
            app_ids.append(f"app_{i:05d}")
            app_cats.append(cat)
            i += 1

    # trim to n (rounding may overshoot slightly)
    app_ids = app_ids[:n]
    app_cats = app_cats[:n]

    df = pd.DataFrame({
        "app_id": app_ids,
        "app_category": app_cats,
        "la1_app_quality": rng.lognormal(0.0, sigma_app, len(app_ids)),
    })
    df = df.sample(frac=1.0,
                   random_state=int(rng.integers(0, 2**31))
                   ).reset_index(drop=True)

    # BN edge #1 (pairing): per-app audience profile LA2 ~ Dirichlet(k_audience *
    # centroid[app_category]). Built ONLY when the pairing edge is ON, from a
    # SEPARATE rng stream, so app_id / category / la1 / order stay byte-identical
    # when OFF. The global archetype mix is restored by the IPF rake at the join.
    if cfg.bn_edges.get("pairing"):
        bn = cfg.benchmarks["bn"]
        arch_labels = list(cfg.archetypes["shares"])
        k_aud = float(bn["k_audience"])
        cents = bn["audience_centroids"]
        rng2 = cfg.rng("app_audience")
        la2 = np.empty((len(df), len(arch_labels)))
        for cat in df["app_category"].unique():
            mask = (df["app_category"] == cat).to_numpy()
            cen = np.asarray(cents[cat], dtype=float)
            cen = cen / cen.sum()
            la2[mask] = rng2.dirichlet(k_aud * cen, size=int(mask.sum()))
        for j, lab in enumerate(arch_labels):
            df[f"la2_{lab}"] = la2[:, j]
    return df


def build_campaigns(cfg) -> pd.DataFrame:
    """Fixed campaign pool with LC1/LC2 latents and budget-share weights.

    Advertiser count comes from profiles.n_advertisers; at least one
    advertiser per scale tier is guaranteed. Campaigns are distributed across
    advertisers proportional to their tier's campaign_share so major-tier
    advertisers get more campaigns even though fewer of them exist.
    """
    n_campaigns = int(cfg.profile["n_campaigns"])
    n_advertisers = int(cfg.profile["n_advertisers"])
    bm = cfg.benchmarks
    scales = bm["advertiser_scale"]
    scale_labels = list(scales.keys())
    rng = cfg.rng("campaign_pool")

    # ── advertiser pool ──────────────────────────────────────────────
    adv_share_p = np.array([scales[s]["advertiser_share"]
                            for s in scale_labels], dtype=float)
    adv_share_p /= adv_share_p.sum()

    # guarantee >=1 advertiser per tier, fill remainder by share draw
    adv_scales = list(scale_labels)          # one of each
    remainder = n_advertisers - len(scale_labels)
    if remainder > 0:
        drawn = rng.choice(scale_labels, size=remainder, p=adv_share_p)
        adv_scales.extend(drawn.tolist())
    rng.shuffle(adv_scales)                  # no ordering artefact

    # ── campaign counts per advertiser ───────────────────────────────
    camp_share = np.array([scales[s]["campaign_share"]
                           for s in scale_labels], dtype=float)
    camp_share /= camp_share.sum()

    scale_idx = {s: i for i, s in enumerate(scale_labels)}
    scale_counts_adv = np.array(
        [adv_scales.count(s) for s in scale_labels], dtype=float)
    # weight per advertiser = campaign_share / n_advertisers_in_tier
    adv_w = np.array(
        [camp_share[scale_idx[s]] / scale_counts_adv[scale_idx[s]]
         for s in adv_scales])
    adv_w /= adv_w.sum()

    # integer campaign counts per advertiser (rounded, fix any drift)
    adv_camp_n = np.round(adv_w * n_campaigns).astype(int)
    adv_camp_n = np.maximum(adv_camp_n, 1)          # each advertiser >=1 campaign
    diff = n_campaigns - int(adv_camp_n.sum())
    adv_camp_n[int(np.argmax(adv_w))] += diff       # absorb rounding error

    # ── campaign rows ────────────────────────────────────────────────
    genre_labels = list(bm["ad_genre_mix"].keys())
    genre_p = np.array([bm["ad_genre_mix"][g]
                        for g in genre_labels], dtype=float)
    genre_p /= genre_p.sum()
    sigma_cre = bm["entity_latents"]["sigma_cre"]
    sigma_game = bm["entity_latents"]["sigma_game"]

    rows = []
    for adv_idx, (scale, n_c) in enumerate(zip(adv_scales, adv_camp_n)):
        genres = rng.choice(genre_labels, size=n_c, p=genre_p)
        lc1 = rng.lognormal(0.0, sigma_cre, n_c)
        lc2 = rng.lognormal(0.0, sigma_game, n_c)
        for j in range(n_c):
            rows.append({
                "campaign_id": f"camp_{len(rows):04d}",
                "advertiser_id": f"adv_{adv_idx:03d}",
                "advertiser_scale": scale,
                "ad_genre": genres[j],
                "lc1_creative_appeal": float(lc1[j]),
                "lc2_game_quality": float(lc2[j]),
            })

    df = pd.DataFrame(rows[:n_campaigns])

    # ── sample_weight ────────────────────────────────────────────────
    tier_counts = df.groupby("advertiser_scale")["campaign_id"] \
                    .transform("count").astype(float)
    camp_share_map = {s: scales[s]["campaign_share"] for s in scale_labels}
    df["sample_weight"] = (df["advertiser_scale"].map(camp_share_map)
                           / tier_counts)
    df["sample_weight"] /= df["sample_weight"].sum()

    return df.reset_index(drop=True)
