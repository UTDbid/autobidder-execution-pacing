# T9 v10 anchoring — published value-of-lost-price-information bands (5 Jul 2026)

**Purpose (proposal §7.2, sign-off Q4):** pick the "faithful niche market" point ρ* on the v10 dial by
matching T9's measurable — the **C3−C1 price-RMSE gain** (`price_rmse_all` / `price_rmse_lost`, now a
standard pipeline output) — to the published real-world value of lost/censored price information.
Sources: the three post-2019 papers from the 2 Jul sweep, PDFs read in full (extraction agents, 5 Jul).

## What each paper provides

| Paper | Setting | The usable contrast | Number |
|---|---|---|---|
| **Wang 2023** (KBS; censored GMM) | iPinYou 2nd-price, simulated censorship, **win rate 31.6%** (≈ T9's 30%) | **GMM (full price info) vs CGMM (interval-censored), same model family — the only genuine full-info-vs-censored arm in the literature we hold** | **RMSE gain 8.1% pooled** (all-rows); 10–13% on heavily-censored campaigns (win<22%); ~0 to −5% when win>43% |
| **Seo 2022** (IEEE BigData; first-price censorship geometry) | iPinYou re-labeled SPA vs FPA regimes | Same-model *information ablation*: exact prices on won rows vs no exact prices anywhere | ANLP +2.7% / BCE +5.6% (campaign spread +1.3%…+34%) — likelihood scale, reverse direction |
| **Li 2022** (KDD; ADM) | FreeWheel production 2nd-price, ~62% censored; **true lost prices used for EVALUATION ONLY, never training** | Censoring-aware NLL vs naive won-only (technique value, not information value) | MAE −16%…−26% (log-price scale) |

**Key structural fact: no published model is ever *trained* on true lost prices** (Li explicitly
forbids it; Seo's FPA regime removes prices; only Wang's GMM/CGMM pair varies the information level
within one family). So Wang 2023 is **the** anchor source; Seo and Li are flanking context
(lower-flavor and technique-upper-flavor respectively).

## The band

**At T9's win-rate regime (~30%): full-price-information is worth ≈ 8% all-rows price-RMSE**
(Wang pooled, matching win rate), with a defensible range **5–13%** (Wang's censored-campaign spread).
Lost-row-only gain runs somewhat above the pooled figure (lost rows are ~2/3 of the data and carry the
censoring; Wang caveat #1) → **lost-row band ≈ 8–15%**.

Caveats carried explicitly: 2nd-price data (their "market price" = highest competing bid = T9's LU7 —
good analogue); Wang's censorship simulated by replay; GMM-vs-CGMM confounds estimator quality with
information (their #2997 negative gain shows the noise floor); metrics in iPinYou price units (relative
% used, not levels). Seo validates T9's first-price censoring geometry exactly (FPA winner's label =
interval z∈(0,b] — precisely C1's `aft_interval_labels`).

## Mapping rule (step 3)

1. From the 1M v10 runs, read T9's C3−C1 `price_rmse_all` (and `_lost`) gain at ρ=0 and ρ=1.
2. Choose ρ* where the T9 gain falls in the **8% central / 5–13% band** (interpolate; confirm with a
   100K mini-grid of the gain vs ρ if the endpoints straddle the band).
3. The anchored reason-3 verdict = the **CLF ranking contrast + economics at ρ\*** (1M, DeLong CI).

**Pre-registered observation (before the 1M lands):** T9's measured C3−C1 all-rows RMSE gain in the
v8/ρ≈0 world was already **−8.3%** (14.48→13.28, 1M; `docs/T9_SSP_Null_Validity_Discussion` test #1) —
i.e. **on the Wang anchor**. If the v10 ρ=0 point reproduces ~8%, the anchor points to **low ρ***,
where the CLF ranking contrast is small (+0.003–0.004 @100K) — reason 3 would *survive* at the
anchored point, with the v10 dose-response bounding how quickly it would fail in more private markets.
If v10's ρ=0 gain comes in below the band, ρ* moves interior and the verdict is read there. Stated
now so the 1M numbers can't be accused of steering the anchor.
