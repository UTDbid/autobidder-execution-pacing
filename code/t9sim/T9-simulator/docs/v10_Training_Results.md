# T9 — V10 Training/Eval Results (anchored point, 1M n=10 · 10M n=10)

**Schema:** V10 "Private-Rival Market" (BUILT + EVALUATED; branch `t9-v10-rival`, not merged; v7 stays
operative pending sign-off). **Generator:** v10 build, flag-gated — `rival_pool` OFF reproduces v7
byte-identical (gate G1, hash-checked). **Operating point:** the externally **anchored** privateness
ρ\* = 0.8 (value-of-lost-price-information band, Wang et al. 2023 GMM-vs-CGMM: 5–13% all-rows RMSE
gain at ~30% win rate; T9 ρ-grid mapping 24.8% @ρ=0 → 3.7% @ρ=1 → ρ\*≈0.8; `v10_Anchor_Bands.md`).

**V10 "ON" = v7 funnel BN edges (#1–#5) + `rival_pool`** (K=8 rival archetypes; LU7 = max over
participating rivals' structured bids; H9 = endogenous rival count, SSP-only) **+** pipeline
`hist_clearing=True` (leak-gated OOF/LOO rolling clearing features, C3/C4).

**Pipeline (extended 5 Jul 2026):** Tier-1 4-head hurdle (XGBoost) · Tier-2 = **AFT + first-class
binary win CLASSIFIER co-reported** (monotone sweep of bid as explicit input) · paired **DeLong**
tests with 95% CIs on AUC contrasts · **ECE/MCE** win-calibration · closed-form lognormal **CRPS** +
price **RMSE** on all/won/lost rows · **TWO bidders** — AFT-curve bidder and classifier-curve (CLF)
bidder — so every economic metric is **model-robust** · temporal split · `--no-shap`.

**Conditions:** **C1** DSP · **C2** DSP+MMP · **C3** DSP+SSP · **C4** all · **oracle** = same pipeline
handed the true latents (upper bound).

**Seeds / n:** 1M = **10 seeds** (90213–90222), no outlier removal; 10M = **10 seeds** (90213–90222),
**one fresh process per seed** (scale10m convention; parquet deleted after eval, reproducible by seed).
Cells are seed means; headline contrasts carry mean [95% t-CI] + sign counts.

**Raw per-seed JSON:** `docs/results/v10_anchor_s13.json` … `s22.json` (1M);
`docs/results/v10_10m_s13.json` … `s22.json` (10M); aggregates `v10_anchor_n10.json`, `v10_10m_n10.json`.
Runners: `scripts/v10_anchor_5seed.py`, `v10_anchor_n10.py`, `v10_10m_worker.py` + `v10_10m_driver.py`.
Earlier stages (100K ablation, ρ-grid, decisive ρ=1 test): `docs/results/v10_*.json`, write-up
`v10_Results_5Jul2026.md`.

---

## 1M — anchored point ρ\*=0.8, n=10 seeds (means)

| Metric | C1 (DSP) | C2 (+MMP) | C3 (+SSP) | C4 (all) | Oracle |
|---|---|---|---|---|---|
| **Tier 1 training** |  |  |  |  |  |
| ev_spearman | 0.445 | **0.552** | 0.445 | **0.552** | 1.000 |
| ev_ratio | 0.633 | **0.968** | 0.633 | **0.968** | 1.000 |
| auc_click | 0.6797 | **0.7100** | 0.6797 | **0.7100** | 0.7843 |
| auc_install | 0.6217 | **0.6725** | 0.6217 | **0.6725** | 0.8609 |
| auc_payer | 0.5563 | **0.6141** | 0.5563 | **0.6141** | 0.8357 |
| rmse_spend, E(spend\|payer) ($)* | 173.2 | **155.9** | 173.2 | **155.9** | 143.6 |
| **Tier 2 training, AFT head (price model)** |  |  |  |  |  |
| auc_win (AFT) | 0.7648 | 0.7648 | **0.7823** | **0.7823** | — |
| ece_win (AFT) | **0.0517** | **0.0517** | 0.0554 | 0.0554 | — |
| mce_win (AFT) | 0.189 | 0.189 | **0.118** | **0.118** | — |
| price_rmse_all | 25.7 | 25.7 | **24.4** | **24.4** | — |
| price_rmse_lost | 29.7 | 29.7 | **28.3** | **28.3** | — |
| price_crps_all | 0.658 | 0.658 | **0.621** | **0.621** | — |
| price_crps_lost | 0.577 | 0.577 | **0.560** | **0.560** | — |
| **Tier 2 training, CLF head (win classifier)** |  |  |  |  |  |
| auc_win (CLF) | 0.8091 | 0.8091 | **0.8227** | **0.8227** | — |
| logloss_win (CLF) | 0.4525 | 0.4525 | **0.4389** | **0.4389** | — |
| **Tier 1+2, Profit(bid) optimizer, AFT bidder** |  |  |  |  |  |
| overpay CPM ($) | 2.29 | 3.38 | **2.13** | 3.15 | 11.87 |
| profit CPM ($) | 9.01 | 14.56 | 8.27 | **16.46** | 184.58 |
| profit, total ($) | 653 | 932 | 585 | **1,059** | 2,427 |
| ROAS | 2.97 | 3.70 | 2.90 | **4.13** | 10.13 |
| n_won | 60,233 | **64,428** | 58,889 | 62,957 | 12,965 |
| **Tier 1+2, Profit(bid) optimizer, CLF bidder** |  |  |  |  |  |
| overpay CPM ($) | **1.10** | 1.22 | **1.10** | 1.23 | — |
| profit CPM ($) | 6.82 | **15.57** | 7.21 | 15.45 | — |
| profit, total ($) | 367 | 778 | 401 | **786** | — |
| ROAS | 3.53 | **6.59** | 3.64 | 6.51 | — |
| n_won | 49,833 | 51,209 | 50,622 | **52,041** | — |

## 10M — anchored point ρ\*=0.8, n=10 seeds (90213–90222) (means)

| Metric | C1 (DSP) | C2 (+MMP) | C3 (+SSP) | C4 (all) | Oracle |
|---|---|---|---|---|---|
| **Tier 1 training** |  |  |  |  |  |
| ev_spearman | 0.589 | **0.633** | 0.589 | **0.633** | 1.000 |
| ev_ratio | 0.524 | **0.889** | 0.524 | **0.889** | 1.000 |
| auc_click | 0.7090 | **0.7212** | 0.7090 | **0.7212** | 0.7842 |
| auc_install | 0.6719 | **0.6994** | 0.6719 | **0.6994** | 0.8595 |
| auc_payer | 0.5855 | **0.6835** | 0.5855 | **0.6835** | 0.8352 |
| rmse_spend, E(spend\|payer) ($)* | 208.0 | **184.3** | 208.0 | **184.3** | 178.2 |
| **Tier 2 training, AFT head (price model)** |  |  |  |  |  |
| auc_win (AFT) | 0.7772 | 0.7772 | **0.7899** | **0.7899** | — |
| ece_win (AFT) | **0.0521** | **0.0521** | 0.0540 | 0.0540 | — |
| mce_win (AFT) | 0.119 | 0.119 | **0.095** | **0.095** | — |
| price_rmse_all | 24.59 | 24.59 | **24.10** | **24.10** | — |
| price_rmse_lost | 28.54 | 28.54 | **27.98** | **27.98** | — |
| price_crps_all | 0.634 | 0.634 | **0.614** | **0.614** | — |
| price_crps_lost | 0.558 | 0.558 | **0.551** | **0.551** | — |
| **Tier 2 training, CLF head (win classifier)** |  |  |  |  |  |
| auc_win (CLF) | 0.8142 | 0.8142 | **0.8280** | **0.8280** | — |
| logloss_win (CLF) | 0.4463 | 0.4463 | **0.4318** | **0.4318** | — |
| **Tier 1+2, Profit(bid) optimizer, AFT bidder** |  |  |  |  |  |
| overpay CPM ($) | 2.15 | 3.77 | **2.13** | 3.79 | 11.89 |
| profit CPM ($) | 15.54 | 20.75 | 15.93 | **21.04** | 148.12 |
| profit, total ($) | 7,783 | 11,798 | 7,904 | **11,858** | 19,202 |
| ROAS | 4.65 | 3.99 | **4.74** | 3.99 | 8.06 |
| n_won | 496,713 | **566,457** | 491,135 | 561,394 | 129,506 |
| **Tier 1+2, Profit(bid) optimizer, CLF bidder** |  |  |  |  |  |
| overpay CPM ($) | **1.18** | 1.45 | 1.19 | 1.45 | — |
| profit CPM ($) | 10.23 | **13.82** | 10.25 | 12.90 | — |
| profit, total ($) | 4,746 | **6,575** | 4,817 | 6,236 | — |
| ROAS | 4.65 | **5.31** | 4.62 | 5.01 | — |
| n_won | 442,763 | 477,606 | 449,195 | **483,266** | — |

*\*rmse_spend (the E(spend|payer) head, RMSE on test payer rows) is whale-tail-dominated: the target
is heavy-tailed, so even the oracle sits within ~3-9% of C2 (little predictable headroom) and the
seed SD is ~40% of the mean. Read it as directional support only; the durable E(LTV) evidence is
ev_ratio / ev_spearman (the EV product contains this head) and profit.*

*Group legend (both tables above): Tier 1 training = funnel/value heads only. Tier 2 training is
split by head: the AFT (a censored price model; its price face is scored by RMSE/CRPS, its derived
probability face by AUC/ECE/MCE) and the CLF (a binary win classifier; AUC/log-loss). Tier 1+2 =
bidder economics, composing
both tiers through the Profit(bid) profit-max optimizer. The grouping is the design's orthogonality
made visible: every Tier-1 row moves only with MMP (C2/C4), every Tier-2 row only with SSP (C3/C4),
and only the Tier-1+2 rows mix. CPM rows (overpay CPM, profit CPM) are $ per 1,000 won impressions
(a won auction = a served impression, so the denominator is implied); the underlying pipeline fields
`overpay_per_won` / `surplus_per_won` are per single won auction, x1000 here. The rows relate as:
profit, total = profit CPM x n_won / 1000. The won-price slice (price_rmse_won) is omitted from these tables for compactness; it lives in the raw per-seed JSONs, and the won-vs-lost metric-illusion argument it supports is made in the method-benchmark doc. The economics block is split by bidder; rows where the two bidders DISAGREE (overpay CPM on C3, ROAS at 10M) are the model-robustness checks - only claims that agree across both bidder subsections are quoted. The oracle bidder rides the AFT curve, so the CLF subsection has no oracle values.*

---

## Statistical support map (paired 95% t-CIs, n=10 per scale)

The claim-sorting key for the two anchored-point tables above, row-for-row (same groups, same
order). **Improved** = did the layer move the metric in the better direction (✔ improved,
✗ worsened, - no claim)? Direction only. **Verdict** = is the change statistically supported
(bar: the 10M paired t-CI excludes zero in the better direction AND the 1M contrast is
directionally consistent)? In the docx the 1M and 10M tables are shaded to this map: green =
SUPPORTED, pink = not supported, unshaded = descriptive (no claim).

| Metric | Improved | Contrast | 10M: mean [95% CI], sign | Verdict |
|---|---|---|---|---|
| **Tier 1 training** |  |  |  |  |
| ev_spearman | ✔ | MMP (C2-C1) | +0.0443 [+0.0240, +0.0646] 10/10 | ✅ SUPPORTED |
| ev_ratio (scored as bias abs(ratio-1)) | ✔ | MMP (C2-C1) | -0.3609 [-0.5093, -0.2125] 0/10 | ✅ SUPPORTED |
| auc_click | ✔ | MMP (C2-C1) | +0.0122 [+0.0115, +0.0129] 10/10 | ✅ SUPPORTED |
| auc_install | ✔ | MMP (C2-C1) | +0.0275 [+0.0245, +0.0306] 10/10 | ✅ SUPPORTED |
| auc_payer | ✔ | MMP (C2-C1) | +0.0979 [+0.0796, +0.1162] 10/10 | ✅ SUPPORTED |
| rmse_spend, E(spend\|payer) ($) | ✔ | MMP (C2-C1) | -23.6 [-73.8, +26.5] 3/10 | ❌ not supported |
| **Tier 2 training, AFT head (price model)** |  |  |  |  |
| auc_win (AFT) | ✔ | SSP (C3-C1) | +0.0128 [+0.0051, +0.0204] 10/10 | ✅ SUPPORTED |
| ece_win (AFT) | ✗ | SSP (C3-C1) | +0.0019 [-0.0087, +0.0124] 6/10 | ❌ not supported |
| mce_win (AFT) | ✔ | SSP (C3-C1) | -0.024 [-0.058, +0.011] 4/10 | ❌ not supported |
| price_rmse_all | ✔ | SSP (C3-C1) | -0.49 [-0.71, -0.26] 0/10 | ✅ SUPPORTED |
| price_rmse_lost | ✔ | SSP (C3-C1) | -0.56 [-0.83, -0.29] 0/10 | ✅ SUPPORTED |
| price_crps_all | ✔ | SSP (C3-C1) | -0.019 [-0.032, -0.007] 1/10 | ✅ SUPPORTED |
| price_crps_lost | ✔ | SSP (C3-C1) | -0.007 [-0.019, +0.005] 4/10 | ❌ not supported |
| **Tier 2 training, CLF head (win classifier)** |  |  |  |  |
| auc_win (CLF) | ✔ | SSP (C3-C1) | +0.0137 [+0.0055, +0.0220] 10/10 | ✅ SUPPORTED |
| logloss_win (CLF) | ✔ | SSP (C3-C1) | -0.0145 [-0.0230, -0.0059] 0/10 | ✅ SUPPORTED |
| **Tier 1+2, Profit(bid) optimizer, AFT bidder** |  |  |  |  |
| overpay CPM ($) | ✔ | SSP (C3-C1) | -0.02 [-0.05, +0.01] 2/10 | ❌ not supported |
| profit CPM ($) | - | - | - | descriptive (no claim) |
| profit, total ($) | ✔ | MMP (C2-C1) | +4,015 [+1,402, +6,627] 10/10 | ✅ SUPPORTED |
| ROAS | - | - | - | descriptive (no claim) |
| n_won | - | - | - | descriptive (no claim) |
| **Tier 1+2, Profit(bid) optimizer, CLF bidder** |  |  |  |  |
| overpay CPM ($) | ✗ | SSP (C3-C1) | +0.01 [-0.01, +0.03] 6/10 | ❌ not supported |
| profit CPM ($) | - | - | - | descriptive (no claim) |
| profit, total ($) | ✔ | MMP (C2-C1) | +1,829 [+245, +3,413] 10/10 | ✅ SUPPORTED |
| ROAS | - | - | - | descriptive (no claim) |
| n_won | - | - | - | descriptive (no claim) |

**Reading notes.** (1) A ✔ with "not supported" (rmse_spend, mce_win, price_crps_lost, AFT-bidder
overpay) means the direction was favourable but inside seed noise; the two ✗ rows (ece_win,
CLF-bidder overpay) moved the wrong way, also inside noise - neither harm nor benefit is claimed.
(2) ev_ratio is scored as bias magnitude abs(ratio-1): MMP shrinks the bias 0.476 → 0.111, which is
why the improvement appears as a negative contrast (and a ✔). Sign counts count positive per-seed
differences, so 0/10 on a falling error = improved in all ten seeds. (3) Each row carries its
CLAIMED contrast; the profit rows carry the MMP claim, and the SSP-side profit contrasts (not
shown) are null: AFT +120 [-43, +283] 7/10, CLF +71 [-113, +255] 4/10 - together with the overpay
rows, that is the SSP economics null. (4) 1M replication: every SUPPORTED row is directionally
consistent at 1M (most also significant); MMP profit at 1M passes a sign test (9/10 CLF) but its
t-CI spans zero (whale-tail variance), so economics are quoted at 10M. Rows significant at 1M only:
mce_win and overpay CPM (AFT bidder) - the latter is the residue of the overpay artifact. Full 1M
CIs regenerate from docs/results/v10_anchor_s*.json. (5) price_crps_lost narrowly fails at 10M
while all-rows CRPS and both RMSE slices pass: quote the all-rows CRPS. (6) The SUPPORTED set is
exactly the paper story: MMP = value ranking + bias correction + profit; SSP = win discrimination +
price recovery; nothing else.

---

## Marginal data-layer contrasts (derived from the seed means above)

Column reading: C2-C1 = MMP value on bare DSP; C4-C3 = MMP value given SSP; C3-C1 = SSP value on
bare DSP; C4-C2 = SSP value given MMP; C4-C1 = both layers combined. Comparing col 1 vs 2 (and 3
vs 4) is an interaction check; on Tier-1 and Tier-2 rows the pairs are identical (and the off-axis
pair is 0) by the censoring design, so the interaction question is meaningful only on the Tier-1+2
economics rows, where the bidder composes both tiers.

**Caveats.** These cells are differences of the rounded seed means above, so they can differ in the
last digit from the paired headline contrasts below (e.g. auc_win CLF +0.0138 here vs the paired
+0.0137). They carry NO confidence intervals: on the economics rows the small SSP-side cells
(C3-C1, C4-C2) are inside seed noise (the paired SSP profit CI spans 0; see the headline table),
so sign flips there must not be over-read. The CI-bearing headline contrasts below remain the
citable numbers.

### 1M (n=10)

| Metric | C2-C1 | C4-C3 | C3-C1 | C4-C2 | C4-C1 |
|---|---|---|---|---|---|
| **Tier 1 training** |  |  |  |  |  |
| ev_spearman | +0.107 | +0.107 | 0 | 0 | +0.107 |
| ev_ratio | +0.335 | +0.335 | 0 | 0 | +0.335 |
| auc_click | +0.0303 | +0.0303 | 0 | 0 | +0.0303 |
| auc_install | +0.0508 | +0.0508 | 0 | 0 | +0.0508 |
| auc_payer | +0.0578 | +0.0578 | 0 | 0 | +0.0578 |
| **Tier 2 training, AFT head (price model)** |  |  |  |  |  |
| auc_win (AFT) | 0 | 0 | +0.0175 | +0.0175 | +0.0175 |
| ece_win (AFT) | 0 | 0 | +0.0037 | +0.0037 | +0.0037 |
| mce_win (AFT) | 0 | 0 | -0.071 | -0.071 | -0.071 |
| price_rmse_all | 0 | 0 | -1.3 | -1.3 | -1.3 |
| price_rmse_lost | 0 | 0 | -1.4 | -1.4 | -1.4 |
| price_crps_all | 0 | 0 | -0.037 | -0.037 | -0.037 |
| price_crps_lost | 0 | 0 | -0.017 | -0.017 | -0.017 |
| **Tier 2 training, CLF head (win classifier)** |  |  |  |  |  |
| auc_win (CLF) | 0 | 0 | +0.0136 | +0.0136 | +0.0136 |
| logloss_win (CLF) | 0 | 0 | -0.0136 | -0.0136 | -0.0136 |
| **Tier 1+2, Profit(bid) optimizer, AFT bidder** |  |  |  |  |  |
| overpay CPM ($) | +1.09 | +1.02 | -0.16 | -0.23 | +0.86 |
| profit CPM ($) | +5.55 | +8.19 | -0.74 | +1.90 | +7.45 |
| profit, total ($) | +279 | +474 | -68 | +127 | +406 |
| ROAS | +0.73 | +1.23 | -0.07 | +0.43 | +1.16 |
| n_won | +4,195 | +4,068 | -1,344 | -1,471 | +2,724 |
| **Tier 1+2, Profit(bid) optimizer, CLF bidder** |  |  |  |  |  |
| overpay CPM ($) | +0.12 | +0.13 | 0.00 | +0.01 | +0.13 |
| profit CPM ($) | +8.75 | +8.24 | +0.39 | -0.12 | +8.63 |
| profit, total ($) | +411 | +385 | +34 | +8 | +419 |
| ROAS | +3.06 | +2.87 | +0.11 | -0.08 | +2.98 |
| n_won | +1,376 | +1,419 | +789 | +832 | +2,208 |

### 10M (n=10)

| Metric | C2-C1 | C4-C3 | C3-C1 | C4-C2 | C4-C1 |
|---|---|---|---|---|---|
| **Tier 1 training** |  |  |  |  |  |
| ev_spearman | +0.044 | +0.044 | 0 | 0 | +0.044 |
| ev_ratio | +0.365 | +0.365 | 0 | 0 | +0.365 |
| auc_click | +0.0122 | +0.0122 | 0 | 0 | +0.0122 |
| auc_install | +0.0275 | +0.0275 | 0 | 0 | +0.0275 |
| auc_payer | +0.0980 | +0.0980 | 0 | 0 | +0.0980 |
| **Tier 2 training, AFT head (price model)** |  |  |  |  |  |
| auc_win (AFT) | 0 | 0 | +0.0127 | +0.0127 | +0.0127 |
| ece_win (AFT) | 0 | 0 | +0.0019 | +0.0019 | +0.0019 |
| mce_win (AFT) | 0 | 0 | -0.024 | -0.024 | -0.024 |
| price_rmse_all | 0 | 0 | -0.49 | -0.49 | -0.49 |
| price_rmse_lost | 0 | 0 | -0.56 | -0.56 | -0.56 |
| price_crps_all | 0 | 0 | -0.020 | -0.020 | -0.020 |
| price_crps_lost | 0 | 0 | -0.007 | -0.007 | -0.007 |
| **Tier 2 training, CLF head (win classifier)** |  |  |  |  |  |
| auc_win (CLF) | 0 | 0 | +0.0138 | +0.0138 | +0.0138 |
| logloss_win (CLF) | 0 | 0 | -0.0145 | -0.0145 | -0.0145 |
| **Tier 1+2, Profit(bid) optimizer, AFT bidder** |  |  |  |  |  |
| overpay CPM ($) | +1.62 | +1.66 | -0.02 | +0.02 | +1.64 |
| profit CPM ($) | +5.21 | +5.11 | +0.39 | +0.29 | +5.50 |
| profit, total ($) | +4,015 | +3,954 | +121 | +60 | +4,075 |
| ROAS | -0.66 | -0.75 | +0.09 | 0.00 | -0.66 |
| n_won | +69,744 | +70,259 | -5,578 | -5,063 | +64,681 |
| **Tier 1+2, Profit(bid) optimizer, CLF bidder** |  |  |  |  |  |
| overpay CPM ($) | +0.27 | +0.26 | +0.01 | 0.00 | +0.27 |
| profit CPM ($) | +3.59 | +2.65 | +0.02 | -0.92 | +2.67 |
| profit, total ($) | +1,829 | +1,419 | +71 | -339 | +1,490 |
| ROAS | +0.66 | +0.39 | -0.03 | -0.30 | +0.36 |
| n_won | +34,843 | +34,071 | +6,432 | +5,660 | +40,503 |

*Interaction reading (10M, CLF bidder): MMP is worth +1,829 profit alone but +1,419 on top of SSP;
SSP is +71 alone and -339 on top of MMP - mildly sub-additive, and the SSP-side cells are
statistically null either way. The economics are carried by MMP at both scales.*

---

## Headline contrasts (mean [95% t-CI], sign count)

| Contrast | 1M, n=10 | 10M, n=10 | Verdict |
|---|---|---|---|
| **SSP prediction** — auc_win CLF, C3−C1 | +0.0137 [0.0053, 0.0220], 10/10 | +0.0137 [0.0055, 0.0220], 10/10 | **real, scale-stable** |
| SSP prediction — auc_win AFT, C3−C1 | +0.0175, 10/10 | +0.0128 [0.0051, 0.0204], 10/10 | real (both heads agree) |
| SSP prediction — price CRPS gain, all rows | +5.3% | +2.9% [1.2, 4.7], 9/10 | real |
| SSP prediction — price RMSE gain, all rows | +4.8% | +2.0% [1.2, 2.8], 10/10 | real, modest |
| **SSP economics** — overpay reduction (%, scale-free) | +6.5% (AFT bidder, 10/10) **but −0.6% [−1.7, +0.4] (CLF bidder, 2/10)** | −0.9% [−2.5, +0.7] (CLF bidder, 1/10) | **AFT artifact — not demonstrated** |
| SSP economics — CLF-bidder profit, C3 vs C1 | CI spans 0 | +1.6% [−3.3, +6.4], 4/10 | not demonstrated |
| **MMP economics** — CLF-bidder profit, C2 vs C1 | ≈ 2.1× (367 → 778) | +39% (4,746 → 6,575) | **robust** |
| MMP value-bias — ev_ratio C1 → C2 | 0.63 → 0.97 | 0.52 → 0.89 | durable (selection bias, uncorrectable by scale) |

**Context — the decisive test that precedes these numbers (ρ=1, 100K/1M):** the v10 SSP contrast
**survives the well-specified classifier** (+0.021 vs v8's +0.0002) — v8's null was DGP-conditional
(conditional-IPV market ⇒ single price is sufficient, Athey–Haile), not a property of SSP data.
The anchored point then reads off the honest niche-market estimate. Full arc: `v10_Results_5Jul2026.md`.


---

## Reading the tables

**Design identities (hold at every scale, every seed):** MMP touches only the funnel/value axis →
`C3 = C1`, `C4 = C2` on ev_spearman/ev_ratio/auc_click/auc_install/auc_payer. SSP touches only the
price/win axis → `C1 = C2`, `C3 = C4` on auc_win/logloss/ECE/RMSE/CRPS. So **MMP value = C2−C1 ≡
C4−C3**; **SSP value = C3−C1 ≡ C4−C2**. The repeated cells in the tables are these identities, not
errors.

**AFT vs CLF (why two of everything):** the v8-era shrink test showed the AFT head under-fits both
conditions equally; every v10 ranking claim is therefore quoted from BOTH heads (they agree), and
every economic claim from BOTH bidders. The two-bidder check is what exposed the overpayment
"reduction" as an AFT-curve artifact — under the classifier-curve bidder it vanishes. Quote SSP
economics only from the CLF bidder.

**Why SSP model-gains don't convert to profit:** the bidder is gated on the **value side** — EV
ranking (ev_spearman 0.44–0.64 vs oracle 1.0) determines which impressions are worth buying; a
better win-curve cannot rescue mispriced value. This is the same gate the 12-Jun oracle-features
diagnostic identified. MMP improves exactly that side (ev_ratio, payer-head AUC) and its profit
advantage is robust under both bidders at both scales.

**ROAS is seed-unstable — report profit, not ROAS.** The n=5 apparent inversion (CLF C2 ROAS 4.41 <
C1 4.65) did **not** survive n=10 (C2 5.31 > C1 4.65): revenue is whale-dominated, so ROAS swings widely
across seeds. Profit is the seed-stable economic metric — the MMP profit gap is robust at both scales
(now **+39%** CLF at 10M, n=10) — while ROAS is reported for completeness only.

**Oracle gap:** all conditions sit far below the oracle on the value axis by design — the latent
slice (LU/LA/LC + rival privateness) is the ablation's headroom, not a defect to be closed.

---

**Bottom line:** at the externally anchored operating point of a market whose competing bids now
arise from named rival mechanisms rather than iid noise, the ablation's SSP verdict is **axis-split
and scale-stable: SSP data improves the model (win-AUC ≈+0.014 at both scales, CRPS +3–5%, all seeds
positive at both scales) but does not demonstrably convert to money (overpay artifact; profit CI
spans zero), while MMP data makes the money (2.1× @1M → +39% @10M classifier-bidder profit; ev_ratio
0.63→0.97 / 0.52→0.89).** Reason 3 of the proof chain is resolved axis-specifically: falsified for
prediction, sustained for economics in this niche-market configuration. (V10 = BUILT + EVALUATED,
branch `t9-v10-rival` not merged; v7 stays operative pending sign-off.)
