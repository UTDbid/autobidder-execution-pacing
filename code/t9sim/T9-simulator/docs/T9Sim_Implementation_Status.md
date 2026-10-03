# T9Sim - Implementation Status

*Companion to the Specification. This file holds the edge wiring status, the declared code-vs-spec divergences, the build status and the audit trail, split out of the Specification so it can stay focused on defining the variables and their laws. Section numbers match the Specification.*

---

## 2.6 Edge inventory and wiring status

*(The standalone `Edge_Register` was folded into this section on 23 Jul 2026 and archived; mentions of it in the audit trail below are historical.)*

**At a glance.** ✓ = edge exists at this level · blank = not present / not applicable · **✗ (red)** = required but missing (a Schema edge needs Code; a dependency edge needs CPT + Code). Dep = member of the flag-gated dependency layer. CPT = a dependency-layer parameter table (`bn_cpts.yaml` or the `bn:` block of `benchmarks.yaml`); the column applies to dependency edges only. Status 22 Jul 2026, branch `t9-v11-mbw`. Parent sets corrected against code 22 Jul; rival-pool parameters now declared in `benchmarks.yaml` (`bn.rival_pool`, `bn.edges.rival_pool`, `auction.rho`), so E1-E4 carry CPT ✓. Defect detail is in 2.7 and Appendix A below.

| Edge (from → to) | Schema | Dep | CPT | Code |
|---|:---:|:---:|:---:|:---:|
| **Archetype tilts** |  |  |  |  |
| LU1 → D2 hour_of_day | ✓ | ✓ | ✓ | ✓ |
| LU1 → A3 os | ✓ | ✓ | ✓ | ✗ |
| LU1 → A5 device_type | ✓ | ✓ | ✓ | ✗ |
| LU1 → D3 day_of_week | ✓ | ✓ | ✓ | ✗ |
| **v7 dependency edges** |  |  |  |  |
| #1 (LA2 × LU1) → B1 app | ✓ | ✓ | ✓ | ✓ |
| #2 LU4, LU5 → C campaign (via LC2-sorted pmf) | ✓ | ✓ | ✓ | ✓ |
| #3 A3 → G2 spend | ✓ | ✓ | ✓ | ✓ |
| #4 D2, D3 → G1 payer | ✓ | ✓ | ✓ | ✓ |
| **v10 rival pool** |  |  |  |  |
| E1 z → b_k | ✓ | ✓ | ✓ | ✓ |
| E2 day → pacing → b_k | ✓ | ✓ | ✓ | ✓ |
| E3 ad_exchange → A_i | ✓ | ✓ | ✓ | ✓ |
| E4 segment → R → b_k | ✓ | ✓ | ✓ | ✓ |
| E5 b_k → LU7; A_i → H9 | ✓ | ✓ |  | ✓ |
| **Base structure (not flag-gated; outside the dependency layer)** |  |  |  |  |
| LU1 → LU2..LU6 | ✓ |  |  | ✓ |
| B2 app_category → LA2 | ✓ |  |  | ✓ |
| A3 → A4 os_version | ✓ |  |  | ✓ |
| A1 → A2 city | ✓ |  |  | ✓ |
| D2, D3, week → D1 timestamp | ✓ |  |  | ✓ |
| LU6, C4 → r_genre | ✓ |  |  | ✓ |
| LU2, LA1, LC1, v_slot, r_genre → E1 click | ✓ |  |  | ✓ |
| LU3, LA1, B2 (ease), r_genre → F1 install | ✓ |  |  | ✓ |
| LU4, r_genre (+ D2, D3 when #4) → G1 payer | ✓ |  |  | ✓ |
| LU5, LC2, B2 (mu_cat) (+ A3 when #3) → G2 spend | ✓ |  |  | ✓ |
| funnel heads → EV | ✓ |  |  | ✓ |
| EV → z | ✓ |  |  | ✓ |
| B6 → H1 floor | ✓ |  |  | ✓ |
| B2, C2, B4/B5, B6 → H2 bid | ✓ |  |  | ✓ |
| H2, LU7, H1 → H3 won | ✓ |  |  | ✓ |
| H3, H2, LU7, H1 → H4 clearing | ✓ |  |  | ✓ |

*Excluded: retired edges (LU1 → A4 direct, LU1 → A1 region, B3), the dormant v8 layer, and the abandoned arms B1/B2 (dropped 24 Jul 2026; they added no edges, and their flags remain in code but are inert and unreported).*

Families: base structure (always-on) / archetype tilts (declared dependency-layer tilts) / dependency edges #1-#5 (flag-gated) / gated group #6 `rival_pool` / dormant / retired. The per-family tables below carry the code evidence behind each tick. rows marked AMEND are register parent-set under-listings logged in Appendix M13 (seven register rows; CODE is the authority on wiring). Parent sets are defined in 1.6; this inventory is the definitive list for wiring.

### 2.6.1 Base structure (always-on)

| # | Edge | Wired | Register agreement |
|---|---|---|---|
| B-1 | LU1 -> LU2..LU6 | yes | agrees |
| B-2 | B2 app_category -> LA2 | yes | agrees; note: the LA2 NODE is drawn only when #1 is ON (existence gated, edge form fixed) |
| B-3 | A3 -> A4 os_version | yes | agrees |
| B-4 | A1 -> A2 city | yes | agrees |
| B-5 | D2, D3, week (D4) -> D1 timestamp | yes | agrees (the register's "week" is D4, M2) |
| B-6 | LU6, C4 -> r_genre | yes | agrees |
| B-7 | LU2, LA1, LC1 -> E1 click | yes | AMEND (M13): v_slot and m_stage also enter via p_click |
| B-8 | LU3, LA1 -> F1 install | yes | AMEND (M13): ease(B2) and m_stage also enter via p_install |
| B-9 | LU4 -> G1 payer | yes | AMEND (M13): m_stage always, t_pay when #4, via p_payer |
| B-10 | LU5, LC2 -> G2 spend | yes | AMEND (M13): mu_cat(B2) always; plat(A3) when #3 |
| B-11 | funnel heads -> EV (ev_truth) | yes | agrees |
| B-12 | EV -> z | yes | agrees |
| B-13 | B6 -> H1 floor | yes | agrees |
| B-14 | B2, C2 -> H2 bid | yes | AMEND (M13): full parents v_slot(B4/B5/B6), ease(B2), eltv_b2(B2), k_cpa(C2), plus scalars k_global/shade/exploration |
| B-15 | H2, LU7, H1 -> H3 won | yes | agrees |
| B-16 | H3, H2, sold_lost, LU7 -> H4 clearing | yes | AMEND (M13): the register row "H3, LU7 -> H4" under-lists H2 (the won-row value); H1 enters through the sold_lost mediator |

### 2.6.2 Base structure: internal edges

| # | Edge | Wired |
|---|---|---|
| B-17 | B6 -> _size -> B4, B5 | yes |
| B-18 | B6 -> base_e | yes |
| B-19 | E1, D1 -> E2 click_timestamp | yes |
| B-20 | F1, E2 -> F2 install_timestamp | yes |
| B-21 | G2 -> G3, G4 | yes |
| B-22 | H3, LU7, H1 -> sold_lost (mediates H4) | yes |
| B-23 | campaign pool structure: C3 -> C1 -> C2; C2 tier -> sample_weight | yes |
| B-24 | r_genre -> m_stage (m_click, m_install, m_pay) | yes |
| B-25 | E1 -> F1 (click gates install) | yes |
| B-26 | F1 -> G1 (install gates payer) | yes |
| B-27 | G1 -> G2 (payer gates spend; 0 for non-payers) | yes |

### 2.6.3 Archetype tilts (declared)

| # | Edge | Wired | Register agreement |
|---|---|---|---|
| T-1 | LU1 -> D2 hour_of_day | YES (CPT `resolved.hour_of_day`, consumed auctions.py:127). This row IS dependency edge #5 (flag `hour`). | agrees (all four register cells ✓) |
| T-2 | LU1 -> A3 os | NO (CPT built `bn_cpts.yaml`:47, never read; flat draw user_profiles.py:50) | agrees (Code ✗); Known_Defects 1 |
| T-3 | LU1 -> A5 device_type | NO (CPT built :70, never read) | agrees (Code ✗) |
| T-4 | LU1 -> D3 day_of_week | NO (CPT built :93, never read even with #5 ON, M18) | agrees (Code ✗) |

### 2.6.4 Dependency edges #1-#5

Construction vocabulary: #1, #2 and #5 are IPF-raked, marginal-preserving constructions; #3 and #4 are mean-normalised multipliers (#3 mean-preserving by recentring, #4 raked to mean 1 over the calibration joint).

| # | Edge | Flag | Construction | Wired | Register agreement |
|---|---|---|---|---|---|
| #1 | (LA2 x LU1) -> B1 app pairing | `pairing` | marginal-preserving IPF joint over (archetype, app); acts symmetrically through the always-visible app_id | yes (auctions.py:144-165, 279-293) | agrees (CPT ✓: pairing_strength_s, k_audience, audience_centroids) |
| #2 | LU4, LU5 -> campaign exposure | `exposure` | value-keyed per-bin IPF-raked pmf (decile discretisation of the continuous tilt, 2.7); no footprint in any always-visible column | yes (auctions.py:167-197, 295-303) | agrees on status; parent note per M13 (campaign-side LC2, sample_weight) |
| #3 | A3 -> G2/e_ltv spend | `os_spend` | plat multiplier + mean-preserving mu_cat recentre by -ln E[plat]; also shifts the os split constant | yes (auctions.py:111-114, 387-394) | agrees |
| #4 | D2, D3 -> G1/p_payer | `payer_timing` | t_pay raked to mean 1 over the calibration (hour, dow) joint | yes (auctions.py:115-121, 380-384) | agrees |
| #5 | LU1 -> D2 (= T-1; hour only, M18) | `hour` | von Mises CPT, IPF-raked to the iPinYou hour marginal | yes | agrees |

Time-structure claim (scoped per Appendix M11): "Time has no stochastic internal edges: D2 and D3 are independent draws, and D1 is deterministically assembled from them plus the week (the D2, D3, week → D1 edge), so there is no D1 → D2 or D1 → D3 direction and no D2-D3 edge." This holds on the #5-ON configuration; on the all-OFF path (hour, dow) are drawn jointly from the iPinYou hour x dow pmf, which couples them.

### 2.6.5 Group #6 `rival_pool` (laws in 2.5)

| # | Edge | Wired | Register agreement |
|---|---|---|---|
| E1 | z -> b_k (via LR1 w_k) | yes (rival_pool.py:113) | agrees (Code ✓, CPT ✗ = no config-declared parameters; Known_Defects 2, M5) |
| E2 | day -> LR3 pacing -> b_k AND participation gate (gate not rho-scaled) | yes (rival_pool.py:107-113) | agrees (Code ✓, CPT ✗) |
| E3 | B3 ad_exchange -> LR4 -> A_i composition | yes (rival_pool.py:109) | agrees (Code ✓, CPT ✗) |
| E4 | observable segment (A3 x A5) -> LR2 R -> b_k | yes (rival_pool.py:113) | agrees (Code ✓, CPT ✗) |
| E5 | {b_k} -> LU7; A_i -> H9 | yes (rival_pool.py:119-120) | agrees (Code ✓; CPT cell blank: deterministic max/count, no parameters) |

Pool-internal notes (no separate register rows): the deterministic role partition sets LR1's draw ranges, the LR4 k=0 override and the LR6 exemption (M1); LR5 noise and base_e enter every b_k; LR6/LR3/LR4 -> Z_k. Observability is the footprint rule (2.5).

### 2.6.6 Dormant (code present, default OFF)

| # | Entry | Status |
|---|---|---|
| V-1 | `competition`: N_t = 1 + Poisson, order-statistic gaming term, emits bid_density (the quantity lives on as H9 under group #6) | wired, OFF; numbering retired (M14) |
| V-2 | `app_quality_price` (M_price multiplier) | wired, OFF |
| V-3 | `exchange_price` | wired, OFF |
| V-4 | `app_price` | wired, OFF |
| V-5 | `time_competition` | wired, OFF |
| V-6 | `ev_market` switch (z input replaced by the genre-marginal _ev_lu7; auto-forced by any of the five price flags V-1 to V-5) | wired, OFF; under rival_pool ON the M_price product never reaches LU7 (M17) |
| V-7 | `rival_hidden` (+`rival_user_corr`) probe: persistent per-segment factor mixed into the OFF-path eps; emits rival_seg | wired, OFF; superseded by #6 rival_pool |
| V-8 | pipeline `--hist-clearing` (default OFF): attaches hist_clearing_dsp (all conditions; won-row H2 cell means) and hist_clearing_ssp (C3/C4; H4 cell means) (pipeline.py:159-202); declared in schema.py:30-33. Pipeline-derived features, never generator columns; the H6/H7 numbers they carried are retired | wired (pipeline), OFF |

---

## 2.7 Declared divergences

*(Split from the Specification's 2.7; the symbol key stays there.)*

- Funnel means are `clip(·, 0, 1)` (`auctions.py:370-376`), making each head a well-defined Bernoulli kernel; pre-clip products can exceed 1.
- `t_pay` has mean 1 over the calibration (hour, dow) joint (`auctions.py:120, 383`), so edge #4 preserves the payer mean only approximately: the `ν^pay` clip, the archetype-hour correlation with `q^pay`, and the realised (hour, dow) joint differing from the calibration joint under #5 each shift it slightly; backstopped by the validation gates. (The auctions.py:379 comment's "population" wording is loose; the rake divisor is computed over the calibration pmf.)
- `z = 0` on `EV <= 0` (`auctions.py:434-436`); inactive users force `q^pay = m_u = 0` (`user_profiles.py:79, 83`), so `EV` carries a point mass at 0 on ~30% of rows. RNG accounting: the inactive payer Beta is drawn then overwritten (the draw consumes the user stream); only LU5's LogNormal is genuinely skipped (M19).
- Under max-of-K rivals, `ρ` scales structured terms directly and is not variance-raked, so total log-bid variance grows with `ρ` (calibration is empirical, via `k_global`).
- `os` is not a parent of `mu_cat` (scalar re-centre only, `auctions.py:389`); it reaches LTV solely via `plat`.
- `make_cpts` builds dow/os/device tilts (the region tilt is computed then dropped, `make_cpts.py:201`, so it never reaches `bn_cpts.yaml`); the engine consumes only the hour CPT (`auctions.py:127, 339`), so those marginals are drawn untilted. Scheduled for wiring: see 2.3 and 4.1.
- `base_e` carries a degenerate-zero guard: `base_e = max(pay_shape·target, 0.01·target)` (`auctions.py:429-430`). Load-bearing: `log(base_e)` feeds every rival bid (`rival_pool.py:103`).
- Edge #2 exposure is a decile discretisation of the schema's continuous per-user tilt: users are grouped into value deciles of `g = z(log(q^pay·m))` and each decile shares one IPF-raked campaign pmf built from the bin-mean `g` (`auctions.py:183-197, 295-303`), so `P(campaign | user)` is a step function of `g` rather than the continuous `budget_c·exp(β_vo·g_u·z(log lc2_c))` weight. Campaign marginals and the monotone value-keying are preserved; `β_vo` is autocalibrated on the binned form.
- The always-on slot `k=0` is the first gaming-DSP archetype (`w_0 ~ U(0.35, 0.75)`), strongly value-loaded, not a value-neutral generalist (the `rival_pool.py:12, 112` "generalist" comments are mislabels, M4). Likewise the module docstring's `phi(LU_u)` is a mislabel for z of the focal row's log ev_truth (M16).
- D1 assembly direction: the schema states D2/D3 derived from D1; the build draws `hour`/`day_of_week`/`week` first and assembles `timestamp` from them plus a uniform within-hour second, giving the same joint law over (D1, D2, D3) at hour resolution.
- Rival-pool day indexing is CHUNK-RELATIVE: day_idx = (timestamp - min timestamp in chunk) // 86400 (`rival_pool.py:98-99`), not anchored to `window_start_utc`. Negligible at production chunk sizes (the chunk min is within seconds of window start); real at small n, where rows near day boundaries can shift one day and blur the E2 time footprint (M12; proposed Known_Defects addition).

### 2.7.1 Paper method section vs released code

*Declared 28 Jul 2026. The entries above are specification vs code. These three are different in kind: the paper's method section describes model machinery the released code does not implement. None of them affects the reported C1 to C4 contrasts, because every view is trained and scored identically, but a reader comparing the paper's Section 5 with `pipeline.py` will find them, so they are stated here rather than left to be discovered.*

- **Probability calibration is measured, not fitted.** Paper Section 5.4 says each view and prediction head is calibrated on a validation split using isotonic regression or Platt scaling. No calibrator is fitted anywhere in the package. The pipeline measures calibration only: expected and maximum calibration error over fifteen equal-width bins (`pipeline.py:559-571`) and a ten-decile reliability table on the win curve. All reported probabilities are therefore raw model output. Because the omission is identical across C1 to C4, the data-layer contrasts are unaffected; absolute probability levels are not calibrated.
- **The win classifier is not monotone-constrained.** The paper's introduction describes the Tier-2 model as fitted as a monotone win classifier. The XGBoost parameters in `train_tier2_clf` (`pipeline.py:607-625`) set no `monotone_constraints`, so the predicted win probability is not guaranteed to be non-decreasing in the bid. At serving time the same row is re-scored at each of the 48 grid bids (`BID_GRID`, `pipeline.py:73`; `pwin_clf`, `pipeline.py:498-507`), so a non-monotone region can in principle move the profit-maximising bid.
- **No ROAS-constrained bid rule exists.** Paper Section 5.3 also gives a ROAS-constrained variant that keeps only candidate bids whose predicted return on ad spend clears a target and abstains when none do. The code implements only the profit-maximising argmax over the bid grid (`_resolve_bidder`, `pipeline.py:529-556`). No reported result uses the constrained variant.

---

---

# Part IV - Status

## 4.1 Build status and reproducibility

**Wired vs declared.** Everything in the edge inventory (2.6) marked "yes" is wired in code; the three declared archetype tilts T-2/T-3/T-4 have CPTs but no consuming code (the released data carries no archetype signal in os/os_version/device_type/day_of_week). Per-edge authority is the edge inventory in 2.6; declared divergences are in 2.7 and the audit trail in Appendix A. The internal registers that once held this material were folded into this document (see the note in 2.6) and are not published separately; where the specification and those registers disagreed, the resolutions are Appendix M12 and M13.

**Where each dial lives** (Q1 resolved 22 Jul 2026, option A; Known_Defects 2). All three declarations are in `config/benchmarks.yaml`: the flag `bn.edges.rival_pool`, the parameter block `bn.rival_pool` (K = 8, n_gaming = 3, beta_R = 0.5, pacing_ar = 0.85, pacing_sigma = 0.30, read by `rival_pool.py:34` with the legacy top-level key as fallback), and the privateness dial `auction.rho` (`auctions.py:99`).

**Shipped defaults.** Flag false, rho 0.0, so the repository default is the all-OFF baseline and the paper configuration is obtained by setting the five dependency-edge flags, `rival_pool: true` and `rho: 0.8`.

**Verified at 100K** (`golden`, seed 90210): declaring the dials in config leaves the all-OFF and paper-configuration hashes byte-identical, and a config-only run (no Python overrides) reproduces the paper-configuration hash exactly.

**Calibrated constants** (in `config/calibrated.yaml`, merged over benchmarks): base_ctr 0.27141, base_ir 2.564963, base_payer 0.257853, LTV mu 0.481154, sigma 1.648169. Win-rate autocalibration: k_global solved on a 50,000-row warm-up so mean H3 hits the 0.30 target.

**Results pointer** (the one allowed): evaluation numbers for this schema are in `docs/v10_Training_Results.md`.

## 4.2 Tunable parameters

| Group | Params |
|---|---|
| Funnel base rates (calibration knobs, *auto-calibrated*) | `base_ctr`, `base_ir`, `base_payer`, LTV `μ/σ` |
| Match strengths (effect-size knobs, *inferred*) | `w_click`, `w_install`, `w_pay` |
| Auction core (effect-size knobs) | `γ` = 0.45, `sigma_g` = 0.35, `mu_x` = -0.45, `sigma_x` = 0.55 (all four OFF-path LU7 dials; the sigma_g/mu_x/sigma_x normals are consumed-and-discarded on the `rival_pool` path for stream alignment), `k_cpa`, `shade`, `σ_explore` |
| **Rival pool** | **K** (8) and **n_gaming** (3): the coded role partition is gaming vs non-gaming (Q2 resolved: 3 gaming + 5 non-gaming, no sub-labels); per-archetype `w_k`, `σ_k`; retargeting `β_R` + segment shifters `R[s,k]` (os×device); pacing `φ_p`, `σ_p` + flight windows `F_k(d)`; participation `π_k(e)`; **ρ** private-structure amplitude dial (anchored ρ*≈0.8; per-run argument, 4.1) |
| Latent spreads (effect-size) | `σ_app`, `σ_cre`, `σ_game`, interest (LU6) and app-audience (LA2) Dirichlet concentrations |
| Categories / mixes | `app_categories` (`B2`), `ad_genre` mix (`C4`), `ad_exchanges` (`B3`), `slot_format` mix (`B6`), `slot_sizes` (`B4`×`B5`), `advertiser_scale` |
| Dependency layer, edges #1-#5 (active) | the five `bn:` edge flags and their CPT parameters (2.6.4) |

## 4.3 Validation targets

Funnel metrics are computed on the **master**; all six gates are verified with `rival_pool` ON.

| Metric | Target |
|---|---|
| Population CTR (mean E1) | 2-5% |
| Click→install (F1\|E1) | 25-40% |
| Install→payer (G1\|F1) | 2-5% |
| Whale concentration | top 5% payers → ~55-65% of total G2 |
| Median payer spend | ~$6 |
| Auction win rate (mean H3) | 20-40% (calibrated-pinned at 30%) |

## 4.4 Version trail

v7 = funnel + dependency-edge core (inherited); v8 = superseded price-feature layer (origin of the retired H5-H8 numbers); v10 = v7 + the Rival-Pool Layer (this spec).

**Retired structure** (no code; kept here for the record): the direct LU1 -> A4 tilt (never coded); the LU1 -> A1 region tilt (computed then dropped, `make_cpts.py:201-204`); and the H5-H8 column numbering, whose quantities live on as H9 (rival count) and as the flag-gated `hist_clearing` pipeline features.

---

---

# Appendix A - Resolved mismatches (audit trail)

Every disagreement between the sources (schema v10, generator formalisation, code, registers), with the resolution this document adopts and the arbiter that decided it. A verifier pass (22 Jul, code-checked) corrected the working inventory before this merge; its corrections are folded into the rows below and listed at the end.

| # | Mismatch | Sources in conflict | Resolution (adopted above) | Arbiter |
|---|---|---|---|---|
| M1 | LR family under-enumerated: schema doc codes LR1-LR4 only; sigma_k sits in its notation table with NO LR code; flights are bundled into LR3. The code inventory additionally proposed LR7 for the role label. | Schema v10 vs code (6 frozen per-rival draw families, rival_pool.py:52-82) vs formalisation (LR1-LR6) | Adopt LR1-LR6 exactly, gap-free (2.5): LR1 w_k, LR2 R, LR3 pace (pacing ONLY), LR4 pi_ke (participation ONLY), LR5 sigma_k, LR6 flight. NO LR7: the role label (gaming iff k < n_gaming; k=0 always-on) is a deterministic index partition, not a sampled ground-truth parameter. NO per-rival base multiplier exists (log base_e is the common anchor). After LR1-LR6 the family is exhaustive: rival_pool.py freezes nothing else (verified: exactly six draw families, no seventh). | CODE (existence) + FORMALISATION (form and codes). Matches Known_Defects 3b. |
| M2 | D4 week missing: schema defines D1-D3 but its own D1 rule and the Edge_Register's "D2, D3, week -> D1" row name a week draw. | Schema v10 vs code (auctions.py:344) + formalisation (D4) | Adopt D4 week, U{0..n_weeks-1}, generator-internal, never a column. | CODE + FORMALISATION. Matches Known_Defects 3b. |
| M3 | Archetype sub-mix "2 e-commerce retargeters / 1 brand buyer / 2 generalists" and the "archetype mix" tunable have no coded counterpart: the pool distinguishes only gaming (k < 3) vs non-gaming; retargeting applies to ALL K rivals; only K and n_gaming exist, as hard-coded defaults. | Schema v10 (market setting, tunables row) vs code (rival_pool.py:35-63) | Present the sub-mix as narrative interpretation of the 3 gaming + 5 non-gaming split, explicitly marked not coded (2.1); tunables row lists K and n_gaming (4.2). Whether to delete the sub-labels outright is open question Q2. | CODE (existence). |
| M4 | k=0 identity: code comments call the always-on slot "generalist" (rival_pool.py:11-13,112), but the code makes k=0 the FIRST GAMING archetype, strongly value-loaded (w_0 ~ U(0.35, 0.75)) with pi = 1 and flight exemption. | Code comments vs code behaviour + schema's declared build divergence + formalisation divergence note | The schema/formalisation account is correct and carries as the declared divergence (2.5, 2.7); the code comments are wrong and should be fixed (comments are not behaviour). | CODE behaviour decides; SCHEMA disclosure stands. |
| M5 | Reproducibility gap: schema calls rho a "fixed dial ... anchored rho* ~ 0.8", but `benchmarks.yaml` has no rho entry (auctions.py:99 defaults 0.0 = the SSP-null regime; 0.8 exists only as a runtime override in the 10M worker script), no `rival_pool:` params block, and no `rival_pool` flag in the `bn:` edges block; K/n_gaming/beta_R/pacing constants are code defaults. | Schema v10 vs `config/benchmarks.yaml` + code defaults | 4.1 states, per dial, where the value actually lives and that config declaration is pending (Known_Defects 2). The registers' records (CPT ✗ on E1-E4) are correct. RESOLVED 22 Jul 2026 (option A): declared in `benchmarks.yaml`, byte-identity verified (4.1). | CODE + both registers. |
| M6 | Stale formulas: the v10 outcome/estimand tables print BASE forms, omitting the active edge factors: t_pay(D2, D3) on G1/p_payer (#4) and plat(A3) + the mean-preserving mu_cat recentring on G2/e_ltv (#3). Sub-items: the base_e degenerate-zero guard (max with 0.01 x format target, auctions.py:430) was undocumented in the schema; the code comment near the t_pay rake says "population" where the divisor is the calibration (hour, dow) pmf, a distinction that is load-bearing because payer-mean preservation is only approximate precisely when the realised joint under #5 differs from the calibration joint. | Schema v10 tables vs code + formalisation (both carry the with-edge forms and the guard) | This document prints the with-edge formulas with their flag conditions (1.6, 2.4), documents the guard, and uses "calibration (hour, dow) joint" throughout. | CODE (wiring) + FORMALISATION (form). |
| M7 | Phantom archetype tilts: the schema declares LU1 -> A3, A5, D2, D3 (A4 via A3); only the D2 hour tilt is consumed (auctions.py:127). The A3/A5/D3 CPTs exist in `bn_cpts.yaml` (lines 47, 70, 93) but nothing reads them, so released data carries no archetype signal in os/os_version/device_type/day_of_week. | Declared design (and several diagrams) vs code (user_profiles.py:50-52) | Keep the schema's accurate disclosure; only LU1 -> D2 is wired (2.6.3); diagrams are the offenders, not the schema text. Wiring status per the registers. | CODE; Edge_Register ✗ rows and Known_Defects 1 agree. |
| M8 | rho symbol collision: v7's edge #3 platform multiplier is code `self._rho` (= `ios_ltv_multiplier`), while the v10 privateness dial is `self.rho_rival`; one document space, two rhos. | Schema v7/v10 notation vs itself | rho is reserved for the privateness dial; the edge #3 multiplier is written plat in prose (the formalisation's symbol); code identifiers stay verbatim as literals. | FORMALISATION + house naming rules. |
| M9 | Naming collision not covered by the naming note: rival graph-edge labels E1-E5 vs click columns E1/E2. | Schema naming note vs the schema's own E1-E5 usage | Naming note extended (1.3): E1-E5 in the rival-market material are edge labels, never the click columns. | SCHEMA (naming semantics are its jurisdiction). |
| M10 | "Users and apps are drawn uniformly" is only marginally true with #1 ON: the IPF pairing tilts the (archetype x app) JOINT while preserving both marginals (uniform app popularity as column sums). | Schema pool-semantics block vs code (auctions.py:144-165) | Scoping note attached to the carried block (2.1): uniform MARGINALS; edge #1 tilts the joint, marginals preserved by IPF. | CODE. |
| M11 | "D2 and D3 are independent draws" holds only on the dependency-layer-ON path (#5: hour from the archetype CPT, dow from the marginal); the all-OFF path draws (hour, dow) jointly from the iPinYou hour x dow pmf, which couples them. | Schema time-structure claim vs code (auctions.py:341-343) | Claim scoped to the #5-ON configuration (2.6.4). | CODE. |
| M12 | Rival-pool day indexing is CHUNK-RELATIVE: day_idx = (timestamp - min timestamp in chunk) // 86400 (rival_pool.py:98-99), not anchored to window_start_utc. Negligible at production chunk sizes; real at small n. | Code vs both registers (in neither) | Documented in 2.7; PROPOSED as a new Known_Defects entry (low severity). Added to Known_Defects as section 3c (22 Jul). | CODE (new fact). |
| M13 | Edge_Register parent under-listing, seven rows: "B2, C2 -> H2 bid" omits v_slot (a B4/B5/B6 function), ease(B2) and k_global (auctions.py:516-520); "LU2, LA1, LC1 -> E1 click" omits v_slot and m_stage (via p_click); "LU3, LA1 -> F1 install" omits ease(B2) and m_stage (via p_install); "LU4 -> G1 payer" omits m_stage and, under #4, t_pay (via p_payer); "LU5, LC2 -> G2 spend" omits mu_cat(B2) and, under #3, plat(A3); "H3, LU7 -> H4 clearing" omits H2 (the won-row value) and the sold_lost mediator; "#2 LU4, LU5 -> C campaign" omits the campaign-side LC2 and sample_weight inputs to the per-bin pmf (auctions.py:175-197). | Edge_Register vs code | The edge inventory (2.6) and master table (1.6) carry the full code-true parent sets; the seven register rows should be amended. This is the one CATEGORY where this document disagrees with a register, and CODE is the authority on wiring. | CODE. |
| M14 | Gated-group "#6" ambiguity: the formalisation labels the rival market "#6 rival_pool"; v8's flag #6 was `competition`; the config gives rival_pool no number. | Formalisation vs v8 numbering (and one diagram script per Known_Defects 3) | This document uses "#6 rival_pool" and states the v8 numbering (#6-#10) is retired along with the H5-H8 column numbers; the dormant v8 flags remain in code under their config names (2.6.7). | FORMALISATION + Known_Defects 3. |
| M15 | The schema's map-to-parquet block and "non-column boxes" label hard-code "LR1-LR4"; adopting LR5/LR6/D4 requires those label edits, with all four counts (41/47/54/55) unchanged (the new codes are generator-internal, never columns). | Schema v10 labels vs the M1/M2 adoption | Labels updated to LR1-LR6 throughout (1.4, 1.7); counts unchanged. | Follows M1/M2. |
| M16 | rival_pool docstring says rival bids read "phi(LU_u)" (user-only value); in fact z is the z-score of the FOCAL row's log ev_truth (user x app x campaign x slot, including LC1/LC2 and genre relevance), since `rival_pool` is not in the `ev_market` auto-force list. | Code comment vs code behaviour + formalisation | The formalisation's form (z of log ev_truth) matches the code and carries (2.5); the docstring is a comment-level mislabel to fix. | CODE behaviour + FORMALISATION. |
| M17 | Mixed-config property: with `rival_pool` ON, the v8 #7-#10 M_price multipliers are computed but never applied to LU7 (they only enter the gaming term of the non-rival branches). Moot at defaults (price edges OFF). | Code vs nothing documented | Noted where mixed configurations are described (2.3 Scope, 2.6.7 V-6). | CODE. |
| M18 | Edge #5's name suggests an archetype tilt on hour AND day; only hour is tilted. Even with #5 ON, day_of_week is drawn from the untilted marginal (the dow tilt table exists, unread: M7). | Edge naming/diagrams vs code (auctions.py:331-343) | #5 described as hour-only; D3 stays a root. Consistent with the register's LU1 -> D3 ✗ row. | CODE. |
| M19 | Inactive-archetype branch: schema says LU4 "Beta per archetype" and LU5 "LogNormal per archetype"; code forces LU4 = 0 and LU5 = 0 for inactive users. RNG accounting: the payer Beta IS drawn for every archetype (user_profiles.py:76) and then overwritten to 0 (:79-84), so the draw still consumes the user-stream RNG; only LU5's LogNormal is genuinely never drawn (its fixed branch replaces the draw). The distinction matters in a document that elsewhere trades on exact stream accounting for byte-identity. | Schema v10 latent table vs code (the formalisation already states "overwritten to 0") | The structural non-payer branch is stated with the draw-then-overwrite accounting (1.6, 2.4). | CODE (formalisation agrees). |
| M20 | What counts as a variable: the formalisation excludes user_id, app_id, advertiser_id, campaign_id as row/entity labels (not generative variables); the schema carries B1/C1/C3 as observed features and notes app/campaign latents are partially recoverable from the ids. | Formalisation scope vs schema observability | The master table keeps B1, C1, C3 as observed variables (always-visible columns; id-recoverability note, 1.8) and user_id as lineage-only; the formalisation's exclusion is recorded as a modelling-scope statement, not a contradiction. | SCHEMA (observability) + CODE (the columns exist). |

**Verifier-pass corrections applied (22 Jul, all code-checked).** (1) V-6 `ev_market` is auto-forced by any of #6-#10, not #7-#10 (`_PRICE_EDGES` includes `competition`, auctions.py:88-91; `rival_pool` is NOT in the list). (2) H2's factor enumeration includes ease(B2) (auctions.py:518; added to 1.6, M13/B-14, and the ease row now notes it feeds H2). (3) M19 rephrased: the inactive payer Beta is drawn then overwritten, consuming the user-stream RNG (user_profiles.py:76-84). (4) the H5-H8 entry rescoped to "numbering retired": hist_clearing_ssp/dsp have live flag-gated pipeline code (V-8; pipeline.py:159-202, schema.py:30-33) and bid_density's v8 emission is dormant (V-1), living on as H9. (5) The dial list carries all four OFF-path v7 auction dials (gamma 0.45, sigma_g 0.35, mu_x -0.45, sigma_x 0.55), not gamma alone. (6) Edge B-16 carries H4's full parent set (H2 added) and is flagged for register amendment; M13's row count corrected to seven, extending the same standard to B-8/B-9/B-10. (7) The funnel-chain edges E1 -> F1, F1 -> G1, G1 -> G2 added to the edge inventory (B-25..B-27). (8) t_pay is raked over the calibration (hour, dow) joint, not the "population joint" (auctions.py:52, 119-121; M6). (9) user_vbin's parent set includes the selection index u_rows (1.6), matching the formalisation.

**Open questions (no arbiter can settle; owner action).**

| # | Question |
|---|---|
| Q1 | RESOLVED (Ken, 22 Jul 2026): option A. `rival_pool` flag, `bn.rival_pool` params and `auction.rho` declared in `benchmarks.yaml`; defaults unchanged; byte-identity and config-only reproduction verified (4.1, Known_Defects 2). |
| Q2 | RESOLVED (Ken, 22 Jul 2026): sub-labels deleted. The specification is the coded partition, 3 gaming + 5 non-gaming. |
| Q3 | RESOLVED (Ken, 22 Jul 2026): applied. `Known_Defects_Register` gained section 3c (chunk-relative day index, M12) and section 2 is marked resolved; `Edge_Register` parent rows amended (M13) and its four rival-pool CPT crosses are now ticks. |
