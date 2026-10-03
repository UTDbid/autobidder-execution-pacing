# Datasheet for T9

Following *Datasheets for Datasets* (Gebru et al., 2021). T9 is a **generator plus a benchmark**,
not a fixed corpus: the data are produced by seed-deterministic simulation, so this datasheet
describes both the generator and the reference datasets released with the paper.

---

## Motivation

**For what purpose was the dataset created?**

To make a class of question answerable that no real advertising log can answer. In real
real-time bidding (RTB), a demand-side platform only observes outcomes for auctions it *won*.
Whether a method that corrects for this censoring actually recovers the missing counterfactual,
or merely looks better on the observable data, is untestable on real data because the
counterfactual was never recorded.

T9 generates every auction with its full outcome (click, install, payer, 90-day spend) on **all**
rows, won or lost, and retains the top competing bid. Four censoring operators then reduce that
one master table to the views real platforms actually hold. Methods can therefore be scored
against retained truth, including on the rows the bidder lost.

The dataset also supports the study it was built for: a controlled ablation measuring the
marginal predictive and economic value of integrating DSP, MMP (attribution) and SSP
(supply-side) data layers.

**Who created it and who funded it?**

Created by the author as part of an MSc dissertation at Queen Mary University of London. No
external funding; no commercial sponsor.

---

## Composition

**What do the instances represent?**

One row is one **ad auction**: a (user, app, campaign, time) opportunity, the bid placed, the
auction outcome, and the full post-auction funnel. Instances are synthetic. No row corresponds
to a real person, device, app, advertiser, or auction.

**How many instances are there?**

The reference releases are **1,000,000** and **10,000,000** auctions per seed, at **10 seeds**
(90213 to 90222), for each scale. A 1M master parquet is about 152 MB; a 10M master is about
1.5 GB. Any size can be generated on demand.

**What data does each instance consist of?**

**55 columns** in the headline configuration, in eight families plus latents:

| Family | Content |
|---|---|
| A | user attributes (region, city, os, os_version, device_type) |
| B | app / placement (app_id, app_category, ad_exchange, slot_format, slot dimensions) |
| C | campaign (campaign_id, advertiser_id, advertiser_scale, ad_genre) |
| D | time (timestamp, hour_of_day, day_of_week) |
| E | click (click, click_timestamp) |
| F | install (install, install_timestamp) |
| G | lifetime value (is_payer, ltv_value, ltv_7d, ltv_30d) |
| H | auction (floor_price, bid_price, won, clearing_price, bid_density) |
| LU / LA / LC | user, app and campaign latents (the generative drivers, e.g. lu1_archetype, la1_app_quality, lc1_creative_appeal) |

`ltv_value` is the 90-day post-install total and is the spend target. `ltv_7d` and `ltv_30d` are
the same figure at earlier recognition points, 40% and 70% of it, not separate targets. `e_ltv`
and `ev_truth` are on the `ltv_value` scale, so training on `ltv_7d` and scoring against them
understates spend by 60%.


Full definitions, domains and parents: `docs/T9Sim_Specification_v10.md` Part I.

**Is there a label or target?**

Several, depending on the task. The benchmark's distinguishing feature is the **retained
ground truth**, which a real dataset cannot contain:

- `p_click`, `p_install`, `p_payer`, `e_ltv` - the true per-row probabilities and expected spend
- `ev_truth` - the true expected value of the impression
- `lu7_competing_bid` - the highest rival bid, present on lost rows too

These are oracle quantities. They are **never** exposed as model features in any condition; they
exist to score predictions.

**Is any information missing?**

By design. The four conditions are censoring operators over the one master table:

| View | Sees | Analogue |
|---|---|---|
| C1 | funnel outcomes on won rows only, no clearing prices | a DSP alone |
| C2 | funnel outcomes on all rows | DSP + MMP |
| C3 | won-rows funnel, plus clearing prices and rival count | DSP + SSP |
| C4 | all layers | integrated stack |

The censoring is both column-conditional (which columns exist) and row-conditional (which rows
carry funnel labels). Implemented in `src/t9sim/censor.py`.

**Does the dataset contain confidential or personally identifiable data?**

No. Every instance is generated. The calibration inputs are aggregated distribution shapes (see
Collection Process), not records. Region and city are anonymised integer codes with a canonical
hierarchy; no real place names, advertiser names, or app names appear.

**Does the dataset contain data that might be offensive or distressing?**

No. The content is numeric auction and funnel data with no text, images, or human-authored
material.

---

## Collection process

**How was the data acquired?**

It was generated, not collected. The generator is a structural causal model: four pools of
persistent entities (users, apps, campaigns, rival bidders) are drawn once, then each auction row
is produced by ancestral sampling through the dependency graph. Generation order, the joint
factorisation, and every conditional law are specified in `docs/T9Sim_Specification_v10.md`
Part II.

**What calibrated it?**

Distribution *shapes* from the public **iPinYou** RTB dataset (Zhang et al., 2014), rescaled to a
2025 US mobile-gaming market. The shipped CSVs in `calibration/` are aggregated distributions
(histograms and summary tables). **No raw iPinYou records are included or redistributed.**
`scripts/calibrate_ipinyou.py` regenerates every CSV from the raw iPinYou season 2 and 3 logs,
which must be downloaded separately (about 28 GB).

Price *levels* are not taken from iPinYou. They are rescaled to 2025 US mobile eCPM benchmarks,
because iPinYou is 2013 China display. Auctions are modelled as **first-price**, consistent with
post-2019 mobile in-app mediation. iPinYou's second-price clearing prices are reused as an
empirical shape for first-price clearing, justified by revenue equivalence. This is a declared
modelling assumption, recorded in the specification.

**Over what timeframe?**

Each generated dataset spans a 28-day window with an hour-by-day-of-week activity shape derived
from iPinYou. The simulator itself was developed between June and July 2026.

**Were ethical review processes conducted?**

Not required. The work involves no human subjects and no personal data. iPinYou is a public
research dataset and is used only in aggregate.

---

## Preprocessing, cleaning, labelling

Not applicable in the usual sense: labels are generated, not annotated, so there is no labelling
noise and no annotator disagreement. The generator applies documented clips and bounds to keep
draws in valid domains; each is recorded with its conditional law in the specification.

Validation is declared in `config/validation.yaml` and run by `src/t9sim/validate.py`: six
calibration target ranges (population CTR, click-to-install, install-to-payer, whale
concentration, median payer spend, auction win rate) and five direction assertions (for example,
that expected value on lost auctions exceeds that on won auctions, so adverse selection is
present). The direction assertions are enforced by the test suite; the level targets are
calibration goals rather than hard gates. A content fingerprint (`src/t9sim/fingerprint.py`)
hashes column values rather than file bytes, so determinism can be verified across machines.

---

## Uses

**What has the dataset been used for?**

A four-condition ablation (C1 to C4) measuring the marginal value of the MMP and SSP data
layers, at 1M and 10M scale, 10 seeds each. The same two-tier bidding algorithm is trained under
each view and scored against retained truth, including a lost-rows slice, and against an oracle
ceiling.

Results: `docs/v10_Training_Results.md`.

Feature-attribution (SHAP) results are deliberately **not** part of this release. The pipeline
retains the capability, so anyone can compute their own, but no attribution figures or values
are published here and none should be attributed to the authors.

**What other tasks could it be used for?**

Learning under selection and censoring generally: propensity and inverse-weighting methods,
censoring-aware survival or price models, off-policy evaluation of bidding policies, and
calibration research where the true probability per row is needed.

**Is there anything that should NOT be done with it?**

- **Do not treat absolute numbers as market estimates.** Levels are calibrated to public
  benchmarks, not measured. The dataset supports *comparisons between conditions and methods*,
  which is what it was built for. It does not support claims about what any real platform earns.
- **Do not train a production bidder on it** and expect transfer. The generator is a simplified
  market.
- **Do not read a null result as universal.** Findings here are conditional on the data
  generating process. The project's own history is the cautionary case: an earlier schema (v8)
  produced a null SSP effect that a later schema (v10) showed to be an artefact of an
  independent-values market rather than a fact about SSP data.

**Is there anything that biases future uses?**

Yes, and it is worth stating plainly. The identification results in Part III of the
specification are properties of *this* generator. The privateness dial `rho` is externally
anchored (`docs/v10_Anchor_Bands.md`) rather than estimated from data, and the build does not
rake log-bid moments, so raising `rho` also inflates total price variance. Declared divergences
between specification and code are listed in `docs/T9Sim_Implementation_Status.md`. Anyone
drawing conclusions should read that file.

**Which limitations bite hardest in practice?**

Three, stated here rather than left to the divergence register.

- **Three declared archetype tilts are not wired.** The specification declares that a user's
  latent archetype tilts their `os`, `os_version`, `device_type` and `day_of_week`. Those CPTs
  are built but no code consumes them, so **the released data carries no archetype signal in
  those four columns** and their marginals are drawn untilted. A method that appears to extract
  user-type information from device or day-of-week here is fitting noise. Detail:
  `docs/T9Sim_Implementation_Status.md` §2.6.3 and the wiring summary.
- **There is no post-auction price feedback on wins.** Auctions are first-price, so
  `clearing_price` on a won row is the winner's own bid. The highest rival bid
  (`lu7_competing_bid`) exists on every row but is a latent, excluded from all four conditions.
  No condition therefore observes its own overpayment, and bid-shading research needs the
  generator modified rather than this data as released.
- **Edge #2 (campaign exposure) is discretised.** The continuous per-user value tilt is applied
  through value deciles rather than continuously, so `P(campaign | user)` is a step function of
  latent value. Marginals and monotonicity are preserved; the functional form is not.

---

## Distribution

**How is it distributed?**

Two ways:

1. **As a recipe.** The generator, configuration and calibration targets are in this repository.
   Generation is seed-deterministic, so `(profile, seed, rho, edges)` plus the pinned config
   reproduces the dataset byte-for-byte. `CHECKSUMS.txt` pins the SHA-256 of every input that
   determines the output.
2. **As frozen archives.** The reference 10M datasets are deposited on Zenodo with a DOI,
   [10.5281/zenodo.21533031](https://doi.org/10.5281/zenodo.21533031), for reviewers who
   prefer not to regenerate.

**Under what licence?**

Split by kind. **Code** (the `t9sim` package, scripts, tests, examples, configuration) is under
the **MIT Licence** (`LICENSE`). **Data** (the calibration tables, the result JSONs, and the
datasets generated by this software and deposited by the authors) is under **CC BY 4.0**
(`LICENSE-DATA`), so reuse is unrestricted provided the work is credited.

The repository is a private preview for collaborators until public release.

**Are there restrictions from third parties?**

The calibration tables are derived from iPinYou. Only aggregated shapes are redistributed here.
Anyone regenerating them from raw logs is bound by iPinYou's own terms.

---

## Maintenance

**Who maintains it, and how can they be contacted?**

The author. Citation metadata is in `CITATION.cff`. Issues and questions
via the repository issue tracker.

**Will it be updated?**

The schema is versioned and older versions are retained for the record. The version this paper
reports is **V10** at the externally anchored `rho* = 0.8`. Corrections and clarifications will
be released as tagged versions; the tag used for any published result is stated with that result.

**Will older versions continue to be supported?**

Yes, in the sense that matters for reproduction: with the v10 edge flags off, the generator
reproduces the earlier v7 baseline byte-for-byte. That identity is checked by
`scripts/neg_control_generator_off.py`, which reports the fingerprint
(`0xdf0ac3e18624cf2b` for the golden profile at the default seed). Superseded schema versions
remain documented so published numbers stay traceable.

**Can others contribute?**

Yes. The benchmark is designed for it: implement a method, take a censored view with
`t9.view(df, "C1")`, and score against the master's retained truth on the all, won, and lost
row slices.
