# T9Sim RTB Auction Data Simulator: Generating, Censoring and Benchmarking DSP/MMP/SSP Adtech Data Layers

Code **MIT** · Data **CC BY 4.0** · Data DOI [10.5281/zenodo.21533031](https://doi.org/10.5281/zenodo.21533031) · **v1.0.1** · Python ≥ 3.11 · Cite: `CITATION.cff` · Datasheet: `DATASHEET.md`

T9Sim ("T9") is a calibrated simulator of mobile in-app game real-time-bidding (RTB) auctions
**with retained ground truth**, and a benchmark built on it. The design is three steps:

1. **Generate** one ground-truth stream of first-price auctions. Every row retains the full
   funnel (click → install → payer → 90-day spend) and the top competing bid, **won or lost** —
   the counterfactual outcomes no real log contains. The latent value drivers and the oracle
   expected value are retained for scoring only.
2. **Censor** that one stream into four views mimicking what real-world platforms can see. Two
   censoring operators — one for funnel outcomes, one for winning prices — add one data layer
   at a time:

   | View | Sees | Real-world analogue |
   |---|---|---|
   | **C1** | funnel labels on won rows only; no winning price on lost auctions | a standard DSP setup — MMP attribution on its own wins only (the "biased view") |
   | **C2** | C1 + funnel labels on *all* rows | deeper MMP integration: funnel outcomes beyond own wins |
   | **C3** | C1 + winning prices on all rows & rival count (funnel still on won rows only) | DSP + SSP (supply-side) |
   | **C4** | all layers | fully integrated stack |

   C2/C4's all-rows outcome visibility is a controlled idealisation of cross-network MMP
   attribution — an upper bound on the value of reducing outcome censoring, not a literal MMP
   capability (see `DATASHEET.md`).
3. **Train the same two-tier bidding algorithm under each view** and score it against the
   retained truth — lost auctions included — and against an oracle ceiling (the same pipeline
   given the true latents). **Tier 1 is the value model**: a four-head funnel — p(click),
   p(install | click), p(payer | install), E(spend | payer) — whose product is the impression's
   expected value (EV). **Tier 2 is the win model**, p(win | b). The bid rule maximises
   (EV − b) · p(win | b). Because only the view varies, the C1→C4 ablation isolates each data
   layer's marginal predictive and economic value.

Because the truth is retained, T9 answers questions no real log can: what each data layer is
worth on its own, how far every view sits below the oracle ceiling, and how large the won-rows
selection bias actually is — measured, not assumed.

## Headline results — the C1–C4 ablation (10M auctions × 10 seeds)

1. **The DSP-only view is biased.** Learning the funnel from won rows alone, C1 recovers only
   **52%** of true expected value (ev_ratio 0.524) — the selection bias of the "biased view".
2. **MMP corrects it, and the money follows.** Funnel labels on all rows lift ev_ratio
   **0.52 → 0.89**, payer-head AUC **+0.098**, and profit **+39%** at 10M — every contrast
   positive in all ten seeds.
3. **SSP improves the win model, not profit.** Winning-price visibility raises win-AUC
   **+0.0137 [+0.0055, +0.0220]** (10/10 seeds, scale-stable at 1M and 10M), but its profit
   contrast spans zero: no supported economic effect in this setting.
4. **The two layers load onto disjoint axes.** MMP moves only the value model, SSP only the
   win model. And even C4 sits well below the oracle (payer AUC 0.68 vs 0.84; ev_ratio 0.89
   vs 1.00) — the residual gap is the latent value no observable data layer carries.

Full tables: `docs/v10_Training_Results.md`.

## Install

```bash
python -m venv venv && source venv/Scripts/activate   # Linux/macOS: source venv/bin/activate
pip install -e .                                      # PowerShell: venv\Scripts\Activate.ps1
pytest                                                # 35 tests, ~1 min
```

Python ≥ 3.11. Dependencies are listed in `requirements.txt` (core: pandas, numpy, scipy,
pyarrow, xgboost, scikit-learn, pyyaml).

## Quickstart (~2 min)

```bash
python examples/quickstart_100k.py
```

generates a 100K master under the paper's operative configuration (schema **V10**, the
private-rival market, at the externally anchored privateness ρ\*=0.8), trains the same
two-tier bidding algorithm under C1–C4, and prints a summary table (single-seed 100K figures
are illustrative — the profit row in particular is noise-dominated at this scale and does not
reproduce the paper's MMP gain, which needs ≥1M and ten seeds). The same loop at 1M scale
(profile `"test"`):

```python
import t9sim.api as t9

t9.generate("test", seed=90213, rho=t9.RHO_STAR, bn_edges=t9.V10_EDGES)  # 1M master
results = t9.evaluate("test")                    # reference models, scored vs truth
```

**Bring your own method.** The harness accepts an external method through one function: load
the master parquet, take a censored view with `t9.view(df, "C1")`, train on what the view
exposes, and score your predictions against the master's retained truth columns — the
estimands (`p_click`, `p_install`, `p_payer`, `e_ltv`), the oracle expected value (`ev_truth`)
and the top competing bid (`lu7_competing_bid`) — on the all / won / **lost** row slices.
Because lost-row outcomes and prices are retained but hidden from training, the harness can
score whether a censoring correction recovers the counterfactual population rather than merely
improving observable-row metrics.

## Reproduce the paper numbers

| Result | Command |
|---|---|
| 1M ablation, n=10 seeds | `python scripts/v10_anchor_5seed.py` then `scripts/v10_anchor_n10.py` |
| 10M ablation, n=10 (one fresh process/seed, ~16 GB) | `python scripts/v10_10m_driver.py` (seeds 90213–17), then `python scripts/v10_10m_worker.py <seed>` for 90218–90222; aggregate the ten per-seed JSONs as in `scripts/v10_anchor_n10.py` (shipped aggregate: `docs/results/v10_10m_n10.json`) |
| Leakage negative control | `python scripts/neg_control_generator_off.py` |
| Rebuild the deposited 10M datasets (~40 min, ~17.6 GB) | `python scripts/deposit_gen_driver.py` |
| The reported results tables, from the per-seed JSONs | `python scripts/aggregate_results.py --out docs/results/v10_paper_tables.json` |

Per-seed result JSONs for the shipped write-ups are under `docs/results/`.

## Data & reproducibility

Generation is **seed-deterministic**: the same (profile, seed, ρ, edges) reproduces the dataset
byte-for-byte (verified via the content fingerprint in `src/t9sim/fingerprint.py`; the flag-OFF
configuration reproduces the v7 baseline fingerprint `0xdf0ac3e18624cf2b` — golden profile at
its default seed 90210). The paper uses seeds **90213–90222**.

The dials are declared in `config/benchmarks.yaml` — the privateness dial `rho`, the
`bn.rival_pool` edge flag, and the rival-pool parameters (`K`, `n_gaming`, `beta_R`,
`pacing_ar`, `pacing_sigma`, at paper values). The operative point is passed per run rather
than read from the config. `RHO_STAR` / ρ\* = 0.8, `rival_pool = True` and
`hist_clearing = True` are passed by the runner scripts and by `t9sim.api`. The first two are
recorded in each dataset manifest, the third in each results JSON.
`CHECKSUMS.txt` pins the SHA-256 of every calibration target and config file that fixes the
output; regenerate it with `python scripts/make_checksums.py`.

Generated data therefore ships as *recipes* (profile + seed + config), not binaries. The frozen
archives are deposited separately, with checksums and a DOI:

> **Datasets:** [10.5281/zenodo.21533031](https://doi.org/10.5281/zenodo.21533031) — the ten 10M
> masters behind the published results, plus a 1M sample. CC BY 4.0.

The ten 10M masters behind the headline results (seeds 90213–90222, ~17.6 GB with the 1M
sample) can be rebuilt locally rather than downloaded:

```bash
python scripts/deposit_gen_driver.py                              # ten 10M masters, ~40 min
python scripts/deposit_gen_sample.py                              # the 1M sample
python scripts/make_deposit_manifest.py output/v10_10m_s* --fingerprint
```

`scripts/make_deposit_package.py` then rebuilds the deposit exactly as published: one archive per
dataset, plus a source snapshot of the tagged release, with the manifest that pins them.

`DEPOSIT_MANIFEST.txt` records what was deposited: MD5 and SHA-256 per archive, row counts per
file, and a **content fingerprint** per dataset. Compare fingerprints. Regeneration on the same
platform and library versions does reproduce the files byte-for-byte — verified, SHA-256 matched
on all forty parquet files of the ten 10M masters — but parquet stores the writer version, so a
different pyarrow release can
change the bytes while the data is identical. The fingerprint hashes column values and survives
that.

Calibration targets in `calibration/` are **aggregated distribution shapes** derived from the
public iPinYou dataset (Zhang et al., 2014), rescaled to a 2025 US mobile-gaming market - no
raw iPinYou records are included. The derivation is fully reproducible:
`scripts/calibrate_ipinyou.py` regenerates every CSV the generator reads from the raw iPinYou
season 2+3 logs (download separately; ~28 GB).

## Repository layout

```
src/t9sim/        the package: generator (auctions.py), censoring (censor.py),
                  training/eval pipeline (pipeline.py), validation, fingerprint, api
config/           all tunable parameters (YAML; every value provenance-tagged)
calibration/      iPinYou-derived distribution shapes (CSV)
docs/             the specification (V10) + implementation status, results write-ups
docs/results/     per-seed and aggregated result JSONs backing the write-ups
scripts/          paper runners: ablation (1M/10M), sensitivity sweeps, deposit
                  rebuild and packaging
tests/            35 tests incl. leakage gates and schema contracts
diagrams/         schema maps (overview, generation, dependencies, rival prices),
                  the generator and dependency-graph figures, per-feature calculation SVGs
examples/         quickstart
```

## Formal model

The generative model is specified in `docs/T9Sim_Specification_v10.md` — the single source of
truth for the schema:

- **Part I — the data.** Every variable with its domain and parents, which layer observes it,
  and the C1–C4 censoring map (column- *and* row-conditional), reconciled against the emitted
  parquet.
- **Part II — the generator.** The four entity pools, the joint factorisation, the dependency
  graph over the anchored variables, each conditional law with its clips, and the rival-pool
  (private-price) layer.
- **Part III — identification.** Propositions **P1–P4**: what each data layer does and does not
  identify, and the two-dial result (value-awareness flat, privateness ρ responsive).

Its companion `docs/T9Sim_Implementation_Status.md` carries the build-side detail kept out of
the specification: edge wiring status, declared code-vs-spec divergences, tunables, validation
targets, and the audit trail.

A **datasheet** for the dataset (motivation, composition, collection, uses and their limits,
distribution, maintenance) is in `DATASHEET.md`.

## Limitations and ethics

The data is **fully synthetic** — no row corresponds to a real person, device or app, and no
PII is present. Results are conditional on the generative model (DGP-conditional) and model one
market slice (the 2025 US mobile in-app gaming market from the DSP's point of view). All-rows
outcome visibility in C2/C4 is an idealised upper bound; bid shading is out of scope for this
release; three of the four archetype tilts (device type, OS, day of week) are declared but
unwired, which lowers absolute recovery but leaves the C1–C4 contrasts intact. See
`DATASHEET.md` for the full limitations and ethics discussion.

## Status, license, citation

Current release: **v1.0.1** (tag `v1.0.1`; version metadata in `CITATION.cff`). Questions and
issues via the repository issue tracker.

**Code** (the `t9sim` package, scripts, tests, examples, config) is **MIT** — see `LICENSE`.
**Data** (calibration tables, result JSONs, and the deposited datasets) is **CC BY 4.0** — see
`LICENSE-DATA`.

Cite via `CITATION.cff`. The deposited datasets have their own DOI:
[10.5281/zenodo.21533031](https://doi.org/10.5281/zenodo.21533031).
