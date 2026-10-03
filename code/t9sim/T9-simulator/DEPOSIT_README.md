# T9Sim RTB Data Simulator: Generating, Censoring and Benchmarking DSP/MMP/SSP Adtech Data Layers

Simulated real-time-bidding (RTB) auction datasets for mobile-game advertising, in which
**every auction carries its full counterfactual**.

In a real advertising log, a demand-side platform observes outcomes only for the auctions it
*won*. Whether a method that corrects for this censoring genuinely recovers the missing
counterfactual, or merely looks better on the data it can see, is untestable on real data,
because the counterfactual was never recorded.

Here it is recorded. Funnel outcomes (click, install, payer, 90-day spend) are generated on
**all** rows, won or lost, and the top competing bid is retained. Methods can therefore be
scored on the auctions the DSP lost.

---

## What is in this deposit

| File | Contents | Size |
|---|---|---|
| `t9_v10_1m_sample_seed90213.zip` | **Start here.** 1M-auction sample, seed 90213 | ~0.17 GB |
| `t9_v10_10m_seed90213.zip` … `…90222.zip` | The ten 10M masters behind the published results | ~1.7 GB each |
| `t9sim-1.0.1-source.zip` | The generator that produced them, tagged release `v1.0.1` | ~1.6 MB |
| `MANIFEST.txt` | Checksums (MD5 + SHA-256), row counts, content fingerprints | — |

Each archive is self-contained. It unpacks to a directory holding the auction master, the three
entity pools it was drawn from, and the metadata describing how it was made.

```
auctions.parquet         the auction master (55 columns)
pool_users.parquet       the user pool
pool_apps.parquet        the app pool
pool_campaigns.parquet   the campaign pool

manifest.json            this run's settings: seed, profile, pool sizes, the
                         privateness dial rho, which dependency edges were on,
                         and the calibrated constants
provenance.csv           every configuration parameter with its value and where
                         that value came from, including which were solved by
                         auto-calibration rather than chosen
validation_report.csv    the six calibration gates, with achieved values
```

The last three mean a downloaded archive can state its own identity and quality without
reference to anything else.

`t9sim-1.0.1-source.zip` is the complete source tree of the generator at the tagged release
`v1.0.1`. The exact commit it was exported from is recorded on the `git =` line of that archive's
entry in `MANIFEST.txt`, which is generated alongside the archive and cannot drift from it. It
holds the package, the
configuration, the iPinYou-derived calibration tables, the tests, the paper runners, the
specification and the datasheet. It is included so that this record
does not depend on the GitHub repository continuing to exist: the data and the code that produced
it are archived together, and either can be checked against the other.

That includes the scripts that rebuild this deposit. `deposit_gen_driver.py` regenerates the ten
10M masters and `deposit_gen_sample.py` the 1M sample, then `make_deposit_package.py` repackages
them into exactly these archives with a matching manifest.

The source snapshot is **MIT** licensed, and carries its own `LICENSE`. The datasets are CC BY
4.0. That is why the record as a whole is marked CC BY 4.0, which is the more restrictive of the
two.

The two scales use different pool sizes:

| | auctions | users | apps | campaigns |
|---|---|---|---|---|
| 10M master | 10,000,000 | 2,200,000 | 1,200 | 200 |
| 1M sample | 1,000,000 | 220,000 | 500 | 100 |

Exact row counts for every file are in `MANIFEST.txt`.

**You do not need the whole deposit.** Take the 1M sample to explore the schema, or a single 10M
archive to run an experiment. The ten seeds exist so that results can carry confidence
intervals; you only need all of them to reproduce the published intervals exactly.

## Quick start

```python
import pandas as pd

df = pd.read_parquet("t9_v10_1m_sample_seed90213/auctions.parquet")
print(df.shape)                      # (1000000, 55)

lost = df[df.won == 0]               # 695,683 auctions the DSP did not win
print(len(lost), lost.click.mean())  # 695683  0.0398
```

Those last two lines are the point of the dataset. They read outcomes for impressions that were
never served, on rows where a real platform would hold no labels at all.

## The four conditions

The datasets are uncensored masters. The benchmark applies **censoring operators** that reduce
the master to the views real platforms actually hold:

| View | Sees | Real-world analogue |
|---|---|---|
| **C1** | own wins' funnel only, no clearing prices | a standard DSP setup — MMP attribution on its own wins only (the "biased view") |
| **C2** | C1 + funnel labels on *all* rows | deeper MMP integration: funnel outcomes beyond own wins |
| **C3** | C1 + clearing prices and rival count (funnel still on won rows only) | DSP + SSP (supply-side) |
| **C4** | all layers | fully integrated stack |

Censoring is both column-conditional (which columns exist) and row-conditional (which rows carry
labels). The implementation is `censor.py` in the software repository, used as
`t9.view(df, "C1")`.

## Ground-truth columns

These are **oracle** quantities. They are what makes the dataset useful, and they are never
exposed as model features in any condition. Use them to score, not to train.

| Column | Meaning |
|---|---|
| `p_click`, `p_install`, `p_payer` | the true per-row probabilities |
| `e_ltv` | true expected spend |
| `ev_truth` | true expected value of the impression |
| `lu7_competing_bid` | highest rival bid, **present on lost rows too** |

Training on these leaks the answer. The software repository's test suite includes leakage gates
for exactly this reason.

## Which spend column to use

`ltv_value` is the 90-day post-install total, and it is the spend target to train and score
against. `ltv_7d` and `ltv_30d` are the same figure at earlier recognition points, 40% and 70% of
it, not separate targets. `e_ltv` and `ev_truth` are on the `ltv_value` scale, so training on
`ltv_7d` and scoring against them understates spend by 60%.

## Verifying what you downloaded

**Did the transfer arrive intact?** Zenodo shows an MD5 for each file. Compare it against the
`md5 =` line for that archive in `MANIFEST.txt`.

**Is this the data behind the paper?** Unpack it and compute the content fingerprint. This needs
only pandas and pyarrow, no install of the simulator:

```python
import pandas as pd, pyarrow.parquet as pq
from pathlib import Path

def fingerprint(d):
    M, total = (1 << 64) - 1, 0
    for f in sorted(Path(d).glob("*.parquet")):
        df = pq.read_table(f).to_pandas()
        total = (total + int(pd.util.hash_pandas_object(
            df[sorted(df.columns)], index=False).sum() & M)) & M
    return total

print(hex(fingerprint("t9_v10_10m_seed90213/")))
```

Compare the result with the `fingerprint =` line for that archive in `MANIFEST.txt`.

If you have the simulator installed, `python -m t9sim.fingerprint <dir>` does the same thing.

Prefer the fingerprint over a file hash when comparing against data you generated yourself.

The **parquet files** do reproduce byte-for-byte on the same platform and library versions. We
checked: a full regeneration matched SHA-256 on all forty parquets, and produced identical
`provenance.csv` and `validation_report.csv` as well. But two things will still differ:

- `manifest.json` records `elapsed_s`, the wall-clock time that run took, so it can never match.
  The **archive as a whole therefore cannot be reproduced byte-for-byte**, even though the data
  in it can.
- Parquet stores the writer version, so a different pyarrow release or platform can change the
  bytes while the data is unchanged.

The fingerprint hashes column *values* only, ignoring both. That is why it, and not a file hash,
is the check to use.

## Provenance: these are the datasets behind the published results

The archives here were generated from source rather than copied from the original experimental
run, whose parquet files had been reclaimed after their metrics were recorded. Because
generation is seed-deterministic, regeneration reproduces the originals exactly. That was
verified rather than assumed, on four independent checks:

| Check | Coverage | Result |
|---|---|---|
| **SHA-256 of every parquet** | all 40 files (10 seeds × 4 parquets) | identical |
| File sizes | all 40 files | identical |
| Parquet row-group statistics (min, max and null count per column) | 46,400 statistics | identical |
| `validation_report.csv`, computed from the data | 10 seeds | identical |
| `provenance.csv`, the parameter set | 10 seeds | identical |
| Generator validation metrics against the original run log | six metrics × five seeds | identical |
| Temporal train/validation/test split counts | seed 90213 | identical |

The regenerated parquets are byte-for-byte identical to the originals, so this is reproduction in
the strictest sense rather than a statistical resemblance. The one file that differs is
`manifest.json`, which records how long the run took.

## Regenerating instead of downloading

The data is fully reproducible from source. Generation is seed-deterministic, and the operative
configuration (the privateness dial `rho`, the `bn.rival_pool` edge flag, and the rival-pool
parameters) is declared in `config/benchmarks.yaml` in the software repository:

```bash
python scripts/deposit_gen_driver.py      # ~40 min, ~18 GB, resumable
python scripts/make_deposit_manifest.py output/v10_10m_s* --fingerprint
```

Then compare fingerprints against `MANIFEST.txt`.

## How it was made, and what that means for interpretation

The generator is a structural causal model. Four pools of persistent entities (users, apps,
campaigns, rival bidders) are drawn once, then each auction row is produced by ancestral
sampling. Calibration uses **aggregated distribution shapes** from the public iPinYou dataset
(Zhang et al., 2014), rescaled to a 2025 US mobile-gaming market. No raw iPinYou records are
included or redistributed.

Prices are modelled as **first-price**, consistent with post-2019 mobile in-app mediation.
iPinYou's second-price clearing prices are reused as an empirical shape, justified by revenue
equivalence. This is a declared modelling assumption, not a measurement.

**Read this before drawing conclusions:**

- **Absolute numbers are not market estimates.** Levels are calibrated to public benchmarks, not
  measured. The dataset supports comparisons *between conditions and methods*. It does not
  support claims about what any real platform earns.
- **Results are conditional on the data generating process.** The project's own history is the
  cautionary case: an earlier schema produced a null SSP effect that a later schema showed to be
  an artefact of an independent-values market, not a fact about SSP data.
- **The privateness dial `rho` is externally anchored, not estimated**, and the build does not
  rake log-bid moments, so raising it also inflates total price variance. This is a declared
  limitation.

The full specification, the identification propositions, and the register of declared
divergences between specification and code are in the software repository.

## Software, specification, and citation

Software: `t9sim-1.0.1-source.zip` **in this record**, and at
<https://github.com/QUMLKY/T9-simulator> (MIT licensed). The two are the same tree; the
repository is where development continues, the snapshot is what these datasets were made with.

Either holds the generator, the censoring operators, the reference model pipeline, the full
specification (`docs/T9Sim_Specification_v10.md`), the datasheet (`DATASHEET.md`), and the
reference benchmark entrants.

```bash
unzip t9sim-1.0.1-source.zip && cd t9sim-1.0.1
python -m venv venv && venv/Scripts/activate    # or source venv/bin/activate
pip install -e . && pytest                      # 35 tests
python examples/quickstart_100k.py              # ~3 min end to end
```

These data are licensed **CC BY 4.0**: free to use, share and adapt, including commercially,
with credit.

Cite as **10.5281/zenodo.21533031**.
