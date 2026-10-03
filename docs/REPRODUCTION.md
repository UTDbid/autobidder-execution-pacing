# Reproduction guide

The paper's results are reproduced end to end: download the three public
datasets, train the five autobidders from the bundled source, evaluate them
under the three pacing controllers, and rebuild every figure and table.

```
download datasets (§2) → train autobidders (§3) → evaluate (§4) → build figures (§5)
```

If you only want the figures and tables as typeset numbers, the source tables
are already checked in and §5 can be run on its own (no data, no training).

---

## 1. Environment

A single Python ≥ 3.11 environment suffices for training, evaluation and
plotting; only the SemBid bidder needs a GPU with the OpenLBM / vLLM stack.

```bash
# training + evaluation + plotting (micro/macro, DT/GAS, κ-selection, figures)
pip install torch numpy pandas pyarrow scipy scikit-learn xgboost matplotlib pyyaml

# T9Sim simulator package (install from the bundled source)
pip install -e code/t9sim/T9-simulator

# SemBid bidder additionally needs the OpenLBM/vLLM dependencies of
# code/bidders/sembid_cpa/ (transformers + the Qwen2.5-0.5B weight).
```

---

## 2. Datasets — download and extraction

All three datasets have public download links. The bundled `data/<dataset>/`
scripts reproduce the download + integrity-audit + extraction steps and are
parameterized by `POMS_DATA_ROOT` (default: this repo's `data/` directory).

### 2.1 AuctionNet / Alimama AIGB

```bash
python data/auctionnet/download_official_alimama.py   # 8 ZIPs → data/auctionnet/
```

The script pins the 8 official archives on the Alimama OSS bucket
(`alimama-bidding-competition.oss-cn-beijing.aliyuncs.com`): period 7–8 and
the trajectory archives, plus the `final/` variants. After download, the raw
CSVs are converted to the evaluator's input format with
`data/auctionnet/convert_period_to_trajectory.py` and the
`data/auctionnet/auctionnet_official/convert_official_raw_to_trajectory.py`
pipeline (validation + conversion + checksum).

The evaluator consumes two files per level:

| Input | Meaning |
| --- | --- |
| `--data-pickle` | raw period pickle (e.g. `raw_data_period-8.a681c387.pickle`) |
| `--source-csv` | advertiser table with `budget` / `cpa` columns (e.g. `period-8.csv`) |

### 2.2 T9Sim (Zenodo 21533031)

```bash
bash data/t9sim/download_t9sim.sh           # 10 × ~1.7 GB zips from Zenodo
bash data/t9sim/extract_audit_t9sim_formal.sh   # → 10 × auctions.parquet (10M rows each)
```

The `code/t9sim/T9-simulator/` package is the simulator itself (the same
source deposited to Zenodo). The replay driver is
`code/t9sim/run_reference_experiment.py` (`--parquet … --references … --kappas …`).

### 2.3 iPinYou

```bash
bash data/ipinyou/download_ipinyou_full.sh     # ipinyou.contest.dataset.7z (~6 GB)
bash data/ipinyou/download_ipinyou_season2.sh  # season-2 supplement
bash data/ipinyou/extract_audit_ipinyou.sh
python code/ipinyou/preprocess_ipinyou.py       # → per-campaign train/test parquet
```

The logged-replay driver is `code/ipinyou/run_ipinyou_reference_experiment.py`
(`--train … --test … --campaign … --output …`). `code/ipinyou/make-ipinyou-data/`
is the upstream (Python 2) conversion tool kept for provenance of the raw
format; the Python-3 path above supersedes it.

---

## 3. Train the autobidders

Each learned bidder's training source is bundled and self-contained. Train them
per the paper's training procedure; the training data is derived from the
public datasets in §2 with the bundled download + conversion scripts.

| Bidder | Training entrypoint |
| --- | --- |
| CQL | `code/bidders/sembid_cpa/bidding_train_env/baseline/cql/cql.py` |
| IQL | `code/bidders/sembid_cpa/bidding_train_env/baseline/iql/iql.py` |
| DT | `code/bidders/sembid_cpa/bidding_train_env/baseline/dt/dt.py` |
| GAS | `code/bidders/gas/run/train_dt_baselines.py` + `code/bidders/gas/run/train_dt_critics.py` |
| SemBid | `code/bidders/sembid_cpa/Training/train_exp23_2048.py` |

The evaluator (§4) then loads the resulting checkpoints. For reference, the
authors' training outputs and their approximate sizes were:

| Bidder | Authors' checkpoint path (relative to their `$BID`) | Approx. size |
| --- | --- | --- |
| CQL | `sembid_new/models/baselines/high/20260725_recovery_v1/cql/full_attempt1/checkpoint_400000/cql_model.pth` + `normalize_dict.pkl` | 1.4 MB |
| IQL | `sembid_new/models/baselines/high/20260725_recovery_v1/iql/full_attempt1/checkpoint_400000/iql_model.pth` + `normalize_dict.pkl` | 0.3 MB |
| DT | `sembid_new/models/baselines/high/20260725_recovery_v1/dt/full_attempt2/checkpoint_400000/dt.pt` + `normalize_dict.pkl` | 32 MB |
| GAS | `sembid_new/models/gas/high/20260727_212220_20260725_recovery_v1_b_high_full1/` (`policy/dt.pt` + `critics/critic_*`) | ~0.3 GB |
| SemBid | `autonomy_assurance_20260810/staging/controller_adapters/sembid_high_200k` + `sembid_new/data/high/train/derived/sembid_cikm2026/20260729_high_qwen05b_800k/embedding_lookup.pkl` | ~1 GB × 16 cp |

---

## 4. Evaluation

The evaluation produces the manuscript's 31 locked-evidence aggregates
(`l01`–`l31`), which `build_derived_data.py` turns into the figure/table source
CSVs.

### 4.1 AuctionNet micro + macro (`l01–l11`, `l21–l30`)

The core evaluator is `code/auctionnet/micro/mandate_eval_v2.py`:

```bash
python code/auctionnet/micro/mandate_eval_v2.py \
  --data-pickle  <period pickle> \
  --source-csv   <period csv> \
  --grid-json    configs/micro_k1_grid.json \
  --model cql|iql|dt|pid|constant|gas|sembid \
  --checkpoint   <path> --normalizer <path> \          # cql/iql/dt (from §3)
  --gas-root     <path> --gas-action-num 5 \           # gas
  --sembid-model-dir <path> --sembid-embedding-lookup <path> \  # sembid
  --reference-mode pacing|traffic_aware|smoothed_controller|response_aware \
  --policy-mode hard --conversion-seed <s> --market-seed <s> \
  --output <results.csv>
```

The frozen grids live in `configs/` (macro `uniform`/`rollout_*` × `pid`/`dual`/
`traffic`) and in `code/kappa_selection/02_configs/` (`p8_retrospective_missing_grid.json`
and split grids). The synchronous 48-advertiser macro evaluator is
`code/auctionnet/macro/src/platformbid_v1/evaluator.py`.

DT and SemBid source is bundled at `code/bidders/sembid_cpa/`; the evaluator
resolves it automatically relative to the repo (`--dt-code-root`,
`--sembid-code-root`, and `GAS_REPO_ROOT` env var).

### 4.2 T9Sim + iPinYou (`l12–l15`)

```bash
python code/t9sim/run_reference_experiment.py  --parquet <auctions.parquet> ... --output cells.csv
python code/ipinyou/run_ipinyou_reference_experiment.py --train <t> --test <t> --campaign <c> --output cells.csv
python code/analysis/analyze_marketing_evidence.py --root <run-root>
```

The `analyze_marketing_evidence.py` aggregation emits the four upstream tables
mapped to `l12–l15`:

| Locked evidence | Upstream output |
| --- | --- |
| `l12_ipinyou_canonical_logged_policy_raw.csv` | `ipinyou_policy_cells.csv` |
| `l13_ipinyou_canonical_logged_policy_contrasts.csv` | `ipinyou_wide_tight_contrasts.csv` |
| `l14_t9sim_policy_cells.csv` | `t9sim_policy_cells.csv` |
| `l15_t9sim_exclusive_purchases.csv` | `t9sim_exclusive_purchase_differences.csv` |

### 4.3 κ-selection (`l16–l20`)

```bash
cd code/kappa_selection/03_code
python prepare_challenge.py                       # freeze grids/manifest (idempotent)
python analyze_selection_challenge.py --phase retrospective
```

The retrospective phase reads the inherited calibration + test surfaces and the
per-cell `results.json` (produced by `mandate_eval_v2.py` above) and writes:

| Locked evidence | Output |
| --- | --- |
| `l16_kappa_selection_calibration.csv` | `calibration_surface.csv` |
| `l17_kappa_selection_evaluation.csv` | `evaluation_surface.csv` |
| `l18_kappa_selection_rules.csv` | `rule_summary.csv` |
| `l19_kappa_selection_gates.csv` | `success_gates.csv` |
| `l20_kappa_selection_decision.json` | `decision.json` |

`KAPPA_INPUT_ROOT` points at the directory holding the unpacked upstream
results. The authors' frozen retrospective outputs are bundled at
`code/kappa_selection/06_results/retrospective/` for reference.

### 4.4 Dataset-scale accounting (`l31`)

```bash
python code/analysis/collect_dataset_raw_statistics.py \
  --project-root <...> --legacy-root <...> --external-root <...> \
  --output dataset_raw_statistics.json
```

→ maps to `l31_dataset_scale_accounting.json`.

---

## 5. Rebuild derived CSVs and figures

Once `l01`–`l31` are at `plotting/evidence/locked/`:

```bash
cd plotting/scripts/server_release
python build_derived_data.py     # l01–l31 → plotting/data/derived/{main,ec}/*.csv
python build_all_figures.py      # derived CSVs → figures/{main,ec}/*.{pdf,svg,png}
```

The `l01`–`l31` filenames are defined in `build_derived_data.py` (`FILES`), and
`FIGURE_TABLE_CROSSWALK.md` maps every manuscript exhibit to its source CSV.

**Figure-only shortcut** — the derived source tables are already checked in, so
the figures can be rebuilt without the data, training or evaluation above:

```bash
cd plotting/scripts/server_release
python build_all_figures.py      # from the checked-in plotting/data/derived/
```

---

## 6. Notes on bundled source

- `code/bidders/sembid_cpa/` serves **both** DT and SemBid: it is the
  `Code/Code` tree, byte-identical to the `legacy_sembid_cpa` tree the DT
  checkpoint was trained from (md5-verified).
- The GAS adapter's `git rev-parse` provenance check is skipped when `.git` is
  absent (this repo strips VCS metadata); it still verifies when `.git` exists.
- The bundled DT/SemBid source still carries a few `/home/wangmeiyi/AuctionNet/…`
  defaults from the authors' machines. They are confined to training/stand-alone
  test scripts and to an `argparse` default inside `test_exp23_standard.py`'s
  `__main__` block, so the evaluator import is unaffected — every evaluation
  data/checkpoint path is passed explicitly on the CLI.
- `code/ipinyou/make-ipinyou-data/` is Python 2 (upstream iPinYou tool), kept
  for raw-format provenance; the Python-3 `preprocess_ipinyou.py` is the path
  actually used.
- One Mid-level recalibration launch script (`run_mid_candidate_grid.sh`) was
  `0700` on the authors' system and could not be read; the Mid-recalibration
  `.py` pipeline it fronted is present in `data/auctionnet/auctionnet_official/`.
