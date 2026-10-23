# Reproduction guide

The repository contains the scripts used for data preparation, training, and
evaluation, together with the source tables behind each exhibit.

Requirements: the three public datasets and trained bidder checkpoints.

1. Download and extract the datasets (§2).
2. Train the five bidders (§3).
3. Run the evaluators (§4), then `build_derived_data.py` to produce the source
   tables in `plotting/data/derived/`.
4. Rebuild the figures from those tables (§5).

---

## 1. Environment

A single Python ≥ 3.11 environment suffices for training, evaluation and
plotting; only the SemBid bidder needs a GPU with the vLLM stack.

```bash
# training + evaluation + plotting (campaign replay, synchronous market, DT/GAS, figures)
pip install torch numpy pandas pyarrow scipy scikit-learn xgboost matplotlib pyyaml

# T9Sim simulator package (install from the bundled source)
pip install -e code/t9sim/T9-simulator

# SemBid bidder additionally needs the vLLM dependencies of
# code/bidders/sembid_cpa/ (transformers + the Qwen2.5-0.5B weight).
```

---

## 2. Datasets — download and extraction

All three datasets are publicly available. The scripts in `data/<dataset>/`
handle download, integrity checks, and extraction; `POMS_DATA_ROOT` sets the
target directory (default: this repo's `data/`).

### 2.1 AuctionNet

```bash
python data/auctionnet/download_official_alimama.py   # 8 ZIPs → data/auctionnet/
```

The eight archives are on the Alimama OSS bucket:

```text
BASE=https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com
$BASE/share/autoBidding_aigb_track_data_period_7-8.zip
$BASE/share/final/autoBidding_aigb_track_final_data_period_7-8.zip
$BASE/share/autoBidding_aigb_track_data_trajectory_data.zip
$BASE/share/autoBidding_aigb_track_data_trajectory_data_extended_1.zip
$BASE/share/autoBidding_aigb_track_data_trajectory_data_extended_2.zip
$BASE/share/final/autoBidding_aigb_track_final_data_trajectory_data_1.zip
$BASE/share/final/autoBidding_aigb_track_final_data_trajectory_data_2.zip
$BASE/share/final/autoBidding_aigb_track_final_data_trajectory_data_3.zip
```

After download, the raw CSVs are converted to the evaluator's input format with
`data/auctionnet/convert_period_to_trajectory.py` and the
`data/auctionnet/auctionnet_official/convert_official_raw_to_trajectory.py`
pipeline (validation, conversion, and checksumming).

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

The files are published under Zenodo record 21533031
(`https://zenodo.org/records/21533031`); the download script reads
`https://zenodo.org/api/records/21533031`.

The replay driver is `code/t9sim/run_reference_experiment.py`
(`--parquet … --references … --kappas …`).

### 2.3 iPinYou

```bash
bash data/ipinyou/download_ipinyou_full.sh     # ipinyou.contest.dataset.7z (~6 GB)
bash data/ipinyou/download_ipinyou_season2.sh  # season-2 supplement
bash data/ipinyou/extract_audit_ipinyou.sh
python code/ipinyou/preprocess_ipinyou.py       # → per-campaign train/test parquet
```

The two source archives are:

- `https://www.dropbox.com/scl/fi/dlam6yqswcwu4x0dnooig/ipinyou.contest.dataset.7z?rlkey=h7q2ey4f7u8n0w7y8euo3zx20&dl=1`
- `https://ndownloader.figshare.com/files/10082688`

The logged-replay driver is `code/ipinyou/run_ipinyou_reference_experiment.py`
(`--train … --test … --campaign … --output …`).  The evaluation uses 19.5
million retained log records: 12,318,831 for model fitting (first 80%),
3,076,427 for calibration (remaining 20%), and 4,100,716 in the official test
split.

---

## 3. Train the autobidders

Training code for all five bidders is in `code/bidders/`. The entrypoints below
train them from the datasets in §2.

| Bidder | Training entrypoint |
| --- | --- |
| CQL | `code/bidders/sembid_cpa/bidding_train_env/baseline/cql/cql.py` |
| IQL | `code/bidders/sembid_cpa/bidding_train_env/baseline/iql/iql.py` |
| DT | `code/bidders/sembid_cpa/bidding_train_env/baseline/dt/dt.py` |
| GAS | `code/bidders/gas/run/train_dt_baselines.py` + `code/bidders/gas/run/train_dt_critics.py` |
| SemBid | `code/bidders/sembid_cpa/Training/train_exp23_2048.py` |

The evaluator loads the resulting checkpoints. For reference, the checkpoint
sizes used in the paper were:

| Bidder | Approx. size |
| --- | --- |
| CQL | 1.4 MB |
| IQL | 0.3 MB |
| DT | 32 MB |
| GAS | ~0.3 GB |
| SemBid | ~1 GB × 16 cp |

---

## 4. Evaluation

The evaluators produce the per-cell results used to build the source tables.

### 4.1 AuctionNet campaign replay + synchronous market

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

The experiment grids live in `configs/` (market `uniform`/`rollout_*` × `pid`/`dual`/
`traffic`). The synchronous 48-advertiser market evaluator is
`code/auctionnet/macro/src/platformbid_v1/evaluator.py`.

DT and SemBid source is bundled at `code/bidders/sembid_cpa/`; the evaluator
resolves it automatically relative to the repo (`--dt-code-root`,
`--sembid-code-root`, and `GAS_REPO_ROOT` env var).

### 4.2 T9Sim + iPinYou

```bash
python code/t9sim/run_reference_experiment.py  --parquet <auctions.parquet> ... --output cells.csv
python code/ipinyou/run_ipinyou_reference_experiment.py --train <t> --test <t> --campaign <c> --output cells.csv
python code/analysis/analyze_marketing_evidence.py --root <run-root>
```

The `analyze_marketing_evidence.py` aggregation emits four upstream tables:

- `ipinyou_policy_cells.csv`
- `ipinyou_wide_tight_contrasts.csv`
- `t9sim_policy_cells.csv`
- `t9sim_exclusive_purchase_differences.csv`

### 4.3 Dataset-scale accounting

```bash
python code/analysis/collect_dataset_raw_statistics.py \
  --project-root <...> --legacy-root <...> --external-root <...> \
  --output dataset_raw_statistics.json
```

---

## 5. Rebuild figures from the checked-in source tables

No data download or training is required:

```bash
cd plotting/scripts/server_release
python build_all_figures.py      # E-Companion figures → figures/ec/*.png
cd ../../.. && python plotting/scripts/rebuild_main_figures.py   # main figures 2–7 → figures/main/*.png
```

`build_derived_data.py` performs this aggregation from the evaluation outputs.

Figure 1 is the preserved conceptual asset
(`plotting/figures/main/fig1_execution_boundary_framework.png`).  Figure 8 is
typeset from `table2a`/`table2b`.  `FIGURE_TABLE_CROSSWALK.md` maps every
manuscript exhibit to its source CSV.

---

## 6. Notes on bundled source

- The bundled DT/SemBid source still carries a few `/home/wangmeiyi/AuctionNet/…`
  defaults from the authors' machines. They are confined to training and
  stand-alone test scripts; evaluation paths are passed explicitly on the CLI.
