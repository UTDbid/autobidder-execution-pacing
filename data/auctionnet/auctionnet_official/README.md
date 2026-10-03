# scripts/data/auctionnet_official — Official AuctionNet Flow

This directory replaces the deprecated `sample_auctionnet_raw.py` (which
replaced all 48 advertisers with `PidBiddingStrategy` and used
`pv_num=500`/`num_tick=24`, both diverging from the official
[`alimama-tech/AuctionNet`](https://github.com/alimama-tech/AuctionNet)
rules).

The scripts here mirror the **official flow** verbatim, with **only one
project-specific override**: the 48-entry CPA constraint list.

## Layout

```
scripts/data/auctionnet_official/
├── generate_official_raw.py
├── convert_official_raw_to_trajectory.py
├── validate_official_raw.py
├── validate_official_trajectory.py
└── README.md
```

The official repo is cloned under `external/auctionnet_official_<ts>/` and
imported via `PYTHONPATH`. The scripts do not modify it.

## Official-flow invariants enforced

1. **Mixed strategy pool** — `Controller.initialize_agents()`
   (`Controller.py:38-64`) instantiates **9 distinct strategy classes**
   across 48 advertisers (per-category parity i%2):
   `PidBiddingStrategy, IqlBiddingStrategy, TD3_BCBiddingStrategy,
   OnlineLpBiddingStrategy, CqlBiddingStrategy, BcBiddingStrategy,
   MbrlMopoBiddingStrategy, BcqBiddingStrategy, MbrlComboMicroBiddingStrategy`.
   `generate_official_raw.py` ASSERTs that the resulting
   `agent_strategy_counts` is NOT all-PID; otherwise it `SystemExit`s.
2. **PV generator** — `NeurIPSPvGen(num_tick=48, num_agent=48,
   num_agent_category=8, num_category=6, pv_num=500000)` (matches
   `config/test.gin`).
3. **Over-cost adjustment** — `run/run_test.py:184-199` while-loop with
   `adjust_over_cost` is preserved verbatim.
4. **`get_winner`** — `run/run_test.py:53-67` helper preserved verbatim.
5. **Raw schema (18 cols)** — `BiddingTracker.train_logging` writes the
   exact 18 columns in the order documented in
   `BiddingTracker.py:99-107` and `run_test.py:217-220`.
6. **`TrainDataGenerator._generate_train_data`** is the OFFICIAL raw →
   trajectory converter; `convert_official_raw_to_trajectory.py` uses it
   directly (no re-implementation).

## Project-specific override (the ONLY one)

The 48-entry `Controller.cpa_constraint_list` (default `[60..130]` ladder
in `Controller.py:103-112`) is affine-rescaled to the level's canonical
band:

| Level | Target CPA range | Rescale rule |
|---|---|---|
| `high` | `[6, 13]` | affine from `[60, 130]` to `[6, 13]` |
| `mid`  | `[20, 60]` | affine from `[60, 130]` to `[20, 60]` |
| `low`  | `[60, 130]` | verbatim (matches official) |

No other override is permitted. If you find yourself wanting to change the
agent pool, the PV generator, the auction env, or the over-cost loop —
STOP — that breaks the official-flow contract.

## Typical usage

```bash
# Set PYTHONPATH so the official repo + sembid_new src are both importable.
export PYTHONPATH=external/auctionnet_official_<ts>:src

# 1) Generate official raw (mid, >=20 GB by appending episodes).
python scripts/data/auctionnet_official/generate_official_raw.py \
    --level mid --split train \
    --num-episodes 6 --num-ticks 48 --pv-num 500000 \
    --output-dir data/mid/raw_official_20gb \
    --manifest data/manifests/sample_mid_train_official_20gb.json \
    --min-total-bytes $((20 * 1024**3))

# 2) Validate the raw (schema, advertiser count, CPA band, size, diversity).
python scripts/data/auctionnet_official/validate_official_raw.py \
    --raw-dir data/mid/raw_official_20gb \
    --generation-manifest data/manifests/sample_mid_train_official_20gb.json \
    --level mid \
    --report data/manifests/validate_mid_raw.json \
    --min-total-bytes $((20 * 1024**3))

# 3) Convert raw -> trajectory (per-period shards to bound memory).
python scripts/data/auctionnet_official/convert_official_raw_to_trajectory.py \
    --raw-dir data/mid/raw_official_20gb \
    --output-dir data/mid/train_official_20gb \
    --manifest data/manifests/convert_mid_official_20gb.json \
    --shard-mode shards

# 4) Validate trajectory (14-col schema, advertiser count, CPA band).
python scripts/data/auctionnet_official/validate_official_trajectory.py \
    --trajectory-dir data/mid/train_official_20gb \
    --source-manifest data/manifests/convert_mid_official_20gb.json \
    --level mid \
    --report data/manifests/validate_mid_trajectory.json
```

Test splits (high/mid/low) follow the same flow with `--split test` and
`--output-dir data/<level>/test_official_20gb`.

## Stop conditions (from the project rules)

A run is **NOT** complete unless ALL of these are true:

- `validate_official_raw.py` exits 0 (schema 18 cols, 48 advertisers,
  timeStepIndex 0..47, CPA in band, agent_strategy_counts NOT all-PID,
  total_bytes >= --min-total-bytes).
- `validate_official_trajectory.py` exits 0 (schema 14 cols, 48 advertisers,
  timeStepIndex 0..47, CPA in band).
- The generation manifest records `is_all_pid=false` and lists the
  per-class agent counts.

If any condition fails, the data must be re-generated before being
declared ready for downstream baseline training.

## Disk budget (per `--pv-num 500000 --num-ticks 48`)

- 1 episode → ~1.4 GB raw + ~50 MB trajectory (≈30× compression).
- mid train target 20 GB raw → ~15 episodes (~30 min wall on this host).
- 1 test split (2 episodes) → ~3 GB raw (sufficient for eval).

## See also

- `docs/technical_reports/<date>-official-auctionnet-mid-regeneration.md`
  for the regeneration report with concrete numbers.
- `external/auctionnet_official_<ts>/README.md` for the upstream official
  documentation.