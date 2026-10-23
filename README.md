<div align="center">

# Execution Authority in Digital Advertising: Coordinating Autobidding and Campaign Control

**Code & Data Release**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#reproduction)
[![Datasets](https://img.shields.io/badge/Datasets-AuctionNet%20T9Sim%20iPinYou-8A2BE2)](#datasets-and-download-links)
[![中文文档](https://img.shields.io/badge/中文-README_CN-EB4B4B)](README_CN.md)

</div>

---

**English** · [**中文**](README_CN.md)

---

<p align="center">
  <img src="plotting/figures/main/fig1_execution_boundary_framework.png" alt="Execution Boundary Framework" width="800">
  <br><em>Execution Boundary Framework</em>
</p>

---

## Contents

- [What is in this repository](#what-is-in-this-repository)
- [The five learned autobidders](#the-five-learned-autobidders)
- [The three pacing controllers](#the-three-pacing-controllers)
- [Repository layout](#repository-layout)
- [Reproduction](#reproduction)
- [Datasets and download links](#datasets-and-download-links)
- [License](#license)

---

## What is in this repository

The paper studies **execution authority** in digital advertising: how much
bidder discretion (`κ`) a pacing-controlled autobidder should have over bid
execution, and how bidder and campaign control should coordinate. We evaluate
an execution boundary (execution width `κ`) across five learned autobidders,
three pacing controllers, and three datasets.

| Block | Contents | Location |
| --- | --- | --- |
| **Core evaluator** | Campaign replay with external competition fixed + synchronous 48-advertiser market + campaign-control policies | `code/auctionnet/` |
| **Autobidders** | Training code for CQL, IQL, DT, GAS, and SemBid | `code/bidders/` |
| **Pacing controllers** | traffic-aware, PID, dual (campaign-control policies) | `code/auctionnet/micro/mandate_core_v2.py`, `code/common/reference_policies.py` |
| **Datasets** | Download and extraction scripts for AuctionNet, T9Sim (synthetic auctions with latent outcomes), and iPinYou (real-world DSP log records) | `data/` |
| **Cross-dataset analysis** | iPinYou/T9Sim aggregation, campaign-load and scenario-calibration analyses, dataset-scale accounting | `code/analysis/`, `plotting/data/load/`, `plotting/data/server_checked/` |
| **Figures & tables** | Source CSVs and plotting code for each exhibit | `plotting/` |
| **Experiment grids** | Execution widths, adoption levels, and seeds | `configs/` |

---

## The five learned autobidders

| Bidder | Type | Source |
| --- | --- | --- |
| **CQL** | Conservative Q-Learning | `code/bidders/sembid_cpa/bidding_train_env/baseline/cql/` |
| **IQL** | Implicit Q-Learning | `code/bidders/sembid_cpa/bidding_train_env/baseline/iql/` |
| **DT** | Decision Transformer (`state_dim=16`, `K=20`) | `code/bidders/sembid_cpa/bidding_train_env/baseline/dt/` |
| **GAS** | Gradient-Ascent Strategy (DT policy + reweighted-search critic) | `code/bidders/gas/run/` |
| **SemBid** | Language-bidding model | `code/bidders/sembid_cpa/Training/` |

Training settings are listed in [`docs/TRAINING.md`](docs/TRAINING.md).

---

## The three pacing controllers

`traffic-aware`, `PID`, and `dual` pacing are implemented as auditable
campaign-control policies in `code/common/reference_policies.py` and
`code/auctionnet/micro/mandate_core_v2.py`. The execution boundary then constrains the
bidder's native action relative to the reference action:

```text
enforce_action(raw, ref, κ)  →  clip to [max{0, ref·(1−κ)}, ref·(1+κ)]
```

---

## Repository layout

```text
├── code/
│   ├── auctionnet/          # core evaluator (campaign replay + synchronous market) + campaign-control policies
│   ├── bidders/             # five learned autobidders' training + inference source
│   ├── t9sim/               # T9Sim simulator package + replay driver
│   ├── ipinyou/             # iPinYou preprocess + logged-replay driver
│   └── analysis/            # cross-dataset aggregation + dataset-scale accounting
├── configs/                 # experiment grids (execution width × adoption × seed)
├── data/
│   ├── auctionnet/          # download + convert scripts for AuctionNet
│   ├── t9sim/               # download + extract scripts for T9Sim (Zenodo)
│   └── ipinyou/             # download + extract scripts for iPinYou
├── plotting/
│   ├── data/derived/        # every figure/table source CSV (checked in)
│   ├── evidence/locked/     # locked evidence package (l01–l31)
│   └── scripts/server_release/  # build_derived_data.py + figure/table builders
└── docs/                    # reproduction guide + training settings
```

---

## Reproduction

The repository includes the scripts used for data preparation, training, and
evaluation, plus the source tables behind each exhibit.

Requirements: the three public datasets (links below).

1. Download and extract the datasets with the scripts in `data/<dataset>/`.
2. Train the five bidders with the code in `code/bidders/`.
3. Run the evaluators in `code/auctionnet/`, `code/t9sim/`, and `code/ipinyou/`.
4. Run `build_derived_data.py` to rebuild the source tables in
   `plotting/data/derived/` from the locked evidence in
   `plotting/evidence/locked/`. `FIGURE_TABLE_CROSSWALK.md` maps each exhibit
   to its table.
5. Rebuild the figures from the source tables:

```bash
pip install pandas numpy matplotlib
cd plotting/scripts/server_release
python build_all_figures.py      # → plotting/figures/ec/*.png
cd ../../.. && python plotting/scripts/rebuild_main_figures.py   # → main figures 2–7
```

---

## Datasets and download links

| Dataset | What it is | Download | License / terms |
| --- | --- | --- | --- |
| **AuctionNet** | Alimama auto-bidding track (period 7–8, trajectory data) | [Alimama OSS](https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/) — 8 ZIPs (period 7–8, trajectory, and `final/` variants; full list in `docs/REPRODUCTION.md` §2.1) | Alimama competition terms |
| **T9Sim** | Synthetic auctions with latent outcomes (10 generated masters, 100M auctions) | [Zenodo 21533031](https://zenodo.org/records/21533031) (DOI 10.5281/zenodo.21533031) | CC-BY-4.0 |
| **iPinYou** | Real-world DSP log records (19.5M: 12.3M model fitting + 3.1M calibration + 4.1M evaluation) | [contest dataset](https://www.dropbox.com/scl/fi/dlam6yqswcwu4x0dnooig/ipinyou.contest.dataset.7z?rlkey=h7q2ey4f7u8n0w7y8euo3zx20&dl=1) · [season 2](https://ndownloader.figshare.com/files/10082688) | iPinYou contest terms |

---

## License

The evaluation/analysis code in `code/` is released for academic reproduction.
Third-party components carry their own licenses: T9Sim (`code/t9sim/T9-simulator/`,
see its `LICENSE` / `LICENSE-DATA`), GAS (`code/bidders/gas/`), and the
iPinYou/AuctionNet datasets under their respective terms. See
`docs/REPRODUCTION.md` for per-component notes.
