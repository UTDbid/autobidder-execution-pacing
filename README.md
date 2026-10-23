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
discretion (`κ`) a pacing-controlled autobidder should have over bid execution,
and how bidder and campaign control should coordinate. We evaluate an execution
boundary (execution width `κ`) across five learned autobidders, three pacing
controllers, and three datasets.

| Block | Contents | Location |
| --- | --- | --- |
| **Core evaluator** | Single-advertiser *micro* replay + 48-advertiser synchronous *macro* market simulator + reference pacing policies | `code/auctionnet/` |
| **Autobidders** | Training code for CQL, IQL, DT, GAS, and SemBid | `code/bidders/` |
| **Pacing controllers** | traffic-aware, PID, dual (reference policies) | `code/auctionnet/micro/mandate_core_v2.py`, `code/common/reference_policies.py` |
| **Datasets** | Download and extraction scripts for AuctionNet, T9Sim (truth-referenced simulator), and iPinYou (logged auctions) | `data/` |
| **Cross-dataset analysis** | iPinYou/T9Sim aggregation, campaign-load and scenario-calibration analyses, dataset-scale accounting | `code/analysis/`, `plotting/data/load/`, `plotting/data/server_checked/` |
| **Figures & tables** | Source CSVs and plotting code for each exhibit | `plotting/` |
| **Frozen configs** | Experiment grids (κ widths, adoption fractions, seeds) | `configs/` |

---

## The five learned autobidders

| Bidder | Type | Source |
| --- | --- | --- |
| **CQL** | Conservative Q-Learning (torch.jit) | `code/bidders/sembid_cpa/` |
| **IQL** | Implicit Q-Learning (torch.jit) | `code/bidders/sembid_cpa/` |
| **DT** | Decision Transformer (`state_dim=16`, `K=20`) | `code/bidders/sembid_cpa/` |
| **GAS** | Gradient-Ascent Strategy (DT policy + reweighted-search critic) | `code/bidders/gas/` |
| **SemBid** | Language-bidding model | `code/bidders/sembid_cpa/` |

---

## The three pacing controllers

`traffic-aware`, `PID`, and `dual` pacing are implemented as small, auditable
reference policies in `code/common/reference_policies.py` and
`code/auctionnet/micro/mandate_core_v2.py`. The execution boundary then constrains the
bidder's native action relative to the reference action:

```text
enforce_action(raw, ref, κ)  →  clip to [max{0, ref·(1−κ)}, ref·(1+κ)]   (relative_hard)
```

---

## Repository layout

```text
├── code/
│   ├── auctionnet/          # core evaluator (micro + macro) + common reference policies
│   ├── bidders/             # five autobidders' training + inference source
│   ├── t9sim/               # T9Sim simulator package + replay driver
│   ├── ipinyou/             # iPinYou preprocess + logged-replay driver
│   └── analysis/            # cross-dataset aggregation + dataset-scale accounting
├── configs/                 # frozen experiment grids (macro/micro κ-width × adoption × seed)
├── data/
│   ├── auctionnet/          # download + convert scripts for AuctionNet
│   ├── t9sim/               # download + extract scripts for T9Sim (Zenodo)
│   └── ipinyou/             # download + extract scripts for iPinYou
├── plotting/
│   ├── data/derived/        # every figure/table source CSV (checked in)
│   └── scripts/server_release/  # build_derived_data.py + figure/table builders
└── docs/                    # reproduction guide + asset inventory
```

---

## Reproduction

The repository includes the scripts used for data preparation, training, and
evaluation, plus the source tables behind each exhibit.

Requirements: the three public datasets (links below) and trained bidder
checkpoints.

1. Download and extract the datasets with the scripts in `data/<dataset>/`.
2. Train the five bidders using the entrypoints in `code/bidders/`.
3. Run the evaluators in `code/auctionnet/`, `code/t9sim/`, and `code/ipinyou/`.
4. Run `build_derived_data.py` to turn the evaluation outputs into the source
   tables in `plotting/data/derived/`. `FIGURE_TABLE_CROSSWALK.md` maps each
   exhibit to its table.
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
| **AuctionNet** | Alimama auto-bidding track (periods 7–8, trajectory data) | 8 ZIPs on the Alimama OSS bucket — see `data/auctionnet/download_official_alimama.py` | Alimama competition terms |
| **T9Sim** | Truth-referenced RTB simulator (10 × 10M-impression seeds) | [Zenodo 21533031](https://zenodo.org/records/21533031) (DOI 10.5281/zenodo.21533031) | CC-BY-4.0 |
| **iPinYou** | Real logged RTB campaign data (19.5M records: 12.3M fitting + 3.1M calibration + 4.1M test) | contest 7z + season-2 zip — see `data/ipinyou/` scripts | iPinYou contest terms |

Each `data/<dataset>/` directory contains the corresponding download,
integrity-audit, and extraction scripts.

---

## License

The evaluation/analysis code in `code/` is released for academic reproduction.
Third-party components carry their own licenses: T9Sim (`code/t9sim/T9-simulator/`,
see its `LICENSE` / `LICENSE-DATA`), GAS (`code/bidders/gas/`), and the
iPinYou/AuctionNet datasets under their respective terms. See
`docs/REPRODUCTION.md` for per-component notes.
