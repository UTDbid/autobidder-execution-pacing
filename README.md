<div align="center">

# Managing Autobidder Execution under Campaign Pacing

**Code & Data Release** · *POM / Marketing Interface*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#reproduction)
[![Datasets](https://img.shields.io/badge/Datasets-AuctionNet%20T9Sim%20iPinYou-8A2BE2)](#datasets-and-download-links)
[![中文文档](https://img.shields.io/badge/中文-README_CN-EB4B4B)](README_CN.md)

</div>

---

**English** · [**中文**](README_CN.md)

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

The paper studies how much *discretion* (`κ`) a pacing-controlled autobidder
should be given over bid execution. It evaluates a **Mandate** mechanism
`(q_scale, κ, h)` — resource release, execution boundary/discretion, review
horizon — on top of **five learned autobidders** under **three pacing
controllers**, across **three datasets**:

| Block | Contents | Location |
| --- | --- | --- |
| **Core evaluator** | Single-advertiser *micro* replay + 48-advertiser synchronous *macro* market simulator + reference pacing policies | `code/auctionnet/` |
| **Autobidders** | CQL, IQL, DT, GAS, SemBid — training source bundled under `code/bidders/` | `code/bidders/` |
| **Pacing controllers** | traffic-aware, PID, dual (reference policies) | `code/auctionnet/micro/mandate_core_v2.py`, `code/common/reference_policies.py` |
| **Datasets** | AuctionNet (Alimama AIGB), T9Sim (truth-referenced simulator), iPinYou (logged auctions) — setup + extraction scripts | `data/` |
| **Cross-dataset analysis** | iPinYou/T9Sim aggregation, κ-selection challenge, dataset-scale accounting | `code/analysis/`, `code/kappa_selection/` |
| **Figures & tables** | Every manuscript exhibit's source CSV + plotting code | `plotting/` |
| **Frozen configs** | The exact experiment grids (κ widths, adoption fractions, seeds) | `configs/` |

---

## The five learned autobidders

| Bidder | Type | Where it lives |
| --- | --- | --- |
| **CQL** | Conservative Q-Learning (torch.jit) | `code/bidders/sembid_cpa/bidding_train_env/baseline/cql/cql.py` |
| **IQL** | Implicit Q-Learning (torch.jit) | `code/bidders/sembid_cpa/bidding_train_env/baseline/iql/iql.py` |
| **DT** | Decision Transformer (`state_dim=16`, `K=20`) | `code/bidders/sembid_cpa/bidding_train_env/baseline/dt/dt.py` |
| **GAS** | Gradient-Ascent Strategy (DT policy + reweighted-search critic) | `code/bidders/gas/` |
| **SemBid** | OpenLBM / Qwen2.5-0.5B language-bidding model | `code/bidders/sembid_cpa/Testing/test_exp23_standard.py` |

> All five bidders are trained from source bundled in this repository
> (`docs/REPRODUCTION.md` §3); the evaluator loads the resulting checkpoints.
> DT and SemBid share the `code/bidders/sembid_cpa/` tree; GAS is bundled
> separately.

---

## The three pacing controllers

`traffic-aware`, `PID`, and `dual` pacing are implemented as pure, auditable
reference policies in `code/common/reference_policies.py` and
`code/auctionnet/micro/mandate_core_v2.py`. The Mandate then constrains the
bidder's *raw* proposal relative to the reference action:

```text
enforce_action(raw, ref, κ)  →  clip to [ref·(1−κ), ref·(1+κ)]   (relative_hard)
```

---

## Repository layout

```text
├── code/
│   ├── auctionnet/          # core evaluator (micro + macro) + common reference policies
│   ├── bidders/             # five autobidders' training + inference source
│   ├── t9sim/               # T9Sim simulator package + replay driver
│   ├── ipinyou/             # iPinYou preprocess + logged-replay driver
│   ├── kappa_selection/     # κ-selection challenge (01_protocol/02_configs/03_code + frozen 06_results)
│   └── analysis/            # cross-dataset aggregation + dataset-scale accounting
├── configs/                 # frozen experiment grids (macro/micro κ-width × adoption × seed)
├── data/
│   ├── auctionnet/          # download + convert scripts for AuctionNet/AIGB
│   ├── t9sim/               # download + extract scripts for T9Sim (Zenodo)
│   └── ipinyou/             # download + extract scripts for iPinYou
├── plotting/
│   ├── data/derived/        # every figure/table source CSV (checked in)
│   └── scripts/server_release/  # build_derived_data.py + figure/table builders
└── docs/                    # reproduction guide + asset inventory
```

---

## Reproduction

The full end-to-end flow (data → train → evaluate → build figures) is in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md):

```text
download datasets  →  train autobidders  →  evaluate (micro/macro, κ, T9Sim, iPinYou)
→  l01–l31 evidence  →  build_derived_data.py  →  derived CSVs  →  build_all_figures.py
```

To rebuild **just the figures and tables** from the checked-in source tables
(no data, no training):

```bash
pip install pandas numpy matplotlib
cd plotting/scripts/server_release
python build_all_figures.py      # → plotting/figures/{main,ec}/*.{pdf,svg,png}
```

This reproduces the manuscript's figures/tables **as typeset numbers** from the
bundled `plotting/data/derived/`. Verified to run end-to-end.

---

## Datasets and download links

| Dataset | What it is | Download | License / terms |
| --- | --- | --- | --- |
| **AuctionNet / AIGB** | Alimama auto-bidding track (periods 7–8, trajectory data) | 8 ZIPs on the Alimama OSS bucket — see `data/auctionnet/download_official_alimama.py` | Alimama AIGB competition terms |
| **T9Sim** | Truth-referenced RTB simulator (10 × 10M-impression seeds) | [Zenodo 21533031](https://zenodo.org/records/21533031) (DOI 10.5281/zenodo.21533031) | CC-BY-4.0 |
| **iPinYou** | Real logged RTB campaign data | contest 7z + season-2 zip — see `data/ipinyou/` scripts | iPinYou contest terms |

Each `data/<dataset>/` directory contains the exact download + integrity-audit
+ extraction scripts used by the authors, parameterized via `POMS_DATA_ROOT`
so they run outside the authors' machines.

---

## License

The evaluation/analysis code in `code/` is released for academic reproduction.
Third-party components carry their own licenses: T9Sim (`code/t9sim/T9-simulator/`,
see its `LICENSE` / `LICENSE-DATA`), GAS (`code/bidders/gas/`), and the
iPinYou/AuctionNet datasets under their respective terms. See
`docs/REPRODUCTION.md` for per-component notes.
