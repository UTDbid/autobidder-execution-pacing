<div align="center">

# 广告活动投放节奏下的自动出价执行管理

*Managing Autobidder Execution under Campaign Pacing* · **代码与数据发布**（POM / Marketing Interface）

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#复现流程)
[![数据集](https://img.shields.io/badge/Datasets-AuctionNet%20T9Sim%20iPinYou-8A2BE2)](#数据集与下载链接)
[![English](https://img.shields.io/badge/English-README-2F81F7)](README.md)

</div>

---

> **端到端复现**——下载公开数据，用仓库自带的训练源码训练五个自动出价器，在三类 pacing 控制器下评估，并重建论文中的每一张图、每一张表。

**中文** · [**English**](README.md)

---

## 目录

- [本仓库包含什么](#overview)
- [五个学习型自动出价器](#bidders)
- [三个 pacing 控制器](#controllers)
- [仓库结构](#layout)
- [复现流程](#reproduction)
- [数据集与下载链接](#datasets)
- [许可](#license)

---

<a id="overview"></a>
## 本仓库包含什么

论文研究「受 pacing 控制的自动出价器」在出价执行上应被赋予多大的自由裁量权（`κ`）。
它在 **五个学习型自动出价器**、**三类 pacing 控制器**、**三个数据集** 之上，评估了一种
**Mandate** 机制 `(q_scale, κ, h)`——资源释放、执行边界/裁量、回顾时域：

| 模块 | 内容 | 位置 |
| --- | --- | --- |
| **核心评估器** | 单广告主*微观*重放 + 48 广告主同步*宏观*市场模拟器 + 参照 pacing 策略 | `code/auctionnet/` |
| **自动出价器** | CQL、IQL、DT、GAS、SemBid——训练源码随仓库打包于 `code/bidders/` | `code/bidders/` |
| **pacing 控制器** | traffic-aware、PID、dual（参照策略） | `code/auctionnet/micro/mandate_core_v2.py`、`code/common/reference_policies.py` |
| **数据集** | AuctionNet（Alimama AIGB）、T9Sim（真值参照模拟器）、iPinYou（日志拍卖）——下载与提取脚本 | `data/` |
| **跨数据集分析** | iPinYou/T9Sim 聚合、κ 选择挑战、数据集规模核算 | `code/analysis/`、`code/kappa_selection/` |
| **图表** | 论文每个图表展示的源 CSV + 画图代码 | `plotting/` |
| **冻结配置** | 精确实验网格（κ 宽度、采纳比例、随机种子） | `configs/` |

---

<a id="bidders"></a>
## 五个学习型自动出价器

| 出价器 | 类型 | 位置 |
| --- | --- | --- |
| **CQL** | Conservative Q-Learning（torch.jit） | `code/bidders/sembid_cpa/bidding_train_env/baseline/cql/cql.py` |
| **IQL** | Implicit Q-Learning（torch.jit） | `code/bidders/sembid_cpa/bidding_train_env/baseline/iql/iql.py` |
| **DT** | Decision Transformer（`state_dim=16`、`K=20`） | `code/bidders/sembid_cpa/bidding_train_env/baseline/dt/dt.py` |
| **GAS** | Gradient-Ascent Strategy（DT 策略 + 重加权搜索 critic） | `code/bidders/gas/` |
| **SemBid** | OpenLBM / Qwen2.5-0.5B 语言出价模型 | `code/bidders/sembid_cpa/Testing/test_exp23_standard.py` |

> 五个出价器均使用本仓库随附的训练源码训练（`docs/REPRODUCTION.md` §3），
> 评估器加载训练产生的 checkpoint。DT 与 SemBid 共用 `code/bidders/sembid_cpa/`
> 目录；GAS 单独打包。

---

<a id="controllers"></a>
## 三个 pacing 控制器

`traffic-aware`、`PID`、`dual` 三种 pacing 以纯、可审计的参照策略实现于
`code/common/reference_policies.py` 与 `code/auctionnet/micro/mandate_core_v2.py`。
Mandate 将出价器的*原始*提案相对参照动作进行约束：

```text
enforce_action(raw, ref, κ)  →  clip 到 [ref·(1−κ), ref·(1+κ)]   (relative_hard)
```

---

<a id="layout"></a>
## 仓库结构

```text
├── code/
│   ├── auctionnet/          # 核心评估器（微观 + 宏观）+ 通用参照策略
│   ├── bidders/             # 五个自动出价器的训练 + 推理源码
│   ├── t9sim/               # T9Sim 模拟器包 + 重放驱动
│   ├── ipinyou/             # iPinYou 预处理 + 日志重放驱动
│   ├── kappa_selection/     # κ 选择挑战（01_protocol/02_configs/03_code + 冻结 06_results）
│   └── analysis/            # 跨数据集聚合 + 数据集规模核算
├── configs/                 # 冻结实验网格（宏观/微观 κ 宽度 × 采纳比例 × 种子）
├── data/
│   ├── auctionnet/          # AuctionNet/AIGB 下载 + 转换脚本
│   ├── t9sim/               # T9Sim（Zenodo）下载 + 提取脚本
│   └── ipinyou/             # iPinYou 下载 + 提取脚本
├── plotting/
│   ├── data/derived/        # 每张图/表的源 CSV（已入库）
│   └── scripts/server_release/  # build_derived_data.py + 图表构建脚本
└── docs/                    # 复现指南 + 资产清单
```

---

<a id="reproduction"></a>
## 复现流程

端到端流程（数据 → 训练 → 评估 → 出图）见
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)：

```text
下载数据集  →  训练自动出价器  →  评估（微观/宏观、κ、T9Sim、iPinYou）
→  l01–l31 证据  →  build_derived_data.py  →  派生 CSV  →  build_all_figures.py
```

仅重建**图表**（无需数据、无需训练）：

```bash
pip install pandas numpy matplotlib
cd plotting/scripts/server_release
python build_all_figures.py      # → plotting/figures/{main,ec}/*.{pdf,svg,png}
```

即以**排版后的数字**从已入库的 `plotting/data/derived/` 复现论文图表，已实测端到端可运行。

---

<a id="datasets"></a>
## 数据集与下载链接

| 数据集 | 说明 | 下载 | 许可 / 条款 |
| --- | --- | --- | --- |
| **AuctionNet / AIGB** | Alimama 自动出价赛道（第 7–8 期、轨迹数据） | Alimama OSS 桶上的 8 个 ZIP——见 `data/auctionnet/download_official_alimama.py` | Alimama AIGB 竞赛条款 |
| **T9Sim** | 真值参照 RTB 模拟器（10 × 1000 万曝光种子） | [Zenodo 21533031](https://zenodo.org/records/21533031)（DOI 10.5281/zenodo.21533031） | CC-BY-4.0 |
| **iPinYou** | 真实日志 RTB 广告活动数据 | 竞赛 7z + season-2 zip——见 `data/ipinyou/` 脚本 | iPinYou 竞赛条款 |

每个 `data/<dataset>/` 目录都包含作者使用过的精确下载 + 完整性校验 + 提取脚本，
通过 `POMS_DATA_ROOT` 参数化，可在作者机器之外运行。

---

<a id="license"></a>
## 许可

`code/` 中的评估/分析代码以学术复现为目的发布。第三方组件遵循各自许可：
T9Sim（`code/t9sim/T9-simulator/`，见其 `LICENSE` / `LICENSE-DATA`）、GAS
（`code/bidders/gas/`），以及 iPinYou/AuctionNet 数据集遵循各自条款。
各组件说明见 `docs/REPRODUCTION.md`。
