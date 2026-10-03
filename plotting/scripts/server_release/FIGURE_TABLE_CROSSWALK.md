# Manuscript Figure/Table Crosswalk

This file maps every final manuscript exhibit to its rendered asset and source data. The paper is organized by operating question rather than by dataset. All assets are self-contained within this release.

## Main-text figures

| Label | Role in the paper | Final asset | Source data |
|---|---|---|---|
| Figure 1 | Pacing–bidder–execution architecture and evidence map | `figures/main/fig1_operating_architecture.pdf` | conceptual; specification in `scripts/fig_main_1.py` |
| Figure 2 | AuctionNet campaign performance–delivery frontier and later temporal matrix | `figures/main/fig2_campaign_frontier.pdf` | `derived/main/figure2a_campaign_frontier.csv`; `derived/main/figure2b_temporal_scatter.csv` |
| Figure 3 | AuctionNet uniform market width, rollout paths, and adoption frontier | `figures/main/fig3_market_width_and_rollout.pdf` | `derived/main/figure3ab_uniform_width.csv`; `figure3c_rollout_paths.csv`; `figure3d_adoption_frontier.csv` |
| Figure 4 | AuctionNet directional diagnostic plus T9Sim truth-level purchase reallocation | `figures/main/fig4_purchase_reallocation.pdf` | `derived/main/figure4a_directional.csv`; `figure4b_t9_true_value.csv`; `figure4c_t9_purchase_efficiency.csv` |
| Figure 5 | AuctionNet opportunity supply plus iPinYou budget pressure | `figures/main/fig5_room_for_selection.pdf` | `derived/main/figure5a_auctionnet_supply.csv`; `figure5b_ipinyou_pressure.csv` |

Figure 3D combines the focal one-advertiser pilot at $\kappa=0.8$ with the aligned $0.3\rightarrow0.8$ rollout nodes at 25%--100% adoption. The bundled pilot evidence contains exactly the three reported reference cells and 20 episodes per reference.

## Main-text tables

| Label | Role in the paper | Standalone asset | Source data |
|---|---|---|---|
| Table 1 | Research design and evidence roles | `tables/main/table1_research_design.pdf` | `derived/main/table1_research_design.csv` |
| Table 2 | AuctionNet 5 bidders × 3 pacing architectures in two temporal blocks | `tables/main/table2_auctionnet_campaign_performance.pdf` | `derived/main/table2a_reference_development_evaluation.csv`; `table2b_later_temporal_evaluation.csv` |
| Table 3 | AuctionNet advertiser, delivery, revenue, price, and spillover outcomes at 50% adoption | `tables/main/table3_auctionnet_market_50pct.pdf` | `derived/main/table3_market_consequences_50pct.csv` |
| Table 4 | T9Sim true value, ROAS, delivery, and exclusive-purchase diagnostics | `tables/main/table4_t9sim_truth_performance.pdf` | `derived/main/table4_t9sim_truth_performance.csv` |
| Table 5 | iPinYou clicks, spend, eCPC, and delivery under budget pressure | `tables/main/table5_ipinyou_logged_performance.pdf` | `derived/main/table5_ipinyou_logged_performance.csv` |

The standalone PDFs are review copies. The same final numbers are typeset natively in `manuscript.tex`, and `scripts/audit_manuscript_alignment.py` checks every numeric row against the CSVs.

## E-Companion figures

| EC figure | Final asset | Source data |
|---|---|---|
| R1 | `figures/ec/fig_r1_full_bidder_frontiers.pdf` | `derived/ec/figure_r1_full_bidder_frontiers.csv` |
| R2 | `figures/ec/fig_r2_reference_bidder_heatmap.pdf` | `derived/ec/figure_r2_architecture_heatmap.csv` |
| R3 | `figures/ec/fig_r3_directional_decomposition.pdf` | `derived/ec/figure_r3_directional.csv` |
| R4 | `figures/ec/fig_r4_opportunity_slack.pdf` | `derived/ec/figure_r4_opportunity_slack.csv` |
| R5 | `figures/ec/fig_r5_sparse_feedback.pdf` | `derived/ec/figure_r5_sparse_feedback.csv` |
| R6 (Figure EC.8) | `figures/ec/fig_r6_complete_market_rollouts_above.pdf` | `data/derived/ec/figure_r6_complete_rollouts.csv` |
| R7 | `figures/ec/fig_r7_targeted_rollout.pdf` | `derived/ec/figure_r7_targeted_rollout.csv` |
| R8 | `figures/ec/fig_r8_ipinyou_campaign_heterogeneity.pdf` | `derived/ec/figure_r8_ipinyou_heterogeneity.csv` |
| R9 | `figures/ec/fig_r9_t9sim_truth_mechanism.pdf` | `derived/ec/figure_r9_t9sim_truth.csv` |
| R10 | `figures/ec/fig_r10_kappa_selection_challenge.pdf` | `derived/ec/figure_r10_kappa_selection.csv` |

## E-Companion tables

| EC table | Final asset or native source | Source data |
|---|---|---|
| A1 | `tables/ec/table_a1_design_audit.pdf` | `derived/ec/table_a1_design_audit.csv` |
| A2 | native in `manuscript.tex`; standalone `tables/ec/table_a2_dataset_scale.pdf` | `derived/ec/table_a2_dataset_scale.csv` |
| A3 | native in `manuscript.tex`; standalone `tables/ec/table_a3_claim_crosswalk.pdf` | `derived/ec/table_a3_claim_crosswalk.csv` |
| A4 | native in `manuscript.tex`; standalone `tables/ec/table_a4_formal_grids.pdf` | `derived/ec/table_a4_formal_grids.csv` |
| A5 | native in `manuscript.tex`; standalone `tables/ec/table_a5_information_boundaries.pdf` | `derived/ec/table_a5_information_boundaries.csv` |
| A6 | native in `manuscript.tex`; standalone `tables/ec/table_a6_outcome_accounting.pdf` | `derived/ec/table_a6_outcome_accounting.csv` |
| B1 | `tables/ec/table_b1_architecture_scope.pdf` (2 pages) | `derived/ec/table_b1_architecture_scope.csv` |
| B2 | `tables/ec/table_b2_complete_temporal.pdf` | `derived/ec/table_b2_complete_temporal.csv` |
| C1 | `tables/ec/table_c1_directional.pdf` | `derived/ec/table_c1_directional.csv` |
| C2 | `tables/ec/table_c2_secondary_diagnostics.pdf` | `derived/ec/table_c2_secondary_diagnostics.csv` |
| D1 | `tables/ec/table_d1_opportunity_slack.pdf` (2 pages) | `derived/ec/table_d1_opportunity_slack.csv` |
| D2 | `tables/ec/table_d2_slack_stress.pdf` (2 pages) | `derived/ec/table_d2_slack_stress.csv` |
| D3 | `tables/ec/table_d3_sparse_feedback.pdf` | `derived/ec/table_d3_sparse_feedback.csv` |
| E1 | `tables/ec/table_e1_reliability.pdf` (2 pages) | `derived/ec/table_e1_reliability.csv` |
| F1 | `tables/ec/table_f1_complete_macro.pdf` (4 pages) | `derived/ec/table_f1_complete_macro.csv` |
| F2 | `tables/ec/table_f2_macro_extensions.pdf` | `derived/ec/table_f2_macro_extensions.csv` |
| G1 | `tables/ec/table_g1_targeted_rollout.pdf` | `derived/ec/table_g1_targeted_rollout.csv` |
| H1 | `tables/ec/table_h1_ipinyou_campaigns.pdf` (4 pages) | `derived/ec/table_h1_ipinyou_campaigns.csv` |
| I1 | `tables/ec/table_i1_t9sim_full.pdf` (2 pages) | `derived/ec/table_i1_t9sim_full.csv` |
| J1 | `tables/ec/table_j1_kappa_selection.pdf` (2 pages) | `derived/ec/table_j1_kappa_selection.csv` |

## Build and audit order

1. `python scripts/verify_locked_evidence.py`
2. `python scripts/build_derived_data.py`
3. render figures/tables with the project Python environment listed in `README.md`
4. `tectonic manuscript.tex`
5. `python scripts/verify_release.py`
6. `python scripts/audit_manuscript_alignment.py`

No command in this crosswalk trains a bidder, reruns a replay, or reconstructs a synchronous market.
