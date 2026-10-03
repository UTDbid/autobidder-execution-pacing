#!/usr/bin/env python3
"""Build every manuscript-facing source table from the locked evidence package.

This script performs aggregation only. It does not run a bidder, simulator, or
market reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "evidence" / "locked"
OUT = ROOT / "data" / "derived"

BIDDERS = ["cql", "iql", "dt", "gas", "sembid"]
STRESS_BIDDERS = ["pid", "constant5"]
REFS = ["traffic_aware", "pid", "dual"]
REFS_MACRO = ["traffic", "pid", "dual"]
WIDTHS = ["0", "0.3", "0.5", "0.8", "1.0", "1.2", "Raw"]


FILES = {
    "micro_dev": "l01_auctionnet_micro_development.csv",
    "micro_later_pd": "l02_auctionnet_later_pid_dual.csv",
    "micro_later_t": "l03_auctionnet_later_traffic_aware.csv",
    "directional": "l04_auctionnet_directional.csv",
    "slack": "l05_auctionnet_opportunity_slack.csv",
    "micro_width": "l06_auctionnet_micro_aligned_widths.csv",
    "micro_k1": "l07_auctionnet_micro_kappa_1_summary.csv",
    "macro_uniform_raw": "l08_auctionnet_macro_uniform_raw.csv",
    "macro_uniform_summary": "l09_auctionnet_macro_uniform_summary.csv",
    "macro_rollout_raw": "l10_auctionnet_macro_rollout_episodes.csv",
    "macro_rollout_summary": "l11_auctionnet_macro_rollout_summary.csv",
    "ipinyou_raw": "l12_ipinyou_canonical_logged_policy_raw.csv",
    "ipinyou_contrasts": "l13_ipinyou_canonical_logged_policy_contrasts.csv",
    "t9_cells": "l14_t9sim_policy_cells.csv",
    "t9_exclusive": "l15_t9sim_exclusive_purchases.csv",
    "selection_cal": "l16_kappa_selection_calibration.csv",
    "selection_eval": "l17_kappa_selection_evaluation.csv",
    "selection_rules": "l18_kappa_selection_rules.csv",
    "selection_gates": "l19_kappa_selection_gates.csv",
    "selection_decision": "l20_kappa_selection_decision.json",
    "one_advertiser": "l21_auctionnet_one_advertiser_adoption_episodes.csv",
    "endpoint": "l22_endpoint_lottery_diagnostic.csv",
    "disagreement": "l23_useful_disagreement_diagnostic.csv",
    "tiers": "l24_transparent_service_tiers.csv",
    "sparse_1": "l25_auctionnet_sparse_feedback_period_1.csv",
    "sparse_2": "l26_auctionnet_sparse_feedback_period_2.csv",
    "reliability": "l27_auctionnet_measurement_reliability_and_cadence.csv",
    "response": "l28_auctionnet_response_aware_market_boundary.csv",
    "future_market": "l29_auctionnet_future_period_market_holdout.csv",
    "targeted": "l30_auctionnet_targeted_rollout_summary.csv",
    "scale": "l31_dataset_scale_accounting.json",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(LOCKED / FILES[name])


def save(frame: pd.DataFrame, name: str) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_pct(new: float, base: float) -> float:
    return np.nan if np.isclose(base, 0) else 100.0 * (new - base) / abs(base)


def build_design_tables() -> None:
    stats = json.loads((LOCKED / FILES["scale"]).read_text(encoding="utf-8"))
    micro7 = stats["auctionnet_micro"]["high_p7"]
    micro8 = stats["auctionnet_micro"]["high_p8"]
    macro11 = stats["auctionnet_macro"]["period11"]
    macro12 = stats["auctionnet_macro"]["period12"]
    ipy = stats["ipinyou"]
    t9 = stats["t9sim"]
    design = pd.DataFrame(
        [
            ["AuctionNet campaign", f"48 advertisers; {micro7['unique_pvs']:,}/{micro8['unique_pvs']:,} opportunities", "CRN seed × bidder × pacing", "Campaign performance–delivery frontier", "Primary campaign evidence"],
            ["AuctionNet market", f"48 synchronous advertisers; {macro11['unique_pvs']:,}/{macro12['unique_pvs']:,} opportunities", "Synchronous-market episode", "Value, prices, delivery, current auction revenue", "Primary market evidence"],
            ["T9Sim", f"{t9['underlying_rows_total']:,} auctions; {t9['generator_seeds']} generator seeds", "Generator seed × information view", "True value and purchase-set efficiency", "Truth-level mechanism"],
            ["iPinYou", f"{ipy['campaigns']} campaigns; {ipy['test_rows']:,} test log records", "Logged campaign", "Clicks, efficiency, spend, delivery", "Historical budget-pressure boundary"],
        ],
        columns=["Environment", "Scale", "Replication unit", "Primary outcomes", "Evidence role"],
    )
    save(design, "main/table1_research_design.csv")

    save(
        pd.DataFrame(
            [
                ["Campaign main", "5 learned bidders × 3 pacing references", "0, 0.3, 0.5, 0.8, 1.0, 1.2; Raw separate", "Fixed opportunity replay"],
                ["Campaign temporal", "5 learned bidders × 3 pacing references", "0 versus 0.8", "Independent later temporal block"],
                ["Market uniform", "CQL × 3 pacing references", "0, 0.3, 0.5, 0.8, 1.0, 1.2; Raw separate", "20 synchronous episodes"],
                ["Market rollout", "CQL × 3 pacing references × 4 paths", "25%, 50%, 75%, 100% adoption", "20 paired episodes per cell"],
                ["T9Sim", "3 pacing references × 4 information views × 3 budgets", "0, 0.3, 0.5, 0.8, 1.0, 1.2; Raw separate", "10 generator seeds"],
                ["iPinYou", "3 pacing references × 9 campaigns × 3 budget levels", "0, 0.3, 0.5, 0.8, 1.0, 1.2; Raw separate", "Logged-support replay"],
                ["κ-selection", "15 bidder–reference pairs × 3 service caps", "0, 0.3, 0.5, 0.8, 1.0", "Retrospective screen; STOP"],
            ],
            columns=["Component", "Design", "Width/adoption menu", "Replication"],
        ),
        "ec/table_a1_design_audit.csv",
    )
    save(design, "ec/table_a2_dataset_scale.csv")
    save(
        pd.DataFrame(
            [
                ["Performance–delivery frontier", "AuctionNet campaign", "Within bidder–reference width contrast", "Supported"],
                ["Current advertiser–platform wedge", "AuctionNet synchronous market", "Paired market-episode contrast", "Supported"],
                ["Downward selectivity is the main regular mechanism", "AuctionNet directional", "Down/up/symmetric decomposition", "Supported; not downward-only optimum"],
                ["Purchase-set quality depends on information and pacing", "T9Sim", "True-value exclusive-set contrast", "Supported conditionally"],
                ["Room for selection moderates value", "AuctionNet supply + iPinYou pressure", "Cross-design comparative statics", "Supported; not literal replication"],
                ["Universal waiting for later opportunities", "T9Sim timing", "Exclusive-set timing contrast", "Rejected"],
                ["Universal optimal κ=1.0", "Aligned width grid", "0.8-to-1.0 pair comparison", "Rejected"],
                ["Deployable κ-selection rule", "Retrospective challenge", "Regret and guardrail gates", "Not supported; STOP"],
            ],
            columns=["Claim", "Evidence", "Estimand", "Status"],
        ),
        "ec/table_a3_claim_crosswalk.csv",
    )
    save(
        pd.DataFrame(
            [
                ["Finite execution width", "0, 0.3, 0.5, 0.8, 1.0, 1.2", "Symmetric multiplicative boundary"],
                ["Raw endpoint", "Raw", "Projection bypass; not large finite κ"],
                ["Market uniform", "same finite menu + Raw", "All 48 advertisers use one width"],
                ["Rollout 1", "0→0.3", "First permission"],
                ["Rollout 2", "0.3→0.5", "Small dose expansion"],
                ["Rollout 3", "0.3→0.8", "Focal operating expansion"],
                ["Rollout 4", "0→0.8", "Direct transition from pacing-only"],
            ],
            columns=["Grid", "Nodes/path", "Interpretation"],
        ),
        "ec/table_a4_formal_grids.csv",
    )
    save(
        pd.DataFrame(
            [
                ["Traffic-aware", "Budget, time, traffic, historical value–price response", "No", "Bidder-independent pacing"],
                ["PID pacing", "Cumulative spend-path error", "No", "Delivery feedback"],
                ["Dual pacing", "Budget shadow price and spend feedback", "No", "Economic pacing"],
                ["Smoothed-controller", "Lagged raw bidder actions and spend", "Yes", "Information-overlap extension"],
                ["Response-aware", "Spend and CPA response", "No raw action", "Value-aware boundary"],
            ],
            columns=["Pacing architecture", "Information used", "Uses raw action", "Paper role"],
        ),
        "ec/table_a5_information_boundaries.csv",
    )
    save(
        pd.DataFrame(
            [
                ["Advertiser value", "Sum of expected acquisition value", "Higher is better", "Primary campaign/market outcome"],
                ["Aggregate CPA", "Spend divided by realized/expected acquisitions", "Lower is better", "Efficiency"],
                ["Underdelivery", "Unspent budget divided by assigned budget", "Lower is better", "Delivery discipline"],
                ["Current auction revenue", "Sum of simulated clearing payments", "Platform-side current yield", "Not long-run profit"],
                ["Clearing price", "Mean positive auction clearing price", "Competition pressure", "Market mediator"],
                ["Non-adopter value", "Value earned by untreated advertisers", "Spillover", "Undefined at full adoption"],
                ["True expected value", "Ground-truth expected value in T9Sim", "Higher is better", "Synthetic mechanism only"],
                ["Clicks", "Logged supported clicks in iPinYou replay", "Higher is better", "No unlogged counterfactual outcomes"],
            ],
            columns=["Outcome", "Definition", "Direction/meaning", "Scope"],
        ),
        "ec/table_a6_outcome_accounting.csv",
    )


def build_width_surfaces() -> None:
    """Aggregate the locked L06 width scan, including the native Raw endpoint."""
    aligned = read("micro_width")
    aligned["reference"] = aligned.reference.replace({"traffic": "traffic_aware"})
    raw = aligned[aligned.kappa_label.eq("Raw")].copy()
    expanded = [aligned[~aligned.kappa_label.eq("Raw")].copy(), raw]
    for ref in ["pid", "dual"]:
        rr = raw.copy()
        rr["reference"] = ref
        expanded.append(rr)
    full = pd.concat(expanded, ignore_index=True)
    base = full[np.isclose(full.kappa, 0)].set_index(["model", "reference", "seed"])
    records = []
    for r in full.itertuples(index=False):
        b = base.loc[(r.model, r.reference, r.seed)]
        records.append(
            {
                "bidder": r.model,
                "reference": r.reference,
                "seed": r.seed,
                "width": r.kappa_label,
                "width_numeric": np.inf if r.kappa_label == "Raw" else float(r.kappa),
                "value": r.total_expected_value,
                "value_delta": r.total_expected_value - b.total_expected_value,
                "relative_value_delta_pct": safe_pct(r.total_expected_value, b.total_expected_value),
                "cpa": r.aggregate_cpa,
                "cpa_delta": r.aggregate_cpa - b.aggregate_cpa,
                "underdelivery_pp": 100 * r.underdelivery,
                "underdelivery_delta_pp": 100 * (r.underdelivery - b.underdelivery),
            }
        )
    frontier_seed = pd.DataFrame(records)
    bidder_frontier = frontier_seed.groupby(["bidder", "reference", "width", "width_numeric"], as_index=False).agg(
        relative_value_delta_pct=("relative_value_delta_pct", "mean"),
        value_delta=("value_delta", "mean"),
        cpa_delta=("cpa_delta", "mean"),
        underdelivery_delta_pp=("underdelivery_delta_pp", "mean"),
        seeds=("seed", "nunique"),
    )
    ref_frontier = bidder_frontier.groupby(["reference", "width", "width_numeric"], as_index=False).agg(
        mean_relative_value_delta_pct=("relative_value_delta_pct", "mean"),
        sd_relative_value_delta_pct=("relative_value_delta_pct", "std"),
        mean_underdelivery_delta_pp=("underdelivery_delta_pp", "mean"),
        sd_underdelivery_delta_pp=("underdelivery_delta_pp", "std"),
        bidders=("bidder", "nunique"),
    )
    save(ref_frontier, "main/figure2a_campaign_frontier.csv")
    save(bidder_frontier, "ec/figure_r1_full_bidder_frontiers.csv")


def build_micro() -> None:
    dev = read("micro_dev")
    rows = []
    for bidder in BIDDERS:
        for ref in REFS:
            d = dev[(dev.controller == bidder) & (dev.reference == ref)]
            b = d[np.isclose(d.kappa, 0)].mean(numeric_only=True)
            w = d[np.isclose(d.kappa, 0.8)].mean(numeric_only=True)
            rows.append(
                {
                    "bidder": bidder,
                    "reference": ref,
                    "value_base": b.total_expected_value,
                    "value_wide": w.total_expected_value,
                    "value_delta": w.total_expected_value - b.total_expected_value,
                    "cpa_base": b.cpa,
                    "cpa_wide": w.cpa,
                    "cpa_delta": w.cpa - b.cpa,
                    "underdelivery_base_pp": 100 * b.underdelivery,
                    "underdelivery_wide_pp": 100 * w.underdelivery,
                    "underdelivery_delta_pp": 100 * (w.underdelivery - b.underdelivery),
                    "seeds": d.seed.nunique(),
                }
            )
    dev_table = pd.DataFrame(rows)
    save(dev_table, "main/table2a_reference_development_evaluation.csv")

    later = pd.concat([read("micro_later_pd"), read("micro_later_t")], ignore_index=True)
    rows = []
    for bidder in BIDDERS:
        for ref in REFS:
            d = later[(later.model == bidder) & (later.reference == ref) & later.kappa.isin([0, 0.8])]
            b = d[np.isclose(d.kappa, 0)].mean(numeric_only=True)
            w = d[np.isclose(d.kappa, 0.8)].mean(numeric_only=True)
            seed_delta = []
            for _, s in d.groupby("seed"):
                seed_delta.append(s[np.isclose(s.kappa, 0.8)].iloc[0].total_expected_value - s[np.isclose(s.kappa, 0)].iloc[0].total_expected_value)
            rows.append(
                {
                    "bidder": bidder,
                    "reference": ref,
                    "value_base": b.total_expected_value,
                    "value_wide": w.total_expected_value,
                    "value_delta": w.total_expected_value - b.total_expected_value,
                    "cpa_base": b.aggregate_cpa,
                    "cpa_wide": w.aggregate_cpa,
                    "cpa_delta": w.aggregate_cpa - b.aggregate_cpa,
                    "underdelivery_base_pp": 100 * b.underdelivery,
                    "underdelivery_wide_pp": 100 * w.underdelivery,
                    "underdelivery_delta_pp": 100 * (w.underdelivery - b.underdelivery),
                    "positive_seed_share": np.mean(np.asarray(seed_delta) > 0),
                    "seeds": d.seed.nunique(),
                }
            )
    later_table = pd.DataFrame(rows)
    save(later_table, "main/table2b_later_temporal_evaluation.csv")
    save(later_table, "main/figure2b_temporal_scatter.csv")
    save(later_table, "ec/table_b2_complete_temporal.csv")

    build_width_surfaces()

    # Architecture matrix at the focal width, including stress bidders and
    # information-overlap pacing references.
    arch_rows = []
    for (bidder, ref, seed), d in dev.groupby(["controller", "reference", "seed"]):
        if not ({0.0, 0.8} <= set(np.round(d.kappa, 6))):
            continue
        b = d[np.isclose(d.kappa, 0)].iloc[0]
        w = d[np.isclose(d.kappa, 0.8)].iloc[0]
        arch_rows.append({"bidder": bidder, "reference": ref, "seed": seed, "value_delta": w.total_expected_value - b.total_expected_value, "cpa_delta": w.cpa - b.cpa, "underdelivery_delta_pp": 100 * (w.underdelivery - b.underdelivery)})
    arch = pd.DataFrame(arch_rows).groupby(["bidder", "reference"], as_index=False).agg(value_delta=("value_delta", "mean"), cpa_delta=("cpa_delta", "mean"), underdelivery_delta_pp=("underdelivery_delta_pp", "mean"), seeds=("seed", "nunique"))
    arch["bidder_role"] = np.where(arch.bidder.isin(BIDDERS), "learned", "stress")
    arch["reference_role"] = np.where(arch.reference.isin(REFS), "regular", "information-overlap")
    save(arch, "ec/table_b1_architecture_scope.csv")
    save(arch, "ec/figure_r2_architecture_heatmap.csv")


def build_mechanism_and_boundaries() -> None:
    d = read("directional")
    base = d[d.condition.eq("reference")].set_index(["reference", "seed"])
    rows = []
    for r in d[~d.condition.eq("reference") & d.reference.isin(base.index.get_level_values(0))].itertuples(index=False):
        b = base.loc[(r.reference, r.seed)]
        rows.append({
            "reference": r.reference,
            "condition": r.condition,
            "seed": r.seed,
            "value_delta": r.total_expected_value - b.total_expected_value,
            "cpa_delta": r.aggregate_cpa - b.aggregate_cpa,
            "underdelivery_delta_pp": 100 * (r.underdelivery - b.underdelivery),
            "lower_binding_delta_pp": 100 * (r.lower_projection_rate - b.lower_projection_rate),
            "upper_binding_delta_pp": 100 * (r.upper_projection_rate - b.upper_projection_rate),
        })
    direction = pd.DataFrame(rows).groupby(["reference", "condition"], as_index=False).agg(
        value_delta=("value_delta", "mean"), cpa_delta=("cpa_delta", "mean"), underdelivery_delta_pp=("underdelivery_delta_pp", "mean"), lower_binding_delta_pp=("lower_binding_delta_pp", "mean"), upper_binding_delta_pp=("upper_binding_delta_pp", "mean"), positive_seed_share=("value_delta", lambda x: np.mean(np.asarray(x) > 0)), seeds=("seed", "nunique"))
    save(direction, "main/figure4a_directional.csv")
    save(direction, "ec/table_c1_directional.csv")
    save(direction, "ec/figure_r3_directional.csv")

    secondary = pd.concat([
        read("endpoint").assign(panel="endpoint_lottery"),
        read("disagreement").assign(panel="useful_disagreement"),
        read("tiers").assign(panel="transparent_tiers"),
    ], ignore_index=True, sort=False)
    save(secondary, "ec/table_c2_secondary_diagnostics.csv")

    slack = read("slack")
    base = slack[np.isclose(slack.kappa, 0)].set_index(["model", "reference", "budget_multiplier", "opportunity_supply", "hash_order", "seed"])
    srows = []
    for r in slack[np.isclose(slack.kappa, 0.8) & slack.policy_mode.eq("hard")].itertuples(index=False):
        b = base.loc[(r.model, r.reference, r.budget_multiplier, r.opportunity_supply, r.hash_order, r.seed)]
        srows.append({"model": r.model, "reference": r.reference, "budget_multiplier": r.budget_multiplier, "opportunity_supply": r.opportunity_supply, "hash_order": r.hash_order, "seed": r.seed, "value_delta": r.total_expected_value - b.total_expected_value, "cpa_delta": r.aggregate_cpa - b.aggregate_cpa, "underdelivery_delta_pp": 100 * (r.underdelivery - b.underdelivery)})
    s = pd.DataFrame(srows)
    summary = s.groupby(["model", "reference", "budget_multiplier", "opportunity_supply"], as_index=False).agg(value_delta=("value_delta", "mean"), cpa_delta=("cpa_delta", "mean"), underdelivery_delta_pp=("underdelivery_delta_pp", "mean"), seed_hash_cells=("value_delta", "size"), positive_cell_share=("value_delta", lambda x: np.mean(np.asarray(x) > 0)))
    main_supply = summary[(summary.model == "cql") & summary.reference.isin(REFS) & np.isclose(summary.budget_multiplier, 1.0)].copy()
    save(main_supply, "main/figure5a_auctionnet_supply.csv")
    save(summary[(summary.model == "cql") & summary.reference.isin(REFS)], "ec/table_d1_opportunity_slack.csv")
    save(summary[summary.model.isin(STRESS_BIDDERS) & summary.reference.isin(REFS)], "ec/table_d2_slack_stress.csv")
    save(summary[(summary.model == "cql") & summary.reference.isin(REFS)], "ec/figure_r4_opportunity_slack.csv")

    sparse = pd.concat([read("sparse_1").assign(temporal_block="sparse_block_1"), read("sparse_2").assign(temporal_block="sparse_block_2")], ignore_index=True)
    sparse = sparse[sparse.model.isin(BIDDERS + STRESS_BIDDERS) & sparse.reference.isin(["pid", "dual"]) & np.isclose(sparse.kappa, 0.8)].copy()
    save(sparse, "ec/table_d3_sparse_feedback.csv")
    save(sparse[sparse.model.isin(BIDDERS)], "ec/figure_r5_sparse_feedback.csv")
    save(read("reliability"), "ec/table_e1_reliability.csv")


def build_macro() -> None:
    uniform = read("macro_uniform_summary").copy()
    zeros = []
    for ref in REFS_MACRO:
        zeros.append({"reference": ref, "width": "0", "episodes": 20, "mean_relative_total_base_expected_value_pct": 0.0, "mean_relative_platform_revenue_pct": 0.0, "mean_delta_aggregate_cpa": 0.0, "mean_delta_underdelivery": 0.0, "mean_relative_mean_clearing_price_all_pct": 0.0})
    uniform = pd.concat([pd.DataFrame(zeros), uniform], ignore_index=True, sort=False)
    save(uniform, "main/figure3ab_uniform_width.csv")

    rollout = read("macro_rollout_summary")
    save(rollout, "main/figure3c_rollout_paths.csv")
    # Combine the one-advertiser pilot with the aligned 25%--100% rollout
    # cells to provide a complete adoption path.
    one_advertiser_raw = read("one_advertiser")
    one_advertiser = one_advertiser_raw.groupby(["reference", "alpha"], as_index=False).agg(
        advertiser_value_delta_pct=("relative_total_base_expected_value_pct", "mean"),
        revenue_delta_pct=("relative_platform_revenue_pct", "mean"),
        clearing_price_delta_pct=("relative_mean_clearing_price_all_pct", "mean"),
        cpa_delta=("delta_aggregate_cpa", "mean"),
        underdelivery_delta_pp=("delta_underdelivery", lambda x: 100 * x.mean()),
        nonadopter_value_delta=("delta_nonadopter_expected_value", "mean"),
        episodes=("market_episode_id", "nunique"),
    )
    aligned = rollout[rollout.rollout_path.eq("0.3_to_0.8")][[
        "reference", "alpha", "mean_relative_total_base_expected_value_pct",
        "mean_relative_platform_revenue_pct", "mean_relative_mean_clearing_price_all_pct",
        "mean_delta_aggregate_cpa", "mean_delta_underdelivery",
        "mean_delta_nonadopter_expected_value", "episodes",
    ]].rename(columns={
        "mean_relative_total_base_expected_value_pct": "advertiser_value_delta_pct",
        "mean_relative_platform_revenue_pct": "revenue_delta_pct",
        "mean_relative_mean_clearing_price_all_pct": "clearing_price_delta_pct",
        "mean_delta_aggregate_cpa": "cpa_delta",
        "mean_delta_underdelivery": "underdelivery_delta_pp",
        "mean_delta_nonadopter_expected_value": "nonadopter_value_delta",
    })
    aligned["underdelivery_delta_pp"] = 100 * aligned["underdelivery_delta_pp"]
    adoption = pd.concat([one_advertiser, aligned], ignore_index=True).sort_values(["reference", "alpha"])
    save(adoption, "main/figure3d_adoption_frontier.csv")

    raw = read("macro_rollout_raw")
    focal = raw[(raw.rollout_path == "0.3_to_0.8") & np.isclose(raw.alpha, 0.5)]
    rows = []
    for ref in REFS_MACRO:
        d = focal[focal.reference == ref]
        rows.append({
            "reference": ref,
            "advertiser_value_base": d.total_base_expected_value_baseline.mean(),
            "advertiser_value_adopt": d.total_base_expected_value.mean(),
            "advertiser_value_delta_pct": d.relative_total_base_expected_value_pct.mean(),
            "cpa_base": d.aggregate_cpa_baseline.mean(),
            "cpa_adopt": d.aggregate_cpa.mean(),
            "cpa_delta": d.delta_aggregate_cpa.mean(),
            "underdelivery_base_pp": 100 * d.underdelivery_baseline.mean(),
            "underdelivery_adopt_pp": 100 * d.underdelivery.mean(),
            "underdelivery_delta_pp": 100 * d.delta_underdelivery.mean(),
            "revenue_base": d.platform_revenue_baseline.mean(),
            "revenue_adopt": d.platform_revenue.mean(),
            "revenue_delta_pct": d.relative_platform_revenue_pct.mean(),
            "clearing_price_base": d.mean_clearing_price_all_baseline.mean(),
            "clearing_price_adopt": d.mean_clearing_price_all.mean(),
            "clearing_price_delta_pct": d.relative_mean_clearing_price_all_pct.mean(),
            "nonadopter_value_base": d.nonadopter_expected_value_baseline.mean(),
            "nonadopter_value_adopt": d.nonadopter_expected_value.mean(),
            "nonadopter_value_delta": d.delta_nonadopter_expected_value.mean(),
            "episodes": d.market_episode_id.nunique(),
        })
    save(pd.DataFrame(rows), "main/table3_market_consequences_50pct.csv")

    full_macro = pd.concat([uniform.assign(panel="uniform_width"), rollout.assign(panel="rollout_paths")], ignore_index=True, sort=False)
    save(full_macro, "ec/table_f1_complete_macro.csv")
    save(rollout, "ec/figure_r6_complete_rollouts.csv")
    response = read("response")
    response_summary = response.groupby("reference", as_index=False).agg(value_delta_pct=("relative_total_base_expected_value_pct", "mean"), cpa_delta=("delta_aggregate_cpa", "mean"), underdelivery_delta_pp=("delta_underdelivery", lambda x: 100 * x.mean()), revenue_delta_pct=("relative_platform_revenue_pct", "mean"), price_delta_pct=("relative_mean_clearing_price_all_pct", "mean"), episodes=("market_episode_id", "nunique"))
    f2 = pd.concat([response_summary.assign(panel="response_aware_market"), read("future_market").assign(panel="future_temporal_market")], ignore_index=True, sort=False)
    save(f2, "ec/table_f2_macro_extensions.csv")
    target = read("targeted")
    save(target, "ec/table_g1_targeted_rollout.csv")
    save(target, "ec/figure_r7_targeted_rollout.csv")


def build_t9_and_ipinyou() -> None:
    cells = read("t9_cells")
    focus = cells[cells.reference.isin(REFS) & cells.condition.isin(["C1", "C2", "C3", "C4"]) & cells.kappa.isin([0, 0.8])]
    exclusive = read("t9_exclusive")
    rows = []
    for budget in [0.5, 1.0, 1.5]:
        for ref in REFS:
            for view in ["C1", "C2", "C3", "C4"]:
                d = focus[(focus.reference == ref) & (focus.condition == view) & np.isclose(focus.budget_multiplier, budget)]
                b = d[np.isclose(d.kappa, 0)].mean(numeric_only=True)
                w = d[np.isclose(d.kappa, 0.8)].mean(numeric_only=True)
                ex = exclusive[(exclusive.reference == ref) & (exclusive.condition == view) & np.isclose(exclusive.budget_multiplier, budget)].set_index("metric")
                rows.append({
                    "reference": ref,
                    "budget_multiplier": budget,
                    "information_view": view,
                    "true_value_base": b.expected_value,
                    "true_value_wide": w.expected_value,
                    "true_value_delta": w.expected_value - b.expected_value,
                    "expected_roas_base": b.expected_roas,
                    "expected_roas_wide": w.expected_roas,
                    "expected_roas_delta": w.expected_roas - b.expected_roas,
                    "underdelivery_base_pp": 100 * b.underdelivery,
                    "underdelivery_wide_pp": 100 * w.underdelivery,
                    "underdelivery_delta_pp": 100 * (w.underdelivery - b.underdelivery),
                    "exclusive_efficiency_delta": ex.loc["delta_true_efficiency_mean", "mean"],
                    "exclusive_timing_delta_steps": ex.loc["delta_mean_step", "mean"],
                    "positive_seed_share_true_value": np.mean((d[np.isclose(d.kappa, 0.8)].set_index("seed").expected_value - d[np.isclose(d.kappa, 0)].set_index("seed").expected_value) > 0),
                    "seeds": d.seed.nunique(),
                })
    t9_full = pd.DataFrame(rows)
    t9 = t9_full[np.isclose(t9_full.budget_multiplier, 1.0)].copy()
    save(t9, "main/table4_t9sim_truth_performance.csv")
    save(t9[["reference", "information_view", "true_value_delta"]], "main/figure4b_t9_true_value.csv")
    save(t9[["reference", "information_view", "exclusive_efficiency_delta", "exclusive_timing_delta_steps"]], "main/figure4c_t9_purchase_efficiency.csv")
    save(t9_full, "ec/table_i1_t9sim_full.csv")
    save(t9_full, "ec/figure_r9_t9sim_truth.csv")

    ipy = read("ipinyou_raw")
    rows = []
    for ref in REFS:
        for budget in [0.03125, 0.125, 0.5]:
            d = ipy[(ipy.reference == ref) & np.isclose(ipy.budget_fraction, budget) & ipy.kappa.isin([0, 0.8])]
            b = d[np.isclose(d.kappa, 0)].mean(numeric_only=True)
            w = d[np.isclose(d.kappa, 0.8)].mean(numeric_only=True)
            rows.append({
                "reference": ref,
                "budget_fraction": budget,
                "campaigns": d.campaign.nunique(),
                "clicks_base": b.clicks,
                "clicks_wide": w.clicks,
                "clicks_delta": w.clicks - b.clicks,
                "ecpc_base": b.ecpc,
                "ecpc_wide": w.ecpc,
                "ecpc_delta": w.ecpc - b.ecpc,
                "spend_base": b.spend,
                "spend_wide": w.spend,
                "spend_delta": w.spend - b.spend,
                "underdelivery_base_pp": 100 * b.underdelivery,
                "underdelivery_wide_pp": 100 * w.underdelivery,
                "underdelivery_delta_pp": 100 * (w.underdelivery - b.underdelivery),
                "mean_won_pctr_base": b.mean_won_pctr,
                "mean_won_pctr_wide": w.mean_won_pctr,
                "mean_won_pctr_delta": w.mean_won_pctr - b.mean_won_pctr,
            })
    ipy_table = pd.DataFrame(rows)
    save(ipy_table, "main/table5_ipinyou_logged_performance.csv")
    save(ipy_table[["reference", "budget_fraction", "clicks_delta", "ecpc_delta", "underdelivery_delta_pp"]], "main/figure5b_ipinyou_pressure.csv")
    contrasts = read("ipinyou_contrasts")
    camp = contrasts[np.isclose(contrasts.kappa, 0.8) & contrasts.reference.isin(REFS) & contrasts.budget_fraction.isin([0.03125, 0.125, 0.5])].copy()
    save(camp, "ec/table_h1_ipinyou_campaigns.csv")
    save(camp, "ec/figure_r8_ipinyou_heterogeneity.csv")


def build_selection() -> None:
    rules = read("selection_rules")
    gates = read("selection_gates")
    decision = json.loads((LOCKED / FILES["selection_decision"]).read_text(encoding="utf-8"))
    rules["decision"] = decision.get("outcome", "STOP")
    save(rules, "ec/figure_r10_kappa_selection.csv")
    save(pd.concat([rules.assign(panel="performance"), gates.assign(panel="success_gates")], ignore_index=True, sort=False), "ec/table_j1_kappa_selection.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_design_tables()
    build_micro()
    build_mechanism_and_boundaries()
    build_macro()
    build_t9_and_ipinyou()
    build_selection()
    outputs = sorted(str(p.relative_to(ROOT)) for p in OUT.rglob("*.csv"))
    (ROOT / "qa" / "DERIVED_DATA_INVENTORY.json").write_text(json.dumps({"status": "PASS", "files": len(outputs), "outputs": outputs}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(outputs)} derived source tables")


if __name__ == "__main__":
    main()
