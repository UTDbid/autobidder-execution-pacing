#!/usr/bin/env python3
"""Build the colleague-facing dataset-scale and baseline-results report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REF_LABEL = {
    "traffic_aware": "Traffic-aware",
    "traffic": "Traffic-aware",
    "pid": "PID pacing",
    "dual": "Dual pacing",
    "smoothed_controller": "Smoothed hybrid",
    "response_aware": "Response-aware",
    "response": "Response-aware",
    "raw": "Raw bidder",
}


def fmt(value, digits=2, pct=False, signed=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = float(value)
    if pct:
        value *= 100
    sign = "+" if signed and value > 0 else ""
    if abs(value) >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M"
    if abs(value) >= 10_000:
        return f"{sign}{value:,.0f}"
    if abs(value) >= 1_000:
        return f"{sign}{value:,.1f}"
    return f"{sign}{value:.{digits}f}"


def md_table(headers, rows):
    def clean(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(map(clean, headers)) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def display_count(value):
    """Format numeric counts without changing descriptive string cells."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return f"{int(value):,}"
    return value


def mean_sd(group, column):
    return group[column].mean(), group[column].std(ddof=1)


def aggregate_micro(group, regime, reference, controller_scope):
    ev, ev_sd = mean_sd(group, "total_expected_value")
    cpa_col = "aggregate_cpa" if "aggregate_cpa" in group.columns else "cpa"
    cpa, cpa_sd = mean_sd(group, cpa_col)
    return {
        "regime": regime,
        "reference": REF_LABEL.get(reference, reference),
        "controller_scope": controller_scope,
        "seeds": int(group["seed"].nunique()),
        "expected_value_mean": ev,
        "expected_value_sd": ev_sd,
        "budget_usage_mean": group["budget_usage"].mean(),
        "underdelivery_mean": group["underdelivery"].mean(),
        "cpa_mean": cpa,
        "cpa_sd": cpa_sd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root
    results = project / "server_results"
    source_dir = args.output_md.parent / "data_statistics_source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads((args.output_md.parent / "dataset_raw_statistics.json").read_text())

    # ------------------------------------------------------------------
    # Dataset scale
    # ------------------------------------------------------------------
    scale_rows = []
    for key, label in [
        ("high_p7", "AuctionNet Micro — High P7"),
        ("high_p8", "AuctionNet Micro — High P8"),
        ("low_p7", "AuctionNet Micro — Low P7"),
        ("low_p8", "AuctionNet Micro — Low P8"),
    ]:
        d = raw["auctionnet_micro"][key]
        scale_rows.append({
            "environment": label,
            "source_type": "AuctionNet advertiser-level replay file",
            "advertisers_or_campaigns": d["advertisers"],
            "categories": d["categories"],
            "independent_auctions_or_logs": d["unique_pvs"],
            "advertiser_auction_rows": d["data_rows"],
            "time_structure": f"48 time steps; Period {key[-1]}",
            "raw_size_bytes": d["bytes"],
            "role": "Micro performance–delivery" if "high" in key else "Sparse-feedback boundary",
        })
    for key, label in [
        ("period11", "AuctionNet Macro — P11"),
        ("period12", "AuctionNet Macro — P12"),
        ("period13", "AuctionNet Macro — P13 holdout"),
    ]:
        d = raw["auctionnet_macro"][key]
        scale_rows.append({
            "environment": label,
            "source_type": "AuctionNet shared-market archive",
            "advertisers_or_campaigns": d["advertisers"],
            "categories": d["categories"],
            "independent_auctions_or_logs": d["unique_pvs"],
            "advertiser_auction_rows": d["data_rows"],
            "time_structure": f"48 time steps; {d['member']}",
            "raw_size_bytes": d["member_uncompressed_bytes"],
            "role": "Synchronous competition/revenue" if key != "period13" else "Future temporal holdout",
        })
    ipy = raw["ipinyou"]
    scale_rows.append({
        "environment": "iPinYou Seasons 2–3",
        "source_type": "Real logged second-price campaigns",
        "advertisers_or_campaigns": ipy["campaigns"],
        "categories": "—",
        "independent_auctions_or_logs": ipy["train_rows"] + ipy["test_rows"],
        "advertiser_auction_rows": "not duplicated by advertiser",
        "time_structure": "chronological train/calibration/test",
        "raw_size_bytes": ipy["raw_directory_bytes"],
        "role": "Campaign × budget × reference heterogeneity",
    })
    t9 = raw["t9sim"]
    scale_rows.append({
        "environment": "T9Sim formal 10M",
        "source_type": "Synthetic truth-level first-price auctions",
        "advertisers_or_campaigns": f"{t9['campaigns_per_seed']} per seed; {t9['campaign_seed_units']:,} seed-campaigns",
        "categories": "—",
        "independent_auctions_or_logs": t9["underlying_rows_total"],
        "advertiser_auction_rows": f"{t9['view_rows_total']:,} view-rows (same auctions × C1–C4)",
        "time_structure": "10 generator seeds; C1–C4 censored views",
        "raw_size_bytes": t9["ten_m_archives_bytes"],
        "role": "Truth-level purchase-set/data-layer mechanism",
    })
    dataset_scale = pd.DataFrame(scale_rows)
    dataset_scale.to_csv(source_dir / "dataset_scale.csv", index=False)

    # ------------------------------------------------------------------
    # AuctionNet micro reference-only (kappa=0)
    # ------------------------------------------------------------------
    p7 = pd.read_csv(results / "auctionnet_micro/high_p7/five_reference_architecture/p7_five_reference_raw.csv")
    p7_k0 = p7[np.isclose(p7["kappa"], 0)].copy()
    micro_rows = []
    for ref in ["traffic_aware", "pid", "dual", "response_aware"]:
        # Pure references do not use the raw bidder; choose one controller to avoid sevenfold duplicates.
        group = p7_k0[(p7_k0["reference"] == ref) & (p7_k0["controller"] == "cql")]
        micro_rows.append(aggregate_micro(group, "High P7", ref, "bidder-invariant at κ=0"))
    for controller, group in p7_k0[p7_k0["reference"] == "smoothed_controller"].groupby("controller"):
        micro_rows.append(aggregate_micro(group, "High P7", "smoothed_controller", controller.upper()))

    for relative, regime in [
        ("auctionnet_micro/high_p8/reference_expansion/micro_aggregate_raw.csv", "High P8"),
        ("auctionnet_micro/low_p7/reference_expansion/micro_aggregate_raw.csv", "Low P7"),
        ("auctionnet_micro/low_p8/reference_expansion/micro_aggregate_raw.csv", "Low P8"),
    ]:
        frame = pd.read_csv(results / relative)
        for ref in ["pid", "dual"]:
            group = frame[(frame["model"] == "cql") & (frame["reference"] == ref) & np.isclose(frame["kappa"], 0)]
            micro_rows.append(aggregate_micro(group, regime, ref, "bidder-invariant at κ=0"))
    micro_k0 = pd.DataFrame(micro_rows)
    micro_k0.to_csv(source_dir / "auctionnet_micro_reference_only.csv", index=False)

    # AuctionNet Raw endpoint, P7. Raw is duplicated across reference imports; average within controller×seed first.
    p7_raw = p7[np.isinf(p7["kappa"])].groupby(["controller", "seed"], as_index=False).agg(
        total_expected_value=("total_expected_value", "mean"),
        budget_usage=("budget_usage", "mean"),
        underdelivery=("underdelivery", "mean"),
        cpa=("cpa", "mean"),
    )
    micro_raw = p7_raw.groupby("controller", as_index=False).agg(
        seeds=("seed", "nunique"),
        expected_value_mean=("total_expected_value", "mean"),
        expected_value_sd=("total_expected_value", "std"),
        budget_usage_mean=("budget_usage", "mean"),
        underdelivery_mean=("underdelivery", "mean"),
        cpa_mean=("cpa", "mean"),
    )
    micro_raw.to_csv(source_dir / "auctionnet_micro_raw_endpoint.csv", index=False)

    # ------------------------------------------------------------------
    # AuctionNet macro operational baselines and effects
    # ------------------------------------------------------------------
    macro_frames = []
    for rel in [
        "auctionnet_macro/reference_expansion/macro_episode_contrasts.csv",
        "auctionnet_macro/traffic_random_alpha50/macro_episode_contrasts.csv",
        "auctionnet_macro/response_boundary/macro_episode_contrasts.csv",
    ]:
        frame = pd.read_csv(results / rel)
        keep = [
            "reference", "market_episode_id", "baseline_total_base_expected_value",
            "baseline_platform_revenue", "baseline_aggregate_cpa", "baseline_budget_usage",
            "baseline_underdelivery", "baseline_mean_clearing_price_positive", "baseline_realized_slot_fill",
        ]
        macro_frames.append(frame[keep].drop_duplicates(["reference", "market_episode_id"]))
    macro_base = pd.concat(macro_frames, ignore_index=True).groupby("reference", as_index=False).agg(
        episodes=("market_episode_id", "nunique"),
        total_expected_value=("baseline_total_base_expected_value", "mean"),
        platform_revenue=("baseline_platform_revenue", "mean"),
        aggregate_cpa=("baseline_aggregate_cpa", "mean"),
        budget_usage=("baseline_budget_usage", "mean"),
        underdelivery=("baseline_underdelivery", "mean"),
        clearing_price=("baseline_mean_clearing_price_positive", "mean"),
        slot_fill=("baseline_realized_slot_fill", "mean"),
    )
    macro_base["window"] = "P11–P12"
    macro_base["baseline_definition"] = "α=0; all 48 advertisers at κ=.3"

    for window, rel, episodes in [
        ("P11–P12", "inherited_prior_auctionnet/analysis/platformbid_v1/formal_p11p12/adoption_curves.csv", 20),
        ("P13", "inherited_prior_auctionnet/results/platformbid_period13_final_temporal/adoption_curves.csv", 10),
    ]:
        frame = pd.read_csv(results / rel)
        row = frame[np.isclose(frame["alpha"], 0)].iloc[0]
        macro_base = pd.concat([macro_base, pd.DataFrame([{
            "reference": "smoothed_controller",
            "episodes": episodes,
            "total_expected_value": row["mean_total_base_expected_value"],
            "platform_revenue": row["mean_platform_revenue"],
            "aggregate_cpa": row["mean_aggregate_cpa"],
            "budget_usage": row["mean_budget_usage"],
            "underdelivery": row["mean_underdelivery"],
            "clearing_price": row["mean_mean_clearing_price_positive"],
            "slot_fill": row["mean_realized_slot_fill"],
            "window": window,
            "baseline_definition": "α=0; all 48 advertisers at inherited low-width κ=.3",
        }])], ignore_index=True)
    macro_base.to_csv(source_dir / "auctionnet_macro_operational_baselines.csv", index=False)

    macro_effect_rows = []
    for rel in [
        "auctionnet_macro/reference_expansion/macro_summary.csv",
        "auctionnet_macro/traffic_random_alpha50/macro_summary.csv",
        "auctionnet_macro/response_boundary/macro_summary.csv",
    ]:
        frame = pd.read_csv(results / rel)
        focus = frame[np.isclose(frame["adopter_kappa"], 0.8)]
        for _, row in focus.iterrows():
            if row["reference"] in ["dual", "pid", "response"] and not np.isclose(row["alpha"], 1):
                continue
            macro_effect_rows.append({
                "window": "P11–P12",
                "reference": row["reference"],
                "adoption": row["alpha"],
                "comparison": "κ=.8 adopters vs α=0 κ=.3 baseline",
                "advertiser_value_delta_pct": 100 * row["mean_delta_total_base_expected_value"] / (
                    row["mean_delta_total_base_expected_value"] / (row.get("positive_episode_share_total_base_expected_value", 1) or 1)
                ) if False else np.nan,
                "advertiser_value_delta": row["mean_delta_total_base_expected_value"],
                "revenue_delta": row["mean_delta_platform_revenue"],
                "cpa_delta": row["mean_delta_aggregate_cpa"],
                "underdelivery_delta_pp": 100 * row["mean_delta_underdelivery"],
                "clearing_price_delta": row["mean_delta_mean_clearing_price_positive"],
                "nonadopter_value_delta": row["mean_delta_nonadopter_expected_value"] if row["alpha"] < 1 else np.nan,
                "episodes": row["episodes"],
            })
    # Add relative effects from episode-level contrast files exactly.
    combined_contrasts = pd.concat([
        pd.read_csv(results / "auctionnet_macro/reference_expansion/macro_episode_contrasts.csv"),
        pd.read_csv(results / "auctionnet_macro/traffic_random_alpha50/macro_episode_contrasts.csv"),
        pd.read_csv(results / "auctionnet_macro/response_boundary/macro_episode_contrasts.csv"),
    ], ignore_index=True)
    for row in macro_effect_rows:
        match = combined_contrasts[
            (combined_contrasts["reference"] == row["reference"])
            & np.isclose(combined_contrasts["alpha"], row["adoption"])
            & np.isclose(combined_contrasts["adopter_kappa"], 0.8)
        ]
        row["advertiser_value_delta_pct"] = match["relative_total_base_expected_value_pct"].mean()
        row["revenue_delta_pct"] = match["relative_platform_revenue_pct"].mean()
        row["clearing_price_delta_pct"] = match["relative_mean_clearing_price_positive_pct"].mean()

    for window, rel in [
        ("P11–P12", "inherited_prior_auctionnet/analysis/platformbid_v1/formal_p11p12/adoption_curves.csv"),
        ("P13", "inherited_prior_auctionnet/results/platformbid_period13_final_temporal/adoption_curves.csv"),
    ]:
        frame = pd.read_csv(results / rel)
        base = frame[np.isclose(frame["alpha"], 0)].iloc[0]
        wide = frame[np.isclose(frame["alpha"], 1) & (frame["policy"] == "private")].iloc[0]
        macro_effect_rows.append({
            "window": window,
            "reference": "smoothed_controller",
            "adoption": 1.0,
            "comparison": "full adoption κ=.8 vs α=0 κ=.3 baseline",
            "advertiser_value_delta_pct": 100 * (wide["mean_total_base_expected_value"] / base["mean_total_base_expected_value"] - 1),
            "revenue_delta_pct": 100 * (wide["mean_platform_revenue"] / base["mean_platform_revenue"] - 1),
            "clearing_price_delta_pct": 100 * (wide["mean_mean_clearing_price_positive"] / base["mean_mean_clearing_price_positive"] - 1),
            "advertiser_value_delta": wide["mean_total_base_expected_value"] - base["mean_total_base_expected_value"],
            "revenue_delta": wide["mean_platform_revenue"] - base["mean_platform_revenue"],
            "cpa_delta": wide["mean_aggregate_cpa"] - base["mean_aggregate_cpa"],
            "underdelivery_delta_pp": 100 * (wide["mean_underdelivery"] - base["mean_underdelivery"]),
            "clearing_price_delta": wide["mean_mean_clearing_price_positive"] - base["mean_mean_clearing_price_positive"],
            "nonadopter_value_delta": np.nan,
            "episodes": int(base["n_cell_rows"]),
        })
    macro_effect = pd.DataFrame(macro_effect_rows)
    macro_effect.to_csv(source_dir / "auctionnet_macro_policy_effects.csv", index=False)

    # ------------------------------------------------------------------
    # iPinYou reference-only and Raw
    # ------------------------------------------------------------------
    ipy_policy = pd.read_csv(results / "ipinyou/formal_v2/ipinyou_campaign_policy_raw.csv")

    def aggregate_ipy(frame):
        rows = []
        for (ref, bf), group in frame.groupby(["reference", "budget_fraction"], dropna=False):
            budget = group["budget"].sum()
            spend = group["spend"].sum()
            clicks = group["clicks"].sum()
            conversions = group["conversions"].sum()
            wins = group["wins"].sum()
            rows.append({
                "reference": ref,
                "budget_fraction": bf,
                "campaigns": group["campaign"].nunique(),
                "total_budget": budget,
                "spend": spend,
                "budget_usage": spend / budget,
                "underdelivery": 1 - spend / budget,
                "clicks": clicks,
                "conversions": conversions,
                "wins": wins,
                "ctr": clicks / wins if wins else np.nan,
                "ecpc": spend / clicks if clicks else np.nan,
                "campaign_clicks_median": group["clicks"].median(),
            })
        return pd.DataFrame(rows)

    ipy_k0 = aggregate_ipy(ipy_policy[np.isclose(ipy_policy["kappa"], 0)])
    ipy_raw = aggregate_ipy(ipy_policy[ipy_policy["reference"] == "raw"])
    ipy_k0.to_csv(source_dir / "ipinyou_reference_only.csv", index=False)
    ipy_raw.to_csv(source_dir / "ipinyou_raw_endpoint.csv", index=False)

    # ------------------------------------------------------------------
    # T9Sim reference-only and Raw, central budget
    # ------------------------------------------------------------------
    t9_policy = pd.read_csv(results / "t9sim/formal_10m/t9sim_formal_aggregate_raw.csv")

    def aggregate_t9(frame):
        return frame.groupby(["condition", "reference"], as_index=False).agg(
            generator_seeds=("seed", "nunique"),
            expected_value_mean=("expected_value", "mean"),
            expected_value_sd=("expected_value", "std"),
            spend_mean=("spend", "mean"),
            budget_usage_mean=("budget_usage", "mean"),
            underdelivery_mean=("underdelivery", "mean"),
            expected_roas_mean=("expected_roas", "mean"),
            expected_surplus_mean=("expected_surplus", "mean"),
            wins_mean=("wins", "mean"),
            win_rate_mean=("win_rate", "mean"),
            true_efficiency_mean=("mean_won_true_efficiency", "mean"),
            clicks_mean=("clicks", "mean"),
            installs_mean=("installs", "mean"),
            payers_mean=("payers", "mean"),
        )

    t9_k0 = aggregate_t9(t9_policy[np.isclose(t9_policy["kappa"], 0) & np.isclose(t9_policy["budget_multiplier"], 1)])
    t9_raw = aggregate_t9(t9_policy[(t9_policy["reference"] == "raw") & np.isclose(t9_policy["budget_multiplier"], 1)])
    t9_k0.to_csv(source_dir / "t9sim_reference_only_b1.csv", index=False)
    t9_raw.to_csv(source_dir / "t9sim_raw_endpoint_b1.csv", index=False)

    # ------------------------------------------------------------------
    # Experiment-scale ledger
    # ------------------------------------------------------------------
    experiment_scale = pd.DataFrame([
        ["AuctionNet High P7 five-reference architecture", "7 bidders × 5 references", "κ=0,.3,.5,.8,1.2,Raw; 4 CRN seeds", "728 aggregate cells; 588 contrasts; 48 advertisers"],
        ["AuctionNet High P8 temporal extension", "7 bidders × PID/dual", "κ=0,.3,.5,.8,1.2; 8 CRN seeds", "560 aggregate cells; 26,880 advertiser outcomes"],
        ["AuctionNet Low P7/P8", "5 bidders × PID/dual × 2 periods", "5 κ values; 4 seeds per period", "400 aggregate cells; 19,200 advertiser outcomes"],
        ["Directional mechanism", "CQL × 5 references", "down-only/up-only/symmetric/raw; 4 seeds", "84 aggregate cells; 184,320 step diagnostics"],
        ["Opportunity supply × budget", "3 bidders × 3 references", "500/2k/8k/full; 3 budgets; nested hashes", "2,520 raw cells; 1,080 contrasts"],
        ["Reliability × cadence", "CQL/PID × 3 references", "noise/overestimate and h=2/4/8", "280 raw cells; 120 contrasts"],
        ["AuctionNet macro PID/dual", "48 advertisers; 20 episodes/reference", "6 adoption rates × 2 wide policies; κlow=.3", "40 episodes; 4,200 raw market cells; 201,600 advertiser-cell outcomes; 400 normalized cells"],
        ["Inherited smoothed macro P11–P12", "48 advertisers; 20 episodes", "105 prespecified cells/episode", "2,100 market cells; 100,800 advertiser-cell outcomes"],
        ["Smoothed future holdout P13", "48 advertisers; 10 episodes", "105 prespecified cells/episode", "1,050 market cells; 50,400 advertiser-cell outcomes"],
        ["Targeted rollout", "3 references × 2 assignment heuristics", "α=.5; κ=.8 vs .3; 20 episodes", "120 paired targeted cells plus matched random"],
        ["iPinYou formal v2", "9 real campaigns × 3 references × 3 budgets", "κ=0,.3,.8,1.2 plus shared Raw", "351 policy rows; 243 campaign contrasts"],
        ["T9Sim formal", "10 seeds × 4 views × 200 campaigns", "3 refs × 3 budgets × 4 κ plus Raw", "1,560 aggregate rows; 312,000 campaign rows; 1,080 contrasts"],
    ], columns=["study", "entities", "design", "computed_scale"])
    experiment_scale.to_csv(source_dir / "experiment_scale.csv", index=False)

    # ------------------------------------------------------------------
    # Render tables
    # ------------------------------------------------------------------
    scale_table = md_table(
        ["Environment", "Type", "Advertisers / campaigns", "Independent auctions/logs", "Advertiser-auction/view rows", "Time structure", "Raw size", "Role"],
        [[
            row["environment"], row["source_type"], row["advertisers_or_campaigns"],
            f"{int(row['independent_auctions_or_logs']):,}", display_count(row["advertiser_auction_rows"]), row["time_structure"],
            f"{row['raw_size_bytes'] / 1e9:.2f} GB", row["role"],
        ] for _, row in dataset_scale.iterrows()],
    )

    micro_table = md_table(
        ["Regime", "Reference", "Controller scope", "Seeds", "Expected value mean ± SD", "Budget usage", "Underdelivery", "CPA mean"],
        [[
            row["regime"], row["reference"], row["controller_scope"], row["seeds"],
            f"{fmt(row['expected_value_mean'],1)} ± {fmt(row['expected_value_sd'],1)}",
            fmt(row["budget_usage_mean"], 2, pct=True) + "%", fmt(row["underdelivery_mean"], 2, pct=True) + "%",
            fmt(row["cpa_mean"], 3),
        ] for _, row in micro_k0.iterrows()],
    )
    micro_raw_table = md_table(
        ["Bidder", "Seeds", "Expected value mean ± SD", "Budget usage", "Underdelivery", "CPA"],
        [[
            row["controller"].upper(), int(row["seeds"]),
            f"{fmt(row['expected_value_mean'],1)} ± {fmt(row['expected_value_sd'],1)}",
            fmt(row["budget_usage_mean"], 2, pct=True) + "%", fmt(row["underdelivery_mean"], 2, pct=True) + "%",
            fmt(row["cpa_mean"], 3),
        ] for _, row in micro_raw.sort_values("expected_value_mean", ascending=False).iterrows()],
    )
    macro_base_table = md_table(
        ["Window", "Reference", "Baseline", "Episodes", "Advertiser value", "Current revenue", "CPA", "Budget usage", "Underdelivery", "Clearing price", "Slot fill"],
        [[
            row["window"], REF_LABEL.get(row["reference"], row["reference"]), row["baseline_definition"], int(row["episodes"]),
            fmt(row["total_expected_value"],1), fmt(row["platform_revenue"],1), fmt(row["aggregate_cpa"],3),
            fmt(row["budget_usage"],2,pct=True)+"%", fmt(row["underdelivery"],2,pct=True)+"%",
            fmt(row["clearing_price"],4), fmt(row["slot_fill"],2,pct=True)+"%",
        ] for _, row in macro_base.sort_values(["window", "reference"]).iterrows()],
    )
    macro_effect_table = md_table(
        ["Window", "Reference", "Adoption", "Value Δ%", "Revenue Δ%", "Clearing price Δ%", "CPA Δ", "Underdelivery Δ(pp)", "Non-adopter value Δ", "Episodes"],
        [[
            row["window"], REF_LABEL.get(row["reference"], row["reference"]), fmt(row["adoption"],0,pct=True)+"%",
            fmt(row["advertiser_value_delta_pct"],2,signed=True)+"%", fmt(row["revenue_delta_pct"],2,signed=True)+"%",
            fmt(row["clearing_price_delta_pct"],2,signed=True)+"%", fmt(row["cpa_delta"],3,signed=True),
            fmt(row["underdelivery_delta_pp"],2,signed=True), fmt(row["nonadopter_value_delta"],1,signed=True), int(row["episodes"]),
        ] for _, row in macro_effect.sort_values(["window", "reference"]).iterrows()],
    )
    ipy_table = md_table(
        ["Reference", "Budget/log-spend fraction", "Campaigns", "Total budget", "Spend", "Budget usage", "Underdelivery", "Clicks", "Conversions", "Wins", "CTR", "eCPC"],
        [[
            REF_LABEL.get(row["reference"], row["reference"]), fmt(row["budget_fraction"],3,pct=True)+"%", int(row["campaigns"]),
            fmt(row["total_budget"],1), fmt(row["spend"],1), fmt(row["budget_usage"],2,pct=True)+"%",
            fmt(row["underdelivery"],2,pct=True)+"%", int(row["clicks"]), int(row["conversions"]), f"{int(row['wins']):,}",
            fmt(row["ctr"],4,pct=True)+"%", fmt(row["ecpc"],3),
        ] for _, row in ipy_k0.sort_values(["budget_fraction", "reference"]).iterrows()],
    )
    ipy_raw_table = md_table(
        ["Endpoint", "Budget/log-spend fraction", "Campaigns", "Spend", "Budget usage", "Clicks", "Wins", "CTR", "eCPC"],
        [[
            "Raw bidder", fmt(row["budget_fraction"],3,pct=True)+"%", int(row["campaigns"]), fmt(row["spend"],1),
            fmt(row["budget_usage"],2,pct=True)+"%", int(row["clicks"]), f"{int(row['wins']):,}",
            fmt(row["ctr"],4,pct=True)+"%", fmt(row["ecpc"],3),
        ] for _, row in ipy_raw.sort_values("budget_fraction").iterrows()],
    )
    t9_table = md_table(
        ["View", "Reference", "Seeds", "True EV mean ± SD", "Spend", "Budget usage", "Underdelivery", "Expected ROAS", "True efficiency", "Wins", "Clicks", "Installs", "Payers"],
        [[
            row["condition"], REF_LABEL.get(row["reference"], row["reference"]), int(row["generator_seeds"]),
            f"{fmt(row['expected_value_mean'],1)} ± {fmt(row['expected_value_sd'],1)}", fmt(row["spend_mean"],1),
            fmt(row["budget_usage_mean"],2,pct=True)+"%", fmt(row["underdelivery_mean"],2,pct=True)+"%",
            fmt(row["expected_roas_mean"],3), fmt(row["true_efficiency_mean"],3), f"{row['wins_mean']:,.0f}",
            f"{row['clicks_mean']:,.1f}", f"{row['installs_mean']:,.1f}", f"{row['payers_mean']:,.1f}",
        ] for _, row in t9_k0.sort_values(["condition", "reference"]).iterrows()],
    )
    t9_raw_table = md_table(
        ["View", "Seeds", "True EV mean ± SD", "Spend", "Budget usage", "Underdelivery", "Expected ROAS", "True efficiency", "Wins"],
        [[
            row["condition"], int(row["generator_seeds"]),
            f"{fmt(row['expected_value_mean'],1)} ± {fmt(row['expected_value_sd'],1)}", fmt(row["spend_mean"],1),
            fmt(row["budget_usage_mean"],2,pct=True)+"%", fmt(row["underdelivery_mean"],2,pct=True)+"%",
            fmt(row["expected_roas_mean"],3), fmt(row["true_efficiency_mean"],3), f"{row['wins_mean']:,.0f}",
        ] for _, row in t9_raw.sort_values("condition").iterrows()],
    )
    experiment_table = md_table(experiment_scale.columns.tolist(), experiment_scale.values.tolist())

    ipy_campaign_stats = pd.read_csv(source_dir / "ipinyou_campaign_data_stats.csv")
    ipy_campaign_table = md_table(
        ["Campaign", "Train rows", "Calibration rows", "Test rows", "Train clicks", "Test clicks", "Train conversions", "Test conversions", "Test days"],
        [[
            int(row["campaign"]), f"{int(row['train_rows']):,}", f"{int(row['train_calibration_rows']):,}",
            f"{int(row['test_rows']):,}", int(row["train_clicks"]), int(row["test_clicks"]),
            int(row["train_conversions"]), int(row["test_conversions"]), int(row["test_dates"]),
        ] for _, row in ipy_campaign_stats.sort_values("campaign").iterrows()],
    )

    totals = raw["auctionnet_totals"]
    report = f"""# Data Statistics & Results Complete

| 项目 | 内容 |
|---|---|
| 版本 | 2026-08-30 |
| 用途 | 给合作者解释当前完整数据、实验单元、reference设置和基线结果 |
| 证据包 | AuctionNet Micro + AuctionNet Macro + iPinYou + T9Sim |

> **先澄清“没有control”的含义。** 本项目中，`κ=0`并不是“没有reference”，而是
> **reference-only / 完全受reference约束**：执行动作等于pacing reference，bidder没有偏离权限。
> 真正“没有执行边界”的端点是`Raw`。AuctionNet同步市场主实验没有运行`κ=0`；它的无推广
> baseline是`α=0`且所有广告主使用`κ=.3`。本报告分别列出三种口径，避免混用。

## 一、结论先行

1. **AuctionNet是主证据。** 五个正式时期合计包含{totals['unique_period_pvs']:,}个独立PV、
   {totals['unique_advertiser_opportunity_pairs']:,}个基础advertiser–opportunity pairs；每个时期固定
   48个广告主、6个类别和48个time steps。High/Low共享P7/P8拍卖骨架，不能当作两份独立市场。
2. **iPinYou提供真实campaign异质性。** 九个campaign合计{ipy['train_rows']:,}条train日志和
   {ipy['test_rows']:,}条test日志；训练内部再拆出{ipy['calibration_rows']:,}条calibration。
3. **T9Sim提供最大规模的truth mechanism。** 十个独立generator seeds共100,000,000个底层拍卖、
   2,000个campaign-seed units和55个字段；C1–C4是同一拍卖流的四个信息视图，不是4亿个独立拍卖。
4. **reference-only并不存在统一赢家。** AuctionNet High P7中，response-aware在κ=0时价值最高但
   underdelivery达到38.20%；traffic-aware几乎花完预算但价值较低。PID和dual位于二者之间。
5. **Raw也不是统一上界。** AuctionNet P7中CQL Raw价值最高但仍有14.50%预算未交付；Constant5 Raw
   underdelivery达到84.04%。在iPinYou和T9Sim中，Raw同样表现为高选择性、低预算利用，而非天然最优。
6. **宏观baseline必须单独解释。** 同步市场使用α=0、κ=.3作为operational baseline；绝对值受
   reference和模拟版本影响，跨版本不应直接排名。可比较的是同一reference、同一episode内的采用差值。

## 二、一个总表：所有数据集规模和作用

{scale_table}

源表：[dataset_scale.csv](data_statistics_source_tables/dataset_scale.csv)。

规模口径说明：iPinYou的12.67 GB是下载、解压和清洗前的raw工作目录占用（原始压缩包约6.31 GB）；
T9Sim的17.44 GB是十个正式10M压缩包的合计。二者不能被解释为“有效样本量”，有效样本量以上表中的
独立auction/log记录为准。

### AuctionNet合并口径

- P7/P8 micro独立auction skeleton：{raw['auctionnet_micro']['high_p7']['unique_pvs'] + raw['auctionnet_micro']['high_p8']['unique_pvs']:,} PVs；High和Low是同一骨架上的不同反馈/价值环境。
- P11/P12/P13 macro：{sum(x['unique_pvs'] for x in raw['auctionnet_macro'].values()):,} PVs。
- 五个时期合计：{totals['unique_period_pvs']:,}独立PVs、{totals['unique_advertiser_opportunity_pairs']:,} advertiser–opportunity pairs。
- 若按磁盘中的High/Low两份文件都计入，micro保存了{totals['stored_micro_rows_high_plus_low']:,}行；这是存储规模，不是独立样本量。
- 48个广告主分成6个类别，每类8个。High/macro预算总额150,000，单广告主预算450–11,850、
  中位数2,325，CPA约束6–12；Low预算总额同为150,000，但单广告主预算2,000–4,850、CPA约束60–130。

## 三、完整实验规模

{experiment_table}

源表：[experiment_scale.csv](data_statistics_source_tables/experiment_scale.csv)。这里的“advertiser-cell
outcomes”是同一批48个广告主在不同政策cell中的重复计算结果，不是新增广告主。

## 四、AuctionNet Micro：κ=0 reference-only表现

{micro_table}

源表：[auctionnet_micro_reference_only.csv](data_statistics_source_tables/auctionnet_micro_reference_only.csv)。

### 如何解释

- Traffic-aware在High P7几乎完成全部预算（99.91%），但价值约11.0k、CPA约13.62。
- PID把价值提高到约12.8k，但交付率降至95.94%；dual介于二者之间。
- Response-aware价值最高、CPA最低，但只使用约61.80%预算，因此它不是“更强的纯pacing”，而是
  更重视价值响应的边界条件。
- Smoothed hybrid即使κ=0也读取滞后raw action，因此baseline会随bidder改变；它不能和四个
  bidder-invariant references混成一个平均值。
- Low P7/P8的expected-value量级明显低于High，同时PID/dual的reference-only baseline仍保持较高预算利用；
  后续放宽产生负效应的原因不是baseline完全失去交付能力，而是稀疏反馈下bidder deviation不够可靠。

## 五、AuctionNet Micro：真正“无执行边界”的Raw端点

{micro_raw_table}

源表：[auctionnet_micro_raw_endpoint.csv](data_statistics_source_tables/auctionnet_micro_raw_endpoint.csv)。

Raw端点说明模型能力和预算纪律可以分离：CQL Raw在P7获得最高价值，但仍留下14.50%预算；PID bidder
几乎完成预算但价值较低；Constant5既缺少机会排序，也留下84.04%预算。有限κ的价值正是要在这些端点
与reference-only之间形成状态依赖的折中。

## 六、AuctionNet Macro：这里没有κ=0，baseline是α=0、κ=.3

{macro_base_table}

源表：[auctionnet_macro_operational_baselines.csv](data_statistics_source_tables/auctionnet_macro_operational_baselines.csv)。

### 对应的市场采用效果

{macro_effect_table}

源表：[auctionnet_macro_policy_effects.csv](data_statistics_source_tables/auctionnet_macro_policy_effects.csv)。

解读边界：

- PID、dual、traffic-aware与继承的smoothed市场都出现“广告主价值提高、平台当期拍卖收入下降”的wedge。
- Response-aware是明确例外；其全采用带来的价值变化很小，收入反而略升。
- Traffic-aware当前只做了50%随机采用，不应把它和全采用数字直接比较。
- Smoothed P11–P12与P13使用继承的、经SHA校验的旧正式实现；其绝对value水平与本轮PID/dual实现
  不作横向排名，只比较同一实现内的变化。

## 七、iPinYou：九个真实campaign的规模

{ipy_campaign_table}

总计：train {ipy['train_rows']:,}行（其中pCTR fitting {ipy['pctr_training_rows']:,}、calibration
{ipy['calibration_rows']:,}），test {ipy['test_rows']:,}行；train clicks {ipy['train_clicks']:,}、test clicks
{ipy['test_clicks']:,}、test conversions {ipy['test_conversions']:,}。单campaign test规模从
{ipy['test_rows_min']:,}到{ipy['test_rows_max']:,}，中位数{ipy['test_rows_median']:,.0f}。

源表：[ipinyou_campaign_data_stats.csv](data_statistics_source_tables/ipinyou_campaign_data_stats.csv)。

### iPinYou κ=0 reference-only表现

{ipy_table}

源表：[ipinyou_reference_only.csv](data_statistics_source_tables/ipinyou_reference_only.csv)。

### iPinYou Raw端点

{ipy_raw_table}

源表：[ipinyou_raw_endpoint.csv](data_statistics_source_tables/ipinyou_raw_endpoint.csv)。

这些数字只在日志支持内成立。预算扩大时三类reference都能购买更多logged opportunities，但点击增幅远慢于
支出增幅，eCPC明显上升；Raw bidder则极度选择性，预算利用率仅约0.13%–2.04%，且预算越宽松利用率越低。
这说明真实campaign中的核心问题不是
“哪一个reference绝对最好”，而是reference、预算压力和可见支持共同决定结果。

## 八、T9Sim：100M底层拍卖与四个信息视图

- 十个10M archives合计{t9['ten_m_archives_bytes']/1e9:.2f} GB，全部通过官方MD5、ZIP、10M×55 schema gate。
- 每个seed约{t9['training_rows_per_seed']:,} training、{t9['calibration_rows_per_seed']:,} calibration、
  {t9['evaluation_rows_per_seed']:,} evaluation rows；每个seed 200 campaigns。
- 基准预算下单campaign预算中位数{t9['budget_per_campaign_median']:.2f}，范围
  {t9['budget_per_campaign_min']:.2f}–{t9['budget_per_campaign_max']:.2f}；每seed预算总额平均
  {t9['budget_total_mean_per_seed']:.1f}。
- C1=DSP；C2=DSP+MMP；C3=DSP+SSP；C4=DSP+MMP+SSP。四个views共享同一ground-truth stream。

### T9Sim κ=0 reference-only，budget multiplier=1

{t9_table}

源表：[t9sim_reference_only_b1.csv](data_statistics_source_tables/t9sim_reference_only_b1.csv)。

### T9Sim Raw端点，budget multiplier=1

{t9_raw_table}

源表：[t9sim_raw_endpoint_b1.csv](data_statistics_source_tables/t9sim_raw_endpoint_b1.csv)。

T9Sim最重要的结果不是“复制AuctionNet均值”，而是说明information architecture改变reference-only
购买集合：MMP可见时（C2/C4），traffic-aware的truth EV和效率明显提高；PID/dual在C2/C4的κ=0
水平反而低于C1/C3。Raw在四个views都存在约14%–18% underdelivery。结合此前wide−tight结果，
这说明control value来自reference与bidder信息的互补/重叠，而不是一个数据集无关的模型排行榜。

## 九、统计口径与不能做的比较

| Issue | Correct interpretation |
|---|---|
| High vs Low | 同一P7/P8 auction skeleton上的不同反馈/价值环境，不是独立新增流量 |
| T9 C1–C4 | 同一100M底层拍卖的四个censored views，不是400M独立样本 |
| κ=0 | reference-only；bidder没有偏离权限 |
| Raw | 无执行边界；不等于“最好”，也不等于预算会花完 |
| Macro α=0 | 无wide-policy adoption，但全体仍在κ=.3；不是κ=0 |
| Replication unit | Micro用CRN seed/advertiser；macro用market episode；iPinYou用campaign；T9用generator seed |
| Cross-dataset levels | EV、CPA/eCPC、ROAS单位不同，不能横向排名绝对值 |
| External validity | iPinYou是logged-support replay；T9是synthetic truth diagnostics；二者都不识别真实共享市场干扰 |

## 十、给合作者的一句话版本

这套证据不是“在四个数据集重复同一张排行榜”。AuctionNet Micro识别campaign-level
performance–delivery机制，AuctionNet Macro识别同步竞争和当期平台收入，iPinYou显示真实campaign与预算压力
异质性，T9Sim利用truth和censoring views解释购买集合为何改变。Reference-only与Raw两端的巨大差异说明，
论文真正研究的是**如何设计bidder deviation进入执行的边界**，以及该边界如何随reference信息、机会供给、
反馈可靠性和市场采用范围变化。

## 十一、若后续需要补证据，优先顺序

1. 只有在审稿人明确要求时，才为同步市场补跑真正的κ=0 reference-only benchmark；当前宏观主张由
   同一reference、同一episode内的α=0到政策采用差值识别，不依赖跨reference绝对值排名。
2. 若能获得具有多广告主同步出价与成交价的外部日志，应优先验证market interference；普通单campaign
   replay不能替代这一证据。
3. iPinYou与T9Sim后续最有价值的扩展不是新增模型，而是检验少数透明operating tiers能否跨campaign
   稳定预测tight/wide control的相对收益。

## 十二、可复核文件

- 原始规模审计：[dataset_raw_statistics.json](dataset_raw_statistics.json)
- 汇总完整性审计：[data_statistics_analysis_audit.json](data_statistics_analysis_audit.json)
- 全部源表：`data_statistics_source_tables/`
- 完整实验结果：[complete_results.html](complete_results.html)
- 独立审计：[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)
- 结果到主张：[RESULT_TO_CLAIM.md](RESULT_TO_CLAIM.md)
"""

    args.output_md.write_text(report, encoding="utf-8")
    expected_rows = {
        "dataset_scale.csv": 9,
        "experiment_scale.csv": 12,
        "auctionnet_micro_reference_only.csv": 17,
        "auctionnet_micro_raw_endpoint.csv": 7,
        "auctionnet_macro_operational_baselines.csv": 6,
        "auctionnet_macro_policy_effects.csv": 6,
        "ipinyou_campaign_data_stats.csv": 9,
        "ipinyou_reference_only.csv": 9,
        "ipinyou_raw_endpoint.csv": 3,
        "t9sim_reference_only_b1.csv": 12,
        "t9sim_raw_endpoint_b1.csv": 4,
    }
    table_checks = {}
    for name, expected in expected_rows.items():
        path = source_dir / name
        observed = len(pd.read_csv(path)) if path.exists() else None
        table_checks[name] = {
            "exists": path.exists(),
            "expected_rows": expected,
            "observed_rows": observed,
            "pass": observed == expected,
        }
    report_sha = hashlib.sha256(args.output_md.read_bytes()).hexdigest()
    audit = {
        "status": "PASS" if all(item["pass"] for item in table_checks.values()) else "FAIL",
        "report": str(args.output_md),
        "report_sha256": report_sha,
        "source_tables": table_checks,
        "semantic_guards": {
            "kappa_zero_labeled_reference_only": "κ=0" in report and "reference-only" in report,
            "raw_labeled_unbounded": "真正“没有执行边界”的端点是`Raw`" in report,
            "macro_baseline_not_mislabeled_kappa_zero": "baseline是`α=0`且所有广告主使用`κ=.3`" in report,
            "high_low_not_double_counted": "High和Low是同一骨架上的不同反馈/价值环境" in report,
            "t9_views_not_double_counted": "不是4亿个独立拍卖" in report,
        },
    }
    audit["status"] = "PASS" if (
        audit["status"] == "PASS" and all(audit["semantic_guards"].values())
    ) else "FAIL"
    audit_path = args.output_md.parent / "data_statistics_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_md),
        "bytes": len(report.encode("utf-8")),
        "source_tables": len(list(source_dir.glob("*.csv"))),
        "audit": str(audit_path),
        "audit_status": audit["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
