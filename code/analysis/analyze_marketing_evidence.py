#!/usr/bin/env python3
"""Aggregate the isolated POMS Marketing evidence expansion.

The script never mutates raw run outputs.  It writes flattened, auditable CSV
tables plus a JSON audit record under ``08_results``.  All contrasts preserve
the frozen statistical unit: T9Sim seed, iPinYou campaign, and AuctionNet
market episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


REFERENCE_ORDER = ["traffic_aware", "pid", "dual"]
CONDITION_ORDER = ["C1", "C2", "C3", "C4"]
CONDITION_LABELS = {
    "C1": "DSP",
    "C2": "DSP+MMP",
    "C3": "DSP+SSP",
    "C4": "DSP+MMP+SSP",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([finite_float(x) for x in values], dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if not n:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "se": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "positive": 0,
                "negative": 0, "zero": 0, "sign_test_p": np.nan}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else np.nan
    se = sd / math.sqrt(n) if n > 1 else np.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
    positive = int((array > 0).sum())
    negative = int((array < 0).sum())
    nonzero = positive + negative
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - critical * se if n > 1 else np.nan,
        "ci_high": mean + critical * se if n > 1 else np.nan,
        "positive": positive,
        "negative": negative,
        "zero": int((array == 0).sum()),
        "sign_test_p": (
            float(stats.binomtest(positive, n=nonzero, p=0.5).pvalue)
            if nonzero else np.nan
        ),
    }


def summarize(df: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(groups, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            rows.append({**base, "metric": metric, **mean_ci(group[metric])})
    return pd.DataFrame(rows)


def analyze_t9(root: Path, out: Path) -> dict[str, Any]:
    files = sorted((root / "06_runs/t9sim_acquisition").glob("seed_*/*.json"))
    flat: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    truth_ok = True
    for path in files:
        payload = json.loads(path.read_text())
        hashes[str(path.relative_to(root))] = sha256(path)
        truth_ok &= payload.get("truth_usage") == "evaluation only"
        seed = int(payload["seed"])
        for condition, condition_data in payload["conditions"].items():
            for row in condition_data["results"]:
                flat.append({
                    "seed": seed,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    **{k: v for k, v in row.items()
                       if k not in {"per_campaign", "spend_by_step", "expected_value_by_step",
                                    "replacement_diagnostic", "delta_vs_reference",
                                    "acquisition_mix_diagnostic"}},
                })
                diag = row.get("acquisition_mix_diagnostic")
                if not diag:
                    continue
                base = {
                    "seed": seed,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "reference": row["reference"],
                    "budget_multiplier": row["budget_multiplier"],
                    "kappa": row["kappa"],
                }
                for purchase_set in ["wide_only", "tight_only", "common"]:
                    diagnostics.append({**base, "purchase_set": purchase_set,
                                        **diag[purchase_set]})
                for cell in diag["value_by_difficulty_migration"]:
                    migrations.append({**base, **cell})

    flat_df = pd.DataFrame(flat)
    if flat_df.empty:
        return {"files": 0, "status": "PENDING"}
    flat_df.to_csv(out / "t9sim_policy_cells.csv", index=False)

    bounded = flat_df[flat_df.reference.isin(REFERENCE_ORDER)].copy()
    width_metrics = [
        "expected_value", "expected_roas", "expected_surplus", "realized_ltv",
        "realized_roas", "spend", "budget_usage", "underdelivery", "wins",
        "win_rate", "clicks", "installs", "payers", "mean_won_true_efficiency",
        "lower_binding_rate", "upper_binding_rate", "mean_reference",
        "reference_log_volatility",
    ]
    summarize(
        bounded,
        ["condition", "condition_label", "reference", "budget_multiplier", "kappa"],
        width_metrics,
    ).to_csv(out / "t9sim_width_response_summary.csv", index=False)
    raw = flat_df[flat_df.reference == "raw"].copy()
    if not raw.empty:
        summarize(
            raw,
            ["condition", "condition_label", "budget_multiplier"],
            [metric for metric in width_metrics if metric in raw.columns],
        ).to_csv(out / "t9sim_raw_endpoint_summary.csv", index=False)

    tight = bounded[np.isclose(bounded.kappa, 0.0)].copy()
    wide = bounded[np.isclose(bounded.kappa, 0.8)].copy()
    keys = ["seed", "condition", "condition_label", "reference", "budget_multiplier"]
    numeric = [
        "expected_value", "expected_roas", "expected_surplus", "realized_ltv",
        "realized_roas", "spend", "budget_usage", "underdelivery", "wins",
        "win_rate", "clicks", "installs", "payers", "mean_won_true_efficiency",
    ]
    paired = wide[keys + numeric].merge(tight[keys + numeric], on=keys,
                                          suffixes=("_wide", "_tight"), validate="one_to_one")
    for metric in numeric:
        paired[f"delta_{metric}"] = paired[f"{metric}_wide"] - paired[f"{metric}_tight"]
    paired.to_csv(out / "t9sim_wide_tight_contrasts.csv", index=False)
    t9_summary = summarize(
        paired,
        ["condition", "condition_label", "reference", "budget_multiplier"],
        [f"delta_{m}" for m in numeric],
    )
    t9_summary.to_csv(out / "t9sim_wide_tight_summary.csv", index=False)

    # Marketing information interaction on the *incremental value of width*.
    gain = paired.pivot_table(
        index=["seed", "reference", "budget_multiplier"],
        columns="condition", values="delta_expected_value", aggfunc="first",
    ).reset_index()
    # Partial monitoring may observe condition files in a staggered order.
    # Preserve a stable schema without imputing any missing experimental cell.
    for condition in CONDITION_ORDER:
        if condition not in gain:
            gain[condition] = np.nan
    gain["MMP_without_SSP"] = gain.C2 - gain.C1
    gain["MMP_with_SSP"] = gain.C4 - gain.C3
    gain["SSP_without_MMP"] = gain.C3 - gain.C1
    gain["SSP_with_MMP"] = gain.C4 - gain.C2
    gain["MMP_x_SSP"] = (gain.C4 - gain.C3) - (gain.C2 - gain.C1)
    gain.to_csv(out / "t9sim_information_interactions.csv", index=False)
    interaction_summary = summarize(
        gain, ["reference", "budget_multiplier"],
        ["MMP_without_SSP", "MMP_with_SSP", "SSP_without_MMP",
         "SSP_with_MMP", "MMP_x_SSP"],
    )
    interaction_summary.to_csv(out / "t9sim_information_interaction_summary.csv", index=False)

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(out / "t9sim_acquisition_profiles.csv", index=False)
    if not diag_df.empty:
        id_cols = ["seed", "condition", "condition_label", "reference",
                   "budget_multiplier", "kappa"]
        wide_diag = diag_df[diag_df.purchase_set == "wide_only"].copy()
        tight_diag = diag_df[diag_df.purchase_set == "tight_only"].copy()
        profile = wide_diag.merge(tight_diag, on=id_cols, suffixes=("_wide", "_tight"),
                                  validate="one_to_one")
        profile_metrics = [
            "count", "true_expected_value_sum", "true_expected_value_mean",
            "true_efficiency_mean", "required_price_mean", "bidder_score_mean",
            "p_click_mean", "p_install_mean", "p_payer_mean",
            "expected_ltv_conditional_mean", "click_rate_realized",
            "install_rate_realized", "payer_rate_realized", "ltv_realized_mean",
            "mean_step",
        ]
        for metric in profile_metrics:
            profile[f"delta_{metric}"] = profile[f"{metric}_wide"] - profile[f"{metric}_tight"]
        profile.to_csv(out / "t9sim_exclusive_purchase_differences.csv", index=False)
        summarize(
            profile,
            ["condition", "condition_label", "reference", "budget_multiplier"],
            [f"delta_{m}" for m in profile_metrics],
        ).to_csv(out / "t9sim_exclusive_purchase_summary.csv", index=False)
    pd.DataFrame(migrations).to_csv(out / "t9sim_value_difficulty_migration.csv", index=False)

    expected_files = 40
    expected_rows = expected_files * (3 * 6 * 3 + 3)
    return {
        "status": "COMPLETE" if len(files) == expected_files else "PARTIAL",
        "files": len(files),
        "expected_files": expected_files,
        "policy_rows": int(len(flat_df)),
        "expected_policy_rows": expected_rows,
        "wide_tight_pairs": int(len(paired)),
        "truth_evaluation_only": bool(truth_ok),
        "source_sha256": hashes,
    }


def analyze_ipinyou(root: Path, out: Path) -> dict[str, Any]:
    files = sorted((root / "06_runs/ipinyou_composition").glob("campaign_*.json"))
    flat: list[dict[str, Any]] = []
    mix: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in files:
        payload = json.loads(path.read_text())
        hashes[str(path.relative_to(root))] = sha256(path)
        campaign = str(payload["campaign"])
        calibrator = payload["score_calibrator"]
        test_diag = payload["test_pctr_diagnostic"]
        audits.append({
            "campaign": campaign,
            "schema_version": payload.get("schema_version"),
            "calibration_revision": payload.get("calibration_revision"),
            "calibration_rows": payload.get("calibration_rows"),
            "test_rows": payload.get("test_rows"),
            "calibration_auc": calibrator.get("calibration_auc"),
            "calibration_mean_pctr": calibrator.get("calibration_mean_pctr"),
            "calibration_ctr": calibrator.get("calibration_ctr"),
            "test_mean_pctr": test_diag.get("mean"),
            "test_pctr_std": test_diag.get("std"),
            "test_nondegenerate": test_diag.get("nondegenerate"),
            "cutpoint_source": payload["purchase_mix_design"].get("cutpoint_source"),
        })
        for row in payload["results"]:
            base = {"campaign": campaign, **{k: v for k, v in row.items()
                    if k not in {"spend_by_step", "clicks_by_step", "purchase_mix"}}}
            flat.append(base)
            for cell in row["purchase_mix"]:
                mix.append({
                    "campaign": campaign,
                    "reference": row["reference"],
                    "kappa": row["kappa"],
                    "budget_fraction": row["budget_fraction"],
                    **cell,
                })

    flat_df = pd.DataFrame(flat)
    audit_df = pd.DataFrame(audits)
    if flat_df.empty:
        return {"files": 0, "status": "PENDING"}
    flat_df.to_csv(out / "ipinyou_policy_cells.csv", index=False)
    audit_df.to_csv(out / "ipinyou_calibration_audit.csv", index=False)
    mix_df = pd.DataFrame(mix)
    mix_df.to_csv(out / "ipinyou_purchase_mix_cells.csv", index=False)

    bounded = flat_df[flat_df.reference.isin(REFERENCE_ORDER)].copy()
    tight = bounded[np.isclose(bounded.kappa, 0.0)]
    wide = bounded[np.isclose(bounded.kappa, 0.8)]
    keys = ["campaign", "reference", "budget_fraction"]
    numeric = ["spend", "budget_usage", "underdelivery", "wins", "clicks",
               "conversions", "ecpc", "ecpa", "ctr", "mean_won_pctr"]
    paired = wide[keys + numeric].merge(tight[keys + numeric], on=keys,
                                         suffixes=("_wide", "_tight"), validate="one_to_one")
    for metric in numeric:
        paired[f"delta_{metric}"] = paired[f"{metric}_wide"] - paired[f"{metric}_tight"]
    paired["pct_delta_clicks"] = np.where(
        paired.clicks_tight > 0,
        100 * paired.delta_clicks / paired.clicks_tight,
        np.nan,
    )
    paired.to_csv(out / "ipinyou_wide_tight_contrasts.csv", index=False)
    summarize(
        paired, ["reference", "budget_fraction"],
        [f"delta_{m}" for m in numeric] + ["pct_delta_clicks"],
    ).to_csv(out / "ipinyou_wide_tight_summary.csv", index=False)
    aggregate = paired.groupby(["reference", "budget_fraction"], as_index=False).agg(
        campaigns=("campaign", "size"),
        clicks_tight=("clicks_tight", "sum"), clicks_wide=("clicks_wide", "sum"),
        spend_tight=("spend_tight", "sum"), spend_wide=("spend_wide", "sum"),
        wins_tight=("wins_tight", "sum"), wins_wide=("wins_wide", "sum"),
        mean_delta_underdelivery=("delta_underdelivery", "mean"),
        positive_campaigns=("delta_clicks", lambda x: int((x > 0).sum())),
        negative_campaigns=("delta_clicks", lambda x: int((x < 0).sum())),
    )
    aggregate["pct_delta_clicks"] = 100 * (aggregate.clicks_wide / aggregate.clicks_tight - 1)
    aggregate["pct_delta_spend"] = 100 * (aggregate.spend_wide / aggregate.spend_tight - 1)
    aggregate["ecpc_tight"] = aggregate.spend_tight / aggregate.clicks_tight
    aggregate["ecpc_wide"] = aggregate.spend_wide / aggregate.clicks_wide
    aggregate["pct_delta_ecpc"] = 100 * (aggregate.ecpc_wide / aggregate.ecpc_tight - 1)
    aggregate.to_csv(out / "ipinyou_aggregate_performance.csv", index=False)

    # Within-campaign composition shifts.  Shares avoid mechanical scaling by
    # a policy's total number of wins or spend.
    mix_keys = ["campaign", "reference", "budget_fraction", "pctr_quintile",
                "paying_price_quintile"]
    tight_mix = mix_df[np.isclose(mix_df.kappa, 0.0)].copy()
    wide_mix = mix_df[np.isclose(mix_df.kappa, 0.8)].copy()
    for frame in (tight_mix, wide_mix):
        totals = frame.groupby(["campaign", "reference", "budget_fraction"])[["wins", "spend"]].transform("sum")
        frame["win_share"] = np.where(totals.wins > 0, frame.wins / totals.wins, 0.0)
        frame["spend_share"] = np.where(totals.spend > 0, frame.spend / totals.spend, 0.0)
        frame["mean_cell_pctr"] = np.where(frame.wins > 0, frame.pctr_sum / frame.wins, np.nan)
    mix_pair = wide_mix.merge(tight_mix, on=mix_keys, suffixes=("_wide", "_tight"),
                              validate="one_to_one")
    for metric in ["wins", "spend", "clicks", "conversions", "pctr_sum",
                   "win_share", "spend_share", "mean_cell_pctr"]:
        mix_pair[f"delta_{metric}"] = mix_pair[f"{metric}_wide"] - mix_pair[f"{metric}_tight"]
    mix_pair.to_csv(out / "ipinyou_purchase_mix_differences.csv", index=False)
    summarize(
        mix_pair,
        ["reference", "budget_fraction", "pctr_quintile", "paying_price_quintile"],
        ["delta_win_share", "delta_spend_share", "delta_mean_cell_pctr"],
    ).to_csv(out / "ipinyou_purchase_mix_summary.csv", index=False)
    composition_parts: list[pd.DataFrame] = []
    definitions = {
        "high_pctr_Q4_Q5": mix_pair.pctr_quintile >= 4,
        "high_price_Q4_Q5": mix_pair.paying_price_quintile >= 4,
        "high_pctr_low_price": (mix_pair.pctr_quintile >= 4) & (mix_pair.paying_price_quintile <= 2),
        "low_pctr_high_price": (mix_pair.pctr_quintile <= 2) & (mix_pair.paying_price_quintile >= 4),
    }
    for name, mask in definitions.items():
        part = mix_pair[mask].groupby(keys, as_index=False).agg(
            delta_win_share=("delta_win_share", "sum"),
            delta_spend_share=("delta_spend_share", "sum"),
        )
        part["composition_region"] = name
        composition_parts.append(part)
    composition = pd.concat(composition_parts, ignore_index=True)
    composition.to_csv(out / "ipinyou_composition_indices.csv", index=False)
    summarize(
        composition, ["reference", "budget_fraction", "composition_region"],
        ["delta_win_share", "delta_spend_share"],
    ).to_csv(out / "ipinyou_composition_index_summary.csv", index=False)

    return {
        "status": "COMPLETE" if len(files) == 9 else "PARTIAL",
        "files": len(files),
        "expected_files": 9,
        "policy_rows": int(len(flat_df)),
        "expected_policy_rows": 9 * 39,
        "wide_tight_pairs": int(len(paired)),
        "all_test_pctr_nondegenerate": bool(audit_df.test_nondegenerate.fillna(False).all()),
        "all_calibration_only_cutpoints": bool(
            audit_df.cutpoint_source.astype(str).str.contains("calibration", case=False).all()
        ),
        "source_sha256": hashes,
    }


def load_network(root: Path, reference: str) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for period in (9, 10):
        path = root / ("06_runs/auctionnet_topology/development_network/"
                       f"{reference}/p{period}/output/competition_overlap.csv")
        edges = pd.read_csv(path)
        matrix = np.zeros((48, 48), dtype=float)
        for row in edges.itertuples(index=False):
            i, j = int(row.agent_i), int(row.agent_j)
            value = float(row.top3_co_count)
            matrix[i, j] = matrix[j, i] = value
        matrices.append(matrix)
    return np.mean(matrices, axis=0)


def analyze_macro(root: Path, out: Path) -> dict[str, Any]:
    run_dirs = sorted((root / "06_runs/auctionnet_topology/formal_v2_with_baseline").glob(
        "*/p*_m*_a*/output"))
    markets: list[pd.DataFrame] = []
    agents: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for directory in run_dirs:
        manifest_path = directory / "run_manifest.json"
        market_path = directory / "market_runs.csv"
        agent_path = directory / "agent_outcomes.csv"
        if not (manifest_path.exists() and market_path.exists() and agent_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        relative = str(manifest_path.relative_to(root))
        hashes[relative] = sha256(manifest_path)
        reference = directory.parts[-3]
        market = pd.read_csv(market_path)
        agent = pd.read_csv(agent_path)
        market["reference"] = reference
        agent["reference"] = reference
        markets.append(market)
        agents.append(agent)
        manifests.append(manifest)
    if not markets:
        return {"files": 0, "status": "PENDING"}
    market_df = pd.concat(markets, ignore_index=True)
    agent_df = pd.concat(agents, ignore_index=True)
    market_df.to_csv(out / "macro_topology_market_cells.csv", index=False)
    agent_df.to_csv(out / "macro_topology_agent_cells.csv", index=False)

    episode_keys = ["reference", "period", "market_seed", "assignment_seed"]
    baseline_market = market_df[np.isclose(market_df.alpha, 0.0)].copy()
    baseline_agent = agent_df[np.isclose(agent_df.alpha, 0.0)].copy()
    if not bool(baseline_market.groupby(episode_keys).size().eq(1).all()):
        raise ValueError("each formal episode must contain exactly one alpha=0 baseline market")
    market_metrics = [
        "total_base_expected_value", "platform_revenue", "aggregate_cpa",
        "budget_usage", "underdelivery", "mean_clearing_price_all",
        "mean_clearing_price_positive", "realized_slot_fill",
    ]
    treated_market = market_df[market_df.alpha > 0].copy()
    joined_market = treated_market.merge(
        baseline_market[episode_keys + market_metrics], on=episode_keys,
        suffixes=("", "_baseline"), validate="many_to_one",
    )
    for metric in market_metrics:
        joined_market[f"delta_{metric}"] = joined_market[metric] - joined_market[f"{metric}_baseline"]
        joined_market[f"pct_delta_{metric}"] = np.where(
            joined_market[f"{metric}_baseline"].abs() > 0,
            100 * joined_market[f"delta_{metric}"] / joined_market[f"{metric}_baseline"],
            np.nan,
        )
    joined_market.to_csv(out / "macro_topology_vs_baseline.csv", index=False)
    summarize(
        joined_market,
        ["reference", "alpha", "adopter_policy"],
        [f"delta_{m}" for m in market_metrics] +
        [f"pct_delta_{m}" for m in market_metrics],
    ).to_csv(out / "macro_topology_vs_baseline_summary.csv", index=False)

    # Topology effects are defined relative to random with the same episode and alpha.
    random_cells = joined_market[joined_market.adopter_policy == "random"].copy()
    topology_cells = joined_market[joined_market.adopter_policy.isin(["clustered", "dispersed"])].copy()
    random_keys = episode_keys + ["alpha"]
    topology = topology_cells.merge(
        random_cells[random_keys + market_metrics], on=random_keys,
        suffixes=("", "_random"), validate="many_to_one",
    )
    for metric in market_metrics:
        topology[f"delta_vs_random_{metric}"] = topology[metric] - topology[f"{metric}_random"]
    topology.to_csv(out / "macro_topology_vs_random.csv", index=False)
    summarize(
        topology, ["reference", "alpha", "adopter_policy"],
        [f"delta_vs_random_{m}" for m in market_metrics],
    ).to_csv(out / "macro_topology_vs_random_summary.csv", index=False)

    # Individual non-adopter spillovers relative to the same agent in alpha=0.
    agent_metrics = ["base_expected_value", "latent_expected_value", "cost",
                     "budget_usage", "underdelivery", "wins", "win_rate"]
    agent_keys = episode_keys + ["agent_id"]
    treated_agents = agent_df[(agent_df.alpha > 0) & (~agent_df.adopter)].copy()
    spill = treated_agents.merge(
        baseline_agent[agent_keys + agent_metrics], on=agent_keys,
        suffixes=("", "_baseline"), validate="many_to_one",
    )
    for metric in agent_metrics:
        spill[f"delta_{metric}"] = spill[metric] - spill[f"{metric}_baseline"]

    exposure_rows: list[dict[str, Any]] = []
    for keys, group in spill.groupby(episode_keys + ["alpha", "adopter_policy"], sort=False):
        reference, period, market_seed, assignment_seed, alpha, policy = keys
        # Adopter status is read from the full cell rather than inferred from non-adopters.
        cell_agents = agent_df[
            (agent_df.reference == reference)
            & (agent_df.period == period)
            & (agent_df.market_seed == market_seed)
            & (agent_df.assignment_seed == assignment_seed)
            & np.isclose(agent_df.alpha, alpha)
            & (agent_df.adopter_policy == policy)
        ]
        adopters = cell_agents.loc[cell_agents.adopter, "agent_id"].astype(int).to_numpy()
        network = load_network(root, reference)
        degree = network.sum(axis=1)
        for agent_id in group.agent_id.astype(int):
            raw = float(network[agent_id, adopters].sum())
            normalized = raw / degree[agent_id] if degree[agent_id] > 0 else 0.0
            exposure_rows.append({
                "reference": reference, "period": period, "market_seed": market_seed,
                "assignment_seed": assignment_seed, "alpha": alpha,
                "adopter_policy": policy, "agent_id": agent_id,
                "adopter_exposure_raw": raw, "adopter_exposure_share": normalized,
            })
    exposure = pd.DataFrame(exposure_rows)
    spill = spill.merge(exposure, on=episode_keys + ["alpha", "adopter_policy", "agent_id"],
                        validate="one_to_one")
    spill.to_csv(out / "macro_nonadopter_spillovers.csv", index=False)
    summarize(
        spill, ["reference", "alpha", "adopter_policy"],
        [f"delta_{m}" for m in agent_metrics],
    ).to_csv(out / "macro_nonadopter_spillover_summary.csv", index=False)

    # Aggregate spillovers only after matching each non-adopter to its own
    # alpha=0 outcome.  Raw market-level ``nonadopter_*`` totals are not
    # comparable with alpha=0 because the latter contains all 48 advertisers.
    cell_keys = episode_keys + ["alpha", "adopter_policy"]
    spill_cell = spill.groupby(cell_keys, as_index=False).agg(
        nonadopters=("agent_id", "size"),
        mean_spillover_value=("delta_base_expected_value", "mean"),
        total_spillover_value=("delta_base_expected_value", "sum"),
        mean_spillover_wins=("delta_wins", "mean"),
        mean_spillover_win_rate=("delta_win_rate", "mean"),
        mean_spillover_cost=("delta_cost", "mean"),
        mean_spillover_underdelivery=("delta_underdelivery", "mean"),
    )
    spill_cell.to_csv(out / "macro_nonadopter_spillover_cells.csv", index=False)
    spill_random = spill_cell[spill_cell.adopter_policy == "random"]
    spill_topology = spill_cell[spill_cell.adopter_policy.isin(["clustered", "dispersed"])].merge(
        spill_random, on=episode_keys + ["alpha"], suffixes=("", "_random"),
        validate="many_to_one",
    )
    for metric in ["mean_spillover_value", "total_spillover_value",
                   "mean_spillover_wins", "mean_spillover_win_rate",
                   "mean_spillover_cost", "mean_spillover_underdelivery"]:
        spill_topology[f"delta_vs_random_{metric}"] = (
            spill_topology[metric] - spill_topology[f"{metric}_random"]
        )
    spill_topology.to_csv(out / "macro_nonadopter_topology_vs_random.csv", index=False)
    summarize(
        spill_topology, ["reference", "alpha", "adopter_policy"],
        [c for c in spill_topology if c.startswith("delta_vs_random_")],
    ).to_csv(out / "macro_nonadopter_topology_vs_random_summary.csv", index=False)

    exposure_stats: list[dict[str, Any]] = []
    for keys, group in spill.groupby(["reference", "alpha", "adopter_policy"], sort=False):
        for outcome in ["delta_base_expected_value", "delta_wins", "delta_win_rate",
                        "delta_cost", "delta_underdelivery"]:
            valid = group[["adopter_exposure_share", outcome]].dropna()
            if len(valid) < 3 or valid.adopter_exposure_share.nunique() < 2:
                rho = pvalue = slope = np.nan
            else:
                rho, pvalue = stats.spearmanr(valid.adopter_exposure_share, valid[outcome])
                slope = stats.linregress(valid.adopter_exposure_share, valid[outcome]).slope
            exposure_stats.append({
                "reference": keys[0], "alpha": keys[1], "adopter_policy": keys[2],
                "outcome": outcome, "n": len(valid), "spearman": rho,
                "spearman_p": pvalue, "ols_slope": slope,
            })
    pd.DataFrame(exposure_stats).to_csv(out / "macro_exposure_spillover_associations.csv", index=False)

    # Episode-level exposure gradients preserve the market episode as the
    # inferential unit.  The pooled associations above remain descriptive and
    # must not be interpreted with row-level p-values as if advertisers across
    # the same reconstructed market were independent.
    episode_exposure_stats: list[dict[str, Any]] = []
    exposure_groups = episode_keys + ["alpha", "adopter_policy"]
    for keys, group in spill.groupby(exposure_groups, sort=False):
        base = dict(zip(exposure_groups, keys))
        for outcome in ["delta_base_expected_value", "delta_wins", "delta_win_rate",
                        "delta_cost", "delta_underdelivery"]:
            valid = group[["adopter_exposure_share", outcome]].dropna()
            if len(valid) < 3 or valid.adopter_exposure_share.nunique() < 2:
                rho = slope = np.nan
            else:
                rho = stats.spearmanr(valid.adopter_exposure_share, valid[outcome]).statistic
                slope = stats.linregress(valid.adopter_exposure_share, valid[outcome]).slope
            episode_exposure_stats.append({
                **base, "outcome": outcome, "nonadopters": len(valid),
                "spearman": rho, "ols_slope": slope,
            })
    episode_exposure = pd.DataFrame(episode_exposure_stats)
    episode_exposure.to_csv(out / "macro_exposure_episode_associations.csv", index=False)
    summarize(
        episode_exposure,
        ["reference", "alpha", "adopter_policy", "outcome"],
        ["spearman", "ols_slope"],
    ).to_csv(out / "macro_exposure_episode_association_summary.csv", index=False)

    expected = 60
    counts_ok = all(int(m.get("counts", {}).get("market_runs.csv", -1)) == 7 for m in manifests)
    return {
        "status": "COMPLETE" if len(manifests) == expected and counts_ok else "PARTIAL",
        "episodes": len(manifests),
        "expected_episodes": expected,
        "market_rows": int(len(market_df)),
        "agent_rows": int(len(agent_df)),
        "seven_cells_per_episode": bool(counts_ok),
        "same_run_baseline_present": bool(len(baseline_market) == len(manifests)),
        "source_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "08_results/marketing_evidence"
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "root": str(root),
        "t9sim": analyze_t9(root, out),
        "ipinyou": analyze_ipinyou(root, out),
        "auctionnet_macro": analyze_macro(root, out),
        "inherited_auctionnet_segments": {
            "audit": str(root / "08_results/auctionnet_campaign_segments_v3/auctionnet_segment_audit.json"),
            "exists": (root / "08_results/auctionnet_campaign_segments_v3/auctionnet_segment_audit.json").exists(),
        },
    }
    status_values = [audit[k].get("status") for k in ["t9sim", "ipinyou", "auctionnet_macro"]]
    audit["status"] = "COMPLETE" if all(x == "COMPLETE" for x in status_values) else "PARTIAL"
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
