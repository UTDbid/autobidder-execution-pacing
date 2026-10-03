#!/usr/bin/env python3
"""Assemble inherited/new AuctionNet surfaces and evaluate transparent κ rules.

The later-period screen is retrospective by construction.  A separate phase can
evaluate the frozen rules on newly generated CRN seeds, but that phase is called
stochastic transport rather than temporal OOS.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("KAPPA_CHALLENGE_ROOT", Path(__file__).resolve().parents[1]))
# KAPPA_INPUT_ROOT must point at the directory that holds the unpacked
# upstream results / checkpoints (see docs/reproduction.md). The fallback is
# the release root so the script fails with a clear "file not found" rather
# than silently reading a stale absolute path.
OLD = Path(os.environ.get("KAPPA_INPUT_ROOT", str(Path(__file__).resolve().parents[2])))
REF = OLD / "poms_reference_expansion_20260829"
WA = OLD / "poms_width_alignment_20260902"
AUTO = OLD / "autonomy_assurance_20260810"
MODELS = ("cql", "iql", "dt", "gas", "sembid")
REFERENCES = ("traffic_aware", "pid", "dual")
WIDTHS = (0.0, 0.3, 0.5, 0.8, 1.0)
CAPS = (5.0, 10.0, 20.0)
CAL_SEEDS = tuple(range(20260428, 20260432))
TEST_SEEDS = tuple(range(20260428, 20260436))
FRESH_SEEDS = tuple(range(2026090400, 2026090408))


TRAFFIC_CAL_PATHS = {
    "cql": "GATEB-V1-CQL-S{suffix}/results.json",
    "iql": "UTD2-HIGH-IQL-S{suffix}/results.json",
    "dt": "GATEB-V1-DT-S{suffix}/results.json",
    "gas": "UTD2-HIGH-GAS-S{suffix}-R1/results.json",
    "sembid": "UTD2-HIGH-SEMBID-S{suffix}-R1/results.json",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_traffic_row(row: dict, width: float) -> bool:
    cfg = row["config"]
    if cfg.get("reference_mode") != "traffic_aware":
        return False
    if abs(float(cfg.get("kappa", -99)) - width) > 1e-9:
        return False
    identifier = str(cfg.get("id", ""))
    return "m=clean" in identifier or "__m=" not in identifier


def row_record(row: dict, period: str, model: str, reference: str, seed: int, source: Path) -> dict:
    return {
        "period": period,
        "model": model,
        "reference": reference,
        "seed": int(seed),
        "kappa": float(row["config"]["kappa"]),
        "value": float(row["total_expected_value"]),
        "cpa": float(row["aggregate_cpa"]),
        "underdelivery": float(row["underdelivery"]),
        "budget": float(row["total_budget"]),
        "score": float(row["avg_score"]),
        "source": str(source),
    }


def collect_calibration() -> pd.DataFrame:
    # The width-alignment audit already compacted the exact inherited P7 JSON
    # cells into one row per model/reference/seed/width.  Reading this audited
    # CSV avoids repeatedly parsing large per-agent traces over NFS.
    path = WA / "06_results/micro_aligned_grid_per_seed.csv"
    raw = pd.read_csv(path)
    raw["reference"] = raw.reference.replace({"traffic": "traffic_aware"})
    raw = raw[
        raw.model.isin(MODELS)
        & raw.reference.isin(REFERENCES)
        & raw.seed.isin(CAL_SEEDS)
        & raw.kappa.isin(WIDTHS)
    ].copy()
    return pd.DataFrame({
        "period": "calibration",
        "model": raw.model,
        "reference": raw.reference,
        "seed": raw.seed.astype(int),
        "kappa": raw.kappa.astype(float),
        "value": raw.total_expected_value.astype(float),
        "cpa": raw.aggregate_cpa.astype(float),
        "underdelivery": raw.underdelivery.astype(float),
        "budget": 150000.0,
        "score": np.nan,
        "source": str(path),
    })


def collect_retrospective() -> pd.DataFrame:
    records: list[dict] = []
    inherited_path = REF / "08_results/auctionnet_micro/high_p8/reference_expansion/micro_aggregate_raw.csv"
    inherited = pd.read_csv(inherited_path)
    inherited = inherited[
        inherited.model.isin(MODELS)
        & inherited.reference.isin(("pid", "dual"))
        & inherited.seed.isin(TEST_SEEDS)
        & inherited.kappa.isin((0.0, 0.3, 0.5, 0.8))
    ]
    for row in inherited.itertuples(index=False):
        records.append({"period": "retrospective", "model": row.model, "reference": row.reference,
            "seed": int(row.seed), "kappa": float(row.kappa), "value": float(row.total_expected_value),
            "cpa": float(row.aggregate_cpa), "underdelivery": float(row.underdelivery), "budget": 150000.0,
            "score": float(row.avg_score), "source": str(inherited_path)})

    traffic_path = REF / "08_results/auctionnet_micro/high_p8/traffic_aware_completion/micro_aggregate_raw.csv"
    traffic = pd.read_csv(traffic_path)
    traffic = traffic[
        traffic.model.isin(MODELS)
        & traffic.seed.isin(TEST_SEEDS)
        & traffic.kappa.isin((0.0, 0.8))
    ]
    for row in traffic.itertuples(index=False):
        records.append({"period": "retrospective", "model": row.model, "reference": "traffic_aware",
            "seed": int(row.seed), "kappa": float(row.kappa), "value": float(row.total_expected_value),
            "cpa": float(row.aggregate_cpa), "underdelivery": float(row.underdelivery), "budget": 150000.0,
            "score": float(row.avg_score), "source": str(traffic_path)})

    for model in MODELS:
        for seed in TEST_SEEDS:
            new = ROOT / "04_runs/retrospective" / model / f"seed_{seed}" / "results.json"
            payload = load_json(new)
            for row in payload["results"]:
                reference = row["config"]["reference_mode"]
                records.append(row_record(row, "retrospective", model, reference, seed, new))
    return pd.DataFrame(records)


def collect_fresh() -> pd.DataFrame:
    records: list[dict] = []
    for model in MODELS:
        for seed in FRESH_SEEDS:
            path = ROOT / "04_runs/fresh_stochastic" / model / f"seed_{seed}" / "results.json"
            payload = load_json(path)
            for row in payload["results"]:
                records.append(
                    row_record(row, "fresh_stochastic", model, row["config"]["reference_mode"], seed, path)
                )
    return pd.DataFrame(records)


def validate_surface(frame: pd.DataFrame, seeds: tuple[int, ...], label: str) -> None:
    expected = len(MODELS) * len(REFERENCES) * len(WIDTHS) * len(seeds)
    if len(frame) != expected:
        raise ValueError(f"{label}: expected {expected} rows, observed {len(frame)}")
    counts = frame.groupby(["model", "reference", "kappa", "seed"]).size()
    if not (counts == 1).all():
        raise ValueError(f"{label}: duplicated or missing cells")


def means(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.groupby(["model", "reference", "kappa"], as_index=False).agg(
        value=("value", "mean"),
        cpa=("cpa", "mean"),
        underdelivery=("underdelivery", "mean"),
        budget=("budget", "mean"),
    )
    base = mean[mean.kappa == 0.0][["model", "reference", "value", "underdelivery"]].rename(
        columns={"value": "value_k0", "underdelivery": "ud_k0"}
    )
    mean = mean.merge(base, on=["model", "reference"], validate="many_to_one")
    mean["value_gain"] = mean.value - mean.value_k0
    mean["delta_ud_pp"] = 100 * (mean.underdelivery - mean.ud_k0)
    return mean


def pick_smallest_best(rows: pd.DataFrame, metric: str = "value") -> float:
    best = rows[metric].max()
    return float(rows[np.isclose(rows[metric], best, rtol=0, atol=1e-9)].kappa.min())


def select_rules(cal: pd.DataFrame, cap: float) -> pd.DataFrame:
    surface = means(cal)
    selected: list[dict] = []
    for (model, reference), pair in surface.groupby(["model", "reference"]):
        feasible = pair[pair.delta_ud_pp <= cap + 1e-9].copy()
        if feasible.empty:
            feasible = pair[pair.kappa == 0.0].copy()
        r1 = pick_smallest_best(feasible)
        max_gain = feasible.value_gain.max()
        if max_gain <= 0:
            r2 = 0.0
        else:
            r2 = float(feasible[feasible.value_gain >= 0.95 * max_gain - 1e-9].kappa.min())

        r3 = 0.0
        for left, right in zip(WIDTHS[:-1], WIDTHS[1:]):
            if abs(r3 - left) > 1e-9:
                break
            left_seed = cal[(cal.model == model) & (cal.reference == reference) & (cal.kappa == left)].set_index("seed")
            right_seed = cal[(cal.model == model) & (cal.reference == reference) & (cal.kappa == right)].set_index("seed")
            changes = right_seed.value - left_seed.value
            next_ud = float(pair.loc[np.isclose(pair.kappa, right), "delta_ud_pp"].iloc[0])
            if changes.mean() > 0 and int((changes > 0).sum()) >= 3 and next_ud <= cap + 1e-9:
                r3 = right
            else:
                break

        pre = feasible[feasible.kappa <= 0.8 + 1e-9]
        r4 = pick_smallest_best(pre)
        if abs(r4 - 0.8) < 1e-9 and (feasible.kappa == 1.0).any():
            k08 = cal[(cal.model == model) & (cal.reference == reference) & (cal.kappa == 0.8)].set_index("seed")
            k10 = cal[(cal.model == model) & (cal.reference == reference) & (cal.kappa == 1.0)].set_index("seed")
            changes = k10.value - k08.value
            if changes.mean() > 0 and int((changes > 0).sum()) >= 3:
                r4 = 1.0
        for rule, width in (("R1_frontier", r1), ("R2_near_best_95", r2), ("R3_sequential", r3), ("R4_kink_aware", r4)):
            selected.append({"cap_pp": cap, "model": model, "reference": reference, "rule": rule, "selected_kappa": width})

    # Pooled calibration baselines.
    global_surface = surface.groupby("kappa", as_index=False).agg(value=("value", "mean"), delta_ud_pp=("delta_ud_pp", "mean"))
    global_feasible = global_surface[global_surface.delta_ud_pp <= cap + 1e-9]
    global_k = pick_smallest_best(global_feasible)
    for model in MODELS:
        for reference in REFERENCES:
            selected.append({"cap_pp": cap, "model": model, "reference": reference, "rule": "B_global_cal", "selected_kappa": global_k})
    for reference, ref_surface in surface.groupby("reference"):
        pooled = ref_surface.groupby("kappa", as_index=False).agg(value=("value", "mean"), delta_ud_pp=("delta_ud_pp", "mean"))
        feasible = pooled[pooled.delta_ud_pp <= cap + 1e-9]
        ref_k = pick_smallest_best(feasible)
        for model in MODELS:
            selected.append({"cap_pp": cap, "model": model, "reference": reference, "rule": "B_reference_cal", "selected_kappa": ref_k})
    for fixed in (0.0, 0.3, 0.8, 1.0):
        for model in MODELS:
            for reference in REFERENCES:
                selected.append({"cap_pp": cap, "model": model, "reference": reference, "rule": f"B_fixed_{fixed:g}", "selected_kappa": fixed})
    return pd.DataFrame(selected)


def evaluate_rules(cal: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_surface = means(test)
    all_selected = pd.concat([select_rules(cal, cap) for cap in CAPS], ignore_index=True)
    oracle_rows = []
    value_oracle_rows = []
    for cap in CAPS:
        for (model, reference), pair in test_surface.groupby(["model", "reference"]):
            feasible = pair[pair.delta_ud_pp <= cap + 1e-9]
            if feasible.empty:
                feasible = pair[pair.kappa == 0.0]
            oracle_rows.append({"cap_pp": cap, "model": model, "reference": reference, "oracle_kappa": pick_smallest_best(feasible)})
            value_oracle_rows.append({
                "cap_pp": cap,
                "model": model,
                "reference": reference,
                "value_oracle_kappa": pick_smallest_best(pair),
            })
    oracle = pd.DataFrame(oracle_rows)
    value_oracle = pd.DataFrame(value_oracle_rows)
    out = all_selected.merge(oracle, on=["cap_pp", "model", "reference"], validate="many_to_one")
    out = out.merge(value_oracle, on=["cap_pp", "model", "reference"], validate="many_to_one")
    selected_surface = test_surface.rename(columns={
        "kappa": "selected_kappa", "value": "selected_value", "underdelivery": "selected_ud",
        "budget": "selected_budget", "delta_ud_pp": "selected_delta_ud_pp", "value_gain": "selected_gain",
        "cpa": "selected_cpa",
    })
    oracle_surface = test_surface.rename(columns={
        "kappa": "oracle_kappa", "value": "oracle_value", "underdelivery": "oracle_ud",
        "budget": "oracle_budget", "delta_ud_pp": "oracle_delta_ud_pp", "value_gain": "oracle_gain",
        "cpa": "oracle_cpa",
    })
    value_oracle_surface = test_surface.rename(columns={
        "kappa": "value_oracle_kappa", "value": "value_oracle_value", "value_gain": "value_oracle_gain",
    })
    keep_sel = ["model", "reference", "selected_kappa", "selected_value", "selected_ud", "selected_budget", "selected_delta_ud_pp", "selected_gain", "selected_cpa"]
    keep_oracle = ["model", "reference", "oracle_kappa", "oracle_value", "oracle_ud", "oracle_budget", "oracle_delta_ud_pp", "oracle_gain", "oracle_cpa"]
    keep_value_oracle = ["model", "reference", "value_oracle_kappa", "value_oracle_value", "value_oracle_gain"]
    out = out.merge(selected_surface[keep_sel], on=["model", "reference", "selected_kappa"], validate="many_to_one")
    out = out.merge(oracle_surface[keep_oracle], on=["model", "reference", "oracle_kappa"], validate="many_to_one")
    out = out.merge(value_oracle_surface[keep_value_oracle], on=["model", "reference", "value_oracle_kappa"], validate="many_to_one")
    # The constrained oracle can have lower value than an infeasible selected policy,
    # producing a negative signed gap.  We therefore use regret to the unconstrained
    # value oracle as the non-negative value-selection metric and disclose delivery
    # violations separately.  The signed constrained gap remains in the audit table.
    out["normalized_regret"] = (out.value_oracle_value - out.selected_value) / out.selected_budget
    out["constrained_signed_value_gap"] = (out.oracle_value - out.selected_value) / out.selected_budget
    out["delivery_violation"] = out.selected_delta_ud_pp > out.cap_pp + 1e-9
    out["violation_severity_pp"] = np.maximum(0.0, out.selected_delta_ud_pp - out.cap_pp)
    out["exact_oracle"] = np.isclose(out.selected_kappa, out.oracle_kappa)
    out["within_one_step"] = [
        abs(WIDTHS.index(float(a)) - WIDTHS.index(float(b))) <= 1
        for a, b in zip(out.selected_kappa, out.oracle_kappa)
    ]
    out["value_capture"] = np.where(out.value_oracle_gain > 1e-9, out.selected_gain / out.value_oracle_gain, np.nan)
    summary = out.groupby(["cap_pp", "rule"], as_index=False).agg(
        mean_normalized_regret=("normalized_regret", "mean"),
        median_normalized_regret=("normalized_regret", "median"),
        violation_rate=("delivery_violation", "mean"),
        mean_violation_severity_pp=("violation_severity_pp", "mean"),
        exact_oracle_rate=("exact_oracle", "mean"),
        within_one_step_rate=("within_one_step", "mean"),
        mean_value_capture=("value_capture", "mean"),
    )
    return all_selected, out, summary


def paired_bootstrap(pair_results: pd.DataFrame, cap: float, rule: str, baseline: str, draws: int = 20000) -> dict:
    selected = pair_results[(pair_results.cap_pp == cap) & (pair_results.rule == rule)].sort_values(["model", "reference"])
    base = pair_results[(pair_results.cap_pp == cap) & (pair_results.rule == baseline)].sort_values(["model", "reference"])
    if list(zip(selected.model, selected.reference)) != list(zip(base.model, base.reference)):
        raise ValueError("paired result keys mismatch")
    improvement = base.normalized_regret.to_numpy() - selected.normalized_regret.to_numpy()
    rng = np.random.default_rng(20260904)
    samples = rng.choice(improvement, size=(draws, len(improvement)), replace=True).mean(axis=1)
    base_mean = float(base.normalized_regret.mean())
    rule_mean = float(selected.normalized_regret.mean())
    rr = np.nan if abs(base_mean) < 1e-12 else 1 - rule_mean / base_mean
    return {
        "cap_pp": cap,
        "rule": rule,
        "baseline": baseline,
        "mean_improvement": float(improvement.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "regret_reduction": float(rr),
        "pairs_not_worse": int((improvement >= -1e-12).sum()),
    }


def decision(pair_results: pd.DataFrame, summary: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cap = 10.0
    candidate_rules = ("R1_frontier", "R2_near_best_95", "R3_sequential", "R4_kink_aware")
    baselines = ("B_global_cal", "B_reference_cal")
    base_summary = summary[(summary.cap_pp == cap) & summary.rule.isin(baselines)].sort_values("mean_normalized_regret")
    strongest = str(base_summary.iloc[0].rule)
    boot = pd.DataFrame([paired_bootstrap(pair_results, cap, rule, strongest) for rule in candidate_rules])
    rows = []
    for _, item in boot.iterrows():
        rule = str(item.rule)
        rs = summary[(summary.cap_pp == cap) & (summary.rule == rule)].iloc[0]
        bs = summary[(summary.cap_pp == cap) & (summary.rule == strongest)].iloc[0]
        paired = pair_results[(pair_results.cap_pp == cap) & pair_results.rule.isin([rule, strongest])]
        pivot = paired.pivot_table(index=["model", "reference"], columns="rule", values="normalized_regret")
        pair_improvement = pivot[strongest] - pivot[rule]
        ref_improvement = pair_improvement.groupby(level="reference").mean()
        loo_bidder_positive = sum(
            float(pair_improvement[pair_improvement.index.get_level_values("model") != model].mean()) >= -1e-12
            for model in MODELS
        )
        loo_reference_positive = sum(
            float(pair_improvement[pair_improvement.index.get_level_values("reference") != reference].mean()) >= -1e-12
            for reference in REFERENCES
        )
        sensitivity_positive = 0
        for other_cap in (5.0, 20.0):
            rs_other = summary[(summary.cap_pp == other_cap) & (summary.rule == rule)].iloc[0]
            bs_other = summary[(summary.cap_pp == other_cap) & (summary.rule == strongest)].iloc[0]
            sensitivity_positive += int(rs_other.mean_normalized_regret <= bs_other.mean_normalized_regret + 1e-12)
        gates = {
            "regret_gate": bool(item.regret_reduction >= 0.15 and item.ci_low > 0),
            "breadth_gate": bool(
                item.pairs_not_worse >= 10
                and int((ref_improvement >= -1e-12).sum()) >= 2
                and loo_bidder_positive == len(MODELS)
                and loo_reference_positive == len(REFERENCES)
            ),
            "delivery_gate": bool(rs.violation_rate <= bs.violation_rate + 0.05 and rs.mean_violation_severity_pp <= bs.mean_violation_severity_pp + 1e-9),
            "guardrail_gate": bool(sensitivity_positive >= 1),
            "proximity_gate": bool(rs.mean_value_capture >= 0.50 or rs.within_one_step_rate >= 0.70),
        }
        rows.append({
            **item.to_dict(),
            "references_not_worse": int((ref_improvement >= -1e-12).sum()),
            "loo_bidder_positive": loo_bidder_positive,
            "loo_reference_positive": loo_reference_positive,
            **gates,
            "all_gates": all(gates.values()),
        })
    gates = pd.DataFrame(rows).sort_values(["all_gates", "regret_reduction"], ascending=[False, False])
    full_pass = gates[gates.all_gates]
    if not full_pass.empty:
        outcome = "GO"
        frozen_rules = list(full_pass.rule.head(2))
        reason = "At least one transparent rule passes all retrospective publication-grade gates."
    else:
        promising = gates[(gates.mean_improvement > 0) & gates.delivery_gate & gates.breadth_gate]
        if not promising.empty:
            outcome = "CONDITIONAL_GO"
            frozen_rules = list(promising.rule.head(2))
            reason = "No rule passes every strong gate, but at least one improves regret broadly without worsening delivery."
        else:
            outcome = "STOP"
            frozen_rules = []
            reason = "Transparent rules do not beat the strongest calibration baseline broadly enough to justify a fresh challenge."
    return {"outcome": outcome, "strongest_baseline": strongest, "frozen_rules": frozen_rules, "reason": reason}, gates


def plot_outputs(summary: pd.DataFrame, selected: pd.DataFrame, pair_results: pd.DataFrame, label: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    order = ["B_fixed_0.8", "B_fixed_1", "B_global_cal", "B_reference_cal", "R1_frontier", "R2_near_best_95", "R3_sequential", "R4_kink_aware"]
    names = {"B_fixed_0.8": "Fixed 0.8", "B_fixed_1": "Fixed 1.0", "B_global_cal": "Global-Cal", "B_reference_cal": "Reference-Cal", "R1_frontier": "Frontier", "R2_near_best_95": "Near-best", "R3_sequential": "Sequential", "R4_kink_aware": "Kink-aware"}
    sub = summary[(summary.cap_pp == 10) & summary.rule.isin(order)].set_index("rule").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#9aa7b2"] * 4 + ["#2a9d8f", "#457b9d", "#e9c46a", "#e76f51"]
    ax.bar(range(len(sub)), sub.mean_normalized_regret, color=colors)
    ax.set_xticks(range(len(sub)), [names[x] for x in sub.index], rotation=25, ha="right")
    ax.set_ylabel("Budget-normalized unconstrained value regret")
    ax.set_title(f"κ-selection regret — {label}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_regret.png", dpi=220)
    fig.savefig(outdir / f"{label}_regret.pdf")
    plt.close(fig)

    rules = ["R1_frontier", "R2_near_best_95", "R3_sequential", "R4_kink_aware"]
    sel = selected[(selected.cap_pp == 10) & selected.rule.isin(rules)].copy()
    matrix = sel.pivot_table(index=["model", "reference"], columns="rule", values="selected_kappa").reindex(columns=rules)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(rules)), ["Frontier", "Near-best", "Sequential", "Kink-aware"])
    ax.set_yticks(range(len(matrix)), [f"{m.upper()}–{r.replace('_aware','').replace('_',' ').title()}" for m, r in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i,j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Selected κ")
    ax.set_title("Calibration-selected κ under the 10 pp guardrail")
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_selected_kappa.png", dpi=220)
    fig.savefig(outdir / f"{label}_selected_kappa.pdf")
    plt.close(fig)

    subp = pair_results[(pair_results.cap_pp == 10) & pair_results.rule.isin(order)].groupby("rule", as_index=False).agg(
        regret=("normalized_regret", "mean"), severity=("violation_severity_pp", "mean"), violation=("delivery_violation", "mean")
    ).set_index("rule").reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for rule, row in subp.iterrows():
        ax.scatter(row.severity, row.regret, s=80, color=colors[order.index(rule)])
        ax.annotate(names[rule], (row.severity, row.regret), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean delivery-violation severity (pp)")
    ax.set_ylabel("Budget-normalized regret")
    ax.set_title(f"Value regret versus service-risk — {label}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_regret_delivery.png", dpi=220)
    fig.savefig(outdir / f"{label}_regret_delivery.pdf")
    plt.close(fig)


def render_report(label: str, decision_data: dict, summary: pd.DataFrame, gates: pd.DataFrame, selected: pd.DataFrame, pair_results: pd.DataFrame, outdir: Path) -> None:
    primary = summary[(summary.cap_pp == 10) & summary.rule.isin(["B_fixed_0.8", "B_fixed_1", "B_global_cal", "B_reference_cal", "R1_frontier", "R2_near_best_95", "R3_sequential", "R4_kink_aware"])].copy()
    primary["mean_normalized_regret"] *= 1000
    primary["violation_rate"] *= 100
    primary["exact_oracle_rate"] *= 100
    primary["within_one_step_rate"] *= 100
    cols = ["rule", "mean_normalized_regret", "violation_rate", "mean_violation_severity_pp", "exact_oracle_rate", "within_one_step_rate", "mean_value_capture"]
    table = primary[cols].rename(columns={
        "rule": "Policy", "mean_normalized_regret": "Regret ×10³", "violation_rate": "Violation %", "mean_violation_severity_pp": "Severity (pp)", "exact_oracle_rate": "Oracle hit %", "within_one_step_rate": "Within-one-step %", "mean_value_capture": "Value capture"
    })
    gate_table = gates[["rule", "regret_reduction", "ci_low", "ci_high", "pairs_not_worse", "references_not_worse", "loo_bidder_positive", "loo_reference_positive", "regret_gate", "breadth_gate", "delivery_gate", "guardrail_gate", "proximity_gate", "all_gates"]].copy()
    gate_table.regret_reduction *= 100
    outcome_cn = {"GO": "通过：值得运行冻结规则的新随机性挑战", "CONDITIONAL_GO": "条件通过：有筛查价值，但尚未达到强规则门槛", "STOP": "停止：不值得继续把论文升级为 κ-selection rule"}[decision_data["outcome"]]
    scope = "已消费的 later-period retrospective screen" if label == "retrospective" else "新 CRN seeds 的 stochastic transport（不是新时间段）"
    doc = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>κ Selection Challenge</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f5f7fa;color:#1f2933}}main{{max-width:1180px;margin:auto;background:white;padding:42px 54px;box-shadow:0 2px 18px #d8dee6}}h1,h2{{color:#17324d}}.hero{{background:linear-gradient(135deg,#eaf4f4,#eef3fb);padding:24px;border-left:6px solid #2a9d8f}}.go{{font-size:1.35rem;font-weight:750;color:#0b6e4f}}.warn{{background:#fff7e6;padding:14px;border-left:5px solid #e9a23b}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0 28px}}th{{background:#dbe8f4}}th,td{{border:1px solid #c9d2dc;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #dde3ea;margin:8px 0 26px}}code{{background:#eef2f6;padding:2px 5px}}small{{color:#607080}}</style></head><body><main>
<h1>κ Selection Challenge：{html.escape(label)}</h1>
<div class='hero'><div class='go'>{html.escape(outcome_cn)}</div><p>{html.escape(decision_data['reason'])}</p><p>最强 calibration benchmark：<b>{html.escape(decision_data['strongest_baseline'])}</b>；冻结候选规则：<b>{html.escape(', '.join(decision_data['frozen_rules']) or '无')}</b>。</p></div>
<p class='warn'><b>证据范围：</b>{html.escape(scope)}。本报告不把已查看数据重新命名为 fresh holdout，也不根据结果修改 95%、3/4 或 10 pp 等阈值。</p>
<h2>主要结果（10 pp service guardrail）</h2>{table.to_html(index=False, float_format=lambda x:f'{x:.3f}', border=0)}
<img src='{label}_regret.png' alt='regret'>
<h2>预设门槛</h2>{gate_table.to_html(index=False, float_format=lambda x:f'{x:.3f}', border=0)}
<h2>规则选出的宽度</h2><img src='{label}_selected_kappa.png' alt='selected kappa'>
<h2>价值后悔与交付风险</h2><img src='{label}_regret_delivery.png' alt='regret delivery'>
<h2>如何解释</h2><ul><li>Value regret 衡量事后 unconstrained value oracle 与可部署规则之间的 value gap；它始终非负，并与 delivery violation 分开披露，从而避免人为设定 value–delivery 权重。</li><li>表内的 oracle-hit 指标仍以满足测试期 guardrail 的 feasible oracle 为参照；完整 CSV 另保留相对 feasible oracle 的 signed value gap。</li><li>Global-Cal 为所有 pair 使用同一 κ；Reference-Cal 允许三种 pacing architecture 各有一个 κ，是更强、也更公平的简单基准。</li><li>若透明规则不能稳定超过这些 calibration baselines，则论文应保留“如何选择更宽或更紧”的机制 insights，而不声称已经得到可靠的 deployment selector。</li></ul>
<small>Generated from frozen server artifacts in {ROOT}. Full pair-level CSVs and hashes are stored beside this report.</small>
</main></body></html>"""
    (outdir / f"{label}_selection_challenge.html").write_text(doc, encoding="utf-8")


def run(phase: str) -> None:
    cal = collect_calibration()
    validate_surface(cal, CAL_SEEDS, "calibration")
    if phase == "retrospective":
        test = collect_retrospective()
        seeds = TEST_SEEDS
    else:
        test = collect_fresh()
        seeds = FRESH_SEEDS
    validate_surface(test, seeds, phase)
    selected, pair_results, summary = evaluate_rules(cal, test)
    decision_data, gates = decision(pair_results, summary)
    outdir = ROOT / "06_results" / phase
    reportdir = ROOT / "09_reports" / phase
    outdir.mkdir(parents=True, exist_ok=True)
    reportdir.mkdir(parents=True, exist_ok=True)
    cal.to_csv(outdir / "calibration_surface.csv", index=False)
    test.to_csv(outdir / "evaluation_surface.csv", index=False)
    selected.to_csv(outdir / "selected_kappa.csv", index=False)
    pair_results.to_csv(outdir / "pair_level_results.csv", index=False)
    summary.to_csv(outdir / "rule_summary.csv", index=False)
    gates.to_csv(outdir / "success_gates.csv", index=False)
    (outdir / "decision.json").write_text(json.dumps(decision_data, indent=2) + "\n", encoding="utf-8")
    source_hashes = {}
    for path in sorted(set(map(Path, pd.concat([cal.source, test.source]).unique()))):
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (outdir / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_outputs(summary, selected, pair_results, phase, reportdir)
    render_report(phase, decision_data, summary, gates, selected, pair_results, reportdir)
    print(json.dumps(decision_data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("retrospective", "fresh_stochastic"), required=True)
    args = parser.parse_args()
    run(args.phase)


if __name__ == "__main__":
    main()
