#!/usr/bin/env python3
"""T9Sim auxiliary study: data visibility x pacing reference x control width.

The raw bidder is the upstream T9 two-tier learned bidder.  C1--C4 alter only
the information visible during training.  Campaign pacing references never
read the raw bidder action.  Generative truth is retained for evaluation only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from reference_policies import (
    DualPacingState,
    PIDPacingState,
    dual_pacing_reference,
    pid_pacing_reference,
)
from t9sim import censor, schema
from t9sim.pipeline import (
    AFT_SCALE,
    BID_GRID,
    PipelineConfig,
    _as_cat,
    funnel_population,
    pwin_aft,
    predict_ev,
    train_tier1,
    train_tier2_aft,
)


TRUTH_ONLY = {"p_click", "p_install", "p_payer", "e_ltv", "ev_truth", "lu7_competing_bid"}
ACTION_GRID = np.geomspace(0.01, 20.0, 96)


def quantile_bins(values: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic evaluation-only quantile bins.

    These bins are used solely to describe which opportunities a policy buys;
    they are never passed to the bidder, pacing reference, or budget allocator.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot bin an empty evaluation variable")
    inner = np.quantile(finite, np.arange(1, n_bins) / n_bins)
    edges = np.r_[-np.inf, np.maximum.accumulate(inner), np.inf]
    codes = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, n_bins - 1)
    return codes.astype(np.int8), edges


def mask_profile(mask: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    required = arrays["required"][mask]
    truth = arrays["ev_truth"][mask]
    return {
        "count": count,
        "true_expected_value_sum": float(np.sum(truth)),
        "true_expected_value_mean": float(np.mean(truth)),
        "true_efficiency_mean": float(np.mean(truth / np.maximum(required, 1e-12))),
        "required_price_mean": float(np.mean(required)),
        "bidder_score_mean": float(np.mean(arrays["bidder_score"][mask])),
        "p_click_mean": float(np.mean(arrays["p_click"][mask])),
        "p_install_mean": float(np.mean(arrays["p_install"][mask])),
        "p_payer_mean": float(np.mean(arrays["p_payer"][mask])),
        "expected_ltv_conditional_mean": float(np.mean(arrays["e_ltv"][mask])),
        "click_rate_realized": float(np.mean(arrays["click"][mask])),
        "install_rate_realized": float(np.mean(arrays["install"][mask])),
        "payer_rate_realized": float(np.mean(arrays["is_payer"][mask])),
        "ltv_realized_mean": float(np.mean(arrays["ltv_value"][mask])),
        "mean_step": float(np.mean(arrays["step"][mask])),
    }


def binned_profile(
    mask: np.ndarray,
    codes: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    label: str,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = max(int(mask.sum()), 1)
    for index in range(n_bins):
        cell = mask & (codes == index)
        profile = mask_profile(cell, arrays)
        rows.append({
            "dimension": label,
            "quintile": index + 1,
            "share_of_selected": float(profile.get("count", 0) / selected),
            **profile,
        })
    return rows


def migration_map(
    wide_only: np.ndarray,
    tight_only: np.ndarray,
    value_bin: np.ndarray,
    difficulty_bin: np.ndarray,
    arrays: dict[str, np.ndarray],
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value_q in range(n_bins):
        for difficulty_q in range(n_bins):
            cell = (value_bin == value_q) & (difficulty_bin == difficulty_q)
            wide = wide_only & cell
            tight = tight_only & cell
            rows.append({
                "true_value_quintile": value_q + 1,
                "acquisition_difficulty_quintile": difficulty_q + 1,
                "wide_only_count": int(wide.sum()),
                "tight_only_count": int(tight.sum()),
                "net_count": int(wide.sum() - tight.sum()),
                "wide_only_true_value": float(arrays["ev_truth"][wide].sum()),
                "tight_only_true_value": float(arrays["ev_truth"][tight].sum()),
                "net_true_value": float(
                    arrays["ev_truth"][wide].sum() - arrays["ev_truth"][tight].sum()
                ),
            })
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_priority(seed: int, row_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(row_ids, dtype=np.uint64)
    x = values ^ np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def choose_affordable(mask: np.ndarray, costs: np.ndarray, remaining: float, priority: np.ndarray) -> np.ndarray:
    keep = np.zeros(mask.size, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0 or remaining <= 0:
        return keep
    order = indices[np.argsort(priority[indices], kind="stable")]
    cumulative = 0.0
    for index in order:
        cost = float(costs[index])
        if cumulative + cost <= remaining + 1e-12:
            keep[index] = True
            cumulative += cost
    return keep


def profit_max_bid(ev_hat: np.ndarray, booster: xgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    pw = pwin_aft(booster, frame, BID_GRID)
    profit = (ev_hat[:, None] - BID_GRID[None, :]) * pw
    best = profit.argmax(axis=1)
    best_profit = profit[np.arange(len(frame)), best]
    return np.where(best_profit > 0, BID_GRID[best], 0.0)


def price_prediction(booster: xgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    prediction_ecpm = booster.predict(
        xgb.DMatrix(_as_cat(frame, schema.TIER2_FEATURES), enable_categorical=True)
    )
    return np.maximum(prediction_ecpm / 1000.0, 1e-9)


@dataclass
class CampaignState:
    budget: float
    remaining: float
    reference: float = 1.0
    pid: PIDPacingState = field(default_factory=PIDPacingState)
    dual: DualPacingState = field(default_factory=DualPacingState)
    spend_by_step: list[float] = field(default_factory=list)
    reference_by_review: list[float] = field(default_factory=list)


def traffic_reference(
    ev_cal: np.ndarray,
    price_cal: np.ndarray,
    *,
    current_count: int,
    target_spend: float,
    fallback: float,
) -> float:
    """Calibration-only spend response, rescaled by current opportunity count."""
    if current_count <= 0 or target_spend <= 0 or ev_cal.size == 0:
        return 0.0 if target_spend <= 0 else float(fallback)
    bids = ev_cal[:, None] * ACTION_GRID[None, :]
    spends = np.sum(np.where(bids >= price_cal[:, None], bids, 0.0), axis=0)
    predicted = spends * (float(current_count) / float(ev_cal.size))
    index = int(np.searchsorted(predicted, target_spend, side="left"))
    return float(ACTION_GRID[min(index, ACTION_GRID.size - 1)])


def prepare_models(
    master: pd.DataFrame,
    condition: str,
    *,
    seed: int,
    max_train: int | None,
) -> dict[str, Any]:
    pc = PipelineConfig(oracle=False)
    train = master[master["_day"] <= 14]
    calibration = master[(master["_day"] >= 15) & (master["_day"] <= 20)]
    evaluation = master[master["_day"] >= 21]
    if max_train is not None and len(train) > max_train:
        train = train.sample(max_train, random_state=seed)
    train_view = censor.view(train, condition)
    calibration_view = censor.view(calibration, condition)
    models = train_tier1(pc, train_view, calibration_view, condition)
    booster = train_tier2_aft(train_view, condition)
    ev_cal, *_ = predict_ev(pc, models, calibration)
    ev_eval, *_ = predict_ev(pc, models, evaluation)
    raw_bid = profit_max_bid(ev_eval, booster, evaluation)
    raw_action = np.divide(raw_bid, ev_eval, out=np.zeros_like(raw_bid), where=ev_eval > 1e-12)
    return {
        "pc": pc,
        "models": models,
        "booster": booster,
        "calibration": calibration,
        "evaluation": evaluation,
        "ev_cal": ev_cal,
        "ev_eval": ev_eval,
        "price_cal": price_prediction(booster, calibration),
        "raw_bid": raw_bid,
        "raw_action": raw_action,
        "training_rows": int(len(train)),
        "censored_master_rows": int(len(train_view)),
        "tier1_training_population": {
            stage: int(len(funnel_population(train_view, condition, stage)))
            for stage in ("click", "install", "payer", "spend")
        },
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(evaluation)),
    }


def policy_id(reference: str, kappa: float | None, budget_multiplier: float) -> str:
    if reference == "raw":
        return f"raw_b{budget_multiplier:g}"
    return f"{reference}_k{kappa:g}_b{budget_multiplier:g}"


def replay_policy(
    data: dict[str, Any],
    *,
    reference: str,
    kappa: float | None,
    budget_multiplier: float,
    seed: int,
    h: int,
    pid_params: dict[str, float],
    dual_params: dict[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    evaluation = data["evaluation"]
    calibration = data["calibration"]
    ev_eval = data["ev_eval"]
    raw_action = data["raw_action"]
    raw_bid = data["raw_bid"]
    required = np.maximum(
        evaluation["floor_price"].to_numpy(),
        evaluation["lu7_competing_bid"].to_numpy(),
    ) / 1000.0
    campaign_values = evaluation["campaign_id"].astype(str).to_numpy()
    steps = evaluation["_step"].to_numpy(dtype=int)
    row_ids = evaluation["_row_id"].to_numpy(dtype=np.uint64)
    priorities = stable_priority(seed, row_ids)
    campaigns = sorted(evaluation["campaign_id"].astype(str).unique())
    horizon = int(evaluation["_step"].max()) + 1

    cal_won_spend = (
        calibration.assign(_spend=np.where(calibration["won"].to_numpy() == 1,
                                            calibration["bid_price"].to_numpy() / 1000.0, 0.0))
        .groupby(calibration["campaign_id"].astype(str), observed=True)["_spend"].sum()
    )
    cal_days = max(int(calibration["_day"].nunique()), 1)
    eval_days = max(int(evaluation["_day"].nunique()), 1)
    fallback_per_row = float(cal_won_spend.sum()) / max(len(calibration), 1)
    eval_counts = pd.Series(campaign_values).value_counts()
    states: dict[str, CampaignState] = {}
    for campaign in campaigns:
        base = float(cal_won_spend.get(campaign, fallback_per_row * eval_counts.get(campaign, 0)))
        base *= float(eval_days) / float(cal_days)
        budget = max(base * budget_multiplier, 1e-6)
        states[campaign] = CampaignState(budget=budget, remaining=budget)

    cal_campaigns = calibration["campaign_id"].astype(str).to_numpy()
    cal_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for campaign in campaigns:
        mask = cal_campaigns == campaign
        cal_lookup[campaign] = (data["ev_cal"][mask], data["price_cal"][mask])

    won_all = np.zeros(len(evaluation), dtype=bool)
    executed_actions = np.zeros(len(evaluation), dtype=float)
    lower_binding = upper_binding = 0
    spend_by_step = np.zeros(horizon, dtype=float)
    value_by_step = np.zeros(horizon, dtype=float)

    for step in range(horizon):
        step_indices = np.flatnonzero(steps == step)
        if step_indices.size == 0:
            continue
        for campaign in campaigns:
            local = step_indices[campaign_values[step_indices] == campaign]
            if local.size == 0:
                continue
            state = states[campaign]
            if state.remaining <= 1e-9:
                continue
            if reference != "raw" and step % h == 0:
                remaining_steps = max(horizon - step, 1)
                target = state.remaining / remaining_steps
                if reference == "traffic_aware":
                    ev_cal, price_cal = cal_lookup[campaign]
                    state.reference = traffic_reference(
                        ev_cal, price_cal, current_count=local.size,
                        target_spend=target, fallback=1.0,
                    )
                elif reference == "pid":
                    state.reference = pid_pacing_reference(
                        state.pid, budget=state.budget, remaining=state.remaining,
                        time_step=step, horizon=horizon, fallback_action=1.0,
                        kp=pid_params["pid_kp"], ki=pid_params["pid_ki"],
                        kd=pid_params["pid_kd"], integral_limit=pid_params["pid_integral_limit"],
                        max_action=20.0,
                    )
                elif reference == "dual":
                    state.reference = dual_pacing_reference(
                        state.dual, budget=state.budget, remaining=state.remaining,
                        time_step=step, horizon=horizon, fallback_action=1.0,
                        eta=dual_params["dual_eta"],
                        log_shadow_limit=dual_params["dual_log_shadow_limit"],
                        max_action=20.0,
                    )
                else:
                    raise ValueError(reference)
                state.reference_by_review.append(float(state.reference))

            if reference == "raw":
                action = raw_action[local]
                bids = raw_bid[local]
            else:
                width = float(kappa)
                low = max(state.reference * (1.0 - width), 0.0)
                high = state.reference * (1.0 + width)
                raw_local = raw_action[local]
                lower_binding += int(np.sum(raw_local < low - 1e-12))
                upper_binding += int(np.sum(raw_local > high + 1e-12))
                action = np.clip(raw_local, low, high)
                bids = action * ev_eval[local]
            candidates = (bids > 0) & (bids >= required[local])
            selected = choose_affordable(candidates, bids, state.remaining, priorities[local])
            won_indices = local[selected]
            won_all[won_indices] = True
            executed_actions[local] = action
            step_spend = float(bids[selected].sum())
            state.remaining = max(state.remaining - step_spend, 0.0)
            state.spend_by_step.append(step_spend)
            spend_by_step[step] += step_spend
            value_by_step[step] += float(evaluation["ev_truth"].to_numpy()[won_indices].sum())

    bid_exec = executed_actions * ev_eval
    if reference == "raw":
        bid_exec = raw_bid
    spend = float(bid_exec[won_all].sum())
    expected_value = float(evaluation["ev_truth"].to_numpy()[won_all].sum())
    realized_ltv = float(evaluation["ltv_value"].to_numpy()[won_all].sum())
    total_budget = float(sum(state.budget for state in states.values()))
    remaining = float(sum(state.remaining for state in states.values()))
    true_efficiency = evaluation["ev_truth"].to_numpy() / np.maximum(required, 1e-12)
    references = [item for state in states.values() for item in state.reference_by_review]
    per_campaign = []
    for campaign, state in states.items():
        mask = campaign_values == campaign
        won = won_all & mask
        campaign_spend = float(bid_exec[won].sum())
        campaign_value = float(evaluation["ev_truth"].to_numpy()[won].sum())
        per_campaign.append({
            "campaign_id": campaign,
            "budget": state.budget,
            "spend": campaign_spend,
            "underdelivery": state.remaining / state.budget,
            "expected_value": campaign_value,
            "expected_roas": campaign_value / campaign_spend if campaign_spend > 0 else None,
            "wins": int(won.sum()),
        })
    result = {
        "policy_id": policy_id(reference, kappa, budget_multiplier),
        "reference": reference,
        "kappa": kappa,
        "budget_multiplier": budget_multiplier,
        "campaigns": len(states),
        "budget": total_budget,
        "spend": spend,
        "budget_usage": spend / total_budget,
        "underdelivery": remaining / total_budget,
        "expected_value": expected_value,
        "expected_roas": expected_value / spend if spend > 0 else None,
        "expected_surplus": expected_value - spend,
        "realized_ltv": realized_ltv,
        "realized_roas": realized_ltv / spend if spend > 0 else None,
        "wins": int(won_all.sum()),
        "win_rate": float(won_all.mean()),
        "clicks": int(evaluation["click"].to_numpy()[won_all].sum()),
        "installs": int(evaluation["install"].to_numpy()[won_all].sum()),
        "payers": int(evaluation["is_payer"].to_numpy()[won_all].sum()),
        "lower_binding_rate": lower_binding / len(evaluation) if reference != "raw" else None,
        "upper_binding_rate": upper_binding / len(evaluation) if reference != "raw" else None,
        "mean_won_true_efficiency": float(true_efficiency[won_all].mean()) if won_all.any() else None,
        "mean_reference": float(np.mean(references)) if references else None,
        "reference_log_volatility": float(np.std(np.log(np.maximum(references, 1e-9)))) if references else None,
        "spend_by_step": spend_by_step.tolist(),
        "expected_value_by_step": value_by_step.tolist(),
        "per_campaign": per_campaign,
    }
    return result, won_all


def add_replacement_diagnostics(
    results: list[dict[str, Any]],
    wins: dict[str, np.ndarray],
    evaluation: pd.DataFrame,
    bidder_score: np.ndarray,
) -> dict[str, Any]:
    truth = evaluation["ev_truth"].to_numpy()
    need = np.maximum(evaluation["floor_price"].to_numpy(), evaluation["lu7_competing_bid"].to_numpy()) / 1000.0
    step = evaluation["_step"].to_numpy()
    arrays = {
        "required": need,
        "ev_truth": truth,
        "bidder_score": np.asarray(bidder_score, dtype=float),
        "p_click": evaluation["p_click"].to_numpy(dtype=float),
        "p_install": evaluation["p_install"].to_numpy(dtype=float),
        "p_payer": evaluation["p_payer"].to_numpy(dtype=float),
        "e_ltv": evaluation["e_ltv"].to_numpy(dtype=float),
        "click": evaluation["click"].to_numpy(dtype=float),
        "install": evaluation["install"].to_numpy(dtype=float),
        "is_payer": evaluation["is_payer"].to_numpy(dtype=float),
        "ltv_value": evaluation["ltv_value"].to_numpy(dtype=float),
        "step": step.astype(float),
    }
    difficulty = need / np.maximum(truth, 1e-12)
    value_bin, value_edges = quantile_bins(truth)
    difficulty_bin, difficulty_edges = quantile_bins(difficulty)
    score_bin, score_edges = quantile_bins(np.asarray(bidder_score, dtype=float))
    pclick_bin, pclick_edges = quantile_bins(arrays["p_click"])
    ltv_bin, ltv_edges = quantile_bins(arrays["e_ltv"])
    lookup = {row["policy_id"]: row for row in results}
    for row in results:
        if row["reference"] == "raw" or float(row["kappa"]) == 0.0:
            continue
        base_id = policy_id(row["reference"], 0.0, row["budget_multiplier"])
        base = wins[base_id]
        wide = wins[row["policy_id"]]
        tight_only = base & ~wide
        wide_only = wide & ~base
        row["replacement_diagnostic"] = {
            "tight_only_count": int(tight_only.sum()),
            "wide_only_count": int(wide_only.sum()),
            "wide_to_tight_count_ratio": float(wide_only.sum() / max(tight_only.sum(), 1)),
            "wide_minus_tight_efficiency": float(
                (truth[wide_only] / np.maximum(need[wide_only], 1e-12)).mean()
                - (truth[tight_only] / np.maximum(need[tight_only], 1e-12)).mean()
            ) if wide_only.any() and tight_only.any() else None,
            "wide_minus_tight_mean_step": float(step[wide_only].mean() - step[tight_only].mean())
            if wide_only.any() and tight_only.any() else None,
        }
        row["delta_vs_reference"] = {
            key: (row[key] - lookup[base_id][key])
            for key in ("expected_value", "spend", "budget_usage", "underdelivery", "expected_surplus")
        }
        # The paper's information/acquisition claim is defined on the fixed
        # main contrast.  Keeping the detailed trace to kappa=0.8 avoids
        # manufacturing dozens of post-hoc subgroup comparisons.
        if np.isclose(float(row["kappa"]), 0.8):
            common = wide & base
            row["acquisition_mix_diagnostic"] = {
                "wide_only": mask_profile(wide_only, arrays),
                "tight_only": mask_profile(tight_only, arrays),
                "common": mask_profile(common, arrays),
                "wide_only_by_dimension": {
                    "true_expected_value": binned_profile(
                        wide_only, value_bin, arrays, label="true_expected_value"
                    ),
                    "acquisition_difficulty": binned_profile(
                        wide_only, difficulty_bin, arrays, label="acquisition_difficulty"
                    ),
                    "bidder_score": binned_profile(
                        wide_only, score_bin, arrays, label="bidder_score"
                    ),
                    "conversion_probability": binned_profile(
                        wide_only, pclick_bin, arrays, label="conversion_probability"
                    ),
                    "conditional_ltv": binned_profile(
                        wide_only, ltv_bin, arrays, label="conditional_ltv"
                    ),
                },
                "tight_only_by_dimension": {
                    "true_expected_value": binned_profile(
                        tight_only, value_bin, arrays, label="true_expected_value"
                    ),
                    "acquisition_difficulty": binned_profile(
                        tight_only, difficulty_bin, arrays, label="acquisition_difficulty"
                    ),
                    "bidder_score": binned_profile(
                        tight_only, score_bin, arrays, label="bidder_score"
                    ),
                    "conversion_probability": binned_profile(
                        tight_only, pclick_bin, arrays, label="conversion_probability"
                    ),
                    "conditional_ltv": binned_profile(
                        tight_only, ltv_bin, arrays, label="conditional_ltv"
                    ),
                },
                "value_by_difficulty_migration": migration_map(
                    wide_only, tight_only, value_bin, difficulty_bin, arrays
                ),
            }
    return {
        "bin_source": "held-out evaluation population; descriptive truth audit only",
        "true_expected_value_edges": value_edges.tolist(),
        "acquisition_difficulty_edges": difficulty_edges.tolist(),
        "bidder_score_edges": score_edges.tolist(),
        "conversion_probability_edges": pclick_edges.tolist(),
        "conditional_ltv_edges": ltv_edges.tolist(),
        "truth_or_bin_codes_used_by_policy": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--references", nargs="+", default=["traffic_aware", "pid", "dual"])
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.8, 1.0, 1.2])
    parser.add_argument("--budget-multipliers", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--seed", type=int, default=90213)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    pf_columns = set(pq.ParquetFile(args.parquet).schema_arrow.names)
    needed = sorted(schema.NEEDED.intersection(pf_columns))
    master = pd.read_parquet(args.parquet, columns=needed)
    master = master.sort_values(["timestamp", "campaign_id"], kind="stable").reset_index(drop=True)
    master["_row_id"] = np.arange(len(master), dtype=np.uint64)
    t0 = int(master["timestamp"].min())
    master["_day"] = ((master["timestamp"] - t0) // 86400 + 1).astype(int)
    master["_step"] = ((master["_day"] - 21) * 24 + master["hour_of_day"]).astype(int)
    if args.max_eval is not None:
        eval_index = master.index[master["_day"] >= 21]
        keep_eval = set(eval_index[: args.max_eval].tolist())
        master = master[(master["_day"] < 21) | master.index.isin(keep_eval)].copy()
    for column in PipelineConfig().t1_cat:
        if column in master:
            master[column] = master[column].astype("category")

    pid_params = {"pid_kp": 4.0, "pid_ki": 0.5, "pid_kd": 0.0, "pid_integral_limit": 1.0}
    dual_params = {"dual_eta": 10.0, "dual_log_shadow_limit": math.log(20.0)}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "T9Sim auxiliary data-integration x reference architecture",
        "parquet": str(args.parquet.resolve()),
        "parquet_sha256": sha256_file(args.parquet),
        "seed": args.seed,
        "split": {"train": "days 1-14", "calibration": "days 15-20", "evaluation": "days 21-28"},
        "truth_only_columns": sorted(TRUTH_ONLY),
        "truth_usage": "evaluation only",
        "pid_params_inherited_from_auctionnet_p7": pid_params,
        "dual_params_inherited_from_auctionnet_p7": dual_params,
        "conditions": {},
    }
    for condition in args.conditions:
        condition_started = time.time()
        data = prepare_models(master, condition, seed=args.seed, max_train=args.max_train)
        results: list[dict[str, Any]] = []
        wins: dict[str, np.ndarray] = {}
        for budget_multiplier in args.budget_multipliers:
            for reference in args.references:
                for kappa in args.kappas:
                    result, mask = replay_policy(
                        data, reference=reference, kappa=kappa,
                        budget_multiplier=budget_multiplier, seed=args.seed,
                        h=args.h, pid_params=pid_params, dual_params=dual_params,
                    )
                    results.append(result)
                    wins[result["policy_id"]] = mask
            raw_result, raw_mask = replay_policy(
                data, reference="raw", kappa=None,
                budget_multiplier=budget_multiplier, seed=args.seed,
                h=args.h, pid_params=pid_params, dual_params=dual_params,
            )
            results.append(raw_result)
            wins[raw_result["policy_id"]] = raw_mask
        acquisition_bin_audit = add_replacement_diagnostics(
            results, wins, data["evaluation"], data["ev_eval"]
        )
        payload["conditions"][condition] = {
            "training_rows": data["training_rows"],
            "censored_master_rows": data["censored_master_rows"],
            "tier1_training_population": data["tier1_training_population"],
            "calibration_rows": data["calibration_rows"],
            "evaluation_rows": data["evaluation_rows"],
            "acquisition_bin_audit": acquisition_bin_audit,
            "elapsed_seconds": time.time() - condition_started,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"CONDITION_DONE {condition} seconds={time.time() - condition_started:.1f}", flush=True)
    payload["elapsed_seconds"] = time.time() - started
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OUTPUT_WRITTEN {args.output}", flush=True)


if __name__ == "__main__":
    main()
