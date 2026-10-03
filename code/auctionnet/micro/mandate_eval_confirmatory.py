#!/usr/bin/env python3
"""Auditable AuctionNet replay for fixed delegated-control mandates.

The evaluator intentionally does not import the historical repository's
OfflineEnv.  It preserves its auction rule (alpha * observed value competes
with leastWinningCost), but uses common random numbers across configurations
and a single explicit budget-enforcement rule.  This removes branch-dependent
conversion redraws in the historical evaluator and makes mechanism comparisons
paired at the impression level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

COMMON_ROOT = Path(__file__).resolve().parents[2] / "common"
BIDDER_ROOT = Path(__file__).resolve().parents[2] / "bidders"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from reference_policies import (
    DualPacingState,
    PIDPacingState,
    dual_pacing_reference,
    pid_pacing_reference,
)

from mandate_core_v2 import (
    Mandate,
    derive_intervention_masks,
    enforce_action,
    enforce_asymmetric_action,
    estimate_reference_action,
    estimate_response_aware_reference_action,
    estimate_smoothed_controller_reference_action,
    estimate_traffic_aware_reference_action,
    expand_cartesian_spec,
    implied_block_share,
    nips_score,
    realized_autonomy,
)


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return NumpyCompatUnpickler(handle).load()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-pickle", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--model", choices=["cql", "dt", "constant"], default="cql")
    parser.add_argument("--checkpoint")
    parser.add_argument("--normalizer")
    parser.add_argument("--dt-code-root", default=str(BIDDER_ROOT / "sembid_cpa"),
                        help="Path containing bidding_train_env/ (DecisionTransformer source).")
    parser.add_argument("--constant-action", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-json", help="JSON list of configuration objects")
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.3)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--policy-mode", choices=["raw", "reference", "hard"], default="hard")
    parser.add_argument(
        "--reference-mode",
        choices=["pacing", "traffic_aware", "pid", "dual", "smoothed_controller", "response_aware"],
        default="pacing",
    )
    parser.add_argument(
        "--autonomy-operator",
        choices=["relative_hard", "log_hard", "soft_blend"],
        default="relative_hard",
    )
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--cpa-multiplier", type=float, default=1.0)
    parser.add_argument("--market-volatility", type=float, default=0.0)
    parser.add_argument("--valuation-noise", type=float, default=0.0)
    parser.add_argument("--valuation-bias", type=float, default=0.0)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--max-action", type=float, default=100.0)
    parser.add_argument("--budget-control", choices=["random_priority", "cheapest_first"], default="random_priority")
    parser.add_argument("--conversion-seed", type=int, default=20260428)
    parser.add_argument("--market-seed", type=int, default=20260810)
    parser.add_argument("--max-advertisers", type=int)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--max-pvs", type=int)
    parser.add_argument(
        "--pv-order-seed",
        type=int,
        help="Optional deterministic within-tick hash ordering used only for post-confirmatory supply diagnostics",
    )
    parser.add_argument(
        "--post-confirmatory-period8",
        action="store_true",
        help="Explicitly mark a diagnostic run after the sealed Period-8 study has already been opened",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanism-output", help="Optional JSONL mechanism records by advertiser and step")
    return parser.parse_args()


def mean_arrays(values: Sequence[np.ndarray]) -> float:
    total, count = 0.0, 0
    for value in values:
        array = np.asarray(value)
        total += float(np.sum(array))
        count += int(array.size)
    return total / count if count else 0.0


def build_state(
    time_step: int,
    observed_values: np.ndarray,
    remaining_budget: float,
    budget: float,
    history: dict[str, list[np.ndarray]],
) -> np.ndarray:
    last = slice(max(0, len(history["bids"]) - 3), None)
    return np.asarray(
        [
            (48 - time_step) / 48.0,
            remaining_budget / budget if budget > 0 else 0.0,
            mean_arrays(history["bids"]),
            mean_arrays(history["bids"][last]),
            mean_arrays(history["lwc"]),
            mean_arrays(history["observed_values"]),
            mean_arrays(history["conversions"]),
            mean_arrays(history["wins"]),
            mean_arrays(history["lwc"][last]),
            mean_arrays(history["observed_values"][last]),
            mean_arrays(history["conversions"][last]),
            mean_arrays(history["wins"][last]),
            float(np.mean(observed_values)) if observed_values.size else 0.0,
            int(observed_values.size),
            sum(x.size for x in history["bids"][last]),
            sum(x.size for x in history["bids"]),
        ],
        dtype=np.float32,
    )


def normalize_state(state: np.ndarray, normalizer: dict) -> np.ndarray:
    out = state.astype(np.float32).copy()
    if "state_mean" in normalizer and "state_std" in normalizer:
        return (out - np.asarray(normalizer["state_mean"], dtype=np.float32)) / (
            np.asarray(normalizer["state_std"], dtype=np.float32) + 1e-8
        )
    for key, value in normalizer.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        low, high = float(value.get("min", 0.0)), float(value.get("max", 0.0))
        out[index] = (out[index] - low) / (high - low) if high > low else 0.0
    return out


class FastController:
    def __init__(self, args: argparse.Namespace):
        self.model = args.model
        self.constant_action = args.constant_action
        self.device = torch.device(args.device)
        self.policy = None
        self.normalizer: dict = {}
        if self.model == "cql":
            if not args.checkpoint or not args.normalizer:
                raise ValueError("CQL requires --checkpoint and --normalizer")
            self.normalizer = load_pickle(Path(args.normalizer))
            self.policy = torch.jit.load(args.checkpoint, map_location=self.device)
            self.policy.eval()

        if self.model == "dt":
            if not args.checkpoint or not args.normalizer:
                raise ValueError("DT requires --checkpoint and --normalizer")
            code_root = Path(args.dt_code_root)
            if str(code_root) not in sys.path:
                sys.path.insert(0, str(code_root))
            from bidding_train_env.baseline.dt.dt import DecisionTransformer

            self.normalizer = load_pickle(Path(args.normalizer))
            self.policy = DecisionTransformer(
                state_dim=16,
                act_dim=1,
                state_mean=self.normalizer["state_mean"],
                state_std=self.normalizer["state_std"],
                action_tanh=False,
                K=20,
                max_ep_len=96,
                scale=3000,
                target_return=8,
            )
            self.policy.load_net(args.checkpoint, device=str(self.device))
            self.policy = self.policy.to(self.device)
            self.policy.state_mean = self.policy.state_mean.to(self.device)
            self.policy.state_std = self.policy.state_std.to(self.device)
            self.policy.device = str(self.device)
            self.policy.eval()

    def reset(self) -> None:
        if self.model == "dt":
            self.policy.init_eval()

    def action(self, state: np.ndarray, previous_reward: float | None = None) -> float:
        if self.model == "constant":
            return float(self.constant_action)
        if self.model == "dt":
            action = self.policy.take_actions(state) if previous_reward is None else self.policy.take_actions(
                state, pre_reward=previous_reward
            )
            return max(float(np.asarray(action).reshape(-1)[0]), 0.0)
        normalized = normalize_state(state, self.normalizer)
        with torch.no_grad():
            tensor = torch.as_tensor(normalized.reshape(1, -1), dtype=torch.float32, device=self.device)
            output = self.policy(tensor)
        return max(float(output.detach().cpu().numpy().reshape(-1)[0]), 0.0)


def prepare_episodes(data_pickle: Path, max_advertisers: int | None) -> list[dict[str, Any]]:
    started = time.time()
    frame = load_pickle(data_pickle)
    required = {
        "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
        "timeStepIndex", "pValue", "pValueSigma", "leastWinningCost",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"data cache missing columns: {sorted(missing)}")
    # Avoid a second multi-gigabyte full-frame copy.  AuctionNet rows need only
    # be ordered within an advertiser episode, so sorting each group is enough.
    grouped = frame.groupby(["deliveryPeriodIndex", "advertiserNumber"], sort=True)
    episodes: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(grouped):
        if max_advertisers is not None and index >= max_advertisers:
            break
        group = group.sort_values("timeStepIndex", kind="stable")
        steps = []
        for _, tick in group.groupby("timeStepIndex", sort=True):
            steps.append(
                {
                    "p": tick["pValue"].to_numpy(dtype=np.float64, copy=True),
                    "sigma": tick["pValueSigma"].to_numpy(dtype=np.float64, copy=True),
                    "lwc": tick["leastWinningCost"].to_numpy(dtype=np.float64, copy=True),
                }
            )
        episodes.append(
            {
                "key": [int(key[0]), int(key[1])],
                "category": int(group["advertiserCategoryIndex"].iloc[0]),
                "budget": float(group["budget"].iloc[0]),
                "cpa": float(group["CPAConstraint"].iloc[0]),
                "steps": steps,
            }
        )
    print(f"DATA_READY episodes={len(episodes)} seconds={time.time() - started:.2f}", flush=True)
    return episodes


def make_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.grid_json:
        raw_configs = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
        configs = expand_cartesian_spec(raw_configs) if isinstance(raw_configs, dict) else raw_configs
        if not isinstance(configs, list) or not configs:
            raise ValueError("grid-json must contain a nonempty list or Cartesian specification")
    else:
        configs = [
            {
                "id": "single",
                "q_scale": args.q_scale,
                "kappa": args.kappa,
                "h": args.h,
                "policy_mode": args.policy_mode,
                "budget_multiplier": args.budget_multiplier,
                "cpa_multiplier": args.cpa_multiplier,
                "market_volatility": args.market_volatility,
                "valuation_noise": args.valuation_noise,
                "valuation_bias": args.valuation_bias,
            }
        ]
    defaults = {
        "q_scale": args.q_scale,
        "kappa": args.kappa,
        "h": args.h,
        "policy_mode": args.policy_mode,
        "budget_multiplier": args.budget_multiplier,
        "cpa_multiplier": args.cpa_multiplier,
        "market_volatility": args.market_volatility,
        "valuation_noise": args.valuation_noise,
        "valuation_bias": args.valuation_bias,
        "reference_mode": args.reference_mode,
        "autonomy_operator": args.autonomy_operator,
    }
    output = []
    for index, raw in enumerate(configs):
        value = {**defaults, **raw}
        value.setdefault("id", f"config_{index:03d}")
        Mandate(float(value["q_scale"]), float(value["kappa"]), int(value["h"]))
        if "kappa_down" in value or "kappa_up" in value:
            down = float(value.get("kappa_down", value["kappa"]))
            up = float(value.get("kappa_up", value["kappa"]))
            if down < 0 or up < 0:
                raise ValueError("kappa_down and kappa_up must be nonnegative")
        if value["policy_mode"] not in {"raw", "reference", "hard"}:
            raise ValueError(f"invalid policy_mode: {value['policy_mode']}")
        if value["reference_mode"] not in {"pacing", "traffic_aware", "pid", "dual", "smoothed_controller", "response_aware"}:
            raise ValueError(f"invalid reference_mode: {value['reference_mode']}")
        if value["autonomy_operator"] not in {"relative_hard", "log_hard", "soft_blend"}:
            raise ValueError(f"invalid autonomy_operator: {value['autonomy_operator']}")
        output.append(value)
    return output


def stable_hash_order(size: int, seed: int, episode_key: Sequence[int], time_step: int) -> np.ndarray:
    """Return a deterministic pseudo-random permutation of canonical PV positions."""

    if size < 0:
        raise ValueError("size must be nonnegative")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    context = hashlib.sha256(
        f"{int(seed)}|{list(episode_key)}|{int(time_step)}".encode("utf-8")
    ).digest()
    offset = np.uint64(int.from_bytes(context[:8], "big"))
    values = np.arange(size, dtype=np.uint64) + offset
    values = (values + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    values = values ^ (values >> np.uint64(31))
    return np.argsort(values, kind="stable").astype(np.int64)


def perturb_values(true_p: np.ndarray, z: np.ndarray, noise: float, bias: float) -> np.ndarray:
    clipped = np.clip(true_p, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-(logits + float(bias) + float(noise) * z)))


def budget_mask(wins: np.ndarray, costs: np.ndarray, remaining: float, priority: np.ndarray, mode: str) -> np.ndarray:
    keep = wins.copy()
    if float(np.sum(costs[keep])) <= remaining + 1e-9:
        return keep
    indices = np.flatnonzero(keep)
    if mode == "random_priority":
        order = indices[np.argsort(priority[indices], kind="stable")]
    else:
        order = indices[np.argsort(costs[indices], kind="stable")]
    keep[:] = False
    cumulative = 0.0
    for index in order:
        item_cost = float(costs[index])
        if cumulative + item_cost <= remaining + 1e-9:
            keep[index] = True
            cumulative += item_cost
    return keep


def evaluate_config(
    args: argparse.Namespace,
    config: dict[str, Any],
    episodes: Sequence[dict[str, Any]],
    controller: FastController,
    mechanism_handle=None,
) -> dict[str, Any]:
    mandate = Mandate(float(config["q_scale"]), float(config["kappa"]), int(config["h"]))
    conversion_rng = np.random.default_rng(args.conversion_seed)
    budget_rng = np.random.default_rng(args.conversion_seed + 100003)
    valuation_rng = np.random.default_rng(args.market_seed + 200003)
    market_rng = np.random.default_rng(args.market_seed)
    per_agent = []
    all_raw_actions: list[float] = []
    all_executed_actions: list[float] = []
    all_reference_actions: list[float] = []
    all_realized_autonomy: list[float] = []
    all_control_intensity: list[float] = []
    projection_count, projection_distance, action_count = 0, 0.0, 0
    total_reviews, total_pvs, total_wins = 0, 0, 0
    aggregate_score = aggregate_conversion = aggregate_expected = aggregate_cost = aggregate_budget = 0.0
    cpa_exceed_count = budget_exceed_count = 0

    for episode in episodes:
        controller.reset()
        budget = float(episode["budget"]) * float(config["budget_multiplier"])
        cpa_limit = float(episode["cpa"]) * float(config["cpa_multiplier"])
        remaining = budget
        history: dict[str, list[np.ndarray]] = {
            "bids": [], "lwc": [], "observed_values": [], "true_values": [], "conversions": [], "wins": [],
            "step_costs": []
        }
        pid_reference_state = PIDPacingState()
        dual_reference_state = DualPacingState()
        current_reference = cpa_limit * mandate.q_scale
        agent_raw: list[float] = []
        agent_executed: list[float] = []
        agent_reference: list[float] = []
        agent_projection = agent_reviews = agent_pvs = agent_wins = 0
        agent_projection_distance = agent_expected = agent_cost = agent_conversion = 0.0
        agent_early_cost = 0.0
        agent_spend_path_errors: list[float] = []
        agent_exhaustion_time: int | None = None
        num_steps = len(episode["steps"])
        if args.max_timesteps is not None:
            num_steps = min(num_steps, args.max_timesteps)
        market_z = market_rng.standard_normal(num_steps)

        for time_step, step in enumerate(episode["steps"][:num_steps]):
            true_p = step["p"]
            base_lwc = step["lwc"]
            full_size = true_p.size
            valuation_z = valuation_rng.standard_normal(full_size)
            conversion_uniform = conversion_rng.random(full_size)
            budget_priority = budget_rng.random(full_size)
            if args.pv_order_seed is not None:
                order = stable_hash_order(full_size, args.pv_order_seed, episode["key"], time_step)
                if args.max_pvs is not None:
                    order = order[: args.max_pvs]
                true_p = true_p[order]
                base_lwc = base_lwc[order]
                valuation_z = valuation_z[order]
                conversion_uniform = conversion_uniform[order]
                budget_priority = budget_priority[order]
            elif args.max_pvs is not None:
                true_p = true_p[: args.max_pvs]
                base_lwc = base_lwc[: args.max_pvs]
                valuation_z = valuation_z[: args.max_pvs]
                conversion_uniform = conversion_uniform[: args.max_pvs]
                budget_priority = budget_priority[: args.max_pvs]
            observed_p = perturb_values(
                true_p, valuation_z, float(config["valuation_noise"]), float(config["valuation_bias"])
            )
            volatility = float(config["market_volatility"])
            market_factor = float(np.exp(volatility * market_z[time_step] - 0.5 * volatility * volatility))
            lwc = base_lwc * market_factor
            control_eligible = remaining >= 0.1
            if not control_eligible:
                raw_action = executed_action = 0.0
                projected = False
                distance = 0.0
            else:
                state = build_state(time_step, observed_p, remaining, budget, history)
                previous_reward = float(np.sum(history["conversions"][-1])) if history["conversions"] else None
                raw_action = controller.action(state, previous_reward=previous_reward)
                if time_step % mandate.h == 0:
                    remaining_steps = max(48 - time_step, 1)
                    target_per_step = mandate.q_scale * remaining / remaining_steps
                    reference_mode = config["reference_mode"]
                    reference_arguments = {
                        "target_spend_per_step": target_per_step,
                        "current_pv_count": observed_p.size,
                        "historical_observed_values": history["observed_values"],
                        "historical_lwc": history["lwc"],
                        "fallback_action": cpa_limit * mandate.q_scale,
                        "lookback": args.lookback,
                        "max_action": args.max_action,
                    }
                    if reference_mode == "pacing":
                        current_reference = estimate_reference_action(**reference_arguments)
                    elif reference_mode == "traffic_aware":
                        current_reference = estimate_traffic_aware_reference_action(**reference_arguments)
                    elif reference_mode == "pid":
                        current_reference = pid_pacing_reference(
                            pid_reference_state,
                            budget=budget,
                            remaining=remaining,
                            time_step=time_step,
                            horizon=num_steps,
                            fallback_action=cpa_limit * mandate.q_scale,
                            kp=float(config.get("pid_kp", 2.0)),
                            ki=float(config.get("pid_ki", 0.2)),
                            kd=float(config.get("pid_kd", 0.0)),
                            integral_limit=float(config.get("pid_integral_limit", 1.0)),
                            max_action=args.max_action,
                        )
                    elif reference_mode == "dual":
                        current_reference = dual_pacing_reference(
                            dual_reference_state,
                            budget=budget,
                            remaining=remaining,
                            time_step=time_step,
                            horizon=num_steps,
                            fallback_action=cpa_limit * mandate.q_scale,
                            eta=float(config.get("dual_eta", 5.0)),
                            log_shadow_limit=float(config.get("dual_log_shadow_limit", np.log(20.0))),
                            max_action=args.max_action,
                        )
                    elif reference_mode == "smoothed_controller":
                        current_reference = estimate_smoothed_controller_reference_action(
                            historical_raw_actions=agent_raw,
                            fallback_action=cpa_limit * mandate.q_scale,
                            lookback=max(args.lookback, mandate.h),
                            max_action=args.max_action,
                            target_spend_per_step=target_per_step,
                            historical_step_costs=history["step_costs"],
                        )
                    else:
                        current_reference = estimate_response_aware_reference_action(
                            target_spend_per_step=target_per_step,
                            current_pv_count=observed_p.size,
                            historical_observed_values=history["observed_values"],
                            historical_lwc=history["lwc"],
                            historical_value_estimates=history["observed_values"],
                            cpa_limit=cpa_limit,
                            fallback_action=cpa_limit * mandate.q_scale,
                            lookback=max(args.lookback, 6),
                            max_action=args.max_action,
                        )
                    agent_reviews += 1
                if config["policy_mode"] == "raw":
                    executed_action, projected, distance = raw_action, False, 0.0
                elif config["policy_mode"] == "reference":
                    executed_action = current_reference
                    projected = abs(executed_action - raw_action) > 1e-10
                    distance = abs(executed_action - raw_action)
                else:
                    if "kappa_down" in config or "kappa_up" in config:
                        executed_action, projected, distance = enforce_asymmetric_action(
                            raw_action,
                            current_reference,
                            float(config.get("kappa_down", mandate.kappa)),
                            float(config.get("kappa_up", mandate.kappa)),
                        )
                    else:
                        executed_action, projected, distance = enforce_action(
                            raw_action,
                            current_reference,
                            mandate.kappa,
                            operator=config["autonomy_operator"],
                        )

            action_autonomy, control_intensity = realized_autonomy(
                raw_action, current_reference, executed_action
            )

            bids = executed_action * observed_p
            raw_bids = raw_action * observed_p
            raw_initial_wins = raw_bids >= lwc
            initial_wins = bids >= lwc
            initial_costs = lwc * initial_wins
            wins = budget_mask(initial_wins, initial_costs, remaining, budget_priority, args.budget_control)
            raw_initial_costs = lwc * raw_initial_wins
            raw_budget_wins = budget_mask(
                raw_initial_wins,
                raw_initial_costs,
                remaining,
                budget_priority,
                args.budget_control,
            )
            remaining_before = remaining
            costs = lwc * wins
            conversions = ((conversion_uniform < true_p) & wins).astype(np.float64)
            intervention_masks = derive_intervention_masks(
                raw_initial_wins, initial_wins, raw_budget_wins, wins
            )
            true_efficiency = true_p / np.maximum(lwc, 1e-12)
            observed_efficiency = observed_p / np.maximum(lwc, 1e-12)
            step_cost = float(np.sum(costs))
            remaining = max(remaining - step_cost, 0.0)
            cumulative_spend = budget - remaining
            target_progress_after_step = min(float(time_step + 1) / max(float(num_steps), 1.0), 1.0)
            actual_progress_after_step = cumulative_spend / budget if budget > 0 else 0.0
            agent_spend_path_errors.append(actual_progress_after_step - target_progress_after_step)
            if time_step < 24:
                agent_early_cost += step_cost
            if agent_exhaustion_time is None and remaining < 0.1:
                agent_exhaustion_time = time_step
            agent_cost += step_cost
            agent_conversion += float(np.sum(conversions))
            agent_expected += float(np.sum(true_p * wins))
            agent_pvs += int(true_p.size)
            agent_wins += int(np.sum(wins))
            agent_projection += int(projected)
            agent_projection_distance += float(distance)
            if control_eligible:
                agent_raw.append(float(raw_action))
                agent_executed.append(float(executed_action))
                agent_reference.append(float(current_reference))
                all_realized_autonomy.append(action_autonomy)
                all_control_intensity.append(control_intensity)
            if mechanism_handle is not None:
                def mask_metrics(mask_name: str) -> dict[str, Any]:
                    mask = intervention_masks[mask_name]
                    return {
                        f"{mask_name}_count": int(np.sum(mask)),
                        f"{mask_name}_cost": float(np.sum(lwc[mask])),
                        f"{mask_name}_true_expected_value": float(np.sum(true_p[mask])),
                        f"{mask_name}_observed_expected_value": float(np.sum(observed_p[mask])),
                        f"{mask_name}_mean_true_efficiency": (
                            float(np.mean(true_efficiency[mask])) if np.any(mask) else None
                        ),
                        f"{mask_name}_mean_observed_efficiency": (
                            float(np.mean(observed_efficiency[mask])) if np.any(mask) else None
                        ),
                    }

                intervention_metrics: dict[str, Any] = {}
                for mask_name in intervention_masks:
                    intervention_metrics.update(mask_metrics(mask_name))
                mechanism_handle.write(
                    json.dumps(
                        {
                            "config_id": config["id"],
                            "episode_key": episode["key"],
                            "time_step": time_step,
                            "raw_action": float(raw_action),
                            "executed_action": float(executed_action),
                            "reference_action": float(current_reference),
                            "intervened": bool(projected),
                            "intervention_magnitude": float(distance),
                            "normalized_intervention_magnitude": float(distance / max(raw_action, 1e-8)),
                            "realized_autonomy": action_autonomy,
                            "control_intensity": control_intensity,
                            "control_eligible": control_eligible,
                            "pvs": int(true_p.size),
                            "raw_initial_wins": int(np.sum(raw_initial_wins)),
                            "executed_initial_wins": int(np.sum(initial_wins)),
                            "raw_budget_wins": int(np.sum(raw_budget_wins)),
                            "executed_budget_wins": int(np.sum(wins)),
                            **intervention_metrics,
                            "executed_cost": step_cost,
                            "executed_expected_value": float(np.sum(true_p * wins)),
                            "starting_budget": budget,
                            "remaining_before": remaining_before,
                            "remaining_budget": remaining,
                            "cumulative_spend": cumulative_spend,
                            "early_period": bool(time_step < 24),
                            "is_budget_exhausted": bool(remaining < 0.1),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            history["bids"].append(bids)
            history["lwc"].append(lwc)
            history["observed_values"].append(observed_p)
            history["true_values"].append(true_p)
            history["conversions"].append(conversions)
            history["wins"].append(wins.astype(np.float64))
            history["step_costs"].append(step_cost)

        cpa_real = agent_cost / (agent_conversion + 1e-10) if agent_cost > 0 else 0.0
        score = nips_score(agent_conversion, agent_cost, cpa_limit)
        cpa_exceeded = cpa_real > cpa_limit
        budget_exceeded = agent_cost > budget + 1e-6
        aggregate_score += score
        aggregate_conversion += agent_conversion
        aggregate_expected += agent_expected
        aggregate_cost += agent_cost
        aggregate_budget += budget
        cpa_exceed_count += int(cpa_exceeded)
        budget_exceed_count += int(budget_exceeded)
        all_raw_actions.extend(agent_raw)
        all_executed_actions.extend(agent_executed)
        all_reference_actions.extend(agent_reference)
        projection_count += agent_projection
        projection_distance += agent_projection_distance
        action_count += len(agent_raw)
        total_reviews += agent_reviews
        total_pvs += agent_pvs
        total_wins += agent_wins
        per_agent.append(
            {
                "key": episode["key"],
                "category": episode["category"],
                "budget": budget,
                "cpa_limit": cpa_limit,
                "cost": agent_cost,
                "realized_conversions": agent_conversion,
                "expected_value": agent_expected,
                "cpa_real": cpa_real,
                "score": score,
                "budget_usage": agent_cost / budget if budget else 0.0,
                "underdelivery": max(budget - agent_cost, 0.0) / budget if budget else 0.0,
                "cpa_exceeded": cpa_exceeded,
                "budget_exceeded": budget_exceeded,
                "mean_raw_action": float(np.mean(agent_raw)) if agent_raw else 0.0,
                "mean_executed_action": float(np.mean(agent_executed)) if agent_executed else 0.0,
                "mean_reference_action": float(np.mean(agent_reference)) if agent_reference else 0.0,
                "spend_path_rmse": float(np.sqrt(np.mean(np.square(agent_spend_path_errors)))) if agent_spend_path_errors else 0.0,
                "reference_log_volatility": float(np.mean(np.abs(np.diff(np.log(np.maximum(agent_reference, 1e-8)))))) if len(agent_reference) > 1 else 0.0,
                "projection_rate": agent_projection / max(len(agent_raw), 1),
                "mean_realized_autonomy": float(np.mean([
                    realized_autonomy(raw, ref, executed)[0]
                    for raw, ref, executed in zip(agent_raw, agent_reference, agent_executed)
                ])) if agent_raw else 1.0,
                "early_spend_share": agent_early_cost / budget if budget else 0.0,
                "budget_exhaustion_time": agent_exhaustion_time,
                "review_count": agent_reviews,
                "win_rate": agent_wins / max(agent_pvs, 1),
            }
        )

    n = max(len(episodes), 1)
    aggregate_cpa = aggregate_cost / (aggregate_conversion + 1e-10) if aggregate_cost > 0 else 0.0
    result = {
        "config": config,
        "num_advertisers": len(episodes),
        "avg_score": aggregate_score / n,
        "total_realized_conversions": aggregate_conversion,
        "total_expected_value": aggregate_expected,
        "total_cost": aggregate_cost,
        "total_budget": aggregate_budget,
        "aggregate_cpa": aggregate_cpa,
        "budget_usage": aggregate_cost / aggregate_budget if aggregate_budget else 0.0,
        "underdelivery": max(aggregate_budget - aggregate_cost, 0.0) / aggregate_budget if aggregate_budget else 0.0,
        "cpa_exceed_rate": cpa_exceed_count / n,
        "budget_exceed_rate": budget_exceed_count / n,
        "mean_raw_action": float(np.mean(all_raw_actions)) if all_raw_actions else 0.0,
        "mean_executed_action": float(np.mean(all_executed_actions)) if all_executed_actions else 0.0,
        "mean_reference_action": float(np.mean(all_reference_actions)) if all_reference_actions else 0.0,
        "projection_rate": projection_count / max(action_count, 1),
        "mean_projection_distance": projection_distance / max(action_count, 1),
        "mean_realized_autonomy": float(np.mean(all_realized_autonomy)) if all_realized_autonomy else 1.0,
        "mean_control_intensity": float(np.mean(all_control_intensity)) if all_control_intensity else 0.0,
        "mean_reviews_per_advertiser": total_reviews / n,
        "win_rate": total_wins / max(total_pvs, 1),
        "mean_implied_initial_block_share": float(
            np.mean([implied_block_share(mandate.q_scale, mandate.h, 48)])
        ),
        "per_agent": per_agent,
    }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.conversion_seed)
    np.random.seed(args.conversion_seed)
    torch.manual_seed(args.conversion_seed)
    started = time.time()
    source_csv = Path(args.source_csv).resolve()
    data_pickle = Path(args.data_pickle).resolve()
    lock_path_raw = os.environ.get("CONFIRMATORY_LOCK_PATH")
    lock_digest_expected = os.environ.get("CONFIRMATORY_LOCK_SHA256")
    if "period-8" in source_csv.name and not args.post_confirmatory_period8:
        if not lock_path_raw or not lock_digest_expected:
            raise RuntimeError("CONFIRMATORY_LOCK_REQUIRED: set locked manifest path and digest")
        lock_path = Path(lock_path_raw).resolve()
        if sha256_file(lock_path) != lock_digest_expected:
            raise RuntimeError("CONFIRMATORY_LOCK_MISMATCH")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "LOCKED_BEFORE_PERIOD8_ACCESS":
            raise RuntimeError("CONFIRMATORY_LOCK_STATUS_INVALID")
        if lock.get("period8_accessed_before_lock") is not False:
            raise RuntimeError("CONFIRMATORY_PRELOCK_ACCESS_FLAG_INVALID")
        expected_evaluator = lock["hashes"]["confirmatory_evaluator_sha256"]
        if sha256_file(Path(__file__).resolve()) != expected_evaluator:
            raise RuntimeError("CONFIRMATORY_EVALUATOR_HASH_MISMATCH")
        expected_grid = lock["hashes"]["grid_sha256"]
        if not args.grid_json or sha256_file(Path(args.grid_json).resolve()) != expected_grid:
            raise RuntimeError("CONFIRMATORY_GRID_HASH_MISMATCH")
    configs = make_configs(args)
    episodes = prepare_episodes(data_pickle, args.max_advertisers)
    controller = FastController(args)
    results = []
    mechanism_handle = None
    if args.mechanism_output:
        mechanism_path = Path(args.mechanism_output)
        mechanism_path.parent.mkdir(parents=True, exist_ok=True)
        mechanism_temporary = mechanism_path.with_suffix(mechanism_path.suffix + ".tmp")
        mechanism_handle = mechanism_temporary.open("x", encoding="utf-8")
    try:
        for index, config in enumerate(configs, start=1):
            print(f"CONFIG_START {index}/{len(configs)} id={config['id']}", flush=True)
            config_started = time.time()
            result = evaluate_config(args, config, episodes, controller, mechanism_handle=mechanism_handle)
            result["seconds"] = time.time() - config_started
            results.append(result)
            print(
                f"CONFIG_DONE id={config['id']} score={result['avg_score']:.6f} "
                f"value={result['total_expected_value']:.3f} cpa={result['aggregate_cpa']:.6f} "
                f"usage={result['budget_usage']:.6f} seconds={result['seconds']:.2f}",
                flush=True,
            )
    finally:
        if mechanism_handle is not None:
            mechanism_handle.close()
    if args.mechanism_output:
        os.replace(mechanism_temporary, mechanism_path)
    payload = {
        "schema_version": 2,
        "mechanism_semantics": "one_step_local_intervention_on_governed_history",
        "created_at_unix": time.time(),
        "command": sys.argv,
        "hostname": os.uname().nodename,
        "source_csv": str(source_csv),
        "source_csv_bytes": source_csv.stat().st_size,
        "data_pickle": str(data_pickle),
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "normalizer": str(Path(args.normalizer).resolve()) if args.normalizer else None,
        "device": str(args.device),
        "conversion_seed": args.conversion_seed,
        "market_seed": args.market_seed,
        "budget_control": args.budget_control,
        "max_advertisers": args.max_advertisers,
        "max_timesteps": args.max_timesteps,
        "max_pvs": args.max_pvs,
        "pv_order_seed": args.pv_order_seed,
        "analysis_role": (
            "post_confirmatory_period8_mechanism_diagnostic"
            if args.post_confirmatory_period8
            else "development_or_locked_confirmatory"
        ),
        "common_random_numbers": True,
        "mechanism_output": str(Path(args.mechanism_output).resolve()) if args.mechanism_output else None,
        "total_seconds": time.time() - started,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"OUTPUT_WRITTEN {output}", flush=True)


if __name__ == "__main__":
    main()
