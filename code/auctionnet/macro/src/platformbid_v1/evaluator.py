"""Synchronous 48-advertiser delegated-autonomy market evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

COMMON_ROOT = Path(__file__).resolve().parents[4] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from reference_policies import dual_pacing_reference, pid_pacing_reference

from .auction import enforce_agent_budgets, uniform_third_price_top2
from .controller import FastController
from .data import PeriodTick, iter_period_ticks
from .design import DesignCell, make_design_cells, select_adopters
from .supervision import (
    AgentHistory,
    enforce_relative_action,
    pacing_reference,
    response_aware_reference,
    smoothed_controller_reference,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(market_seed: int, period: int, time_step: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(deterministic_seed(market_seed, period, time_step, stream))


@dataclass
class CellState:
    cell: DesignCell
    adopters: np.ndarray
    budgets: np.ndarray
    cpas: np.ndarray
    categories: np.ndarray
    remaining: np.ndarray
    histories: list[AgentHistory]
    current_references: np.ndarray
    cost: np.ndarray
    conversions: np.ndarray
    base_expected: np.ndarray
    latent_expected: np.ndarray
    wins: np.ndarray
    pvs: np.ndarray
    raw_action_sum: np.ndarray
    executed_action_sum: np.ndarray
    reference_action_sum: np.ndarray
    action_count: np.ndarray
    projection_count: np.ndarray
    projection_distance: np.ndarray
    review_count: np.ndarray
    clearing_sum: float = 0.0
    clearing_positive_sum: float = 0.0
    clearing_positive_count: int = 0
    opportunity_count: int = 0
    provisional_slots: int = 0
    filled_slots: int = 0
    budget_dropped_slots: int = 0
    assignment_rule: str = "random_block_hash"


def initialize_cell_state(
    cell: DesignCell,
    tick: PeriodTick,
    assignment_seed: int,
    *,
    assignment_rule: str = "random_block_hash",
    ranking_scores: np.ndarray | None = None,
) -> CellState:
    adopters = select_adopters(
        cell.alpha,
        tick.budgets,
        tick.cpa_constraints,
        tick.categories,
        assignment_seed=assignment_seed,
        period=tick.period,
        rotation_agent=cell.rotation_agent,
        ranking_scores=ranking_scores,
    )
    n_agents = tick.budgets.size
    zeros = lambda dtype=np.float64: np.zeros(n_agents, dtype=dtype)
    return CellState(
        cell=cell,
        adopters=adopters,
        budgets=tick.budgets.copy(),
        cpas=tick.cpa_constraints.copy(),
        categories=tick.categories.copy(),
        remaining=tick.budgets.copy(),
        histories=[AgentHistory() for _ in range(n_agents)],
        current_references=tick.cpa_constraints.copy(),
        cost=zeros(),
        conversions=zeros(),
        base_expected=zeros(),
        latent_expected=zeros(),
        wins=zeros(np.int64),
        pvs=zeros(np.int64),
        raw_action_sum=zeros(),
        executed_action_sum=zeros(),
        reference_action_sum=zeros(),
        action_count=zeros(np.int64),
        projection_count=zeros(np.int64),
        projection_distance=zeros(),
        review_count=zeros(np.int64),
        assignment_rule=assignment_rule,
    )


def validate_static_contracts(state: CellState, tick: PeriodTick) -> None:
    if not (
        np.array_equal(state.budgets, tick.budgets)
        and np.array_equal(state.cpas, tick.cpa_constraints)
        and np.array_equal(state.categories, tick.categories)
    ):
        raise ValueError("advertiser contracts changed within a period")


def kappa_vector(state: CellState) -> np.ndarray:
    return np.where(state.adopters, state.cell.adopter_kappa, state.cell.low_kappa).astype(np.float64)


def raw_policy_mask(state: CellState) -> np.ndarray:
    """Return agents whose native bidder action bypasses the projection envelope.

    Raw is an explicit execution mode rather than a large finite kappa.  This
    distinction matters whenever the pacing reference is zero, because every
    finite relative envelope then also has a zero upper bound.
    """

    adopter_raw = state.cell.adopter_policy_mode == "raw"
    low_raw = state.cell.low_policy_mode == "raw"
    return np.where(state.adopters, adopter_raw, low_raw).astype(bool)


def nips_score(conversions: np.ndarray, cost: np.ndarray, cpa_constraint: np.ndarray) -> np.ndarray:
    conversions = np.asarray(conversions, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    constraints = np.asarray(cpa_constraint, dtype=np.float64)
    cpa = np.divide(cost, conversions, out=np.zeros_like(cost), where=conversions > 0)
    penalty = np.where(cpa <= constraints, 1.0, np.square(constraints / np.maximum(cpa, 1e-10)))
    return conversions * penalty


def update_references(
    state: CellState,
    tick: PeriodTick,
    *,
    h: int,
    q_scale: float,
    reference_mode: str,
    lookback: int,
    max_action: float,
    horizon: int,
    reference_params: dict[str, Any],
) -> None:
    if tick.time_step % h != 0:
        return
    remaining_steps = max(horizon - tick.time_step, 1)
    for agent, history in enumerate(state.histories):
        target = q_scale * state.remaining[agent] / remaining_steps
        fallback = state.cpas[agent] * q_scale
        if reference_mode == "pacing":
            reference = pacing_reference(
                target,
                tick.p_values.shape[1],
                list(history.reference_values)[-lookback:],
                list(history.reference_prices)[-lookback:],
                fallback,
                max_action=max_action,
            )
        elif reference_mode == "pid":
            reference = pid_pacing_reference(
                history.pid_reference_state,
                budget=state.budgets[agent],
                remaining=state.remaining[agent],
                time_step=tick.time_step,
                horizon=horizon,
                fallback_action=fallback,
                kp=float(reference_params.get("pid_kp", 2.0)),
                ki=float(reference_params.get("pid_ki", 0.2)),
                kd=float(reference_params.get("pid_kd", 0.0)),
                integral_limit=float(reference_params.get("pid_integral_limit", 1.0)),
                max_action=max_action,
            )
        elif reference_mode == "dual":
            reference = dual_pacing_reference(
                history.dual_reference_state,
                budget=state.budgets[agent],
                remaining=state.remaining[agent],
                time_step=tick.time_step,
                horizon=horizon,
                fallback_action=fallback,
                eta=float(reference_params.get("dual_eta", 5.0)),
                log_shadow_limit=float(
                    reference_params.get("dual_log_shadow_limit", np.log(20.0))
                ),
                max_action=max_action,
            )
        elif reference_mode == "smoothed_controller":
            reference = smoothed_controller_reference(
                history.raw_actions,
                history.step_costs,
                target,
                fallback,
                lookback=max(lookback, h),
                max_action=max_action,
            )
        elif reference_mode == "response_aware":
            reference = response_aware_reference(
                target,
                tick.p_values.shape[1],
                list(history.reference_values),
                list(history.reference_prices),
                state.cpas[agent],
                fallback,
                lookback=max(lookback, 6),
                max_action=max_action,
            )
        else:
            raise ValueError(f"unsupported reference_mode: {reference_mode}")
        state.current_references[agent] = reference
        state.review_count[agent] += 1


def step_market_cells(
    cells: list[CellState],
    tick: PeriodTick,
    controller: FastController,
    *,
    market_seed: int,
    h: int,
    q_scale: float,
    reference_mode: str,
    lookback: int,
    max_action: float,
    horizon: int,
    reference_params: dict[str, Any] | None = None,
) -> None:
    """Advance every treatment cell on one shared traffic tick."""

    n_agents, n_pvs = tick.p_values.shape
    tie_break = rng_for(market_seed, tick.period, tick.time_step, "tie").random((n_agents, n_pvs))
    budget_priority = rng_for(market_seed, tick.period, tick.time_step, "budget").random((n_agents, n_pvs))
    latent_z = rng_for(market_seed, tick.period, tick.time_step, "latent_value").standard_normal((n_agents, n_pvs))
    conversion_uniform = rng_for(market_seed, tick.period, tick.time_step, "conversion").random((n_agents, n_pvs))
    latent_probability = np.clip(tick.p_values + tick.p_value_sigmas * latent_z, 0.0, 1.0)
    potential_conversions = conversion_uniform < latent_probability

    state_rows: list[np.ndarray] = []
    for state in cells:
        validate_static_contracts(state, tick)
        for agent, history in enumerate(state.histories):
            state_rows.append(
                history.state(
                    tick.time_step,
                    tick.p_values[agent],
                    state.remaining[agent],
                    state.budgets[agent],
                    horizon=horizon,
                )
            )
    raw_matrix = controller.actions(np.stack(state_rows)).reshape(len(cells), n_agents)

    for cell_index, state in enumerate(cells):
        raw_actions = raw_matrix[cell_index]
        for agent, history in enumerate(state.histories):
            history.raw_actions.append(float(raw_actions[agent]))
        update_references(
            state,
            tick,
            h=h,
            q_scale=q_scale,
            reference_mode=reference_mode,
            lookback=lookback,
            max_action=max_action,
            horizon=horizon,
            reference_params=reference_params or {},
        )
        kappas = kappa_vector(state)
        raw_mask = raw_policy_mask(state)
        executed_actions = np.empty(n_agents, dtype=np.float64)
        projected = np.empty(n_agents, dtype=bool)
        distances = np.empty(n_agents, dtype=np.float64)
        for agent in range(n_agents):
            if state.remaining[agent] < 0.01:
                executed_actions[agent], projected[agent], distances[agent] = 0.0, False, 0.0
            elif raw_mask[agent]:
                executed_actions[agent], projected[agent], distances[agent] = (
                    max(float(raw_actions[agent]), 0.0),
                    False,
                    0.0,
                )
            else:
                executed_actions[agent], projected[agent], distances[agent] = enforce_relative_action(
                    raw_actions[agent], state.current_references[agent], kappas[agent]
                )
        bids = executed_actions[:, None] * tick.p_values
        auction = uniform_third_price_top2(bids, tie_break)
        final_winners, final_payments = enforce_agent_budgets(
            auction.winners,
            auction.payments,
            state.remaining,
            budget_priority,
        )
        realized_conversions = potential_conversions & final_winners
        step_costs = np.sum(final_payments, axis=1)
        step_conversions = np.sum(realized_conversions, axis=1)
        state.remaining = np.maximum(state.remaining - step_costs, 0.0)
        state.cost += step_costs
        state.conversions += step_conversions
        state.base_expected += np.sum(tick.p_values * final_winners, axis=1)
        state.latent_expected += np.sum(latent_probability * final_winners, axis=1)
        state.wins += np.sum(final_winners, axis=1)
        state.pvs += n_pvs
        active = state.remaining + step_costs >= 0.01
        state.raw_action_sum += raw_actions * active
        state.executed_action_sum += executed_actions * active
        state.reference_action_sum += state.current_references * active
        state.action_count += active.astype(np.int64)
        state.projection_count += projected.astype(np.int64)
        state.projection_distance += distances
        state.clearing_sum += float(np.sum(auction.clearing_prices))
        positive_prices = auction.clearing_prices[auction.clearing_prices > 0]
        state.clearing_positive_sum += float(np.sum(positive_prices))
        state.clearing_positive_count += int(positive_prices.size)
        state.opportunity_count += n_pvs
        state.provisional_slots += int(np.sum(auction.winners))
        state.filled_slots += int(np.sum(final_winners))
        state.budget_dropped_slots += int(np.sum(auction.winners) - np.sum(final_winners))

        for agent, history in enumerate(state.histories):
            history.bids.append(bids[agent])
            history.lwc.append(auction.clearing_prices)
            history.observed_values.append(tick.p_values[agent])
            history.conversions.append(realized_conversions[agent].astype(np.float64))
            history.wins.append(final_winners[agent].astype(np.float64))
            history.reference_values.append(tick.p_values[agent].copy())
            history.reference_prices.append(auction.clearing_prices.copy())
            history.step_costs.append(float(step_costs[agent]))
            history.previous_reward = float(step_conversions[agent])


def agent_rows(state: CellState, period: int, market_seed: int, assignment_seed: int) -> list[dict[str, Any]]:
    score = nips_score(state.conversions, state.cost, state.cpas)
    cpa = np.divide(state.cost, state.conversions, out=np.zeros_like(state.cost), where=state.conversions > 0)
    rows = []
    kappas = kappa_vector(state)
    raw_mask = raw_policy_mask(state)
    for agent in range(state.budgets.size):
        count = max(int(state.action_count[agent]), 1)
        rows.append(
            {
                "market_episode_id": f"p{period}__m{market_seed}__a{assignment_seed}",
                "period": period,
                "market_seed": market_seed,
                "assignment_seed": assignment_seed,
                "cell_id": state.cell.cell_id,
                "alpha": state.cell.alpha,
                "adopter_policy": state.cell.adopter_policy,
                "rotation_agent": state.cell.rotation_agent,
                "agent_id": agent,
                "adopter": bool(state.adopters[agent]),
                "assignment_rule": state.assignment_rule,
                "execution_mode": "raw" if raw_mask[agent] else "bounded",
                "kappa": None if raw_mask[agent] else float(kappas[agent]),
                "budget": float(state.budgets[agent]),
                "cpa_constraint": float(state.cpas[agent]),
                "category": int(state.categories[agent]),
                "cost": float(state.cost[agent]),
                "realized_conversions": float(state.conversions[agent]),
                "base_expected_value": float(state.base_expected[agent]),
                "latent_expected_value": float(state.latent_expected[agent]),
                "realized_cpa": float(cpa[agent]),
                "score": float(score[agent]),
                "budget_usage": float(state.cost[agent] / state.budgets[agent]) if state.budgets[agent] else 0.0,
                "underdelivery": float(max(state.budgets[agent] - state.cost[agent], 0.0) / state.budgets[agent]) if state.budgets[agent] else 0.0,
                "cpa_exceeded": bool(cpa[agent] > state.cpas[agent]),
                "wins": int(state.wins[agent]),
                "win_rate": float(state.wins[agent] / max(state.pvs[agent], 1)),
                "mean_raw_action": float(state.raw_action_sum[agent] / count),
                "mean_executed_action": float(state.executed_action_sum[agent] / count),
                "mean_reference_action": float(state.reference_action_sum[agent] / count),
                "projection_rate": float(state.projection_count[agent] / count),
                "mean_projection_distance": float(state.projection_distance[agent] / count),
                "review_count": int(state.review_count[agent]),
            }
        )
    return rows


def _group_sum(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(np.asarray(values)[np.asarray(mask, dtype=bool)]))


def market_row(state: CellState, period: int, market_seed: int, assignment_seed: int) -> dict[str, Any]:
    score = nips_score(state.conversions, state.cost, state.cpas)
    cpa = float(np.sum(state.cost) / max(float(np.sum(state.conversions)), 1e-10))
    nonadopters = ~state.adopters
    return {
        "market_episode_id": f"p{period}__m{market_seed}__a{assignment_seed}",
        "period": period,
        "market_seed": market_seed,
        "assignment_seed": assignment_seed,
        "cell_id": state.cell.cell_id,
        "alpha": state.cell.alpha,
        "adopter_policy": state.cell.adopter_policy,
        "adopter_kappa": state.cell.adopter_kappa,
        "adopter_policy_mode": state.cell.adopter_policy_mode,
        "low_kappa": state.cell.low_kappa,
        "low_policy_mode": state.cell.low_policy_mode,
        "rotation_agent": state.cell.rotation_agent,
        "num_adopters": int(np.sum(state.adopters)),
        "assignment_rule": state.assignment_rule,
        "total_score": float(np.sum(score)),
        "total_realized_conversions": float(np.sum(state.conversions)),
        "total_base_expected_value": float(np.sum(state.base_expected)),
        "total_latent_expected_value": float(np.sum(state.latent_expected)),
        "platform_revenue": float(np.sum(state.cost)),
        "aggregate_cpa": cpa,
        "total_budget": float(np.sum(state.budgets)),
        "budget_usage": float(np.sum(state.cost) / np.sum(state.budgets)),
        "underdelivery": float(np.sum(np.maximum(state.budgets - state.cost, 0.0)) / np.sum(state.budgets)),
        "cpa_exceed_rate": float(np.mean(np.divide(state.cost, state.conversions, out=np.zeros_like(state.cost), where=state.conversions > 0) > state.cpas)),
        "mean_clearing_price_all": state.clearing_sum / max(state.opportunity_count, 1),
        "mean_clearing_price_positive": state.clearing_positive_sum / max(state.clearing_positive_count, 1),
        "positive_clearing_share": state.clearing_positive_count / max(state.opportunity_count, 1),
        "provisional_slot_fill": state.provisional_slots / max(2 * state.opportunity_count, 1),
        "realized_slot_fill": state.filled_slots / max(2 * state.opportunity_count, 1),
        "budget_dropped_slots": state.budget_dropped_slots,
        "adopter_expected_value": _group_sum(state.base_expected, state.adopters),
        "adopter_cost": _group_sum(state.cost, state.adopters),
        "adopter_conversions": _group_sum(state.conversions, state.adopters),
        "nonadopter_expected_value": _group_sum(state.base_expected, nonadopters),
        "nonadopter_cost": _group_sum(state.cost, nonadopters),
        "nonadopter_conversions": _group_sum(state.conversions, nonadopters),
    }


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return len(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"alphas", "adopter_policies", "low_kappa", "q_scale", "h", "reference_mode", "lookback", "max_action"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if str(config.get("status", "")).startswith("TEMPLATE_"):
        raise RuntimeError("CONFIG_NOT_LOCKED: formal template cannot be executed")
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--market-seed", type=int, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--model", choices=["cql", "constant"], default="cql")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--constant-action", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--max-pvs", type=int)
    parser.add_argument("--cell-regex")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument(
        "--assignment-score-json",
        type=Path,
        help="Frozen pre-treatment advertiser score file for a targeted assignment diagnostic",
    )
    parser.add_argument(
        "--assignment-score-field",
        choices=["private_gain", "gain_to_footprint"],
        help="Score column used to rank advertisers",
    )
    parser.add_argument("--chunk-rows", type=int, default=2_000_000)
    return parser.parse_args(argv)


def load_assignment_scores(path: Path, field: str, n_agents: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("scores")
    if not isinstance(rows, list):
        raise ValueError("assignment score JSON must contain a scores list")
    values = np.full(n_agents, np.nan, dtype=np.float64)
    for row in rows:
        agent = int(row["agent_id"])
        if not 0 <= agent < n_agents or np.isfinite(values[agent]):
            raise ValueError("assignment score JSON has duplicate or invalid agent_id")
        values[agent] = float(row[field])
    if not np.all(np.isfinite(values)):
        raise ValueError("assignment score JSON does not cover every advertiser")
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = load_config(args.config)
    cells = make_design_cells(
        config["alphas"], config["adopter_policies"], low_kappa=config["low_kappa"], n_agents=48
    )
    if args.cell_regex:
        expression = re.compile(args.cell_regex)
        cells = [cell for cell in cells if expression.search(cell.cell_id)]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    if not cells:
        raise ValueError("cell filter produced an empty design")
    ranking_scores = None
    assignment_rule = "random_block_hash"
    if args.assignment_score_json is not None:
        if args.assignment_score_field is None:
            raise ValueError("assignment-score-field is required with assignment-score-json")
        ranking_scores = load_assignment_scores(args.assignment_score_json, args.assignment_score_field, 48)
        assignment_rule = f"ranked_{args.assignment_score_field}"
    elif args.assignment_score_field is not None:
        raise ValueError("assignment-score-json is required with assignment-score-field")
    controller = FastController(
        args.model,
        checkpoint=args.checkpoint,
        normalizer=args.normalizer,
        device=args.device,
        constant_action=args.constant_action,
    )
    controller.reset(len(cells) * 48)
    states: list[CellState] | None = None
    steps = 0
    for tick in iter_period_ticks(
        args.archive,
        args.member,
        expected_period=args.period,
        phase=args.phase,
        n_agents=48,
        chunk_rows=args.chunk_rows,
        max_timesteps=args.max_timesteps,
        max_pvs=args.max_pvs,
    ):
        if states is None:
            states = [
                initialize_cell_state(
                    cell,
                    tick,
                    args.assignment_seed,
                    assignment_rule=assignment_rule,
                    ranking_scores=ranking_scores,
                )
                for cell in cells
            ]
        step_market_cells(
            states,
            tick,
            controller,
            market_seed=args.market_seed,
            h=int(config["h"]),
            q_scale=float(config["q_scale"]),
            reference_mode=str(config["reference_mode"]),
            lookback=int(config["lookback"]),
            max_action=float(config["max_action"]),
            horizon=int(config.get("horizon", 48)),
            reference_params=dict(config.get("reference_params", {})),
        )
        steps += 1
        print(f"TICK_DONE period={tick.period} time_step={tick.time_step} cells={len(cells)} pvs={tick.pv_indices.size}", flush=True)
    if states is None or steps == 0:
        raise RuntimeError("no period ticks were evaluated")
    market_rows = [market_row(state, args.period, args.market_seed, args.assignment_seed) for state in states]
    agents = [row for state in states for row in agent_rows(state, args.period, args.market_seed, args.assignment_seed)]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    design_rows = [
        {
            "cell_id": cell.cell_id,
            "alpha": cell.alpha,
            "adopter_policy": cell.adopter_policy,
            "adopter_kappa": cell.adopter_kappa,
            "adopter_policy_mode": cell.adopter_policy_mode,
            "low_kappa": cell.low_kappa,
            "low_policy_mode": cell.low_policy_mode,
            "rotation_agent": cell.rotation_agent,
            "assignment_rule": assignment_rule,
        }
        for cell in cells
    ]
    counts = {
        "design_cells.csv": write_csv_atomic(output / "design_cells.csv", design_rows),
        "market_runs.csv": write_csv_atomic(output / "market_runs.csv", market_rows),
        "agent_outcomes.csv": write_csv_atomic(output / "agent_outcomes.csv", agents),
    }
    manifest = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
        "phase": args.phase,
        "period": args.period,
        "market_seed": args.market_seed,
        "assignment_seed": args.assignment_seed,
        "assignment_rule": assignment_rule,
        "assignment_score_json": (
            str(args.assignment_score_json.resolve()) if args.assignment_score_json else None
        ),
        "assignment_score_sha256": (
            sha256_file(args.assignment_score_json) if args.assignment_score_json else None
        ),
        "assignment_score_field": args.assignment_score_field,
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256_file(args.archive),
        "member": args.member,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "code_files_sha256": {
            path.name: sha256_file(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "normalizer": str(args.normalizer.resolve()) if args.normalizer else None,
        "normalizer_sha256": sha256_file(args.normalizer) if args.normalizer else None,
        "device": args.device,
        "steps": steps,
        "max_timesteps": args.max_timesteps,
        "max_pvs": args.max_pvs,
        "cell_regex": args.cell_regex,
        "max_cells": args.max_cells,
        "common_random_numbers": {
            "scope": "period x market_seed x time_step x stream",
            "streams": ["tie", "budget", "latent_value", "conversion"],
            "branch_dependent_redraws": False,
        },
        "auction_rule": "two positive top bids; deterministic CRN tie break; uniform third-highest-bid payment",
        "budget_rule": "CRN-priority affordable subset; no rerank; no redraw",
        "original_lwc_role": "input-alignment diagnostic only; not used for reconstructed clearing",
        "counts": counts,
        "seconds": time.time() - started,
    }
    temporary = output / "run_manifest.json.tmp"
    final = output / "run_manifest.json"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, final)
    print(f"OUTPUT_WRITTEN {output}", flush=True)
    return manifest


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
