"""Pure numerical primitives for delegated auto-bidding.

This module deliberately has no Torch/Pandas dependency so its semantics can be
unit-tested independently from the historical AuctionNet codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


def expand_cartesian_spec(spec: dict) -> list[dict]:
    """Expand a compact, auditable Cartesian configuration specification."""

    if "cartesian" not in spec:
        raise ValueError("configuration object must contain 'cartesian'")
    base = dict(spec.get("base", {}))
    axes = spec["cartesian"]
    if not isinstance(axes, dict) or not axes:
        raise ValueError("cartesian must be a nonempty object")
    keys = list(axes)
    values = [axes[key] for key in keys]
    if any(not isinstance(items, list) or not items for items in values):
        raise ValueError("every Cartesian axis must be a nonempty list")
    output = []
    for combination in product(*values):
        item = {**base, **dict(zip(keys, combination))}
        item.setdefault("id", "__".join(f"{key}={value}" for key, value in zip(keys, combination)))
        output.append(item)
    for item in spec.get("include", []):
        output.append({**base, **item})
    return output


@dataclass(frozen=True)
class Mandate:
    """Executable control mandate.

    q_scale multiplies uniform remaining-budget pacing.  The implied block
    spend share at review time is q_scale * h / remaining_steps (clipped to 1).
    kappa is the symmetric relative action allowance around a_ref.
    h is the number of auction-control periods before review.
    """

    q_scale: float
    kappa: float
    h: int

    def __post_init__(self) -> None:
        if self.q_scale <= 0:
            raise ValueError("q_scale must be positive")
        if self.kappa < 0:
            raise ValueError("kappa must be nonnegative")
        if self.h < 1:
            raise ValueError("h must be at least one")


def implied_block_share(q_scale: float, h: int, remaining_steps: int) -> float:
    """Return the target fraction of current remaining budget for this block."""

    if remaining_steps < 1:
        return 0.0
    return float(np.clip(q_scale * min(h, remaining_steps) / remaining_steps, 0.0, 1.0))


def enforce_action(
    raw_action: float,
    reference_action: float,
    kappa: float,
    operator: str = "relative_hard",
) -> tuple[float, bool, float]:
    """Apply a declared autonomy operator around the supervisory reference."""

    raw = max(float(raw_action), 0.0)
    ref = max(float(reference_action), 0.0)
    width = float(kappa)
    if operator == "relative_hard":
        lo = max(ref * (1.0 - width), 0.0)
        hi = ref * (1.0 + width)
        executed = float(np.clip(raw, lo, hi))
    elif operator == "log_hard":
        lo = ref * float(np.exp(-width))
        hi = ref * float(np.exp(width))
        executed = float(np.clip(raw, lo, hi))
    elif operator == "soft_blend":
        # For the soft operator kappa is an autonomy weight rather than a
        # relative interval width: kappa=0 reproduces the reference and
        # kappa=1 reproduces the raw controller.  Cross-operator comparisons
        # still use realized_autonomy(), not nominal kappa.
        autonomy_weight = float(np.clip(width, 0.0, 1.0))
        executed = float((1.0 - autonomy_weight) * ref + autonomy_weight * raw)
    else:
        raise ValueError(f"unsupported autonomy operator: {operator}")
    distance = abs(executed - raw)
    return executed, bool(distance > 1e-10), distance


def enforce_asymmetric_action(
    raw_action: float,
    reference_action: float,
    kappa_down: float,
    kappa_up: float,
) -> tuple[float, bool, float]:
    """Apply a directional relative-hard boundary around the reference.

    This primitive is diagnostic only. The paper's operating policy remains
    the symmetric relative-hard rule. kappa_down permits movement below the
    reference and kappa_up permits movement above it.
    """

    raw = max(float(raw_action), 0.0)
    ref = max(float(reference_action), 0.0)
    down = float(kappa_down)
    up = float(kappa_up)
    if down < 0 or up < 0:
        raise ValueError("directional widths must be nonnegative")
    lo = max(ref * (1.0 - down), 0.0)
    hi = ref * (1.0 + up)
    executed = float(np.clip(raw, lo, hi))
    distance = abs(executed - raw)
    return executed, bool(distance > 1e-10), distance


def realized_autonomy(raw_action: float, reference_action: float, executed_action: float) -> tuple[float, float]:
    """Return (realized autonomy, control intensity) on the raw-reference path.

    The scale is operator-invariant: 0 is the reference, 1 is the raw action.
    If raw and reference coincide there is no meaningful intervention and the
    observation is defined as fully autonomous.
    """

    gap = abs(float(raw_action) - float(reference_action))
    if gap <= 1e-10:
        return 1.0, 0.0
    intensity = float(np.clip(abs(float(executed_action) - float(raw_action)) / gap, 0.0, 1.0))
    return 1.0 - intensity, intensity


def derive_intervention_masks(
    raw_initial_wins: np.ndarray,
    executed_initial_wins: np.ndarray,
    raw_budget_wins: np.ndarray,
    executed_budget_wins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose a one-step action intervention without causal overclaiming.

    ``direct_*`` isolates bid-eligibility changes. ``allocation_*`` compares
    budget-feasible allocations using the same state, remaining budget, and
    priority.  These masks describe a local intervention, not an untreated
    closed-loop trajectory.
    """

    raw_initial = np.asarray(raw_initial_wins, dtype=bool)
    executed_initial = np.asarray(executed_initial_wins, dtype=bool)
    raw_budget = np.asarray(raw_budget_wins, dtype=bool)
    executed_budget = np.asarray(executed_budget_wins, dtype=bool)
    shapes = {array.shape for array in (raw_initial, executed_initial, raw_budget, executed_budget)}
    if len(shapes) != 1:
        raise ValueError("all intervention masks must have the same shape")
    return {
        "forced_eligible": executed_initial & ~raw_initial,
        "blocked_eligible": raw_initial & ~executed_initial,
        "direct_forced_actual": executed_budget & ~raw_initial,
        "direct_blocked_feasible": raw_budget & ~executed_initial,
        "allocation_exec_only": executed_budget & ~raw_budget,
        "allocation_raw_only": raw_budget & ~executed_budget,
    }


def estimate_traffic_aware_reference_action(
    target_spend_per_step: float,
    current_pv_count: int,
    historical_observed_values: Sequence[np.ndarray],
    historical_lwc: Sequence[np.ndarray],
    fallback_action: float,
    lookback: int = 3,
    max_action: float = 100.0,
) -> float:
    """Traffic-aware pacing reference using lagged opportunity volume."""

    historical_counts = [np.asarray(values).size for values in historical_observed_values[-lookback:]]
    expected_count = float(np.mean(historical_counts)) if historical_counts else float(current_pv_count)
    traffic_ratio = float(current_pv_count) / max(expected_count, 1.0)
    adjusted_target = max(float(target_spend_per_step), 0.0) * traffic_ratio
    return estimate_reference_action(
        target_spend_per_step=adjusted_target,
        current_pv_count=current_pv_count,
        historical_observed_values=historical_observed_values,
        historical_lwc=historical_lwc,
        fallback_action=fallback_action,
        lookback=lookback,
        max_action=max_action,
    )


def estimate_smoothed_controller_reference_action(
    historical_raw_actions: Sequence[float],
    fallback_action: float,
    lookback: int = 4,
    max_action: float = 100.0,
    target_spend_per_step: float | None = None,
    historical_step_costs: Sequence[float] = (),
) -> float:
    """Competent but slow action, scaled by a lagged spend-response estimate."""

    actions = np.asarray(list(historical_raw_actions)[-lookback:], dtype=np.float64)
    actions = actions[np.isfinite(actions) & (actions >= 0)]
    if actions.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    base_action = float(np.median(actions))
    costs = np.asarray(list(historical_step_costs)[-lookback:], dtype=np.float64)
    costs = costs[np.isfinite(costs) & (costs >= 0)]
    if target_spend_per_step is not None and costs.size:
        spend_ratio = max(float(target_spend_per_step), 0.0) / max(float(np.mean(costs)), 1e-8)
        base_action *= float(np.clip(spend_ratio, 0.5, 2.0))
    return float(np.clip(base_action, 0.0, max_action))


def estimate_response_aware_reference_action(
    target_spend_per_step: float,
    current_pv_count: int,
    historical_observed_values: Sequence[np.ndarray],
    historical_lwc: Sequence[np.ndarray],
    historical_value_estimates: Sequence[np.ndarray],
    cpa_limit: float,
    fallback_action: float,
    lookback: int = 6,
    max_action: float = 100.0,
) -> float:
    """Observable response-aware reference satisfying spend and CPA roots.

    Both eligibility and expected outcome use lagged value estimates visible to
    the controller.  Environment truth is deliberately absent from this API.
    """

    observed = list(historical_observed_values)[-lookback:]
    lwcs = list(historical_lwc)[-lookback:]
    value_estimates = list(historical_value_estimates)[-lookback:]
    if not observed or not lwcs or not value_estimates:
        return float(np.clip(fallback_action, 0.0, max_action))
    observed_array = np.concatenate([np.asarray(value, dtype=np.float64) for value in observed])
    lwc_array = np.concatenate([np.asarray(value, dtype=np.float64) for value in lwcs])
    estimate_array = np.concatenate([np.asarray(value, dtype=np.float64) for value in value_estimates])
    valid = (
        np.isfinite(observed_array)
        & np.isfinite(lwc_array)
        & np.isfinite(estimate_array)
        & (observed_array > 0)
        & (lwc_array >= 0)
        & (estimate_array >= 0)
    )
    observed_array, lwc_array, estimate_array = (
        observed_array[valid], lwc_array[valid], estimate_array[valid]
    )
    if observed_array.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    ratios = lwc_array / np.maximum(observed_array, 1e-12)
    order = np.argsort(ratios, kind="stable")
    ratios = ratios[order]
    costs = lwc_array[order]
    expected_values = estimate_array[order]
    traffic_scale = float(current_pv_count) / float(observed_array.size)
    cumulative_cost = np.cumsum(costs) * traffic_scale
    cumulative_value = np.cumsum(expected_values) * traffic_scale
    spend_hit = int(np.searchsorted(cumulative_cost, max(float(target_spend_per_step), 0.0), side="left"))
    spend_action = float(max_action) if spend_hit >= ratios.size else float(ratios[spend_hit])
    expected_cpa = cumulative_cost / np.maximum(cumulative_value, 1e-12)
    feasible_indices = np.flatnonzero(expected_cpa <= max(float(cpa_limit), 0.0))
    cpa_action = float(ratios[feasible_indices[-1]]) if feasible_indices.size else 0.0
    return float(np.clip(min(spend_action, cpa_action), 0.0, max_action))


def estimate_reference_action(
    target_spend_per_step: float,
    current_pv_count: int,
    historical_observed_values: Sequence[np.ndarray],
    historical_lwc: Sequence[np.ndarray],
    fallback_action: float,
    lookback: int = 3,
    max_action: float = 100.0,
) -> float:
    """Estimate a pacing reference from prior revealed auction responses.

    No current/future least-winning costs are used.  For each candidate alpha,
    historical cost per opportunity is rescaled by the current opportunity
    count.  The smallest alpha meeting the target is selected; this makes the
    q-to-reference mapping deterministic and monotone conditional on history.
    """

    target = max(float(target_spend_per_step), 0.0)
    if target <= 0 or current_pv_count <= 0:
        return 0.0
    values = list(historical_observed_values)[-lookback:]
    lwcs = list(historical_lwc)[-lookback:]
    if not values or not lwcs:
        return float(np.clip(fallback_action, 0.0, max_action))
    v = np.concatenate([np.asarray(x, dtype=np.float64) for x in values])
    c = np.concatenate([np.asarray(x, dtype=np.float64) for x in lwcs])
    valid = np.isfinite(v) & np.isfinite(c) & (v > 0) & (c >= 0)
    v, c = v[valid], c[valid]
    if v.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    ratios = c / np.maximum(v, 1e-12)
    order = np.argsort(ratios, kind="stable")
    ratios, c = ratios[order], c[order]
    predicted_cumulative = np.cumsum(c) * (float(current_pv_count) / float(v.size))
    hit = int(np.searchsorted(predicted_cumulative, target, side="left"))
    if hit >= ratios.size:
        return float(max_action)
    return float(np.clip(ratios[hit], 0.0, max_action))


def nips_score(conversions: float, cost: float, cpa_constraint: float) -> float:
    """AuctionNet/NIPS score: conversions with a squared CPA penalty."""

    conv = float(conversions)
    spend = float(cost)
    cpa = spend / (conv + 1e-10) if spend > 0 else 0.0
    penalty = 1.0 if cpa <= cpa_constraint else (float(cpa_constraint) / (cpa + 1e-10)) ** 2
    return float(conv * penalty)
