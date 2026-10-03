"""Stateful pacing references shared by AuctionNet micro and macro studies.

The policies in this module are deliberately independent of a bidder's raw
action and of current/future auction prices.  They use only the campaign
budget state and information available before the current auction block.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _spend_error(*, budget: float, remaining: float, time_step: int, horizon: int) -> float:
    """Target-minus-actual cumulative spend as a fraction of initial budget."""

    if budget <= 0 or horizon <= 0:
        return 0.0
    target_progress = _clip(float(time_step) / float(horizon), 0.0, 1.0)
    actual_progress = _clip((float(budget) - float(remaining)) / float(budget), 0.0, 1.0)
    return target_progress - actual_progress


@dataclass
class PIDPacingState:
    integral: float = 0.0
    previous_error: float = 0.0
    initialized: bool = False
    last_reference: float | None = None
    saturation_count: int = 0


def pid_pacing_reference(
    state: PIDPacingState,
    *,
    budget: float,
    remaining: float,
    time_step: int,
    horizon: int,
    fallback_action: float,
    kp: float,
    ki: float,
    kd: float,
    integral_limit: float = 1.0,
    min_action: float = 1e-4,
    max_action: float = 100.0,
) -> float:
    """Log-action PID pacing with conditional-integration anti-windup.

    Positive error means the campaign is behind its cumulative spend target,
    so the reference action rises.  The output is anchored at the campaign's
    fallback action rather than at the bidder action.
    """

    if min_action <= 0 or max_action <= min_action:
        raise ValueError("PID action bounds must satisfy 0 < min < max")
    if integral_limit < 0:
        raise ValueError("PID integral_limit must be nonnegative")
    error = _spend_error(
        budget=budget, remaining=remaining, time_step=time_step, horizon=horizon
    )
    derivative = error - state.previous_error if state.initialized else 0.0
    candidate_integral = _clip(
        state.integral + error, -float(integral_limit), float(integral_limit)
    )
    anchor = _clip(float(fallback_action), min_action, max_action)

    def output(integral: float) -> tuple[float, float]:
        signal = float(kp) * error + float(ki) * integral + float(kd) * derivative
        unconstrained = anchor * math.exp(_clip(signal, -50.0, 50.0))
        return unconstrained, _clip(unconstrained, min_action, max_action)

    unconstrained, reference = output(candidate_integral)
    high_windup = unconstrained > max_action and error > 0
    low_windup = unconstrained < min_action and error < 0
    if high_windup or low_windup:
        state.saturation_count += 1
        _, reference = output(state.integral)
    else:
        state.integral = candidate_integral
    state.previous_error = error
    state.initialized = True
    state.last_reference = reference
    return float(reference)


@dataclass
class DualPacingState:
    log_shadow_price: float = 0.0
    last_spent: float = 0.0
    last_time_step: int = 0
    initialized: bool = False
    projection_count: int = 0
    last_reference: float | None = None


def dual_pacing_reference(
    state: DualPacingState,
    *,
    budget: float,
    remaining: float,
    time_step: int,
    horizon: int,
    fallback_action: float,
    eta: float,
    log_shadow_limit: float = math.log(20.0),
    min_action: float = 1e-4,
    max_action: float = 100.0,
) -> float:
    """Projected mirror-descent update for a budget shadow price.

    Spending above the budget trajectory raises the shadow price and lowers
    the bid multiplier; underspending does the opposite.  Updates use the
    realized spend since the previous review and never inspect bidder actions.
    """

    if budget <= 0 or horizon <= 0:
        return 0.0
    if eta < 0 or log_shadow_limit <= 0:
        raise ValueError("dual eta and log_shadow_limit must be positive")
    if min_action <= 0 or max_action <= min_action:
        raise ValueError("dual action bounds must satisfy 0 < min < max")
    spent = _clip(float(budget) - float(remaining), 0.0, float(budget))
    if state.initialized:
        elapsed = max(int(time_step) - state.last_time_step, 0)
        target_increment = float(budget) * float(elapsed) / float(horizon)
        actual_increment = spent - state.last_spent
        gradient = (actual_increment - target_increment) / float(budget)
        unconstrained = state.log_shadow_price + float(eta) * gradient
        projected = _clip(unconstrained, -float(log_shadow_limit), float(log_shadow_limit))
        state.projection_count += int(not math.isclose(unconstrained, projected, abs_tol=1e-12))
        state.log_shadow_price = projected
    else:
        state.initialized = True
    state.last_spent = spent
    state.last_time_step = int(time_step)
    anchor = _clip(float(fallback_action), min_action, max_action)
    reference = _clip(
        anchor * math.exp(-state.log_shadow_price), min_action, max_action
    )
    state.last_reference = reference
    return float(reference)
