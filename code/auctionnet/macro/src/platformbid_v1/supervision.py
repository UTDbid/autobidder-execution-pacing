"""Low-dimensional history and delegated-control primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

COMMON_ROOT = Path(__file__).resolve().parents[4] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from reference_policies import DualPacingState, PIDPacingState


@dataclass
class MomentHistory:
    total_sum: float = 0.0
    total_count: int = 0
    recent: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=3))

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        item = (float(np.sum(array)), int(array.size))
        self.total_sum += item[0]
        self.total_count += item[1]
        self.recent.append(item)

    @property
    def mean(self) -> float:
        return self.total_sum / self.total_count if self.total_count else 0.0

    @property
    def recent_mean(self) -> float:
        count = sum(item[1] for item in self.recent)
        return sum(item[0] for item in self.recent) / count if count else 0.0

    @property
    def recent_count(self) -> int:
        return sum(item[1] for item in self.recent)


@dataclass
class AgentHistory:
    bids: MomentHistory = field(default_factory=MomentHistory)
    lwc: MomentHistory = field(default_factory=MomentHistory)
    observed_values: MomentHistory = field(default_factory=MomentHistory)
    conversions: MomentHistory = field(default_factory=MomentHistory)
    wins: MomentHistory = field(default_factory=MomentHistory)
    raw_actions: list[float] = field(default_factory=list)
    step_costs: list[float] = field(default_factory=list)
    reference_values: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    reference_prices: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    previous_reward: float | None = None
    pid_reference_state: PIDPacingState = field(default_factory=PIDPacingState)
    dual_reference_state: DualPacingState = field(default_factory=DualPacingState)

    def state(
        self,
        time_step: int,
        current_values: np.ndarray,
        remaining_budget: float,
        original_budget: float,
        *,
        horizon: int = 48,
    ) -> np.ndarray:
        return np.asarray(
            [
                (horizon - time_step) / float(horizon),
                remaining_budget / original_budget if original_budget > 0 else 0.0,
                self.bids.mean,
                self.bids.recent_mean,
                self.lwc.mean,
                self.observed_values.mean,
                self.conversions.mean,
                self.wins.mean,
                self.lwc.recent_mean,
                self.observed_values.recent_mean,
                self.conversions.recent_mean,
                self.wins.recent_mean,
                float(np.mean(current_values)) if current_values.size else 0.0,
                int(current_values.size),
                self.bids.recent_count,
                self.bids.total_count,
            ],
            dtype=np.float32,
        )


def enforce_relative_action(raw_action: float, reference_action: float, kappa: float) -> tuple[float, bool, float]:
    raw = max(float(raw_action), 0.0)
    reference = max(float(reference_action), 0.0)
    width = max(float(kappa), 0.0)
    executed = float(np.clip(raw, max(reference * (1.0 - width), 0.0), reference * (1.0 + width)))
    distance = abs(executed - raw)
    return executed, bool(distance > 1e-10), distance


def pacing_reference(
    target_spend_per_step: float,
    current_pv_count: int,
    historical_values: Sequence[np.ndarray],
    historical_prices: Sequence[np.ndarray],
    fallback_action: float,
    *,
    max_action: float = 100.0,
) -> float:
    """Lag-only monotone pacing response; no current/future prices are used."""

    if target_spend_per_step <= 0 or current_pv_count <= 0:
        return 0.0
    if not historical_values or not historical_prices:
        return float(np.clip(fallback_action, 0.0, max_action))
    values = np.concatenate([np.asarray(item, dtype=np.float64) for item in historical_values])
    prices = np.concatenate([np.asarray(item, dtype=np.float64) for item in historical_prices])
    valid = np.isfinite(values) & np.isfinite(prices) & (values > 0) & (prices >= 0)
    values, prices = values[valid], prices[valid]
    if values.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    ratios = prices / np.maximum(values, 1e-12)
    order = np.argsort(ratios, kind="stable")
    ratios, prices = ratios[order], prices[order]
    cumulative = np.cumsum(prices) * (float(current_pv_count) / values.size)
    index = int(np.searchsorted(cumulative, max(float(target_spend_per_step), 0.0), side="left"))
    return float(max_action if index >= ratios.size else np.clip(ratios[index], 0.0, max_action))


def smoothed_controller_reference(
    raw_actions: Sequence[float],
    step_costs: Sequence[float],
    target_spend_per_step: float,
    fallback_action: float,
    *,
    lookback: int,
    max_action: float = 100.0,
) -> float:
    actions = np.asarray(list(raw_actions)[-lookback:], dtype=np.float64)
    actions = actions[np.isfinite(actions) & (actions >= 0)]
    if actions.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    reference = float(np.median(actions))
    costs = np.asarray(list(step_costs)[-lookback:], dtype=np.float64)
    costs = costs[np.isfinite(costs) & (costs >= 0)]
    if costs.size:
        ratio = max(float(target_spend_per_step), 0.0) / max(float(np.mean(costs)), 1e-8)
        reference *= float(np.clip(ratio, 0.5, 2.0))
    return float(np.clip(reference, 0.0, max_action))


def response_aware_reference(
    target_spend_per_step: float,
    current_pv_count: int,
    historical_values: Sequence[np.ndarray],
    historical_prices: Sequence[np.ndarray],
    cpa_limit: float,
    fallback_action: float,
    *,
    lookback: int = 6,
    max_action: float = 100.0,
) -> float:
    """Lag-only pacing response constrained by an expected-CPA root.

    AuctionNet p-values are observable expected-conversion estimates, so the
    same lagged arrays supply both auction attractiveness and expected outcome.
    Latent probabilities, future conversions, and current clearing prices are
    deliberately absent from this API.
    """

    values_history = list(historical_values)[-lookback:]
    prices_history = list(historical_prices)[-lookback:]
    if not values_history or not prices_history:
        return float(np.clip(fallback_action, 0.0, max_action))
    values = np.concatenate([np.asarray(item, dtype=np.float64) for item in values_history])
    prices = np.concatenate([np.asarray(item, dtype=np.float64) for item in prices_history])
    valid = np.isfinite(values) & np.isfinite(prices) & (values > 0) & (prices >= 0)
    values, prices = values[valid], prices[valid]
    if values.size == 0:
        return float(np.clip(fallback_action, 0.0, max_action))
    ratios = prices / np.maximum(values, 1e-12)
    order = np.argsort(ratios, kind="stable")
    ratios, prices, expected_values = ratios[order], prices[order], values[order]
    traffic_scale = float(current_pv_count) / float(values.size)
    cumulative_cost = np.cumsum(prices) * traffic_scale
    cumulative_value = np.cumsum(expected_values) * traffic_scale
    spend_hit = int(
        np.searchsorted(
            cumulative_cost, max(float(target_spend_per_step), 0.0), side="left"
        )
    )
    spend_action = float(max_action) if spend_hit >= ratios.size else float(ratios[spend_hit])
    expected_cpa = cumulative_cost / np.maximum(cumulative_value, 1e-12)
    feasible = np.flatnonzero(expected_cpa <= max(float(cpa_limit), 0.0))
    cpa_action = float(ratios[feasible[-1]]) if feasible.size else 0.0
    return float(np.clip(min(spend_action, cpa_action), 0.0, max_action))
