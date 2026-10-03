from __future__ import annotations

import math

from reference_policies import (
    DualPacingState,
    PIDPacingState,
    dual_pacing_reference,
    pid_pacing_reference,
)


def test_pid_raises_action_when_behind_and_lowers_when_ahead() -> None:
    behind = pid_pacing_reference(
        PIDPacingState(), budget=100, remaining=95, time_step=24, horizon=48,
        fallback_action=10, kp=2, ki=0, kd=0,
    )
    ahead = pid_pacing_reference(
        PIDPacingState(), budget=100, remaining=30, time_step=24, horizon=48,
        fallback_action=10, kp=2, ki=0, kd=0,
    )
    assert behind > 10 > ahead


def test_pid_anti_windup_keeps_integral_bounded() -> None:
    state = PIDPacingState()
    for _ in range(20):
        value = pid_pacing_reference(
            state, budget=100, remaining=100, time_step=47, horizon=48,
            fallback_action=10, kp=100, ki=100, kd=0,
            integral_limit=0.5, max_action=20,
        )
        assert value <= 20
    assert abs(state.integral) <= 0.5
    assert state.saturation_count > 0


def test_dual_shadow_price_moves_against_spend_error() -> None:
    state = DualPacingState()
    initial = dual_pacing_reference(
        state, budget=100, remaining=100, time_step=0, horizon=10,
        fallback_action=10, eta=5,
    )
    overspent = dual_pacing_reference(
        state, budget=100, remaining=70, time_step=1, horizon=10,
        fallback_action=10, eta=5,
    )
    underspent = dual_pacing_reference(
        state, budget=100, remaining=70, time_step=5, horizon=10,
        fallback_action=10, eta=5,
    )
    assert math.isclose(initial, 10)
    assert overspent < initial
    assert underspent > overspent


def test_state_inputs_contain_no_bidder_action_or_current_price() -> None:
    pid_fields = set(PIDPacingState.__dataclass_fields__)
    dual_fields = set(DualPacingState.__dataclass_fields__)
    forbidden = {"raw_action", "bidder_action", "current_price", "future_price"}
    assert not pid_fields.intersection(forbidden)
    assert not dual_fields.intersection(forbidden)
