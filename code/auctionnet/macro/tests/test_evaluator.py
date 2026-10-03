from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from platformbid_v1.controller import FastController
from platformbid_v1.data import PeriodTick
from platformbid_v1.design import DesignCell
from platformbid_v1.evaluator import (
    initialize_cell_state,
    market_row,
    raw_policy_mask,
    step_market_cells,
)


def tick(time_step=0, n_pvs=5):
    p_values = np.linspace(0.01, 0.20, 48)[:, None] * np.ones((1, n_pvs))
    return PeriodTick(
        period=9,
        time_step=time_step,
        pv_indices=np.arange(n_pvs),
        p_values=p_values,
        p_value_sigmas=np.zeros_like(p_values),
        original_lwc=np.ones(n_pvs),
        budgets=np.full(48, 100.0),
        cpa_constraints=np.full(48, 8.0),
        categories=np.arange(48) % 6,
    )


def test_two_cells_share_crn_but_can_have_distinct_governance():
    current = tick()
    low = DesignCell(0.0, "baseline_low", 0.1, 0.1, None)
    high = DesignCell(1.0, "high", 1.2, 0.1, None)
    states = [initialize_cell_state(low, current, 11), initialize_cell_state(high, current, 11)]
    controller = FastController("constant", constant_action=20)
    step_market_cells(
        states,
        current,
        controller,
        market_seed=28,
        h=4,
        q_scale=1.0,
        reference_mode="pacing",
        lookback=3,
        max_action=100,
        horizon=48,
    )
    assert states[0].projection_count.sum() == 48
    # kappa=1.2 is high but not algebraically unbounded: a raw action below
    # ref*(1-kappa) is free, while an action above ref*(1+kappa) is clipped.
    # The decisive semantic check is weaker intervention than kappa=0.1.
    assert states[1].projection_distance.sum() < states[0].projection_distance.sum()
    assert states[0].provisional_slots <= 2 * current.pv_indices.size
    assert states[1].provisional_slots <= 2 * current.pv_indices.size
    # Every agent saw the same count of shared opportunities.
    assert np.all(states[0].pvs == 5)
    assert np.all(states[1].pvs == 5)
    row = market_row(states[1], 9, 28, 11)
    assert row["market_episode_id"] == "p9__m28__a11"
    assert row["num_adopters"] == 48
    assert row["platform_revenue"] == pytest.approx(states[1].cost.sum())


def test_repeating_same_cell_and_seed_is_exactly_deterministic():
    current = tick(n_pvs=17)
    cell = DesignCell(0.25, "private", 0.8, 0.1, None)
    controller = FastController("constant", constant_action=12)
    outcomes = []
    for _ in range(2):
        state = initialize_cell_state(cell, current, 99)
        step_market_cells(
            [state], current, controller, market_seed=31, h=4, q_scale=1,
            reference_mode="pacing", lookback=3, max_action=100, horizon=48,
        )
        outcomes.append((state.adopters.copy(), state.cost.copy(), state.conversions.copy(), state.wins.copy()))
    for left, right in zip(outcomes[0], outcomes[1]):
        assert np.array_equal(left, right)


def test_raw_policy_bypasses_zero_reference_projection():
    current = tick(n_pvs=7)
    bounded_cell = DesignCell(1.0, "bounded", 100.0, 0.3, None)
    raw_cell = DesignCell(1.0, "raw", 0.0, 0.3, None, "raw", "bounded")
    bounded = initialize_cell_state(bounded_cell, current, 12)
    raw = initialize_cell_state(raw_cell, current, 12)
    bounded.current_references[:] = 0.0
    raw.current_references[:] = 0.0
    assert not raw_policy_mask(bounded).any()
    assert raw_policy_mask(raw).all()

    controller = FastController("constant", constant_action=20)
    step_market_cells(
        [bounded, raw], current, controller, market_seed=33, h=99, q_scale=0,
        reference_mode="pacing", lookback=3, max_action=100, horizon=48,
    )
    assert bounded.executed_action_sum.sum() == 0.0
    assert raw.executed_action_sum.sum() == pytest.approx(48 * 20.0)
    assert raw.projection_count.sum() == 0
