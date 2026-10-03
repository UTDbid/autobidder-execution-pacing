import numpy as np

from platformbid_v1.supervision import (
    AgentHistory,
    enforce_relative_action,
    response_aware_reference,
)


def test_state_matches_gatea_16_dimensional_semantics():
    history = AgentHistory()
    history.bids.append(np.array([1.0, 3.0]))
    history.lwc.append(np.array([0.5, 1.5]))
    history.observed_values.append(np.array([0.1, 0.2]))
    history.conversions.append(np.array([0.0, 1.0]))
    history.wins.append(np.array([1.0, 0.0]))
    state = history.state(4, np.array([0.2, 0.4, 0.6]), 50, 100)
    assert state.shape == (16,)
    assert state[0] == np.float32(44 / 48)
    assert state[1] == np.float32(0.5)
    assert state[2] == np.float32(2.0)
    assert state[12] == np.float32(0.4)
    assert state[13] == 3
    assert state[14] == 2
    assert state[15] == 2


def test_relative_autonomy_operator():
    assert enforce_relative_action(20, 10, 0) == (10.0, True, 10.0)
    assert enforce_relative_action(20, 10, 1) == (20.0, False, 0.0)


def test_response_aware_reference_obeys_spend_and_cpa_roots():
    values = [np.asarray([0.1, 0.2, 0.4])]
    prices = [np.asarray([1.0, 5.0, 30.0])]
    value = response_aware_reference(
        target_spend_per_step=6.0,
        current_pv_count=3,
        historical_values=values,
        historical_prices=prices,
        cpa_limit=20.0,
        fallback_action=10.0,
        max_action=100.0,
    )
    assert value == 25.0


def test_response_aware_reference_falls_back_without_history():
    assert response_aware_reference(1.0, 10, [], [], 20.0, 7.0) == 7.0
