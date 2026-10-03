from __future__ import annotations

import numpy as np

from mandate_core_v2 import (
    Mandate,
    derive_intervention_masks,
    enforce_action,
    estimate_reference_action,
    estimate_response_aware_reference_action,
    estimate_smoothed_controller_reference_action,
    estimate_traffic_aware_reference_action,
    expand_cartesian_spec,
    implied_block_share,
    nips_score,
    realized_autonomy,
)


def test_mandate_validation() -> None:
    assert Mandate(1.0, 0.3, 4).h == 4
    for args in [(0, 0.3, 4), (1, -0.1, 4), (1, 0.3, 0)]:
        try:
            Mandate(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid mandate accepted: {args}")


def test_enforcement_interval_and_projection() -> None:
    executed, projected, distance = enforce_action(20, 10, 0.3)
    assert np.isclose(executed, 13)
    assert projected and np.isclose(distance, 7)
    executed, projected, _ = enforce_action(11, 10, 0.3)
    assert np.isclose(executed, 11) and not projected
    executed, _, _ = enforce_action(2, 10, 1.5)
    assert np.isclose(executed, 2)


def test_alternative_autonomy_operators() -> None:
    executed, projected, _ = enforce_action(20, 10, np.log(1.5), operator="log_hard")
    assert np.isclose(executed, 15) and projected
    executed, projected, _ = enforce_action(20, 10, 0.5, operator="soft_blend")
    assert np.isclose(executed, 15) and projected
    executed, projected, _ = enforce_action(20, 10, 1.0, operator="soft_blend")
    assert np.isclose(executed, 20) and not projected
    autonomy, intensity = realized_autonomy(20, 10, 15)
    assert np.isclose(autonomy, 0.5) and np.isclose(intensity, 0.5)
    assert realized_autonomy(10, 10, 10) == (1.0, 0.0)


def test_q_mapping_is_monotone_given_history() -> None:
    values = [np.array([0.05, 0.10, 0.20, 0.40])]
    lwc = [np.array([0.20, 0.50, 1.50, 4.00])]
    refs = [estimate_reference_action(t, 4, values, lwc, 8.0) for t in (0.2, 0.7, 2.0, 5.0)]
    assert refs == sorted(refs)


def test_no_lookahead_fallback_and_block_share() -> None:
    assert np.isclose(estimate_reference_action(10, 100, [], [], 7.5), 7.5)
    assert np.isclose(implied_block_share(1.0, 4, 20), 0.2)
    assert np.isclose(implied_block_share(1.5, 20, 10), 1.0)


def test_reference_variants_are_lagged_and_bounded() -> None:
    values = [np.array([0.05, 0.10, 0.20, 0.40])]
    value_estimates = [np.array([0.04, 0.08, 0.18, 0.35])]
    lwc = [np.array([0.20, 0.50, 1.50, 4.00])]
    traffic = estimate_traffic_aware_reference_action(1.0, 8, values, lwc, 7.5)
    smoothed = estimate_smoothed_controller_reference_action([4, 8, 12], 7.5)
    response = estimate_response_aware_reference_action(1.0, 8, values, lwc, value_estimates, 10.0, 7.5)
    assert 0 <= traffic <= 100
    assert np.isclose(smoothed, 8)
    assert 0 <= response <= 100


def test_local_intervention_mask_identities() -> None:
    raw_initial = np.array([1, 1, 0, 1, 0], dtype=bool)
    executed_initial = np.array([1, 0, 1, 1, 0], dtype=bool)
    raw_budget = np.array([1, 0, 0, 1, 0], dtype=bool)
    executed_budget = np.array([0, 0, 1, 1, 0], dtype=bool)
    masks = derive_intervention_masks(raw_initial, executed_initial, raw_budget, executed_budget)
    assert np.array_equal(masks["forced_eligible"], [0, 0, 1, 0, 0])
    assert np.array_equal(masks["blocked_eligible"], [0, 1, 0, 0, 0])
    assert np.array_equal(masks["direct_forced_actual"], [0, 0, 1, 0, 0])
    assert np.array_equal(masks["direct_blocked_feasible"], [0, 0, 0, 0, 0])
    assert np.array_equal(masks["allocation_exec_only"], [0, 0, 1, 0, 0])
    assert np.array_equal(masks["allocation_raw_only"], [1, 0, 0, 0, 0])


def test_score_penalty() -> None:
    assert np.isclose(nips_score(10, 80, 10), 10)
    assert np.isclose(nips_score(10, 200, 10), 2.5)


def test_cartesian_grid_expansion() -> None:
    spec = {
        "base": {"policy_mode": "hard", "q_scale": 1.0},
        "cartesian": {"h": [2, 4], "kappa": [0.0, 0.3]},
        "include": [{"id": "raw", "policy_mode": "raw", "h": 2, "kappa": 99}],
    }
    result = expand_cartesian_spec(spec)
    assert len(result) == 5
    assert result[0]["id"] == "h=2__kappa=0.0"
    assert result[-1]["id"] == "raw"
