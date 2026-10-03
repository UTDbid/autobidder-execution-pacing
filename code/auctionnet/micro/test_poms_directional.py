import numpy as np

from mandate_core_v2 import enforce_action, enforce_asymmetric_action
from mandate_eval_confirmatory import stable_hash_order


def test_symmetric_directional_matches_legacy_relative_hard():
    for raw in [0.0, 2.0, 10.0, 18.0, 30.0]:
        legacy = enforce_action(raw, 10.0, 0.8, operator="relative_hard")
        directional = enforce_asymmetric_action(raw, 10.0, 0.8, 0.8)
        assert directional == legacy


def test_directional_semantics():
    assert np.isclose(enforce_asymmetric_action(1.0, 10.0, 0.8, 0.0)[0], 2.0)
    assert np.isclose(enforce_asymmetric_action(20.0, 10.0, 0.8, 0.0)[0], 10.0)
    assert np.isclose(enforce_asymmetric_action(1.0, 10.0, 0.0, 0.8)[0], 10.0)
    assert np.isclose(enforce_asymmetric_action(20.0, 10.0, 0.0, 0.8)[0], 18.0)


def test_hash_orders_are_deterministic_nested_permutations():
    first = stable_hash_order(100, 17, [7, 0], 3)
    second = stable_hash_order(100, 17, [7, 0], 3)
    other = stable_hash_order(100, 18, [7, 0], 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(100))
    assert not np.array_equal(first, other)
    np.testing.assert_array_equal(first[:20], stable_hash_order(100, 17, [7, 0], 3)[:20])
