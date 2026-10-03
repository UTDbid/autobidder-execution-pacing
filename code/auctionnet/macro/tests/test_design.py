import numpy as np

from platformbid_v1.design import make_design_cells, select_adopters


def contracts():
    return (
        np.arange(1, 49, dtype=float),
        np.tile(np.arange(6, 14, dtype=float), 6),
        np.repeat(np.arange(6), 8),
    )


def test_assignment_has_exact_size_and_is_deterministic():
    budget, cpa, category = contracts()
    first = select_adopters(0.25, budget, cpa, category, assignment_seed=7, period=9)
    second = select_adopters(0.25, budget, cpa, category, assignment_seed=7, period=9)
    assert first.sum() == 12
    assert np.array_equal(first, second)


def test_single_adopter_rotation_is_explicit():
    budget, cpa, category = contracts()
    selected = select_adopters(1 / 48, budget, cpa, category, assignment_seed=7, period=9, rotation_agent=31)
    assert selected.sum() == 1
    assert selected[31]


def test_design_has_one_baseline_and_96_private_rotation_cells():
    cells = make_design_cells(
        [0, 1 / 48, 0.25, 0.5, 0.75, 1],
        [{"label": "private", "kappa": 0.8}, {"label": "high", "kappa": 1.2}],
        low_kappa=0.1,
    )
    assert len([cell for cell in cells if cell.alpha == 0]) == 1
    assert len([cell for cell in cells if np.isclose(cell.alpha, 1 / 48)]) == 96
    # Baseline + 96 rotations + four positive non-single alpha levels x 2 policies.
    assert len(cells) == 105
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_ranked_assignment_selects_highest_scores_exactly():
    budgets = np.ones(8)
    cpas = np.ones(8)
    categories = np.zeros(8, dtype=np.int64)
    scores = np.asarray([0.2, 9.0, 0.1, 4.0, 2.0, 3.0, -1.0, 8.0])
    selected = select_adopters(
        0.5,
        budgets,
        cpas,
        categories,
        assignment_seed=1,
        period=11,
        ranking_scores=scores,
    )
    assert np.flatnonzero(selected).tolist() == [1, 3, 5, 7]


def test_raw_policy_is_explicit_and_does_not_change_baseline_mode():
    cells = make_design_cells(
        [0, 1],
        [{"label": "raw", "policy_mode": "raw"}],
        low_kappa=0.3,
    )
    baseline, raw = cells
    assert baseline.adopter_policy_mode == "bounded"
    assert baseline.low_policy_mode == "bounded"
    assert raw.adopter_policy_mode == "raw"
    assert raw.low_policy_mode == "bounded"


def test_unknown_policy_mode_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "unsupported policy_mode"):
        make_design_cells(
            [1],
            [{"label": "bad", "policy_mode": "huge-kappa"}],
            low_kappa=0.3,
        )
