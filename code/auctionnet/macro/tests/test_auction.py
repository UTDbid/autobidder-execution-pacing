import numpy as np

from platformbid_v1.auction import enforce_agent_budgets, uniform_third_price_top2


def test_top2_only_and_both_pay_third_price():
    bids = np.array([[10.0], [9.0], [8.0], [7.0]])
    outcome = uniform_third_price_top2(bids, np.zeros_like(bids))
    assert outcome.winners[:, 0].tolist() == [True, True, False, False]
    assert outcome.clearing_prices.tolist() == [8.0]
    assert outcome.payments[:, 0].tolist() == [8.0, 8.0, 0.0, 0.0]


def test_third_bidder_does_not_win_when_bid_equals_threshold():
    bids = np.array([[10.0], [8.0], [8.0], [1.0]])
    ties = np.array([[0.9], [0.2], [0.1], [0.0]])
    outcome = uniform_third_price_top2(bids, ties)
    assert outcome.winners[:, 0].sum() == 2
    assert outcome.winners[0, 0]
    assert outcome.winners[2, 0]
    assert not outcome.winners[1, 0]
    assert outcome.clearing_prices[0] == 8.0


def test_tie_break_is_crn_then_agent_id():
    bids = np.full((4, 2), 5.0)
    ties = np.array([[0.4, 0.1], [0.2, 0.1], [0.3, 0.9], [0.1, 0.8]])
    outcome = uniform_third_price_top2(bids, ties)
    assert outcome.winners[:, 0].tolist() == [False, True, False, True]
    assert outcome.winners[:, 1].tolist() == [True, True, False, False]


def test_zero_bids_leave_slots_empty():
    bids = np.zeros((4, 3))
    outcome = uniform_third_price_top2(bids, np.zeros_like(bids))
    assert not outcome.winners.any()
    assert not outcome.payments.any()


def test_budget_mask_is_agent_local_and_affordable_without_redraw():
    provisional = np.array([[True, True, True], [True, False, False]])
    payments = np.array([[4.0, 4.0, 4.0], [2.0, 0.0, 0.0]])
    priorities = np.array([[0.3, 0.1, 0.2], [0.9, 0.0, 0.0]])
    winners, costs = enforce_agent_budgets(provisional, payments, np.array([8.0, 1.0]), priorities)
    assert winners[0].tolist() == [False, True, True]
    assert not winners[1].any()
    assert np.sum(costs[0]) == 8.0
    assert np.sum(costs[1]) == 0.0
