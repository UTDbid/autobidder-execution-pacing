"""Pure market-clearing and budget-allocation primitives.

The PlatformBid reference implementation first computes a third-highest bid
threshold and then tests ``bid >= threshold``.  Under ties (and even without
ties for the third-ranked bidder), that can allocate more than two slots.  The
functions here make the intended mechanism explicit: exactly the top two
positive bids are provisionally allocated, both pay the third-highest bid, and
a deterministic CRN tie-break resolves equal bids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuctionOutcome:
    """Provisional synchronous-auction outcome before budget enforcement."""

    winners: np.ndarray
    payments: np.ndarray
    clearing_prices: np.ndarray
    ranking: np.ndarray


def uniform_third_price_top2(
    bids: np.ndarray,
    tie_break: np.ndarray,
    *,
    num_slots: int = 2,
) -> AuctionOutcome:
    """Rank all agents jointly and allocate two slots at the third price.

    Parameters
    ----------
    bids:
        Matrix ``[agent, opportunity]``.  Negative/nonfinite bids are treated
        as zero.
    tie_break:
        CRN matrix with the same shape.  Smaller values win an exact bid tie;
        agent id is the final deterministic key.
    num_slots:
        Kept as an explicit assertion so accidental changes to the mechanism
        fail loudly.  This evaluator implements the PlatformBid two-slot case.
    """

    if num_slots != 2:
        raise ValueError("uniform_third_price_top2 requires exactly two slots")
    bid_matrix = np.asarray(bids, dtype=np.float64)
    tie_matrix = np.asarray(tie_break, dtype=np.float64)
    if bid_matrix.ndim != 2 or bid_matrix.shape != tie_matrix.shape:
        raise ValueError("bids and tie_break must be same-shape 2D matrices")
    n_agents, n_opportunities = bid_matrix.shape
    if n_agents < 3:
        raise ValueError("third-price auction requires at least three agents")
    safe_bids = np.where(np.isfinite(bid_matrix) & (bid_matrix > 0), bid_matrix, 0.0)
    safe_ties = np.where(np.isfinite(tie_matrix), tie_matrix, np.inf)
    agent_ids = np.broadcast_to(np.arange(n_agents, dtype=np.int64)[:, None], safe_bids.shape)
    # np.lexsort uses the last key as primary: bid descending, then the CRN
    # draw ascending, then agent id ascending as a collision-proof final key.
    ranking = np.lexsort((agent_ids, safe_ties, -safe_bids), axis=0)
    ordered_bids = np.take_along_axis(safe_bids, ranking, axis=0)
    clearing_prices = ordered_bids[2].copy()
    winners = np.zeros_like(safe_bids, dtype=bool)
    payments = np.zeros_like(safe_bids, dtype=np.float64)
    columns = np.arange(n_opportunities)
    for slot in range(2):
        winning_agents = ranking[slot]
        active = ordered_bids[slot] > 0
        winners[winning_agents[active], columns[active]] = True
        payments[winning_agents[active], columns[active]] = clearing_prices[active]
    return AuctionOutcome(winners=winners, payments=payments, clearing_prices=clearing_prices, ranking=ranking)


def enforce_agent_budgets(
    provisional_winners: np.ndarray,
    provisional_payments: np.ndarray,
    remaining_budgets: np.ndarray,
    budget_priority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a paired, branch-free budget mask independently to each agent.

    When a provisional winner cannot afford all opportunities in a tick, CRN
    priorities select an affordable subset.  Removed slots are left unfilled;
    the function never redraws conversions or reranks the auction.
    """

    winners = np.asarray(provisional_winners, dtype=bool)
    payments = np.asarray(provisional_payments, dtype=np.float64)
    budgets = np.asarray(remaining_budgets, dtype=np.float64)
    priorities = np.asarray(budget_priority, dtype=np.float64)
    if winners.ndim != 2 or payments.shape != winners.shape or priorities.shape != winners.shape:
        raise ValueError("winner, payment, and priority matrices must share a 2D shape")
    if budgets.shape != (winners.shape[0],):
        raise ValueError("remaining_budgets must contain one value per agent")
    if np.any(payments < 0) or np.any(~np.isfinite(payments)):
        raise ValueError("payments must be finite and nonnegative")
    final_winners = winners.copy()
    for agent in range(winners.shape[0]):
        indices = np.flatnonzero(final_winners[agent])
        if indices.size == 0:
            continue
        costs = payments[agent, indices]
        if float(np.sum(costs)) <= budgets[agent] + 1e-9:
            continue
        order = indices[np.argsort(priorities[agent, indices], kind="stable")]
        final_winners[agent] = False
        cumulative = 0.0
        for opportunity in order:
            cost = float(payments[agent, opportunity])
            if cumulative + cost <= budgets[agent] + 1e-9:
                final_winners[agent, opportunity] = True
                cumulative += cost
    return final_winners, payments * final_winners
