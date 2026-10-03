"""Treatment-cell and adopter-assignment design."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DesignCell:
    alpha: float
    adopter_policy: str
    adopter_kappa: float
    low_kappa: float
    rotation_agent: int | None
    adopter_policy_mode: str = "bounded"
    low_policy_mode: str = "bounded"

    @property
    def cell_id(self) -> str:
        alpha_label = "1of48" if np.isclose(self.alpha, 1.0 / 48.0) else f"{self.alpha:g}"
        rotation = "none" if self.rotation_agent is None else f"a{self.rotation_agent:02d}"
        return f"alpha={alpha_label}__policy={self.adopter_policy}__rotation={rotation}"


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _quantile_bins(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("binning input must be one-dimensional")
    # Rank-based bins remain deterministic when empirical quantiles coincide.
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    return np.minimum((ranks * n_bins) // max(values.size, 1), n_bins - 1).astype(np.int64)


def select_adopters(
    alpha: float,
    budgets: np.ndarray,
    cpa_constraints: np.ndarray,
    categories: np.ndarray,
    *,
    assignment_seed: int,
    period: int,
    rotation_agent: int | None = None,
    ranking_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Select an exact-sized, block-hash-randomized adopter set.

    Budget and CPA rank terciles plus advertiser category define operational
    blocks.  A within-block fractional rank, randomized by a stable hash, is
    used as the global selection score.  This yields approximately proportional
    block representation while retaining an exact adopter count.
    """

    budgets = np.asarray(budgets, dtype=np.float64)
    cpas = np.asarray(cpa_constraints, dtype=np.float64)
    categories = np.asarray(categories, dtype=np.int64)
    if budgets.ndim != 1 or budgets.shape != cpas.shape or budgets.shape != categories.shape:
        raise ValueError("budget, CPA, and category vectors must share a 1D shape")
    n_agents = budgets.size
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    target = int(round(float(alpha) * n_agents))
    selected = np.zeros(n_agents, dtype=bool)
    if target == 0:
        return selected
    if target == 1 and rotation_agent is not None:
        if not 0 <= int(rotation_agent) < n_agents:
            raise ValueError("rotation_agent out of range")
        selected[int(rotation_agent)] = True
        return selected
    if target == n_agents:
        selected[:] = True
        return selected
    if ranking_scores is not None:
        scores = np.asarray(ranking_scores, dtype=np.float64)
        if scores.shape != (n_agents,) or not np.all(np.isfinite(scores)):
            raise ValueError("ranking_scores must be a finite vector with one value per advertiser")
        chosen = np.lexsort((np.arange(n_agents), -scores))[:target]
        selected[chosen] = True
        return selected

    budget_bins = _quantile_bins(budgets)
    cpa_bins = _quantile_bins(cpas)
    blocks: dict[tuple[int, int, int], list[int]] = {}
    for agent in range(n_agents):
        blocks.setdefault((int(budget_bins[agent]), int(cpa_bins[agent]), int(categories[agent])), []).append(agent)
    scores = np.empty(n_agents, dtype=np.float64)
    for block, members in blocks.items():
        ordered = sorted(
            members,
            key=lambda agent: (_stable_uniform(assignment_seed, period, *block, agent), agent),
        )
        size = len(ordered)
        for rank, agent in enumerate(ordered):
            jitter = _stable_uniform("jitter", assignment_seed, period, *block, agent)
            scores[agent] = (rank + jitter) / size
    chosen = np.argsort(scores, kind="stable")[:target]
    selected[chosen] = True
    return selected


def make_design_cells(
    alphas: Iterable[float],
    adopter_policies: Iterable[dict],
    *,
    low_kappa: float,
    n_agents: int = 48,
    rotate_single_adopter: bool = True,
) -> list[DesignCell]:
    """Expand the frozen alpha-by-autonomy design without duplicate baseline."""

    cells: list[DesignCell] = []
    if low_kappa < 0:
        raise ValueError("low_kappa must be nonnegative")
    for alpha in alphas:
        alpha_value = float(alpha)
        if np.isclose(alpha_value, 0.0):
            cells.append(
                DesignCell(
                    0.0,
                    "baseline_low",
                    float(low_kappa),
                    float(low_kappa),
                    None,
                    "bounded",
                    "bounded",
                )
            )
            continue
        rotations: list[int | None]
        if rotate_single_adopter and int(round(alpha_value * n_agents)) == 1:
            rotations = list(range(n_agents))
        else:
            rotations = [None]
        for policy in adopter_policies:
            policy_mode = str(policy.get("policy_mode", "bounded"))
            if policy_mode not in {"bounded", "raw"}:
                raise ValueError(f"unsupported policy_mode: {policy_mode}")
            policy_kappa = float(policy.get("kappa", 0.0))
            if policy_kappa < 0:
                raise ValueError("adopter kappa must be nonnegative")
            for rotation in rotations:
                cells.append(
                    DesignCell(
                        alpha_value,
                        str(policy["label"]),
                        policy_kappa,
                        float(low_kappa),
                        rotation,
                        policy_mode,
                        "bounded",
                    )
                )
    identifiers = [cell.cell_id for cell in cells]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("design contains duplicate cells")
    return cells
