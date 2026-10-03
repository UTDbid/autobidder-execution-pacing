"""Read official AuctionNet periods as synchronized 48-agent ticks."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "deliveryPeriodIndex",
    "advertiserNumber",
    "advertiserCategoryIndex",
    "budget",
    "CPAConstraint",
    "timeStepIndex",
    "pvIndex",
    "pValue",
    "pValueSigma",
    "leastWinningCost",
]


@dataclass(frozen=True)
class PeriodTick:
    period: int
    time_step: int
    pv_indices: np.ndarray
    p_values: np.ndarray
    p_value_sigmas: np.ndarray
    original_lwc: np.ndarray
    budgets: np.ndarray
    cpa_constraints: np.ndarray
    categories: np.ndarray


def assert_unsealed_period_input(archive: Path, member: str, expected_period: int, phase: str) -> None:
    text = f"{archive.name}|{member}".lower().replace("_", "-")
    if re.search(r"period-?13", text) or int(expected_period) == 13:
        raise RuntimeError("SEALED_SPLIT_GUARD: period 13 is forbidden")
    if re.search(r"period-?8", text) or int(expected_period) == 8:
        raise RuntimeError("SEALED_SPLIT_GUARD: period 8 is unrelated and forbidden")
    allowed = {"pilot": {9, 10}, "formal": {11, 12}}
    if phase not in allowed or int(expected_period) not in allowed[phase]:
        raise RuntimeError(f"PHASE_BOUNDARY: {phase} cannot read period {expected_period}")


def _frame_to_tick(frame: pd.DataFrame, expected_period: int, n_agents: int) -> PeriodTick:
    periods = np.unique(frame["deliveryPeriodIndex"].to_numpy(dtype=np.int64))
    time_steps = np.unique(frame["timeStepIndex"].to_numpy(dtype=np.int64))
    if periods.tolist() != [expected_period] or time_steps.size != 1:
        raise ValueError("tick contains unexpected period or multiple time steps")
    work = frame.sort_values(["pvIndex", "advertiserNumber"], kind="stable")
    pv = work["pvIndex"].to_numpy(dtype=np.int64)
    agents = work["advertiserNumber"].to_numpy(dtype=np.int64)
    unique_pv, counts = np.unique(pv, return_counts=True)
    if np.any(counts != n_agents) or work.shape[0] != unique_pv.size * n_agents:
        raise ValueError("each opportunity must contain exactly one row per agent")
    agent_matrix = agents.reshape(unique_pv.size, n_agents)
    expected_agents = np.arange(n_agents, dtype=np.int64)
    if not np.all(agent_matrix == expected_agents[None, :]):
        raise ValueError("advertiser ids are missing, duplicated, or outside 0..47")

    def matrix(column: str) -> np.ndarray:
        return work[column].to_numpy(dtype=np.float64).reshape(unique_pv.size, n_agents).T.copy()

    budgets_by_agent = work.groupby("advertiserNumber", sort=True)["budget"].first().to_numpy(dtype=np.float64)
    cpas_by_agent = work.groupby("advertiserNumber", sort=True)["CPAConstraint"].first().to_numpy(dtype=np.float64)
    categories_by_agent = work.groupby("advertiserNumber", sort=True)["advertiserCategoryIndex"].first().to_numpy(dtype=np.int64)
    original_lwc_matrix = matrix("leastWinningCost")
    # The logged LWC is a diagnostic from the original market.  It must be
    # common across advertisers for a shared opportunity but is not used to
    # clear the reconstructed simultaneous market.
    if not np.allclose(original_lwc_matrix, original_lwc_matrix[0:1], rtol=0, atol=1e-10):
        raise ValueError("leastWinningCost differs across agents for a shared opportunity")
    return PeriodTick(
        period=expected_period,
        time_step=int(time_steps[0]),
        pv_indices=unique_pv,
        p_values=matrix("pValue"),
        p_value_sigmas=matrix("pValueSigma"),
        original_lwc=original_lwc_matrix[0].copy(),
        budgets=budgets_by_agent,
        cpa_constraints=cpas_by_agent,
        categories=categories_by_agent,
    )


def iter_period_ticks(
    archive: Path,
    member: str,
    *,
    expected_period: int,
    phase: str,
    n_agents: int = 48,
    chunk_rows: int = 2_000_000,
    max_timesteps: int | None = None,
    max_pvs: int | None = None,
) -> Iterator[PeriodTick]:
    """Stream complete time-step groups from a zipped official CSV or cache."""

    archive = Path(archive)
    assert_unsealed_period_input(archive, member, expected_period, phase)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    pending: pd.DataFrame | None = None
    last_yielded = -1
    yielded = 0
    if archive.suffix.lower() == ".zip":
        bundle = zipfile.ZipFile(archive, "r")
        if member not in bundle.namelist():
            bundle.close()
            raise FileNotFoundError(f"{member} not found in {archive}")
        handle = bundle.open(member, "r")
    else:
        if archive.name != Path(member).name:
            raise ValueError("plain CSV archive path must match --member basename")
        bundle = None
        handle = archive.open("rb")
    try:
        with handle:
            chunks = pd.read_csv(handle, usecols=REQUIRED_COLUMNS, chunksize=chunk_rows)
            for chunk in chunks:
                if pending is not None:
                    chunk = pd.concat([pending, chunk], ignore_index=True)
                steps = chunk["timeStepIndex"].to_numpy(dtype=np.int64)
                if steps.size == 0:
                    pending = chunk
                    continue
                maximum = int(np.max(steps))
                complete = chunk.loc[steps < maximum]
                pending = chunk.loc[steps == maximum].copy()
                for time_step, group in complete.groupby("timeStepIndex", sort=True):
                    integer_step = int(time_step)
                    if integer_step <= last_yielded:
                        raise ValueError("input CSV is not globally ordered by timeStepIndex")
                    tick = _frame_to_tick(group, expected_period, n_agents)
                    if max_pvs is not None:
                        tick = PeriodTick(
                            period=tick.period,
                            time_step=tick.time_step,
                            pv_indices=tick.pv_indices[:max_pvs],
                            p_values=tick.p_values[:, :max_pvs],
                            p_value_sigmas=tick.p_value_sigmas[:, :max_pvs],
                            original_lwc=tick.original_lwc[:max_pvs],
                            budgets=tick.budgets,
                            cpa_constraints=tick.cpa_constraints,
                            categories=tick.categories,
                        )
                    yield tick
                    yielded += 1
                    last_yielded = integer_step
                    if max_timesteps is not None and yielded >= max_timesteps:
                        return
            if pending is not None and not pending.empty:
                for time_step, group in pending.groupby("timeStepIndex", sort=True):
                    if max_timesteps is not None and yielded >= max_timesteps:
                        return
                    integer_step = int(time_step)
                    if integer_step <= last_yielded:
                        raise ValueError("input CSV is not globally ordered by timeStepIndex")
                    tick = _frame_to_tick(group, expected_period, n_agents)
                    if max_pvs is not None:
                        tick = PeriodTick(
                            tick.period, tick.time_step, tick.pv_indices[:max_pvs],
                            tick.p_values[:, :max_pvs], tick.p_value_sigmas[:, :max_pvs],
                            tick.original_lwc[:max_pvs], tick.budgets,
                            tick.cpa_constraints, tick.categories,
                        )
                    yield tick
                    yielded += 1
    finally:
        if bundle is not None:
            bundle.close()
