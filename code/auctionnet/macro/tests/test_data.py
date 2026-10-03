from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from platformbid_v1.data import _frame_to_tick, assert_unsealed_period_input, iter_period_ticks


def synthetic_frame(period=9, time_step=0, n_agents=48, n_pvs=3):
    rows = []
    for pv in range(n_pvs):
        for agent in range(n_agents):
            rows.append(
                {
                    "deliveryPeriodIndex": period,
                    "advertiserNumber": agent,
                    "advertiserCategoryIndex": agent % 6,
                    "budget": 100 + agent,
                    "CPAConstraint": 6 + agent % 8,
                    "timeStepIndex": time_step,
                    "pvIndex": pv,
                    "pValue": 0.01 * (agent + 1),
                    "pValueSigma": 0.001,
                    "leastWinningCost": 0.2 + pv,
                }
            )
    return pd.DataFrame(rows)


def test_frame_to_tick_preserves_pv_and_agent_alignment():
    frame = synthetic_frame().sample(frac=1, random_state=42)
    tick = _frame_to_tick(frame, expected_period=9, n_agents=48)
    assert tick.p_values.shape == (48, 3)
    assert tick.pv_indices.tolist() == [0, 1, 2]
    assert tick.p_values[7, 2] == pytest.approx(0.08)
    assert tick.original_lwc.tolist() == pytest.approx([0.2, 1.2, 2.2])


def test_duplicate_or_missing_agent_fails():
    frame = synthetic_frame(n_pvs=1).iloc[:-1]
    with pytest.raises(ValueError, match="exactly one row"):
        _frame_to_tick(frame, expected_period=9, n_agents=48)


def test_lwc_disagreement_across_agents_fails():
    frame = synthetic_frame(n_pvs=1)
    frame.loc[1, "leastWinningCost"] = 9.0
    with pytest.raises(ValueError, match="differs across agents"):
        _frame_to_tick(frame, expected_period=9, n_agents=48)


@pytest.mark.parametrize(
    "archive,member,period,phase",
    [
        ("period_13.zip", "period-13.csv", 13, "formal"),
        ("period_9_10.zip", "period-10.csv", 10, "formal"),
        ("period_11_12.zip", "period-11.csv", 11, "pilot"),
        ("period_8.zip", "period-8.csv", 8, "pilot"),
    ],
)
def test_sealed_and_phase_guards(archive, member, period, phase):
    with pytest.raises(RuntimeError):
        assert_unsealed_period_input(Path(archive), member, period, phase)


def test_plain_csv_cache_streams_complete_ticks(tmp_path: Path):
    path = tmp_path / "period-9.csv"
    pd.concat([synthetic_frame(time_step=0), synthetic_frame(time_step=1)]).to_csv(path, index=False)
    ticks = list(
        iter_period_ticks(
            path, "period-9.csv", expected_period=9, phase="pilot", n_agents=48, chunk_rows=155
        )
    )
    assert [tick.time_step for tick in ticks] == [0, 1]
    assert all(tick.p_values.shape == (48, 3) for tick in ticks)
