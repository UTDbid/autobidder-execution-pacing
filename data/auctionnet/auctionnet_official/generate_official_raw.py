#!/usr/bin/env python3
"""Generate raw AuctionNet period CSVs using the OFFICIAL AuctionNet flow.

This script mirrors the structure of ``run/run_test.py`` from the official
``alimama-tech/AuctionNet`` repository (cloned under
``external/auctionnet_official_<timestamp>/``). It preserves:

  * ``Controller.initialize_agents()`` mixed strategy pool (9 distinct classes
    across 48 advertisers — see external/auctionnet_official_*/simul_bidding_env/
    Controller/Controller.py:38-64). The agent_strategy_counts manifest field
    must NOT be all-PID; this script MUST fail if that invariant is broken.
  * The official over-cost adjustment loop
    (``while ratio_max > 0: adjust_over_cost(...)`` — run/run_test.py:184-199).
  * ``NeurIPSPvGen`` PV generator + ``BiddingTracker.train_logging`` 18-column
    raw schema (deliveryPeriodIndex, advertiserNumber, ..., isEnd).

The ONLY project-specific override allowed is the 48-entry CPA list:

    high (CVR high):  affine rescale [60..130] -> [6..13]
    mid  (CVR mid ):  affine rescale [60..130] -> [20..60]
    low  (CVR low ):  [60..130] verbatim (matches official)

Budget list, agent pool, PV generator, over-cost loop, tracker schema,
auction env — ALL preserved verbatim from the official repo.

Usage::

    PYTHONPATH=external/auctionnet_official_<ts>:src \
    python scripts/data/auctionnet_official/generate_official_raw.py \
        --level mid \
        --num-episodes 6 \
        --num-ticks 48 \
        --pv-num 500000 \
        --output-dir data/mid/raw_official_20gb \
        --manifest data/manifests/sample_mid_train_official_20gb.json

Each episode writes ``<output-dir>/period-<episode>.csv`` (18 cols, raw schema).
The manifest records: agent_strategy_counts, rows, size_bytes, schema,
CPA min/max, advertiser count, timeStepIndex range, episode count, pv_num.
The script ASSERTs: agent_strategy_counts is NOT all-PID (else raises).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402

# Make the OFFICIAL repo importable. Caller must also pass our own src/
# on PYTHONPATH for project_paths.
# Path: scripts/data/auctionnet_official/generate_official_raw.py
#   parents[0]=scripts/data/auctionnet_official
#   parents[1]=scripts/data
#   parents[2]=scripts
#   parents[3]=<repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_PATH = Path(os.environ.get(
    "AUCTIONNET_OFFICIAL_PATH",
    str(_REPO_ROOT / "external" / "auctionnet_official_20260618_091733"),
))
if not _OFFICIAL_PATH.exists():
    raise SystemExit(f"Official repo not found: {_OFFICIAL_PATH}")
if str(_OFFICIAL_PATH) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_PATH))
# strategy_train_env/ contains bidding_train_env.* if anything in generate
# needs it later (e.g. inline trajectory conversion). Safe to add.
_STRATEGY_TRAIN_ENV = _OFFICIAL_PATH / "strategy_train_env"
if _STRATEGY_TRAIN_ENV.exists() and str(_STRATEGY_TRAIN_ENV) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_TRAIN_ENV))
# Also insert src/ for project_paths.
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Official imports (must come AFTER sys.path edits).
from simul_bidding_env.Controller.Controller import Controller  # noqa: E402
from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker  # noqa: E402
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy  # noqa: E402

from common.project_paths import get_repo_paths  # noqa: E402


# Project-level CPA profile (the ONLY override vs. official).
# Official default is the 48-entry [60..130] list in Controller.py:103-112.
# We rescale that list affinely to the project's canonical CPA bands.
_OFFICIAL_CPA_LIST = [
    100, 70, 90, 110, 60, 130, 120, 80,
    70, 130, 100, 110, 120, 90, 60, 80,
    130, 80, 110, 100, 90, 120, 60, 70,
    120, 60, 90, 70, 100, 110, 130, 80,
    120, 90, 70, 80, 100, 110, 60, 130,
    90, 100, 110, 80, 60, 70, 130, 120,
]
assert len(_OFFICIAL_CPA_LIST) == 48

_LEVEL_CPA_RANGE = {
    "high": (6.0, 13.0),   # CVR high -> small CPA
    "mid": (20.0, 60.0),
    "low": (60.0, 130.0),  # verbatim from official
}


def _rescale_cpa_list(level: str) -> list[float]:
    """Affine rescale the official 48-entry CPA list to the level's range.

    The relative ordering is preserved; only the numeric range changes.
    """
    if level not in _LEVEL_CPA_RANGE:
        raise ValueError(f"Unknown level: {level}")
    if level == "low":
        return [float(x) for x in _OFFICIAL_CPA_LIST]
    lo_target, hi_target = _LEVEL_CPA_RANGE[level]
    src_min = min(_OFFICIAL_CPA_LIST)
    src_max = max(_OFFICIAL_CPA_LIST)
    out = []
    for v in _OFFICIAL_CPA_LIST:
        norm = (v - src_min) / (src_max - src_min)
        out.append(round(lo_target + norm * (hi_target - lo_target), 4))
    return out


def _install_cpa_profile(controller: Controller, level: str) -> list[float]:
    """Override Controller.cpa_constraint_list with the level's profile.

    Returns the new list (48 floats).
    """
    import numpy as np
    new_cpa = _rescale_cpa_list(level)
    controller.cpa_constraint_list = np.asarray(new_cpa, dtype=float)
    # Also patch each agent's cpa so subsequent bidding() calls see it.
    for i, agent in enumerate(controller.agents):
        agent.cpa = float(new_cpa[i])
    return new_cpa


def _run_episode(
    controller: Controller,
    tracker: BiddingTracker,
    *,
    episode: int,
    num_tick: int,
    pv_generator,
    envs,
    agents,
    agents_cpa,
    agents_category,
    budgets: list[float],
) -> dict[str, Any]:
    """Run one episode using the OFFICIAL loop (mirrors run_test.py:138-221)."""
    tracker.reset()

    rewards = [0.0] * len(agents)
    costs = [0.0] * len(agents)
    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    controller.reset(episode=episode)

    total_pv_num = 0
    for tick_index in range(num_tick):
        pv_values = pv_generator.pv_values[tick_index]
        pvalue_sigmas = pv_generator.pValueSigmas[tick_index]

        bids = [
            agent.bidding(
                tick_index,
                pv_values[:, i],
                pvalue_sigmas[:, i],
                [x[i] for x in history_pvalue_infos],
                [x[i] for x in history_bids],
                [x[i] for x in history_auction_results],
                [x[i] for x in history_impression_results],
                history_least_winning_costs,
            )
            if agent.remaining_budget >= envs.min_remaining_budget
            else np.zeros(pv_values.shape[0])
            for i, agent in enumerate(agents)
        ]
        bids = np.array(bids).transpose()
        bids[bids < 0] = 0

        remaining_budget_list = np.array([agent.remaining_budget for agent in agents])
        done_list = (
            np.ones(len(agents), dtype=int)
            if tick_index == (num_tick - 1)
            else (remaining_budget_list < envs.min_remaining_budget).astype(int)
        )

        # Official over-cost adjustment loop (run_test.py:184-199).
        ratio_max = None
        winner_pit = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                over_cost_ratio = np.maximum(
                    (cost - remaining_budget_list) / (cost + 1e-4), 0
                )
                _adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)
            xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, least_winning_cost_pit, _ = (
                envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
            )
            real_cost = cost_pit * is_exposed_pit
            cost = real_cost.sum(axis=1)
            reward = conversion_action_pit.sum(axis=1)
            winner_pit = _get_winner(slot_pit)
            over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]
        for i in range(len(agents)):
            rewards[i] += reward[i]
            costs[i] += cost[i]

        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        history_pvalue_infos.append(pvalue_info)
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        history_auction_results.append(auction_info)
        impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        history_impression_results.append(impression_info)

        tracker.train_logging(
            episode, tick_index, pv_values, budgets, agents_cpa, agents_category,
            remaining_budget_list, total_pv_num, pvalue_sigmas, bids,
            xi_pit, slot_pit, cost_pit, is_exposed_pit,
            conversion_action_pit, least_winning_cost_pit, done_list,
        )
        total_pv_num += pv_values.shape[0]

    return {
        "episode": episode,
        "rewards": rewards,
        "costs": costs,
        "total_pv_num": total_pv_num,
    }


def _adjust_over_cost(bids, over_cost_ratio, envs_slots, winner_pit):
    """Verbatim copy of run_test.py:70-83."""
    overcost_agent_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_agent_indices:
        for i, _ in enumerate(envs_slots):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_to_drop > 0:
                dropped = rng.choice(pv_indices, num_to_drop, replace=False)
                bids[dropped, agent_index] = 0


def _get_winner(slot_pit):
    """Verbatim copy of run_test.py:53-67."""
    slot_pit = slot_pit.T
    num_pv, num_agent = slot_pit.shape
    num_slot = 3
    winner = np.full((num_pv, num_slot), -1, dtype=int)
    for pos in range(1, num_slot + 1):
        winning_agents_indices = np.argwhere(slot_pit == pos)
        if winning_agents_indices.size > 0:
            pv_indices, agent_indices = winning_agents_indices.T
            winner[pv_indices, pos - 1] = agent_indices
    return winner


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="generate_official_raw",
        description=(
            "Generate raw AuctionNet period CSVs using the OFFICIAL mixed-strategy "
            "Controller flow. Only the CPA list is project-overridden."
        ),
    )
    parser.add_argument("--level", required=True, choices=["high", "mid", "low"],
                        help="CPA profile: high=6-13, mid=20-60, low=60-130 (official)")
    parser.add_argument("--split", required=True, choices=["train", "test"],
                        help="Logical split — informational; used by downstream")
    parser.add_argument("--num-episodes", type=int, default=2,
                        help="Episodes to run (default: 2, matches config/test.gin)")
    parser.add_argument("--episode-start", type=int, default=0,
                        help=("Starting source episode id (default 0). Use >=100000 for "
                              "test split to avoid colliding with train."))
    parser.add_argument("--num-ticks", type=int, default=48,
                        help="Ticks per episode (default: 48, matches config/test.gin)")
    parser.add_argument("--pv-num", type=int, default=500000,
                        help="PVs per tick (default: 500000, matches config/test.gin)")
    parser.add_argument("--player-index", type=int, default=0,
                        help="Player slot in the 48-agent population")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write period-<episode>.csv files")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest JSON output path")
    parser.add_argument("--min-total-bytes", type=int, default=20 * 1024 ** 3,
                        help="Loop continues appending episodes until total raw size >= this")
    parser.add_argument("--max-episodes", type=int, default=64,
                        help="Hard cap to avoid infinite loop")
    return _run_main(parser.parse_args())


def _run_main(args) -> int:
    import numpy as np

    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Build Controller with official defaults + project CPA override.
    player_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    controller = Controller(
        player_index=args.player_index,
        player_agent=player_agent,
        num_tick=args.num_ticks,
        num_agent_category=8,
        num_category=6,
        pv_num=args.pv_num,
        pv_generator_type="neuripsPvGen",
    )
    agents = controller.agents
    agents_category = np.array([agent.category for agent in agents])
    new_cpa = _install_cpa_profile(controller, args.level)
    agents_cpa = np.asarray(new_cpa, dtype=float)

    # Strategy-class diversity assertion (the headline guardrail).
    strategy_counts = Counter(type(a).__name__ for a in agents)
    distinct_classes = {k: v for k, v in strategy_counts.items()
                        if k != "PlayerAgentWrapper"}
    is_all_pid = list(distinct_classes.keys()) == ["PidBiddingStrategy"]
    if is_all_pid:
        raise SystemExit(
            f"FATAL: agent_strategy_counts is all PID (counts={strategy_counts}). "
            f"This indicates the official mixed-strategy pool was not loaded. "
            f"Check that the official repo path {str(_OFFICIAL_PATH)} is correct "
            f"and that no monkey-patching replaced the agents."
        )
    print(f"agent_strategy_counts: {dict(strategy_counts)}")

    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator
    budgets = [agent.budget for agent in agents]
    tracker = BiddingTracker(f"official_raw_tracker_{args.level}_{args.split}")

    started = time.time()
    episode_results = []
    total_bytes = 0

    for offset in range(args.num_episodes):
        episode = args.episode_start + offset
        out_path = output_dir / f"period-{episode}.csv"
        # Always rewrite to keep determinism.
        tracker.reset()
        ep_result = _run_episode(
            controller, tracker,
            episode=episode,
            num_tick=args.num_ticks,
            pv_generator=pv_generator,
            envs=envs,
            agents=agents,
            agents_cpa=agents_cpa,
            agents_category=agents_category,
            budgets=budgets,
        )
        # Write via the official generate_train_data helper so the schema is
        # guaranteed identical to official (18 cols in the documented order).
        tracker.generate_train_data(str(out_path))
        size = out_path.stat().st_size
        total_bytes += size
        episode_results.append({
            "episode": episode,
            "source_episode_id": episode,
            "path": str(out_path),
            "size_bytes": size,
            "elapsed_s": round(time.time() - started, 1),
            **ep_result,
        })
        print(f"[episode {offset + 1}/{args.num_episodes} src_id={episode}] wrote {out_path} "
              f"({size / 1024**3:.2f} GB) total={total_bytes / 1024**3:.2f} GB")

        if total_bytes >= args.min_total_bytes:
            print(f"Reached min_total_bytes={args.min_total_bytes} after {offset+1} episodes.")
            break

    # Write manifest.
    manifest = {
        "task": "generate_official_raw",
        "level": args.level,
        "split": args.split,
        "num_episodes": len(episode_results),
        "episode_start": args.episode_start,
        "episode_range": (
            [args.episode_start, args.episode_start + len(episode_results) - 1]
            if episode_results else None
        ),
        "num_ticks": args.num_ticks,
        "pv_num": args.pv_num,
        "official_repo_path": str(_OFFICIAL_PATH),
        "agent_strategy_counts": dict(strategy_counts),
        "is_all_pid": is_all_pid,
        "cpa_profile": {
            "level": args.level,
            "values": [float(x) for x in new_cpa],
            "min": float(min(new_cpa)),
            "max": float(max(new_cpa)),
            "is_rescaled": args.level != "low",
        },
        "schema": [
            "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex",
            "budget", "CPAConstraint", "timeStepIndex", "remainingBudget",
            "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot",
            "cost", "isExposed", "conversionAction", "leastWinningCost", "isEnd",
        ],
        "total_bytes": total_bytes,
        "output_dir": str(output_dir),
        "elapsed_s": round(time.time() - started, 1),
        "episodes": episode_results,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"MANIFEST: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())