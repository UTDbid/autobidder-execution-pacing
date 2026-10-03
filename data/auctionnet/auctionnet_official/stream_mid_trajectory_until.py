#!/usr/bin/env python3
"""Resume-safe, multi-worker streaming generator for mid trajectory shards.

For each episode this script:
  1. Runs ONE official AuctionNet episode (Controller mixed-strategy pool,
     mid CPA profile [20, 60]) to produce a raw 18-col CSV.
  2. Converts the raw CSV to a 14-col trajectory shard via
     ``TrainDataGenerator._generate_train_data``.
  3. Writes the trajectory shard atomically to ``--output-dir``.
  4. Updates a per-worker JSON state file atomically.
  5. Deletes the giant raw CSV (we only keep trajectory shards).

Multi-worker partitioning is by ``(episode - episode_start) % num_workers``.
Each worker stops when its share of the global ``--target-bytes`` is reached
(``target_bytes / num_workers``), OR ``--max-episodes`` is hit.

On startup the script reconciles the output dir with disk: it validates each
existing shard with :func:`is_valid_trajectory_shard`. Invalid shards (e.g.
shards left behind by an aborted previous run that crashed mid-write) are
deleted. Valid shards owned by this worker count toward
``accumulated_bytes``. Valid shards owned by other workers are SKIPPED
(legitimate resume case).

Benchmark mode (``--benchmark-only``) runs a few episodes, captures per-step
timings + sizes, computes ETAs for 100 MB / 1 GB targets at 1 / 4 workers,
and writes a benchmark manifest — without touching the regular state file.

Usage (regular run, 4 workers, Phase 1 100 MB target)::

    PYTHONPATH=external/auctionnet_official_<ts>:external/auctionnet_official_<ts>/strategy_train_env:src \\
    python scripts/data/auctionnet_official/stream_mid_trajectory_until.py \\
        --target-bytes 104857600 \\
        --num-workers 4 \\
        --worker-id 0 \\
        --output-dir data/mid/train_official_streaming

Usage (benchmark)::

    PYTHONPATH=... python scripts/data/auctionnet_official/stream_mid_trajectory_until.py \\
        --benchmark-only --benchmark-episodes 2 \\
        --benchmark-manifest data/manifests/stream_mid_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Path bootstrap — same convention as generate_official_raw.py
# ---------------------------------------------------------------------------
# parents[0]=scripts/data/auctionnet_official
# parents[1]=scripts/data
# parents[2]=scripts
# parents[3]=<repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_PATH = Path(os.environ.get(
    "AUCTIONNET_OFFICIAL_PATH",
    str(_REPO_ROOT / "external" / "auctionnet_official_20260618_091733"),
))
if not _OFFICIAL_PATH.exists():
    raise SystemExit(f"Official repo not found: {_OFFICIAL_PATH}")
if str(_OFFICIAL_PATH) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_PATH))
_STRATEGY_TRAIN_ENV = _OFFICIAL_PATH / "strategy_train_env"
if _STRATEGY_TRAIN_ENV.exists() and str(_STRATEGY_TRAIN_ENV) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_TRAIN_ENV))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

# Official imports.
from simul_bidding_env.Controller.Controller import Controller  # noqa: E402
from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker  # noqa: E402
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy  # noqa: E402

from bidding_train_env.train_data_generator.train_data_generator import (  # noqa: E402
    TrainDataGenerator,
)

# Project-side validator (added by sub-agent 1).
from common.baseline_registry import is_valid_trajectory_shard  # noqa: E402


# ---------------------------------------------------------------------------
# CPA profile + episode runner — copied verbatim from generate_official_raw.py
# (kept inline so this script is self-contained and doesn't depend on the
# sibling script being importable as a module).
# ---------------------------------------------------------------------------

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
    "high": (6.0, 13.0),
    "mid": (20.0, 60.0),
    "low": (60.0, 130.0),
}


def _rescale_cpa_list(level: str) -> list[float]:
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
    new_cpa = _rescale_cpa_list(level)
    controller.cpa_constraint_list = np.asarray(new_cpa, dtype=float)
    for i, agent in enumerate(controller.agents):
        agent.cpa = float(new_cpa[i])
    return new_cpa


def _apply_pvalue_scale(controller: Controller, scale: float, log: logging.Logger | None = None) -> None:
    """Scale generated pValue and pValueSigma arrays in-place.

    The official NeurIPSPvGen produces ``pv_values`` and ``pValueSigmas``
    during reset. Mid adjustment keeps the official shape/temporal/category
    profile but raises expected conversion probability.
    """
    if scale <= 0:
        raise ValueError(f"pvalue scale must be > 0, got {scale}")
    if abs(scale - 1.0) < 1e-12:
        return
    pv_generator = controller.pvGenerator
    pv_generator.pv_values = [np.clip(arr * scale, 0.0, 1.0) for arr in pv_generator.pv_values]
    pv_generator.pValueSigmas = [np.clip(arr * scale, 0.0, 1.0) for arr in pv_generator.pValueSigmas]
    if log is not None:
        first_mean = float(np.mean(pv_generator.pv_values[0])) if pv_generator.pv_values else 0.0
        first_sigma_mean = float(np.mean(pv_generator.pValueSigmas[0])) if pv_generator.pValueSigmas else 0.0
        log.info(
            "PVALUE_SCALE applied scale=%.6g first_tick_mean_pvalue=%.8g first_tick_mean_sigma=%.8g",
            scale,
            first_mean,
            first_sigma_mean,
        )


def _adjust_over_cost(bids, over_cost_ratio, envs_slots, winner_pit):
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
    tracker.reset()

    rewards = [0.0] * len(agents)
    costs = [0.0] * len(agents)
    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    controller.reset(episode=episode)
    pvalue_scale = getattr(controller, "_sembid_pvalue_scale", 1.0)
    if abs(float(pvalue_scale) - 1.0) >= 1e-12:
        _apply_pvalue_scale(controller, float(pvalue_scale), None)

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

        ratio_max = None
        winner_pit = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                over_cost_ratio = np.maximum(
                    (cost - remaining_budget_list) / (cost + 1e-4), 0
                )
                _adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)
            (
                xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit,
                least_winning_cost_pit, _,
            ) = envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
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


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_TRAJ_SHARD_RE = re.compile(r"^trajectory_data__period-(\d+)\.csv$")


@dataclass
class WorkerState:
    worker_id: int
    num_workers: int
    episode_start: int
    target_bytes: int
    completed_episodes: list[int] = field(default_factory=list)
    accumulated_bytes: int = 0
    next_episode: int = 0
    started_at_iso: str = ""
    updated_at_iso: str = ""
    profile_name: str = ""
    pvalue_scale: float = 1.0
    agent_strategy_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _save_state(state_file: Path, state: WorkerState) -> None:
    state.updated_at_iso = _now_iso()
    _write_json_atomic(state_file, state.to_json())


def _parse_episode_from_shard(name: str) -> int | None:
    m = _TRAJ_SHARD_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _reconcile_output_dir(
    *,
    output_dir: Path,
    state: WorkerState,
    log: logging.Logger,
) -> None:
    """Scan output_dir for existing shards, validate, classify by worker.

    Mutates ``state.completed_episodes`` / ``state.accumulated_bytes`` /
    ``state.next_episode`` in place.
    """
    if not output_dir.exists():
        return
    completed: set[int] = set()
    acc_bytes = 0
    shards = sorted(output_dir.glob("trajectory_data__period-*.csv"))
    for shard in shards:
        ep = _parse_episode_from_shard(shard.name)
        if ep is None:
            continue
        ok, reason = is_valid_trajectory_shard(shard)
        if not ok:
            log.warning("RECONCILE: deleting invalid shard %s (%s)", shard.name, reason)
            try:
                shard.unlink()
            except FileNotFoundError:
                pass
            continue
        # Classify by worker.
        if (ep - state.episode_start) % state.num_workers == state.worker_id:
            if ep not in completed:
                completed.add(ep)
                acc_bytes += shard.stat().st_size
                log.info("RECONCILE: own shard ep=%d size=%d bytes",
                         ep, shard.stat().st_size)
        else:
            log.info("RECONCILE: skipping shard ep=%d owned by worker %d",
                     ep, (ep - state.episode_start) % state.num_workers)
    state.completed_episodes = sorted(completed)
    state.accumulated_bytes = acc_bytes
    # Compute next_episode (first uncompleted modulo-class episode >=
    # episode_start + worker_id).
    if completed:
        cursor = max(completed) + state.num_workers
    else:
        cursor = state.episode_start + state.worker_id
    state.next_episode = cursor


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stream_mid_trajectory")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if invoked twice in same process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Controller setup
# ---------------------------------------------------------------------------

def _build_controller(
    *,
    level: str,
    num_ticks: int,
    pv_num: int,
    player_index: int,
    pvalue_scale: float,
    profile_name: str,
    log: logging.Logger,
) -> tuple[Controller, list[float], np.ndarray, np.ndarray, BiddingTracker, dict[str, int]]:
    player_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    controller = Controller(
        player_index=player_index,
        player_agent=player_agent,
        num_tick=num_ticks,
        num_agent_category=8,
        num_category=6,
        pv_num=pv_num,
        pv_generator_type="neuripsPvGen",
    )
    agents = controller.agents
    agents_category = np.array([agent.category for agent in agents])
    new_cpa = _install_cpa_profile(controller, level)
    _apply_pvalue_scale(controller, pvalue_scale, log)
    controller._sembid_pvalue_scale = float(pvalue_scale)
    controller._sembid_profile_name = profile_name
    log.info(
        "conversion_profile level=%s profile_name=%s pvalue_scale=%.6g cpa_min=%.4f cpa_max=%.4f",
        level,
        profile_name,
        pvalue_scale,
        float(np.min(new_cpa)),
        float(np.max(new_cpa)),
    )
    agents_cpa = np.asarray(new_cpa, dtype=float)
    budgets = [agent.budget for agent in agents]

    strategy_counts = Counter(type(a).__name__ for a in agents)
    distinct = {k: v for k, v in strategy_counts.items() if k != "PlayerAgentWrapper"}
    is_all_pid = list(distinct.keys()) == ["PidBiddingStrategy"]
    if is_all_pid:
        raise SystemExit(
            f"FATAL: agent_strategy_counts is all PID (counts={dict(strategy_counts)}). "
            f"This indicates the official mixed-strategy pool was not loaded. "
            f"Check that the official repo path {str(_OFFICIAL_PATH)} is correct."
        )
    log.info("agent_strategy_counts: %s", dict(strategy_counts))

    tracker = BiddingTracker(f"stream_mid_tracker_w{player_index}")
    return controller, budgets, agents_cpa, agents_category, tracker, dict(strategy_counts)


# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------

def _process_one_episode(
    *,
    episode: int,
    controller: Controller,
    tracker: BiddingTracker,
    pv_generator,
    envs,
    agents,
    agents_cpa,
    agents_category,
    budgets,
    num_ticks: int,
    tmp_dir: Path,
    output_dir: Path,
    keep_raw_dir: Path | None,
    log: logging.Logger,
) -> dict[str, Any]:
    """Run + convert + atomic-write one episode shard. Returns timing dict."""
    raw_csv = tmp_dir / f"period-{episode}.csv"
    if raw_csv.exists():
        raw_csv.unlink()

    t0 = time.time()
    _run_episode(
        controller, tracker,
        episode=episode,
        num_tick=num_ticks,
        pv_generator=pv_generator,
        envs=envs,
        agents=agents,
        agents_cpa=agents_cpa,
        agents_category=agents_category,
        budgets=budgets,
    )
    tracker.generate_train_data(str(raw_csv))
    t_raw = time.time()

    raw_size = raw_csv.stat().st_size
    df_raw = pd.read_csv(raw_csv)
    traj_df = TrainDataGenerator()._generate_train_data(df_raw)
    t_convert = time.time()

    out_path = output_dir / f"trajectory_data__period-{episode}.csv"
    tmp_path = output_dir / f"trajectory_data__period-{episode}.csv.tmp"
    traj_df.to_csv(tmp_path, index=False)
    with open(tmp_path, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)
    t_write = time.time()

    # Validate the freshly-written shard before claiming it.
    ok, reason = is_valid_trajectory_shard(out_path)
    if not ok:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"freshly-written shard failed validation: {reason}")

    traj_size = out_path.stat().st_size
    if keep_raw_dir is not None:
        keep_raw_dir.mkdir(parents=True, exist_ok=True)
        keep_path = keep_raw_dir / raw_csv.name
        os.replace(raw_csv, keep_path)
        raw_csv = keep_path
    else:
        try:
            raw_csv.unlink()
        except FileNotFoundError:
            pass

    log.info(
        "EPISODE ep=%d raw=%.1fs convert=%.1fs write=%.1fs total=%.1fs "
        "raw_bytes=%d traj_bytes=%d",
        episode, t_raw - t0, t_convert - t_raw, t_write - t_convert,
        t_write - t0, raw_size, traj_size,
    )
    return {
        "episode": episode,
        "raw_seconds": round(t_raw - t0, 3),
        "convert_seconds": round(t_convert - t_raw, 3),
        "write_seconds": round(t_write - t_convert, 3),
        "total_seconds": round(t_write - t0, 3),
        "raw_size_bytes": raw_size,
        "traj_size_bytes": traj_size,
    }


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def _run_benchmark(
    args,
    *,
    log: logging.Logger,
) -> int:
    output_dir: Path = args.output_dir.expanduser().resolve()
    tmp_dir: Path = args.tmp_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = args.benchmark_manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "BENCHMARK MODE: episodes=%d output_dir=%s tmp_dir=%s manifest=%s",
        args.benchmark_episodes, output_dir, tmp_dir, manifest_path,
    )

    controller, budgets, agents_cpa, agents_category, tracker, strategy_counts = (
        _build_controller(
            level="mid",
            num_ticks=args.num_ticks,
            pv_num=args.pv_num,
            player_index=0,
            pvalue_scale=args.pvalue_scale,
            profile_name=args.profile_name,
            log=log,
        )
    )
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator
    agents = controller.agents

    episodes_info: list[dict[str, Any]] = []
    for i in range(args.benchmark_episodes):
        # Walk strided like a worker would (worker_id=0, num_workers stride).
        ep = args.episode_start + args.worker_id + i * args.num_workers
        info = _process_one_episode(
            episode=ep,
            controller=controller,
            tracker=tracker,
            pv_generator=pv_generator,
            envs=envs,
            agents=agents,
            agents_cpa=agents_cpa,
            agents_category=agents_category,
            budgets=budgets,
            num_ticks=args.num_ticks,
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            keep_raw_dir=args.keep_raw_dir,
            log=log,
        )
        episodes_info.append(info)

    if not episodes_info:
        raise SystemExit("benchmark produced no episodes")

    avg_bytes = sum(e["traj_size_bytes"] for e in episodes_info) / len(episodes_info)
    avg_seconds = sum(e["total_seconds"] for e in episodes_info) / len(episodes_info)

    def _eta_hours(target_bytes: int, workers: int) -> float:
        if avg_bytes <= 0:
            return float("inf")
        episodes_needed = target_bytes / avg_bytes
        per_worker_episodes = episodes_needed / workers
        return per_worker_episodes * avg_seconds / 3600.0

    target_100mb = 100 * 1024 ** 2
    target_1gb = 1 * 1024 ** 3

    manifest = {
        "task": "stream_mid_benchmark",
        "started_at_iso": _now_iso(),
        "num_ticks": args.num_ticks,
        "pv_num": args.pv_num,
        "episode_start": args.episode_start,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "profile_name": args.profile_name,
        "pvalue_scale": args.pvalue_scale,
        "agent_strategy_counts": strategy_counts,
        "episodes": episodes_info,
        "avg_traj_bytes_per_episode": round(avg_bytes, 2),
        "avg_seconds_per_episode": round(avg_seconds, 3),
        "eta_100mb_serial_hours": round(_eta_hours(target_100mb, 1), 4),
        "eta_100mb_4worker_hours": round(_eta_hours(target_100mb, 4), 4),
        "eta_1gb_serial_hours": round(_eta_hours(target_1gb, 1), 4),
        "eta_1gb_4worker_hours": round(_eta_hours(target_1gb, 4), 4),
        "output_dir": str(output_dir),
    }
    _write_json_atomic(manifest_path, manifest)
    log.info("BENCHMARK MANIFEST: %s", manifest_path)
    return 0


# ---------------------------------------------------------------------------
# Regular streaming loop
# ---------------------------------------------------------------------------

def _run_streaming(args, *, log: logging.Logger) -> int:
    output_dir: Path = args.output_dir.expanduser().resolve()
    tmp_dir: Path = args.tmp_dir.expanduser().resolve()
    state_file: Path = args.state_file.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    per_worker_target = args.target_bytes // max(args.num_workers, 1)
    log.info(
        "STREAM START worker_id=%d num_workers=%d target_bytes=%d per_worker=%d "
        "episode_start=%d output_dir=%s",
        args.worker_id, args.num_workers, args.target_bytes, per_worker_target,
        args.episode_start, output_dir,
    )

    # Build controller first so we can stamp strategy_counts into state.
    controller, budgets, agents_cpa, agents_category, tracker, strategy_counts = (
        _build_controller(
            level="mid",
            num_ticks=args.num_ticks,
            pv_num=args.pv_num,
            player_index=0,
            pvalue_scale=args.pvalue_scale,
            profile_name=args.profile_name,
            log=log,
        )
    )
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator
    agents = controller.agents

    state = WorkerState(
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        episode_start=args.episode_start,
        target_bytes=per_worker_target,
        profile_name=args.profile_name,
        pvalue_scale=float(args.pvalue_scale),
        started_at_iso=_now_iso(),
        agent_strategy_counts=strategy_counts,
    )
    state.next_episode = args.episode_start + args.worker_id
    _reconcile_output_dir(output_dir=output_dir, state=state, log=log)
    _save_state(state_file, state)
    log.info(
        "RECONCILED state: completed=%d acc_bytes=%d next_ep=%d",
        len(state.completed_episodes), state.accumulated_bytes, state.next_episode,
    )

    episodes_processed_this_run = 0
    while True:
        if state.accumulated_bytes >= per_worker_target:
            log.info(
                "STOP: per-worker target reached acc_bytes=%d target=%d",
                state.accumulated_bytes, per_worker_target,
            )
            break
        if episodes_processed_this_run >= args.max_episodes:
            log.info(
                "STOP: max_episodes reached (%d this run)",
                episodes_processed_this_run,
            )
            break

        ep = state.next_episode
        info = _process_one_episode(
            episode=ep,
            controller=controller,
            tracker=tracker,
            pv_generator=pv_generator,
            envs=envs,
            agents=agents,
            agents_cpa=agents_cpa,
            agents_category=agents_category,
            budgets=budgets,
            num_ticks=args.num_ticks,
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            keep_raw_dir=args.keep_raw_dir,
            log=log,
        )
        state.completed_episodes.append(ep)
        state.accumulated_bytes += info["traj_size_bytes"]
        state.next_episode = ep + args.num_workers
        _save_state(state_file, state)
        episodes_processed_this_run += 1
        log.info(
            "PROGRESS acc_bytes=%d / %d (%.1f%%) episodes_this_run=%d",
            state.accumulated_bytes, per_worker_target,
            100.0 * state.accumulated_bytes / max(per_worker_target, 1),
            episodes_processed_this_run,
        )

    log.info(
        "DONE worker_id=%d episodes_total=%d acc_bytes=%d",
        args.worker_id, len(state.completed_episodes), state.accumulated_bytes,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_per_worker_path(template: str, worker_id: int) -> Path:
    return Path(template.replace("<id>", str(worker_id)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stream_mid_trajectory_until",
        description=(
            "Resume-safe streaming generator for mid trajectory shards. "
            "Runs official AuctionNet episodes, converts each to a 14-col "
            "trajectory shard, writes atomically, deletes the raw."
        ),
    )
    parser.add_argument("--target-bytes", type=int, default=100 * 1024 ** 2,
                        help="Global target trajectory size in bytes; each "
                             "worker stops at target_bytes / num_workers "
                             "(default: 100MB)")
    parser.add_argument("--num-ticks", type=int, default=48,
                        help="Ticks per episode (default: 48)")
    parser.add_argument("--pv-num", type=int, default=500_000,
                        help="PVs per tick (default: 500000)")
    parser.add_argument("--pvalue-scale", type=float, default=1.0,
                        help="Scale generated pValue and pValueSigma arrays after each official reset; default 1.0")
    parser.add_argument("--profile-name", type=str, default=None,
                        help="Profile label written to logs/manifests; default mid_pvscale_<scale>")
    parser.add_argument("--keep-raw-dir", type=Path, default=None,
                        help="If set, keep a copy of each raw period CSV in this directory")
    parser.add_argument("--episode-start", type=int, default=200_000,
                        help="Starting source episode id (default: 200000 to "
                             "avoid colliding with train/test 0..99999)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="0..num_workers-1 (default: 0)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total worker count (default: 1)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/mid/train_official_streaming"),
                        help="Shared output dir for trajectory shards")
    parser.add_argument("--tmp-dir", type=Path, default=None,
                        help="Per-worker tmp dir (default: "
                             "data/_tmp/mid_streaming_worker_<id>)")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Per-worker state JSON (default: "
                             "data/manifests/stream_mid_worker_<id>.state.json)")
    parser.add_argument("--log", type=Path, default=None,
                        help="Per-worker log file (default: "
                             "data/logs/mid_streaming_worker_<id>.log)")
    parser.add_argument("--max-episodes", type=int, default=2000,
                        help="Hard cap on episodes processed THIS RUN")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run a small benchmark and exit (does not touch "
                             "state file)")
    parser.add_argument("--benchmark-episodes", type=int, default=2,
                        help="Episodes to time in benchmark mode (default: 2)")
    parser.add_argument("--benchmark-manifest", type=Path,
                        default=Path("data/manifests/stream_mid_benchmark.json"),
                        help="Benchmark manifest output path")
    return parser


def _resolve_per_worker_defaults(args: argparse.Namespace) -> None:
    if args.tmp_dir is None:
        args.tmp_dir = _default_per_worker_path(
            "data/_tmp/mid_streaming_worker_<id>", args.worker_id,
        )
    if args.state_file is None:
        args.state_file = _default_per_worker_path(
            "data/manifests/stream_mid_worker_<id>.state.json", args.worker_id,
        )
    if args.log is None:
        args.log = _default_per_worker_path(
            "data/logs/mid_streaming_worker_<id>.log", args.worker_id,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        raise SystemExit(
            f"--worker-id must be in [0, num_workers), got {args.worker_id}/{args.num_workers}"
        )

    _resolve_per_worker_defaults(args)
    if args.pvalue_scale <= 0:
        raise SystemExit(f"--pvalue-scale must be > 0, got {args.pvalue_scale}")
    if args.profile_name is None:
        scale_label = str(args.pvalue_scale).replace(".", "p")
        args.profile_name = f"mid_pvscale_{scale_label}"
    log = _setup_logging(args.log.expanduser().resolve())

    log.info("ARGS: %s", vars(args))

    if args.benchmark_only:
        return _run_benchmark(args, log=log)
    return _run_streaming(args, log=log)


if __name__ == "__main__":
    raise SystemExit(main())
