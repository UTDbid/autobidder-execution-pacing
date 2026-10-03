"""Stateful adapters for GAS and SemBid inside the mandate replay.

These classes import canonical, read-only implementations and checkpoints. A
controller observes governed (executed) history, so its next raw proposal is a
closed-loop response to the mandate rather than an open-loop counterfactual.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


GAS_COMMIT = "abb0ba9a1dbe1a3d0e1eefb458dcd576a9d98e69"

# Release layout: bidder source trees live under <repo>/code/bidders/. The
# paths below resolve relative to this file so the release is relocatable;
# override with the GAS_REPO_ROOT env var to point at another tree.
_BIDDER_ROOT = Path(__file__).resolve().parents[2] / "bidders"


def safe_norm_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    span = float(array.max() - array.min()) if array.size else 0.0
    if span <= 1e-12:
        return np.zeros_like(array)
    return (array - array.min()) / span


class GasMandateController:
    def __init__(self, root: str | Path, *, action_num: int = 5, seed: int = 20260814) -> None:
        self.root = Path(root).resolve()
        self.action_num = int(action_num)
        self.seed = int(seed)
        config = json.loads((self.root / "pipeline_config.json").read_text(encoding="utf-8"))
        self.policy_method = str(config.get("policy_method", "vanilla_dt"))
        self.critic_method = str(config.get("critic_method", "dt_reweight_search_Q"))
        self.reweight_w = float(config.get("reweight_w", 0.2))
        self.policy_dir = self.root / "policy"
        self.critic_dirs = sorted(path for path in (self.root / "critics").glob("critic_*") if path.is_dir())
        if not (self.policy_dir / "dt.pt").exists() or not self.critic_dirs:
            raise FileNotFoundError(f"incomplete GAS root: {self.root}")
        gas_repo = Path(os.environ.get("GAS_REPO_ROOT", str(_BIDDER_ROOT / "gas"))).resolve()
        if (gas_repo / ".git").exists():
            actual = subprocess.run(
                ["git", "-C", str(gas_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
            if actual != GAS_COMMIT:
                raise RuntimeError(f"GAS commit mismatch: expected {GAS_COMMIT}, got {actual}")
        for name in [key for key in list(sys.modules) if key.startswith("bidding_train_env") or key == "run" or key.startswith("run.")]:
            del sys.modules[name]
        sys.path.insert(0, str(gas_repo))
        from bidding_train_env.strategy import PlayerBiddingCritic, PlayerBiddingStrategy

        self.strategy_class = PlayerBiddingStrategy
        self.critic_class = PlayerBiddingCritic
        self.agent = None
        self.critics: list[Any] = []
        self.rng = np.random.default_rng(self.seed)
        self.history: dict[str, list[np.ndarray]] = {}
        self.actual_executed_action: np.ndarray | None = None

    def reset(self, *, budget: float, cpa_limit: float, episode_key: Sequence[int], **_: Any) -> None:
        key_seed = sum((index + 1) * int(value) for index, value in enumerate(episode_key))
        self.rng = np.random.default_rng(self.seed + key_seed)
        self.agent = self.strategy_class(
            budget=float(budget), cpa=float(cpa_limit), load_dir=str(self.policy_dir),
            baseline_method=self.policy_method, reweight_w=self.reweight_w,
        )
        self.critics = [
            self.critic_class(
                budget=float(budget), cpa=float(cpa_limit), load_dir=str(path),
                baseline_method=self.critic_method, reweight_w=self.reweight_w,
            )
            for path in self.critic_dirs
        ]
        self.history = {"pvalue": [], "bids": [], "auction": [], "impression": [], "lwc": []}
        self.actual_executed_action = None

    def action(
        self,
        *,
        time_step: int,
        observed_values: np.ndarray,
        p_value_sigma: np.ndarray,
        **_: Any,
    ) -> float:
        if self.agent is None:
            raise RuntimeError("GAS controller not reset")
        _bids, alpha = self.agent.bidding(
            int(time_step), observed_values, p_value_sigma,
            self.history["pvalue"], self.history["bids"], self.history["auction"],
            self.history["impression"], self.history["lwc"], self.actual_executed_action,
        )
        multipliers = self.rng.uniform(0.9, 1.1, self.action_num).astype(np.float32)
        multipliers[-1] = 1.0
        proposals = (multipliers * float(np.asarray(alpha).reshape(-1)[0])).reshape(-1, 1)
        values = []
        for proposal in proposals:
            votes = []
            for critic in self.critics:
                value_1, value_2 = critic.access_value(
                    proposal, int(time_step), observed_values, p_value_sigma,
                    self.history["pvalue"], self.history["bids"], self.history["auction"],
                    self.history["impression"], self.history["lwc"],
                )
                votes.append(min(value_1[0, -1, 0].cpu().item(), value_2[0, -1, 0].cpu().item()))
            values.append(votes)
        normalized = np.asarray([safe_norm_rank(row) for row in np.asarray(values).T]).T.mean(axis=1)
        selected = int(np.argmax(normalized))
        return max(float(proposals[selected, 0]), 0.0)

    def observe(
        self,
        *,
        executed_action: float,
        executed_bids: np.ndarray,
        observed_values: np.ndarray,
        p_value_sigma: np.ndarray,
        least_winning_cost: np.ndarray,
        wins: np.ndarray,
        costs: np.ndarray,
        conversions: np.ndarray,
        remaining_budget: float,
        **_: Any,
    ) -> None:
        self.agent.remaining_budget = float(remaining_budget)
        self.actual_executed_action = np.asarray([float(executed_action)], dtype=np.float32)
        self.history["pvalue"].append(
            np.asarray([(observed_values[i], p_value_sigma[i]) for i in range(observed_values.size)])
        )
        self.history["bids"].append(np.asarray(executed_bids))
        self.history["lwc"].append(np.asarray(least_winning_cost))
        self.history["auction"].append(np.asarray([(wins[i], wins[i], costs[i]) for i in range(wins.size)]))
        self.history["impression"].append(
            np.asarray([(conversions[i], conversions[i]) for i in range(conversions.size)])
        )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SemBidMandateController:
    def __init__(
        self,
        model_dir: str | Path,
        embedding_lookup: str | Path,
        *,
        code_root: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        with Path(embedding_lookup).open("rb") as handle:
            self.embedding_lookup = pickle.load(handle)
        code_root = Path(code_root if code_root is not None else _BIDDER_ROOT / "sembid_cpa").resolve()
        algorithms = code_root / "Algorithms"
        for path in (algorithms, code_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        module = load_module(code_root / "Testing" / "test_exp23_standard.py", "sembid_exp23_standard_adapter")
        self.strategy_class = module.Exp23BiddingStrategy
        self.agent = None
        self.history: dict[str, list[Any]] = {}

    @staticmethod
    def episode_stats(episode: Mapping[str, Any]) -> dict[str, np.ndarray]:
        steps = episode["steps"]
        num_steps = len(steps)
        pvalue_mean = np.asarray([float(np.mean(step["p"])) if step["p"].size else 0.0 for step in steps], dtype=np.float32)
        lwc_mean = np.asarray([float(np.mean(step["lwc"])) if step["lwc"].size else 0.0 for step in steps], dtype=np.float32)
        volume = np.asarray([float(step["p"].size) for step in steps], dtype=np.float32)
        historical = np.zeros(num_steps, dtype=np.float32)
        last_three = np.zeros(num_steps, dtype=np.float32)
        for index in range(num_steps):
            historical[index] = float(np.sum(volume[:index]))
            last_three[index] = float(np.sum(volume[max(0, index - 3) : index]))
        return {
            "pvalue_mean": pvalue_mean, "lwc_mean": lwc_mean,
            "xi_mean": np.zeros(num_steps, dtype=np.float32), "volume": volume,
            "historical_volume": historical, "last3_volume": last_three,
        }

    def reset(
        self,
        *,
        budget: float,
        cpa_limit: float,
        category: int,
        episode: Mapping[str, Any],
        **_: Any,
    ) -> None:
        self.agent = self.strategy_class(
            model_dir=str(self.model_dir), budget=float(budget), cpa=float(cpa_limit), category=int(category),
            shared_encoder=None, language_emb_dim=2048, embedding_lookup=self.embedding_lookup, device=self.device,
        )
        self.agent.precomputed_stats = self.episode_stats(episode)
        self.history = {"bids": [], "auction": [], "impression": []}

    def action(
        self,
        *,
        time_step: int,
        observed_values: np.ndarray,
        p_value_sigma: np.ndarray,
        least_winning_cost: np.ndarray,
        **_: Any,
    ) -> float:
        if self.agent is None:
            raise RuntimeError("SemBid controller not reset")
        self.agent.bidding(
            int(time_step), observed_values, p_value_sigma, self.history["bids"],
            self.history["auction"], self.history["impression"], [least_winning_cost],
        )
        return max(float(self.agent.last_action), 0.0)

    def observe(
        self,
        *,
        executed_action: float,
        executed_bids: np.ndarray,
        wins: np.ndarray,
        costs: np.ndarray,
        conversions: np.ndarray,
        remaining_budget: float,
        **_: Any,
    ) -> None:
        self.agent.remaining_budget = float(remaining_budget)
        self.agent.last_reward = float(np.sum(conversions))
        self.agent.bid_mean_hist.append(float(np.mean(executed_bids)) if executed_bids.size else 0.0)
        self.agent.conv_mean_hist.append(float(np.mean(conversions)) if conversions.size else 0.0)
        if self.agent.actions:
            replacement = torch.tensor([float(executed_action)], device=self.agent.device, dtype=torch.float32)
            self.agent.actions[-1] = replacement
            self.agent.last_action = float(executed_action)
        self.history["bids"].append(np.asarray(executed_bids))
        self.history["auction"].append([(wins[i], wins[i], costs[i]) for i in range(wins.size)])
        self.history["impression"].append(
            [(conversions[i], conversions[i]) for i in range(conversions.size)]
        )

