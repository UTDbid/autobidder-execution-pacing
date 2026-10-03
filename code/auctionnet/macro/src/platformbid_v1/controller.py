"""Fast-controller adapter matching the frozen AuctionNet CQL semantics."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return NumpyCompatUnpickler(handle).load()


def normalize_states(states: np.ndarray, normalizer: dict) -> np.ndarray:
    """Apply exactly the normalization accepted by Gate A FastController."""

    output = np.asarray(states, dtype=np.float32).copy()
    if output.ndim == 1:
        output = output.reshape(1, -1)
    if "state_mean" in normalizer and "state_std" in normalizer:
        return (output - np.asarray(normalizer["state_mean"], dtype=np.float32)) / (
            np.asarray(normalizer["state_std"], dtype=np.float32) + 1e-8
        )
    for key, value in normalizer.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        low, high = float(value.get("min", 0.0)), float(value.get("max", 0.0))
        output[:, index] = (output[:, index] - low) / (high - low) if high > low else 0.0
    return output


class FastController:
    """Batched TorchScript CQL or explicit constant negative control.

    CQL loading, input normalization, nonnegative clipping, and output
    extraction deliberately mirror ``gatea_v2/src/mandate_eval.py``.  One
    immutable TorchScript policy is shared by all advertisers and market cells;
    policy state is supplied entirely through the 16-dimensional input.
    """

    def __init__(
        self,
        model: str,
        *,
        checkpoint: Path | None = None,
        normalizer: Path | None = None,
        device: str = "cpu",
        constant_action: float = 8.0,
    ) -> None:
        if model not in {"cql", "constant"}:
            raise ValueError("minimal PlatformBid evaluator supports cql or constant")
        self.model = model
        self.device = torch.device(device)
        self.constant_action = float(constant_action)
        self.policy = None
        self.normalizer: dict = {}
        if model == "cql":
            if checkpoint is None or normalizer is None:
                raise ValueError("CQL requires checkpoint and normalizer")
            self.normalizer = load_pickle(Path(normalizer))
            self.policy = torch.jit.load(str(checkpoint), map_location=self.device)
            self.policy.eval()

    def reset(self, n_trajectories: int) -> None:
        # TorchScript CQL is memoryless conditional on the provided state.
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be positive")

    def actions(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 16:
            raise ValueError("controller states must have shape [batch, 16]")
        if self.model == "constant":
            return np.full(states.shape[0], max(self.constant_action, 0.0), dtype=np.float64)
        normalized = normalize_states(states, self.normalizer)
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            output = self.policy(tensor)
        actions = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if actions.size != states.shape[0]:
            raise ValueError("TorchScript controller returned an unexpected batch shape")
        return np.maximum(actions, 0.0)
