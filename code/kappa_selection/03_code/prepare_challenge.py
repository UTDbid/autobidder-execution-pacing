#!/usr/bin/env python3
"""Freeze missing-cell grids, queues, and an auditable execution manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("cql", "iql", "dt", "gas", "sembid")
REFERENCES = ("traffic_aware", "pid", "dual")
RETRO_SEEDS = tuple(range(20260428, 20260436))
FRESH_SEEDS = tuple(range(2026090400, 2026090408))
MAIN_WIDTHS = (0.0, 0.3, 0.5, 0.8, 1.0)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config(reference: str, kappa: float) -> dict:
    row = {
        "id": f"{reference}_k{kappa:g}",
        "q_scale": 1.0,
        "kappa": kappa,
        "h": 4,
        "policy_mode": "hard",
        "reference_mode": reference,
        "autonomy_operator": "relative_hard",
        "budget_multiplier": 1.0,
        "cpa_multiplier": 1.0,
        "market_volatility": 0.0,
        "valuation_noise": 0.0,
        "valuation_bias": 0.0,
    }
    if reference == "pid":
        row.update(pid_kp=4.0, pid_ki=0.5, pid_kd=0.0, pid_integral_limit=1.0)
    elif reference == "dual":
        row.update(dual_eta=10.0, dual_log_shadow_limit=2.995732274)
    return row


def write_queues(name: str, seeds: tuple[int, ...], workers: int = 6) -> int:
    jobs = [(model, seed) for seed in seeds for model in MODELS]
    for worker in range(workers):
        rows = [
            f"{model}\t{seed}\n"
            for index, (model, seed) in enumerate(jobs)
            if index % workers == worker
        ]
        path = ROOT / "02_configs" / "queues" / name / f"worker_{worker}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(rows), encoding="utf-8")
    return len(jobs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Existing later-period coverage: traffic {0, .8}; PID/Dual {0, .3, .5, .8, 1.2}.
    # Only the following five cells per bidder/seed are genuinely missing from the main menu.
    retro_grid = [
        config("traffic_aware", 0.3),
        config("traffic_aware", 0.5),
        config("traffic_aware", 1.0),
        config("pid", 1.0),
        config("dual", 1.0),
    ]
    full_grid = [config(reference, width) for reference in REFERENCES for width in MAIN_WIDTHS]
    dump_json(ROOT / "02_configs" / "p8_retrospective_missing_grid.json", retro_grid)
    dump_json(ROOT / "02_configs" / "p8_fresh_stochastic_full_grid.json", full_grid)

    retro_jobs = write_queues("retrospective", RETRO_SEEDS)
    fresh_jobs = write_queues("fresh_stochastic", FRESH_SEEDS)
    inputs = sorted((ROOT / "01_protocol").glob("*.md")) + sorted(
        (ROOT / "02_configs").glob("*.yaml")
    ) + sorted((ROOT / "02_configs").glob("*.json"))
    manifest = {
        "status": "FROZEN_RETROSPECTIVE_SCREEN",
        "created": "2026-09-04",
        "selection_unit": "bidder_reference_pair",
        "models": MODELS,
        "references": REFERENCES,
        "main_widths": MAIN_WIDTHS,
        "retrospective": {
            "seeds": RETRO_SEEDS,
            "jobs": retro_jobs,
            "new_cells_per_job": len(retro_grid),
            "config_seed_evaluations": retro_jobs * len(retro_grid),
            "scientific_role": "consumed later-period screening only",
        },
        "fresh_stochastic": {
            "seeds": FRESH_SEEDS,
            "jobs": fresh_jobs,
            "cells_per_job": len(full_grid),
            "config_seed_evaluations": fresh_jobs * len(full_grid),
            "scientific_role": "conditional stochastic transport; not temporal OOS",
            "launch_status": "BLOCKED_UNTIL_RETROSPECTIVE_GO_AND_RULE_FREEZE",
        },
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    dump_json(ROOT / "08_audit" / "execution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
