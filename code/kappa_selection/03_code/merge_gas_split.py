#!/usr/bin/env python3
"""Merge independently scheduled GAS cells without changing scientific content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = (
    "traffic_aware_k0.3",
    "traffic_aware_k0.5",
    "traffic_aware_k1",
    "pid_k1",
    "dual_k1",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    outdir = args.root / "04_runs" / "retrospective" / "gas" / f"seed_{args.seed}"
    payloads = []
    for config_id in EXPECTED:
        path = outdir / "split_parts" / f"{config_id}.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert len(payload.get("results", [])) == 1, path
        assert payload["results"][0]["config"]["id"] == config_id, path
        payloads.append(payload)

    merged = dict(payloads[0])
    merged["command"] = ["merge_gas_split.py", "--seed", str(args.seed)]
    merged["total_seconds"] = sum(float(p.get("total_seconds", 0)) for p in payloads)
    merged["parallel_walltime_note"] = "five cells scheduled independently; total_seconds is summed compute time"
    merged["results"] = [p["results"][0] for p in payloads]
    assert [r["config"]["id"] for r in merged["results"]] == list(EXPECTED)
    assert all(r["num_advertisers"] == 48 for r in merged["results"])

    target = outdir / "results.json"
    temporary = outdir / "results.json.tmp"
    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
