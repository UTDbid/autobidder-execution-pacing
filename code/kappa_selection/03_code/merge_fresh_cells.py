#!/usr/bin/env python3
"""Merge 15 independently scheduled fresh-seed cells for one model/seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCES = ("traffic_aware", "pid", "dual")
WIDTH_LABELS = ("0", "0.3", "0.5", "0.8", "1")
EXPECTED = tuple(f"{reference}_k{width}" for reference in REFERENCES for width in WIDTH_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.root / "04_runs" / "fresh_stochastic" / args.model / f"seed_{args.seed}"
    payloads = []
    for config_id in EXPECTED:
        path = outdir / "split_parts" / f"{config_id}.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["model"] == args.model, path
        assert len(payload.get("results", [])) == 1, path
        assert payload["results"][0]["config"]["id"] == config_id, path
        payloads.append(payload)
    merged = dict(payloads[0])
    merged["command"] = ["merge_fresh_cells.py", "--model", args.model, "--seed", str(args.seed)]
    merged["total_seconds"] = sum(float(p.get("total_seconds", 0)) for p in payloads)
    merged["parallel_walltime_note"] = "15 cells scheduled independently; total_seconds is summed compute time"
    merged["results"] = [p["results"][0] for p in payloads]
    assert len(merged["results"]) == 15
    target = outdir / "results.json"
    temporary = outdir / "results.json.tmp"
    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
