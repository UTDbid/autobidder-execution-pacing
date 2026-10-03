#!/usr/bin/env python3
"""Validate OFFICIAL AuctionNet trajectory CSVs (14-col RL schema).

Asserts:
  * 14-column trajectory schema in official order.
  * advertiserNumber range covers 0..47 (48 unique agents).
  * timeStepIndex range is 0..num_ticks-1.
  * CPA range matches level expectation.
  * At least one shard per period file in the source manifest.

Usage::

    python scripts/data/auctionnet_official/validate_official_trajectory.py \
        --trajectory-dir data/mid/train_official_20gb \
        --source-manifest data/manifests/convert_mid_official_20gb.json \
        --level mid \
        --report data/manifests/validate_mid_trajectory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SCHEMA = [
    "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex",
    "budget", "CPAConstraint", "realAllCost", "realAllConversion",
    "timeStepIndex", "state", "action", "reward", "reward_continuous",
    "done", "next_state",
]
LEVEL_CPA_RANGE = {
    "high": (6.0, 13.0),
    "mid": (20.0, 60.0),
    "low": (60.0, 130.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="validate_official_trajectory")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Manifest written by convert_official_raw_to_trajectory.py")
    parser.add_argument("--level", required=True, choices=list(LEVEL_CPA_RANGE))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-ticks-expected", type=int, default=48)
    args = parser.parse_args()

    traj_dir: Path = args.trajectory_dir.expanduser().resolve()
    src_manifest: Path = args.source_manifest.expanduser().resolve()
    report_path: Path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    src = json.loads(src_manifest.read_text())
    shard_files = sorted(traj_dir.glob("trajectory_data__period-*.csv"))
    if not shard_files:
        raise SystemExit(f"No trajectory shards under {traj_dir}")

    errors = []
    cpa_lo, cpa_hi = LEVEL_CPA_RANGE[args.level]
    per_shard = []
    for sf in shard_files:
        df = pd.read_csv(sf, nrows=0)
        cols = list(df.columns)
        if cols != EXPECTED_SCHEMA:
            errors.append(f"{sf.name}: schema mismatch ({cols})")
            continue
        # Sample full file (each shard is bounded by single-period rows).
        full = pd.read_csv(sf)
        adv_unique = int(full["advertiserNumber"].nunique())
        ts_min = int(full["timeStepIndex"].min())
        ts_max = int(full["timeStepIndex"].max())
        cpa_min = float(full["CPAConstraint"].min())
        cpa_max = float(full["CPAConstraint"].max())
        if adv_unique != 48:
            errors.append(f"{sf.name}: advertiser unique={adv_unique} (expected 48)")
        if not (ts_min == 0 and ts_max == args.num_ticks_expected - 1):
            errors.append(
                f"{sf.name}: timeStepIndex range {ts_min}..{ts_max} "
                f"(expected 0..{args.num_ticks_expected - 1})"
            )
        if not (cpa_lo - 0.01 <= cpa_min and cpa_max <= cpa_hi + 0.01):
            errors.append(
                f"{sf.name}: CPA [{cpa_min}, {cpa_max}] out of [{cpa_lo}, {cpa_hi}]"
            )
        per_shard.append({
            "path": str(sf),
            "rows": len(full),
            "advertiser_unique": adv_unique,
            "timeStepIndex_range": [ts_min, ts_max],
            "CPA_range": [cpa_min, cpa_max],
            "size_bytes": sf.stat().st_size,
        })

    report = {
        "task": "validate_official_trajectory",
        "level": args.level,
        "trajectory_dir": str(traj_dir),
        "source_manifest": str(src_manifest),
        "schema_expected": EXPECTED_SCHEMA,
        "num_shards": len(shard_files),
        "per_shard": per_shard,
        "total_trajectory_rows": src.get("total_trajectory_rows"),
        "pass": not errors,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"REPORT: {report_path}")
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    total_rows = sum(p["rows"] for p in per_shard)
    print(f"PASS: {len(shard_files)} shards, {total_rows:,} rows total, "
          f"CPA [{cpa_lo}, {cpa_hi}], timeStepIndex 0..{args.num_ticks_expected - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())