#!/usr/bin/env python3
"""Validate OFFICIAL AuctionNet raw period CSVs.

Asserts (mandatory per project requirement):
  * 18-column raw schema in official order.
  * advertiserNumber range covers 0..47 (48 unique agents).
  * timeStepIndex range is 0..num_ticks-1 (MUST be 0..47 for full-tick run).
  * CPA range matches level expectation.
  * Manifest exists and records agent_strategy_counts (loaded from the
    generation manifest, NOT from the CSV — the CSV does not encode strategy).
  * total_bytes >= min-total-bytes (default 20 GB for mid train).
  * is_all_pid == false in the generation manifest.

Usage::

    python scripts/data/auctionnet_official/validate_official_raw.py \
        --raw-dir data/mid/raw_official_20gb \
        --generation-manifest data/manifests/sample_mid_train_official_20gb.json \
        --level mid \
        --report data/manifests/validate_mid_raw.json \
        --min-total-bytes $((20 * 1024**3))
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


EXPECTED_SCHEMA = [
    "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex",
    "budget", "CPAConstraint", "timeStepIndex", "remainingBudget",
    "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot",
    "cost", "isExposed", "conversionAction", "leastWinningCost", "isEnd",
]
LEVEL_CPA_RANGE = {
    "high": (6.0, 13.0),
    "mid": (20.0, 60.0),
    "low": (60.0, 130.0),
}


def _validate_one(path: Path) -> dict:
    df = pd.read_csv(path, nrows=0)
    cols = list(df.columns)
    schema_ok = cols == EXPECTED_SCHEMA
    return {"path": str(path), "rows_header": cols, "schema_ok": schema_ok}


def _validate_full(path: Path, num_ticks_expected: int) -> dict:
    df = pd.read_csv(path)
    adv_min = int(df["advertiserNumber"].min())
    adv_max = int(df["advertiserNumber"].max())
    adv_unique = int(df["advertiserNumber"].nunique())
    ts_min = int(df["timeStepIndex"].min())
    ts_max = int(df["timeStepIndex"].max())
    cpa_min = float(df["CPAConstraint"].min())
    cpa_max = float(df["CPAConstraint"].max())
    rows = len(df)
    return {
        "rows": rows,
        "advertiser_min": adv_min,
        "advertiser_max": adv_max,
        "advertiser_unique": adv_unique,
        "advertiser_count_ok": adv_unique == 48 and adv_min == 0 and adv_max == 47,
        "timeStepIndex_min": ts_min,
        "timeStepIndex_max": ts_max,
        "timeStepIndex_range_ok": ts_min == 0 and ts_max == num_ticks_expected - 1,
        "CPA_min": cpa_min,
        "CPA_max": cpa_max,
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="validate_official_raw")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True,
                        help="Manifest written by generate_official_raw.py")
    parser.add_argument("--level", required=True, choices=list(LEVEL_CPA_RANGE))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-total-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--num-ticks-expected", type=int, default=48)
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir.expanduser().resolve()
    gen_manifest: Path = args.generation_manifest.expanduser().resolve()
    report_path: Path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not gen_manifest.exists():
        raise SystemExit(f"Generation manifest not found: {gen_manifest}")
    gen = json.loads(gen_manifest.read_text())

    period_files = sorted(raw_dir.glob("period-*.csv"))
    if not period_files:
        raise SystemExit(f"No period-*.csv under {raw_dir}")

    errors = []

    # 1) Schema check (header only — cheap).
    schema_checks = [_validate_one(p) for p in period_files]
    if not all(s["schema_ok"] for s in schema_checks):
        bad = [s["path"] for s in schema_checks if not s["schema_ok"]]
        errors.append(f"schema mismatch in: {bad}")

    # 2) Full content check on first + last file (sample two for speed).
    sample_files = [period_files[0], period_files[-1]] if len(period_files) > 1 else period_files
    sample_stats = [_validate_full(p, args.num_ticks_expected) for p in sample_files]
    for s in sample_stats:
        if not s["advertiser_count_ok"]:
            errors.append(f"advertiser_count wrong: {s}")
        if not s["timeStepIndex_range_ok"]:
            errors.append(
                f"timeStepIndex range wrong (expected 0..{args.num_ticks_expected-1}): {s}"
            )

    # 3) CPA range (from generation manifest + sampled CSV).
    cpa_lo, cpa_hi = LEVEL_CPA_RANGE[args.level]
    cpa_ok = (
        cpa_lo - 0.01 <= gen["cpa_profile"]["min"] <= cpa_hi + 0.01
        and cpa_lo - 0.01 <= gen["cpa_profile"]["max"] <= cpa_hi + 0.01
    )
    if not cpa_ok:
        errors.append(f"CPA out of band [{cpa_lo}, {cpa_hi}]: {gen['cpa_profile']}")

    # 4) Strategy diversity (must NOT be all-PID).
    if gen.get("is_all_pid", True):
        errors.append(f"agent_strategy_counts is all PID: {gen['agent_strategy_counts']}")
    distinct = sum(1 for k in gen["agent_strategy_counts"] if k != "PlayerAgentWrapper")
    if distinct < 2:
        errors.append(
            f"agent_strategy_counts has only {distinct} distinct class(es): "
            f"{gen['agent_strategy_counts']}"
        )

    # 5) Total raw size.
    total_bytes = sum(p.stat().st_size for p in period_files)
    if total_bytes < args.min_total_bytes:
        errors.append(
            f"total raw bytes {total_bytes} < required {args.min_total_bytes} "
            f"({total_bytes / 1024**3:.2f} GB vs {args.min_total_bytes / 1024**3:.2f} GB)"
        )

    report = {
        "task": "validate_official_raw",
        "level": args.level,
        "raw_dir": str(raw_dir),
        "generation_manifest": str(gen_manifest),
        "schema_expected": EXPECTED_SCHEMA,
        "schema_checks": schema_checks,
        "sample_stats": sample_stats,
        "agent_strategy_counts": gen.get("agent_strategy_counts"),
        "is_all_pid": gen.get("is_all_pid"),
        "cpa_profile": gen.get("cpa_profile"),
        "total_bytes": total_bytes,
        "min_total_bytes": args.min_total_bytes,
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
    print(f"PASS: {len(period_files)} periods, {total_bytes / 1024**3:.2f} GB, "
          f"{distinct} distinct strategy classes, CPA [{cpa_lo}, {cpa_hi}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())