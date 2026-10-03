#!/usr/bin/env python3
"""Gate M1: advertiser-level structural calibration report for mid candidates.

Measures the canonical high/low endpoint raw test files and every candidate's
retained raw episodes, computes per-advertiser opportunity index
(``mean_pvalue * CPAConstraint``) distributions, evaluates the five M1 gate
conditions from the recalibration design spec, and selects the candidate
closest to the geometric endpoint target. Score is not used to break ties
unless structural distances are effectively equal.

Outputs a JSON manifest (``--output``) and a human-readable summary on stdout.

Usage::

    python scripts/data/auctionnet_official/mid_recalibration_m1_report.py \\
        --run-id 20260715 \\
        --output data/manifests/mid_recalibration_m1_selection_20260715.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AOFF = _REPO_ROOT / "scripts" / "data" / "auctionnet_official"
if str(_AOFF) not in sys.path:
    sys.path.insert(0, str(_AOFF))

from measure_conversion_profile import _series_stats  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aggregate_raw(paths: list[Path], chunksize: int) -> dict[str, Any]:
    """Aggregate per-advertiser stats across one or more raw 18-col CSV files."""
    total_rows = 0
    total_pvalue = 0.0
    total_conv = 0.0
    total_exposed = 0.0
    per_adv_rows: dict[int, int] = {}
    per_adv_pvalue: dict[int, float] = {}
    per_adv_conv: dict[int, float] = {}
    per_adv_cpa: dict[int, float] = {}
    per_adv_budget: dict[int, float] = {}
    timesteps: set[int] = set()
    schema: list[str] | None = None

    for path in paths:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            rows = len(chunk)
            total_rows += rows
            if schema is None:
                schema = list(chunk.columns)
            conv = chunk.get("conversionAction", pd.Series(np.zeros(rows))).astype(float)
            exposed = chunk.get("isExposed", pd.Series(np.zeros(rows))).astype(float)
            pvalue = chunk.get("pValue", pd.Series(np.zeros(rows))).astype(float)
            total_conv += float(conv.sum())
            total_exposed += float(exposed.sum())
            total_pvalue += float(pvalue.sum())
            if "timeStepIndex" in chunk.columns:
                timesteps.update(int(x) for x in chunk["timeStepIndex"].unique())
            if "advertiserNumber" in chunk.columns:
                adv = chunk["advertiserNumber"].astype(int)
                g = pd.DataFrame({"adv": adv, "conv": conv, "pvalue": pvalue}).groupby("adv")
                for k, v in g.size().items():
                    per_adv_rows[int(k)] = per_adv_rows.get(int(k), 0) + int(v)
                for k, v in g["conv"].sum().items():
                    per_adv_conv[int(k)] = per_adv_conv.get(int(k), 0.0) + float(v)
                for k, v in g["pvalue"].sum().items():
                    per_adv_pvalue[int(k)] = per_adv_pvalue.get(int(k), 0.0) + float(v)
                if "CPAConstraint" in chunk.columns:
                    for k, v in chunk.groupby(adv)["CPAConstraint"].first().items():
                        per_adv_cpa[int(k)] = float(v)
                if "budget" in chunk.columns:
                    for k, v in chunk.groupby(adv)["budget"].first().items():
                        per_adv_budget[int(k)] = float(v)

    opp_index: list[float] = []
    bud_over_cpa: list[float] = []
    per_adv_opp: dict[int, float] = {}
    for adv, n in sorted(per_adv_rows.items()):
        if n <= 0:
            continue
        mean_pv = per_adv_pvalue.get(adv, 0.0) / n
        cpa = per_adv_cpa.get(adv)
        if cpa and cpa > 0:
            per_adv_opp[adv] = mean_pv * cpa
            opp_index.append(mean_pv * cpa)
            bud_over_cpa.append(per_adv_budget.get(adv, 0.0) / cpa)
    cpa_vals = list(per_adv_cpa.values())
    bud_vals = list(per_adv_budget.values())
    mean_pvalue = total_pvalue / max(total_rows, 1)
    cpa_mean = float(np.mean(cpa_vals)) if cpa_vals else 0.0
    return {
        "paths": [str(p) for p in paths],
        "rows": total_rows,
        "mean_pvalue": mean_pvalue,
        "conversion_per_pv": total_conv / max(total_rows, 1),
        "exposure_rate": total_exposed / max(total_rows, 1),
        "advertiser_count": len(per_adv_rows),
        "timestep_count": len(timesteps),
        "timestep_min": min(timesteps) if timesteps else None,
        "timestep_max": max(timesteps) if timesteps else None,
        "schema": schema,
        "schema_col_count": len(schema) if schema else 0,
        "cpa_min": min(cpa_vals) if cpa_vals else None,
        "cpa_max": max(cpa_vals) if cpa_vals else None,
        "cpa_mean": cpa_mean,
        "budget_mean": float(np.mean(bud_vals)) if bud_vals else None,
        "global_opportunity_index": mean_pvalue * cpa_mean,
        "opportunity_index": _series_stats(pd.Series(opp_index, dtype=float)),
        "opportunity_index_values": opp_index,
        "per_advertiser_cpa": {int(k): v for k, v in sorted(per_adv_cpa.items())},
        "per_advertiser_opportunity_index": {int(k): v for k, v in sorted(per_adv_opp.items())},
        "budget_over_cpa": _series_stats(pd.Series(bud_over_cpa, dtype=float)),
    }


def _measure_endpoints(args, cache: Path) -> tuple[dict, dict]:
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["high"], d["low"]
    print(f"[M1] measuring high endpoint: {args.high_raw}", flush=True)
    high = _aggregate_raw([args.high_raw], args.chunksize)
    print(f"[M1] measuring low endpoint: {args.low_raw}", flush=True)
    low = _aggregate_raw([args.low_raw], args.chunksize)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"high": high, "low": low, "cached_at_iso": _now_iso()},
                                indent=2, ensure_ascii=False) + "\n")
    print(f"[M1] cached endpoint profile: {cache}", flush=True)
    return high, low


def _evaluate_candidate(cand: dict, high: dict, low: dict,
                        target_index: float, envelope: tuple[float, float]) -> dict:
    opp_vals = cand["opportunity_index_values"]
    opp_median = cand["opportunity_index"]["median"]
    cond1 = high["mean_pvalue"] > cand["mean_pvalue"] > low["mean_pvalue"]
    cond2 = abs(opp_median - target_index) / max(target_index, 1e-12) <= 0.20
    if opp_vals:
        inside = sum(1 for v in opp_vals if envelope[0] <= v <= envelope[1])
        cond3 = inside / len(opp_vals) >= 0.90
    else:
        cond3 = False
    cond4 = (cand["cpa_min"] is not None and cand["cpa_min"] >= 20.0
             and cand["cpa_max"] is not None and cand["cpa_max"] <= 60.0)
    cond5 = (cand["advertiser_count"] == 48 and cand["timestep_count"] == 48
             and cand["schema_col_count"] == 18)
    passed = cond1 and cond2 and cond3 and cond4 and cond5
    return {
        "mean_pvalue": cand["mean_pvalue"],
        "global_opportunity_index": cand["global_opportunity_index"],
        "opp_index_median": opp_median,
        "opp_index_target": target_index,
        "opp_index_rel_err": abs(opp_median - target_index) / max(target_index, 1e-12),
        "envelope": list(envelope),
        "cond1_high_gt_cand_gt_low": cond1,
        "cond2_within_20pct_target": cond2,
        "cond3_90pct_in_envelope": cond3,
        "cond4_cpa_in_mid_band": cond4,
        "cond5_schema_adv_ts": cond5,
        "passed": passed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="M1 structural calibration report")
    p.add_argument("--run-id", required=True)
    p.add_argument("--high-raw", type=Path,
                   default=_REPO_ROOT / "data" / "high" / "test" / "period-0.csv")
    p.add_argument("--low-raw", type=Path,
                   default=_REPO_ROOT / "data" / "low" / "test" / "period-0.csv")
    p.add_argument("--candidates-root", type=Path,
                   default=_REPO_ROOT / "data" / "calibration" / "mid")
    p.add_argument("--scales", default="2.00,2.25,2.50,3.00")
    p.add_argument("--target-index", type=float, default=0.04243,
                   help="Geometric endpoint target (spec sec 3). Default 0.04243.")
    p.add_argument("--endpoints-cache", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    args = p.parse_args()

    if args.endpoints_cache is None:
        args.endpoints_cache = (_REPO_ROOT / "data" / "manifests"
                                / f"mid_recalibration_endpoints_{args.run_id}.json")

    high, low = _measure_endpoints(args, args.endpoints_cache)
    # Recompute target from observed endpoints as a cross-check against spec.
    target_observed = math.sqrt(high["global_opportunity_index"] * low["global_opportunity_index"])
    target_index = args.target_index
    # Combined robust endpoint envelope (per-advertiser min/max) with 25% tolerance.
    hi_vals = high["opportunity_index_values"]
    lo_vals = low["opportunity_index_values"]
    endpoint_vals = hi_vals + lo_vals
    if endpoint_vals:
        lo_env = min(endpoint_vals) * 0.75
        hi_env = max(endpoint_vals) * 1.25
    else:
        lo_env, hi_env = 0.0, float("inf")
    envelope = (lo_env, hi_env)

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    candidates: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for scale in scales:
        raw_dir = args.candidates_root / args.run_id / f"scale_{scale}" / "raw"
        raw_files = sorted(raw_dir.glob("period-*.csv")) if raw_dir.exists() else []
        if not raw_files:
            print(f"[M1] WARN: no raw files for scale {scale} at {raw_dir}", flush=True)
            evaluations[scale] = {"passed": False, "error": "no raw files", "raw_dir": str(raw_dir)}
            continue
        print(f"[M1] measuring candidate scale={scale}: {len(raw_files)} raw file(s)", flush=True)
        cand = _aggregate_raw(raw_files, args.chunksize)
        candidates[scale] = cand
        evaluations[scale] = _evaluate_candidate(cand, high, low, target_index, envelope)

    # Select candidate closest to target among those passing all conditions.
    passing = [(s, evaluations[s]) for s in scales if evaluations[s].get("passed")]
    passing.sort(key=lambda kv: kv[1]["opp_index_rel_err"])
    selected = passing[0][0] if passing else None

    manifest = {
        "task": "mid_recalibration_m1_selection",
        "run_id": args.run_id,
        "generated_at_iso": _now_iso(),
        "target_index_spec": target_index,
        "target_index_observed": target_observed,
        "envelope": list(envelope),
        "high_mean_pvalue": high["mean_pvalue"],
        "low_mean_pvalue": low["mean_pvalue"],
        "high_global_opportunity_index": high["global_opportunity_index"],
        "low_global_opportunity_index": low["global_opportunity_index"],
        "candidates": candidates,
        "evaluations": evaluations,
        "selected_scale": selected,
        "note": ("Selection is by structural distance to target only; score is not "
                 "used unless structural distances are effectively equal."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[M1] wrote manifest: {args.output}", flush=True)

    # Human-readable summary.
    print("\n=== M1 Selection Summary ===", flush=True)
    print(f"target_index(spec)={target_index:.5f}  target_index(observed)={target_observed:.5f}", flush=True)
    print(f"envelope(25% tol)=[{envelope[0]:.6f}, {envelope[1]:.6f}]", flush=True)
    print(f"high mean_pvalue={high['mean_pvalue']:.8f}  low mean_pvalue={low['mean_pvalue']:.8f}", flush=True)
    print(f"\n{'scale':>6} {'mean_pvalue':>14} {'opp_median':>12} {'rel_err':>10} "
          f"{'c1':>4} {'c2':>4} {'c3':>4} {'c4':>4} {'c5':>4} {'pass':>5}", flush=True)
    for s in scales:
        e = evaluations[s]
        if "error" in e:
            print(f"{s:>6}  ERROR: {e['error']}", flush=True)
            continue
        print(f"{s:>6} {e['mean_pvalue']:>14.8f} {e['opp_index_median']:>12.6f} "
              f"{e['opp_index_rel_err']:>10.3f} {str(e['cond1_high_gt_cand_gt_low']):>4} "
              f"{str(e['cond2_within_20pct_target']):>4} {str(e['cond3_90pct_in_envelope']):>4} "
              f"{str(e['cond4_cpa_in_mid_band']):>4} {str(e['cond5_schema_adv_ts']):>4} "
              f"{str(e['passed']):>5}", flush=True)
    print(f"\nSELECTED: {selected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
