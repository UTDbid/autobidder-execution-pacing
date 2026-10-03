#!/usr/bin/env python3
"""Measure raw/test and trajectory conversion profile statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _series_stats(s: pd.Series) -> dict[str, float | int]:
    if len(s) == 0:
        return {"n": 0, "mean": 0.0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    q = s.quantile([0.25, 0.5, 0.75])
    return {
        "n": int(len(s)),
        "mean": _safe_float(s.mean()),
        "min": _safe_float(s.min()),
        "q25": _safe_float(q.loc[0.25]),
        "median": _safe_float(q.loc[0.5]),
        "q75": _safe_float(q.loc[0.75]),
        "max": _safe_float(s.max()),
    }


def measure_raw_csv(path: Path, chunksize: int) -> dict[str, Any]:
    total_rows = 0
    total_conversions = 0.0
    total_exposed = 0.0
    total_pvalue = 0.0
    total_cost = 0.0
    per_adv_rows: dict[int, int] = {}
    per_adv_conv: dict[int, float] = {}
    per_adv_pvalue: dict[int, float] = {}
    per_adv_cpa: dict[int, float] = {}
    per_adv_budget: dict[int, float] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows = len(chunk)
        total_rows += rows
        conv = chunk.get("conversionAction", pd.Series(np.zeros(rows))).astype(float)
        exposed = chunk.get("isExposed", pd.Series(np.zeros(rows))).astype(float)
        pvalue = chunk.get("pValue", pd.Series(np.zeros(rows))).astype(float)
        cost = chunk.get("cost", pd.Series(np.zeros(rows))).astype(float)
        total_conversions += float(conv.sum())
        total_exposed += float(exposed.sum())
        total_pvalue += float(pvalue.sum())
        total_cost += float(cost.sum())
        if "advertiserNumber" in chunk.columns:
            adv = chunk["advertiserNumber"].astype(int)
            grouped = pd.DataFrame({"adv": adv, "conv": conv, "pvalue": pvalue}).groupby("adv")
            counts = grouped.size()
            convs = grouped["conv"].sum()
            pvals = grouped["pvalue"].sum()
            for k, v in counts.items():
                per_adv_rows[int(k)] = per_adv_rows.get(int(k), 0) + int(v)
            for k, v in convs.items():
                per_adv_conv[int(k)] = per_adv_conv.get(int(k), 0.0) + float(v)
            for k, v in pvals.items():
                per_adv_pvalue[int(k)] = per_adv_pvalue.get(int(k), 0.0) + float(v)
            # CPAConstraint and budget are constant per advertiser within a
            # delivery period; record them for the opportunity-index and
            # budget/CPA distributions required by the recalibration spec.
            if "CPAConstraint" in chunk.columns:
                for k, v in chunk.groupby(adv)["CPAConstraint"].first().items():
                    per_adv_cpa[int(k)] = float(v)
            if "budget" in chunk.columns:
                for k, v in chunk.groupby(adv)["budget"].first().items():
                    per_adv_budget[int(k)] = float(v)

    per_adv_rates = []
    per_adv_mean_pvalues = []
    per_adv_opportunity_index = []
    per_adv_budget_over_cpa = []
    for adv, n in sorted(per_adv_rows.items()):
        if n <= 0:
            continue
        mean_pv = per_adv_pvalue.get(adv, 0.0) / n
        per_adv_rates.append(per_adv_conv.get(adv, 0.0) / n)
        per_adv_mean_pvalues.append(mean_pv)
        cpa = per_adv_cpa.get(adv)
        if cpa and cpa > 0:
            per_adv_opportunity_index.append(mean_pv * cpa)
            bud = per_adv_budget.get(adv, 0.0)
            per_adv_budget_over_cpa.append(bud / cpa)

    return {
        "kind": "raw",
        "path": str(path),
        "rows": total_rows,
        "conversion_per_pv": total_conversions / max(total_rows, 1),
        "conversion_per_exposure": total_conversions / max(total_exposed, 1.0),
        "exposure_rate": total_exposed / max(total_rows, 1),
        "mean_pvalue": total_pvalue / max(total_rows, 1),
        "mean_cost": total_cost / max(total_rows, 1),
        "total_conversions": total_conversions,
        "total_exposed": total_exposed,
        "per_advertiser_conversion_rate": _series_stats(pd.Series(per_adv_rates, dtype=float)),
        "per_advertiser_mean_pvalue": _series_stats(pd.Series(per_adv_mean_pvalues, dtype=float)),
        "per_advertiser_opportunity_index": _series_stats(pd.Series(per_adv_opportunity_index, dtype=float)),
        "per_advertiser_budget_over_cpa": _series_stats(pd.Series(per_adv_budget_over_cpa, dtype=float)),
        "per_advertiser_opportunity_index_values": per_adv_opportunity_index,
        "per_advertiser_count": len(per_adv_opportunity_index),
    }


def measure_trajectory_csv(path: Path, chunksize: int) -> dict[str, Any]:
    total_rows = 0
    reward_values = []
    reward_cont_values = []
    action_values = []
    real_conv_values = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)
        if "reward" in chunk.columns:
            reward_values.append(chunk["reward"].astype(float))
        if "reward_continuous" in chunk.columns:
            reward_cont_values.append(chunk["reward_continuous"].astype(float))
        if "action" in chunk.columns:
            action_values.append(chunk["action"].astype(float))
        if "realAllConversion" in chunk.columns:
            real_conv_values.append(chunk["realAllConversion"].astype(float))
    rewards = pd.concat(reward_values, ignore_index=True) if reward_values else pd.Series(dtype=float)
    reward_cont = pd.concat(reward_cont_values, ignore_index=True) if reward_cont_values else pd.Series(dtype=float)
    actions = pd.concat(action_values, ignore_index=True) if action_values else pd.Series(dtype=float)
    real_conv = pd.concat(real_conv_values, ignore_index=True) if real_conv_values else pd.Series(dtype=float)
    return {
        "kind": "trajectory",
        "path": str(path),
        "rows": total_rows,
        "reward": _series_stats(rewards),
        "reward_continuous": _series_stats(reward_cont),
        "action": _series_stats(actions),
        "realAllConversion": _series_stats(real_conv),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--trajectory-csv", type=Path, default=None)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    args = p.parse_args()

    if args.raw_csv is None and args.trajectory_csv is None:
        raise SystemExit("provide --raw-csv and/or --trajectory-csv")

    result: dict[str, Any] = {"raw": None, "trajectory": None}
    if args.raw_csv is not None:
        if not args.raw_csv.exists():
            raise SystemExit(f"raw csv not found: {args.raw_csv}")
        result["raw"] = measure_raw_csv(args.raw_csv, args.chunksize)
    if args.trajectory_csv is not None:
        if not args.trajectory_csv.exists():
            raise SystemExit(f"trajectory csv not found: {args.trajectory_csv}")
        result["trajectory"] = measure_trajectory_csv(args.trajectory_csv, args.chunksize)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
