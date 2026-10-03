#!/usr/bin/env python3
"""iPinYou real-log auxiliary study for reference architecture and control width."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score

from reference_policies import (
    DualPacingState,
    PIDPacingState,
    dual_pacing_reference,
    pid_pacing_reference,
)


FEATURES = [
    "weekday", "hour", "os_browser", "region", "city", "adexchange", "domain",
    "slotid", "slotwidth", "slotheight", "slotvisibility", "slotformat",
    "slotprice", "creative", "keypage", "usertag",
]
EVAL_COLUMNS = ["timestamp", "date", "hour", "payprice", "click", "conversion"] + FEATURES
PCTR_FLOOR = 1e-8


def make_action_grid(calibration: pd.DataFrame) -> np.ndarray:
    """Create an action grid from development support only.

    iPinYou bid prices and pCTR are on very different numerical scales.  A fixed
    upper cap can therefore make every pacing rule saturate.  The largest action
    here is the development-period price required to cover the declared pCTR
    floor; no official-test price or outcome is used.
    """
    maximum_payprice = max(float(calibration["payprice"].max()), 1.0)
    maximum_action = max(maximum_payprice / PCTR_FLOOR * 1.05, 1_000_000.0)
    return np.geomspace(1.0, maximum_action, 240)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_tokens(frame: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in frame.itertuples(index=False):
        values = item._asdict()
        tokens = []
        for name in FEATURES:
            value = values[name]
            if name == "usertag":
                tags = str(value).split(",")[:20]
                tokens.extend(f"tag={tag}" for tag in tags if tag and tag != "null")
            elif name == "slotprice":
                bucket = int(np.log1p(max(float(value), 0.0)) * 4)
                tokens.append(f"slotprice_bucket={bucket}")
            elif name in {"slotwidth", "slotheight"}:
                tokens.append(f"{name}={int(value)}")
            else:
                tokens.append(f"{name}={value}")
        rows.append(tokens)
    return rows


def chronological_cutoff(path: Path) -> int:
    values = pq.read_table(path, columns=["timestamp"])["timestamp"].to_numpy()
    return int(np.quantile(values, 0.8, method="nearest"))


def train_pctr(path: Path, cutoff: int, seed: int) -> tuple[FeatureHasher, SGDClassifier, dict[str, int]]:
    hasher = FeatureHasher(n_features=2**18, input_type="string", alternate_sign=False)
    model = SGDClassifier(loss="log_loss", alpha=1e-6, average=True, random_state=seed)
    first = True
    counts = {"rows": 0, "clicks": 0}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=FEATURES + ["timestamp", "click"]):
        frame = batch.to_pandas()
        frame = frame[frame["timestamp"] <= cutoff]
        if frame.empty:
            continue
        x = hasher.transform(feature_tokens(frame[FEATURES]))
        y = frame["click"].to_numpy(dtype=int)
        model.partial_fit(x, y, classes=np.array([0, 1])) if first else model.partial_fit(x, y)
        first = False
        counts["rows"] += len(frame)
        counts["clicks"] += int(y.sum())
    if first:
        raise RuntimeError("no pCTR training rows before cutoff")
    return hasher, model, counts


def predict_split(path: Path, hasher: FeatureHasher, model: SGDClassifier, *, after_cutoff: int | None = None) -> pd.DataFrame:
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=EVAL_COLUMNS):
        frame = batch.to_pandas()
        if after_cutoff is not None:
            frame = frame[frame["timestamp"] > after_cutoff]
        if frame.empty:
            continue
        frame = frame.copy()
        frame["raw_score"] = model.decision_function(
            hasher.transform(feature_tokens(frame[FEATURES]))
        )
        parts.append(frame[["timestamp", "date", "hour", "payprice", "click", "conversion", "raw_score"]])
    if not parts:
        raise RuntimeError("prediction split is empty")
    return pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)


def fit_score_calibrator(calibration: pd.DataFrame) -> tuple[dict[str, Any], Any]:
    """Fit a calibration-only Platt map over the hashed-logistic score.

    The upstream high-dimensional SGD score occasionally has a severely
    shifted intercept in sparse campaigns.  A one-dimensional calibration map
    restores probabilistic scale without reading test rows or test outcomes.
    """
    score = calibration["raw_score"].to_numpy(dtype=float)
    click = calibration["click"].to_numpy(dtype=int)
    center = float(np.median(score))
    scale = float(np.quantile(score, 0.75) - np.quantile(score, 0.25))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(score))
    scale = max(scale, 1e-8)
    z = ((score - center) / scale).reshape(-1, 1)
    if np.unique(click).size < 2:
        prior = float((click.sum() + 0.5) / (len(click) + 1.0))

        def transform(values: np.ndarray) -> np.ndarray:
            return np.full(len(values), prior, dtype=float)

        info = {
            "method": "empirical_prior_fallback",
            "center": center,
            "scale": scale,
            "calibration_rows": len(click),
            "calibration_clicks": int(click.sum()),
            "calibration_auc": None,
        }
        return info, transform
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=0)
    model.fit(z, click)

    def transform(values: np.ndarray) -> np.ndarray:
        standardized = ((np.asarray(values, dtype=float) - center) / scale).reshape(-1, 1)
        return np.clip(model.predict_proba(standardized)[:, 1], PCTR_FLOOR, 1 - PCTR_FLOOR)

    calibrated = transform(score)
    info = {
        "method": "platt_logistic_on_chronological_calibration",
        "center": center,
        "scale": scale,
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "calibration_rows": len(click),
        "calibration_clicks": int(click.sum()),
        "calibration_auc": float(roc_auc_score(click, calibrated)),
        "calibration_mean_pctr": float(calibrated.mean()),
        "calibration_ctr": float(click.mean()),
        "calibration_pctr_quantiles": np.quantile(
            calibrated, [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
    }
    return info, transform


def quantile_edges(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Return monotone calibration-only cut points with open outer bounds."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot build quantile edges from an empty array")
    inner = np.quantile(finite, np.arange(1, n_bins) / n_bins)
    return np.r_[-np.inf, np.maximum.accumulate(inner), np.inf]


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # searchsorted handles coincident empirical quantiles deterministically.
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def purchase_mix(
    frame: pd.DataFrame,
    won: np.ndarray,
    *,
    pctr_edges: np.ndarray,
    price_edges: np.ndarray,
) -> list[dict[str, Any]]:
    p_bin = bin_index(frame["pctr"].to_numpy(dtype=float), pctr_edges)
    price_bin = bin_index(frame["payprice"].to_numpy(dtype=float), price_edges)
    cost = frame["payprice"].to_numpy(dtype=float) / 1000.0
    click = frame["click"].to_numpy(dtype=int)
    conversion = frame["conversion"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for pctr_quintile in range(5):
        for price_quintile in range(5):
            mask = won & (p_bin == pctr_quintile) & (price_bin == price_quintile)
            rows.append({
                "pctr_quintile": pctr_quintile + 1,
                "paying_price_quintile": price_quintile + 1,
                "wins": int(mask.sum()),
                "spend": float(cost[mask].sum()),
                "clicks": int(click[mask].sum()),
                "conversions": int(conversion[mask].sum()),
                "pctr_sum": float(frame["pctr"].to_numpy(dtype=float)[mask].sum()),
            })
    return rows


def spend_under_budget(bids: np.ndarray, frame: pd.DataFrame, budget: float) -> tuple[float, int, int, int]:
    candidate = bids >= frame["payprice"].to_numpy()
    spend = 0.0
    clicks = conversions = wins = 0
    pay = frame["payprice"].to_numpy() / 1000.0
    click = frame["click"].to_numpy(dtype=int)
    conversion = frame["conversion"].to_numpy(dtype=int)
    for index in np.flatnonzero(candidate):
        cost = float(pay[index])
        if spend + cost <= budget + 1e-12:
            spend += cost
            wins += 1
            clicks += int(click[index])
            conversions += int(conversion[index])
    return spend, wins, clicks, conversions


def calibrate_bidder(calibration: pd.DataFrame) -> dict[str, float]:
    historical_cost = float(calibration["payprice"].sum() / 1000.0)
    budget = max(historical_cost / 8.0, 1e-6)
    pctr = calibration["pctr"].to_numpy()
    best = None
    for c in (10.0, 30.0, 60.0, 100.0, 200.0):
        for lam in np.geomspace(1e-8, 1e-3, 36):
            bids = np.sqrt(c * c + c * pctr / lam) - c
            spend, wins, clicks, conversions = spend_under_budget(bids, calibration, budget)
            row = (clicks, conversions, float(pctr[bids >= calibration["payprice"].to_numpy()].sum()), spend, -c, -lam)
            if best is None or row > best[0]:
                best = (row, c, float(lam), spend, wins, clicks, conversions)
    assert best is not None
    _, c, lam, spend, wins, clicks, conversions = best
    return {
        "c": c, "lambda": lam, "calibration_budget": budget,
        "calibration_spend": spend, "calibration_wins": wins,
        "calibration_clicks": clicks, "calibration_conversions": conversions,
    }


def calibrate_linear_reference(
    calibration: pd.DataFrame, budget: float, action_grid: np.ndarray
) -> float:
    pctr = calibration["pctr"].to_numpy()
    pay = calibration["payprice"].to_numpy()
    costs = pay / 1000.0
    spends = np.asarray([costs[action * pctr >= pay].sum() for action in action_grid])
    index = int(np.argmin(np.abs(spends - budget)))
    return float(action_grid[index])


def traffic_reference(
    calibration: pd.DataFrame,
    current_count: int,
    target_spend: float,
    fallback: float,
    action_grid: np.ndarray,
) -> float:
    pctr = calibration["pctr"].to_numpy()
    pay = calibration["payprice"].to_numpy()
    costs = pay / 1000.0
    spends = np.asarray([costs[action * pctr >= pay].sum() for action in action_grid])
    spends *= float(current_count) / max(len(calibration), 1)
    index = int(np.searchsorted(spends, target_spend, side="left"))
    return float(action_grid[min(index, len(action_grid) - 1)]) if target_spend > 0 else 0.0


def replay(
    test: pd.DataFrame,
    calibration: pd.DataFrame,
    bidder: dict[str, float],
    *,
    reference: str,
    kappa: float | None,
    budget_fraction: float,
    fallback_action: float,
    action_grid: np.ndarray,
    pctr_edges: np.ndarray,
    price_edges: np.ndarray,
    h: int = 4,
) -> tuple[dict[str, Any], np.ndarray]:
    historical_cost = float(test["payprice"].sum() / 1000.0)
    budget = max(historical_cost * budget_fraction, 1e-6)
    remaining = budget
    pctr = test["pctr"].to_numpy()
    payprice = test["payprice"].to_numpy()
    cost = payprice / 1000.0
    raw_bid = np.sqrt(bidder["c"] ** 2 + bidder["c"] * pctr / bidder["lambda"]) - bidder["c"]
    raw_action = raw_bid / np.maximum(pctr, 1e-12)
    dates = pd.Categorical(test["date"], categories=sorted(test["date"].unique()), ordered=True).codes
    steps = dates * 24 + test["hour"].to_numpy(dtype=int)
    horizon = int(steps.max()) + 1
    pid_state, dual_state = PIDPacingState(), DualPacingState()
    current_reference = fallback_action
    won = np.zeros(len(test), dtype=bool)
    executed_bid = np.zeros(len(test), dtype=float)
    lower_binding = upper_binding = 0
    references = []
    spend_by_step = np.zeros(horizon)
    clicks_by_step = np.zeros(horizon)
    for step in range(horizon):
        indices = np.flatnonzero(steps == step)
        if indices.size == 0 or remaining <= 1e-12:
            continue
        if reference != "raw" and step % h == 0:
            target = remaining / max(horizon - step, 1)
            if reference == "traffic_aware":
                current_reference = traffic_reference(
                    calibration, indices.size, target, fallback_action, action_grid
                )
            elif reference == "pid":
                current_reference = pid_pacing_reference(
                    pid_state, budget=budget, remaining=remaining, time_step=step,
                    horizon=horizon, fallback_action=fallback_action,
                    kp=4.0, ki=0.5, kd=0.0, integral_limit=1.0,
                    max_action=float(action_grid[-1]),
                )
            elif reference == "dual":
                current_reference = dual_pacing_reference(
                    dual_state, budget=budget, remaining=remaining, time_step=step,
                    horizon=horizon, fallback_action=fallback_action,
                    eta=10.0, log_shadow_limit=math.log(20.0),
                    max_action=float(action_grid[-1]),
                )
            else:
                raise ValueError(reference)
            references.append(float(current_reference))
        if reference == "raw":
            bids = raw_bid[indices]
        else:
            low = max(current_reference * (1.0 - float(kappa)), 0.0)
            high = current_reference * (1.0 + float(kappa))
            lower_binding += int(np.sum(raw_action[indices] < low))
            upper_binding += int(np.sum(raw_action[indices] > high))
            bids = np.clip(raw_action[indices], low, high) * pctr[indices]
        executed_bid[indices] = bids
        for local_index, global_index in enumerate(indices):
            if bids[local_index] < payprice[global_index]:
                continue
            item_cost = float(cost[global_index])
            if item_cost <= remaining + 1e-12:
                won[global_index] = True
                remaining -= item_cost
                spend_by_step[step] += item_cost
                clicks_by_step[step] += int(test["click"].iat[global_index])
    spend = float(cost[won].sum())
    clicks = int(test["click"].to_numpy()[won].sum())
    conversions = int(test["conversion"].to_numpy()[won].sum())
    result = {
        "reference": reference,
        "kappa": kappa,
        "budget_fraction": budget_fraction,
        "budget": budget,
        "spend": spend,
        "budget_usage": spend / budget,
        "underdelivery": remaining / budget,
        "wins": int(won.sum()),
        "clicks": clicks,
        "conversions": conversions,
        "ecpc": spend / clicks if clicks else None,
        "ecpa": spend / conversions if conversions else None,
        "ctr": clicks / max(int(won.sum()), 1),
        "mean_won_pctr": float(pctr[won].mean()) if won.any() else None,
        "lower_binding_rate": lower_binding / len(test) if reference != "raw" else None,
        "upper_binding_rate": upper_binding / len(test) if reference != "raw" else None,
        "mean_reference": float(np.mean(references)) if references else None,
        "reference_log_volatility": float(np.std(np.log(np.maximum(references, 1e-12)))) if references else None,
        "spend_by_step": spend_by_step.tolist(),
        "clicks_by_step": clicks_by_step.tolist(),
        "purchase_mix": purchase_mix(
            test, won, pctr_edges=pctr_edges, price_edges=price_edges
        ),
    }
    return result, won


def add_differential_purchase_mix(
    results: list[dict[str, Any]],
    wins: dict[tuple[float, str, float | None], np.ndarray],
    test: pd.DataFrame,
    *,
    pctr_edges: np.ndarray,
    price_edges: np.ndarray,
) -> None:
    for row in results:
        reference = str(row["reference"])
        kappa = row["kappa"]
        if reference == "raw" or float(kappa) == 0.0:
            continue
        budget_fraction = float(row["budget_fraction"])
        base = wins[(budget_fraction, reference, 0.0)]
        wide = wins[(budget_fraction, reference, float(kappa))]
        row["wide_only_purchase_mix"] = purchase_mix(
            test, wide & ~base, pctr_edges=pctr_edges, price_edges=price_edges
        )
        row["tight_only_purchase_mix"] = purchase_mix(
            test, base & ~wide, pctr_edges=pctr_edges, price_edges=price_edges
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    cutoff = chronological_cutoff(args.train)
    hasher, model, train_counts = train_pctr(args.train, cutoff, args.seed)
    calibration = predict_split(args.train, hasher, model, after_cutoff=cutoff)
    test = predict_split(args.test, hasher, model)
    calibrator_info, calibrator = fit_score_calibrator(calibration)
    calibration["pctr"] = calibrator(calibration["raw_score"].to_numpy(dtype=float))
    test["pctr"] = calibrator(test["raw_score"].to_numpy(dtype=float))
    bidder = calibrate_bidder(calibration)
    action_grid = make_action_grid(calibration)
    fallback = calibrate_linear_reference(
        calibration, bidder["calibration_budget"], action_grid
    )
    pctr_edges = quantile_edges(calibration["pctr"].to_numpy(dtype=float))
    price_edges = quantile_edges(calibration["payprice"].to_numpy(dtype=float))
    results = []
    wins: dict[tuple[float, str, float | None], np.ndarray] = {}
    for budget_fraction in (1 / 32, 1 / 8, 1 / 2):
        for reference in ("traffic_aware", "pid", "dual"):
            for kappa in (0.0, 0.3, 0.8, 1.2):
                result, won = replay(
                    test, calibration, bidder, reference=reference, kappa=kappa,
                    budget_fraction=budget_fraction, fallback_action=fallback,
                    action_grid=action_grid, pctr_edges=pctr_edges,
                    price_edges=price_edges,
                )
                results.append(result)
                wins[(float(budget_fraction), reference, float(kappa))] = won
        raw_result, raw_won = replay(
            test, calibration, bidder, reference="raw", kappa=None,
            budget_fraction=budget_fraction, fallback_action=fallback,
            action_grid=action_grid, pctr_edges=pctr_edges,
            price_edges=price_edges,
        )
        results.append(raw_result)
        wins[(float(budget_fraction), "raw", None)] = raw_won
    add_differential_purchase_mix(
        results, wins, test, pctr_edges=pctr_edges, price_edges=price_edges
    )
    payload = {
        "schema_version": 2,
        "study": "iPinYou real-log support-limited reference response",
        "calibration_revision": "v3_development_supported_action_grid_plus_platt_score_calibration",
        "campaign": args.campaign,
        "train": str(args.train.resolve()), "test": str(args.test.resolve()),
        "train_sha256": sha256_file(args.train), "test_sha256": sha256_file(args.test),
        "chronological_cutoff": cutoff, "pctr_train": train_counts,
        "score_calibrator": calibrator_info,
        "test_pctr_diagnostic": {
            "mean": float(test["pctr"].mean()),
            "std": float(test["pctr"].std()),
            "quantiles": np.quantile(test["pctr"], [0, 0.1, 0.5, 0.9, 1]).tolist(),
            "nondegenerate": bool(test["pctr"].std() > 1e-10),
        },
        "calibration_rows": len(calibration), "test_rows": len(test),
        "native_bidder": {"family": "ORTB1 over hashed-logistic pCTR", **bidder},
        "reference_fallback_action": fallback,
        "action_grid": {
            "selection_data": "chronological calibration split only",
            "pctr_floor": PCTR_FLOOR,
            "minimum": float(action_grid[0]),
            "maximum": float(action_grid[-1]),
            "points": len(action_grid),
            "fallback_at_upper_cap": bool(np.isclose(fallback, action_grid[-1])),
        },
        "pid_dual_parameter_source": "inherited unchanged from AuctionNet High Period 7",
        "support_limit": "Logged second-price replay; outcomes for opportunities absent from the campaign log are not recovered.",
        "purchase_mix_design": {
            "pctr_quintile_edges": pctr_edges.tolist(),
            "paying_price_quintile_edges": price_edges.tolist(),
            "cutpoint_source": "chronological train/calibration support only",
            "cells": "pCTR quintile x logged paying-price quintile",
        },
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"OUTPUT_WRITTEN {args.output}")


if __name__ == "__main__":
    main()
