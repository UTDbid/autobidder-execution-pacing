#!/usr/bin/env python3
"""Gate M2 - fixed-policy evaluation on mid recalibration candidates + endpoints.

Evaluates two deterministic / rule policies on every M1-passing candidate plus
the high/low endpoint references, per spec Gate M2:

1. PID (heuristic controller) - re-simulates budget control via
   ``run_pid_eval.py``. Produces reward / NIPS score / CPA ratio / exceed rate /
   budget usage. This is the policy that can actually reveal a trivial
   (zero-exceedance) or impossible environment.
2. Logged-agent - replays the bid / cost / isExposed / conversionAction already
   recorded in the raw period CSV (the official mixed-strategy pool), scored
   with GAS ``getScore_nips`` (beta=2). Reference / upper baseline.

Spec requirements enforced here: same evaluator revision + seed across
candidates, ``budget_ratio=1.0``, all 48 advertisers, NIPS score formula.

PID results are cached on disk (``eval/checkpoint_static/original.json``); a
re-run skips candidates whose PID output already exists, so this script is safe
to resume alongside a concurrently-running single PID eval.

Usage::

    python scripts/data/auctionnet_official/mid_recalibration_m2_eval.py \\
        --run-id 20260715 \\
        --output data/manifests/mid_recalibration_m2_eval_20260715.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # sembid_new (scripts/data/auctionnet_official/)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from compute_logged_agent_scores import _scan_level, _per_level_report  # noqa: E402

CAL_ROOT = REPO_ROOT / "data" / "calibration" / "mid"


def _inputs(run_id: str) -> list[tuple[str, str, str, Path]]:
    """Return (label, raw_csv, conv_level, pid_output_root) tuples."""
    items: list[tuple[str, str, str, Path]] = []
    for scale in ("2.00", "2.25", "2.50"):
        raw = CAL_ROOT / run_id / f"scale_{scale}" / "raw" / "period-200000.csv"
        out = CAL_ROOT / run_id / f"scale_{scale}" / "reports" / "pid"
        items.append((f"cand_{scale}", str(raw), "mid", out))
    items.append((
        "high_ref",
        str(REPO_ROOT / "data" / "high" / "test" / "period-0.csv"),
        "high",
        CAL_ROOT / run_id / "high_ref" / "reports" / "pid",
    ))
    items.append((
        "low_ref",
        str(REPO_ROOT / "data" / "low" / "test" / "period-0.csv"),
        "low",
        CAL_ROOT / run_id / "low_ref" / "reports" / "pid",
    ))
    return items


def _pid_original_path(out_root: Path) -> Path:
    return out_root / "eval" / "checkpoint_static" / "original.json"


def run_pid(py, label, raw, conv, out_root, seed, budget, log):
    orig = _pid_original_path(out_root)
    if orig.exists():
        log(f"[M2-PID] {label}: cached -> {orig}")
        return json.loads(orig.read_text())
    log(f"[M2-PID] {label}: evaluating {raw}")
    cmd = [
        py, str(REPO_ROOT / "scripts" / "eval" / "run_pid_eval.py"),
        "--conv", conv, "--test-file", raw,
        "--budget-multiplier", str(budget), "--seed", str(seed),
        "--output-root", str(out_root),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[M2-PID] {label}: FAILED rc={r.returncode}")
        log(r.stderr[-2000:])
        return {"error": r.stderr[-2000:], "returncode": r.returncode}
    log(f"[M2-PID] {label}: done in {time.time() - t0:.0f}s")
    if orig.exists():
        return json.loads(orig.read_text())
    return {"error": "no original.json produced", "stdout": r.stdout[-1000:]}


def run_logged(label, raw, num_adv, chunksize, log):
    log(f"[M2-LOG] {label}: scanning {raw}")
    t0 = time.time()
    advs = list(range(num_adv))
    scan = _scan_level(Path(raw), advs, chunksize=chunksize, expected_period=None)
    rep = _per_level_report(label, Path(raw), scan, advs)
    log(
        f"[M2-LOG] {label}: done in {time.time() - t0:.0f}s "
        f"score={rep['average_score']:.4f} exceed={rep['exceed_rate']:.2%}"
    )
    return rep


def _fmt(v, w, prec=4):
    if isinstance(v, (int, float)):
        return f"{v:>{w}.{prec}f}"
    return f"{str(v):>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="20260715")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--num-advertisers", type=int, default=48)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260428)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--skip-pid", action="store_true")
    ap.add_argument("--skip-logged", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated labels to eval")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    inputs = _inputs(args.run_id)
    if args.only:
        want = set(args.only.split(","))
        inputs = [i for i in inputs if i[0] in want]

    results = {}
    for label, raw, conv, out_root in inputs:
        if not Path(raw).exists():
            log(f"[M2] SKIP {label}: missing {raw}")
            continue
        entry = {"raw": raw, "conv": conv, "pid_output_root": str(out_root)}
        if not args.skip_pid:
            entry["pid"] = run_pid(
                args.python, label, raw, conv, out_root, args.seed, args.budget, log
            )
        if not args.skip_logged:
            entry["logged"] = run_logged(
                label, raw, args.num_advertisers, args.chunksize, log
            )
        results[label] = entry

    log("\n=== M2 Summary ===")
    log(f"{'label':12} {'policy':7} {'reward':>14} {'score':>10} {'cpa':>10} {'exceed%':>8} {'budget%':>8}")
    for label, e in results.items():
        for pol in ("pid", "logged"):
            if pol not in e:
                continue
            d = e[pol]
            if "error" in d:
                log(f"{label:12} {pol:7} ERROR: {str(d['error'])[:60]}")
                continue
            if pol == "pid":
                log(
                    f"{label:12} {pol:7} {d['total_reward']:>14.1f} "
                    f"{d['avg_score']:>10.4f} {d['avg_cpa']:>10.4f} "
                    f"{d['exceed_rate']:>7.1f}% {d['budget_usage']:>7.1f}%"
                )
            else:
                log(
                    f"{label:12} {pol:7} {d['average_reward']:>14.4f} "
                    f"{d['average_score']:>10.4f} {d['average_cpa_real_finite']:>10.4f} "
                    f"{d['exceed_rate'] * 100:>7.2f}% {'n/a':>8}"
                )

    payload = {
        "task": "mid_recalibration_m2_eval",
        "run_id": args.run_id,
        "seed": args.seed,
        "budget_ratio": args.budget,
        "num_advertisers": args.num_advertisers,
        "generated_at_epoch": int(time.time()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"\n[M2] wrote: {args.output}")


if __name__ == "__main__":
    main()
