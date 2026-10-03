"""Multi-seed confidence intervals at 1M (Priority 2b).

Regenerates the 1M (test) dataset under several seeds, runs the C1-C4 pipeline
on each, and aggregates mean +/- std per metric and per lift-vs-C1 — so the
reported lifts carry error bars (vs the gamma-sweep, which varies a structural
parameter instead of random noise). Writes output/multiseed.json.
"""
import json
import time

import numpy as np

from t9sim import simulate, pipeline
from t9sim.paths import OUTPUT_DIR

SEEDS = [90211, 131, 271, 314, 415]            # first = canonical test seed
METRICS = ["auc_click", "auc_install", "auc_payer", "ev_ratio", "ev_spearman",
           "auc_win", "roas", "profit"]
CONDS = ("C1", "C2", "C3", "C4")


def main():
    t0 = time.time()
    per_seed = []
    for s in SEEDS:
        name = f"seed_{s}"
        print(f"\n=== seed {s} -> {name} ===")
        simulate.run(profile="test", seed=s, out_name=name)
        res = pipeline.run_profile(name, do_shap=False)
        per_seed.append(res["conditions"])

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4),
                len(vals)]                      # mean, std, n

    out = {"scale": "1M (test)", "seeds": SEEDS, "n_seeds": len(SEEDS),
           "per_condition": {}, "lifts_vs_C1": {}, "format": "[mean, std, n]"}
    for c in CONDS:
        out["per_condition"][c] = {m: stats([ps[c].get(m) for ps in per_seed])
                                   for m in METRICS}
    for c in ("C2", "C3", "C4"):
        out["lifts_vs_C1"][c] = {}
        for m in METRICS:
            diffs = [ps[c].get(m) - ps["C1"].get(m) for ps in per_seed
                     if ps[c].get(m) is not None and ps["C1"].get(m) is not None]
            out["lifts_vs_C1"][c][m] = stats(diffs)
    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUTPUT_DIR / "multiseed.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")
    # clean temp per-seed datasets (keep the json)
    import shutil
    for s in SEEDS:
        shutil.rmtree(OUTPUT_DIR / f"seed_{s}", ignore_errors=True)
    print("\nwrote", OUTPUT_DIR / "multiseed.json")
    print(json.dumps(out["lifts_vs_C1"], indent=2))


if __name__ == "__main__":
    main()
