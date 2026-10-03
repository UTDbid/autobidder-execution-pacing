"""Gamma sensitivity sweep (protocol #6 / FN-2) at 1M scale, single seed.

For each gamma in the grid: regenerate a 1M dataset with that market
value-sensitivity (win-rate re-pinned per gamma), run the C1-C4 pipeline,
and record the ROAS lifts. gamma=0 is the sanity floor (layers should add
nothing). Writes output/gamma_sweep.json.

NOTE this is the cheaper meaningful scale; the definitive sweep is a 10M+/HPC
job. Single seed here (CIs are a separate multi-seed run).
"""
import json
import time

from t9sim import simulate, pipeline
from t9sim.paths import OUTPUT_DIR

GAMMAS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.90]


def main():
    t0 = time.time()
    rows = []
    for g in GAMMAS:
        name = f"gsweep_g{g:.2f}"
        print(f"\n=== gamma={g} -> {name} ===")
        simulate.run(profile="test", gamma=g, out_name=name)
        res = pipeline.run_profile(name, do_shap=False)
        cd = res["conditions"]
        r = {c: cd.get(c, {}).get("roas") for c in ("C1", "C2", "C3", "C4")}
        # CLEAN mechanism signal (bidder-independent): MMP's click-AUC lift
        # should emerge from adverse selection -> vanish at gamma=0.
        auc = {c: cd.get(c, {}).get("auc_click") for c in ("C1", "C2", "C3", "C4")}
        aucw = {c: cd.get(c, {}).get("auc_win") for c in ("C1", "C2", "C3", "C4")}

        def _d(a, b):
            return round(a - b, 4) if a is not None and b is not None else None
        row = {"gamma": g,
               "auc_click": auc, "auc_win": aucw, "roas_CONFOUNDED": r,
               "auc_lift_MMP_C2": _d(auc["C2"], auc["C1"]),
               "auc_win_lift_SSP_C3": _d(aucw["C3"], aucw["C1"]),
               "roas_lift_C2_CONFOUNDED": _d(r["C2"], r["C1"])}
        rows.append(row)
        print(g, "auc_lift_MMP", row["auc_lift_MMP_C2"],
              "auc_win_lift_SSP", row["auc_win_lift_SSP_C3"])
    out = {"scale": "1M (test profile)", "seed": 90211, "rows": rows,
           "elapsed_s": round(time.time() - t0, 1)}
    (OUTPUT_DIR / "gamma_sweep.json").write_text(json.dumps(out, indent=2),
                                                 encoding="utf-8")
    print("\nwrote", OUTPUT_DIR / "gamma_sweep.json")


if __name__ == "__main__":
    main()
