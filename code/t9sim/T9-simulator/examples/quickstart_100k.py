"""Quickstart: generate a 100K master (schema V10, anchored rho*=0.8), run the
reference ablation under all four censored views, and print the headline table.

    python examples/quickstart_100k.py          # ~1.5 min on an idle laptop

For the paper-scale runs use profile "test" (1M) or "scale10m" (10M, one fresh
process per seed — see scripts/v10_10m_driver.py).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import t9sim.api as t9

SEED = 90213

print(f"1) generating 100K master (V10 edges, rho={t9.RHO_STAR}, seed {SEED})")
t9.generate("golden", seed=SEED, rho=t9.RHO_STAR, bn_edges=t9.V10_EDGES)

print("\n2) training + scoring the reference stack under C1-C4")
res = t9.evaluate("golden")

print("\n3) headline (100K, single seed - see docs/ for the n=10 tables)")
ROWS = [("ev_spearman", "EV rank corr"), ("ev_ratio", "EV ratio (1.0=unbiased)"),
        ("auc_install", "install AUC"), ("auc_win_clf", "win AUC (classifier)"),
        ("clf_bidder.profit", "profit $")]


def cell(cond, key):
    """Fetch a metric, following one level of nesting for clf_bidder.*."""
    d = res["conditions"][cond]
    if "." in key:
        outer, inner = key.split(".", 1)
        return (d.get(outer) or {}).get(inner)
    return d.get(key)


print(f"{'metric':26} " + " ".join(f"{c:>9}" for c in t9.CONDITIONS))
for key, label in ROWS:
    cells = [cell(c, key) for c in t9.CONDITIONS]
    print(f"{label:26} " + " ".join(
        f"{v:9.4f}" if isinstance(v, float) else f"{v!s:>9}" for v in cells))

print("\nDesign identities to expect: C1==C3, C2==C4 on funnel metrics "
      "(MMP axis); C1==C2, C3==C4 on win metrics (SSP axis).")
print("The prediction rows already show the paper's pattern. Profit at 100K on "
      "one seed is noise-dominated and does NOT reproduce the paper's MMP gain, "
      "which needs 1M+ and ten seeds - see docs/v10_Training_Results.md.")
