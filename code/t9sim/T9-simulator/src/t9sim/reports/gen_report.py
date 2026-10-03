"""Clean C1-C4 results table from a pipeline results.json.

Computes the Bayes-ceiling AUCs from the run's master (rank actuals by the
TRUE probability) and prints a markdown table: per-model logloss / AUC / RMSE
/ ev_ratio / ev_spearman across conditions, plus the ceiling and % -achievable.

Usage:  python -m t9sim.reports.gen_report output/scale10m/pipeline/results.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score as auc

CONDS = ("C1", "C2", "C3", "C4")


def ceilings(master_path):
    d = pd.read_parquet(master_path, columns=["click", "install", "is_payer",
                                              "p_click", "p_install", "p_payer"])
    ci, ii = d[d.click == 1], d[d.install == 1]
    return {"click": round(auc(d.click, d.p_click), 4),
            "install": round(auc(ci.install, ci.p_install), 4),
            "payer": round(auc(ii.is_payer, ii.p_payer), 4)}


def pct_achievable(a, ceil):
    if a is None or ceil is None or ceil <= 0.5:
        return None
    return round(100 * (a - 0.5) / (ceil - 0.5), 0)


def main():
    rj = Path(sys.argv[1])
    d = json.loads(rj.read_text())
    master = rj.parents[1] / "auctions.parquet"
    cl = ceilings(master) if master.exists() else {}
    C = d["conditions"]

    def row(label, key, fmt="{}"):
        cells = " | ".join(fmt.format(C[c].get(key)) for c in CONDS)
        return f"| {label} | {cells} |"

    print(f"# C1-C4 results — {d['profile']}  "
          f"(n={d['n_rows']:,}, test={d['split_counts']['test']:,})\n")
    print("| metric | C1 | C2 | C3 | C4 |")
    print("|---|---|---|---|---|")
    print(row("P(click) AUC", "auc_click"))
    print(row("P(click) logloss", "logloss_click"))
    print(row("P(install) AUC", "auc_install"))
    print(row("P(install) logloss", "logloss_install"))
    print(row("P(payer) AUC", "auc_payer"))
    print(row("P(payer) logloss", "logloss_payer"))
    print(row("E(spend|payer) RMSE", "rmse_spend"))
    print(row("EV ev_ratio (pred/true)", "ev_ratio"))
    print(row("EV ev_spearman (ranking)", "ev_spearman"))
    print(row("P(win) AUC", "auc_win"))
    print(row("P(win) logloss", "logloss_win"))
    print(row("ROAS [confounded]", "roas"))
    print(row("profit $ [confounded]", "profit"))

    if cl:
        c2 = C["C2"]
        print("\n## Ceilings (Bayes-optimal AUC, rank by TRUE prob) & "
              "% of achievable captured (C2)\n")
        print("| stage | model AUC (C2) | Bayes ceiling | % achievable |")
        print("|---|---|---|---|")
        print(f"| P(click) | {c2.get('auc_click')} | {cl['click']} | "
              f"{pct_achievable(c2.get('auc_click'), cl['click'])}% |")
        print(f"| P(install) | {c2.get('auc_install')} | {cl['install']} | "
              f"{pct_achievable(c2.get('auc_install'), cl['install'])}% |")
        print(f"| P(payer) | {c2.get('auc_payer')} | {cl['payer']} | "
              f"{pct_achievable(c2.get('auc_payer'), cl['payer'])}% |")

    o, h = d.get("conditions", {}).get("oracle", {}), d.get("h2_incumbent", {})
    print(f"\n**Reference bidders:** oracle ROAS {o.get('roas')} "
          f"(profit ${o.get('profit')}) · H2 ROAS {h.get('roas')} "
          f"(profit ${h.get('profit')})")
    if "oracle_features" in d:
        print(f"\n*(oracle-features run: latents exposed to Tier-1)*")


if __name__ == "__main__":
    main()
