"""Print the diagnostic fields from a pipeline results.json.

Usage:  python -m t9sim.reports.show_diag <path/to/results.json>
"""
import json
import sys
from pathlib import Path


def main(path):
    p = Path(path)
    d = json.loads(p.read_text())
    print(f"=== {d['profile']} diagnostics (common all-rows eval) ===")
    hdr = ("cond", "auc_clk", "auc_inst", "ev_ratio", "ev_spear", "ev_rmse",
           "calib_err", "roas", "profit")
    print("  ".join(f"{h:>9}" for h in hdr))
    for c in ("C1", "C2", "C3", "C4"):
        r = d["conditions"][c]
        row = [c, r.get("auc_click"), r.get("auc_install"), r.get("ev_ratio"),
               r.get("ev_spearman"), r.get("ev_rmse"),
               r.get("wincurve_calib_err"), r.get("roas"), r.get("profit")]
        print("  ".join(f"{str(v):>9}" for v in row))
    print("\nEV pred vs true mean (ev_ratio = pred/true; <1 under-predicts):")
    for c in ("C1", "C2", "C3", "C4"):
        r = d["conditions"][c]
        print(f"  {c}: pred={r.get('ev_pred_mean')}  true={r.get('ev_true_mean')}")
    print("\nWin-curve reliability  decile -> [predicted P(win), actual win rate]:")
    for c in ("C1", "C2", "C3", "C4"):
        print(f"  {c}: {d['conditions'][c].get('wincurve_reliability')}")
    print("\noracle:", d.get("conditions", {}).get("oracle"),
          "\nh2:", d.get("h2_incumbent"))


if __name__ == "__main__":
    main(sys.argv[1])
