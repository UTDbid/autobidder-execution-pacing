"""v6 validation: target-vs-achieved table + direction checks.

Funnel metrics are computed on the MASTER (all rows carry outcomes in v6).
Levels (validation.yaml targets) are reported but only become pass/fail gates
after the full auto-calibration loop; the golden run gates on the DIRECTION
checks and the invariants in tests/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(df):
    clicks = df[df["click"] == 1]
    installs = df[df["install"] == 1]
    payers = df[df["is_payer"] == 1]

    whale_share = np.nan
    if len(payers) >= 20:
        spend = payers["ltv_value"].sort_values(ascending=False)
        top = max(1, int(np.ceil(0.05 * len(spend))))
        whale_share = float(spend.iloc[:top].sum() / spend.sum())

    return {
        "population_ctr": float(df["click"].mean()),
        "click_to_install": float(clicks["install"].mean())
        if len(clicks) else np.nan,
        "install_to_payer": float(installs["is_payer"].mean())
        if len(installs) else np.nan,
        "whale_concentration": whale_share,
        "median_payer_spend_usd": float(payers["ltv_value"].median())
        if len(payers) else np.nan,
        "auction_win_rate": float(df["won"].mean()),
    }


def directions(df):
    """The five v6 direction checks. Returns {name: (ok, detail)}."""
    out = {}

    # Archetype value ordering on the DETERMINISTIC ground-truth E[ltv]
    # (e_ltv), not realised payer-spend medians: the latter need ~thousands of
    # payers per archetype to be stable and tie/invert by noise at small scale.
    by_arch = df.groupby("lu1_archetype")["e_ltv"].mean()
    order = ["whale", "engaged_spender", "casual", "time_filler", "inactive"]
    have = [a for a in order if a in by_arch.index]
    vals = [by_arch[a] for a in have]
    out["ltv_by_archetype_monotone"] = (
        all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
        "E[ltv]: " + " > ".join(f"{a}:{v:.2f}" for a, v in zip(have, vals)))

    cats = [c.replace("lu6_", "") for c in df.columns
            if c.startswith("lu6_")]
    lu6 = df[[f"lu6_{c}" for c in cats]].to_numpy()
    gidx = pd.Series(df["ad_genre"]).map(
        {c: j for j, c in enumerate(cats)}).to_numpy()
    r = lu6[np.arange(len(df)), gidx]
    hi, lo = r > np.median(r), r <= np.median(r)
    ctr_hi, ctr_lo = df["click"][hi].mean(), df["click"][lo].mean()
    out["relevance_lifts_ctr"] = (
        bool(ctr_hi > ctr_lo),
        f"ctr(hi r)={ctr_hi:.4f} > ctr(lo r)={ctr_lo:.4f}")

    sold = df[df["clearing_price"].notna()]
    h4 = sold.groupby("slot_format")["clearing_price"].mean()
    ok = (h4.get(3, np.nan) > h4.get(2, np.nan) > h4.get(1, np.nan))
    out["format_ecpm_ordering"] = (
        bool(ok), f"rewarded={h4.get(3, np.nan):.2f} > "
        f"interstitial={h4.get(2, np.nan):.2f} > banner={h4.get(1, np.nan):.2f}")

    # log-EV on EV>0 rows: raw means are dominated by a few whale rows
    # (heavy tail) and can flip sign on an unlucky seed
    pos = df["ev_truth"] > 0
    lev_lost = np.log(df.loc[pos & (df["won"] == 0), "ev_truth"]).mean()
    lev_won = np.log(df.loc[pos & (df["won"] == 1), "ev_truth"]).mean()
    out["adverse_selection"] = (
        bool(lev_lost > lev_won),
        f"logEV(lost)={lev_lost:.2f} > logEV(won)={lev_won:.2f}")

    # within format - pooled quartiles confound bid level with format
    # (rewarded bids are highest AND face the highest hurdles)
    ok_all, details = True, []
    for code, sub in df.groupby("slot_format"):
        q = pd.qcut(sub["bid_price"], 4, labels=False, duplicates="drop")
        wr = sub.groupby(q)["won"].mean()
        ok_all &= bool(wr.iloc[-1] > wr.iloc[0])
        details.append(f"fmt{code}: {wr.iloc[0]:.3f}->{wr.iloc[-1]:.3f}")
    out["win_rises_with_bid_within_format"] = (bool(ok_all),
                                               "; ".join(details))
    return out


def report(df, cfg):
    rows = []
    m = metrics(df)
    for name, target in cfg.validation["targets"].items():
        v = m.get(name, np.nan)
        if isinstance(target, list):
            status = "in-band" if (not np.isnan(v)
                                   and target[0] <= v <= target[1]) else "off"
            tgt = f"[{target[0]}, {target[1]}]"
        else:
            status = "in-band" if (not np.isnan(v)
                                   and abs(v - target) <= 0.5 * target) \
                else "off"
            tgt = f"~{target}"
        rows.append({"metric": name, "achieved": None if np.isnan(v)
                     else round(v, 4), "target": tgt, "status": status})
    table = pd.DataFrame(rows)

    print("\n=== Target vs achieved (levels gate AFTER full autocal) ===")
    print(table.to_string(index=False))
    print("\n=== Direction checks (golden-run gates) ===")
    ok_all = True
    for name, (ok, detail) in directions(df).items():
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return table, ok_all
