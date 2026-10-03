"""FN-1 auto-calibration loop (route: auto-calibrated).

Solves the calibration knobs TOP-DOWN the funnel - each knob changes its own
stage and the population reaching later stages, never upstream, so the
dependency is triangular and a top-down pass converges:

  1. base_ctr    -> population CTR        (1-D rescale, target band mid 0.035)
  2. base_ir     -> click->install        (1-D rescale, mid 0.325)
  3. base_payer  -> install->payer        (1-D rescale, mid 0.035)
  4. ltv mu      -> median payer spend $6 (1-D in log space)
     ltv sigma   -> whale top-5% share    (bisection, target 0.60)
     (mu and sigma are nearly decoupled: a lognormal's median is independent
      of sigma, so sigma moves the whale share without disturbing the median)

Only parameters tagged `route: auto-calibrated` move; quoted/inferred values
stay fixed - the benchmarks constrain the OUTPUTS, the knobs absorb the
adjustment. Solved values are written to config/calibrated.yaml (merged over
benchmarks.yaml by config_loader at load time, keeping the base file and its
provenance comments untouched).

Usage:  python -m t9sim.autocal
"""
from __future__ import annotations

import time
from datetime import date

import numpy as np
import yaml

from . import user_profiles
from .auctions import AuctionEngine
from .catalogue import build_apps, build_campaigns
from .config_loader import Config
from .paths import CONFIG_DIR

TARGETS = {
    "population_ctr": 0.035,        # band [0.02, 0.05]
    "click_to_install": 0.325,      # band [0.25, 0.40]
    "install_to_payer": 0.035,      # band [0.02, 0.05]
    "median_payer_spend_usd": 6.0,
    "whale_concentration": 0.60,    # band [0.55, 0.65]
}


def _whale_share(spend):
    s = np.sort(spend[spend > 0])[::-1]
    top = max(1, int(np.ceil(0.05 * len(s))))
    return float(s[:top].sum() / s.sum())


def run():
    t0 = time.time()
    cfg = Config("test")            # 220k-user pools for payer diversity
    bm = cfg.benchmarks
    before = {k: bm["funnel"][k] for k in ("base_ctr", "base_ir",
                                           "base_payer")}
    before["mu"], before["sigma"] = bm["ltv"]["lognormal_mu"], \
        bm["ltv"]["lognormal_sigma"]

    users = user_profiles.generate_users(cfg)
    apps, camps = build_apps(cfg), build_campaigns(cfg)

    def engine():
        return AuctionEngine(cfg, users, apps, camps)

    eng = engine()
    it_count = 0

    def sample(n):
        nonlocal it_count
        it_count += 1
        return eng.funnel_sample(n, cfg.rng(f"autocal-{it_count}"))

    # ── 1-3: funnel rate knobs (iterated 1-D rescales; clip-aware) ──
    for knob, metric, n in (("base_ctr", "ctr", 300_000),
                            ("base_ir", "c2i", 400_000),
                            ("base_payer", "i2p", 700_000)):
        for _ in range(4):
            s = sample(n)
            if metric == "ctr":
                achieved = s["click"].mean()
                target = TARGETS["population_ctr"]
            elif metric == "c2i":
                achieved = s.loc[s["click"], "install"].mean()
                target = TARGETS["click_to_install"]
            else:
                achieved = s.loc[s["install"], "payer"].mean()
                target = TARGETS["install_to_payer"]
            ratio = target / max(achieved, 1e-9)
            bm["funnel"][knob] *= ratio
            if abs(ratio - 1) < 0.02:
                break
        print(f"  {knob}: {before[knob]:.4g} -> {bm['funnel'][knob]:.4g} "
              f"(achieved {achieved:.4f}, target {target})")

    # ── 4a: ltv mu -> median payer spend (log-space 1-D) ───────────
    # large samples: whale-share & median are heavy-tail quantities estimated
    # from few payers (payer rate ~4e-4), so use 5M draws + many bisection
    # iters or the sigma solve is noise-dominated (whale share is very
    # sensitive to sigma in the lognormal tail).
    LTV_SAMPLE = 5_000_000
    for _ in range(3):
        s = sample(LTV_SAMPLE)
        med = float(np.median(s.loc[s["payer"], "spend"]))
        bm["ltv"]["lognormal_mu"] += float(np.log(
            TARGETS["median_payer_spend_usd"] / med))
        eng = engine()                    # mu_cat / eltv tables are cached
        if abs(np.log(TARGETS["median_payer_spend_usd"] / med)) < 0.03:
            break
    print(f"  ltv mu: {before['mu']:.4g} -> {bm['ltv']['lognormal_mu']:.4g} "
          f"(median ${med:.2f})")

    # ── 4b: ltv sigma -> whale top-5% share (bisection) ────────────
    lo, hi = 0.6, 2.8
    for _ in range(12):
        mid = (lo + hi) / 2
        bm["ltv"]["lognormal_sigma"] = mid
        eng = engine()
        s = sample(LTV_SAMPLE)
        share = _whale_share(s.loc[s["payer"], "spend"].to_numpy())
        if share > TARGETS["whale_concentration"]:
            hi = mid
        else:
            lo = mid
    bm["ltv"]["lognormal_sigma"] = (lo + hi) / 2
    print(f"  ltv sigma: {before['sigma']:.4g} -> "
          f"{bm['ltv']['lognormal_sigma']:.4g} (whale share {share:.3f})")

    # sigma moved -> re-trim the median once (median is sigma-independent
    # in theory; the finite-sample re-measure is a safety pass)
    eng = engine()
    s = sample(LTV_SAMPLE)
    med = float(np.median(s.loc[s["payer"], "spend"]))
    bm["ltv"]["lognormal_mu"] += float(np.log(
        TARGETS["median_payer_spend_usd"] / med))

    # ── write config/calibrated.yaml ────────────────────────────────
    stamp = date.today().isoformat()
    note = lambda what, tgt: (f"SOLVED by autocal {stamp} against "  # noqa: E731
                              f"{what} target {tgt} (route: auto-calibrated)")
    solved = {
        "benchmarks.funnel.base_ctr": {
            "value": round(float(bm["funnel"]["base_ctr"]), 6),
            "note": note("population CTR", "0.02-0.05")},
        "benchmarks.funnel.base_ir": {
            "value": round(float(bm["funnel"]["base_ir"]), 6),
            "note": note("click->install", "0.25-0.40")},
        "benchmarks.funnel.base_payer": {
            "value": round(float(bm["funnel"]["base_payer"]), 6),
            "note": note("install->payer", "0.02-0.05")},
        "benchmarks.ltv.lognormal_mu": {
            "value": round(float(bm["ltv"]["lognormal_mu"]), 6),
            "note": note("median payer spend", "$6")},
        "benchmarks.ltv.lognormal_sigma": {
            "value": round(float(bm["ltv"]["lognormal_sigma"]), 6),
            "note": note("whale top-5% share", "0.55-0.65")},
    }
    out = CONFIG_DIR / "calibrated.yaml"
    out.write_text(
        "# Written by t9sim.autocal - FN-1 solved calibration knobs.\n"
        "# Merged over benchmarks.yaml by config_loader; regenerate by\n"
        "# re-running `python -m t9sim.autocal`. Only route:auto-calibrated "
        "params appear.\n"
        + yaml.safe_dump({"meta": {"date": stamp, "samples": it_count},
                          "solved": solved}, sort_keys=False),
        encoding="utf-8")
    print(f"\nwrote {out} ({time.time() - t0:.0f}s, {it_count} samples)")


if __name__ == "__main__":
    run()
