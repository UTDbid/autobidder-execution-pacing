"""Generate the BN dependency-layer CPTs (the DESIGNED tilt family).

For every archetype -> observable-feature edge we build a conditional table
P(feature | archetype) that is MARGINAL-PRESERVING: sum_k pi_k * P(v|k) = m_v,
where m_v is the feature's existing calibrated marginal (config / iPinYou) and
pi_k the archetype shares. So switching the BN on never disturbs calibration —
that is exactly what the validation row in each table proves.

Method: start from per-archetype multiplicative tilt weights w_{k,v} (domain
judgment; von Mises for hour), seed U = m_v * w_{k,v}, then ITERATIVE
PROPORTIONAL FITTING (raking): alternately rescale columns so the pi-weighted
column mass equals m_v, and renormalise each row to sum 1. Converges to the
unique table satisfying both constraints. tilts -> 0 recovers the independent
(current) generator exactly.

Edges WITHOUT a designed CPT:
  - id -> entity latent (app_id->LA*, campaign_id->LC*): key lookup, not a tilt.
  - archetype -> LU2..LU6: continuous (Beta/LogNormal/Dirichlet) — parameter
    tables, summarised at the end, not discrete CPTs.
  - A2 city|A1 region, A4 os_version|A3 os: nesting CPTs inherited unchanged
    from iPinYou/StatCounter (os_version tabulated; city too high-cardinality).

Outputs:  ../../BN_CPTs.md   and   ../config/bn_cpts.yaml
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from t9sim.paths import CALIBRATION_DIR, CONFIG_DIR, PKG_ROOT

ROOT = PKG_ROOT.parent          # project root, for BN_CPTs.md
CFG = CONFIG_DIR
CAL = CALIBRATION_DIR

ARCH = ["whale", "engaged_spender", "casual", "time_filler", "inactive"]
ALABEL = {"whale": "whale", "engaged_spender": "engaged", "casual": "casual",
          "time_filler": "time_filler", "inactive": "inactive"}


def _load_yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _val(node):
    return node["value"] if isinstance(node, dict) and "value" in node else node


# ── inputs: shares + marginals ───────────────────────────────────────
arche = _load_yaml(CFG / "archetypes.yaml")
bench = _load_yaml(CFG / "benchmarks.yaml")
PI = np.array([_val(arche["shares"][a]) for a in ARCH], float)
PI = PI / PI.sum()

OS_M = bench["device"]["os_split"]
M_OS = {"iOS": _val(OS_M["iOS"]), "Android": _val(OS_M["Android"])}
DEV_M = bench["device"]["device_split"]
M_DEV = {"phone": _val(DEV_M["phone"]), "tablet": _val(DEV_M["tablet"])}

# hour & day-of-week base shape from the iPinYou hour x dow distribution
_hh = defaultdict(float)
_dd = defaultdict(float)
with open(CAL / "ipinyou_hourdow_distribution.csv", newline="") as f:
    for r in csv.DictReader(f):
        p = float(r["proportion"])
        _hh[int(r["hour"])] += p
        _dd[int(r["dow"])] += p
HOURS = list(range(24))
M_HOUR = np.array([_hh[h] for h in HOURS]); M_HOUR /= M_HOUR.sum()
DOWS = list(range(7))
DOW_LAB = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]   # our Mon=0 convention
M_DOW = np.array([_dd[d] for d in DOWS]); M_DOW /= M_DOW.sum()
WEEKEND = {5, 6}

# region -> 3 tiers by share mass (PLACEHOLDER metro/mid/rural proxy)
reg = []
with open(CAL / "ipinyou_region_distribution.csv", newline="") as f:
    for r in csv.DictReader(f):
        reg.append(float(r["proportion"]))
reg = sorted(reg, reverse=True)
s = sum(reg); cum = 0.0; tiers = [0.0, 0.0, 0.0]
for p in reg:
    t = 0 if cum < s / 3 else (1 if cum < 2 * s / 3 else 2)
    tiers[t] += p; cum += p
M_REG = np.array(tiers); M_REG /= M_REG.sum()
REG_LAB = ["tier1_metro", "tier2_mid", "tier3_rural"]


# ── IPF (raking) to a marginal-preserving CPT ────────────────────────
def ipf(weights, marg, pi=PI, iters=2000, tol=1e-12):
    """weights: (K,V) multiplicative tilt; marg: (V,) target marginal.
    Returns P (K,V): rows sum to 1 AND pi @ P == marg."""
    U = marg[None, :] * weights
    P = U / U.sum(1, keepdims=True)
    for _ in range(iters):
        mc = pi @ P                       # current pi-weighted column mass
        P = P * (marg / mc)[None, :]      # column step
        P = P / P.sum(1, keepdims=True)   # row step
        if np.max(np.abs(pi @ P - marg)) < tol:
            break
    return P


def logit_w(tilts):
    """per-archetype scalar tilt -> 2-col weights [exp(t), 1] (level0 favoured)."""
    return np.array([[np.exp(t), 1.0] for t in tilts])


# ── designed tilts (domain judgment; route: inferred, sensitivity-tested) ──
# OS: iOS skews higher-ARPU segments (well documented iOS>Android ARPU)
T_OS = {"whale": 0.80, "engaged_spender": 0.50, "casual": 0.05,
        "time_filler": -0.25, "inactive": -0.20}
# device: tablets skew dedicated/older higher-spend gamers (mild)
T_DEV = {"whale": 0.50, "engaged_spender": 0.40, "casual": 0.0,
         "time_filler": -0.15, "inactive": -0.10}
# day-of-week PLAY tilt: MODEST only — sources (#30 GameAnalytics) show weekly
# PLAY is near-flat (Sat 15.9% vs ~14% weekday). The strong weekend SPEND skew
# is NOT here; it is the separate t_pay(hour,dow) multiplier on p_payer (edge #4).
T_WKND = {"whale": 0.12, "engaged_spender": 0.08, "casual": 0.0,
          "time_filler": -0.05, "inactive": -0.03}
# hour: von Mises peak (mu) + concentration (kappa) per archetype
VM = {"whale": (22, 0.60), "engaged_spender": (21, 0.50), "casual": (20, 0.30),
      "time_filler": (12, 0.40), "inactive": (0.0, 0.0)}
# region tier: spenders skew metro tier-1 (PLACEHOLDER mapping)
T_REG = {"whale": [0.55, 0.0, -0.45], "engaged_spender": [0.40, 0.0, -0.35],
         "casual": [0.05, 0.0, -0.05], "time_filler": [-0.15, 0.0, 0.12],
         "inactive": [-0.10, 0.0, 0.08]}


def build():
    out = {}
    # OS  (iOS, Android)
    w = logit_w([T_OS[a] for a in ARCH])
    out["os"] = (["iOS", "Android"], np.array([M_OS["iOS"], M_OS["Android"]]),
                 ipf(w, np.array([M_OS["iOS"], M_OS["Android"]])))
    # device (phone, tablet) -> tilt favours level0=phone? we tilt tablet, so
    # put tablet as level0 of the logit, then reorder to [phone,tablet]
    wt = logit_w([T_DEV[a] for a in ARCH])           # [tablet, phone-ref]
    w = np.column_stack([wt[:, 1], wt[:, 0]])         # [phone, tablet]
    out["device_type"] = (["phone", "tablet"],
                          np.array([M_DEV["phone"], M_DEV["tablet"]]),
                          ipf(w, np.array([M_DEV["phone"], M_DEV["tablet"]])))
    # day-of-week (7): weekend uplift
    w = np.ones((len(ARCH), 7))
    for i, a in enumerate(ARCH):
        for d in DOWS:
            if d in WEEKEND:
                w[i, d] = np.exp(T_WKND[a])
    out["day_of_week"] = (DOW_LAB, M_DOW, ipf(w, M_DOW))
    # hour (24): von Mises
    w = np.ones((len(ARCH), 24))
    for i, a in enumerate(ARCH):
        mu, k = VM[a]
        for h in HOURS:
            w[i, h] = np.exp(k * np.cos(2 * np.pi * (h - mu) / 24))
    out["hour_of_day"] = ([str(h) for h in HOURS], M_HOUR, ipf(w, M_HOUR))
    # region tier (3): additive-style tilt via exp()
    w = np.array([[np.exp(t) for t in T_REG[a]] for a in ARCH])
    out["region_tier"] = (REG_LAB, M_REG, ipf(w, M_REG))
    return out


def md_table(name, levels, marg, P):
    err = float(np.max(np.abs(PI @ P - marg)))
    cols = " | ".join(levels)
    lines = [f"### P({name} | archetype)\n",
             f"| archetype (pi) | {cols} |",
             "|" + "---|" * (len(levels) + 1)]
    for i, a in enumerate(ARCH):
        row = " | ".join(f"{P[i, j]:.3f}" for j in range(len(levels)))
        lines.append(f"| {ALABEL[a]} ({PI[i]:.2f}) | {row} |")
    recov = " | ".join(f"{(PI @ P)[j]:.3f}" for j in range(len(levels)))
    targ = " | ".join(f"{marg[j]:.3f}" for j in range(len(levels)))
    lines.append(f"| **pi-weighted (recovered)** | {recov} |")
    lines.append(f"| **target marginal** | {targ} |")
    lines.append(f"\n*marginal-preservation max error = {err:.2e}*\n")
    return "\n".join(lines), err


def main():
    tables = build()
    maxerr = 0.0
    md = ["# T9 — BN dependency-layer CPTs (v7 — active spec)\n\n"
          "> **Status (v7).** The BN dependency layer is the active spec "
          "(`T9_Simulator_Schema (v7)`); edges #1/#2 pending Dr Klimis sign-off. "
          "Tilt CPTs below are resolved; tabulated values are rounded summaries "
          "— `config/bn_cpts.yaml` (`resolved`) is canonical. Archetype "
          "`engaged` = config `engaged_spender` (display alias).\n",
          "Conditional tables for the **designed tilt family** "
          "(archetype LU1 -> observable feature). Every table is "
          "**marginal-preserving**: the `pi-weighted (recovered)` row equals the "
          "`target marginal` row, so enabling the BN does not change any "
          "calibrated marginal. Marginals are the existing generation values "
          "(config / iPinYou); per-archetype tilts are **inferred / "
          "sensitivity-tested** design priors. Setting all tilts to 0 recovers "
          "the current independent generator.\n",
          f"Archetype shares pi = " +
          ", ".join(f"{ALABEL[a]} {PI[i]:.2f}" for i, a in enumerate(ARCH)) + ".\n"]
    # region_tier DROPPED (research 17 Jun 2026): anonymised iPinYou codes are
    # 35 Chinese provinces; no defensible affluence/urbanicity lookup exists under
    # the US-2025 reframing, and archetype<->region coupling is only supported at
    # country level. Region stays an UNTILTED marginal (independent of archetype).
    order = ["os", "device_type", "day_of_week", "hour_of_day"]
    names = {"os": "A3 os", "device_type": "A5 device_type",
             "day_of_week": "D3 day_of_week",
             "hour_of_day": "D2 hour_of_day (Compact B, von Mises)"}
    for key in order:
        levels, marg, P = tables[key]
        block, err = md_table(names[key], levels, marg, P)
        maxerr = max(maxerr, err)
        md.append(block)

    # region tilt dropped (research)
    md.append("### A1 region — tilt DROPPED (kept untilted / independent)\n")
    md.append("Region is **not** archetype-tilted. The 35 codes are anonymised iPinYou "
              "*Chinese provinces*; under the US-2025 reframing no defensible "
              "affluence/urbanicity lookup exists, and archetype<->region coupling is only "
              "supported at country level. Region stays an independent marginal (drawn as "
              "today). *(Research 17 Jun 2026.)*\n")

    # inherited nesting CPT: P(os_version | os) — NOT archetype-tilted
    osv = bench["device"]["os_versions"]
    md.append("### P(A4 os_version | A3 os)  — inherited nesting (NOT tilted)\n")
    for parent in ("iOS", "Android"):
        vers = {k: _val(v) for k, v in osv[parent].items()}
        cells = " · ".join(f"{k} {v:.2f}" for k, v in vers.items())
        md.append(f"- **{parent}:** {cells}")
    md.append("\n*A2 city | A1 region is likewise inherited (iPinYou nested "
              "lookup); too high-cardinality to tabulate.*\n")

    # latent conditional tables (continuous) — reference
    md.append("## Latent conditional tables (continuous — parameters, not CPTs)\n")
    md.append("Archetype -> LU2..LU6 are conditional *distributions*, not discrete "
              "CPTs. Means/parameters per archetype (from `archetypes.yaml`):\n")
    prop = arche["propensity"]
    md.append("| archetype | P(click) LU2 | P(install) LU3 | P(payer) LU4 | "
              "LU5 LTV mult (median) |")
    md.append("|---|---|---|---|---|")
    ltm = arche["ltv_multiplier"]
    for a in ARCH:
        def beta_mean(pair):
            x = _val(pair); return x[0] / (x[0] + x[1])
        c = beta_mean(prop[a]["click"]); ins = beta_mean(prop[a]["install"])
        pay = beta_mean(prop[a]["payer"])
        if a == "inactive":
            mult = "0 (structural non-spender)"
        else:
            mu = _val(ltm[a]["mu"]); mult = f"{np.exp(mu):.2f}x"
        md.append(f"| {ALABEL[a]} | {c:.2f} | {ins:.2f} | {pay:.2f} | {mult} |")
    md.append("\n*LU6 interest_vector ~ Dirichlet(k=20 * centroid) per archetype "
              "(centroids in archetypes.yaml).*\n")

    (ROOT / "BN_CPTs.md").write_text("\n".join(md), encoding="utf-8")

    # YAML: tilt specs + resolved probabilities (generator-consumable)
    ydoc = {"_note": "BN tilt CPTs (v7 active). marginal-preserving via IPF; "
                     "tilts are inferred/sensitivity-tested. tilts=0 -> current generator.",
            "archetype_shares": {a: float(PI[i]) for i, a in enumerate(ARCH)},
            "region_tier": "DROPPED (research 17 Jun 2026): codes are 35 Chinese "
                           "provinces; no defensible affluence lookup under US reframe; "
                           "region stays untilted (independent of archetype).",
            "tilts": {"os_ios_logit": T_OS, "device_tablet_logit": T_DEV,
                      "weekend_logit": T_WKND,
                      "hour_vonmises_mu_kappa": {a: list(VM[a]) for a in ARCH}},
            "resolved": {}}
    for key in order:
        levels, marg, P = tables[key]
        ydoc["resolved"][key] = {
            "levels": levels,
            "marginal": [float(x) for x in marg],
            "cpt": {a: [round(float(P[i, j]), 5) for j in range(len(levels))]
                    for i, a in enumerate(ARCH)}}
    (CFG / "bn_cpts.yaml").write_text(
        yaml.safe_dump(ydoc, sort_keys=False, default_flow_style=False),
        encoding="utf-8")

    print(f"wrote {ROOT / 'BN_CPTs.md'}")
    print(f"wrote {CFG / 'bn_cpts.yaml'}")
    print(f"max marginal-preservation error across all CPTs = {maxerr:.2e}")
    assert maxerr < 1e-6, "marginal not preserved!"
    print("OK — all CPTs marginal-preserving.")


if __name__ == "__main__":
    main()
