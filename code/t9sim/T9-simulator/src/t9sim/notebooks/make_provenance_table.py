"""Emit the schema's Data-source audit-trail table (markdown) from the live
provenance registry, grouping per-archetype / per-category leaves into one
family row each. Output: stdout + calibration/provenance_table.md
"""
import re

from t9sim.config_loader import Config
from t9sim.paths import CALIBRATION_DIR

# family-grouping rules: (regex on param path) -> family key
RULES = [
    (r"^benchmarks\.funnel\.w_stage\.", "funnel.w_stage"),
    (r"^benchmarks\.funnel\.install_delay\.", "funnel.install_delay"),
    (r"^benchmarks\.app_categories\.[^.]+\.share$", "app_categories.share"),
    (r"^benchmarks\.app_categories\.[^.]+\.install_ease$",
     "app_categories.install_ease"),
    (r"^benchmarks\.app_categories\.[^.]+\.ltv_multiplier$",
     "app_categories.ltv_multiplier"),
    (r"^benchmarks\.ad_genre_mix\.", "ad_genre_mix"),
    (r"^benchmarks\.advertiser_scale\.[^.]+\.advertiser_share$",
     "advertiser_scale.advertiser_share"),
    (r"^benchmarks\.advertiser_scale\.[^.]+\.campaign_share$",
     "advertiser_scale.campaign_share"),
    (r"^benchmarks\.advertiser_scale\.[^.]+\.k_cpa$",
     "advertiser_scale.k_cpa"),
    (r"^benchmarks\.ad_exchanges\.", "ad_exchanges"),
    (r"^benchmarks\.slot_format_shares\.", "slot_format_shares"),
    (r"^benchmarks\.slot_sizes\.banner\.", "slot_sizes.banner"),
    (r"^benchmarks\.slot_sizes\.interstitial\.", "slot_sizes.interstitial"),
    (r"^benchmarks\.slot_sizes\.rewarded\.", "slot_sizes.rewarded"),
    (r"^benchmarks\.slot_quality\.format_weight\.",
     "slot_quality.format_weight"),
    (r"^benchmarks\.slot_quality\.size_weight\.",
     "slot_quality.size_weight"),
    (r"^benchmarks\.ecpm_targets_usd\.", "ecpm_targets_usd"),
    (r"^benchmarks\.device\.os_split\.", "device.os_split"),
    (r"^benchmarks\.device\.device_split\.", "device.device_split"),
    (r"^benchmarks\.device\.os_versions\.iOS\.", "device.os_versions.iOS"),
    (r"^benchmarks\.device\.os_versions\.Android\.",
     "device.os_versions.Android"),
    (r"^archetypes\.shares\.", "archetypes.shares (LU1)"),
    (r"^archetypes\.propensity\.[^.]+\.click$", "propensity.click (LU2)"),
    (r"^archetypes\.propensity\.[^.]+\.install$",
     "propensity.install (LU3)"),
    (r"^archetypes\.propensity\.[^.]+\.payer$", "propensity.payer (LU4)"),
    (r"^archetypes\.ltv_multiplier\.[^.]+\.mu$", "ltv_multiplier.mu (LU5)"),
    (r"^archetypes\.ltv_multiplier\.[^.]+\.sigma$",
     "ltv_multiplier.sigma (LU5)"),
    (r"^archetypes\.interest\.centroids\.", "interest.centroids (LU6)"),
]

USE = {  # family/param -> schema feature(s) it drives
    "funnel.base_ctr": "E1 P(click)", "funnel.base_ir": "F1 P(install|click)",
    "funnel.base_payer": "G1 is_payer", "funnel.w_stage": "m_stage(r), E1/F1/G1",
    "funnel.click_delay_mean_s": "E2", "funnel.install_delay": "F2",
    "ltv.base_median_usd": "G2 anchor", "ltv.base_mean_usd": "G2 anchor",
    "ltv.lognormal_mu": "G2 base LogNormal", "ltv.lognormal_sigma": "G2 base LogNormal",
    "ltv.decay_d7": "G3", "ltv.decay_d30": "G4",
    "app_categories.share": "B2 mix", "app_categories.install_ease": "F1, H2 (ease)",
    "app_categories.ltv_multiplier": "G2 mu_cat, H2",
    "ad_genre_mix": "C4 mix", "advertiser_scale.advertiser_share": "C2 pool counts",
    "advertiser_scale.campaign_share": "C2 campaign volume (~spend)",
    "advertiser_scale.k_cpa": "H2 bid level",
    "ad_exchanges": "B3 mix", "slot_format_shares": "B6 mix",
    "slot_sizes.banner": "B4xB5 | banner", "slot_sizes.interstitial": "B4xB5 | interstitial",
    "slot_sizes.rewarded": "B4xB5 | rewarded",
    "slot_quality.format_weight": "v_slot (E1, H2)", "slot_quality.size_weight": "v_slot (E1, H2)",
    "ecpm_targets_usd": "H1/H4/LU7 price levels",
    "auction.gamma": "LU7 value-sensitivity", "auction.sigma_g": "LU7 gaming noise",
    "auction.mu_x": "LU7 non-gaming demand", "auction.sigma_x": "LU7 non-gaming noise",
    "auction.shade": "H2 shading", "auction.sigma_explore": "H2 exploration",
    "auction.k_global": "H2 level (win-rate pin)",
    "device.os_split": "A3", "device.device_split": "A5",
    "device.os_versions.iOS": "A4 | iOS", "device.os_versions.Android": "A4 | Android",
    "entity_latents.sigma_app": "LA1", "entity_latents.sigma_cre": "LC1",
    "entity_latents.sigma_game": "LC2", "entity_latents.audience_dirichlet_k": "LA2 (inert)",
    "archetypes.shares (LU1)": "LU1 archetype mix",
    "propensity.click (LU2)": "LU2 Beta(a,b)", "propensity.install (LU3)": "LU3 Beta(a,b)",
    "propensity.payer (LU4)": "LU4 Beta(a,b)",
    "ltv_multiplier.mu (LU5)": "LU5 LogNormal mu", "ltv_multiplier.sigma (LU5)": "LU5 LogNormal sigma",
    "interest.concentration_k": "LU6 Dirichlet k", "interest.centroids (LU6)": "LU6 centroids",
    "time.window_days": "D1 window", "prices.paying_shape_csv": "base_eCPM shape (IP)",
    "prices.floor_shape_csv": "H1 shape (IP)",
}


def family(path):
    for pat, fam in RULES:
        if re.match(pat, path):
            return fam
    return path.replace("benchmarks.", "").replace("archetypes.", "")


def label(path, fam):
    """The distinguishing token for grouped rows: the LAST path token that
    is not shared with the family key (so sizes/versions beat formats/os)."""
    fam_toks = set(re.split(r"[.\s(]", fam))
    toks = path.split(".")
    i = len(toks) - 1
    while i >= 0:
        tok = toks[i]
        if tok == "x" and i > 0:           # version keys like "18.x" split
            tok = toks[i - 1] + ".x"
            i -= 1
        if tok not in fam_toks and tok not in ("benchmarks", "archetypes"):
            return tok
        i -= 1
    return toks[-1]


def clean(s):
    return str(s).replace("|", "/")        # pipes break the GFM table


def fmt_val(v):
    if isinstance(v, list):
        return "/".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


cfg = Config("golden")
df = cfg.provenance.copy()
df["family"] = df["param"].map(family)

rows = []
for fam, grp in df.groupby("family", sort=False):
    if len(grp) == 1:
        val = fmt_val(grp.iloc[0]["value"])
    else:
        val = " · ".join(f"{label(p, fam)} {fmt_val(v)}"
                         for p, v in zip(grp["param"], grp["value"]))
    routes = sorted(grp["route"].unique())
    route = routes[0] if len(routes) == 1 else " + ".join(routes)
    src = max(grp["source"], key=len)
    if len(src) > 110:
        src = src[:107] + "..."
    if len(val) > 95:
        val = val[:92] + "..."
    rows.append((clean(fam), clean(USE.get(fam, "")), clean(val),
                 route, clean(src), len(grp)))

lines = ["| # | Parameter (family) | Drives | Value(s) | Route | Source |",
         "|---|---|---|---|---|---|"]
for i, (fam, use, val, route, src, n) in enumerate(rows, 1):
    fam_disp = f"{fam}" + (f" ({n} leaves)" if n > 1 else "")
    lines.append(f"| {i} | `{fam_disp}` | {use} | {val} | {route} | {src} |")

md = "\n".join(lines)
out = CALIBRATION_DIR / "provenance_table.md"
out.write_text(md, encoding="utf-8")
print(f"{len(rows)} family rows from {len(df)} leaf parameters")
print(md[:1500])
print(f"...\nwritten to {out}")
