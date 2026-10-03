"""Full parameter map: ALL observable + latent parameters in the defined
generative structure, in 4 columns:
  far-left   = the 3 POOLS (latents: user LU*, app LA*, campaign LC*)
  centre-left= OBSERVABLE parameters (A, B, C, D, H1, H2)
  centre-right= ESTIMANDS + AUCTION (p_*, ev_truth, LU7)
  right      = OUTCOME LABELS (E, F, G, H3, H4)

Only DEFINED edges are drawn; funnel parentage is shown as formula text in the
estimand nodes; parameters with no defined relationship (e.g. B3 ad_exchange)
are left unconnected. Pure-stdlib SVG; svg_checks-guarded.
Output: <PKG_ROOT>/Schema diagrams/Full_parameter_map.svg
"""
from xml.sax.saxutils import escape

from t9sim.diagrams.svg_checks import check_layout
from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1450, 965

TYPE = {
    "userL": ("#BBDEFB", "#1565C0"),
    "appL":  ("#C8E6C9", "#2E7D32"),
    "campL": ("#FFE0B2", "#EF6C00"),
    "obs":   ("#FFFFFF", "#546E7A"),
    "est":   ("#E1BEE7", "#6A1B9A"),
    "out":   ("#FFCDD2", "#C62828"),
}
# name: (x, y, w, h, label, type)
NODES = {
    # ── COL 1: POOLS (latents) ──
    "LU1": (45, 110, 150, 42, "LU1 archetype\n(latent hub)", "userL"),
    "LU2": (45, 162, 150, 24, "LU2 click_prop", "userL"),
    "LU3": (45, 190, 150, 24, "LU3 install_prop", "userL"),
    "LU4": (45, 218, 150, 24, "LU4 pay_prop", "userL"),
    "LU5": (45, 246, 150, 24, "LU5 spend/whale", "userL"),
    "LU6": (45, 274, 150, 24, "LU6 interest_vec", "userL"),
    "LA1": (45, 392, 150, 26, "LA1 app_quality", "appL"),
    "LA2": (45, 424, 150, 26, "LA2 audience_profile", "appL"),
    "LC1": (45, 540, 150, 26, "LC1 creative_appeal", "campL"),
    "LC2": (45, 572, 150, 26, "LC2 game_quality", "campL"),
    # ── COL 2: OBSERVABLES ──
    "A3":  (345, 112, 185, 24, "A3 os", "obs"),
    "A4":  (345, 140, 185, 24, "A4 os_version", "obs"),
    "A5":  (345, 168, 185, 24, "A5 device_type", "obs"),
    "A1":  (345, 204, 185, 24, "A1 region (untilted)", "obs"),
    "A2":  (345, 232, 185, 24, "A2 city", "obs"),
    "D1":  (345, 272, 185, 24, "D1 timestamp", "obs"),
    "D2":  (345, 300, 185, 24, "D2 hour_of_day", "obs"),
    "D3":  (345, 328, 185, 24, "D3 day_of_week", "obs"),
    "B1":  (345, 420, 185, 26, "B1 app_id", "obs"),
    "B2":  (345, 452, 185, 26, "B2 app_category", "obs"),
    "B3":  (345, 484, 185, 26, "B3 ad_exchange", "obs"),
    "B4B5":(345, 516, 185, 26, "B4/B5 slot size", "obs"),
    "B6":  (345, 548, 185, 26, "B6 slot_format", "obs"),
    "C13": (345, 632, 185, 26, "C1/C3 campaign_id", "obs"),
    "C2":  (345, 664, 185, 26, "C2 advertiser_scale", "obs"),
    "C4":  (345, 696, 185, 26, "C4 ad_genre", "obs"),
    "H1":  (345, 772, 185, 26, "H1 floor_price", "obs"),
    "H2":  (345, 806, 185, 42, "H2 bid_price\n=k_cpa·v_slot·ease·E[ltv|B2]", "obs"),
    # ── COL 3: ESTIMANDS + AUCTION ──
    "pclick":   (735, 158, 215, 34, "p_click\n← LU2·v_slot·m(r)·LA1·LC1", "est"),
    "pinstall": (735, 210, 215, 34, "p_install\n← LU3·ease(B2)·m(r)·LA1", "est"),
    "ppayer":   (735, 262, 215, 30, "p_payer ← LU4·m_pay(r)", "est"),
    "eltv":     (735, 310, 215, 30, "e_ltv ← LU5·LC2·μ(B2)", "est"),
    "evtruth":  (735, 400, 215, 40, "ev_truth\n= p_click·p_install·p_payer·e_ltv", "est"),
    "LU7":      (735, 600, 215, 30, "LU7 competing_bid", "userL"),
    # ── COL 4: OUTCOME LABELS ──
    "E":  (1105, 158, 270, 30, "E1 click", "out"),
    "F":  (1105, 210, 270, 30, "F1 install", "out"),
    "Gp": (1105, 262, 270, 30, "G1 is_payer", "out"),
    "Gl": (1105, 310, 270, 30, "G2/G3/G4 LTV", "out"),
    "H3": (1105, 596, 270, 40, "H3 won\n=[H2 ≥ max(LU7,floor)]", "out"),
    "H4": (1105, 656, 270, 40, "H4 clearing\n= H2 | LU7 | NaN", "out"),
}
GROUPS = [
    (25, 78, 270, 246, "USER POOL — latents"),
    (25, 360, 270, 100, "APP POOL — latents"),
    (25, 508, 270, 100, "CAMPAIGN POOL — latents"),
    (325, 78, 315, 282, "OBSERVABLES — user · time"),
    (325, 396, 315, 182, "OBSERVABLES — app · context"),
    (325, 608, 315, 124, "OBSERVABLES — campaign"),
    (325, 748, 315, 116, "OBSERVABLES — price"),
    (715, 120, 255, 340, "ESTIMANDS — ground truth (master only)"),
    (715, 560, 255, 100, "AUCTION"),
    (1075, 120, 330, 240, "OUTCOME LABELS — funnel"),
    (1075, 556, 330, 160, "OUTCOME LABELS — auction"),
]
# (src, dst, kind)
EDGES = [
    ("LU1", "LU2", "hier"), ("LU1", "LU3", "hier"), ("LU1", "LU4", "hier"),
    ("LU1", "LU5", "hier"), ("LU1", "LU6", "hier"),
    ("LU1", "A3", "tilt"), ("LU1", "A5", "tilt"),
    ("A1", "A2", "nest"), ("A3", "A4", "nest"), ("D1", "D2", "nest"), ("D1", "D3", "nest"),
    ("B1", "LA1", "key"), ("B1", "LA2", "key"),
    ("C13", "LC1", "key"), ("C13", "LC2", "key"),
    # ── v7 BUILD edges (this plan #1-#5) ──
    ("LA2", "LU1", "pairing"),                 # #1 user<->app pairing (build LA2)
    ("LU5", "LC2", "expose"),                  # #2 value-optimised campaign exposure
    ("A3", "eltv", "v7"),                      # #3 platform os -> spend
    ("D2", "ppayer", "v7"), ("D3", "ppayer", "v7"),   # #4 payer-timing
    ("LU1", "D2", "v7"),                       # #5 archetype -> hour
    ("LU2", "pclick", "funnel"), ("LU3", "pinstall", "funnel"),
    ("LU4", "ppayer", "funnel"), ("LU5", "eltv", "funnel"),
    ("pclick", "E", "funnel"), ("pinstall", "F", "funnel"),
    ("ppayer", "Gp", "funnel"), ("eltv", "Gl", "funnel"),
    ("pclick", "evtruth", "ev"), ("pinstall", "evtruth", "ev"),
    ("ppayer", "evtruth", "ev"), ("eltv", "evtruth", "ev"),
    ("evtruth", "LU7", "ev"),
    ("LU7", "H3", "auction"), ("LU7", "H4", "auction"),
]
EKIND = {
    "hier":     ("#1565C0", "1.5", "",    "archetype -> value latents (hierarchy)"),
    "tilt":     ("#C62828", "2.4", "",    "CPT tilt archetype -> A-feature (WIRED)"),
    "v7":       ("#00897B", "2.4", "",    "v7 BUILD: os->spend, timing, hour (#3-5)"),
    "expose":   ("#00897B", "2.4", "",    "#2 value->LC2 exposure (v7 BUILD)"),
    "pairing":  ("#00897B", "2.4", "",    "#1 user<->app pairing (v7 BUILD)"),
    "nest":     ("#9E9E9E", "1.3", "",    "nesting / derivation"),
    "key":      ("#616161", "1.6", "7,4", "key lookup: id -> entity latent"),
    "funnel":   ("#00838F", "1.6", "",    "funnel: latent -> estimand -> outcome"),
    "ev":       ("#8E24AA", "1.8", "",    "EV chain -> competing bid"),
    "auction":  ("#EF6C00", "1.8", "",    "auction -> won / clearing"),
}


def _c(n):
    x, y, w, h = NODES[n][:4]
    return x + w / 2, y + h / 2, w, h


def _edge(a, b, kind):
    ax, ay, aw, ah = _c(a)
    bx, by, bw, bh = _c(b)
    if abs(bx - ax) < 6:                       # vertical-ish
        sx = ex = ax
        sy = ay + ah / 2 if by > ay else ay - ah / 2
        ey = by - bh / 2 if by > ay else by + bh / 2
    else:
        sx = ax + aw / 2 if bx > ax else ax - aw / 2
        ex = bx - bw / 2 if bx > ax else bx + bw / 2
        sy, ey = ay, by
    col, wdt, dash, _ = EKIND[kind]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" '
            f'stroke="{col}" stroke-width="{wdt}"{da} marker-end="url(#a{kind})"/>')


def _pairing(a, b, offset=24):                 # left-margin elbow (cross-pool latent edge)
    ax, ay, aw, _ = _c(a)
    bx, by, bw, _ = _c(b)
    lx = min(ax - aw / 2, bx - bw / 2) - offset
    return (f'<polyline points="{ax-aw/2:.0f},{ay:.0f} {lx:.0f},{ay:.0f} '
            f'{lx:.0f},{by:.0f} {bx-bw/2:.0f},{by:.0f}" fill="none" '
            f'stroke="#00897B" stroke-width="2.2" marker-end="url(#apairing)"/>')


def render():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
    s.append('<defs>')
    for k, (col, *_r) in EKIND.items():
        s.append(f'<marker id="a{k}" markerWidth="9" markerHeight="9" refX="7.5" '
                 f'refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{col}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    s.append(f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="21" '
             f'font-weight="700" fill="#212121">T9 full parameter map (v7 build plan) — '
             f'pools &#8594; observables &#8594; estimands/auction &#8594; outcomes</text>')
    s.append(f'<text x="{W/2:.0f}" y="52" text-anchor="middle" font-size="11.5" '
             f'fill="#546E7A">teal = v7 BUILD edges #1-#5 (proposed); funnel parentage in '
             f'estimand-node text; B3 etc. left unconnected</text>')
    for gx, gy, gw, gh, lab in GROUPS:
        s.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" '
                 f'fill="#F7F9FB" stroke="#CFD8DC" stroke-width="1.2"/>')
        s.append(f'<text x="{gx+12}" y="{gy+18}" font-size="11.5" font-weight="700" '
                 f'fill="#455A64">{escape(lab)}</text>')
    for a, b, k in EDGES:
        if k == "pairing":
            s.append(_pairing(a, b, 24))
        elif k == "expose":
            s.append(_pairing(a, b, 40))
        else:
            s.append(_edge(a, b, k))
    for n, (x, y, w, h, lab, t) in NODES.items():
        fill, stroke = TYPE[t]
        sw = 2.4 if n == "LU1" else 1.3
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = lab.split("\n")
        y0 = y + h / 2 - (len(lines) - 1) * 6.5 + 3.5
        for i, ln in enumerate(lines):
            fw = "700" if (i == 0 and t in ("userL", "est", "out")) else "400"
            fs = 11 if i == 0 else 8.5
            s.append(f'<text x="{x+w/2:.0f}" y="{y0+i*12:.0f}" text-anchor="middle" '
                     f'font-size="{fs}" font-weight="{fw}" fill="#1A1A1A">{escape(ln)}</text>')
    items = list(EKIND.items())
    lx, ly = 30, H - 34
    for i, (k, (col, wdt, dash, txt)) in enumerate(items):
        if i == 5:
            lx, ly = 30, H - 16
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{col}" '
                 f'stroke-width="2.2"{da}/>')
        s.append(f'<text x="{lx+30}" y="{ly+4}" font-size="9.5" fill="#37474F">{escape(txt)}</text>')
        lx += 36 + len(txt) * 5.9
    s.append('</svg>')
    check_layout(NODES, GROUPS, name="Full_parameter_map.svg")
    (OUT / "Full_parameter_map.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / "Full_parameter_map.svg")


if __name__ == "__main__":
    render()
