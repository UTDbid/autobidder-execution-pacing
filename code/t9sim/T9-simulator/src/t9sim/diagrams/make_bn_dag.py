"""BN dependency-layer DAG (proposed) — observable <-> latent coupling.

Two distinct edge families:
  (1) CPT tilt (generative archetype -> feature): LU1 archetype is the parent of
      A3 os / A5 device_type (red) and, as Compact B (proposed), of D2 hour /
      D3 day-of-week via a per-archetype von Mises active-hour prior (magenta).
      A1 region is UNTILTED (tilt dropped 17 Jun 2026 — anonymised iPinYou codes
      are Chinese provinces; no defensible affluence lookup under the US reframe).
      Marginal-preserving; the Bayes inverse feature->archetype is the predictive signal.
  (2) key lookup (id -> entity latent): app_id (B1) and campaign_id (C1/C3) are KEYS
      to per-entity latents drawn once at pool build (LA1/LA2, LC1/LC2) — not tilts.
LA2 couples user<->app pairing. Existing latent->outcome / nesting edges shown faded.

v7 BUILD PLAN edges (teal, #1-#5 — BN_Build_Plan.md): #1 LA2 user<->app pairing,
#2 LU value-latents -> LC2 campaign exposure, #3 os -> spend, #4 D2/D3 -> payer-timing,
#5 archetype -> hour. (#6 app<->campaign dropped.) These are PROPOSED (pending v7 + Klimis).

Pure-stdlib SVG (matches the other Schema diagrams). Output:
  <PKG_ROOT>/Schema diagrams/BN_dependency_DAG.svg
"""
from xml.sax.saxutils import escape

from t9sim.diagrams.svg_checks import check_layout
from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1200, 660

TYPE = {                       # fill, stroke
    "userL": ("#BBDEFB", "#1565C0"),
    "appL": ("#C8E6C9", "#2E7D32"),
    "campL": ("#FFE0B2", "#EF6C00"),
    "obs": ("#FFFFFF", "#546E7A"),
    "out": ("#FFCDD2", "#C62828"),
}
# name: (x, y, w, h, label, type)
NODES = {
    "LU1":   (40, 150, 165, 50, "LU1 archetype\n(latent hub)", "userL"),
    "A3":    (255, 78, 120, 32, "A3 os", "obs"),
    "A4":    (415, 78, 135, 32, "A4 os_version", "obs"),
    "A1":    (255, 128, 140, 40, "A1 region\n(untilted)", "obs"),
    "A2":    (415, 132, 135, 32, "A2 city", "obs"),
    "A5":    (255, 186, 130, 32, "A5 device_type", "obs"),
    "D2":    (255, 240, 150, 32, "D2 hour_of_day", "obs"),
    "D3":    (415, 240, 150, 32, "D3 day_of_week", "obs"),
    "LU26":  (40, 240, 165, 46, "LU2-LU6\nvalue latents", "userL"),
    "B1":    (40, 360, 165, 36, "B1 app_id (key)", "obs"),
    "LA1":   (270, 350, 165, 32, "LA1 app_quality", "appL"),
    "LA2":   (270, 396, 165, 32, "LA2 audience_profile", "appL"),
    "C13":   (40, 510, 165, 36, "C1/C3 campaign_id (key)", "obs"),
    "LC1":   (270, 486, 175, 32, "LC1 creative_appeal", "campL"),
    "LC2":   (270, 530, 175, 32, "LC2 game_quality", "campL"),
    "C4":    (270, 574, 175, 32, "C4 ad_genre", "obs"),
    "EV":    (760, 300, 360, 92,
             "EV = P(click)·P(install)·P(payer)·E[spend]\n"
             "→ LU7 competing bid\n→ H3 won / H4 clearing", "out"),
}
# (src, dst, kind)  kind: new(CPT tilt, wired) | v7(BUILD this plan) | key | exist
EDGES = [
    ("LU1", "A3", "new"), ("LU1", "A5", "new"),
    ("A3", "A4", "exist"), ("A1", "A2", "exist"),
    ("LU1", "LU26", "exist"),
    ("B1", "LA1", "key"), ("B1", "LA2", "key"),
    ("C13", "LC1", "key"), ("C13", "LC2", "key"),
    # ── v7 BUILD edges (this plan #1-#5) ──
    ("LA2", "LU1", "v7"),                       # #1 user<->app pairing (build LA2)
    ("LU26", "LC2", "v7"),                      # #2 value-optimised campaign exposure
    ("A3", "EV", "v7"),                         # #3 platform os -> spend
    ("D2", "EV", "v7"), ("D3", "EV", "v7"),     # #4 D2/D3 payer-timing
    ("LU1", "D2", "v7"),                        # #5 archetype -> hour
    ("LU26", "EV", "exist"), ("LA1", "EV", "exist"),
    ("LC1", "EV", "exist"), ("LC2", "EV", "exist"),
]
GROUPS = [
    (15, 45, 560, 245, "USER POOL  (per user)"),
    (15, 305, 560, 130, "APP POOL  (per app)"),
    (15, 455, 560, 175, "CAMPAIGN POOL  (per campaign)"),
    (655, 230, 480, 235, "OUTCOMES  (existing latent → outcome edges)"),
]
EKIND = {"new": ("#C62828", "2.4", ""),          # CPT tilt: archetype -> A-feature (wired)
         "v7": ("#00897B", "2.8", ""),           # v7 BUILD edge (this plan #1-#5)
         "compactb": ("#AD1457", "2.4", "2,3"),  # (retired -> folded into v7)
         "key": ("#616161", "1.7", "7,4"),       # key lookup: id -> entity latent
         "exist": ("#9E9E9E", "1.3", ""),
         "couple": ("#6A1B9A", "2.2", "7,5")}    # (retired -> pairing is v7 #1)


def _c(n):
    x, y, w, h, *_ = NODES[n]
    return x + w / 2, y + h / 2, w, h


def _edge(a, b, kind):
    ax, ay, aw, ah = _c(a)
    bx, by, bw, bh = _c(b)
    # exit a / enter b on the side facing the other node
    sx = ax + aw / 2 if bx > ax else ax - aw / 2 if bx < ax - 5 else ax
    ex = bx - bw / 2 if bx > ax else bx + bw / 2 if bx < ax - 5 else bx
    if abs(bx - ax) < 6:                     # vertical-ish
        sx, ex = ax, bx
        sy = ay + ah / 2 if by > ay else ay - ah / 2
        ey = by - bh / 2 if by > ay else by + bh / 2
    else:
        sy, ey = ay, by
    col, wdt, dash = EKIND[kind]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" '
            f'stroke="{col}" stroke-width="{wdt}"{da} marker-end="url(#a{kind})"/>')


def render():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
    s.append('<defs>')
    for k in EKIND:
        col = EKIND[k][0]
        s.append(f'<marker id="a{k}" markerWidth="9" markerHeight="9" refX="7.5" '
                 f'refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" '
                 f'fill="{col}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    s.append(f'<text x="{W/2:.0f}" y="30" text-anchor="middle" font-size="20" '
             f'font-weight="700" fill="#212121">BN dependency layer (v7 build plan) — '
             f'observable &#8596; latent coupling</text>')
    # group boxes
    for gx, gy, gw, gh, lab in GROUPS:
        s.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" '
                 f'fill="#F7F9FB" stroke="#CFD8DC" stroke-width="1.2"/>')
        s.append(f'<text x="{gx+12}" y="{gy+20}" font-size="12.5" '
                 f'font-weight="700" fill="#455A64">{escape(lab)}</text>')
    # edges (behind nodes); D2->EV routed below the pools so it isn't hidden by D3
    for a, b, k in EDGES:
        if (a, b) == ("D2", "EV"):
            cx, cy, _, ch = _c("D2")
            col = EKIND["v7"][0]
            s.append(f'<polyline points="{cx:.0f},{cy+ch/2:.0f} {cx:.0f},296 '
                     f'700,296 760,330" fill="none" stroke="{col}" '
                     f'stroke-width="2.8" marker-end="url(#av7)"/>')
        else:
            s.append(_edge(a, b, k))
    # v7 BUILD edge tags (#1-#5)
    v7lab = {("LA2", "LU1"): "#1 pairing", ("LU26", "LC2"): "#2 exposure",
             ("A3", "EV"): "#3 os->spend", ("D3", "EV"): "#4 D2/D3->payer",
             ("LU1", "D2"): "#5 hour"}
    for (a, b), lab in v7lab.items():
        ax, ay, _, _ = _c(a)
        bx, by, _, _ = _c(b)
        mx, my = (ax + bx) / 2, (ay + by) / 2 - 4
        s.append(f'<text x="{mx:.0f}" y="{my:.0f}" text-anchor="middle" font-size="9.5" '
                 f'fill="#00695C" stroke="#FFFFFF" stroke-width="3" paint-order="stroke" '
                 f'font-weight="700">{escape(lab)}</text>')
    # nodes
    for n, (x, y, w, h, lab, t) in NODES.items():
        fill, stroke = TYPE[t]
        sw = 2.4 if n == "LU1" else 1.4
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = lab.split("\n")
        y0 = y + h / 2 - (len(lines) - 1) * 7 + 4
        for i, ln in enumerate(lines):
            fw = "700" if (i == 0 and t in ("userL", "out")) else "400"
            s.append(f'<text x="{x+w/2:.0f}" y="{y0+i*14:.0f}" '
                     f'text-anchor="middle" font-size="11.5" '
                     f'font-weight="{fw}" fill="#1A1A1A">{escape(ln)}</text>')
    # legend
    ly = H - 26
    items = [("#C62828", "", "CPT tilt: archetype->A-feature (wired)"),
             ("#00897B", "", "v7 BUILD edge (this plan #1-#5)"),
             ("#616161", "7,4", "key lookup: id->entity latent"),
             ("#9E9E9E", "", "existing (nesting / latent->outcome)")]
    lx = 30
    for col, dash, txt in items:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.4"{da}/>')
        s.append(f'<text x="{lx+32}" y="{ly+4}" font-size="10.5" '
                 f'fill="#37474F">{escape(txt)}</text>')
        lx += 36 + len(txt) * 6.0
    s.append('</svg>')
    check_layout(NODES, GROUPS, name="BN_dependency_DAG.svg")
    (OUT / "BN_dependency_DAG.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / "BN_dependency_DAG.svg")


if __name__ == "__main__":
    render()
