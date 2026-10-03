"""Predictive pathway: observable features -> latent ground-truth -> outcome labels.

Shows WHY each observable carries predictive signal: a feature is only predictive
if there is a path from it INTO the latents that the funnel/auction uses to generate
the outcome labels.

  - B/C observables already have a path: they enter the funnel as direct multipliers
    (slot, ease, genre) OR an ID acts as the key to a per-entity latent (app_id->LA*,
    campaign_id->LC*).
  - User A-features get a path ONLY once the BN couples them to the archetype LU1
    (red edges). Without the BN they are independent of LU2-LU6 -> noise w.r.t.
    outcomes (the 12 Jun oracle-features finding).

Pure-stdlib SVG (matches the other Schema diagrams). Output:
  <PKG_ROOT>/Schema diagrams/Predictive_pathway.svg
"""
from xml.sax.saxutils import escape

from t9sim.diagrams.svg_checks import check_layout
from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1320, 755

TYPE = {                       # fill, stroke
    "obs":   ("#FFFFFF", "#546E7A"),
    "userL": ("#BBDEFB", "#1565C0"),
    "appL":  ("#C8E6C9", "#2E7D32"),
    "campL": ("#FFE0B2", "#EF6C00"),
    "ev":    ("#E1BEE7", "#6A1B9A"),
    "out":   ("#FFCDD2", "#C62828"),
}
# name: (x, y, w, h, label, type)
NODES = {
    # observables (left)
    "A1":  (40, 92, 170, 30, "A1 region", "obs"),
    "A3":  (40, 130, 170, 30, "A3 os", "obs"),
    "A5":  (40, 168, 170, 30, "A5 device_type", "obs"),
    "D2":  (40, 210, 170, 30, "D2 hour_of_day", "obs"),
    "D3":  (40, 248, 170, 30, "D3 day_of_week", "obs"),
    "B1":  (40, 286, 170, 30, "B1 app_id", "obs"),
    "B2":  (40, 324, 170, 30, "B2 app_category", "obs"),
    "SLOT":(40, 362, 170, 30, "B4/B5/B6 slot", "obs"),
    "C13": (40, 452, 170, 30, "C1/C3 campaign_id", "obs"),
    "C2":  (40, 490, 170, 30, "C2 advertiser_scale", "obs"),
    "C4":  (40, 528, 170, 30, "C4 ad_genre", "obs"),
    # latent hub + value latents (middle)
    "LU1": (296, 132, 132, 42, "LU1 archetype\n(hub)", "userL"),
    "LU2": (470, 142, 160, 28, "LU2 click prop.", "userL"),
    "LU3": (470, 176, 160, 28, "LU3 install prop.", "userL"),
    "LU4": (470, 210, 160, 28, "LU4 pay prop.", "userL"),
    "LU5": (470, 244, 160, 28, "LU5 spend / whale", "userL"),
    "LU6": (470, 278, 160, 28, "LU6 interest vec.", "userL"),
    "LA1": (470, 326, 160, 28, "LA1 app_quality", "appL"),
    "LA2": (470, 360, 160, 28, "LA2 audience", "appL"),
    "LC1": (470, 446, 160, 28, "LC1 creative_appeal", "campL"),
    "LC2": (470, 480, 160, 28, "LC2 game_quality", "campL"),
    # auction mini-column
    "EV":  (720, 452, 172, 44, "EV (truth)\n= product of Tier-1 latents", "ev"),
    "LU7": (720, 524, 172, 30, "LU7 competing_bid", "userL"),
    # outcome labels (right)
    "E1":  (1060, 138, 214, 32, "E1 click", "out"),
    "F1":  (1060, 184, 214, 32, "F1 install", "out"),
    "G1":  (1060, 230, 214, 32, "G1 is_payer", "out"),
    "G24": (1060, 276, 214, 32, "G2 / G3 / G4 LTV", "out"),
    "H3":  (1060, 496, 214, 32, "H3 won", "out"),
    "H4":  (1060, 542, 214, 32, "H4 clearing", "out"),
}
# (src, dst, kind)  kind: bn | key | lat | ev
EDGES = [
    # NEW BN pathway: user feature -> archetype -> value latents
    ("A1", "LU1", "bn"), ("A3", "LU1", "bn"), ("A5", "LU1", "bn"),
    ("LU1", "LU2", "bn"), ("LU1", "LU3", "bn"), ("LU1", "LU4", "bn"),
    ("LU1", "LU5", "bn"), ("LU1", "LU6", "bn"),
    # Compact B (proposed): time -> which archetype is active (active-hour prior)
    ("D2", "LU1", "bnTime"), ("D3", "LU1", "bnTime"),
    # ID acts as key to per-entity latent (already predictive)
    ("B1", "LA1", "key"), ("B1", "LA2", "key"),
    ("C13", "LC1", "key"), ("C13", "LC2", "key"),
    # latent -> outcome (data-generating funnel): user (teal), app (green),
    # campaign (amber) coloured by the latent's pool
    ("LU2", "E1", "lat"), ("LU6", "E1", "lat"),
    ("LU3", "F1", "lat"), ("LU4", "G1", "lat"), ("LU5", "G24", "lat"),
    ("LA1", "E1", "appE"), ("LA1", "F1", "appE"),
    ("LC1", "E1", "campE"), ("LC2", "F1", "campE"), ("LC2", "G24", "campE"),
    # value latents -> EV -> competing bid -> won / clearing
    ("LU4", "EV", "ev"), ("LU5", "EV", "ev"),
    ("EV", "LU7", "ev"), ("LU7", "H3", "ev"), ("LU7", "H4", "ev"),
]
GROUPS = [
    (24, 70, 202, 510, "OBSERVABLE FEATURES  (visible -> ML training)"),
    (286, 110, 350, 480, "LATENT GROUND-TRUTH  (hidden from all conditions)"),
    (1046, 108, 230, 220, "OUTCOME LABELS - funnel"),
    (1046, 474, 230, 122, "OUTCOME LABELS - auction"),
]
EKIND = {"bn":    ("#C62828", "2.4", ""),
         "bnTime":("#AD1457", "2.4", "2,3"),   # Compact B: time -> archetype
         "key":   ("#9E9E9E", "1.4", "6,4"),
         "lat":   ("#00838F", "1.6", ""),      # user latent -> outcome
         "appE":  ("#2E7D32", "1.8", ""),      # app latent (LA1/LA2) -> outcome
         "campE": ("#EF6C00", "1.8", ""),      # campaign latent (LC1/LC2) -> outcome
         "ev":    ("#6A1B9A", "1.9", "")}


def _pt(n):
    x, y, w, h = NODES[n][:4]
    return {"cx": x + w / 2, "cy": y + h / 2, "l": x, "r": x + w,
            "t": y, "b": y + h}


def _edge(a, b, kind):
    A, B = _pt(a), _pt(b)
    if abs(A["cx"] - B["cx"]) < 70:                 # same column
        if B["cy"] >= A["cy"]:
            sx, sy, ex, ey = A["cx"], A["b"], B["cx"], B["t"]
        else:
            sx, sy, ex, ey = A["cx"], A["t"], B["cx"], B["b"]
    elif B["cx"] > A["cx"]:
        sx, sy, ex, ey = A["r"], A["cy"], B["l"], B["cy"]
    else:
        sx, sy, ex, ey = A["l"], A["cy"], B["r"], B["cy"]
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
             f'font-weight="700" fill="#212121">Predictive pathway: observable '
             f'features &#8594; latent ground-truth &#8594; outcome labels</text>')
    s.append(f'<text x="{W/2:.0f}" y="50" text-anchor="middle" font-size="12" '
             f'fill="#546E7A">a feature is predictive only if a path reaches it '
             f'into the latents that generate the labels</text>')
    # group boxes
    for gx, gy, gw, gh, lab in GROUPS:
        s.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" '
                 f'fill="#F7F9FB" stroke="#CFD8DC" stroke-width="1.2"/>')
        s.append(f'<text x="{gx+12}" y="{gy+19}" font-size="11.5" '
                 f'font-weight="700" fill="#455A64">{escape(lab)}</text>')
    # edges behind nodes
    for a, b, k in EDGES:
        s.append(_edge(a, b, k))
    # nodes
    for n, (x, y, w, h, lab, t) in NODES.items():
        fill, stroke = TYPE[t]
        sw = 2.6 if n == "LU1" else 1.4
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = lab.split("\n")
        y0 = y + h / 2 - (len(lines) - 1) * 7 + 4
        for i, ln in enumerate(lines):
            fw = "700" if (i == 0 and t in ("userL", "ev", "out")) else "400"
            fs = 11.5 if i == 0 else 9.5
            s.append(f'<text x="{x+w/2:.0f}" y="{y0+i*13:.0f}" '
                     f'text-anchor="middle" font-size="{fs}" '
                     f'font-weight="{fw}" fill="#1A1A1A">{escape(ln)}</text>')
    # note box: direct funnel multipliers (no latent) + nesting
    nx, ny = 40, 580
    s.append(f'<rect x="{nx}" y="{ny}" width="590" height="92" rx="9" '
             f'fill="#FFFDE7" stroke="#E6C200" stroke-width="1.2"/>')
    note = [
        "Direct funnel multipliers (observable -> outcome, no latent):",
        "  slot B4/B5/B6 -> click ;  ease(B2) -> install ;  C4 ad_genre x LU6 -> click ;  C2 -> bid",
        "Nesting (observable -> observable):  A2 city <- A1 region ;  A4 os_version <- A3 os",
        "won = [ bid >= max(LU7, floor H1) ] ;  clearing = bid if won, LU7 if lost-sold",
    ]
    for i, ln in enumerate(note):
        fw = "700" if i == 0 else "400"
        s.append(f'<text x="{nx+12}" y="{ny+20+i*18:.0f}" font-size="10.5" '
                 f'font-weight="{fw}" fill="#5D4E00">{escape(ln)}</text>')
    # legend
    ly = H - 16
    items = [("#C62828", "NEW BN edge: user feature -> archetype -> value latents (the pathway BN creates)"),
             ("#AD1457", "Compact B (proposed): time D2/D3 -> archetype mix (per-archetype von Mises active-hour prior; gated on BN; sensitivity)"),
             ("#9E9E9E", "ID -> per-entity latent (key; already predictive)"),
             ("#00838F", "user latent (LU) -> outcome"),
             ("#2E7D32", "app latent (LA1/LA2) -> outcome"),
             ("#EF6C00", "campaign latent (LC1/LC2) -> outcome"),
             ("#6A1B9A", "value latents -> EV -> competing bid -> won / clearing")]
    lx = 660
    s.append(f'<text x="{lx}" y="596" font-size="11" font-weight="700" '
             f'fill="#37474F">Legend</text>')
    yy = 614
    for col, txt in items:
        s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+26}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.6"/>')
        s.append(f'<text x="{lx+32}" y="{yy+4}" font-size="10.5" '
                 f'fill="#37474F">{escape(txt)}</text>')
        yy += 20
    s.append('</svg>')
    check_layout(NODES, GROUPS, name="Predictive_pathway.svg")
    (OUT / "Predictive_pathway.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / "Predictive_pathway.svg")


if __name__ == "__main__":
    render()
