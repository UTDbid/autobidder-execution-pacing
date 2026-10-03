"""Direct (non-archetype) relationships: observable feature -> funnel / auction outcome.

The observable mechanics ALREADY in the generator (schema Matching Engine + Auction),
independent of the BN layer. Shows the four latent-free direct multipliers
(v_slot, ease, k_cpa, E[ltv|B2]), the ONE observable x latent interaction
(relevance m(r) = LU6[C4]), the bid->won/clearing auction wiring, and the two
nesting edges. Only two latents (LU6, LU7) touch this view.

Pure-stdlib SVG (matches the other Schema diagrams). Output:
  <PKG_ROOT>/Schema diagrams/Direct_relationships.svg
"""
from xml.sax.saxutils import escape

from t9sim.diagrams.svg_checks import check_layout
from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1120, 740

TYPE = {
    "obs":    ("#FFFFFF", "#546E7A"),
    "latent": ("#BBDEFB", "#1565C0"),
    "deriv":  ("#FFF8E1", "#F9A825"),
    "bid":    ("#FFE0B2", "#EF6C00"),
    "out":    ("#FFCDD2", "#C62828"),
}
# name: (x, y, w, h, label, type)
NODES = {
    # ---- FUNNEL band (observables left, derived mid, outcomes right) ----
    "SLOT": (40, 96, 150, 38, "B4/B5/B6\nslot size+format", "obs"),
    "B2":   (40, 162, 150, 30, "B2 app_category", "obs"),
    "C4":   (40, 244, 150, 30, "C4 ad_genre", "obs"),
    "LU6":  (40, 290, 150, 30, "LU6 interest (latent)", "latent"),
    "VSLOT":(280, 100, 165, 28, "v_slot (B6,B4,B5)", "deriv"),
    "EASE": (280, 156, 165, 28, "ease (B2)", "deriv"),
    "ELTV": (280, 200, 165, 28, "E[ltv | B2]", "deriv"),
    "MR":   (280, 250, 165, 36, "m(r) = LU6[C4]\n(relevance)", "deriv"),
    "E1":   (840, 100, 210, 30, "E1 click", "out"),
    "F1":   (840, 160, 210, 30, "F1 install", "out"),
    "G1":   (840, 220, 210, 30, "G1 is_payer", "out"),
    # ---- AUCTION band ----
    "C2":   (40, 410, 150, 30, "C2 advertiser_scale", "obs"),
    "H1":   (40, 466, 150, 30, "H1 floor_price", "obs"),
    "LU7":  (40, 510, 150, 30, "LU7 competing_bid (latent)", "latent"),
    "KCPA": (280, 410, 165, 28, "k_cpa (C2)", "deriv"),
    "H2":   (548, 396, 160, 42, "H2 bid", "bid"),
    "H3":   (840, 438, 210, 30, "H3 won", "out"),
    "H4":   (840, 498, 210, 30, "H4 clearing", "out"),
    # ---- NESTING band ----
    "A1":   (40, 614, 150, 28, "A1 region", "obs"),
    "A3":   (40, 654, 150, 28, "A3 os", "obs"),
    "A2":   (280, 614, 165, 28, "A2 city", "obs"),
    "A4":   (280, 654, 165, 28, "A4 os_version", "obs"),
}
# (src, dst, kind)  kind: deriv | lat | fun | bid | auc | nest
EDGES = [
    # observable -> derived term
    ("SLOT", "VSLOT", "deriv"), ("B2", "EASE", "deriv"),
    ("B2", "ELTV", "deriv"), ("C2", "KCPA", "deriv"), ("C4", "MR", "deriv"),
    ("LU6", "MR", "lat"),
    # derived -> funnel outcome (no latent except m(r))
    ("VSLOT", "E1", "fun"), ("EASE", "F1", "fun"),
    ("MR", "E1", "fun"), ("MR", "F1", "fun"), ("MR", "G1", "fun"),
    # derived -> bid
    ("KCPA", "H2", "bid"), ("VSLOT", "H2", "bid"),
    ("EASE", "H2", "bid"), ("ELTV", "H2", "bid"),
    # bid / latent / floor -> auction outcome
    ("H2", "H3", "auc"), ("H2", "H4", "auc"),
    ("LU7", "H3", "auc"), ("LU7", "H4", "auc"),
    ("H1", "H3", "auc"), ("H1", "H4", "auc"),
    # nesting
    ("A1", "A2", "nest"), ("A3", "A4", "nest"),
]
GROUPS = [
    (24, 64, 1040, 286, "FUNNEL multipliers  (observable -> click / install / payer)"),
    (24, 366, 1040, 198, "AUCTION  (bid mechanics -> won / clearing)"),
    (24, 582, 470, 110, "NESTING  (observable -> observable)"),
]
EKIND = {"deriv": ("#9E9E9E", "1.4", ""),
         "lat":   ("#1565C0", "1.8", ""),
         "fun":   ("#00838F", "1.6", ""),
         "bid":   ("#EF6C00", "1.6", ""),
         "auc":   ("#6A1B9A", "1.7", ""),
         "nest":  ("#9E9E9E", "1.4", "6,4")}


def _pt(n):
    x, y, w, h = NODES[n][:4]
    return {"cx": x + w / 2, "cy": y + h / 2, "l": x, "r": x + w,
            "t": y, "b": y + h}


def _edge(a, b, kind):
    A, B = _pt(a), _pt(b)
    if abs(A["cx"] - B["cx"]) < 70:
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
    s.append(f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="19" '
             f'font-weight="700" fill="#212121">Direct (non-archetype) '
             f'relationships: observable feature &#8594; outcome</text>')
    s.append(f'<text x="{W/2:.0f}" y="47" text-anchor="middle" font-size="11.5" '
             f'fill="#546E7A">the observable mechanics already in the generator '
             f'(schema Matching Engine + Auction) - independent of the BN layer</text>')
    for gx, gy, gw, gh, lab in GROUPS:
        s.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" '
                 f'fill="#F7F9FB" stroke="#CFD8DC" stroke-width="1.2"/>')
        s.append(f'<text x="{gx+12}" y="{gy+19}" font-size="11.5" '
                 f'font-weight="700" fill="#455A64">{escape(lab)}</text>')
    for a, b, k in EDGES:
        s.append(_edge(a, b, k))
    for n, (x, y, w, h, lab, t) in NODES.items():
        fill, stroke = TYPE[t]
        sw = 2.2 if n == "H2" else 1.4
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = lab.split("\n")
        y0 = y + h / 2 - (len(lines) - 1) * 7 + 4
        for i, ln in enumerate(lines):
            fw = "700" if (i == 0 and t in ("latent", "bid", "out")) else "400"
            fs = 11 if i == 0 else 9.5
            s.append(f'<text x="{x+w/2:.0f}" y="{y0+i*13:.0f}" '
                     f'text-anchor="middle" font-size="{fs}" '
                     f'font-weight="{fw}" fill="#1A1A1A">{escape(ln)}</text>')
    # cross-reference: time acts via the archetype (Compact B), NOT a direct edge
    s.append('<text x="470" y="361" font-size="10.5" font-style="italic" '
             'fill="#455A64">Time (D2 hour, D3 day) act on behaviour via the '
             'ARCHETYPE (Compact B) - see Predictive_pathway; not a direct edge.</text>')
    # footnote box with exact schema formulas + line cites
    nx, ny = 510, 582
    s.append(f'<rect x="{nx}" y="{ny}" width="554" height="114" rx="9" '
             f'fill="#FFFDE7" stroke="#E6C200" stroke-width="1.2"/>')
    note = [
        "Schema formulas (v6):",
        "P(click)  = base_ctr . LU2 . v_slot . m(r) . LA1 . LC1            [L218/292]",
        "P(install|click) = base_ir . LU3 . ease(B2) . m(r) . LA1          [L220/293]",
        "v_slot = format_weight(B6) . size_weight(B4,B5)                   [L216]",
        "m(r) = LU6[ad_genre]  (observable C4 x LATENT LU6)               [L205-207]",
        "H2 bid = k_cpa(C2) . v_slot . ease(B2) . E[ltv|B2] . shade . noise [L250/302]",
        "won = [H2 >= max(LU7, floor)] ; clearing = H2 if won, LU7 if lost-sold [L257-260]",
    ]
    for i, ln in enumerate(note):
        fw = "700" if i == 0 else "400"
        s.append(f'<text x="{nx+12}" y="{ny+18+i*15:.0f}" font-size="9.5" '
                 f'font-weight="{fw}" fill="#5D4E00" '
                 f'font-family="Consolas, monospace">{escape(ln)}</text>')
    # legend
    ly = H - 12
    items = [("#9E9E9E", "observable -> derived term"),
             ("#1565C0", "latent input (only LU6, LU7 touch this view)"),
             ("#00838F", "-> funnel outcome"),
             ("#EF6C00", "-> bid"),
             ("#6A1B9A", "-> won / clearing")]
    lx = 30
    for col, txt in items:
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        s.append(f'<text x="{lx+30}" y="{ly+4}" font-size="10" '
                 f'fill="#37474F">{escape(txt)}</text>')
        lx += 34 + len(txt) * 6.0
    s.append('</svg>')
    check_layout(NODES, GROUPS, name="Direct_relationships.svg")
    (OUT / "Direct_relationships.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / "Direct_relationships.svg")


if __name__ == "__main__":
    render()
