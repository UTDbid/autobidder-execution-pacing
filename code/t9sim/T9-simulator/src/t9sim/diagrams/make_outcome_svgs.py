"""Generate one SVG per T9 *outcome* feature, showing how it is calculated:
every input is a node, feeding intermediate ops, then the output column.

Layout: layered left->right, colour-coded by source type. Edges are ORTHOGONAL
(horizontal + vertical segments only). Layering is "proper" — every edge spans
exactly one column — so each edge's vertical bend lies in the empty gutter between
two columns and never crosses an unrelated box. Pure-Python (no graphviz/cairo).

Outputs to  <PKG_ROOT>/Schema diagrams/  (i.e. t9_sim/Schema diagrams/).
"""
from xml.sax.saxutils import escape

from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# node type -> (fill, stroke)
TYPE = {
    "config":  ("#ECEFF1", "#90A4AE"),
    "user":    ("#BBDEFB", "#1976D2"),
    "app":     ("#C8E6C9", "#388E3C"),
    "camp":    ("#FFE0B2", "#F57C00"),
    "context": ("#FFFFFF", "#607D8B"),
    "draw":    ("#E1BEE7", "#8E24AA"),
    "op":      ("#FFF9C4", "#FBC02D"),
    "out":     ("#FFCDD2", "#D32F2F"),
}
LEGEND = [
    ("config", "config constant"), ("user", "user-pool (hidden LU*)"),
    ("app", "app-pool (hidden)"), ("camp", "campaign-pool"),
    ("context", "visible / prior column"), ("draw", "random draw"),
    ("op", "intermediate op"), ("out", "OUTPUT"),
]

CHARW, FS, LH = 6.7, 12.5, 16
PAD_X, PAD_Y = 14, 12
HGAP, VGAP = 96, 26
M = 28
TITLE_H, LEGEND_H = 56, 64
BUS = 26          # vertical-bend offset left of the target (sits in the gutter)


def _wh(label):
    lines = label.split("\n")
    w = max(len(l) for l in lines) * CHARW + 2 * PAD_X
    h = len(lines) * LH + 2 * PAD_Y
    return max(w, 92), h


def render(title, nodes, edges, fname):
    layers = {}
    for k, (lab, typ, lyr) in nodes.items():
        layers.setdefault(lyr, []).append(k)
    order = sorted(layers)

    size = {k: _wh(v[0]) for k, v in nodes.items()}
    colw = {lyr: max(size[k][0] for k in layers[lyr]) for lyr in order}

    xcent, x = {}, M
    for lyr in order:
        xcent[lyr] = x + colw[lyr] / 2
        x += colw[lyr] + HGAP
    total_w = x - HGAP + M

    # legend: only the node types actually used in THIS diagram (canonical order)
    used = {typ for (_, typ, _) in nodes.values()}
    legend = [(t, d) for (t, d) in LEGEND if t in used]
    legend_w = M + sum(22 + len(d) * 6.0 + 14 for _, d in legend) + M
    total_w = max(total_w, legend_w)

    colh = {lyr: sum(size[k][1] for k in layers[lyr]) + VGAP * (len(layers[lyr]) - 1)
            for lyr in order}
    body_h = max(colh.values())
    pos = {}
    for lyr in order:
        y = TITLE_H + (body_h - colh[lyr]) / 2
        for k in layers[lyr]:
            w, h = size[k]
            pos[k] = (xcent[lyr], y + h / 2, w, h)
            y += h + VGAP
    total_h = TITLE_H + body_h + LEGEND_H

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" '
         f'height="{total_h:.0f}" viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
         f'font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    s.append('<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" '
             'refY="3" orient="auto" markerUnits="strokeWidth">'
             '<path d="M0,0 L8,3 L0,6 Z" fill="#455A64"/></marker></defs>')
    s.append(f'<rect width="{total_w:.0f}" height="{total_h:.0f}" fill="#FAFAFA"/>')
    s.append(f'<text x="{total_w/2:.0f}" y="34" text-anchor="middle" '
             f'font-size="20" font-weight="700" fill="#263238">{escape(title)}</text>')

    # orthogonal edges (H -> V -> H), bend in the gutter just left of target
    for a, b in edges:
        ax, ay, aw, ah = pos[a]
        bx, by, bw, bh = pos[b]
        x1, y1 = ax + aw / 2, ay              # source right-centre
        x2, y2 = bx - bw / 2, by              # target left-centre
        busx = x2 - BUS
        if busx <= x1 + 6:                    # fallback (shouldn't trigger w/ proper layering)
            busx = (x1 + x2) / 2
        pts = (f"{x1:.0f},{y1:.0f} {busx:.0f},{y1:.0f} "
               f"{busx:.0f},{y2:.0f} {x2:.0f},{y2:.0f}")
        s.append(f'<polyline points="{pts}" fill="none" stroke="#455A64" '
                 f'stroke-width="1.5" marker-end="url(#arr)"/>')

    for k, (lab, typ, lyr) in nodes.items():
        cx, cy, w, h = pos[k]
        fill, stroke = TYPE[typ]
        sw = 2.6 if typ == "out" else 1.4
        s.append(f'<rect x="{cx-w/2:.0f}" y="{cy-h/2:.0f}" width="{w:.0f}" '
                 f'height="{h:.0f}" rx="9" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"/>')
        lines = lab.split("\n")
        y0 = cy - (len(lines) - 1) * LH / 2
        for i, ln in enumerate(lines):
            fw = (700 if typ == "out" else 500) if i == 0 else 400
            fs = FS if i == 0 else FS - 1.5
            s.append(f'<text x="{cx:.0f}" y="{y0 + i*LH + 4:.0f}" '
                     f'text-anchor="middle" font-size="{fs}" font-weight="{fw}" '
                     f'fill="#263238">{escape(ln)}</text>')

    lx, ly = M, total_h - LEGEND_H + 18
    s.append(f'<text x="{lx}" y="{ly-4}" font-size="11" font-weight="700" '
             f'fill="#546E7A">NODE TYPES</text>')
    cx = lx
    for typ, desc in legend:
        fill, stroke = TYPE[typ]
        s.append(f'<rect x="{cx}" y="{ly+6}" width="13" height="13" rx="3" '
                 f'fill="{fill}" stroke="{stroke}"/>')
        s.append(f'<text x="{cx+18}" y="{ly+16}" font-size="10.5" '
                 f'fill="#37474F">{escape(desc)}</text>')
        cx += 22 + len(desc) * 6.0 + 14
    s.append('</svg>')
    (OUT / fname).write_text("\n".join(s), encoding="utf-8")
    return fname


# ── feature specs (PROPER layering: every edge spans exactly one column) ────
FEATURES = {}

FEATURES["E1_click"] = ("E1  click", {
    "B6": ("B6 slot_format", "context", 0),
    "B45": ("B4×B5 slot_size", "context", 0),
    "LU6": ("LU6 interest_vec", "user", 0),
    "C4": ("C4 ad_genre", "camp", 0),
    "ctr": ("base_ctr\n(config)", "config", 1),
    "LU2": ("LU2\nclick_propensity", "user", 1),
    "vslot": ("v_slot\n= fmt_wt(B6) × size_wt(B4,B5)", "op", 1),
    "mr": ("relevance m(r)\n=(1−w_click)+w_click·LU6[ad_genre]", "op", 1),
    "LA1": ("LA1 app_quality", "app", 1),
    "LC1": ("LC1 creative_appeal", "camp", 1),
    "p": ("P(click)\n= base_ctr·LU2·v_slot·m(r)·LA1·LC1", "op", 2),
    "out": ("E1 click\n~ Bernoulli(P(click))", "out", 3),
}, [("B6", "vslot"), ("B45", "vslot"), ("LU6", "mr"), ("C4", "mr"),
    ("ctr", "p"), ("LU2", "p"), ("vslot", "p"), ("mr", "p"), ("LA1", "p"),
    ("LC1", "p"), ("p", "out")])

FEATURES["E2_click_timestamp"] = ("E2  click_timestamp", {
    "D1": ("D1 timestamp", "context", 0),
    "click": ("E1 click (0/1)", "context", 0),
    "exp": ("Exp(mean=30s)\nrandom draw", "draw", 0),
    "out": ("E2 click_timestamp\n= D1 + Exp  if click=1  else −1", "out", 1),
}, [("D1", "out"), ("click", "out"), ("exp", "out")])

FEATURES["F1_install"] = ("F1  install", {
    "LU6": ("LU6 interest_vec", "user", 0),
    "C4": ("C4 ad_genre", "camp", 0),
    "ir": ("base_ir\n(config)", "config", 1),
    "LU3": ("LU3\ninstall_propensity", "user", 1),
    "ease": ("ease(B2)\ninstall-ease by genre", "context", 1),
    "mr": ("relevance m(r)\n=(1−w_install)+w_install·LU6[ad_genre]", "op", 1),
    "LA1": ("LA1 app_quality", "app", 1),
    "p": ("P(install|click)\n= base_ir·LU3·ease(B2)·m(r)·LA1", "op", 2),
    "click": ("E1 click=1\n(gate)", "context", 2),
    "out": ("F1 install\n~ Bernoulli  if click=1  else 0", "out", 3),
}, [("LU6", "mr"), ("C4", "mr"), ("ir", "p"), ("LU3", "p"), ("ease", "p"),
    ("mr", "p"), ("LA1", "p"), ("p", "out"), ("click", "out")])

FEATURES["F2_install_timestamp"] = ("F2  install_timestamp", {
    "E2": ("E2 click_timestamp", "context", 0),
    "inst": ("F1 install=1\n(gate)", "context", 0),
    "ln": ("LogNormal\n(minutes→hours)", "draw", 0),
    "out": ("F2 install_timestamp\n= E2 + LogNormal  if install=1  else −1", "out", 1),
}, [("E2", "out"), ("inst", "out"), ("ln", "out")])

FEATURES["G1_is_payer"] = ("G1  is_payer", {
    "LU6": ("LU6 interest_vec", "user", 0),
    "C4": ("C4 ad_genre", "camp", 0),
    "mr": ("relevance m(r)\n=(1−w_pay)+w_pay·LU6[ad_genre]", "op", 1),
    "LU4": ("LU4\npayer_probability", "user", 1),
    "bp": ("base_payer\n(calibration knob)", "config", 1),
    "p": ("P(pay) = base_payer · LU4 · m(r)", "op", 2),
    "inst": ("F1 install=1\n(gate)", "context", 2),
    "out": ("G1 is_payer\n~ Bernoulli  if install=1  else 0", "out", 3),
}, [("LU6", "mr"), ("C4", "mr"), ("LU4", "p"), ("bp", "p"), ("mr", "p"),
    ("p", "out"), ("inst", "out")])

FEATURES["G2_ltv_value"] = ("G2  ltv_value", {
    "mu": ("lognormal_μ\n(config)", "config", 0),
    "mult": ("ltv_mult(B2)\nby genre", "context", 0),
    "mucat": ("μ_cat = lognormal_μ + ln(ltv_mult(B2))", "op", 1),
    "sig": ("σ\n(config)", "config", 1),
    "base": ("LogNormal(μ_cat, σ)", "op", 2),
    "LU5": ("LU5 ltv_multiplier", "user", 2),
    "LC2": ("LC2 game_quality", "camp", 2),
    "pay": ("G1 is_payer=1\n(gate)", "context", 2),
    "out": ("G2 ltv_value\n= LogNormal·LU5·game_quality  if payer  else 0", "out", 3),
}, [("mu", "mucat"), ("mult", "mucat"), ("mucat", "base"), ("sig", "base"),
    ("base", "out"), ("LU5", "out"), ("LC2", "out"), ("pay", "out")])

FEATURES["G3_ltv_7d"] = ("G3  ltv_7d", {
    "G2": ("G2 ltv_value", "context", 0),
    "d7": ("decay_d7 = 0.40\n(config)", "config", 0),
    "out": ("G3 ltv_7d\n= G2 × 0.40", "out", 1),
}, [("G2", "out"), ("d7", "out")])

FEATURES["G4_ltv_30d"] = ("G4  ltv_30d", {
    "G2": ("G2 ltv_value", "context", 0),
    "d30": ("decay_d30 = 0.70\n(config)", "config", 0),
    "out": ("G4 ltv_30d\n= G2 × 0.70", "out", 1),
}, [("G2", "out"), ("d30", "out")])

FEATURES["EV_expected_value"] = ("EV  expected value (ground truth)", {
    "pc": ("P(click)", "op", 0),
    "pi": ("P(install|click)", "op", 0),
    "pp": ("P(payer|install)", "op", 0),
    "el": ("E[ltv|payer]", "op", 0),
    "note": ("(each P built from user LU2–LU6,\napp LA1, campaign LC1/LC2)", "context", 0),
    "out": ("EV\n= P(click)·P(install|click)·P(payer|install)·E[ltv|payer]", "out", 1),
}, [("pc", "out"), ("pi", "out"), ("pp", "out"), ("el", "out"), ("note", "out")])

FEATURES["LU7_competing_bid"] = ("LU7  competing_bid (value-aware)", {
    "g": ("γ value-sensitivity\n(config)", "config", 0),
    "EV": ("EV (log, standardised)", "op", 0),
    "eg": ("ε_g  gaming noise", "draw", 0),
    "ecpm": ("base_eCPM(B6)\n= iPinYou shape × eCPM target", "context", 0),
    "ex": ("ε_x  non-gaming noise", "draw", 0),
    "lg": ("LU7_gaming\n= base_eCPM · exp(γ·z(logEV) + ε_g)", "op", 1),
    "ln": ("LU7_nongaming\n= base_eCPM · exp(ε_x)", "op", 1),
    "out": ("LU7 competing_bid\n= max(LU7_gaming, LU7_nongaming)", "out", 2),
}, [("g", "lg"), ("EV", "lg"), ("eg", "lg"), ("ecpm", "lg"),
    ("ecpm", "ln"), ("ex", "ln"), ("lg", "out"), ("ln", "out")])

FEATURES["H2_bid_price"] = ("H2  bid_price (DSP, observables only)", {
    "k": ("k_cpa(C2)\nby advertiser_scale", "context", 0),
    "vslot": ("v_slot\n= fmt_wt(B6) × size_wt(B4,B5)", "op", 0),
    "ease": ("ease(B2)", "context", 0),
    "eltv": ("E[ltv | B2]", "context", 0),
    "ve": ("value_estimate\n= k_cpa · v_slot · ease(B2) · E[ltv|B2]", "op", 1),
    "shade": ("shade ≈ 0.7–1.0\n(config)", "config", 1),
    "noise": ("LogNormal(0, σ_explore)\nexploration draw", "draw", 1),
    "out": ("H2 bid_price\n= value_estimate · shade · noise", "out", 2),
}, [("k", "ve"), ("vslot", "ve"), ("ease", "ve"), ("eltv", "ve"),
    ("ve", "out"), ("shade", "out"), ("noise", "out")])

FEATURES["H3_won"] = ("H3  won", {
    "LU7": ("LU7 competing_bid", "context", 0),
    "H1": ("H1 floor_price", "context", 0),
    "mx": ("max(LU7, floor)", "op", 1),
    "H2": ("H2 bid_price", "context", 1),
    "out": ("H3 won\n= 1 if H2 ≥ max(LU7, floor) else 0", "out", 2),
}, [("LU7", "mx"), ("H1", "mx"), ("mx", "out"), ("H2", "out")])

FEATURES["H4_clearing_price"] = ("H4  clearing_price (first-price)", {
    "H3": ("H3 won\n(branch selector)", "context", 0),
    "H2": ("H2 bid_price", "context", 0),
    "LU7": ("LU7 competing_bid", "context", 0),
    "H1": ("H1 floor_price", "context", 0),
    "out": ("H4 clearing_price\n= H2 if won\n= LU7 if lost & LU7 ≥ H1\n= NaN if unsold (max(H2, LU7) < H1)", "out", 1),
}, [("H3", "out"), ("H2", "out"), ("LU7", "out"), ("H1", "out")])


if __name__ == "__main__":
    for fname, (title, nodes, edges) in FEATURES.items():
        render(title, nodes, edges, fname + ".svg")
    print(f"Wrote {len(FEATURES)} SVGs to: {OUT}")
