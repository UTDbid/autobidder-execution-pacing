"""Provenance matrix SVG: data source(s) + feature-inputs for every T9 feature.

Rows = features grouped by pool (User / App / Campaign / Auction), covering
A-H + LU + LA + LC. Columns = 5 data sources + a "Calculated from" column listing
the other feature columns used to compute each row.

Output: <PKG_ROOT>/Schema diagrams/Data_sources_by_feature.svg   (pure stdlib)
"""
from xml.sax.saxutils import escape

from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# code -> colour
SOURCES = [
    ("IP", "#00897B"), ("BM", "#3949AB"), ("IN", "#FB8C00"),
    ("SP", "#43A047"), ("CP", "#E53935"),
]
SRC_FULL = {
    "IP": "iPinYou — empirical distribution shapes & anchors",
    "BM": "Benchmark config — external US-2025 reports (StatCounter, Sensor Tower / Newzoo, IAB, mediation reports, AppsFlyer / Liftoff …)",
    "IN": "Inferred config — set by documented judgment (flagged INFERRED)",
    "SP": "Synthetic pool — Stage-1 generation: synthetic ids + latent draws",
    "CP": "Computed — calculated from the features in the right-hand column (Stage-2 formulas + random draws)",
}

# (label, source codes, features-used-to-calculate-this-row)
SECTIONS = [
    ("USER POOL", [
        ("A1  region", "IP", ""), ("A2  city", "IP", "A1"),
        ("A3  os", "BM", ""), ("A4  os_version", "BM", "A3"),
        ("A5  device_type", "BM", ""),
        ("LU1  archetype", "SP BM", ""),
        ("LU2  click_propensity", "SP BM IN", "LU1"),
        ("LU3  install_propensity", "SP BM IN", "LU1"),
        ("LU4  payer_probability", "SP BM IN", "LU1"),
        ("LU5  ltv_multiplier", "SP BM", "LU1"),
        ("LU6  interest_vector", "SP IN", "LU1"),
    ]),
    ("APP POOL", [
        ("B1  app_id", "SP", ""), ("B2  app_category", "BM", "B1"),
        ("LA1  app_quality", "SP IN", ""), ("LA2  app_audience_profile", "SP IN", ""),
    ]),
    ("CAMPAIGN POOL", [
        ("C1  advertiser_id", "IP SP", ""), ("C2  advertiser_scale", "BM IN", ""),
        ("C3  campaign_id", "SP", "C1"), ("C4  ad_genre", "BM IN", ""),
        ("LC1  creative_appeal", "SP IN", ""), ("LC2  game_quality", "SP IN", ""),
    ]),
    ("AUCTION  (per-auction context + outcomes)", [
        ("B3  ad_exchange", "BM", ""),
        ("B4  slot_width", "BM IN", "B6"), ("B5  slot_height", "BM IN", "B6"),
        ("B6  slot_format", "BM IN", ""),
        ("D1  timestamp", "IP", ""), ("D2  hour_of_day", "CP", "D1"),
        ("D3  day_of_week", "CP", "D1"),
        ("E1  click", "CP", "LU2, B4, B5, B6, LU6, C4, LA1, LC1"),
        ("E2  click_timestamp", "CP", "D1, E1"),
        ("F1  install", "CP", "LU3, B2, LU6, C4, LA1, E1"),
        ("F2  install_timestamp", "CP", "E2, F1"),
        ("G1  is_payer", "CP", "F1, C4, LU4, LU6"),
        ("G2  ltv_value", "CP", "G1, B2, LU5, LC2"),
        ("G3  ltv_7d", "CP", "G2"), ("G4  ltv_30d", "CP", "G2"),
        ("H1  floor_price", "IP BM CP", "B6"),
        ("H2  bid_price", "CP", "C2, B2, B4, B5, B6"),
        ("H3  won", "CP", "H2, LU7, H1"),
        ("H4  clearing_price", "CP", "H2, LU7, H1"),
        ("LU7  competing_bid  (auction-level)", "IP BM CP",
         "LU2–LU6, B2, B4–B6, C4, LA1, LC1, LC2"),
    ]),
]

# Computation formulas for the CP (Computed) rows (padded id, expression)
FORMULAS = [
    ("E1   ", "~ Bernoulli( base_ctr · LU2 · v_slot · m(r) · LA1 · LC1 )"),
    ("F1   ", "~ Bernoulli( base_ir · LU3 · ease(B2) · m(r) · LA1 )      [0 if no click]"),
    ("G1   ", "~ Bernoulli( base_payer · LU4 · m(r) )      [0 if no install]"),
    ("G2   ", "= LogNormal(μ_cat, σ) · LU5 · LC2      [μ_cat = lognormal_μ + ln(ltv_mult(B2));  0 if non-payer]"),
    ("H2   ", "= value_estimate · shade · LogNormal(0, σ_explore)      [value_estimate = k_cpa(C2) · v_slot · ease(B2) · E[ltv|B2]]"),
    ("H3   ", "= 1 if H2 ≥ max(LU7, H1) else 0"),
    ("H4   ", "= H2 if won  |  LU7 if lost & LU7 ≥ H1  |  NaN if unsold (max(H2,LU7) < H1)"),
    ("LU7  ", "= max( base_eCPM(B6)·exp(γ·z(logEV)+ε_g) ,  base_eCPM(B6)·exp(ε_x) )"),
]
WHERE = [
    "where:  m(r) = (1 − w_stage) + w_stage · LU6[ad_genre]        v_slot = fmt_wt(B6) · size_wt(B4,B5)",
    "        EV   = P(click) · P(install|click) · P(payer|install) · E[ltv|payer]",
]

NOTE = ("Notes:  A3–A5 = US-mobile config targets (StatCounter; BM) — current YAML values are placeholders pending that calibration pass.  "
        "B4/B5 = IAB-standard sizes (BM) with an inferred size distribution (IN).  "
        "For rows combining SP with BM/IN, SP marks the Stage-1 draw itself while "
        "the BM/IN chips give the provenance of the drawing distribution's "
        "PARAMETERS (BM = quoted from, or auto-calibrated to, benchmark targets; "
        "IN = documented judgment, e.g. per-archetype spreads & orderings, "
        "Dirichlet concentrations).  "
        "'Calculated from' lists the other feature COLUMNS used (config constants & "
        "random draws omitted); conditional draws (e.g. A2←A1, LU2–6←LU1) "
        "show their parent feature.")

M = 30
TITLE_H, HEADER_H, ROWH, SECTION_H = 52, 42, 25, 28
LABEL_W, SRCW, USEDW = 250, 56, 318
LEGEND_H = 430


def _wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def render():
    nrows = sum(len(items) for _, items in SECTIONS)
    nsec = len(SECTIONS)
    src_x0 = M + LABEL_W
    used_x0 = src_x0 + len(SOURCES) * SRCW
    total_w = used_x0 + USEDW + M
    body_h = nrows * ROWH + nsec * SECTION_H
    total_h = TITLE_H + HEADER_H + body_h + LEGEND_H

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h:.0f}" '
         f'viewBox="0 0 {total_w} {total_h:.0f}" '
         f'font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    s.append(f'<rect width="{total_w}" height="{total_h:.0f}" fill="#FFFFFF"/>')
    s.append(f'<text x="{total_w/2:.0f}" y="34" text-anchor="middle" font-size="21" '
             f'font-weight="700" fill="#212121">Data sources by feature — T9 simulator</text>')

    # column headers
    for i, (code, col) in enumerate(SOURCES):
        cx = src_x0 + i * SRCW + SRCW / 2
        s.append(f'<rect x="{cx-SRCW/2+6:.0f}" y="{TITLE_H+6:.0f}" width="{SRCW-12}" '
                 f'height="{HEADER_H-12}" rx="7" fill="{col}"/>')
        s.append(f'<text x="{cx:.0f}" y="{TITLE_H+26:.0f}" text-anchor="middle" '
                 f'font-size="14" font-weight="700" fill="#FFFFFF">{code}</text>')
    s.append(f'<text x="{used_x0+10:.0f}" y="{TITLE_H+26:.0f}" font-size="12.5" '
             f'font-weight="700" fill="#37474F">Calculated from</text>')

    y = TITLE_H + HEADER_H
    for sec_title, items in SECTIONS:
        s.append(f'<rect x="{M}" y="{y:.0f}" width="{total_w-2*M}" height="{SECTION_H}" '
                 f'fill="#37474F"/>')
        s.append(f'<text x="{M+10}" y="{y+19:.0f}" font-size="12.5" font-weight="700" '
                 f'fill="#FFFFFF">{escape(sec_title)}</text>')
        y += SECTION_H
        for r, (label, codes, used) in enumerate(items):
            if r % 2 == 0:
                s.append(f'<rect x="{M}" y="{y:.0f}" width="{total_w-2*M}" height="{ROWH}" '
                         f'fill="#F5F5F5"/>')
            s.append(f'<text x="{M+10}" y="{y+17:.0f}" font-size="12" '
                     f'fill="#263238">{escape(label)}</text>')
            have = set(codes.split())
            for i, (code, col) in enumerate(SOURCES):
                cx = src_x0 + i * SRCW + SRCW / 2
                cy = y + ROWH / 2
                if code in have:
                    s.append(f'<rect x="{cx-8:.0f}" y="{cy-8:.0f}" width="16" height="16" '
                             f'rx="4" fill="{col}"/>')
                else:
                    s.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="1.6" fill="#CFD8DC"/>')
            if used:
                lines = _wrap(used, 56)
                y0 = y + ROWH / 2 - (len(lines) - 1) * 6
                for j, ln in enumerate(lines):
                    s.append(f'<text x="{used_x0+10:.0f}" y="{y0+j*12+4:.0f}" '
                             f'font-size="10.5" fill="#37474F" '
                             f'font-family="Consolas, monospace">{escape(ln)}</text>')
            else:
                s.append(f'<text x="{used_x0+10:.0f}" y="{y+17:.0f}" font-size="10.5" '
                         f'fill="#B0BEC5">—</text>')
            y += ROWH

    # vertical grid lines (source block + used divider)
    for i in range(len(SOURCES) + 1):
        gx = src_x0 + i * SRCW
        s.append(f'<line x1="{gx}" y1="{TITLE_H+HEADER_H}" x2="{gx}" y2="{y:.0f}" '
                 f'stroke="#E0E0E0" stroke-width="1"/>')
    s.append(f'<line x1="{used_x0}" y1="{TITLE_H+HEADER_H}" x2="{used_x0}" y2="{y:.0f}" '
             f'stroke="#BDBDBD" stroke-width="1.5"/>')

    # legend
    ly = y + 22
    s.append(f'<text x="{M}" y="{ly:.0f}" font-size="12" font-weight="700" '
             f'fill="#37474F">DATA SOURCES</text>')
    ly += 8
    for code, col in SOURCES:
        ly += 18
        s.append(f'<rect x="{M}" y="{ly-11:.0f}" width="14" height="14" rx="3" fill="{col}"/>')
        s.append(f'<text x="{M+22}" y="{ly:.0f}" font-size="11" fill="#37474F">'
                 f'<tspan font-weight="700">{code}</tspan>  {escape(SRC_FULL[code])}</text>')
    # computation formulas (the CP / Computed rows)
    ly += 26
    s.append(f'<text x="{M}" y="{ly:.0f}" font-size="12" font-weight="700" '
             f'fill="#37474F">COMPUTATION FORMULAS  (CP rows)</text>')
    for fid, expr in FORMULAS:
        ly += 16
        s.append(f'<text x="{M}" y="{ly:.0f}" xml:space="preserve" font-size="10.5" '
                 f'font-family="Consolas, monospace" fill="#263238">'
                 f'<tspan font-weight="700" fill="#C62828">{escape(fid)}</tspan>'
                 f'{escape(expr)}</text>')
    for wl in WHERE:
        ly += 14
        s.append(f'<text x="{M}" y="{ly:.0f}" xml:space="preserve" font-size="10" '
                 f'font-family="Consolas, monospace" fill="#78909C">{escape(wl)}</text>')

    ly += 24
    for k, chunk in enumerate(_wrap(NOTE, 150)):
        s.append(f'<text x="{M}" y="{ly+k*14:.0f}" font-size="10" fill="#78909C">'
                 f'{escape(chunk)}</text>')

    s.append('</svg>')
    (OUT / "Data_sources_by_feature.svg").write_text("\n".join(s), encoding="utf-8")
    print("Wrote:", OUT / "Data_sources_by_feature.svg")


if __name__ == "__main__":
    render()
