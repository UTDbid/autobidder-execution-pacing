"""T9 architecture — SIMPLE structure overview (v4), dark house style.

Decluttered for quick reading: no scale numbers, no repeated 'user/app/campaign'
labels, trimmed field lists. The fuller version is t9_data_flow_v5.svg.
Output: <PKG_ROOT>/Schema diagrams/T9_architecture.svg   — pure stdlib.
"""
from xml.sax.saxutils import escape

from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams" / "T9_architecture.svg"

W, H = 900, 760
BG = "#0E1117"
FONT = "'Anthropic Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
GREEN = ("#085041", "#5DCAA5", "#9FE1CB", "#5DCAA5")
PURPLE = ("#3C3489", "#AFA9EC", "#CECBF6", "#AFA9EC")
BLUE = ("#0C447C", "#85B7EB", "#B5D4F4", "#85B7EB")
GRAY = ("#444441", "#B4B2A9", "#D3D1C7", "#B4B2A9")
PANEL = ("#11161D", "#39424D")
SECLBL = "#FAF9F5"
ARROW = "#C2C0B6"

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">']
s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
s.append('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
         'markerHeight="6" orient="auto-start-reverse">'
         f'<path d="M2 1L8 5L2 9" fill="none" stroke="{ARROW}" stroke-width="1.6" '
         'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def box(x, y, w, h, pal, lines, ts=12, ss=9.5, rx=9):
    fill, stroke, tcol, scol = pal
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1"/>')
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (ss + 4) / 2 - (2 if n > 1 else -4)
    for i, ln in enumerate(lines):
        fs, col, fw = (ts, tcol, 600) if i == 0 else (ss, scol, 400)
        dy = y0 + (0 if i == 0 else ts + (i - 1) * (ss + 4) + 1)
        s.append(f'<text x="{x+w/2:.0f}" y="{dy:.0f}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{fw}" fill="{col}">{escape(ln)}</text>')


def panel(x, y, w, h):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL[0]}" '
             f'stroke="{PANEL[1]}" stroke-width="0.8"/>')


def label(x, y, t, size=12.5, col=SECLBL, w=600, anchor="start"):
    s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" '
             f'text-anchor="{anchor}" fill="{col}">{escape(t)}</text>')


def arrow(x1, y1, x2, y2, col=ARROW, sw=1.4):
    s.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
             f'stroke="{col}" stroke-width="{sw}" marker-end="url(#a)"/>')


# ── inputs ──────────────────────────────────────────────────────────────
iw, iy, ih = 250, 24, 58
ix = [30, 325, 620]
box(ix[0], iy, iw, ih, GREEN, ["iPinYou dataset", "distributions · prices"])
box(ix[1], iy, iw, ih, PURPLE, ["Industry reports", "CTR · install/payer · LTV · eCPM"])
box(ix[2], iy, iw, ih, PURPLE, ["Archetype design", "behaviour · causal mechanisms"])
for x in ix:
    arrow(x + iw / 2, iy + ih, x + iw / 2, 130, ARROW, 1.1)

# ── simulator ───────────────────────────────────────────────────────────
s.append('<rect x="18" y="130" width="864" height="402" rx="14" fill="none" '
         'stroke="rgba(222,220,209,0.3)" stroke-width="1" stroke-dasharray="5 3"/>')
label(38, 154, "T9 Simulator", 15)

colx = [44, 318, 592]
cw = 264
ctr = [x + cw / 2 for x in colx]

# 1. pools
panel(32, 168, 836, 148)
label(46, 188, "1.  Pools", 12.5)
pools = [
    ("USER", ["A — user features", "region · OS · device"],
     ["LU1–LU6  latents", "archetype · propensities · LTV · interest"]),
    ("APP", ["B — app", "app_id · category"],
     ["LA1–LA2  latents", "app_quality · audience"]),
    ("CAMPAIGN", ["C — campaign", "advertiser · ad_genre"],
     ["LC1–LC2  latents", "creative · game quality"]),
]
for cx, (hd, g, p) in zip(ctr, pools):
    label(cx, 206, hd, 10.5, SECLBL, 600, "middle")
    box(cx - cw / 2, 212, cw, 44, GREEN, g)
    box(cx - cw / 2, 262, cw, 48, PURPLE, p)

arrow(450, 316, 450, 336)

# 2. auction + outcome generator
panel(32, 336, 836, 188)
label(46, 356, "2.  Auction + outcome generator", 12.5)
for cx, hd in zip(ctr, ["DSP", "MMP", "SSP"]):
    label(cx, 374, hd, 10.5, SECLBL, 600, "middle")
box(colx[0], 382, cw, 38, GREEN, ["A · B · C · D   context"], ts=11)
box(colx[0], 424, cw, 38, GREEN, ["E   click"], ts=11)
box(colx[0], 466, cw, 38, GREEN, ["H1 floor · H2 bid · H3 won"], ts=11)
box(colx[1], 382, cw, 38, PURPLE, ["F   install"], ts=11)
box(colx[1], 424, cw, 38, PURPLE, ["G   payer / LTV"], ts=11)
box(colx[2], 382, cw, 38, GREEN, ["H4   clearing price"], ts=11)
box(colx[2], 424, cw, 38, PURPLE, ["LU7   competing bid"], ts=11)

# ── tiers + optimiser ───────────────────────────────────────────────────
arrow(380, 532, 240, 560)
arrow(520, 532, 660, 560)
box(55, 562, 370, 78, BLUE, ["Tier 1 — User-value prediction",
    "→ P(click), P(install), P(payer), E(spend)"], ts=13, ss=11)
box(475, 562, 370, 78, BLUE, ["Tier 2 — Win probability",
    "→ P(win | bid, context)"], ts=13, ss=11)

arrow(240, 640, 350, 674)
arrow(660, 640, 550, 674)
label(295, 666, "EV", 10.5, ARROW, 600, "middle")
label(605, 666, "P(win|b)", 10.5, ARROW, 600, "middle")
box(250, 676, 400, 64, GRAY, ["Profit-max optimiser  (inference-only)",
    "bid* = argmax[ EV·P(win|b) − b·P(win|b) ]"], ts=12.5, ss=11)

s.append('</svg>')
OUT.write_text("\n".join(s), encoding="utf-8")
print("Wrote:", OUT)
