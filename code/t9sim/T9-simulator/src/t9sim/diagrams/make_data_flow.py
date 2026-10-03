"""T9 data-flow architecture SVG (v4), in the dark 'v5' house style.

inputs -> 3 pools + auction/outcome gen -> Tier 1 / Tier 2 -> profit-max optimiser.
Output: <PKG_ROOT.parent>/t9_data_flow_v5.svg   (MSc Project root)   — pure stdlib.
"""
from xml.sax.saxutils import escape

from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT.parent / "t9_data_flow_v5.svg"

W, H = 1000, 1010
BG = "#0E1117"
FONT = "'Anthropic Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
# palette: (fill, stroke, title-text, sub-text)  — matches the v5 house style
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


def box(x, y, w, h, pal, lines, ts=13, ss=10, rx=10):
    fill, stroke, tcol, scol = pal
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1"/>')
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (ss + 4) / 2 - (3 if n > 1 else -4)
    for i, ln in enumerate(lines):
        fs, col, fw = (ts, tcol, 600) if i == 0 else (ss, scol, 400)
        dy = y0 + (0 if i == 0 else ts + (i - 1) * (ss + 4) + 2)
        s.append(f'<text x="{x+w/2:.0f}" y="{dy:.0f}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{fw}" fill="{col}">{escape(ln)}</text>')


def panel(x, y, w, h):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL[0]}" '
             f'stroke="{PANEL[1]}" stroke-width="0.8"/>')


def label(x, y, t, size=12.5, col=SECLBL, w=600, anchor="start"):
    s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" '
             f'text-anchor="{anchor}" fill="{col}">{escape(t)}</text>')


def arrow(x1, y1, x2, y2, col=ARROW, sw=1.5):
    s.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
             f'stroke="{col}" stroke-width="{sw}" marker-end="url(#a)"/>')


# ── inputs ──────────────────────────────────────────────────────────────
iw, iy, ih = 280, 26, 74
ix = [30, 350, 670]
box(ix[0], iy, iw, ih, GREEN, ["iPinYou dataset", "Demographics, prices,", "auction distributions"])
box(ix[1], iy, iw, ih, PURPLE, ["Industry reports", "Base CTR, install & payer", "rate, LTV, eCPM targets"])
box(ix[2], iy, iw, ih, PURPLE, ["Archetype design", "User / app / campaign", "behaviour · causal mechanisms"])
for x, col in zip(ix, ["#1F7A63", "#534AB7", "#534AB7"]):
    arrow(x + iw / 2, iy + ih, x + iw / 2, 150, col, 1.2)

# ── simulator container ────────────────────────────────────────────────
s.append(f'<rect x="20" y="150" width="960" height="548" rx="16" fill="none" '
         f'stroke="rgba(222,220,209,0.3)" stroke-width="1" stroke-dasharray="5 3"/>')
label(40, 176, "T9 Simulator", 16)

# 1. pools
panel(36, 192, 928, 196)
label(50, 214, "1.  Three pools  (user / app / campaign — pre-generated)   ·   10M users", 12.5)
colx = [52, 356, 660]
pools = [
    ("USER POOL",
     ["A — user features", "region · city · OS · OS-ver · device"],
     ["LU1–LU6  (latent)", "archetype · click/install/pay", "LTV mult · interest vector"]),
    ("APP POOL",
     ["B — app", "app_id · app_category"],
     ["LA1–LA2  (latent)", "app_quality · audience_profile"]),
    ("CAMPAIGN POOL",
     ["C — campaign", "advertiser · scale · campaign · ad_genre"],
     ["LC1–LC2  (latent)", "creative_appeal · game_quality"]),
]
for cx, (hd, g, p) in zip(colx, pools):
    label(cx + 142, 232, hd, 11, SECLBL, 600, "middle")
    box(cx, 238, 284, 60, GREEN, g, ts=11.5, ss=9.5)
    box(cx, 318, 284, 60, PURPLE, p, ts=11.5, ss=9.5)

arrow(500, 388, 500, 410)

# 2. auction + outcome generator
panel(36, 410, 928, 272)
label(50, 432, "2.  Auction + outcome generator   ·   up to 100M bids, 30M won", 12.5)
label(colx[0] + 142, 456, "DSP  (bid-side)", 11, SECLBL, 600, "middle")
label(colx[1] + 142, 456, "MMP  (post-install)", 11, SECLBL, 600, "middle")
label(colx[2] + 142, 456, "SSP  (auction)", 11, SECLBL, 600, "middle")
box(colx[0], 464, 284, 50, GREEN, ["A user · B app/slot/exchange · C campaign · D time"], ts=10.5)
box(colx[0], 520, 284, 44, GREEN, ["E — Click label"], ts=11.5)
box(colx[0], 570, 284, 44, GREEN, ["H1 floor · H2 bid · H3 won"], ts=11.5)
box(colx[1], 464, 284, 50, PURPLE, ["F — Install label"], ts=11.5)
box(colx[1], 520, 284, 50, PURPLE, ["G — Payer / LTV (G1–G4)"], ts=11.5)
box(colx[2], 464, 284, 50, GREEN, ["H4 — Clearing price"], ts=11.5)
box(colx[2], 520, 284, 50, PURPLE, ["LU7 — competing bid (latent)"], ts=10.5)

# ── Tier 1 / Tier 2 ────────────────────────────────────────────────────
arrow(420, 698, 250, 742)
arrow(580, 698, 752, 742)
box(60, 744, 384, 120, BLUE, [
    "Tier 1 — User-value prediction", "Features: A, B, C, D  (+ H1 floor)",
    "Labels: E, F, G", "→ P(click), P(install), P(payer), E(spend)"], ts=13, ss=11)
box(558, 744, 384, 120, BLUE, [
    "Tier 2 — Win probability", "Features: A, B, C, D, H1, H2, H4",
    "Label: H3 (won / lost)", "→ P(win | bid, context)"], ts=13, ss=11)

# ── optimiser ──────────────────────────────────────────────────────────
arrow(252, 864, 360, 906)
arrow(750, 864, 642, 906)
label(300, 884, "EV", 11, ARROW, 600, "middle")
label(700, 884, "P(win|b)", 11, ARROW, 600, "middle")
box(210, 908, 580, 86, GRAY, [
    "Profit-max optimiser  (inference-only, not trained)",
    "bid*  =  argmax [  EV · P(win|b)  −  b · P(win|b)  ]"], ts=13, ss=12)

s.append('</svg>')
# DETAILED data-flow (with field lists + scale numbers). The simpler structure
# overview is generated separately by make_architecture_svg.py -> T9_architecture.svg.
OUT.write_text("\n".join(s), encoding="utf-8")
print("Wrote:", OUT)
