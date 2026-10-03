"""Five lifecycle-stage diagrams: how observables / latent ground truth / outcome
labels relate at each stage. One SVG per stage, SAME spatial template so they are
comparable — what changes is which arrows are live, their DIRECTION (colour), and
each role's AVAILABILITY (box shading + status pill).

Stages: 1 design, 2 generation, 3 training, 4 testing, 5 production.
Pure-stdlib SVG; calls svg_checks.check_layout before writing each file.
Output: <PKG_ROOT>/Schema diagrams/Lifecycle_S{1..5}_*.svg
"""
from xml.sax.saxutils import escape

from t9sim.diagrams.svg_checks import check_layout
from t9sim.paths import PKG_ROOT

OUT = PKG_ROOT / "Schema diagrams"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1180, 740

# master geometry: name -> (x, y, w, h, title, sub)
GEO = {
    "OBS":   (40, 100, 280, 120, "OBSERVABLES  (X)",
              "A user · B app · C campaign · D time\nkeys: app_id/campaign_id · context: slot/time"),
    "LAT":   (450, 90, 290, 142, "LATENT GROUND TRUTH  (L)",
              "LU1-7 · LA1/2 · LC1/2\nestimands p_*, e_ltv, ev_truth"),
    "OUT":   (855, 100, 285, 120, "OUTCOME LABELS  (Y)",
              "E click · F install · G payer/LTV\nH3 won · H4 clearing"),
    "CENSOR":(615, 250, 215, 38, "censor.view (per condition)", ""),
    "MODEL": (450, 310, 290, 92, "MODEL",
              "Tier-1 4-head hurdle\nTier-2 AFT  ->  P(win|b)"),
    "PRED":  (800, 310, 330, 82, "PREDICTIONS",
              "EV_hat = product of 4 heads\nP(win | b)"),
    "ACT":   (800, 448, 330, 66, "ACTION",
              "bid* = argmax (EV_hat - b)*P(win|b)"),
    "EVAL":  (420, 486, 360, 96, "EVALUATOR  (yardstick)",
              "AUC/log-loss/RMSE · ev_ratio/spearman\nprofit/ROAS vs oracle/H2"),
}

# status -> (fill, stroke, pill text, pill colour)
STATUS = {
    "designed":       ("#FFFFFF", "#90A4AE", "DESIGNED — law only", "#546E7A"),
    "realised":       ("#E8F5E9", "#2E7D32", "REALISED — generator owns", "#2E7D32"),
    "realised_out":   ("#E8F5E9", "#2E7D32", "REALISED on ALL rows", "#2E7D32"),
    "input":          ("#E3F2FD", "#1565C0", "INPUT — the only features", "#1565C0"),
    "live":           ("#E3F2FD", "#1565C0", "LIVE INPUT (only X)", "#1565C0"),
    "target":         ("#E0F2F1", "#00838F", "TARGET — censored per condition", "#00838F"),
    "scoring_target": ("#E0F2F1", "#00838F", "SCORING TARGET — common all-rows", "#00838F"),
    "hidden":         ("#FCE4EC", "#C62828", "HIDDEN — forbidden (leakage guard)", "#C62828"),
    "yardstick":      ("#FFF8E1", "#F9A825", "YARDSTICK ONLY — evaluator, not a feature", "#F57F17"),
    "absent":         ("#ECEFF1", "#B0BEC5", "ABSENT — permanently unobservable", "#90A4AE"),
    "future":         ("#EDE7F6", "#6A1B9A", "FUTURE — partial feedback", "#6A1B9A"),
    "frozen":         ("#EFEBE9", "#6D4C41", "FROZEN", "#6D4C41"),
    "fitting":        ("#FFF3E0", "#EF6C00", "FITTING", "#EF6C00"),
    "gate":           ("#FFFDE7", "#E6C200", "", "#9E7E00"),
    "pred":           ("#F3E5F5", "#8E24AA", "", "#6A1B9A"),
    "action":         ("#E1F5FE", "#0277BD", "", "#0277BD"),
    "eval":           ("#FFF8E1", "#F9A825", "", "#F57F17"),
}

# edge kind -> (colour, width, dash) + legend text
EKIND = {
    "generative":  ("#1B5E20", "2.2", "",    "generative (build the world): L -> X, L -> Y"),
    "inert":       ("#BDBDBD", "1.6", "3,4", "designed but INERT (A perp L pre-BN)"),
    "lookup":      ("#616161", "1.6", "7,4", "key lookup: id -> entity latent"),
    "inferential": ("#1565C0", "2.1", "",    "inference: model consumes X  (X -> Y_hat)"),
    "target":      ("#00838F", "2.1", "",    "supervision / labels"),
    "scoring":     ("#6A1B9A", "2.1", "",    "predict / score"),
    "future":      ("#8E24AA", "2.1", "6,4", "after the bid (future / world)"),
    "feedback":    ("#EF6C00", "2.1", "6,4", "-> next training data"),
}

# per-stage spec: nodes {name:status}, geo overrides, edges, essence, notes
STAGES = [
  {"id": 1, "file": "Lifecycle_S1_design.svg", "title": "Stage 1 — Parameter & BN design",
   "essence": "Nothing realised yet: we author the joint law + parameters; the three roles exist only as designed distributions.",
   "nodes": {"OBS": "designed", "LAT": "designed", "OUT": "designed"},
   "edges": [
     ("LAT", "L0.30", "OBS", "R0.30", "generative", "archetype -> A/D (CPT tilt)"),
     ("OBS", "R0.72", "LAT", "L0.72", "lookup", "id -> entity latent (key)"),
     ("LAT", "R0.50", "OUT", "L0.50", "generative", "estimands -> labels ; EV -> LU7"),
   ],
   "notes": ["Latents are INVENTED here (iPinYou has no latent ground truth — only marginal shapes).",
             "Tilts are marginal-preserving (sum_k pi_k*delta=0) so generating latent-first keeps calibration.",
             "EV_hat is the PRODUCT of 4 heads, never a fitted head on ev_truth."]},

  {"id": 2, "file": "Lifecycle_S2_generation.svg", "title": "Stage 2 — Data generation (master)",
   "essence": "Ancestral sampling: latents drawn FIRST, observables constructed from them, labels realised on ALL rows. Nothing hidden yet.",
   "nodes": {"OBS": "realised", "LAT": "realised", "OUT": "realised_out"},
   "edges": [
     ("LAT", "L0.30", "OBS", "R0.30", "inert", "archetype -> A/D  (INERT: A perp L)"),
     ("OBS", "R0.72", "LAT", "L0.72", "lookup", "id -> entity latent (active)"),
     ("LAT", "R0.50", "OUT", "L0.50", "generative", "labels on ALL rows ; EV -> LU7"),
   ],
   "notes": ["Master is uncensored — every row has E/F/G; the only NaN is H4 on unsold rows.",
             "Generator OWNS the latents: 'latent' = hidden from the LEARNER, not from the simulator.",
             "Lost-row outcomes are the declared counterfactual (no formula reads `won`)."]},

  {"id": 3, "file": "Lifecycle_S3_training.svg", "title": "Stage 3 — Model training",
   "essence": "Fit P(Y|X) on the censored train split: model sees only X and per-condition-visible Y; latents are hidden and must be implicitly inverted.",
   "nodes": {"OBS": "input", "LAT": "hidden", "OUT": "target", "CENSOR": "gate", "MODEL": "fitting"},
   "edges": [
     ("OBS", "B0.55", "MODEL", "L0.30", "inferential", "features X"),
     ("OUT", "B0.12", "CENSOR", "T0.92", "target", "labels y"),
     ("CENSOR", "B0.40", "MODEL", "T0.72", "target", "won-only C1/C3 · all-rows C2/C4"),
     ("MODEL", "T0.20", "LAT", "B0.50", "inferential", "must invert X->L->Y (L unseen)"),
     ("LAT", "R0.62", "OUT", "L0.55", "inert", "gen. law (learned about, not seen)"),
   ],
   "notes": ["Dropped + leakage-guarded: latents (LU/LA/LC), estimands (p_*, e_ltv, ev_truth), user_id.",
             "Per-condition TRAIN population (won-only vs all-rows) IS the MMP ablation mechanism.",
             "Temporal split (days 1-20) precedes censoring; C3/C4 un-censor exact LU7 via H4 as the AFT label bound."]},

  {"id": 4, "file": "Lifecycle_S4_testing.svg", "title": "Stage 4 — Model testing / evaluation",
   "essence": "Frozen X->prediction maps scored on the COMMON all-rows test set; latents re-enter ONLY as the evaluator's yardstick, never as features.",
   "nodes": {"OBS": "input", "LAT": "yardstick", "OUT": "scoring_target",
             "MODEL": "frozen", "PRED": "pred", "EVAL": "eval"},
   "geo": {"EVAL": (760, 470, 380, 100, "EVALUATOR  (yardstick)",
                    "AUC/log-loss/RMSE · ev_ratio/spearman\nprofit/ROAS vs oracle/H2")},
   "edges": [
     ("OBS", "B0.55", "MODEL", "L0.30", "inferential", "features X (base_test)"),
     ("MODEL", "R0.50", "PRED", "L0.50", "scoring", "frozen forward"),
     ("PRED", "B0.50", "EVAL", "T0.55", "scoring", "predictions"),
     ("OUT", "B0.55", "EVAL", "R0.30", "target", "realised labels (all-rows)"),
     ("LAT", "B0.85", "EVAL", "L0.30", "scoring", "yardstick: ev_truth, lu7"),
   ],
   "notes": ["Latents come back — but only in scoring code (ev_truth, lu7); the feature lists are A/B/C/D only.",
             "ALL conditions scored on the same uncensored all-rows test set (no won-only restriction).",
             "oracle-features = diagnostic ceiling (value-funnel latents only); report profit + EV-bias, not raw ROAS."]},

  {"id": 5, "file": "Lifecycle_S5_production.svg", "title": "Stage 5 — Inference in production (AXON-analogue)",
   "essence": "At bid time only observables exist; frozen models -> EV_hat & P(win|b) -> one bid. Latents permanently absent; outcomes are in the future.",
   "nodes": {"OBS": "live", "LAT": "absent", "OUT": "future",
             "MODEL": "frozen", "PRED": "pred", "ACT": "action"},
   "edges": [
     ("OBS", "B0.55", "MODEL", "L0.30", "inferential", "live X"),
     ("MODEL", "R0.50", "PRED", "L0.50", "scoring", "frozen"),
     ("PRED", "B0.50", "ACT", "T0.50", "scoring", "EV_hat, P(win|b)"),
     ("ACT", "T0.55", "OUT", "B0.55", "future", "bid -> outcome (future)"),
     ("LAT", "R0.50", "OUT", "L0.40", "future", "world uses L (bidder blind)"),
     ("OUT", "B0.30", "MODEL", "T0.88", "feedback", "obs. = C1-C4 map -> next train set (loop TBD)"),
   ],
   "notes": ["Latents are permanently ABSENT; the oracle (latents-as-inputs) is impossible in production.",
             "id-keys recover frozen per-entity latents baked into the model (the only hidden-quality channel pre-BN).",
             "What returns (won-only vs all-rows, H4?) = the C1-C4 censoring map -> tomorrow's training data."]},
]


def _port(geo, spec):
    x, y, w, h = geo[:4]
    side, frac = spec[0], float(spec[1:])
    if side == "L": return (x, y + frac * h)
    if side == "R": return (x + w, y + frac * h)
    if side == "T": return (x + frac * w, y)
    return (x + frac * w, y + h)               # B


def render(stage):
    geo = dict(GEO)
    geo.update(stage.get("geo", {}))
    nodes = stage["nodes"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
    s.append('<defs>')
    for k, (col, *_rest) in EKIND.items():
        s.append(f'<marker id="m{k}" markerWidth="9" markerHeight="9" refX="7.5" '
                 f'refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{col}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    s.append(f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="21" '
             f'font-weight="700" fill="#212121">{escape(stage["title"])}</text>')
    s.append(f'<text x="{W/2:.0f}" y="54" text-anchor="middle" font-size="11.5" '
             f'fill="#546E7A">{escape(stage["essence"])}</text>')
    # zone captions for the three roles (clear strip above the boxes)
    s.append('<text x="180" y="84" text-anchor="middle" font-size="11" font-weight="700" '
             'fill="#90A4AE">OBSERVABLES</text>')
    s.append('<text x="595" y="78" text-anchor="middle" font-size="11" font-weight="700" '
             'fill="#90A4AE">LATENT GROUND TRUTH</text>')
    s.append('<text x="997" y="84" text-anchor="middle" font-size="11" font-weight="700" '
             'fill="#90A4AE">OUTCOME LABELS</text>')

    # edges first (behind nodes)
    used = []
    for src, sp, dst, dp, kind, lab in stage["edges"]:
        used.append(kind)
        x1, y1 = _port(geo[src], sp)
        x2, y2 = _port(geo[dst], dp)
        col, wdt, dash, _ = EKIND[kind]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                 f'stroke="{col}" stroke-width="{wdt}"{da} marker-end="url(#m{kind})"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 5
        s.append(f'<text x="{mx:.0f}" y="{my:.0f}" text-anchor="middle" font-size="9" '
                 f'fill="{col}" stroke="#FFFFFF" stroke-width="3" paint-order="stroke" '
                 f'font-weight="600">{escape(lab)}</text>')

    # nodes
    for name, status in nodes.items():
        x, y, w, h, title, sub = geo[name]
        fill, stroke, pill, pcol = STATUS[status]
        dash = ' stroke-dasharray="5,4"' if status in ("designed", "absent") else ""
        op = ' opacity="0.6"' if status == "absent" else ""
        s.append(f'<g{op}><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{dash}/>')
        s.append(f'<text x="{x+w/2:.0f}" y="{y+22:.0f}" text-anchor="middle" '
                 f'font-size="13" font-weight="700" fill="#1A1A1A">{escape(title)}</text>')
        for i, ln in enumerate(sub.split("\n") if sub else []):
            s.append(f'<text x="{x+w/2:.0f}" y="{y+40+i*14:.0f}" text-anchor="middle" '
                     f'font-size="9.5" fill="#37474F">{escape(ln)}</text>')
        if pill:
            s.append(f'<text x="{x+w/2:.0f}" y="{y+h-9:.0f}" text-anchor="middle" '
                     f'font-size="9.5" font-style="italic" font-weight="700" '
                     f'fill="{pcol}">[{escape(pill)}]</text>')
        s.append('</g>')

    # notes box
    ny = 600
    s.append(f'<rect x="40" y="{ny}" width="1100" height="78" rx="9" '
             f'fill="#F7F9FB" stroke="#CFD8DC" stroke-width="1.2"/>')
    s.append(f'<text x="52" y="{ny+19}" font-size="10.5" font-weight="700" '
             f'fill="#455A64">Key points</text>')
    for i, ln in enumerate(stage["notes"]):
        s.append(f'<text x="52" y="{ny+37+i*15:.0f}" font-size="10" '
                 f'fill="#37474F">- {escape(ln)}</text>')

    # legend (only kinds used this stage) + status note
    lx, ly = 40, 700
    seen = []
    for k in EKIND:
        if k in used and k not in seen:
            seen.append(k)
    for k in seen:
        col, _, dash, txt = EKIND[k]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{col}" '
                 f'stroke-width="2.4"{da} marker-end="url(#m{k})"/>')
        s.append(f'<text x="{lx+30}" y="{ly+4}" font-size="9.5" fill="#37474F">{escape(txt)}</text>')
        lx += 34 + len(txt) * 5.7
    s.append(f'<text x="40" y="{ly+20}" font-size="9.5" font-style="italic" '
             f'fill="#90A4AE">box colour + [pill] = the role\'s availability at this stage</text>')

    s.append('</svg>')
    used_geo = {n: geo[n] for n in nodes}
    check_layout(used_geo, name=stage["file"])
    (OUT / stage["file"]).write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / stage["file"])


if __name__ == "__main__":
    for st in STAGES:
        render(st)
