"""Layout sanity checks for the BESPOKE DAG SVG generators
(make_bn_dag / make_predictive_pathway / make_direct_relationships).

Those diagrams place every box at a hand-chosen (x, y), so nothing guarantees
separation. Call `check_layout(NODES, GROUPS, name=...)` just before writing the
SVG:

  * HARD-FAILS (AssertionError) if any two node rectangles overlap — so a hidden
    overlap aborts the generator instead of shipping.
  * WARNS (prints) if a group-title text row sits on top of a node rectangle
    (the "title half-hidden behind a box" failure we have shipped before).

Node/group tuples are expected as (x, y, w, h, ...) — only the first four are
read. The check is pure-stdlib; import it from the same diagrams/ directory.
"""
from __future__ import annotations


def _rect(v):
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


def _overlap(a, b):
    """True iff rectangles a, b share positive area (edge-touching is allowed)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def check_layout(nodes, groups=None, name="", title_dx=12, title_top=10,
                 title_h=12, char_w=6.2):
    """nodes: dict name->(x,y,w,h,...) OR iterable of (x,y,w,h,...).
       groups: iterable of (gx,gy,gw,gh,label); the title text row is modelled
       as the label's extent just under the group's top edge."""
    items = (list(nodes.items()) if isinstance(nodes, dict)
             else [(str(i), v) for i, v in enumerate(nodes)])
    rects = [(k, _rect(v)) for k, v in items]

    # 1) node vs node — HARD FAIL
    bad = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _overlap(rects[i][1], rects[j][1]):
                bad.append(f"{rects[i][0]} & {rects[j][0]}")
    assert not bad, (f"[layout FAIL] {name}: overlapping node boxes -> "
                     + "; ".join(bad))

    # 2) group-title text row vs node — WARN ONLY
    warns = []
    for g in (groups or []):
        gx, gy, _, _ = _rect(g)
        label = g[4] if len(g) > 4 else ""
        trow = (gx + title_dx, gy + title_top,
                max(len(label) * char_w, 10.0), title_h)
        for k, r in rects:
            if _overlap(trow, r):
                warns.append(f"title '{label[:28]}' overlaps node {k}")
    for w in warns:
        print(f"[layout WARN] {name}: {w}")

    print(f"[layout OK] {name}: {len(rects)} nodes, no box overlaps"
          + (f"; {len(warns)} title warning(s)" if warns else ""))
    return not warns
