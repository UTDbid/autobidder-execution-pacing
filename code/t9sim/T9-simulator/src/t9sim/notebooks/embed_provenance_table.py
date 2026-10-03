"""Embed calibration/provenance_table.md into schema v7 + export docx (all-borders).

Uses t9sim.notebooks.docx_export.export so every table gets visible borders
(Ken's standing preference); never bare pypandoc. Guarded so importing the
module does NOT rewrite the schema — run it explicitly:

  python -m t9sim.notebooks.embed_provenance_table
"""
from t9sim.notebooks.docx_export import export
from t9sim.paths import CALIBRATION_DIR, PKG_ROOT


def main():
    root = PKG_ROOT.parent
    schema = root / "Simulator_Schema - June 10 (v7).md"
    table = (CALIBRATION_DIR / "provenance_table.md").read_text(encoding="utf-8")
    text = schema.read_text(encoding="utf-8")

    MARK_NEW = "### Data-source audit trail (all parameters)"
    MARK_OLD = "**Data-source audit trail (where the provenance trail lives).**"
    start = text.index(MARK_NEW) if MARK_NEW in text else text.index(MARK_OLD)
    end = text.index("\n\n### General notes", start)

    new = (
        "### Data-source audit trail (all parameters)\n\n"
        "Every number is traceable end-to-end through five persistent artifacts: "
        "(1) `config/*.yaml` — each parameter stored as "
        "`{value, route, source}` with its citation/rationale in-line; "
        "(2) `config/calibrated.yaml` — solved values of every auto-calibrated "
        "knob (dated, with target and sample size; the base file is never "
        "overwritten); (3) `provenance.csv` — emitted with **every** generated "
        "dataset, snapshotting all parameters as used in that run; "
        "(4) `Schema diagrams/Data_sources_by_feature.svg` — the per-feature "
        "source matrix (approved 10–11 Jun 2026); (5) "
        "`Industry_Source_Reports.md` §21 — verified citations with exact "
        "quotes and URLs. Schema versions v1–v6 are retained as the "
        "design-decision trail.\n\n"
        "The registry below — **50 parameter families covering all 122 leaf "
        "values** — is generated from the live configs by "
        "`make_provenance_table.py` (t9_sim; regenerate after any config "
        "change; auto-calibrated values shown as solved 18 Jun 2026). Grouped "
        "rows list per-archetype / per-category leaves; full source text lives "
        "in the YAMLs.\n\n" + table.rstrip() + "\n"
    )
    schema.write_text(text[:start] + new + text[end:], encoding="utf-8")
    print("embedded; schema now", len(schema.read_text(
        encoding="utf-8").splitlines()), "lines")

    out = root / "Simulator_Schema - June 10 (v7).docx"
    try:
        export(schema, out)
    except Exception:
        export(schema, root / "Simulator_Schema - June 10 (v7) rev2.docx")
        print("(main docx locked -> wrote rev2)")


if __name__ == "__main__":
    main()
