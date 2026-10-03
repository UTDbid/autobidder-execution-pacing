"""Markdown -> docx export with VISIBLE table borders ('all borders').

pandoc's default reference docx renders tables borderless in Word; Ken's
standing preference is all-borders on every table. This wraps the usual
pypandoc gfm export and post-processes word/document.xml, injecting a
<w:tblBorders> block (single, 0.5pt, auto colour) into every table's
<w:tblPr>, anchored right after <w:tblW> to respect the element order.

Optionally (--shade-groups) it also gives every run of rows sharing a first-column
value its own pale background, so a long table grouped by that column reads as
blocks rather than as one undifferentiated grid. Markdown cannot express cell
shading, so this is the only place it can happen. Tables whose first column is
all-distinct (an id column, not a group key) are left alone.

Usage:
  python -m t9sim.notebooks.docx_export <input.md> <output.docx> [--toc] [--shade-groups]
Or import:  from t9sim.notebooks.docx_export import export
"""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pypandoc

BORDERS = (
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '</w:tblBorders>'
)


# pale, print-safe tints; cycled in first-appearance order of the group key
GROUP_FILLS = ["EAF2FB", "EAF7EE", "FFF6E5", "F3EEF8", "FDEDEC", "E8F6F6",
               "F2F2F2", "FBF0F5"]
MAX_GROUPS = 12          # above this the first column is an id, not a group key


def _cell_text(tc_xml):
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", tc_xml)).strip()


def _shade_cell(tc_xml, fill):
    """Insert <w:shd/> into one cell, respecting CT_TcPr's element order."""
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
    if "<w:shd " in tc_xml:
        return tc_xml
    if "<w:tcPr/>" in tc_xml:
        return tc_xml.replace("<w:tcPr/>", f"<w:tcPr>{shd}</w:tcPr>", 1)
    if "<w:tcPr>" in tc_xml:
        # after tcBorders if present, else after tcW, else first inside tcPr
        for anchor in (r"(<w:tcBorders>.*?</w:tcBorders>)", r"(<w:tcW[^>]*/>)"):
            m = re.search(anchor, tc_xml, flags=re.S)
            if m:
                return tc_xml[:m.end()] + shd + tc_xml[m.end():]
        return tc_xml.replace("<w:tcPr>", f"<w:tcPr>{shd}", 1)
    return re.sub(r"(<w:tc>)", r"\1" + f"<w:tcPr>{shd}</w:tcPr>", tc_xml, count=1)


def _shade_one_table(tbl_xml):
    rows = re.findall(r"<w:tr\b.*?</w:tr>", tbl_xml, flags=re.S)
    if len(rows) < 3:
        return tbl_xml, 0
    body = rows[1:]                                   # row 0 is the header
    keys = [_cell_text(re.search(r"<w:tc>.*?</w:tc>", r, flags=re.S).group(0))
            if re.search(r"<w:tc>.*?</w:tc>", r, flags=re.S) else ""
            for r in body]
    distinct = list(dict.fromkeys(k for k in keys if k))
    if not distinct or len(distinct) > MAX_GROUPS or len(distinct) == len(keys):
        return tbl_xml, 0                             # id column, not a group key
    fill_of = {k: GROUP_FILLS[i % len(GROUP_FILLS)] for i, k in enumerate(distinct)}
    out, n = tbl_xml, 0
    for row, key in zip(body, keys):
        if key not in fill_of:
            continue
        new = re.sub(r"<w:tc>.*?</w:tc>",
                     lambda m: _shade_cell(m.group(0), fill_of[key]), row, flags=re.S)
        out = out.replace(row, new, 1)
        n += 1
    return out, n


def _shade_groups(xml):
    total_rows, total_tables = 0, 0
    for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", xml, flags=re.S):
        new, n = _shade_one_table(tbl)
        if n:
            xml = xml.replace(tbl, new, 1)
            total_rows += n
            total_tables += 1
    return xml, total_tables, total_rows


def _add_borders(docx_path, update_fields=False, shade=False):
    docx_path = Path(docx_path)
    tmp = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(docx_path) as z:
            z.extractall(tmp)
        doc = tmp / "word" / "document.xml"
        xml = doc.read_text(encoding="utf-8")
        n = len(re.findall(r"<w:tblW[^>]*/>", xml))
        # anchor after <w:tblW/> (order: ...tblW, jc, ..., tblBorders...)
        xml = re.sub(r"(<w:tblW[^>]*/>)", r"\1" + BORDERS, xml)
        # fallback: tblPr without tblW
        xml = re.sub(r"(<w:tblPr>(?:(?!</w:tblPr>|<w:tblW|<w:tblBorders).)*?)"
                     r"(</w:tblPr>)", r"\1" + BORDERS + r"\2", xml)
        shaded = (0, 0)
        if shade:
            xml, nt, nr = _shade_groups(xml)
            shaded = (nt, nr)
        doc.write_text(xml, encoding="utf-8")
        # make Word rebuild the (pandoc-empty) TOC field on open
        if update_fields:
            sett = tmp / "word" / "settings.xml"
            if sett.exists():
                s = sett.read_text(encoding="utf-8")
                if "w:updateFields" not in s:
                    s = re.sub(r"(<w:settings[^>]*>)",
                               r'\1<w:updateFields w:val="true"/>', s, count=1)
                    sett.write_text(s, encoding="utf-8")
        out = docx_path.with_suffix(".tmp.docx")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(tmp))
        out.replace(docx_path)
        return n, shaded
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def export(md_path, docx_path, toc=False, toc_depth=3, shade=False):
    extra = ["--toc", f"--toc-depth={toc_depth}"] if toc else []
    pypandoc.convert_file(str(md_path), "docx", format="gfm",
                          outputfile=str(docx_path), extra_args=extra)
    n, (nt, nr) = _add_borders(docx_path, update_fields=toc, shade=shade)
    msg = (f"{Path(docx_path).name}: written, all-borders applied to {n} tables"
           f"{' + table of contents' if toc else ''}")
    if shade:
        msg += f"; group shading on {nr} rows across {nt} tables"
    print(msg)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    export(pos[0], pos[1], toc="--toc" in sys.argv,
           shade="--shade-groups" in sys.argv)
