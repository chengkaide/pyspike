"""Build the single file ``docs/double-spike-primer.html``.

The document is generated, not hand maintained: the prose lives in
``docs/primer_template.html`` and the numbers/tables come from
``docs/dspike_catalog.json``.  This script glues them together and inlines the
matplotlib figures, so the result is one self contained HTML file with no
network access and no external assets.

    G:/Python39/python.exe tools/dspike_catalog.py     # the numbers
    G:/Python39/python.exe tools/build_primer.py       # the document

Re-run both after changing the isotope data file or the error model; the
document then cannot drift away from what the code computes.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

TEMPLATE = ROOT / "docs" / "primer_template.html"
CATALOG = ROOT / "docs" / "dspike_catalog.json"
OUT = ROOT / "docs" / "double-spike-primer.html"

FIGS = {
    "<!--FIG_FE_CURVES-->": (
        "fig_fe_errorcurves",
        "图 6 · 六种 Fe 双稀释剂的误差曲线（<code>errorcurve</code> 的输出，纵轴为 ppm/amu）。"
        "每条曲线都用该组合<strong>自己的</strong>最优稀释剂组成画，这样比较才公平。"
        "实线是理想纯稀释剂，虚线是 Oak Ridge 的市售稀释剂。"
        "纵轴上限截到了 220 ppm/amu：<sup>54</sup>Fe–<sup>58</sup>Fe 的最低点在 20% 附近而不是 50%，"
        "而且曲线在最低点两侧升得比另外两条陡得多——它虽然被广泛使用过，但容错性最差。",
    ),
    "<!--FIG_FE_SURFACE-->": ("fig_fe_errorsurface", None),
}


def clean_svg(text):
    """Turn a matplotlib SVG file into an inlineable fragment."""
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # let the page control the size; keep the viewBox
    text = re.sub(r'\s(width|height)="[^"]*"', "", text, count=2)
    return text.strip()


def main():
    if not CATALOG.exists():
        sys.exit(f"missing {CATALOG}: run tools/dspike_catalog.py first")

    catalog_text = CATALOG.read_text(encoding="utf-8")
    catalog = json.loads(catalog_text)
    # `<` cannot appear in the JSON we write, but be explicit rather than lucky:
    # the payload goes inside a <script> element.
    embedded = catalog_text.replace("<", "\\u003c")

    import dspike_figures  # noqa: E402  (needs ROOT/src on sys.path)

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*CATALOG_JSON*/", embedded)
    html = html.replace("<!--TBL_OVERVIEW-->", '<div id="tbl-overview"></div>')

    # Figures are rendered straight into memory -- no scratch directory, so
    # there is nothing to clean up and nothing for ``git add -A`` to trip over.
    # (An earlier version wrote them to a temporary directory next to the
    # repository; on Windows that cleanup sometimes silently failed and left the
    # directory behind.)
    svgs = {
        "fig_fe_errorcurves": dspike_figures.fe_error_curves(),
        "fig_fe_errorsurface": dspike_figures.fe_error_surface(),
    }
    for placeholder, (name, caption) in FIGS.items():
        svg = clean_svg(svgs[name])
        if caption:
            svg = f'<figure class="fig">\n{svg}\n<figcaption>{caption}</figcaption>\n</figure>'
        html = html.replace(placeholder, svg)
        if placeholder in html:
            sys.exit(f"placeholder {placeholder} not substituted")

    for placeholder in ("/*CATALOG_JSON*/", "<!--TBL_OVERVIEW-->", "<!--FIG_FE_CURVES-->",
                        "<!--FIG_FE_SURFACE-->"):
        if placeholder in html:
            sys.exit(f"placeholder {placeholder} left in the output")

    OUT.write_text(html, encoding="utf-8")
    n_el = len(catalog["elements"])
    print(f"wrote {OUT}  ({len(html) / 1024:.0f} KB, {n_el} isotope systems)")


if __name__ == "__main__":
    main()
