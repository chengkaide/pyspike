"""Assemble the single file web GUI.

    webgui/template.html + webgui/src/{style.css,data.js,math.js,app.js}
        ->  webgui/doublespike-gui.html

The result is completely self contained: no network access, no dependencies, no
build tooling.  Open it by double clicking, or serve the directory with any
static file server.

Usage (from the repository root):

    python webgui/build.py

If ``src/data.js`` is missing, regenerate it first with::

    PYTHONPATH=src python webgui/make_data.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

PLACEHOLDERS = {
    "/*@STYLE@*/": "src/style.css",
    "/*@DATA@*/": "src/data.js",
    "/*@MATH@*/": "src/math.js",
    "/*@APP@*/": "src/app.js",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    template = read(os.path.join(HERE, "template.html"))
    for marker, rel in PLACEHOLDERS.items():
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            raise SystemExit(f"missing {rel}; run webgui/make_data.py first")
        template = template.replace(marker, read(path))

    left = re.findall(r"/\*@\w+@\*/", template)
    if left:
        raise SystemExit(f"unreplaced placeholders: {left}")

    out = os.path.join(HERE, "doublespike-gui.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
