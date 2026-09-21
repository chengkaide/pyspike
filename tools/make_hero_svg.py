#!/usr/bin/env python3
"""Lift the landing page's hero figure out as a standalone SVG for the README.

docs/index.html is hand written and its figures colour themselves with the
page's CSS variables.  Those cannot survive being lifted out: an SVG loaded as
an image has no access to the page's stylesheet, so every ``var(--x)`` would
resolve to nothing and the figure would come out black.

This script reads the light palette out of ``:root`` in index.html, substitutes
it into the first figure, strips the animation classes (there is no stylesheet
to animate them, and their default state is the finished state), and writes
``docs/figure-hero.svg``.  GitHub renders that file in README.md -- the
repository's front page, which until now was text only.

The palette is taken from index.html rather than repeated here, so the README
figure cannot drift away from the page it came from.  tools/check_readme.py
re-runs this transformation and fails if the committed file is stale.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "index.html"
OUT = ROOT / "docs" / "figure-hero.svg"


def build() -> str:
    page = PAGE.read_text(encoding="utf-8")

    root_block = re.search(r":root\s*\{(.*?)\}", page, re.S)
    if not root_block:
        sys.exit("index.html 里找不到 :root，调色板无法读取")
    palette = dict(re.findall(r"--([a-z-]+):\s*([^;]+);", root_block.group(1)))
    if len(palette) < 8:
        sys.exit(f"只解析出 {len(palette)} 个颜色变量，解析逻辑可能失效了")

    svg = re.search(r"<svg\b.*?</svg>", page, re.S)
    if not svg:
        sys.exit("index.html 里找不到 <svg>")
    svg = svg.group(0)

    # 一定要先长名后短名：var(--ink) 不会误伤 var(--ink-soft)（有右括号挡住），
    # 但反过来写就会把 --ink-soft 截断成 --ink 替换后剩下的 "-soft)"。
    for name in sorted(palette, key=len, reverse=True):
        svg = svg.replace(f"var(--{name})", palette[name].strip())

    left = re.findall(r"var\(--[a-z-]+\)", svg)
    if left:
        sys.exit(f"还有没解析的颜色变量: {sorted(set(left))}")

    svg = re.sub(r'\s+class="[^"]*"', "", svg)
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n"


def main() -> int:
    svg = build()
    if "--check" in sys.argv:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old == svg:
            print(f"{OUT.name} 与 index.html 一致（{len(svg)} 字节）")
            return 0
        print(f"{OUT.name} 已过期：index.html 改过而这张图没重新生成")
        print("重新生成：python tools/make_hero_svg.py")
        return 1
    OUT.write_text(svg, encoding="utf-8")
    print(f"已写出 {OUT.relative_to(ROOT)}（{len(svg)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
