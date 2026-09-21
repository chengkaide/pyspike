#!/usr/bin/env python3
"""Check every link and every code block in README.md.

Two failure modes this catches, both of which have hit this repository:

* a relative link that points at a file which does not exist (renaming a file
  and forgetting the README is otherwise completely silent -- GitHub renders
  the link, the reader gets a 404);
* an unbalanced code fence, which swallows the rest of the file into one
  preformatted blob from the fence onwards.

Anchors are deliberately not checked: GitHub's anchor algorithm depends on
which characters it strips, and we do not want to encode a guess about it here.
If an intra-page anchor is ever needed, link to the section by name instead.

Exit code is non-zero if anything fails.

It also checks docs/index.html, the GitHub Pages landing page.  That page is the
opposite of the primer: hand written, not generated.  Its figures are therefore
the one place in this repository where a plotted number can drift away from the
data without anything else noticing, so every value on them is re-derived here --
the catalogue for the ranking bars, the package's own error model for the
precision curve.  Same principle as tools/check_primer.py, applied to the page
that a first-time visitor actually sees.
"""
import json
import math
import re
import sys
from pathlib import Path

# An alternative path can be given so that the checker itself can be tested
# against a deliberately broken copy.  Relative links in a README resolve
# against the directory holding it, so that is what ROOT means here.
README = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent.parent / "README.md"
)
ROOT = README.parent

failures = []
checked = 0


def check(ok, label, detail=""):
    global checked
    checked += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        failures.append(label)


text = README.read_text(encoding="utf-8")
lines = text.splitlines()

print("README.md")

# ---- code fences ----------------------------------------------------------
fences = [i for i, line in enumerate(lines, 1) if line.lstrip().startswith("```")]
check(
    len(fences) % 2 == 0,
    "代码围栏成对",
    f"{len(fences)} 个围栏，奇数说明有一个没闭合（第 {fences[-1]} 行）"
    if len(fences) % 2
    else "",
)

# ---- links ----------------------------------------------------------------
# [label](target) -- the label may itself contain brackets, hence the lazy match
# up to the last ] before the (.
links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text)
links = [(label, target) for label, target in links if not target.startswith("#")]
check(bool(links), f"找到 {len(links)} 个链接", "一个链接都没有，解析逻辑可能失效了")

for label, target in links:
    if target.startswith(("http://", "https://", "mailto:")):
        checked += 1
        print(f"  ok   {target}  (外部，不做可达性检查)")
        continue
    path = (ROOT / target.split("#")[0]).resolve()
    check(
        path.exists(),
        f"相对链接 [{label}]({target})",
        f"目标不存在: {target}",
    )

# ---- the primer must be reachable from the very top -----------------------
# This is the whole point of the layout, so it is worth asserting rather than
# trusting the eye: the primer link has to appear before any other section
# heading, and inside a blockquote callout.
body = [line for line in lines if not line.startswith("# ")]
first_primer = next(
    (i for i, line in enumerate(lines, 1) if "double-spike-primer.html" in line),
    None,
)
first_section = next(
    (i for i, line in enumerate(lines, 1) if line.startswith("## ")),
    None,
)
check(
    first_primer is not None,
    "README 里有 primer 链接",
    "找不到 double-spike-primer.html",
)
if first_primer is not None and first_section is not None:
    check(
        first_primer < first_section,
        "primer 出现在首屏（位于任何二级标题之前）",
        f"primer 在第 {first_primer} 行，第一个二级标题在第 {first_section} 行"
        " -- 被埋到正文里，读者没有理由往下翻",
    )
if first_primer is not None:
    check(
        lines[first_primer - 1].lstrip().startswith(">"),
        "primer 链接带引用块样式（视觉上突出）",
        "不在引用块里，只是普通段落",
    )

# ---- the primer must be linked from exactly one place ---------------------
# Guards against the description being left behind in a second section, which
# is what happened when the primer was moved to the top: the old "Documentation"
# heading stayed put with its own copy of the same paragraph.
#
# Two earlier versions of this check were wrong, and both are worth remembering:
# the first matched an exact English sentence, so it reported a false failure
# against any README that worded it differently; the second counted the sections
# that merely *mention* the filename, and tripped over the Testing section, which
# legitimately names `docs/double-spike-primer.html` in prose.  Counting markdown
# *links* to it is the distinction that actually matters: prose may name the
# file, but the reader should be sent there from exactly one place.
primer_links = [target for _, target in links if target == "docs/double-spike-primer.html"]
check(
    len(primer_links) == 1,
    "primer 只被链接一处",
    f"有 {len(primer_links)} 处链接指向 primer，说明移动时留下了重复段落",
)

# ---- the GitHub Pages landing page ----------------------------------------
# docs/index.html is the only hand written file in docs/ (everything else there
# is generated), and it points at the primer by its filename.  Renaming the
# primer therefore breaks the published site silently -- exactly the failure
# mode this script exists to catch.  Its own relative links are checked against
# docs/, not against the repository root.
LANDING = ROOT / "docs" / "index.html"
if LANDING.exists():
    print()
    print("docs/index.html")
    page = LANDING.read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"]+)"', page)
    refs = [
        r for r in refs
        if not r.startswith(("#", "http://", "https://", "mailto:", "data:"))
    ]
    check(bool(refs), f"找到 {len(refs)} 个站内引用", "一个都没有，解析逻辑可能失效了")
    for ref in refs:
        target = (LANDING.parent / ref.split("#")[0]).resolve()
        check(target.exists(), f"站内链接 {ref}", f"目标不存在: docs/{ref}")
    check(
        'lang="zh-CN"' in page,
        "落地页标明了中文（与 primer 语言一致）",
        "primer 是中文的，落地页的 lang 属性应与之一致",
    )

    # ---- 结构：图必须是 figure + svg + figcaption 三件套 -------------------
    svgs = re.findall(r"<svg\b.*?</svg>", page, re.S)
    check(
        len(svgs) == page.count("<figure") == page.count("<figcaption>") == page.count("</figcaption>"),
        f"每张图都是 figure + svg + figcaption 三件套（{len(svgs)} 张）",
        f"svg/figure/figcaption = {len(svgs)}/{page.count('<figure')}/{page.count('<figcaption>')}",
    )
    check(all('viewBox="' in s for s in svgs), "每张图都声明了 viewBox")

    # 内联 SVG 里出现 HTML 的 breakout 标签会让解析器提前跳出 SVG 上下文：
    # 之后所有图形都不渲染，剩下文字挤成一行，整张图看起来"塌了"。
    BREAK = ("sup|sub|span|var|i|b|p|div|code|em|strong|small|u|s|br|hr|li|ul|ol|table"
             "|tt|pre|h1|h2|h3|h4|h5|h6|center|font|nobr|ruby|listing|menu|body|head"
             "|meta|embed|img|big|blockquote|dd|dt|dl|a")
    hits = sorted({t for s in svgs for t in re.findall(r"<(" + BREAK + r")(?=[\s>/])", s)})
    check(not hits, "SVG 内没有 HTML 破出标签", f"发现 {hits}，整张图会塌掉")
    check(not re.search(r"@@[A-Z0-9_]+@@", page), "没有未替换的占位符")

    figs = [int(m) for m in re.findall(r"<figcaption><b>图 (\d+) ·", page)]
    check(
        figs == list(range(1, len(figs) + 1)) and len(figs) == len(svgs),
        f"图号 1..{len(svgs)} 连续，且每张图都有编号",
        f"读到的图号: {figs}",
    )

    # ---- 图 2：Fe 的 6 对纯稀释剂。标签、数值、条形长度三者都要对得上 -------
    print()
    print("docs/index.html -- 图上的数字")

    sys.path.insert(0, str(ROOT / "src"))
    from doublespike import IsoData

    cat = json.loads((ROOT / "docs" / "dspike_catalog.json").read_text(encoding="utf-8"))
    els = cat["elements"]

    n_el = len(els)
    n_spk = sum(len(v.get("spikes", [])) for v in els.values())
    n_real = sum(len(v.get("real", [])) for v in els.values())
    check(f"<b>{n_el}</b><span>个同位素体系</span>" in page,
          f"数字条：{n_el} 个同位素体系", "与 docs/dspike_catalog.json 不符")
    check(f"<b>{n_spk}</b><span>个实际可购稀释剂</span>" in page,
          f"数字条：{n_spk} 个实际可购稀释剂", "与目录里 spikes 的条目数不符")
    check(f"<b>{n_real}</b><span>组稀释剂组合的完整排名</span>" in page,
          f"数字条：{n_real} 组稀释剂组合", "与目录里 real 的行数不符")

    fe = sorted(els["Fe"]["pure"], key=lambda r: r["ppmperamu"])[:6]
    fe_scale = 480.0 / fe[-1]["ppmperamu"]
    # \s+ 而不是单个空格：手写 HTML 里为了对齐会多打空格，为此误报过一轮
    labels = re.findall(r'<text x="100" y="[\d.]+"\s+text-anchor="end" font-size="11.5"'
                        r'\s+fill="var\(--ink\)">(\d+)Fe–(\d+)Fe</text>', page)
    values = re.findall(r'<text x="[\d.]+" y="[\d.]+"\s+font-size="10.5"'
                        r'\s+fill="var\(--ink-soft\)">([\d.]+)</text>', page)
    widths = re.findall(r'class="grow" x="108" y="[\d.]+"\s+width="([\d.]+)"', page)
    check(
        len(labels) == len(values) == len(widths) == len(fe),
        f"图 2 读到 {len(fe)} 行条形",
        f"标签/数值/条形 = {len(labels)}/{len(values)}/{len(widths)}，解析逻辑可能失效了",
    )
    for i, row in enumerate(fe):
        want_w = round(row["ppmperamu"] * fe_scale, 1)
        got = (labels[i], float(values[i]), float(widths[i]))
        check(
            got[0] == (str(row["pair"][0]), str(row["pair"][1]))
            and abs(got[1] - row["ppmperamu"]) < 0.05
            and abs(got[2] - want_w) < 0.15,
            f"图 2 第 {i + 1} 行 {row['pair'][0]}Fe–{row['pair'][1]}Fe = "
            f"{row['ppmperamu']:.1f} ppm/amu",
            f"页面上是 {got[0]} / {got[1]} / 条长 {got[2]}（应为 {want_w}）",
        )

    spread = fe[-1]["ppmperamu"] / fe[0]["ppmperamu"]
    check(f"差 {spread:.1f} 倍" in page,
          f"图 2 标注的「最好的与最差的差 {spread:.1f} 倍」", "倍数与目录不符")

    SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    sup = lambda n: str(n).translate(SUP)  # noqa: E731
    sr = sorted(els["Sr"]["pure"], key=lambda r: r["ppmperamu"])
    sr_best, sr_worst = sr[0]["pair"], sr[-1]["pair"]
    sr_spread = sr[-1]["ppmperamu"] / sr[0]["ppmperamu"]
    check(
        f"{sup(sr_worst[0])}Sr–{sup(sr_worst[1])}Sr" in page
        and f"{sup(sr_best[0])}Sr–{sup(sr_best[1])}Sr" in page
        and f"差 <b>{sr_spread:.1f} 倍</b>" in page,
        f"图注里的 Sr 例子：{sr_worst} 与 {sr_best} 差 {sr_spread:.1f} 倍",
        "稀释剂对或倍数与目录不符",
    )

    # ---- 图 4：精度曲线。49 个采样点逐点重算，外加被标注的那个点 ----------
    iso = IsoData("Fe")
    iso.set_errormodel()
    em = iso.errormodel["measured"]
    a, b = em["a"][0], em["b"][0]
    X0, X1, Y0, Y1 = 70.0, 620.0, 200.0, 56.0
    x_of = lambda I: X0 + (I - 1.0) / 99.0 * (X1 - X0)  # noqa: E731
    y_of = lambda p: Y0 - (p / 50.0) * (Y0 - Y1)        # noqa: E731
    ppm = lambda I: 1e6 * math.sqrt(a + b * I) / I      # noqa: E731

    m = re.search(r'id="sigcurve".*?points="([^"]+)"', page, re.S)
    got_pts = [tuple(float(v) for v in p.split(",")) for p in m.group(1).split()] if m else []
    want_pts = [(x_of(I), y_of(ppm(I))) for I in (1.0 + k * (99.0 / 48.0) for k in range(49))]
    max_dev = max((max(abs(gx - wx), abs(gy - wy))
                   for (gx, gy), (wx, wy) in zip(got_pts, want_pts)), default=9e9)
    check(
        len(got_pts) == 49 and max_dev < 0.15,
        "图 4 的 49 个采样点逐点由误差模型现算",
        f"读到 {len(got_pts)} 点，最大偏差 {max_dev:.2f} px",
    )

    dot = re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)"', page)
    check(
        dot is not None and abs(float(dot.group(2)) - y_of(ppm(10))) < 0.6,
        "图 4 高亮圆点落在 10 V 的曲线上",
        f"圆点在 y={dot.group(2) if dot else '?'}，曲线的 10 V 处是 y={y_of(ppm(10)):.1f}",
    )
    check(f"{ppm(10):.1f} ppm" in page, f"图 4 标注「10 V 束流 → {ppm(10):.1f} ppm」")

    share = 100 * a / (a + b * 10)
    check(
        f"热噪声只占方差的 {share:.0f}%" in page and f"占了 <b>{100 - share:.0f}%</b>" in page,
        f"图 4 的方差占比：热噪声 {share:.0f}%，计数统计 {100 - share:.0f}%",
        "与 a、b 的实算比例不符",
    )
    check(
        f"1 伏的 {ppm(1):.0f} ppm" in page and f"100 伏的 {ppm(100):.1f} ppm" in page,
        "图 4 的图片说明与曲线首末点一致",
        f"曲线实算是 {ppm(1):.0f} ppm 和 {ppm(100):.1f} ppm",
    )

# ---- README 里那张图必须与落地页同步 --------------------------------------
# The README's hero figure is lifted out of docs/index.html with the page's
# palette substituted in (see tools/make_hero_svg.py).  Edit the page and the
# committed SVG silently becomes a picture of the old figure -- the same class
# of drift this whole script exists to catch.
hero = ROOT / "docs" / "figure-hero.svg"
check(hero.exists(), "README 引用的 docs/figure-hero.svg 存在",
      "README 顶部会显示一个坏图")
if hero.exists():
    sys.path.insert(0, str(ROOT / "tools"))
    import make_hero_svg  # noqa: E402  (同目录的构建脚本，与它共用一套配色解析)

    check(
        make_hero_svg.build() == hero.read_text(encoding="utf-8"),
        "figure-hero.svg 与 docs/index.html 同步",
        "index.html 改过而图没重新生成，跑 python tools/make_hero_svg.py",
    )

print()
print(f"{checked - len(failures)}/{checked} 项通过")
if failures:
    print("失败项:", ", ".join(failures))
    sys.exit(1)
print("README.md 一致")
