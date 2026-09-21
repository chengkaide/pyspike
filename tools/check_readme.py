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
"""
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

# ---- the primer section must not be described twice -----------------------
n_desc = text.count("why two spikes are needed")
check(
    n_desc == 1,
    "primer 的说明只出现一次",
    f"出现了 {n_desc} 次，说明移动时留下了重复段落",
)

print()
print(f"{checked - len(failures)}/{checked} 项通过")
if failures:
    print("失败项:", ", ".join(failures))
    sys.exit(1)
print("README.md 一致")
