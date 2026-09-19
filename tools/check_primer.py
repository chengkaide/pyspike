"""Fail loudly if the primer drifts away from the code or the data.

``docs/double-spike-primer.html`` mixes three sources of truth, and each of them
can rot independently:

* numbers **rendered by JavaScript** from ``docs/dspike_catalog.json`` (the
  per-element cards, the overview table).  These cannot drift on their own, but
  the *prose around them* quotes the same numbers by hand.
* numbers **hand written** into ``docs/primer_template.html`` as literal HTML
  tables (the Ca table in section 6.2, the impure-spike table in 6.4, the
  literature-anchor table in 4.5).  These are the ones that rot: a catalogue
  re-run changes the truth and nobody notices the table.
* numbers **computed on the fly** by this script from the package itself.

So this file checks all three layers against each other:

    G:/Python39/python.exe tools/check_primer.py

Exit status is non-zero if anything disagrees, so it can gate a push.  It is
deliberately not part of the package test suite (``src/test/``) because it needs
the catalogue and takes about a minute.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from doublespike import IsoData, dsinversion  # noqa: E402
from doublespike.errors import errorestimate  # noqa: E402
from doublespike.isodata import default_data, elementarycharge, k  # noqa: E402
from doublespike.optimal import singleoptimalspike  # noqa: E402

# ``check_against_literature`` is written as a script and does all of its work at
# import time, so it cannot be imported here.  ``dspike_catalog`` is safe,
# because it keeps everything behind ``main()``.
from dspike_catalog import error_model_defaults, spike_table  # noqa: E402


def idx(iso, *masses):
    """Isotope mass numbers -> indices into the isotope arrays."""
    return [iso.isoindex(int(m)) for m in masses]


def spike_col(iso, mass):
    """The rawspike column that is *enriched in* ``mass``."""
    for s in spike_table(iso):
        if s["isotope"] == mass:
            return s["spike_index"]
    raise KeyError(f"no single spike enriched in {mass}")


def optimum(element, type_, masses, inv):
    """One optimisation, configured exactly as the primer describes it."""
    iso = IsoData(element)
    iso.set_errormodel()
    isospike = (idx(iso, *masses) if type_ == "pure"
                else [spike_col(iso, m) for m in masses])
    spike, prop, err, _, ppm = singleoptimalspike(iso, type_, isospike, idx(iso, *inv))
    return iso, spike, prop, ppm

TEMPLATE = ROOT / "docs" / "primer_template.html"
OUTPUT = ROOT / "docs" / "double-spike-primer.html"
CATALOG = ROOT / "docs" / "dspike_catalog.json"

FAILURES = []
CHECKS = [0]


def check(ok, label, detail=""):
    CHECKS[0] += 1
    if ok:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}   {detail}")
        FAILURES.append(f"{label}   {detail}")


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def rows_of_table(html, header_text):
    """The ``<tbody>`` rows of the table whose header contains ``header_text``."""
    out = []
    for tbl in re.findall(r"<table[^>]*>.*?</table>", html, re.S):
        if header_text in tbl:
            for body in re.findall(r"<tbody>(.*?)</tbody>", tbl, re.S):
                out.extend(re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S))
    return out


def cells(row):
    return [re.sub(r"<[^>]+>", "", c).strip() for c in re.split(r"</td>", row) if c.strip()]


# --------------------------------------------------------------------------
print("=" * 78)
print("0. INPUTS")
print("=" * 78)
for path in (TEMPLATE, OUTPUT, CATALOG):
    check(path.exists(), f"{path.name} 存在", str(path))
if FAILURES:
    sys.exit("missing inputs; build the document first")

tmpl = TEMPLATE.read_text(encoding="utf-8")
html = OUTPUT.read_text(encoding="utf-8")
cat = json.loads(CATALOG.read_text(encoding="utf-8"))
els = cat["elements"]


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("1. THE GENERATED FILE IS COMPLETE AND SELF CONTAINED")
print("=" * 78)

left = re.findall(r"<!--FIG_|/\*CATALOG_JSON\*/|<!--TBL_|<!--SEC_", html)
check(not left, "无未替换的占位符", str(left))

ext = re.findall(r'<img[^>]+src="(?:https?:|\.)', html)
ext += re.findall(r'<link[^>]+href="https?:', html)
ext += re.findall(r'<script[^>]+src=', html)
check(not ext, "无外部资源引用（离线可开）", str(ext[:3]))

check(len(re.findall(r"<svg\b", html)) == 7, "7 张图全部内联为 SVG",
      str(len(re.findall(r'<svg\b', html))))

n_figs = sorted(int(m) for m in re.findall(r"图 (\d+) ·", html))
check(n_figs == list(range(1, 8)), "图号 1..7 连续", str(n_figs))

for kw in ("作者是初学者", "AI 参与", "刚开始学"):
    check(kw in html, f"声明里含「{kw}」")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("2. CROSS REFERENCES RESOLVE")
print("=" * 78)

# every "图 N" mentioned in the prose must have a caption
mentioned_figs = {int(m) for m in re.findall(r"图 (\d+)", html)}
check(mentioned_figs <= set(n_figs), "正文引用的图号都有图注",
      str(sorted(mentioned_figs - set(n_figs))))

# every "§x.y" mentioned must be a real heading.  Section numbers live on the
# <h2>; the sub-numbers live on the <h3>.
heads = set(re.findall(r"<h3>([\d.]+)", html))
heads |= {m.split(".")[0] for m in heads}
heads |= set(re.findall(r'<span class="num">(\d+)</span>', html))
cited = set(re.findall(r"§([\d.]+)", html))
missing = {c for c in cited if c not in heads and c.split(".")[0] not in heads}
check(not missing, "正文引用的节号都存在", str(sorted(missing)))

# every in-page anchor target exists
ids = set(re.findall(r'id="([^"]+)"', html))
targets = {h for h in re.findall(r'href="#([^"]+)"', html) if h}
check(targets <= ids, "所有内部锚点都有对应 id", str(sorted(targets - ids)))

# table of contents count matches the number of top level sections
n_toc = len(re.findall(r'<li><a href="#', html.split("</nav>")[0]))
n_h2 = len(re.findall(r'<h2 id="', html))
check(n_toc == n_h2, f"目录条目数({n_toc}) = 章节数({n_h2})", f"{n_toc} vs {n_h2}")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("3. COUNTING CLAIMS IN THE PROSE")
print("=" * 78)

sys.path.insert(0, str(ROOT / "src"))
total = len(default_data)
ge4 = [e for e in default_data if IsoData(e).nisos >= 4]
with_spikes = [e for e in ge4 if IsoData(e).nrawspikes >= 2]
n_spikes = sum(len(v.get("spikes", [])) for v in els.values())
n_odd = sum(1 for v in els.values() for s in v.get("spikes", [])
            if s["isotope"] != s["top_isotope"])

print(f"  (data file: {total} elements, {len(ge4)} with >=4 isotopes, "
      f"{len(with_spikes)} with usable single spikes, {n_spikes} spikes, "
      f"{n_odd} mis-named)")

check(len(els) == len(ge4), f"目录覆盖全部 {len(ge4)} 个可用体系", f"{len(els)}")
check(f"共 {total} 个元素" in tmpl, f"正文写「共 {total} 个元素」")
check(f"{len(ge4)} 个的天然同位素数 ≥ 4" in tmpl, f"正文写「{len(ge4)} 个 …≥ 4」")
check(f"{len(with_spikes)} 个元素共列了 {n_spikes} 个可用单稀释剂" in tmpl,
      f"正文写「{len(with_spikes)} 个元素共列了 {n_spikes} 个单稀释剂」")
check(f"有 {n_odd} 个并不是以它名字里的那个同位素为主" in tmpl,
      f"正文写「有 {n_odd} 个并不是以它名字里的…」")
check(f"{total} 个元素的数据表本身" in tmpl, f"模块地图写「{total} 个元素的数据表」")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("4. HAND WRITTEN TABLES vs THE CATALOGUE")
print("=" * 78)

# --- section 6.4: spikes that are not dominated by the isotope in their name
odd_rows = rows_of_table(tmpl, "丰度最大的同位素")
check(len(odd_rows) == n_odd, f"§6.4 表格有 {n_odd} 行", str(len(odd_rows)))

by_key = {(k, s["isotope"]): s for k, v in els.items() for s in v.get("spikes", [])}
seen = set()
for row in odd_rows:
    m = re.search(r"([A-Za-z]{1,2}) 的 <sup>(\d+)</sup>", row)
    nums = re.findall(r'class="num">([\d.]+)%</td><td class="num">([\d.]+)×</td>'
                      r'<td class="num"><sup>(\d+)</sup>[A-Za-z]{1,2} ([\d.]+)%', row)
    if not m or not nums:
        check(False, f"§6.4 行格式可解析: {row[:60]}")
        continue
    el, mass = m.group(1), int(m.group(2))
    frac, enr, top, topf = (float(x) for x in nums[0])
    s = by_key.get((el, mass))
    if s is None:
        check(False, f"§6.4 {el}-{mass} 在目录里存在")
        continue
    seen.add((el, mass))
    check(close(frac, 100 * s["fraction"], 0.05), f"§6.4 {el}-{mass} 占比 {frac}%",
          f"catalogue {100 * s['fraction']:.1f}%")
    check(close(enr, s["enrichment"], 1.0), f"§6.4 {el}-{mass} 富集 {enr}×",
          f"catalogue {s['enrichment']:.0f}×")
    check(top == s["top_isotope"], f"§6.4 {el}-{mass} 丰度最大者 {top}",
          f"catalogue {s['top_isotope']}")
    check(close(topf, 100 * s["top_fraction"], 0.05), f"§6.4 {el}-{mass} 丰度 {topf}%",
          f"catalogue {100 * s['top_fraction']:.1f}%")

odd_set = {(el, iso) for (el, iso), s in by_key.items()
           if s["isotope"] != s["top_isotope"]}
check(seen == odd_set, "§6.4 覆盖了全部名不副实的稀释剂（不多不少）",
      f"table {sorted(seen)} vs catalogue {sorted(odd_set)}")


# --- section 4.5 anchors that can be read straight out of the catalogue
def best_row(el, pair, real=False):
    rows = els[el].get("real" if real else "pure", [])
    hits = [r for r in rows if tuple(r["pair"]) == tuple(pair)]
    return min(hits, key=lambda r: r["ppmperamu"]) if hits else None


for el, pair, ppm, mix in [("Fe", (56, 58), 57.1, (55.40, 44.60)),
                           ("Fe", (54, 58), 165.6, (21.48, 78.52))]:
    r = best_row(el, pair)
    if r is None:
        check(False, f"目录含 {el} {pair}")
        continue
    check(close(r["ppmperamu"], ppm, 0.05), f"目录 {el} {pair} = {ppm} ppm/amu",
          f"{r['ppmperamu']:.2f}")
    check(close(100 * r["prop"], mix[0], 0.005),
          f"目录 {el} {pair} 混合比 {mix[0]}:{mix[1]}",
          f"{100 * r['prop']:.2f}:{100 * (1 - r['prop']):.2f}")

r = best_row("Zn", (64, 67), real=True)
check(r is not None and close(100 * r["prop"], 58.19, 0.005),
      "目录 Zn 64-67(市售) 混合比 58.19:41.81",
      "missing" if r is None else f"{100 * r['prop']:.2f}")

r = best_row("Ge", (70, 73), real=True)
check(r is not None and close(r["ppmperamu"], 35.0, 0.2),
      "目录 Ge 70-73(市售) ≈ 35 ppm/amu（修正反演组合后的新值）",
      "missing" if r is None else f"{r['ppmperamu']:.2f}")

r = best_row("Zn", (64, 67), real=True)
check(r is not None and close(r["ppmperamu"], 53.6, 0.3),
      "目录 Zn 64-67(市售) ≈ 53.6 ppm/amu",
      "missing" if r is None else f"{r['ppmperamu']:.2f}")

# --- the ranking sentences in 6.1 and 6.3, which are prose claims about the
#     catalogue rather than literature anchors
fe_pure = sorted(els["Fe"]["pure"], key=lambda r: r["ppmperamu"])
check(tuple(fe_pure[0]["pair"]) == (56, 58), "§6.1 Fe 最优 = 56-58",
      str(fe_pure[0]["pair"]))
check(tuple(fe_pure[1]["pair"]) == (57, 58), "§6.1 Fe 次优 = 57-58",
      str(fe_pure[1]["pair"]))
check(tuple(fe_pure[3]["pair"]) == (54, 58), "§6.1 Fe 第四 = 54-58",
      str(fe_pure[3]["pair"]))
ratio = fe_pure[3]["ppmperamu"] / fe_pure[0]["ppmperamu"]
# the prose says "about 3x", so the tolerance has to be as loose as the word
# "about" -- the exact value is 2.90
check(close(ratio, 3, 0.2), "§6.1「54-58 与最优差约 3 倍」", f"computed {ratio:.2f}x")

zn_pure = sorted(els["Zn"]["pure"], key=lambda r: r["ppmperamu"])
check(tuple(zn_pure[0]["pair"]) == (66, 70), "§6.3 Zn 最优 = 66-70",
      str(zn_pure[0]["pair"]))
check(close(zn_pure[0]["ppmperamu"], 17, 0.5), "§6.3 Zn 66-70 ≈ 17 ppm/amu",
      f"computed {zn_pure[0]['ppmperamu']:.2f}")
check(tuple(zn_pure[1]["pair"]) == (67, 70), "§6.3 Zn 次优 = 67-70",
      str(zn_pure[1]["pair"]))


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("5. HAND WRITTEN TABLES vs A FRESH COMPUTATION")
print("=" * 78)

# --- section 6.2: the Ca table, which is the document's own result, not a
#     literature anchor
ca_claims = [
    ((42, 43), (40, 42, 43, 44), 60.0),
    ((43, 48), (40, 43, 44, 48), 36.7),
    ((42, 48), (40, 42, 44, 48), 34.9),
    ((43, 48), (40, 42, 43, 48), 143.3),
]
ca_rows = rows_of_table(tmpl, "最优精度")
print(f"  (§6.2 Ca 表：{len(ca_rows)} 行，逐行实算)")
ca_txt = " ".join(ca_rows)
for pair, inv, ppm in ca_claims:
    iso, sp, pr, got = optimum("Ca", "pure", pair, inv)
    check(close(got, ppm, 0.1),
          f"§6.2 {pair[0]}Ca-{pair[1]}Ca on {','.join(map(str, inv))} = {ppm}",
          f"computed {got:.2f}")
    check(f"{ppm:g} ppm/amu" in tmpl, f"§6.2 表里确实写着 {ppm:g} ppm/amu")

r_4243 = optimum("Ca", "pure", (42, 43), (40, 42, 43, 44))[3]
r_4348 = optimum("Ca", "pure", (43, 48), (40, 43, 44, 48))[3]
check(close(r_4243 / r_4348, 1.6, 0.05),
      "§6.2 结论「43-48 比 42-43 好约 1.6 倍」", f"{r_4243 / r_4348:.2f}")

# --- section 6.3: the Ni table, with its sensitivity columns
ni_claims = [
    ((60, 62), (43.7, 56.3), 33.9, 64, 50),
    ((61, 62), (64.1, 35.9), 48.7, 2, 3),
]
iso = IsoData("Ni")
iso.set_errormodel()
ni_txt = " ".join(rows_of_table(tmpl, "混合比偏低"))
for pair, mix, ppm, up, down in ni_claims:
    sp, pr, err, _, got = singleoptimalspike(
        iso, "real", [spike_col(iso, pair[0]), spike_col(iso, pair[1])],
        idx(iso, 58, 60, 61, 62))
    check(close(got, ppm, 0.1), f"§6.3 Ni {pair[0]}-{pair[1]} = {ppm} ppm/amu",
          f"computed {got:.2f}")
    check(close(100 * pr, mix[0], 0.05), f"§6.3 Ni {pair[0]}-{pair[1]} 混合比 {mix[0]}:{mix[1]}",
          f"computed {100 * pr:.1f}:{100 * (1 - pr):.1f}")
    for dp, expect, name in ((-0.10, up, "偏低"), (0.10, down, "偏高")):
        e, _ = errorestimate(iso, min(max(pr + dp, 0.01), 0.99), sp,
                             idx(iso, 58, 60, 61, 62))
        pct = 100 * (e / err - 1)
        check(close(pct, expect, 1.0),
              f"§6.3 Ni {pair[0]}-{pair[1]} 混合比{name} 10 个百分点 → +{expect}%",
              f"computed +{pct:.0f}%")
    check(f"{ppm} ppm/amu" in ni_txt, f"§6.3 表里确实写着 {ppm} ppm/amu")

# the prose sentence right below the table quotes the same two numbers
check("（33.9 对 48.7）" in tmpl, "§6.3 正文里「33.9 对 48.7」与表一致")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("6. THE EXPERIMENT SECTION'S NUMBERS")
print("=" * 78)

# section 2.2 claims: 50.0 mg of a rock with 6.0% Fe -> 0.0537 mmol, and a
# 1.24:1 spike:sample mole ratio needs 3.72 mg Fe, weighed as 1.33 g of solution
mmol = 50.0 * 0.060 / 55.845          # mg rock x mass fraction / (mg per mmol)
check(close(mmol, 0.0537, 0.00005), "§2.2 50.0 mg × 6.0% → 0.0537 mmol",
      f"computed {mmol:.5f}")
mg_fe = 1.24 * mmol * 55.845          # 1.24 mol of spike Fe per mol of sample Fe
check(close(mg_fe, 3.72, 0.01), "§2.2 摩尔比 1.24:1 → 需要 3.72 mg Fe",
      f"computed {mg_fe:.3f}")

# section 2.5 claims the mixed sample's 58Fe/56Fe is pushed from 0.31 to 15.2
# (figure 5 plots that ratio x100, which is where the decimal points come from)
iso_fe = IsoData("Fe")
iso_fe.set_errormodel()
sp, pr, err, _, ppm = singleoptimalspike(iso_fe, "pure", idx(iso_fe, 56, 58),
                                         idx(iso_fe, 54, 56, 57, 58))
nat = np.asarray(iso_fe.standard, dtype=float)
mix = (1 - pr) * nat + pr * sp
k56 = idx(iso_fe, 56)[0]
r_nat = 100 * nat[idx(iso_fe, 58)[0]] / nat[k56]
r_mix = 100 * mix[idx(iso_fe, 58)[0]] / mix[k56]
check(close(r_nat, 0.31, 0.005), "§2.5 天然 58Fe/56Fe = 0.31", f"computed {r_nat:.3f}")
check(close(r_mix, 15.2, 0.1), "§2.5 混合样 58Fe/56Fe = 15.2", f"computed {r_mix:.2f}")
check(close(r_mix / r_nat, 49, 1.0), "§2.5「抬高 49 倍」", f"computed {r_mix / r_nat:.1f}x")

# section 2.6 claims dsinversion returns prop = 0.5540 on that mixture
out = dsinversion(iso_fe, mix, spike=sp, isoinv=[54, 56, 57, 58])
check(close(out["prop"], 0.5540, 0.0005), "§2.6 反演还回 prop = 0.5540",
      f"computed {out['prop']:.4f}")
check(abs(out["alpha"]) < 1e-6 and abs(out["beta"]) < 1e-6,
      "§2.6 反演还回 α = β ≈ 0",
      f"alpha={out['alpha']:.2e} beta={out['beta']:.2e}")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("7. THE ERROR MODEL THE DOCUMENT ADVERTISES")
print("=" * 78)

# The meta line on the first screen states the assumptions behind every number
# in the document.  It is rendered from the catalogue, so the risk is not that
# the page disagrees with the catalogue -- it is that both drift away from the
# package.  Compare against the function signature itself.
advertised = cat["errormodel"]
defaults = error_model_defaults()
for key, value in defaults.items():
    check(key in advertised, f"目录 errormodel 含 {key}")
    check(advertised.get(key) == value,
          f"目录 errormodel.{key} = 包默认值", f"{advertised.get(key)!r} vs {value!r}")

# every field the page reads must exist, or the meta line shows "undefined"
fields = set(re.findall(r"DATA\.errormodel\.(\w+)", html))
fields |= set(re.findall(r"\bem\.(\w+)", html))
missing_fields = {f for f in fields if f not in advertised}
check(not missing_fields, "页面引用的 errormodel 字段都存在",
      f"missing {sorted(missing_fields)} from catalogue")

top_fields = set(re.findall(r"DATA\.(\w+)", html)) - {"errormodel"}
missing_top = {f for f in top_fields if f not in cat}
check(not missing_top, "页面引用的顶层字段都存在",
      f"{sorted(missing_top)} missing from catalogue, which has {sorted(cat)}")

check("fixed-total" in html, "页面写明了 measured_type = fixed-total")

# section 8's first bullet works through the noise model with real numbers, so
# those have to hold too.  Take the constants from the package, not from here.
noise_a = 4 * k * 300.0 * (1e11**2) / (8.0 * 1e11)
noise_b = elementarycharge * 1e11 / 8.0
sigma_10v = math.sqrt(noise_a + noise_b * 10.0)
check(close(noise_a, 2.1e-10, 0.05e-10), "§8 热噪声项 a ≈ 2.1e-10 V²",
      f"computed {noise_a:.3e}")
check(close(noise_b, 2.0e-9, 0.05e-9), "§8 计数统计项 b ≈ 2.0e-9 V",
      f"computed {noise_b:.3e}")
check(close(1e6 * sigma_10v / 10.0, 14, 0.5), "§8 单束流 ≈ 14 ppm",
      f"computed {1e6 * sigma_10v / 10.0:.2f} ppm")
check(close(100 * noise_a / (noise_a + noise_b * 10.0), 1, 0.3),
      "§8 热噪声约占 1%（计数统计占 99%）",
      f"computed {100 * noise_a / (noise_a + noise_b * 10.0):.2f}%")
check(close(1e12 * 10.0 / 1e11, 100, 0.5), "§8 10 V @ 1e11 Ω = 100 pA")
check("方程 (35)" in tmpl, "§8 引用的方程编号与代码注释一致（(35)，不是 (34)）")

# section 8 quotes what the *other* beam convention would have given, so that
# number has to hold too
iso_fe2 = IsoData("Fe")
iso_fe2.set_errormodel(measured_type="fixed-sample")
sp2, pr2, err2, _, ppm2 = singleoptimalspike(iso_fe2, "pure", idx(iso_fe2, 56, 58),
                                             idx(iso_fe2, 54, 56, 57, 58))
check(close(ppm2, 31.5, 0.1), "§8 fixed-sample 下 Fe 56-58 = 31.5 ppm/amu",
      f"computed {ppm2:.2f}")
check(close(100 * pr2, 90, 0.5), "§8 fixed-sample 最优混合比 90:10",
      f"computed {100 * pr2:.1f}:{100 * (1 - pr2):.1f}")


# --------------------------------------------------------------------------
print()
print("=" * 78)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS[0]} checks")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"all {CHECKS[0]} checks passed")
