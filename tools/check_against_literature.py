"""Audit the computed catalogue against numbers already published in the literature.

This is the honesty check behind docs/double-spike-primer.html.  Several papers
quote optimum double spike compositions to four significant figures, so if the
implementation is faithful those numbers have to come back out.

Two things are checked:

1.  **Literature anchors** (``literature`` section).  Each entry reproduces the
    exact configuration the paper used -- in particular the *same four inversion
    isotopes*.  Comparing a 43Ca-48Ca spike evaluated on 40,42,43,48 with a
    42Ca-43Ca spike evaluated on 40,42,43,44 is the only fair comparison, and
    getting this wrong is what produces nonsense like a 300 ppm/amu 64Zn-67Zn.

2.  **Internal consistency of ``docs/dspike_catalog.json``** (``audit`` section):
    errors sorted ascending, real-spike rows only from inversion sets that
    contain both spike isotopes, no NaN/inf, no empty rows.

    G:/Python39/python.exe tools/dspike_catalog.py            # first
    G:/Python39/python.exe tools/check_against_literature.py  # then
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doublespike import IsoData  # noqa: E402
from doublespike.errors import errorestimate  # noqa: E402
from doublespike.optimal import singleoptimalspike  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from dspike_catalog import spike_table  # noqa: E402  (shared spike identification)


def load_catalog():
    path = ROOT / "docs" / "dspike_catalog.json"
    if not path.exists():
        return {"elements": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def idx(iso, *masses):
    """Isotope mass numbers -> indices into the isotope arrays."""
    return [iso.isoindex(int(m)) for m in masses]


def spike_col(iso, mass):
    """Find the rawspike column that is *enriched in* ``mass``.

    Deliberately shares ``spike_table`` with the catalogue so the two can never
    disagree about which column is which spike.
    """
    for s in spike_table(iso):
        if s["isotope"] == mass:
            return s["spike_index"]
    raise KeyError(f"no single spike enriched in {mass}")


def optimum(element, type_, masses, inv, real_masses=None):
    """Run one optimisation exactly as a given paper would have configured it."""
    iso = IsoData(element)
    iso.set_errormodel()
    if type_ == "pure":
        isospike = idx(iso, *masses)
    else:
        isospike = [spike_col(iso, m) for m in (real_masses or masses)]
    spike, prop, err, _, ppm = singleoptimalspike(
        iso, type_, isospike, idx(iso, *inv)
    )
    return iso, spike, prop, ppm


def line(label, iso, pair, spike, prop, ppm):
    comp = "  ".join(
        f"{int(iso.isonum[i])}{iso.element}={100 * spike[i]:.2f}%"
        for i in np.nonzero(spike > 1e-6)[0]
    )
    print(f"  {label:<28} {comp:<44} "
          f"mix {100 * prop:5.2f}:{100 * (1 - prop):5.2f}   {ppm:7.2f} ppm/amu")
    return comp


def best_error_for_composition(element, comp, inv):
    """The best achievable precision *for one fixed double spike composition*.

    Used to ask how much is actually lost when a published optimum differs from
    ours: if the error surface is flat, a 1 percentage point difference in the
    spike composition costs almost nothing, and that is worth saying explicitly
    rather than hiding behind "the number did not match".
    """
    iso = IsoData(element)
    iso.set_errormodel()
    order = [int(v) for v in iso.isonum]
    vec = np.array([comp.get(m, 0.0) for m in order], dtype=float)
    vec = vec / vec.sum()
    ps = np.linspace(0.02, 0.98, 600)
    _, ppm = errorestimate(iso, ps, np.tile(vec, (len(ps), 1)), idx(iso, *inv))
    k = int(np.argmin(ppm))
    return float(ps[k]), float(ppm[k])


print("=" * 100)
print("1. LITERATURE ANCHORS")
print("=" * 100)

print("\n[Fe]  Rudge et al. (2009) Table 1, pure spikes")
print("      literature: 56Fe-58Fe = 77.28%:22.72%, mix 55.40:44.60, 57 ppm/amu")
print("                  54Fe-58Fe = 79.96%:20.04%, mix 21.48:78.52, 166 ppm/amu")
for m in [(56, 58), (54, 58)]:
    iso, sp, pr, ppm = optimum("Fe", "pure", m, (54, 56, 57, 58))
    line(f"Fe {m[0]}-{m[1]}", iso, m, sp, pr, ppm)

print("\n[Ge]  Woelfer et al. (2025), who ran this toolbox with inversion on 70,72,73,74")
print("      literature: 70Ge = 73.18% of the double spike, 52.15% spike in the mixture")
iso, sp, pr, ppm = optimum("Ge", "pure", (70, 73), (70, 72, 73, 74))
line("Ge 70-73 on 70,72,73,74", iso, (70, 73), sp, pr, ppm)
iso, sp2, pr2, ppm2 = optimum("Ge", "pure", (72, 76), (70, 72, 74, 76))
line("(the code's own best) 72-76", iso, (72, 76), sp2, pr2, ppm2)

print("\n[Ca]  is 43Ca-48Ca really about twice as good as 42Ca-43Ca?")
print("      literature (Rudge et al. 2009, as quoted in the Ca literature): about 2x")
ca_results = {}
for label, m, inv in [
    ("42Ca-43Ca on 40,42,43,44", (42, 43), (40, 42, 43, 44)),
    ("43Ca-48Ca on 40,42,43,48", (43, 48), (40, 42, 43, 48)),
    ("43Ca-48Ca on 40,43,44,48", (43, 48), (40, 43, 44, 48)),
    ("42Ca-48Ca on 40,42,43,48", (42, 48), (40, 42, 43, 48)),
    ("42Ca-48Ca on 40,42,44,48", (42, 48), (40, 42, 44, 48)),
    ("42Ca-46Ca on 40,42,43,46", (42, 46), (40, 42, 43, 46)),
    ("43Ca-46Ca on 40,42,43,46", (43, 46), (40, 42, 43, 46)),
]:
    iso, sp, pr, ppm = optimum("Ca", "pure", m, inv)
    ca_results[label] = (sp, pr, ppm)
    line(label, iso, m, sp, pr, ppm)
best_4243 = ca_results["42Ca-43Ca on 40,42,43,44"][2]
best_4348 = min(v[2] for k, v in ca_results.items() if k.startswith("43Ca-48Ca"))
print(f"      -> 43-48 / 42-43 = {best_4348 / best_4243:.2f}"
      f"  (a ratio of 0.5 would mean 43-48 is twice as good)")

print("\n[Zn]  Rudge et al. cocktail list: optimal 64Zn/67Zn with ORNL spikes is 4.88")
iso = IsoData("Zn")
iso.set_errormodel()
j64, j67 = spike_col(iso, 64), spike_col(iso, 67)
sp, pr, err, _, ppm = singleoptimalspike(iso, "real", [j64, j67], idx(iso, 64, 66, 67, 68))
r = sp[idx(iso, 64)[0]] / sp[idx(iso, 67)[0]]
line("ORNL 64Zn-67Zn on 64,66,67,68", iso, (64, 67), sp, pr, ppm)
print(f"      -> 64Zn/67Zn in the double spike = {r:.2f}   (literature: 4.88)")
print(f"      spike used: 64Zn column purity {100 * iso.rawspike[j64][idx(iso, 64)[0]]:.2f}%, "
      f"67Zn column purity {100 * iso.rawspike[j67][idx(iso, 67)[0]]:.2f}%")

print("\n[Sr]  a published 84Sr-87Sr double spike has 84Sr 41.94%, 87Sr 45.53%")
print("      (Geochemical Perspectives Letters: 'close to the optimum of the cocktail list')")
iso, sp, pr, ppm = optimum("Sr", "real", (84, 87), (84, 86, 87, 88))
line("ORNL 84Sr-87Sr", iso, (84, 87), sp, pr, ppm)

print("\n[Ni]  61Ni-62Ni (used in practice) vs 60Ni-62Ni (higher peak precision)")
print("      literature: 60-62 is slightly better at the optimum but much more")
print("      sensitive to the sample:spike ratio; 61-62 has a broad minimum")
iso = IsoData("Ni")
iso.set_errormodel()
for label, pair in [("61Ni-62Ni", (61, 62)), ("60Ni-62Ni", (60, 62))]:
    sp, pr, err, _, ppm = singleoptimalspike(
        iso, "real", [spike_col(iso, pair[0]), spike_col(iso, pair[1])],
        idx(iso, 58, 60, 61, 62))
    # how much worse is it if the mixture is 10 percentage points off?
    rough = []
    for dp in (-0.10, 0.0, 0.10):
        e, _ = errorestimate(iso, min(max(pr + dp, 0.01), 0.99), sp, idx(iso, 58, 60, 61, 62))
        rough.append(e)
    line(f"{label}", iso, pair, sp, pr, ppm)
    print(f"        error at mix {100 * pr:.0f}% is {rough[1]:.2e}; "
          f"10 points lower {rough[0]:.2e} (+{100 * (rough[0] / rough[1] - 1):.0f}%), "
          f"10 points higher {rough[2]:.2e} (+{100 * (rough[2] / rough[1] - 1):.0f}%)")

print("\n[Ti]  Millet & Dauphas (2014): 47Ti-49Ti, ca. 50% of each spike, ca. 52% sample")
iso, sp, pr, ppm = optimum("Ti", "pure", (47, 49), (46, 47, 48, 49))
line("Ti 47-49 on 46,47,48,49", iso, (47, 49), sp, pr, ppm)
iso, sp, pr, ppm = optimum("Ti", "pure", (47, 49), (47, 48, 49, 50))
line("Ti 47-49 on 47,48,49,50", iso, (47, 49), sp, pr, ppm)

print("\n[HOW MUCH DOES A 1 POINT DIFFERENCE COST?]  published composition vs our optimum")
for label, element, comp, inv, ours in [
    ("Ge 70-73", "Ge", {70: 0.7318, 73: 0.2682}, (70, 72, 73, 74), 37.61),
    ("Sr 84-87", "Sr",
     {84: 0.4194, 86: 0.0228, 87: 0.4553, 88: 0.1026}, (84, 86, 87, 88), 44.72),
]:
    p_lit, ppm_lit = best_error_for_composition(element, comp, inv)
    lit = ", ".join(f"{k}={100 * v:.2f}%" for k, v in comp.items())
    print(f"  {label}: literature composition ({lit})")
    print(f"      best mix for it is {100 * p_lit:.1f}% spike, giving {ppm_lit:.2f} ppm/amu;"
          f"  our optimum gives {ours:.2f} ppm/amu"
          f"  -> {100 * (ppm_lit / ours - 1):+.1f}%")


print()
print("=" * 100)
print("2. INTERNAL AUDIT of docs/dspike_catalog.json")
print("=" * 100)
CAT = load_catalog()
if not CAT["elements"]:
    print("  (no catalogue yet -- run tools/dspike_catalog.py first)")
    sys.exit(0)
problems = []
n_rows = 0
for el, e in CAT["elements"].items():
    for kind in ("pure", "real"):
        rows = e.get(kind) or []
        n_rows += len(rows)
        prev = -np.inf
        for r in rows:
            if not np.isfinite(r["ppmperamu"]):
                problems.append(f"{el}/{kind}: non-finite error")
            if r["ppmperamu"] < prev - 1e-12:
                problems.append(f"{el}/{kind}: errors not sorted ({r['ppmperamu']} after {prev})")
            prev = r["ppmperamu"]
            if len(r["pair"]) != 2:
                problems.append(f"{el}/{kind}: pair is {r['pair']}")
            if kind == "real" and not set(r["pair"]) <= set(r["isoinv"]):
                problems.append(
                    f"{el}/real: inversion set {r['isoinv']} misses the spike {r['pair']}")
        if rows and kind == "real" and e.get("spikes"):
            if len(rows) < 2:
                problems.append(f"{el}/real: suspiciously few rows ({len(rows)})")

print(f"  rows audited: {n_rows}")
if problems:
    for p in problems[:25]:
        print("  PROBLEM:", p)
    print(f"  {len(problems)} problem(s)")
else:
    print("  no problems found")
