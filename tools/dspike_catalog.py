"""Build the double spike catalogue behind docs/double-spike-primer.html.

For every isotope system in ``src/doublespike/data/maininput.csv`` this script re-runs
the package's own optimal spike search, twice:

``pure``
    Every double spike the system allows, built from two isotopically pure single
    spikes, is tried over every choice of four inversion isotopes.  This is the
    theoretical best the element can do.

``real``
    Only the single spikes actually listed in the data file as available are used
    (these are the Oak Ridge / laboratory spikes, with their real impurities),
    and the inversion isotopes are searched over every four-isotope subset that
    contains *both* of the spike's dominant isotopes.  That restriction matters:
    with the inversion fixed to another element-wide "best" set, a spike whose
    isotope is not among the inversion isotopes is degenerate and propagates to a
    nonsensical error (a 64Zn-67Zn double spike evaluated on 64,66,68,70 comes out
    at 300 ppm/amu instead of 50).  Restricting to subsets that can actually see
    both spikes also keeps the enumeration at the same size as the pure case,
    instead of multiplying it by every four-isotope subset.

The numbers that end up in the primer -- the ranked spike pairs, the optimal
double spike composition, the optimal spike:sample proportion and the resulting
precision -- are written to ``docs/dspike_catalog.json``.  Run it with

    G:/Python39/python.exe tools/dspike_catalog.py

and the JSON is regenerated; the HTML primer embeds it verbatim, so the document
cannot drift away from what the code actually computes.

Everything uses the package's default error model (``IsoData.set_errormodel()``,
i.e. 10 V total beam, 8 s integration, 10^11 ohm resistors, 300 K) so that the
precisions are comparable between elements.  Absolute numbers scale with that
assumption; the *ranking* of spike pairs does not.
"""

import inspect
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doublespike import IsoData, optimalspike  # noqa: E402
from doublespike.isodata import default_data  # noqa: E402
from doublespike.optimal import singleoptimalspike  # noqa: E402

OUT = ROOT / "docs" / "dspike_catalog.json"

#: how many ranked rows to keep for each element and each spike type
NROWS = 12
#: an entry of the optimal spike composition above this counts as "one of the
#: two spike isotopes"; pure spikes are exact unit vectors so the gap is huge
INTERESTING = 1e-6


def spike_table(iso):
    """Identify every available single spike: which isotope is it enriched in?

    The obvious guess -- "the most abundant isotope in the column" -- is wrong for
    *nine* of the columns in the data file, because a commercial spike can be
    dominated by an isotope it is not enriched in.  The Oak Ridge 46Ca spike, for
    example, is 60.8% 40Ca and only 30.9% 46Ca; the 36S spike is 90.1% 32S.  A
    spike is defined by what it is *enriched in*, i.e. by
    ``column / standard``, not by its largest entry.

    That ratio alone is still not enough: the Os 192Os spike has 2.44x enrichment
    of 192Os but 2.5x of 184Os, simply because 184Os is nearly absent from the
    standard.  So the ratio is only compared over isotopes that make up at least
    1% of the column -- enough to exclude rounding-off traces, small enough to
    keep a 30.9% 46Ca in play.  Without that cutoff the count would be ten, and
    the extra one is that 184Os trace (0.05% of the column, 250x enriched).
    """
    out = []
    for j in range(iso.nrawspikes):
        col = np.asarray(iso.rawspike[j], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            enrich = np.where(iso.standard > 0, col / iso.standard, 0.0)
        enrich = np.where(np.isfinite(enrich), enrich, 0.0)
        strong = col >= 0.01 * col.max()
        i = int(np.argmax(np.where(strong, enrich, 0.0)))
        top = int(np.argmax(col))
        out.append(
            {
                "spike_index": j,
                # the isotope this spike is enriched in -- its name in the lab
                "isotope": int(iso.isonum[i]),
                "fraction": float(col[i]),
                "enrichment": float(enrich[i]),
                # the most abundant isotope in the column, which is what
                # "purity" used to be confused with
                "top_isotope": int(iso.isonum[top]),
                "top_fraction": float(col[top]),
            }
        )
    return out


def error_model_defaults():
    """The default error model, read out of ``IsoData.set_errormodel`` itself.

    The primer prints these numbers on its first screen ("intensity 10 V,
    integration 8 s, ..."), and they are the assumptions behind every number in
    it.  Typing them in here as literals would mean that changing a default in
    the package silently leaves the prose describing a model the code no longer
    uses, so they are introspected instead.

    ``radiogenic`` and ``R_reference`` are reported but do not characterise the
    model in the same way: ``radiogenic=None`` means "decide per element" (Pb,
    Sr, Hf, Os and Nd get an extra error term on the standard), and
    ``R_reference`` is only the resistance used to *describe* the beam, so a
    10 V total beam means 100 pA at 1e11 ohm.
    """
    params = inspect.signature(IsoData.set_errormodel).parameters
    return {
        "intensity_V": params["intensity"].default,
        "deltat_s": params["deltat"].default,
        "R_ohm": params["R"].default,
        "T_K": params["T"].default,
        "measured_type": params["measured_type"].default,
        "radiogenic": params["radiogenic"].default,
        "R_reference_ohm": params["R_reference"].default,
    }


def rows_of(iso, res, nrows=NROWS):
    """Turn one ``optimalspike`` result dict (pure spikes) into plain JSON rows."""
    out = []
    n = min(nrows, len(res["opterr"]))
    for k in range(n):
        spike = res["optspike"][k]
        # a pure double spike is a mixture of two unit vectors, so the pair is
        # simply where the composition is non-zero
        pair = [int(iso.isonum[i]) for i in np.nonzero(spike > INTERESTING)[0]]
        out.append(
            {
                "isoinv": [int(i) for i in res["optisoinv"][k]],
                "pair": sorted(int(p) for p in pair),
                "spike": [float(v) for v in spike],
                "prop": float(res["optprop"][k]),
                "err": float(res["opterr"][k]),
                "ppmperamu": float(res["optppmperamu"][k]),
                "spikeprop": None,
            }
        )
    return out


def real_rows(iso, spike_map, nrows=NROWS):
    """Rank the double spikes that can actually be made from the available spikes.

    Every pair of single spikes is tried against every four-isotope inversion set
    that contains both dominant isotopes of the pair.  ``optimalspike`` cannot
    express that constraint (it takes a single ``isoinv``), so the loop is written
    out here; each individual optimisation is the same ``singleoptimalspike`` call
    the package itself uses.
    """
    isonum = [int(v) for v in iso.isonum]
    dominant = [s["isotope"] for s in spike_map]
    rows = []
    for pair_idx in combinations(range(iso.nrawspikes), 2):
        want = {dominant[pair_idx[0]], dominant[pair_idx[1]]}
        if len(want) < 2:
            continue  # two spikes enriched in the same isotope cannot form one
        for subset in combinations(range(iso.nisos), 4):
            if not want <= {isonum[i] for i in subset}:
                continue
            try:
                spike, prop, err, spikeprop, ppm = singleoptimalspike(
                    iso, "real", list(pair_idx), list(subset)
                )
            except Exception:  # noqa: BLE001 - a failed combination just ranks last
                continue
            if not np.isfinite(ppm):
                continue
            used = np.nonzero(spikeprop > INTERESTING)[0]
            rows.append(
                {
                    "isoinv": [isonum[i] for i in subset],
                    "pair": sorted(int(i) for i in want),
                    "spike": [float(v) for v in spike],
                    "prop": float(prop),
                    "err": float(err),
                    "ppmperamu": float(ppm),
                    "spikeprop": [
                        {
                            "isotope": spike_map[int(i)]["isotope"],
                            "fraction": spike_map[int(i)]["fraction"],
                            "top_isotope": spike_map[int(i)]["top_isotope"],
                            "top_fraction": spike_map[int(i)]["top_fraction"],
                            "mix_fraction": float(spikeprop[int(i)]),
                        }
                        for i in used
                    ],
                }
            )
    rows.sort(key=lambda r: r["ppmperamu"])
    # keep every distinct spike pair's best result, then the global best rows, so
    # that the document can show both "what is the best spike" and "how does the
    # spike everyone actually uses compare"
    best_per_pair = {}
    for r in rows:
        best_per_pair.setdefault(tuple(r["pair"]), r)
    keep = rows[:nrows]
    for r in sorted(best_per_pair.values(), key=lambda r: r["ppmperamu"])[:nrows]:
        if r not in keep:
            keep.append(r)
    keep.sort(key=lambda r: r["ppmperamu"])
    return keep[: 2 * nrows]


def element_entry(element):
    iso = IsoData(element)
    iso.set_errormodel()
    entry = {
        "element": element,
        "isonum": [int(i) for i in iso.isonum],
        "mass": [float(m) for m in iso.mass],
        "standard": [float(v) for v in iso.standard],
        "spikes": spike_table(iso) if iso.nrawspikes else [],
        "ncombinations": None,
        "pure": [],
        "real": [],
        "note": None,
    }

    pure = optimalspike(iso, "pure")
    if not pure:
        entry["note"] = "no double spike possible"
        return entry
    entry["ncombinations"] = int(len(pure["opterr"]))
    entry["pure"] = rows_of(iso, pure)

    if iso.nrawspikes < 2:
        entry["note"] = "no single spikes in the data file, pure spikes only"
        return entry

    entry["real"] = real_rows(iso, spike_table(iso))
    if not entry["real"]:
        entry["note"] = "no usable combination of the available single spikes"
    return entry


def cost_hint(element):
    """Rough number of optimisations an element will need; used for scheduling."""
    iso = IsoData(element)
    return iso.nisos * iso.nrawspikes * max(iso.nisos - 2, 1) * max(iso.nrawspikes - 1, 1)


def _worker(element):
    t = time.time()
    return element, element_entry(element), time.time() - t


def main():
    wanted = [el for el in default_data if IsoData(el).nisos >= 4]
    # The real-spike search is the expensive half -- every pair of available
    # single spikes against every four-isotope inversion set that contains both
    # of its dominant isotopes -- and the elements are independent of each
    # other, so they are distributed over the available cores, longest first so
    # that the tail is short.  Without this the full run takes well over half an
    # hour on this four core machine.
    order = sorted(wanted, key=cost_hint, reverse=True)
    print("elements with at least four isotopes:", ", ".join(wanted), flush=True)
    print("scheduling order (estimated cost, high to low):", ", ".join(order), flush=True)

    import multiprocessing as mp

    catalog = {}
    t0 = time.time()
    with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
        for el, entry, dt in pool.imap_unordered(_worker, order, chunksize=1):
            catalog[el] = entry
            n = entry["ncombinations"]
            print(f"  {el:3s} {n:5d} pure combinations   {dt:6.1f} s", flush=True)
    # keep a stable key order in the JSON regardless of completion order
    catalog = {el: catalog[el] for el in wanted}

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generator": "tools/dspike_catalog.py",
        "package_version": __import__("doublespike").__version__,
        # Read the defaults out of the function rather than typing them in: the
        # document prints these, and a silently changed default would otherwise
        # leave the prose describing a model the code no longer uses.
        "errormodel": error_model_defaults(),
        "rows_kept": NROWS,
        "note": (
            "optimalspike is a local search; on the top ranked spikes it agrees with the "
            "original implementation to machine precision, on the worst ranked ones by "
            "up to about 1%."
        ),
        "elements": catalog,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}  ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
