"""Generate the isotope data and the reference values used by the web GUI.

    webgui/src/data.js         -- every isotope system in data/maininput.csv,
                                  in the same normalised form that IsoData uses
    webgui/src/reference.json  -- error estimates and inversions computed with
                                  the Python library, so that
                                  tools/test_webgui_math.js can prove the
                                  JavaScript port agrees with it

Usage (from the repository root):

    PYTHONPATH=src python webgui/make_data.py
"""

import json
import os

import numpy as np

import doublespike as ds

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
os.makedirs(SRC, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. the element data
# ---------------------------------------------------------------------------
elements = {}
for el, d in ds.isodata.default_data.items():
    raw = d["rawspike"]
    elements[el] = {
        "isonum": [int(v) for v in d["isonum"]],
        "mass": [float(v) for v in d["mass"]],
        "standard": [float(v) for v in d["standard"]],
        "rawspike": [] if raw is None else [[float(v) for v in row] for row in raw],
    }

# four isotope systems first, then by number of isotopes, then alphabetical
order = sorted(
    elements,
    key=lambda e: (len(elements[e]["isonum"]) < 4, len(elements[e]["isonum"]), e),
)

with open(os.path.join(SRC, "data.js"), "w", encoding="utf-8") as f:
    f.write("// Isotope system data, generated from src/doublespike/data/maininput.csv\n")
    f.write("// by webgui/make_data.py -- do not edit by hand.\n")
    f.write("const ELEMENTS = ")
    json.dump({e: elements[e] for e in order}, f, separators=(",", ":"))
    f.write(";\nconst ELEMENT_ORDER = ")
    json.dump(order, f, separators=(",", ":"))
    f.write(";\n")

# ---------------------------------------------------------------------------
# 2. reference values for the JavaScript port
# ---------------------------------------------------------------------------
ref = {"elements": {}, "error_estimates": [], "inversions": []}

rng = np.random.default_rng(20260917)
for el in ["Fe", "Ca", "Mo", "Ni", "Sr", "Cd", "W", "Pb", "Ge"]:
    iso = ds.IsoData(el)
    iso.set_errormodel()
    n = iso.nisos
    isoinv = np.sort(iso.isonum[np.argsort(-iso.standard)[:4]])
    iso.isoinv = isoinv
    ref["elements"][el] = {"isoinv": [int(v) for v in isoinv]}

    spike = ds.optimalspike(iso, "pure")["optspike"][0]

    for _ in range(6):
        sp = spike if rng.random() < 0.5 else rng.uniform(0, 1, n)
        if sp.sum() == 0:
            continue
        prop = float(rng.uniform(0.05, 0.95))
        alpha = float(rng.uniform(-0.5, 0.5))
        beta = float(rng.uniform(-5, 5))
        er = [int(isoinv[3]), int(isoinv[0])]
        e, ppm = ds.errorestimate(iso, prop, sp, isoinv, None, alpha, beta)
        e2, ppm2 = ds.errorestimate(iso, prop, sp, isoinv, er, alpha, beta)
        ref["error_estimates"].append({
            "element": el,
            "spike": [float(v) for v in sp],
            "prop": prop,
            "alpha": alpha,
            "beta": beta,
            "isoinv": [int(v) for v in isoinv],
            "errorratio": er,
            "error": float(e),
            "ppmperamu": float(ppm),
            "error_ratio": float(e2),
            "ppmperamu_ratio": float(ppm2),
        })

    for _ in range(6):
        prop = float(rng.uniform(0.1, 0.9))
        alpha = float(rng.uniform(-0.4, 0.4))
        beta = float(rng.uniform(-6, 6))
        sample = ds.isodata.normalise_composition(iso.standard * iso.mass ** (-alpha))
        measured = ds.isodata.normalise_composition(
            (prop * spike + (1 - prop) * sample) * iso.mass**beta
        )
        out = ds.dsinversion(iso, measured, spike, isoinv)
        ref["inversions"].append({
            "element": el,
            "isoinv": [int(v) for v in isoinv],
            "spike": [float(v) for v in spike],
            "measured": [float(v) for v in measured],
            "true_alpha": alpha,
            "true_beta": beta,
            "true_prop": prop,
            "alpha": float(out["alpha"]),
            "beta": float(out["beta"]),
            "prop": float(out["prop"]),
            "lambda": float(out["lambda_ratio"]),
        })

with open(os.path.join(SRC, "reference.json"), "w", encoding="utf-8") as f:
    json.dump(ref, f)

print(f"wrote {os.path.join(SRC, 'data.js')} ({len(order)} elements)")
print(f"wrote {os.path.join(SRC, 'reference.json')} "
      f"({len(ref['error_estimates'])} error estimates, "
      f"{len(ref['inversions'])} inversions)")
