"""End to end numerical verification of the double spike toolbox.

Three independent checks:

1. forward model -> inversion -> do we get the parameters back, for every
   isotope system that has at least four isotopes (small fractionations);
2. the same under extreme conditions (beta from -12 to +12, alpha from -1 to +1,
   spike proportions from 0 to 1), where the original fsolve based solver used to
   fail silently on about 6% of cases;
3. the error propagation is consistent with Monte Carlo simulation.

Usage (from the repository root):

    PYTHONPATH=src python tools/verify.py
"""

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import doublespike as ds
from doublespike.inversion import dscorrection_legacy
from doublespike.isodata import normalise_composition, ratio


def usable_systems():
    out = []
    for el, d in ds.isodata.default_data.items():
        if len(d["isonum"]) < 4:
            continue
        if np.any(d["standard"] <= 0) or np.any(d["mass"] <= 0):
            continue
        out.append(el)
    return out


def ratios(iso, measured, spike, isoinv):
    """Ratios as dsinversion would build them.

    The denominator has to be the isotope the spike contains most of -- the same
    choice dsinversion makes -- otherwise a spike that contains none of the first
    inversion isotope gives a division by zero and the comparison is meaningless.
    """
    idx = np.asarray(iso.isoindex(np.asarray(isoinv)))
    spike = np.asarray(spike)
    if np.any(spike[idx] < 0.001):
        k = int(np.argmax(spike[idx]))
        idx = np.concatenate(([idx[k]], idx[idx != idx[k]]))
    return (
        np.log(ratio(iso.mass, idx)),
        ratio(iso.standard, idx),
        ratio(spike, idx),
        ratio(np.asarray(measured), idx),
    )


def case(iso, isoinv, spike, prop, alpha, beta):
    sample = normalise_composition(iso.standard * iso.mass ** (-alpha))
    mixture = prop * spike + (1 - prop) * sample
    return normalise_composition(mixture * iso.mass**beta)


def check_closure():
    print("1. forward model -> inversion -> parameters recovered")
    rng = np.random.default_rng(20260917)
    elements = usable_systems()
    worst = 0.0
    n = 0
    bad = []
    for el in elements:
        iso = ds.IsoData(el)
        iso.set_errormodel()
        isoinv = np.sort(iso.isonum[np.argsort(-iso.standard)[:4]])
        iso.isoinv = isoinv
        opt = ds.optimalspike(iso, "pure")
        if not opt:
            continue
        spike = opt["optspike"][0]
        prop0 = float(opt["optprop"][0])
        for _ in range(25):
            prop = float(np.clip(prop0 + rng.uniform(-0.25, 0.25), 0.02, 0.98))
            alpha, beta = rng.uniform(-0.3, 0.3), rng.uniform(-6, 6)
            measured = case(iso, isoinv, spike, prop, alpha, beta)
            out = dsinversion_safe = ds.dsinversion(iso, measured, spike, isoinv)
            err = max(
                abs(out["alpha"] - alpha), abs(out["beta"] - beta), abs(out["prop"] - prop)
            )
            n += 1
            worst = max(worst, err)
            if not out["converged"] or err > 1e-7:
                bad.append((el, isoinv, prop, alpha, beta, err))
    print(f"   {n} cases over {len(elements)} isotope systems")
    print(f"   worst |error| = {worst:.3e}, failures = {len(bad)}")
    for row in bad[:5]:
        print("   FAIL", row)
    return not bad


def check_stress():
    print("2. extreme conditions")
    rng = np.random.default_rng(7)
    systems = {
        "Fe": [54, 56, 57, 58], "Ca": [40, 42, 44, 48], "Mo": [95, 97, 98, 100],
        "Ni": [58, 60, 61, 62], "Sr": [84, 86, 87, 88], "Cd": [106, 108, 110, 116],
    }
    total = worst = fails = 0
    legacy_fails = 0
    legacy_total = 0
    for el, isoinv in systems.items():
        iso = ds.IsoData(el)
        iso.isoinv = isoinv
        spike = ds.optimalspike(iso, "pure")["optspike"][0]
        n = 4000
        beta = rng.uniform(-12, 12, n)
        alpha = rng.uniform(-1.0, 1.0, n)
        prop = np.clip(rng.uniform(0.0, 1.0, n), 0.005, 0.995)
        sample = normalise_composition(iso.standard * iso.mass ** (-alpha[:, None]))
        mixture = prop[:, None] * spike + (1 - prop[:, None]) * sample
        measured = normalise_composition(mixture * iso.mass ** beta[:, None])

        out = ds.dsinversion(iso, measured, spike, isoinv)
        err = np.maximum.reduce([
            np.abs(out["alpha"] - alpha), np.abs(out["beta"] - beta),
            np.abs(out["prop"] - prop),
        ])
        fails += int(((~out["converged"]) | (err > 1e-6)).sum())
        total += n
        worst = max(worst, float(err.max()))

        P, nn, TT, mm = ratios(iso, measured, spike, isoinv)
        for i in range(0, n, 7):
            legacy_total += 1
            try:
                z = dscorrection_legacy(P, nn, TT, mm[i])
                if not np.isfinite(z).all() or abs(z[1] - alpha[i]) > 1e-6:
                    legacy_fails += 1
            except Exception:  # noqa: BLE001
                legacy_fails += 1
        print(f"   {el:3s} new solver failures {int(((~out['converged']) | (err > 1e-6)).sum())}/{n}")
    print(f"   new solver   : {fails} failures in {total}, worst error {worst:.3e}")
    print(f"   legacy fsolve: {legacy_fails} failures in {legacy_total} sampled points")
    return fails == 0


def check_monte_carlo():
    print("3. error propagation vs Monte Carlo")
    ok = True
    for el, spike, prop, alpha, beta in [
        ("Fe", [0, 0, 0.5, 0.5], 0.5, -0.2, 1.8),
        ("Mo", None, 0.5, 0.1, -0.5),
        ("Ni", None, 0.4, 0.0, 1.0),
    ]:
        iso = ds.IsoData(el)
        iso.set_errormodel()
        iso.isoinv = np.sort(iso.isonum[np.argsort(-iso.standard)[:4]])
        if spike is None:
            spike = ds.optimalspike(iso, "pure")["optspike"][0]
        measured = ds.monterun(iso, prop, spike, alpha, beta, n=4000, seed=1234)
        out = ds.dsinversion(iso, measured, spike)
        got = float(np.std(out["alpha"]))
        want = ds.errorestimate(iso, prop, spike, alpha=alpha, beta=beta)[0]
        ratio_ = got / want
        flag = "ok" if 0.9 < ratio_ < 1.1 else "OFF"
        ok &= 0.9 < ratio_ < 1.1
        print(f"   {el:3s} monte carlo 1SD {got:.6e}, linear prediction {want:.6e}, "
              f"ratio {ratio_:.3f} {flag}")
    return ok


def main():
    t0 = time.perf_counter()
    a = check_closure()
    b = check_stress()
    c = check_monte_carlo()
    print()
    print(f"all checks {'PASSED' if (a and b and c) else 'FAILED'} "
          f"in {time.perf_counter() - t0:.1f} s")
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    sys.exit(main())
