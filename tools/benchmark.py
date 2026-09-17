"""Timings for the double spike toolbox.

Usage (from the repository root):

    PYTHONPATH=src python tools/benchmark.py
"""

import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import doublespike as ds


def timed(label, fn, repeat=1):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    dt = (time.perf_counter() - t0) / repeat
    print(f"   {label:44s} {dt * 1e3:10.2f} ms")
    return out


def main():
    print("error propagation")
    iso = ds.IsoData("Fe")
    iso.spike = [0, 0, 0.5, 0.5]
    spike = [0, 0, 0.5, 0.5]
    er = timed("errorestimate, scalar", lambda: ds.errorestimate(iso, 0.5, spike))

    n = 20000
    props = np.linspace(0.05, 0.95, n)
    timed("errorestimate over 20 000 proportions",
          lambda: ds.errorestimate_many(iso, props, spike))

    ni = ds.IsoData("Ni")
    ni.isoinv = [58, 60, 61, 62]
    ni.set_errormodel()
    q = np.linspace(0.05, 0.95, 200)
    spikes = np.zeros((1, 200, 5))
    spikes[0, :, 2] = q
    spikes[0, :, 4] = 1 - q
    timed("errorestimate_many on a 200 x 200 grid",
          lambda: ds.errorestimate_many(ni, np.linspace(0.05, 0.95, 200)[:, None], spikes))

    import matplotlib
    matplotlib.use("Agg")
    timed("errorcurve2d (resolution 100)",
          lambda: ds.errorcurve2d(ni, "real", isospike=[60, 62]))

    print("optimal spike")
    ca = ds.IsoData("Ca")
    ca.set_errormodel()
    timed("optimalspike, one spike pair",
          lambda: ds.optimalspike(ca, "real", isoinv=[40, 42, 44, 48], isospike=[2, 5]))

    print("inversion")
    fe = ds.IsoData("Fe")
    fe.isoinv = [57, 58, 54, 56]
    sp = np.array([0.0, 0.0, 0.5, 0.5])
    N = 20000
    rng = np.random.default_rng(1)
    beta = rng.uniform(-2, 2, N)
    prop = rng.uniform(0.2, 0.8, N)
    mixture = prop[:, None] * sp + (1 - prop[:, None]) * fe.standard
    measured = ds.isodata.normalise_composition(mixture * fe.mass**beta[:, None])
    out = timed(f"dsinversion, {N} measurements",
                lambda: ds.dsinversion(fe, measured, sp, [57, 58, 54, 56]))
    print(f"   -> {int(out['converged'].sum())}/{N} converged, "
          f"worst |beta error| {np.max(np.abs(out['beta'] - beta)):.2e}")


if __name__ == "__main__":
    main()
