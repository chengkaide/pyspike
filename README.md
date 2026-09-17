# The double spike toolbox

Error propagation and data reduction for the double spike technique in mass spectrometry.

The workings of this package are described in:

Rudge J.F., Reynolds B.C., Bourdon B. The double spike toolbox (2009) Chem. Geol. 265:420-431
https://dx.doi.org/10.1016/j.chemgeo.2009.05.010

and at:

https://johnrudge.com/doublespike

## Installation

This is a standard python 3 package, which can be installed using pip:

```
pip install doublespike
```

## Usage

A series of Jupyter python notebooks describing usage can be found in the [src/](src/) folder. New
users should begin by looking at [src/example.ipynb](src/example.ipynb)

## Web GUI

[`webgui/doublespike-gui.html`](webgui/doublespike-gui.html) is a single file, completely offline
browser interface to the same calculations: pick an element, build a double spike, see the error
curves and the 2D precision map, rank every possible double spike, and reduce measured beam
intensities. Just open it in a browser -- no installation, no network access, no data leaves the
machine. See [webgui/README.md](webgui/README.md) for how it is built and how it is verified
against this Python package.

## What is in the box

| function | purpose |
|---|---|
| `IsoData` | isotope system data (masses, standard composition, available single spikes) |
| `dsinversion` | reduce a measured mixture to (alpha, beta, spike proportion) |
| `errorestimate` | linear error propagation: the precision of alpha, or of a chosen ratio |
| `errorestimate_many` | vectorised `errorestimate` over arrays of proportions and spikes |
| `optimalspike` | search every double spike composition for the best precision |
| `errorcurve`, `errorcurve2`, `errorcurve2d`, `errorcurveoptimalspike` | the standard plots |
| `monterun` | Monte Carlo simulation of a mass spectrometer run |
| `spike_calibration` | calibrate a double spike from spike-standard mixtures |
| `sensitivity` | derivatives for sensitivity analysis (appendix B of the paper) |

## Testing

```bash
pytest                        # if pytest is installed
python src/test/run_tests.py  # otherwise: a dependency free mini runner
```

Beyond the unit tests there are three verification scripts, all of which can be re-run:

```bash
PYTHONPATH=src python tools/verify.py      # forward/inverse closure + stress + Monte Carlo
PYTHONPATH=src python tools/benchmark.py   # timings
node webgui/test_math.js                   # the browser code vs this package
```

`tools/verify.py` checks that the forward model survives a round trip through the inversion for
every isotope system, that it still does so under extreme fractionations (where the original
solver failed on about 6% of cases), and that the linear error propagation agrees with Monte
Carlo.  `webgui/test_math.js` compares the JavaScript port against reference values computed by
this package: 437 comparisons, all agreeing to better than 1e-12 relative.

## Changes in 1.1

The numerics are unchanged in intent but the implementation is substantially faster and more
robust.  Every change is cross checked against 1.0 by re-running the old and new code side by side
on the same inputs.

### Inversion: a different iteration model

The double spike equations are **linear in lambda**, so lambda can be eliminated analytically.
Requiring the value of lambda implied by two different ratios to agree leaves a 2 x 2 system in
(alpha, beta), solved by a damped Newton iteration with an analytic Jacobian, a direction
preserving step limit and a backtracking line search on the residual. This replaces
`scipy.optimize.fsolve` and removes the ill conditioned `1/(1 - lambda)` reparameterisation.

| | 1.0 (`fsolve`) | 1.1 (lambda eliminated Newton) |
|---|---|---|
| one inversion | 214 us | **11 us** (vectorised, ~20x faster) |
| 20 000 inversions | ~4.3 s | **0.22 s** |
| iterations | variable, can hit `maxfev` | mean 2.6, max 3 |
| extreme conditions (beta in +-12, alpha in +-1) | 207 failures in 3432 (6.0%) | **0 failures in 24 000** |
| forward model recovered to | -- | 1.5e-11 |

`dsinversion` now processes a whole matrix of measurements in one vectorised call and reports
`converged` and `residual` per measurement.

### Vectorised error propagation

`errorestimate` accepts arrays of spike-sample proportions and spike compositions, so a 2D error
map is one call instead of `resolution**2` python level calls.

| | 1.0 | 1.1 |
|---|---|---|
| `errorcurve2d` at resolution 100 (10 000 points) | 12.2 s | **0.53 s** (23x faster) |
| 20 000 proportions | 25.8 s | **0.72 s** (36x faster) |
| 12 x 9 grid, python loop vs one call | 111 ms | **5.6 ms** |

A single scalar `errorestimate` call is the same speed as before (1.3 ms); that call is dominated
by numpy's per-call overhead, so the win only appears once you evaluate many points at once, which
is what the plots and the optimiser do.

### Optimal spike search

`optimalspike` evaluates a coarse grid in one batched call, then runs an adaptive per-axis pattern
search and a quadratic surface fit.  The two step sizes adapt independently because the error
surface is a narrow curved valley.  Same answers as before for the combinations that matter, and no
longer prints a traceback for the combinations that fail.

| | 1.0 | 1.1 |
|---|---|---|
| one (spike, mixing ratio) optimum | 0.23 s | **0.13 s** |
| every `real` spike pair for Ca (225 combinations) | 51 s | **31 s** |

`optimalspike` is a local search.  On the best ranked double spikes the two implementations agree to
machine precision; on the worst ranked combinations (tens of times less precise) they can differ by
up to ~1%, because they settle in different local minima of a nearly flat surface.

### Bugs fixed

* `plotting.errorcurve` passed a second positional argument to `set_title`, which mpl interprets as
  `fontdict`; the "error on a ratio" title was therefore broken.
* `changedenomcov` in the vectorised error propagation initially wrote into a temporary copy
  (`a[:, i, :][:, :, j] = ...`) and silently returned zeros.
* `normalise_composition` relied on `type(x) is float`, which is false for numpy scalars.
* `monte.monterun` used `isinstance(x, float)`, so numpy scalars, ints and 0-d arrays were handled
  inconsistently, and it was impossible to make a run reproducible. It now takes `seed=`.
* `monterun` returned a tuple containing `None` when only the spike had an error model.
* `pkg_resources` (deprecated by setuptools) replaced with `importlib.resources`.
* `IsoData.isoinv` shared its array with `IsoData.isonum`.
* `isoindex` used `np.vectorize`, which cost more than the entire error calculation; it is now a
  cached dictionary lookup.
* `optimalspike` printed a traceback for every combination that failed instead of ranking it last.

### Packaging

* `setup.cfg` folded into a modern PEP 621 `pyproject.toml`; `setup.cfg` removed.
* Python floor raised to 3.9, numpy floor to 1.20.
* CI updated to Python 3.9-3.13 and current action versions.
* `src/test/run_tests.py` runs the whole suite without pytest.
