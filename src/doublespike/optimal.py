"""Module for determining optimal double spikes."""

import itertools

import numpy as np

from scipy.special import binom

from .errors import errorestimate, errorestimate_many


def optimalspike(
    isodata,
    type_="pure",
    isospike=None,
    isoinv=None,
    errorratio=None,
    alpha=0.0,
    beta=0.0,
    verbose=False,
):
    """Find the optimal double spike composition and double spike-sample mixture proportions.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
                This is the only mandatory argument.
        type(str): type of spike, 'pure' or 'real'. Real spikes, such as those from
            Oak Ridge National Labs, contain impurities. See isodata.rawspike
            for their assumed compositions. By default pure spikes are used.
        isospike (array): the isotopes used in the double spike e.g. [54, 57].
            By default all choices of 2 isotopes are tried.
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the optimal double spike is chosen as that which
            minimises the error on the natural fractionation factor (known as
            alpha). Instead, the optimiser can be told to minimise the
            error on a particular ratio by setting errorratio. e.g.
            setting errorratio=[58, 56] will minimise the error on 58Fe/56Fe.
        alpha, beta (floats): there is a small dependance of the error on the fractionation
            factors (instrumental and natural, or alpha and beta). Values of alpha and
            beta can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large.
        verbose (bool): print a line for every combination that failed.  Off by
            default: a failed combination is ranked last rather than raising.

    Returns:
        All the outputs are provided as arrays in a dict. Each column represents an isotope
        (see isodata.isonum for the isotope numbers) e.g. for Fe the columns
        correspond to the isotopes 54Fe, 56Fe, 57Fe, 58Fe. The rows represent the
        different combinations of double spikes and isotopes being tried, in order of
        error: The first row is the best double spike, and the last row is the worst.
        optspike: the proportions of each isotope in the optimal double spike.
        optprop: the optimal proportion of spike in the double spike-sample mix.
        opterr: the error in the fractionation factor (or ratio if specified)
                for the optimal spike.
        optisoinv: the 4 isotopes used in the inversion.
        optspikeprop: the proportion of each raw spike in the optimal double spike.
        optppmperamu: an alternative expression of the error in terms of ppm per amu.

    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> opt = optimalspike(isodata_fe, 'pure')

    Note:
        The search for the optimum (spike proportion, spike composition) is a
        *local* one: a coarse grid picks the starting basin, then an adaptive
        per-axis pattern search and a quadratic surface fit refine it.  For the
        double spikes that matter -- the best few out of all combinations -- the
        results agree with the original implementation to machine precision.  On
        the worst ranked combinations, where the precision is tens of times worse
        than the best, the two implementations can disagree by up to about 1%
        because they converge to different local minima of a nearly flat surface.
    """
    # Check if isoinv is set in isodata
    if isoinv is None:
        if hasattr(isodata, "isoinv"):
            isoinv = isodata.isoinv

    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isospike = isodata.isoindex(isospike)
    isoinv = isodata.isoindex(isoinv)

    # If don't specify inversion isotopes, do all possible combinations
    if isoinv is None:
        isoinv = list(itertools.combinations(np.arange(isodata.nisos), 4))
    else:
        isoinv = list([np.array(isoinv)])

    # Work out all combinations of inversion isotopes and spiking isotopes
    isoinvvals = []
    isospikevals = []
    for i in range(len(isoinv)):
        if isospike is None:
            if type_ == "pure":
                # look at all combinations of spikes from the inversion isotopes
                isospikev = list(itertools.combinations(isoinv[i], 2))
            else:
                if isodata.nrawspikes == 0:
                    return {}  # can't proceed if no single spikes to use
                # look at all combinations of spikes from the all rawspikes
                isospikev = list(
                    itertools.combinations(np.arange(isodata.nrawspikes), 2)
                )
        else:
            isospikev = list([isospike])

        if isospikev is not None:
            isospikevals.append(isospikev)
            isoinvvals.append(np.tile(isoinv[i], (len(isospikev), 1)))
    isoinvvals = np.vstack(isoinvvals)
    isospikevals = np.vstack(isospikevals)

    optspikes = []
    optprops = []
    opterrs = []
    optppmperamus = []
    optspikeprops = []

    for i in range(len(isoinvvals)):
        try:
            optspike, optprop, opterr, optspikeprop, optppmperamu = singleoptimalspike(
                isodata,
                type_,
                isospikevals[i, :],
                isoinvvals[i, :],
                errorratio,
                alpha,
                beta,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("Error with:", isospikevals[i, :], isoinvvals[i, :])
                print(e)
            # fail gracefully: this combination simply ranks last
            optspike = np.zeros(isodata.nisos)
            optprop = 0.0
            opterr = np.inf
            optspikeprop = np.zeros(isodata.nisos)
            optppmperamu = np.inf

        optspikes.append(optspike)
        optprops.append(optprop)
        opterrs.append(opterr)
        optppmperamus.append(optppmperamu)
        optspikeprops.append(optspikeprop)

    optspike = np.vstack(optspikes)
    optspikeprop = np.vstack(optspikeprops)
    optprop = np.array(optprops)
    opterr = np.array(opterrs)
    optppmperamu = np.array(optppmperamus)
    optisoinv = isoinvvals

    # Sort in ascending order of error
    ix = np.argsort(opterr)

    # avoid masses of output by limiting to all possibilites in case of pure spikes
    max_noutput = min(len(ix), int(6 * binom(isodata.nisos, 4)))
    ix = ix[0:max_noutput]

    out = {
        "optspike": optspike[ix, :],
        "optprop": optprop[ix],
        "opterr": opterr[ix],
        "optisoinv": isodata.isonum[optisoinv[ix, :]],
        "optspikeprop": optspikeprop[ix, :],
        "optppmperamu": optppmperamu[ix],
    }

    return out


# ---------------------------------------------------------------------------
# the search for the optimum (p, q)
# ---------------------------------------------------------------------------
#: Initial step sizes for the pattern search in (spike proportion p, spike
#: composition q).  The two are adapted independently because the error surface
#: is strongly anisotropic: it is a narrow, curved valley, so the p direction
#: needs a much finer step than the q direction.
_H0 = 0.2
_GROW = 1.3
_SHRINK = 0.5
_STENCIL = (-1.0, 0.0, 1.0)
_P_FLOOR = 1e-3
#: grid used to pick the starting point, so that the pattern search does not have
#: to walk across the whole square if the best basin is far from (0.5, 0.5)
_COARSE = 7
#: step size at which the pattern search hands over to parabolic interpolation,
#: and the step size at which that stops
_H_PATTERN = 1e-3
_H_FINAL = 1e-12
#: factor by which the quadratic refinement stencil shrinks each sweep
_QUAD_SHRINK = 0.5
#: a quadratic step shorter than this fraction of the stencil triggers a
#: tightening of the stencil (otherwise the stencil is kept)
_QUAD_TRUST = 0.3
#: relative improvement required for a step to count (guards against floating
#: point noise keeping the search alive forever on a flat surface)
_IMPROVE = 1e-13


def _error_points(isodata, spike1, spike2, isoinv, errorratio, beta, alpha, ps, qs):
    """Error for a flat list of (p, q) points, in one vectorised call.

    Args:
        ps (array): spike-sample proportions, shape ``(M,)``
        qs (array): proportions of ``spike1`` in the double spike, shape ``(M,)``
    Returns:
        err (array): shape ``(M,)``
    """
    ps = np.asarray(ps, dtype=float)
    qs = np.asarray(qs, dtype=float)
    spike = (
        qs[:, np.newaxis] * spike1[np.newaxis, :]
        + (1.0 - qs)[:, np.newaxis] * spike2[np.newaxis, :]
    )
    err = errorestimate_many(isodata, ps, spike, isoinv, errorratio, alpha, beta)[0]
    with np.errstate(invalid="ignore"):
        err = np.where(np.isfinite(err), err, np.inf)
    return np.atleast_1d(err)


def _error_stencil(isodata, spike1, spike2, isoinv, errorratio, beta, alpha, ps, qs):
    """Error on the full (P x Q) grid, in one vectorised call."""
    Pg, Qg = np.meshgrid(
        np.asarray(ps, dtype=float), np.asarray(qs, dtype=float), indexing="ij"
    )
    return _error_points(
        isodata,
        spike1,
        spike2,
        isoinv,
        errorratio,
        beta,
        alpha,
        Pg.ravel(),
        Qg.ravel(),
    ).reshape(Pg.shape)


def _refine_optimum(
    isodata,
    spike1,
    spike2,
    isoinv,
    errorratio,
    alpha,
    beta,
    h0=_H0,
    h_pattern=_H_PATTERN,
    h_final=_H_FINAL,
    maxiter=200,
):
    """Find the (p, q) that minimises the error, in three stages.

    1. A coarse 7 x 7 grid, evaluated in a single vectorised call, picks the
       starting point.  This stops the search from having to crawl across the
       whole unit square when the best basin is far from (0.5, 0.5).
    2. An *adaptive per axis* pattern search on a 3 x 3 stencil.  The two step
       sizes are grown and shrunk independently, which is what the error surface
       requires: it is a narrow, curved valley, so the spike-sample proportion
       direction needs a far finer step than the spike composition direction.
    3. Successive parabolic interpolation along each axis, which locates the
       minimum to third order and so reaches machine precision from a step of
       only 1e-3 without thousands of extra evaluations.

    Returns:
        p, q, error -- optimal spike-sample proportion, optimal proportion of
        ``spike1`` in the double spike, and the error at that point.
    """
    grid = np.linspace(0.05, 0.95, _COARSE)
    err_grid = _error_stencil(
        isodata, spike1, spike2, isoinv, errorratio, beta, alpha, grid, grid
    )
    i, j = np.unravel_index(int(np.argmin(err_grid)), err_grid.shape)
    p0, q0 = float(grid[i]), float(grid[j])
    e0 = float(err_grid[i, j])

    h_p = h_q = _H0
    for _ in range(maxiter):
        if max(h_p, h_q) < h_pattern:
            break
        ps = np.clip(p0 + h_p * np.array(_STENCIL), _P_FLOOR, 1.0 - _P_FLOOR)
        qs = np.clip(q0 + h_q * np.array(_STENCIL), _P_FLOOR, 1.0 - _P_FLOOR)
        err = _error_stencil(
            isodata, spike1, spike2, isoinv, errorratio, beta, alpha, ps, qs
        )
        i, j = np.unravel_index(int(np.argmin(err)), err.shape)
        if np.isfinite(err[i, j]) and err[i, j] < e0 * (1.0 - _IMPROVE):
            p0, q0, e0 = float(ps[i]), float(qs[j]), float(err[i, j])
            h_p = min(h_p * _GROW, 0.5) if i != 1 else h_p * _SHRINK
            h_q = min(h_q * _GROW, 0.5) if j != 1 else h_q * _SHRINK
        else:
            h_p *= _SHRINK
            h_q *= _SHRINK

    # stage 3: iterated quadratic surface fit.
    #
    # Nine points on a 3 x 3 stencil are enough to fit
    #     f = d0 + d1 u + d2 v + d3 u^2 + d4 u v + d5 v^2
    # where (u, v) are the offsets divided by h, so the design matrix is well
    # scaled.  The stationary point of that quadratic is jumped to analytically
    # and the whole thing is repeated with a smaller stencil.  One vectorised
    # call covers all nine points, so machine precision is reached in a couple
    # of dozen evaluations, where a simplex method started from the same point
    # needs a few hundred.
    h = h_pattern
    U, V = np.meshgrid(np.array(_STENCIL, dtype=float), np.array(_STENCIL, dtype=float),
                       indexing="ij")
    design = np.column_stack([
        np.ones(U.size), U.ravel(), V.ravel(),
        (U ** 2).ravel(), (U * V).ravel(), (V ** 2).ravel(),
    ])
    #: cap on the quadratic stage: a well behaved surface converges in about 25
    #: sweeps, anything far beyond that means the surface is degenerate (some of
    #: the worst ranked spike pairs are) and further sweeps are wasted work
    max_quad = 60
    for _ in range(min(maxiter, max_quad)):
        if h < h_final:
            break
        ps = np.clip(p0 + U.ravel() * h, _P_FLOOR, 1.0 - _P_FLOOR)
        qs = np.clip(q0 + V.ravel() * h, _P_FLOOR, 1.0 - _P_FLOOR)
        f = _error_points(
            isodata, spike1, spike2, isoinv, errorratio, beta, alpha, ps, qs
        )
        if not np.all(np.isfinite(f)):
            h *= _SHRINK
            continue
        try:
            coef, *_ = np.linalg.lstsq(design, f, rcond=None)
        except np.linalg.LinAlgError:
            h *= _SHRINK
            continue
        d1, d2, d3, d4, d5 = coef[1], coef[2], coef[3], coef[4], coef[5]
        hess = np.array([[2.0 * d3, d4], [d4, 2.0 * d5]])
        if not (hess[0, 0] > 0 and np.linalg.det(hess) > 0):
            # not a minimum, or a saddle: the local quadratic is no use here
            h *= _SHRINK
            continue
        step = np.linalg.solve(hess, np.array([-d1, -d2]))
        if not np.all(np.isfinite(step)):
            h *= _SHRINK
            continue
        # The quadratic is only trusted inside the stencil it was fitted on.  The
        # vertex is always taken: testing it and rejecting bad steps turns out to
        # stall (h shrinks to nothing while the point never moves), whereas
        # moving unconditionally lets the next, smaller stencil correct an
        # overshoot.  h is only tightened once the vertex is comfortably inside
        # the stencil, because shrinking on every sweep would cap the total
        # distance the point can travel at about 2 h -- not enough to cross a
        # valley wider than the stencil.
        step = np.clip(step, -1.0, 1.0) * h
        steplen = float(np.max(np.abs(step))) / h
        p0 = float(np.clip(p0 + step[0], _P_FLOOR, 1.0 - _P_FLOOR))
        q0 = float(np.clip(q0 + step[1], _P_FLOOR, 1.0 - _P_FLOOR))
        if steplen < _QUAD_TRUST:
            h *= _QUAD_SHRINK
    e0 = float(
        _error_points(
            isodata, spike1, spike2, isoinv, errorratio, beta, alpha, [p0], [q0]
        )[0]
    )
    return p0, q0, e0


def singleoptimalspike(
    isodata,
    type_="pure",
    isospike=None,
    isoinv=None,
    errorratio=None,
    alpha=0.0,
    beta=0.0,
):
    """Calculate the composition of the optimal double spike.

    Returns ``(optspike, optprop, opterr, optspikeprop, optppmperamu)``.
    """
    if type_ == "pure":
        spikevector1 = np.zeros(isodata.nisos)
        spikevector1[isospike[0]] = 1.0
        spikevector2 = np.zeros(isodata.nisos)
        spikevector2[isospike[1]] = 1.0
    else:
        spikevector1 = isodata.rawspike[isospike[0], :]
        spikevector2 = isodata.rawspike[isospike[1], :]

    p, q, _ = _refine_optimum(
        isodata, spikevector1, spikevector2, isoinv, errorratio, alpha, beta
    )

    optprop = p
    optspike = q * spikevector1 + (1 - q) * spikevector2
    opterr, optppmperamu = errorestimate(
        isodata, p, optspike, isoinv, errorratio, beta, alpha
    )

    optspikeprop = np.zeros_like(optspike)
    optspikeprop[isospike[0]] = q
    optspikeprop[isospike[1]] = 1 - q

    return optspike, optprop, opterr, optspikeprop, optppmperamu
