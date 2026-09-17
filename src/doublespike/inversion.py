"""Routines for performing the double spike inversion.

The double spike inversion takes

* ``m`` - the isotopic ratios measured on the spike-sample *mixture*,
* ``T`` - the isotopic ratios of the double *spike*,
* ``n`` - the isotopic ratios of an unspiked *standard*,

and returns the three model parameters

* ``lambda`` - the proportion of spike in the mixture, in *ratio* space,
* ``alpha``  - the natural (mass dependent) fractionation of the sample,
* ``beta``   - the instrumental (mass dependent) fractionation of the measurement,

by solving the three equations (Rudge et al. 2009, Chem. Geol. 265:420-431,
equation 10)

.. math::

    \\lambda T_i + (1-\\lambda) n_i e^{-\\alpha P_i} = m_i e^{-\\beta P_i},
    \\qquad i = 1,2,3

where :math:`P_i = \\ln(M_j / M_k)` is the natural logarithm of the ratio of the
atomic masses defining ratio :math:`i`.

Two solvers are available
-------------------------

``dscorrection_newton`` (default)
    Eliminates :math:`\\lambda` **analytically**.  Because equation (10) is
    *linear* in :math:`\\lambda`, for any trial :math:`(\\alpha, \\beta)` we can
    write :math:`\\lambda (T_i - N_i) = m_i e^{-\\beta P_i} - N_i` with
    :math:`N_i = n_i e^{-\\alpha P_i}`.  Requiring the value of
    :math:`\\lambda` obtained from different ratios to agree gives the
    :math:`2 \\times 2` system

    .. math::

        G_x(\\alpha,\\beta) = S_{i_0} U_x - U_{i_0} S_x = 0,
        \\qquad S_i = T_i - N_i, \\quad U_i = M_i - N_i

    which is solved by a damped Newton-Raphson iteration with an analytic
    Jacobian, a backtracking line search and multi-start fallbacks.  This
    removes the ill-conditioned :math:`1/(1-\\lambda)` reparameterisation and
    the reliance on MINPACK's ``fsolve``, converges in a handful of iterations
    and can be evaluated for thousands of measurements simultaneously.

``dscorrection_legacy``
    The original ``scipy.optimize.fsolve`` based implementation, retained for
    reference and used as a last-resort fallback.

``dsinversion`` is the supported user-facing entry point and now processes all
rows of a matrix of measurements in a single vectorised call.
"""

import numpy as np

__all__ = [
    "dsinversion",
    "dscorrection",
    "dscorrection_newton",
    "dscorrection_legacy",
    "F_params",
    "F",
    "J",
]


# ---------------------------------------------------------------------------
# sanity helpers
# ---------------------------------------------------------------------------
def _pivot_layout(T, n):
    """Return the pivot index and the indices of the two working ratios.

    The pivot is the ratio for which the spike differs most from the standard,
    i.e. ``argmax|T - n|``.  This is the ratio that carries the most spike
    information and therefore stays furthest from the degenerate
    ``S = T - N = 0`` limit as ``alpha`` is varied.  The next two largest
    entries complete the 2 x 2 system; any further ratios are redundant
    (they are still available for residual checking).
    """
    B, nr = T.shape
    order = np.argsort(-np.abs(T - n), axis=1)
    if nr < 3:
        raise ValueError("Need at least three isotopic ratios to invert.")
    i0 = order[:, 0]
    others = order[:, 1:3]
    return i0, others


def _take(a, idx):
    """``a[rows, idx]`` for a (B, n) array and an (B, k) index array."""
    return np.take_along_axis(a, idx, axis=1)


def _linear_start(P, n, T, m):
    """Initial guess from the linearised problem (Rudge et al. 2009, eq. 11-14).

    Writing equation (10) to first order in ``alpha`` and ``beta`` gives the
    linear system ``A y = b`` with

        b = m - n,   A = [ T - n,  -n P,  m P ],   y = (lambda, (1-lambda)alpha, beta)

    All arrays have shape ``(B, 3)`` because the caller restricts them to the
    three ratios used by the inversion; the stacked 3 x 3 solve is therefore a
    single vectorised LAPACK call.
    """
    B = m.shape[0]
    A = np.empty((B, 3, 3))
    A[:, :, 0] = T - n
    A[:, :, 1] = -n * P
    A[:, :, 2] = m * P
    b = m - n
    ok = np.ones(B, dtype=bool)
    try:
        y0 = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        # solve row by row so that a single degenerate row cannot kill the batch
        y0 = np.zeros((B, 3))
        for i in range(B):
            try:
                y0[i, :] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                ok[i] = False
    if not np.all(np.isfinite(y0)):
        ok &= np.all(np.isfinite(y0), axis=1)
    y0 = np.where(np.isfinite(y0), y0, 0.0)

    # cap on |alpha| and |beta| beyond which the linear guess is worthless
    cap = 1.0 / np.max(np.abs(P))
    lam0 = y0[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha0 = np.where(np.abs(1 - lam0) > 1e-12, y0[:, 1] / (1 - lam0), 0.0)
    beta0 = y0[:, 2]
    bad = (
        ~ok
        | (np.abs(alpha0) > cap)
        | (np.abs(beta0) > cap)
        | (lam0 < -0.5)
        | (lam0 > 1.5)
    )
    alpha0 = np.where(bad, 0.0, alpha0)
    beta0 = np.where(bad, 0.0, beta0)
    return alpha0, beta0, ~bad


def _residual(P, n, T, m, i0, others, alpha, beta):
    """Residual vector ``G`` and its Jacobian for the lambda-eliminated system.

    With ``N = n exp(-alpha P)``, ``M = m exp(-beta P)``, ``S = T - N`` and
    ``U = M - N`` the double spike equations read ``lambda S_i = U_i``.  For a
    trial ``(alpha, beta)`` the value of ``lambda`` implied by ratio ``i`` is
    ``U_i / S_i``; requiring the pivot ratio ``i0`` and a second ratio ``x`` to
    agree gives

        G_x = S_i0 U_x - U_i0 S_x = 0.

    ``others`` contains the two ``x`` values, so ``G`` has two components for
    the two unknowns.
    """
    N = n * np.exp(-alpha[:, None] * P)
    M = m * np.exp(-beta[:, None] * P)
    S = T - N
    U = M - N

    S0 = _take(S, i0[:, None])[:, 0]
    U0 = _take(U, i0[:, None])[:, 0]
    N0 = _take(N, i0[:, None])[:, 0]
    M0 = _take(M, i0[:, None])[:, 0]
    Sx = _take(S, others)  # (B, 2)
    Ux = _take(U, others)
    Nx = _take(N, others)
    Mx = _take(M, others)
    Px = P[others]  # (B, 2)
    P0 = P[i0]  # (B,)

    G = S0[:, None] * Ux - U0[:, None] * Sx  # (B, 2)

    # dG/dalpha = P0 N0 Ux + S0 Px Nx - Px Nx U0 - Sx P0 N0
    dGda = (
        (P0 * N0)[:, None] * Ux
        + S0[:, None] * (Px * Nx)
        - (Px * Nx) * U0[:, None]
        - Sx * (P0 * N0)[:, None]
    )
    # dG/dbeta  = -S0 Px Mx + Sx P0 M0
    dGdb = -S0[:, None] * (Px * Mx) + Sx * (P0 * M0)[:, None]

    J = np.stack((dGda, dGdb), axis=-1)  # (B, 2, 2)

    # relative size of the residual, used as a scale-free convergence test
    scale = np.abs(S0[:, None] * Ux) + np.abs(U0[:, None] * Sx) + 1e-300
    return G, J, scale, S0, U0


def _newton_phase(
    P, n, T, m, i0, others, alpha, beta, tol, maxiter, maxstep
):
    """Damped Newton iteration on the lambda-eliminated equations, run in place.

    The step is scaled as a whole when its infinity norm exceeds ``maxstep`` so
    that the descent direction of the Newton step is preserved -- clipping the
    components of the step individually destroys the descent property and is a
    classic cause of stalling.  The backtracking line search uses the absolute
    max-norm of the residual as its merit function, while convergence is judged
    on the scale-free *relative* residual so that the test behaves the same way
    for a 54Fe/56Fe ratio (order 0.1) and a 48Ca/42Ca ratio (order 500).

    Returns:
        converged (array of bool), iterations (array of int)
    """
    B = alpha.size
    converged = np.zeros(B, dtype=bool)
    iterations = np.zeros(B, dtype=int)

    active = np.arange(B)
    for _ in range(maxiter):
        if active.size == 0:
            break
        aa, bb = alpha[active], beta[active]
        nr_ = n[active], T[active], m[active]
        i0_, ot_ = i0[active], others[active]
        G, J, scale = _residual(P, nr_[0], nr_[1], nr_[2], i0_, ot_, aa, bb)[:3]
        rm = np.max(np.abs(G), axis=1)  # absolute merit for the line search
        rel = np.max(np.abs(G) / scale, axis=1)  # scale free convergence test

        done = rel < tol
        if np.any(done):
            converged[active[done]] = True
        keep = ~done
        if not np.any(keep):
            break
        active, aa, bb = active[keep], aa[keep], bb[keep]
        nr_ = nr_[0][keep], nr_[1][keep], nr_[2][keep]
        i0_, ot_ = i0_[keep], ot_[keep]
        G, J, rm = G[keep], J[keep], rm[keep]

        # Newton step
        try:
            step = np.linalg.solve(J, -G[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(G)
            for k in range(G.shape[0]):
                try:
                    step[k] = np.linalg.solve(J[k], -G[k])
                except np.linalg.LinAlgError:
                    step[k] = 0.0
        step = np.where(np.isfinite(step), step, 0.0)

        # direction preserving step limiting
        snorm = np.max(np.abs(step), axis=1)
        shrink = np.where(snorm > maxstep, maxstep / np.where(snorm > 0, snorm, 1.0), 1.0)
        step = step * shrink[:, None]

        # backtracking line search on the absolute residual norm
        t = np.ones(step.shape[0])
        ok = np.zeros(step.shape[0], dtype=bool)
        for _ in range(50):
            cand_a = aa + t * step[:, 0]
            cand_b = bb + t * step[:, 1]
            Gc, _, _sc = _residual(P, nr_[0], nr_[1], nr_[2], i0_, ot_, cand_a, cand_b)[:3]
            rmc = np.max(np.abs(Gc), axis=1)
            ok = rmc <= (1.0 - 1e-4 * t) * rm
            if np.all(ok):
                break
            t = np.where(ok, t, 0.5 * t)

        alpha[active] = aa + t * step[:, 0]
        beta[active] = bb + t * step[:, 1]
        iterations[active] += 1

        # drop rows where no step length produced a decrease: the Newton
        # direction is not a descent direction there, so restarting from a
        # different point is more productive than grinding on
        active = active[ok]
        tiny = np.maximum(np.abs(alpha[active]), np.abs(beta[active]))
        steplen = np.max(np.abs(step[ok]) * t[ok][:, None], axis=1)
        if np.all(steplen < 1e-14 * np.maximum(1.0, tiny)):
            converged[active] = True
            break

    return converged, iterations


def _multistart(P, n, T, m, i0, others, failed, tol, maxiter, maxstep):
    """Retry the rows that failed, from a small grid of physically sensible starts.

    ``failed`` is a boolean mask over all rows; the returned arrays are aligned
    with ``np.flatnonzero(failed)`` and contain only the rows that the restart
    managed to solve.
    """
    if not np.any(failed):
        return None
    idx = np.flatnonzero(failed)
    alpha = np.zeros(idx.size)
    beta = np.zeros(idx.size)
    best = np.full(idx.size, np.inf)
    for a0 in (0.0, -0.2, 0.2, -0.05, 0.05):
        for b0 in (0.0, -3.0, 3.0, -6.0, 6.0, -1.5, 1.5):
            aa = np.full(idx.size, a0)
            bb = np.full(idx.size, b0)
            conv, _ = _newton_phase(
                P, n[idx], T[idx], m[idx], i0[idx], others[idx],
                aa, bb, tol, maxiter, maxstep,
            )
            G, _, scale = _residual(
                P, n[idx], T[idx], m[idx], i0[idx], others[idx], aa, bb
            )[:3]
            rel = np.max(np.abs(G) / scale, axis=1)
            better = conv & (rel < best)
            best = np.where(better, rel, best)
            alpha[better] = aa[better]
            beta[better] = bb[better]
    good = np.isfinite(best) & (best < 1e-8)
    return alpha[good], beta[good], idx[good]


def dscorrection_newton(P, n, T, m, tol=1e-14, maxiter=60, maxstep=2.0):
    """Solve the double spike equations by lambda elimination + damped Newton.

    Args:
        P (array): log of ratio of atomic masses, shape ``(nr,)``
        n (array): isotope ratios of standard / unspiked run, shape ``(B, nr)``
        T (array): isotope ratios of the spike, shape ``(B, nr)``
        m (array): isotope ratios of the measurement, shape ``(B, nr)``
        tol (float): relative tolerance on the residual
        maxiter (int): maximum number of Newton iterations per start
        maxstep (float): maximum infinity norm of a single Newton step

    Returns:
        alpha (array): natural fractionation, shape ``(B,)``
        beta (array): instrumental fractionation, shape ``(B,)``
        lambda_ (array): proportion of spike in ratio space, shape ``(B,)``
        converged (array): boolean array, True where the iteration converged
        iterations (array): number of iterations used
    """
    P = np.asarray(P, dtype=float).ravel()
    n = np.atleast_2d(np.asarray(n, dtype=float))
    T = np.atleast_2d(np.asarray(T, dtype=float))
    m = np.atleast_2d(np.asarray(m, dtype=float))
    n, T, m = np.broadcast_arrays(n, T, m)
    B, nr = m.shape
    if nr != P.size:
        raise ValueError(
            f"P has {P.size} ratios but n, T, m have {nr}; they must agree."
        )
    if nr < 3:
        raise ValueError("Need at least three isotopic ratios to invert.")

    # Restrict to the three ratios that carry the inversion (the pivot plus the
    # two next most sensitive ratios) and build the linear initial guess.
    i0, others = _pivot_layout(T, n)
    sel = np.concatenate((i0[:, None], others), axis=1)
    alpha, beta, _ = _linear_start(
        P[sel],
        np.take_along_axis(n, sel, axis=1),
        np.take_along_axis(T, sel, axis=1),
        np.take_along_axis(m, sel, axis=1),
    )

    phase = _newton_phase(P, n, T, m, i0, others, alpha, beta, tol, maxiter, maxstep)
    converged = np.asarray(phase[0], dtype=bool)
    iterations = np.asarray(phase[1])

    failed = ~converged
    if np.any(failed):
        # second attempt: same algorithm restarted from a small grid of
        # physically sensible (alpha, beta) combinations
        ra, rb, good_idx = _multistart(
            P, n, T, m, i0, others, failed, tol, maxiter, maxstep
        )
        if good_idx.size:
            alpha[good_idx] = ra
            beta[good_idx] = rb
            iterations[good_idx] += maxiter
            converged[good_idx] = True

    # final residual check for anything that claimed to converge
    G, _, scale, S0, U0 = _residual(P, n, T, m, i0, others, alpha, beta)
    rel = np.max(np.abs(G) / scale, axis=1)
    converged = converged & (rel < max(tol * 1e6, 1e-9)) & np.isfinite(alpha) & np.isfinite(beta)

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = U0 / S0
    converged = converged & np.isfinite(lam)
    lam = np.where(np.isfinite(lam), lam, 0.0)

    return alpha, beta, lam, converged, iterations


def dscorrection(P, n, T, m, **kwargs):
    """Routine for double spike fractionation correction using isotope ratios as inputs.

    Args:
        P (array): log of ratio of atomic masses
        n (array): isotope ratios of standard / unspiked run
        T (array): isotope ratios of spike
        m (array): isotope ratios of measured
        **kwargs: retained for backwards compatibility and passed on to the
            legacy solver if the Newton solver has to be used as a fallback

    Returns:
        z (array): Spike ratio proportion (lambda), natural fractionation
                   (alpha) and instrumental fractionation (beta) as a vector
                   ``z = (lambda, alpha, beta)``.
    """
    P = np.asarray(P, dtype=float)
    n = np.asarray(n, dtype=float)
    T = np.asarray(T, dtype=float)
    m = np.asarray(m, dtype=float)
    single = m.ndim == 1

    alpha, beta, lam, converged, _ = dscorrection_newton(P, n, T, m)
    if not np.all(converged):
        # extremely rare: fall back to the legacy MINPACK solver for the
        # handful of stubborn rows so that behaviour never regresses
        bad = np.flatnonzero(~converged)
        mb = np.atleast_2d(m)[bad]
        nb = np.atleast_2d(n)[bad]
        Tb = np.atleast_2d(T)[bad]
        for k in range(mb.shape[0]):
            try:
                z = dscorrection_legacy(P, nb[k], Tb[k], mb[k], **kwargs)
                lam[bad[k]], alpha[bad[k]], beta[bad[k]] = z[0], z[1], z[2]
            except Exception:
                pass

    z = np.column_stack((lam, alpha, beta))
    return z[0] if single else z


def dscorrection_legacy(P, n, T, m, **kwargs):
    """Original ``fsolve`` based solver, kept for reference and as a fallback.

    Returns:
        z (array): vector ``z = (lambda, alpha, beta)``.
    """
    from scipy.optimize import fsolve

    b = np.transpose(m - n)
    A = np.array([T - n, -n * P, m * P])
    try:
        y0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        y0 = np.array([0.5, 0.0, 0.0])

    lambda_lin = y0[0]
    if abs(1 - lambda_lin) < 1e-12:
        alpha_lin = 0.0
    else:
        alpha_lin = y0[1] / (1 - lambda_lin)
    beta_lin = y0[2]

    # linear approximations are awful if |alpha| > |1/P|, cap if getting alpha this large
    alpha_max = 1.0 / max(abs(P))
    alpha_min = -1.0 / max(abs(P))

    if (
        alpha_lin > alpha_max
        or alpha_lin < alpha_min
        or beta_lin > alpha_max
        or beta_lin < alpha_min
    ):
        # alternate starting guess if linear approximation goes way out
        y0 = np.array([0.5, 0.0, 0.0])

    y = fsolve(F, y0, args=(P, n, T, m), fprime=J, xtol=1e-12, col_deriv=True, **kwargs)
    z = np.array(y, dtype=float)
    z[1] = y[1] / (1 - y[0])
    return z


# ---------------------------------------------------------------------------
# user facing routine
# ---------------------------------------------------------------------------
def dsinversion(isodata, measured, spike=None, isoinv=None, standard=None):
    """Perform the double spike inversion for a given set of measurements.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        measured (array): a matrix of beam intensities. Columns correspond to the
            different isotopes e.g. for Fe, first column is 54Fe, second is 56Fe,
            third is 57Fe, fourth is 58Fe. The matrix should have the same number
            of columns as there are isotopes available.
        spike (array): a composition vector for the spike. e.g. [0, 0, 0.5, 0.5] is a 50-50
            mix of 57Fe and 58Fe. If None this is read from isodata.
        isoinv (array): the four isotopes to use in the inversion, e.g [54, 56, 57, 58]. If
            None this is read from isodata.
        standard (array): standard composition or unspiked run data. If
            None this is read from isodata.

    Returns:
        This routine performs the double spike inversion on measured data to return the
        "true" composition of the sample. Output is returned as a dictionary with the
        following fields
            alpha: the inferred natural fractionations
            beta: the inferred instrumental fractionations
            prop: the inferred proportions of spike to sample
            sample: the inferred compositions of the sample
            mixture: the inferred compositions of the mixture

        In addition the following diagnostic fields are returned (new in 1.1)
            converged: boolean array, True where the inversion converged
            residual: relative residual of the double spike equations
            lambda_ratio: the raw (ratio space) proportion of spike

    Example:
        >>> dsinversion(IsoData('Fe'), measured, [0, 0, 0.5, 0.5], [54, 56, 57, 58])
    """
    from .isodata import ratioproptorealprop, normalise_composition, ratio

    # Get data from isodata if not supplied as arguments
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given.")
        spike = isodata.spike
    if isoinv is None:
        if isodata.isoinv is None:
            raise Exception("Inversion isotopes not specified.")
        isoinv = isodata.isoinv
    if standard is None:
        standard = isodata.standard
    if standard is None:
        raise Exception("Standard composition not specified.")

    # Convert to numpy array if not already
    measured = np.array(measured, dtype=float)
    spike = np.array(spike, dtype=float)
    standard = np.array(standard, dtype=float)
    isoinv = np.array(isoinv, dtype=int)
    if isoinv.size != 4:
        raise Exception("Need exactly 4 inversion isotopes, got %d" % isoinv.size)

    # Duplicate so all matrices same size
    nspike, nstandard, nmeasured = 1, 1, 1
    if spike.ndim > 1:
        nspike = spike.shape[0]
    if measured.ndim > 1:
        nmeasured = measured.shape[0]
    if standard.ndim > 1:
        nstandard = standard.shape[0]
    nobs = max(nspike, nmeasured, nstandard)

    if spike.ndim == 1:
        spike = np.tile(spike, (nobs, 1))
    if measured.ndim == 1:
        measured = np.tile(measured, (nobs, 1))
    if standard.ndim == 1:
        standard = np.tile(standard, (nobs, 1))

    # Avoid division by zero errors for small values
    isoinv = isodata.isoindex(isoinv)
    if np.any(spike[0, isoinv] < 0.001):
        ix = np.argmax(spike[0, isoinv])
        deno = isoinv[ix]
        nume = isoinv[isoinv != deno]
        isoinv = np.concatenate((np.array([deno]), nume))

    # Take ratios based on the isotopes we are inverting
    P = np.log(ratio(isodata.mass, isoinv))
    n = ratio(standard, isoinv)
    T = ratio(spike, isoinv)
    m = ratio(measured, isoinv)

    # Solve the double spike equations.  dscorrection runs the lambda-eliminated
    # damped Newton iteration, restarts from alternative points if needed and
    # only falls back on scipy's fsolve for the odd stubborn measurement.
    z = dscorrection(P, n, T, m)
    lambda_, alpha, beta = z[:, 0], z[:, 1], z[:, 2]

    # residual of the double spike equations, expressed relative to the size of
    # the terms involved so that it is comparable between isotope systems
    with np.errstate(divide="ignore", invalid="ignore"):
        N = n * np.exp(-alpha[:, None] * P)
        M = m * np.exp(-beta[:, None] * P)
        res = lambda_[:, None] * T + (1 - lambda_[:, None]) * N - M
    denom = np.abs(lambda_[:, None] * T) + np.abs((1 - lambda_[:, None]) * N) + np.abs(M)
    residual = np.max(np.abs(res) / np.maximum(denom, 1e-300), axis=1)
    converged = np.isfinite(residual) & (residual < 1e-8)

    out = {}
    out["alpha"] = alpha
    out["beta"] = beta
    out["lambda_ratio"] = lambda_
    out["converged"] = converged
    out["residual"] = residual

    isonum = np.arange(isodata.nisos)
    isonum = isonum[isonum != isoinv[0]]
    isonum = np.concatenate((np.array([isoinv[0]]), isonum))

    AP = np.log(ratio(isodata.mass, isonum))
    AT = ratio(spike, isonum)
    An = ratio(standard, isonum)
    Am = ratio(measured, isonum)

    # Calculate sample and mixture proportion, and proportion by mole
    AM = Am * np.exp(-AP[np.newaxis, :] * beta[:, np.newaxis])
    AN = An * np.exp(-AP[np.newaxis, :] * alpha[:, np.newaxis])
    out["prop"] = ratioproptorealprop(lambda_, AT, AN)
    out["sample"] = np.ones_like(measured)
    out["mixture"] = np.ones_like(measured)
    out["sample"][:, isonum[1:]] = AN
    out["mixture"][:, isonum[1:]] = AM
    out["sample"] = normalise_composition(out["sample"])
    out["mixture"] = normalise_composition(out["mixture"])

    if nobs == 1:
        # For single measurements make the output more compact
        out["alpha"] = out["alpha"][0]
        out["beta"] = out["beta"][0]
        out["prop"] = out["prop"][0]
        out["lambda_ratio"] = out["lambda_ratio"][0]
        out["converged"] = bool(out["converged"][0])
        out["residual"] = out["residual"][0]
        out["sample"] = np.squeeze(out["sample"])
        out["mixture"] = np.squeeze(out["mixture"])

    return out


def F_params(y, P, n, T, m):
    """Determine main variables in the objective function."""
    lambda_ = y[0]
    alpha = y[1] / (1 - lambda_)
    beta = y[2]
    N = n * np.exp(-alpha * P)
    M = m * np.exp(-beta * P)
    return lambda_, alpha, beta, N, M


def F(y, P, n, T, m):
    """The nonlinear equations to solve."""
    lambda_, alpha, beta, N, M = F_params(y, P, n, T, m)
    return lambda_ * T + (1 - lambda_) * N - M  # equation (10)


def J(y, P, n, T, m):
    """The Jacobian of the nonlinear equations -- can speed up root finding, but is not required."""
    lambda_, alpha, beta, N, M = F_params(y, P, n, T, m)
    dfdlambdaprime = T - N * (1 + alpha * P)
    dfdu = -N * P
    dfdbeta = M * P
    return np.array([dfdlambdaprime, dfdu, dfdbeta])  # equation (15)
