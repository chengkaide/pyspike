"""Module for performing linear error propagation.

The routines here implement the linear error propagation of Rudge et al. (2009)
appendix A.  All of the linear algebra is performed on *stacks* of matrices, so
``errorestimate`` accepts arrays of proportions and spike compositions and
returns arrays of errors.  This is what makes the two dimensional error maps in
:mod:`doublespike.plotting` and the search in :mod:`doublespike.optimal`
practical: a 200 x 200 error map is a single vectorised call rather than 40 000
python level calls to the old scalar routine.

Notation follows the paper: ``n`` is the unspiked standard, ``T`` the double
spike, ``AN`` the (unknown) sample, ``AM`` the spike-sample mixture and ``Am``
the measured mixture.  ``alpha`` is natural fractionation, ``beta`` is
instrumental fractionation.
"""

import numpy as np

from .isodata import normalise_composition, realproptoratioprop


# ---------------------------------------------------------------------------
# public, user facing
# ---------------------------------------------------------------------------
def errorestimate(
    isodata, prop, spike=None, isoinv=None, errorratio=None, alpha=0.0, beta=0.0
):
    """Calculate the error in the natural fractionation factor or a chosen ratio by linear error propagation.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        prop (float/array): proportion of double spike in double spike-sample mix.
            An array can be supplied, in which case the returned errors have the
            same shape.
        spike (array): the isotopic composition of the spike e.g. [0, 0.5, 0, 0.5]
            corresponds to a 50-50 mixture of the 2nd and 4th isotopes
            (56Fe and 58Fe) in the case of Fe.  A matrix of compositions can be
            supplied, in which case it is broadcast against ``prop``.
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the error on the natural fractionation
            factor (known as alpha) is given. Instead, the error on a
            particular ratio can be given by setting errorratio. e.g.
            setting errorratio=[58, 56] will give the error on 58Fe/56Fe.
        alpha, beta (floats): there is a small dependance of the error on the fractionation
            factors (instrumental and natural, or alpha and beta). Values of alpha and
            beta can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large.

        If spike or isoinv are set as None, values from isodata will be used instead.

    Returns:
        error (float/array): the error on the fractionation factor, or the specified ratio.
        ppmperamu (float/array): the error converted to an approximate ppm per atomic mass unit

    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> error, ppmperamu = errorestimate(isodata_fe, 0.5, [0, 0.5, 0, 0.5])
    """
    err, ppm = errorestimate_many(
        isodata, prop, spike, isoinv, errorratio, alpha, beta
    )
    if np.ndim(err) == 0:
        return float(err), float(ppm)
    return err, ppm


def errorestimate_many(
    isodata, prop, spike=None, isoinv=None, errorratio=None, alpha=0.0, beta=0.0
):
    """Vectorised :func:`errorestimate`.

    ``prop`` (and optionally ``spike``, ``alpha``, ``beta``) may be arrays.  All
    array arguments are broadcast against each other and the result keeps the
    broadcast shape.
    """
    # ---- defaults -------------------------------------------------------
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        spike = isodata.spike
    if isoinv is None:
        if isodata.isoinv is None:
            raise Exception("No inversion isotopes set")
        isoinv = isodata.isoinv

    nisos = isodata.nisos
    spike_arr = np.asarray(spike, dtype=float)
    if spike_arr.ndim == 0:
        raise ValueError("spike must be a composition vector")
    spike_lead = spike_arr.shape[:-1]
    spike_arr = spike_arr / spike_arr.sum(axis=-1, keepdims=True)

    prop_arr = np.asarray(prop, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    # broadcast to a common leading shape.  A prop of shape (P, 1) and a spike of
    # shape (1, S, nisos) therefore produce a (P, S) grid of errors, which is
    # exactly what the 2D error maps need.
    shape = np.broadcast_shapes(
        prop_arr.shape, spike_lead, alpha_arr.shape, beta_arr.shape
    )
    prop_arr = np.broadcast_to(prop_arr, shape).reshape(-1)
    alpha_arr = np.broadcast_to(alpha_arr, shape).reshape(-1)
    beta_arr = np.broadcast_to(beta_arr, shape).reshape(-1)
    spike_arr = np.broadcast_to(spike_arr, shape + (nisos,)).reshape(-1, nisos)

    # ---- denominator / ratio bookkeeping --------------------------------
    isoinv = isodata.isoindex(np.asarray(isoinv))
    if np.size(isoinv) != 4:
        raise Exception("Need exactly 4 inversion isotopes")
    # The denominator isotope is the isotope the spike contains most of.  Which
    # ratio you use as denominator does not change the physics or the resulting
    # error, it is purely a numerical conditioning choice, so a single
    # denominator is fixed for the whole batch (this is what keeps error maps
    # smooth when the spike sweeps between two single isotope spikes).
    mean_spike = spike_arr.mean(axis=0)
    deno = int(isoinv[np.argmax(mean_spike[isoinv])])
    nume = isoinv[isoinv != deno]
    isoinv = np.concatenate(([deno], nume))

    errorratio = isodata.isoindex(errorratio)
    invrat = isodata.invrat(isoinv)

    # ---- composition and ratio data -------------------------------------
    z, AP, An, AT, Am, AN, AM = ratiodata_many(
        isodata, deno, prop_arr, spike_arr, alpha_arr, beta_arr
    )
    measured = _composition_many(Am, deno, nisos)

    # ---- covariance matrices of n, T and m ------------------------------
    VAn = _calcratiocov_many(isodata.standard, isodata.errormodel["standard"], deno)
    VAT = _calcratiocov_many(spike_arr, isodata.errormodel["spike"], deno)
    VAm = _calcratiocov_many(measured, isodata.errormodel["measured"], deno, prop_arr)

    Vz, VAN, _ = _fcerrorpropagation_many(z, AP, An, AT, Am, VAn, VAT, VAm, invrat)

    # ---- error to return ------------------------------------------------
    if errorratio is None:
        error = np.sqrt(np.maximum(Vz[:, 1, 1], 0.0))
        ppmperamu = (1000000.0 * error) / np.mean(isodata.mass)  # equation (51)
    else:
        newVAN = _changedenomcov_many(AN, VAN, deno, int(errorratio[1]))
        isonums = np.arange(nisos)
        newAni = isonums[isonums != errorratio[1]]
        erat = int(np.where(errorratio[0] == newAni)[0][0])
        error = np.sqrt(np.maximum(newVAN[:, erat, erat], 0.0))
        stdratio = isodata.standard[errorratio[0]] / isodata.standard[errorratio[1]]
        massdiff = np.abs(isodata.mass[errorratio[0]] - isodata.mass[errorratio[1]])
        ppmperamu = (1000000.0 * error) / (stdratio * massdiff)  # equation (46)

    return error.reshape(shape), ppmperamu.reshape(shape)


# ---------------------------------------------------------------------------
# scalar helpers (kept as published API, used by the notebooks)
# ---------------------------------------------------------------------------
def calcratiocov(composition, errormodel, di, isonorm=None, prop=0.0):
    """Calculate the covariance matrix of the ratios based on the given error model and composition."""
    squeeze = np.ndim(composition) == 1
    out = _calcratiocov_many(composition, errormodel, di, prop, isonorm)
    return out[0] if squeeze else out


def calcbeamcov(meanbeams, errormodel):
    """Calculate beam covariance matrix."""
    beamvar = (
        errormodel["a"]
        + errormodel["b"] * meanbeams
        + errormodel["c"] * (meanbeams**2)
    )  # equation (34)
    return np.diag(beamvar)


def covbeamtoratio(meanbeams, covbeams, di):
    """Convert a covariance matrix for beams to one for ratios."""
    squeeze = np.ndim(meanbeams) == 1
    out = _covbeamtoratio_many(
        np.atleast_2d(meanbeams), np.atleast_3d(covbeams), di
    )
    return out[0] if squeeze else out


def changedenomcov(data, datacov, olddi, newdi):
    """Change denominator of covariance matrix for given set of ratios."""
    data = np.atleast_2d(data)
    datacov = np.asarray(datacov, dtype=float)
    if datacov.ndim == 2:
        datacov = datacov[np.newaxis, :, :]
    return _changedenomcov_many(data, datacov, olddi, newdi)[0]


def ratiodata(isodata, di, prop, spike=None, alpha=0.0, beta=0.0):
    """Calculate isotopic ratios describing system.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        di (int): denominator isotope, e.g. 56
        prop (float): proportion of double spike in double spike-sample mix.
        spike (array): the isotopic composition of the spike. If None taken from isodata.spike
        alpha (float): natural fractionation factor
        beta (float): instrumental fractionation factor

    Returns:
        z (array): vector of model parameters (lambda, alpha, beta)
        AP (array): log of ratio of atomic masses
        An (array): isotopic ratios of standard/ unspiked run
        AT (array): isotopic ratios of spike
        Am (array): isotopic ratios of measurement
        AN (array): isotopic ratios of sample
        AM (array): isotopic ratios of mixture
    """
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        spike = isodata.spike
    di = int(isodata.isoindex(di))
    spike = np.atleast_2d(np.asarray(spike, dtype=float))
    z, AP, An, AT, Am, AN, AM = ratiodata_many(
        isodata,
        di,
        np.atleast_1d(np.asarray(prop, dtype=float)),
        spike,
        np.atleast_1d(np.asarray(alpha, dtype=float)),
        np.atleast_1d(np.asarray(beta, dtype=float)),
    )
    squeeze = np.ndim(prop) == 0
    if squeeze:
        z = z[0]
        AT, Am, AN, AM = AT[0], Am[0], AN[0], AM[0]
    return z, AP, An, AT, Am, AN, AM


def z_sensitivity(z, P, n, T, m):
    """Calculate the derivatives of the z=(lambda,alpha, beta) vector with respect to the input n, T, m ratios."""
    dzdn, dzdm, dzdT = _z_sensitivity_many(
        np.atleast_2d(z),
        np.atleast_2d(P),
        np.atleast_2d(n),
        np.atleast_2d(T),
        np.atleast_2d(m),
    )
    return dzdn[0], dzdm[0], dzdT[0]


def sensitivity(z, AP, An, AT, Am, invrat):
    """Returns the partial derivatives describing the sensitivity of model outputs to model inputs.

    Args:
        z (array): vector of (lambda, alpha, beta)
        AP (array): log of ratio of atomic masses
        An (array): isotopic ratios of standard/ unspiked run
        AT (array): isotopic ratios of spike
        Am (array): isotopic ratios of measurement
        invrat: indices of isotopic ratios used in inversion

    Returns the derivatives:
        dzdAn, dzdAT, dzdAm, dANdAn, dANdAT, dANdAm, dAMdAn, dAMdAT, dAMdAm
    """
    out = _sensitivity_many(
        np.atleast_2d(z), np.asarray(AP), An, AT, Am, invrat
    )
    return tuple(a[0] for a in out)


def fcerrorpropagation(z, AP, An, AT, Am, VAn, VAT, VAm, invrat):
    """Linear error propagation for the fractionation correction."""
    out = _fcerrorpropagation_many(
        np.atleast_2d(z), np.asarray(AP), An, AT, Am, VAn, VAT, VAm, invrat
    )
    return tuple(a[0] for a in out)


# ---------------------------------------------------------------------------
# batched core
# ---------------------------------------------------------------------------
def _composition_many(ratios, di, nisos):
    """Convert an array of isotopic ratios into an array of compositional vectors."""
    ratios = np.atleast_2d(ratios)
    comp = np.ones((ratios.shape[0], nisos))
    ni = np.arange(nisos)
    ni = ni[ni != di]
    comp[:, ni] = ratios
    return normalise_composition(comp)


def ratiodata_many(isodata, di, prop, spike, alpha, beta):
    """Vectorised version of the ratio bookkeeping; all arguments are already batched."""
    P = np.log(isodata.mass)
    AP = P[np.arange(isodata.nisos) != di] - P[di]  # log(mass_i/mass_di) == log ratio of masses
    AT = spike[:, np.arange(isodata.nisos) != di] / spike[:, di, np.newaxis]
    An = isodata.standard[np.arange(isodata.nisos) != di] / isodata.standard[di]

    alpha = np.asarray(alpha).reshape(-1, 1)
    beta = np.asarray(beta).reshape(-1, 1)
    prop = np.asarray(prop).reshape(-1)

    AN = An[np.newaxis, :] * np.exp(-alpha * AP[np.newaxis, :])
    lambda_ = realproptoratioprop(prop, AT, AN)
    z = np.column_stack((lambda_, alpha[:, 0], beta[:, 0]))
    AM = lambda_[:, np.newaxis] * AT + (1 - lambda_)[:, np.newaxis] * AN
    Am = AM * np.exp(beta * AP[np.newaxis, :])

    return z, AP, An, AT, Am, AN, AM


def _calcratiocov_many(composition, errormodel, di, prop=0.0, isonorm=None):
    """Batched covariance matrix of the ratios for a given error model."""
    composition = np.atleast_2d(np.asarray(composition, dtype=float))
    nisos = composition.shape[-1]
    if isonorm is None:
        isonorm = np.arange(nisos)
    composition = composition / composition.sum(axis=-1, keepdims=True)
    meanbeams = errormodel["intensity"] * composition / composition[:, isonorm].sum(
        axis=-1, keepdims=True
    )
    prop = np.asarray(prop, dtype=float)
    if errormodel["type"] == "fixed-sample":
        prop = prop.reshape(-1, 1) if prop.ndim else prop
        meanbeams = meanbeams / (1.0 - prop)
    covbeams = np.zeros((meanbeams.shape[0], nisos, nisos))
    beamvar = (
        errormodel["a"]
        + errormodel["b"] * meanbeams
        + errormodel["c"] * (meanbeams**2)
    )  # equation (34)
    idx = np.arange(nisos)
    covbeams[:, idx, idx] = beamvar
    return _covbeamtoratio_many(meanbeams, covbeams, di)


def _covbeamtoratio_many(meanbeams, covbeams, di):
    """Convert a stack of beam covariance matrices to ratio covariance matrices."""
    meanbeams = np.atleast_2d(meanbeams)
    covbeams = np.atleast_3d(covbeams)
    nisos = meanbeams.shape[-1]
    isonums = np.arange(nisos)
    ni = isonums[isonums != di]
    k = ni.size
    n = meanbeams[:, ni]
    d = meanbeams[:, di]
    ii = np.concatenate((ni, np.array([di])))  # move denominator to end
    M = covbeams[:, ii, :][:, :, ii]

    A = np.zeros((meanbeams.shape[0], k, nisos))
    rows = np.arange(k)
    A[:, rows, rows] = 1.0 / d[:, np.newaxis]
    A[:, :, k] = -n / (d[:, np.newaxis] ** 2)  # equation (38)
    return A @ M @ np.swapaxes(A, -1, -2)


def _changedenomcov_many(data, datacov, olddi, newdi):
    """Change denominator of a stack of covariance matrices for a given set of ratios."""
    nisos = data.shape[-1] + 1
    B = data.shape[0]
    oldni = np.concatenate((np.arange(olddi), np.arange(olddi + 1, nisos)))
    dataplus = np.concatenate(
        (data[:, :olddi], np.ones((B, 1)), data[:, olddi:]), axis=1
    )
    newni = np.concatenate((np.arange(newdi), np.arange(newdi + 1, nisos)))

    datacovplus = np.zeros((B, nisos, nisos))
    # single advanced-index assignment: a chained ``a[:, i, :][:, :, i] = ...``
    # would write into a temporary copy and silently do nothing
    datacovplus[:, oldni[:, np.newaxis], oldni[np.newaxis, :]] = datacov

    A = np.eye(nisos)[np.newaxis, :, :] / dataplus[:, newdi][:, np.newaxis, np.newaxis]
    A[:, :, newdi] = A[:, :, newdi] - dataplus / (
        dataplus[:, newdi][:, np.newaxis] ** 2
    )
    newdatacovplus = A @ datacovplus @ np.swapaxes(A, -1, -2)
    return newdatacovplus[:, newni[:, np.newaxis], newni[np.newaxis, :]]


def _stacked_solve(A, B):
    """``np.linalg.solve`` over a stack, tolerating singular rows.

    A handful of points on an error map can be exactly degenerate (for example
    where the double spike equations lose rank).  Rather than aborting the whole
    batch, those rows fall back to the minimum norm least squares solution.
    """
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        A = np.asarray(A)
        B = np.asarray(B)
        if A.ndim < 3:
            return np.linalg.solve(A, B)
        lead = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        Ab = np.broadcast_to(A, lead + A.shape[-2:])
        Bb = np.broadcast_to(B, lead + B.shape[-2:])
        out = np.full(lead + (A.shape[-1], B.shape[-1]), np.nan)
        for i in np.ndindex(*lead):
            try:
                out[i] = np.linalg.solve(Ab[i], Bb[i])
            except np.linalg.LinAlgError:
                out[i] = np.linalg.pinv(Ab[i]) @ Bb[i]
        return out


def _z_sensitivity_many(z, P, n, T, m):
    """Batched derivatives of z=(lambda, alpha, beta) with respect to n, T, m."""
    lambda_ = z[:, 0][:, np.newaxis]
    alpha = z[:, 1][:, np.newaxis]
    beta = z[:, 2][:, np.newaxis]

    N = n * np.exp(-P * alpha)
    M = m * np.exp(-P * beta)

    dfdlambda = T - N * (1 + alpha * P)
    dfdu = -N * P
    dfdbeta = M * P
    # (B, 3, 3) with dfdy[i, j] = dF_i / dy_j -- equation (15).  Built explicitly
    # rather than with np.stack so that the axis convention is unambiguous.
    dfdy = np.empty((z.shape[0], 3, 3))
    dfdy[:, :, 0] = dfdlambda
    dfdy[:, :, 1] = dfdu
    dfdy[:, :, 2] = dfdbeta

    k = P.shape[1]
    dfdT = lambda_[:, :, np.newaxis] * np.eye(k)[np.newaxis, :, :]  # equation (20)
    dfdm = -np.exp(-beta * P)[:, :, np.newaxis] * np.eye(k)[np.newaxis, :, :]
    dfdn = (1 - lambda_)[:, :, np.newaxis] * np.exp(-alpha * P)[:, :, np.newaxis] * np.eye(k)[np.newaxis, :, :]

    # matrix to convert (lambda, (1-lambda)alpha, beta) to (lambda, alpha, beta),
    # equation (22)
    B = z.shape[0]
    K = np.zeros((B, 3, 3))
    K[:, 0, 0] = 1.0
    K[:, 1, 0] = alpha[:, 0] / (1 - lambda_[:, 0])
    K[:, 1, 1] = 1.0 / (1 - lambda_[:, 0])
    K[:, 2, 2] = 1.0

    # one stacked solve for all three right hand sides: dfdy is the same for all
    # of them, and a single call costs a third of three separate ones
    rhs = _stacked_solve(
        dfdy, np.concatenate((dfdT, dfdm, dfdn), axis=-1)
    )  # (B, 3, 3k)
    rhs_T, rhs_m, rhs_n = rhs[:, :, :k], rhs[:, :, k : 2 * k], rhs[:, :, 2 * k :]
    dzdT = -K @ rhs_T  # equation (19)
    dzdm = -K @ rhs_m  # equation (18)
    dzdn = -K @ rhs_n  # equation (17)

    return dzdn, dzdm, dzdT


def _sensitivity_many(z, AP, An, AT, Am, invrat):
    """Batched sensitivity matrices."""
    z = np.atleast_2d(z)
    An = np.atleast_2d(An)
    AT = np.atleast_2d(AT)
    Am = np.atleast_2d(Am)
    B = z.shape[0]
    An = np.broadcast_to(An, (B, An.shape[-1]))
    AT = np.broadcast_to(AT, (B, AT.shape[-1]))
    Am = np.broadcast_to(Am, (B, Am.shape[-1]))

    alpha = z[:, 1][:, np.newaxis]
    beta = z[:, 2][:, np.newaxis]
    AM = Am * np.exp(-AP * beta)
    AN = An * np.exp(-AP * alpha)

    P = np.broadcast_to(AP[invrat], (B, len(invrat)))
    n = An[:, invrat]
    T = AT[:, invrat]
    m = Am[:, invrat]
    dzdn, dzdm, dzdT = _z_sensitivity_many(z, P, n, T, m)

    nratios = An.shape[1]
    dzdAT = np.zeros((B, 3, nratios))
    dzdAn = np.zeros((B, 3, nratios))
    dzdAm = np.zeros((B, 3, nratios))
    dzdAT[:, :, invrat] = dzdT
    dzdAn[:, :, invrat] = dzdn
    dzdAm[:, :, invrat] = dzdm

    NP = AN * AP  # (B, nratios)

    dANdAT = -NP[:, :, np.newaxis] * dzdAT[:, 1, :][:, np.newaxis, :]  # equation (26)
    dANdAn = (
        np.exp(-alpha * AP)[:, :, np.newaxis] * np.eye(nratios)[np.newaxis, :, :]
        - NP[:, :, np.newaxis] * dzdAn[:, 1, :][:, np.newaxis, :]
    )  # equation (24)
    dANdAm = -NP[:, :, np.newaxis] * dzdAm[:, 1, :][:, np.newaxis, :]  # equation (25)

    MP = AM * AP
    dAMdAT = -MP[:, :, np.newaxis] * dzdAT[:, 2, :][:, np.newaxis, :]  # equation (33)
    dAMdAn = -MP[:, :, np.newaxis] * dzdAn[:, 2, :][:, np.newaxis, :]  # equation (31)
    dAMdAm = (
        np.exp(-beta * AP)[:, :, np.newaxis] * np.eye(nratios)[np.newaxis, :, :]
        - MP[:, :, np.newaxis] * dzdAm[:, 2, :][:, np.newaxis, :]
    )  # equation (32)

    return (
        dzdAn, dzdAT, dzdAm,
        dANdAn, dANdAT, dANdAm,
        dAMdAn, dAMdAT, dAMdAm,
    )


def _fcerrorpropagation_many(z, AP, An, AT, Am, VAn, VAT, VAm, invrat):
    """Batched linear error propagation for the fractionation correction."""
    (
        dzdAn, dzdAT, dzdAm,
        dANdAn, dANdAT, dANdAm,
        dAMdAn, dAMdAT, dAMdAm,
    ) = _sensitivity_many(z, AP, An, AT, Am, invrat)

    def sandwich(A, V):
        return A @ V @ np.swapaxes(A, -1, -2)

    Vz = sandwich(dzdAn, VAn) + sandwich(dzdAT, VAT) + sandwich(dzdAm, VAm)
    VAN = sandwich(dANdAn, VAn) + sandwich(dANdAT, VAT) + sandwich(dANdAm, VAm)
    VAM = sandwich(dAMdAn, VAn) + sandwich(dAMdAT, VAT) + sandwich(dAMdAm, VAm)
    return Vz, VAN, VAM
