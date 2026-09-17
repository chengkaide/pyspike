"""Routines for Monte Carlo mass spec simulation."""
import numpy as np

from .isodata import normalise_composition


def monterun(isodata, prop, spike=None, alpha=0.0, beta=0.0, n=1000, seed=None, rng=None):
    """Generate a fake mass spectrometer run by Monte-Carlo simulation.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        prop (float/array): proportion of spike in double spike-sample mix.
            A vector of values can be specified if desired, to reflect changes over a run.
        spike (array): a composition vector for the spike. e.g. [0, 0, 0.5, 0.5] is a 50-50
            mix of 57Fe and 58Fe. If None this is read from isodata.
        alpha (float/array): the natural fractionations (can be float or array of values)
        beta (float/array): the instrumental fractionations (can be float or array of values)
        n (int): number of Monte-Carlo samples to take. Default is 1000.
        seed (int): optional seed for the random number generator.  Setting this
            makes a Monte Carlo run exactly reproducible, which matters when the
            result is quoted in a paper.
        rng (Generator): an alternative to ``seed``; a numpy random Generator to
            draw from.

        Note that behaviour depends on the error model specified in isodata.errormodel.

    Example:
        >>> isodata = IsoData('Fe')
        >>> measured = monterun(isodata, 0.5, [0, 0, 0.5, 0.5], seed=42)
    """
    # Get spike from isodata if not supplied as argument
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        else:
            spike = isodata.spike
    spike = np.array(spike, dtype=float)

    if isodata.errormodel == {}:
        raise Exception("Need to set errormodel")

    if rng is None:
        rng = np.random.default_rng(seed)

    standard = isodata.standard
    mass = isodata.mass
    emodel = isodata.errormodel
    spike = spike / sum(spike)

    # accept plain scalars, numpy scalars, 0-d arrays and lists alike
    def as_vector(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return arr * np.ones(n)
        if arr.ndim == 1:
            if arr.size != n:
                raise ValueError(f"{name} must have length n={n}, got {arr.size}")
            return arr
        raise ValueError(f"{name} must be a scalar or a 1-D array")

    alpha = as_vector(alpha, "alpha")
    beta = as_vector(beta, "beta")
    prop = as_vector(prop, "prop")

    P = np.log(mass)[np.newaxis, :]
    alpha = alpha[:, np.newaxis]
    beta = beta[:, np.newaxis]
    prop = prop[:, np.newaxis]

    standard = standard[np.newaxis, :]
    spike = spike[np.newaxis, :]

    sample = standard * np.exp(-P * alpha)
    sample = normalise_composition(sample)

    mixture = prop * spike + (1 - prop) * sample

    measured = mixture * np.exp(+P * beta)
    measured = normalise_composition(measured)

    # Always perform Monte Carlo simulation for the spike-sample mix
    measuredv = monte_single(measured, emodel["measured"], rng=rng)

    # If errormodel has variance on standard, do Monte Carlo simulation for it
    standardv = None
    standardi, standardivar = calc_var(standard, emodel["standard"], prop)
    if np.any(standardivar > 0.0):
        standardv = monte_single(standard, emodel["standard"], n, rng=rng)

    # If errormodel has variance on spike, do Monte Carlo simulation for it
    spikev = None
    spikei, spikeivar = calc_var(spike, emodel["spike"])
    if np.any(spikeivar > 0.0):
        spikev = monte_single(spike, emodel["spike"], n, rng=rng)

    if standardv is None and spikev is None:
        return measuredv
    elif standardv is not None and spikev is None:
        return measuredv, standardv
    elif standardv is None and spikev is not None:
        return measuredv, spikev
    else:
        return measuredv, standardv, spikev


def calc_var(data, emod, prop=None):
    """Calculate the beam variances."""
    datai = data * emod["intensity"]
    if emod["type"] == "fixed-sample":
        if prop is None:
            raise ValueError("prop must be supplied for a fixed-sample error model")
        datai = datai / (1.0 - prop)
    dataivar = emod["a"] + emod["b"] * datai + emod["c"] * (datai**2)  # equation (34)
    return datai, dataivar


def monte_single(composition, emod, n=None, rng=None):
    """Perform Monte-Carlo simulation for a given composition and error model."""
    if rng is None:
        rng = np.random.default_rng()
    if n is not None:
        composition = np.tile(composition, (n, 1))
    composition = normalise_composition(composition)

    datai, dataivar = calc_var(composition, emod)

    datav = rng.normal(loc=datai, scale=np.sqrt(np.maximum(dataivar, 0.0)))
    return datav
