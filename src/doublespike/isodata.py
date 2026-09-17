"""Module which handles the data on isotope systems."""
import csv
import os
from importlib import resources

import numpy as np

# Fundamental constants
elementarycharge = 1.60217646e-19  # coulombs
k = 1.3806504e-23  # m^2 kg s^-2 K^-1, Boltzmann constant


def _datafile():
    """Path of the bundled isotope system data file.

    ``importlib.resources`` replaces ``pkg_resources``, which was deprecated by
    setuptools and is no longer guaranteed to be installed.
    """
    try:  # python >= 3.9
        return str(resources.files(__package__).joinpath("data", "maininput.csv"))
    except AttributeError:  # pragma: no cover - very old python
        with resources.path(__package__, "data/maininput.csv") as p:
            return str(p)


def loadrawdata(filename=None):
    """Read in isotope system datafile and return a python dictionary with data."""
    if filename is None:
        filename = _datafile()
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find isotope system datafile {filename!r}")

    with open(filename, newline="") as f:
        csvreader = csv.reader(f)

        data = {}
        next(csvreader)  # ignore header row
        els = []
        el = None

        for row in csvreader:
            if len(row) < 4:
                continue
            if len(row[0]) > 0:
                el = row[0].strip('"').strip()
                els.append(el)
                data[el] = {
                    "element": el,
                    "isonum": [],
                    "mass": [],
                    "standard": [],
                    "rawspike": [],
                }
            if el is None:
                raise ValueError(
                    "Malformed isotope data file: the first data row is missing "
                    "an element name."
                )
            data[el]["isonum"].append(int(row[1]))
            data[el]["mass"].append(float(row[2]))
            data[el]["standard"].append(float(row[3]))
            rs = [float(r) for r in row[4:] if len(r) > 0]
            data[el]["rawspike"].append(rs)

    # Renormalise compositions, convert to numpy format
    for el in els:
        data[el]["isonum"] = np.array(data[el]["isonum"])
        data[el]["mass"] = np.array(data[el]["mass"])

        s = np.array(data[el]["standard"])
        data[el]["standard"] = s / sum(s)

        rs = np.array(data[el]["rawspike"])
        rs = rs / rs.sum(axis=0)
        data[el]["rawspike"] = rs.T
        if len(data[el]["rawspike"]) == 0:
            data[el]["rawspike"] = None

    return data


default_data = loadrawdata()


class IsoData:
    """Object which stores data on an isotope system.

    Args:
        element (str): the element string (e.g. 'Fe')

    Attributes:
        element (str): the element e.g. 'Fe'
        isonum (array): the isotopes of this element e.g. [54, 56, 57, 58]
        mass (array): the atomic masses
        standard (array): composition of a standard as a vector
        spike (array): composition of a double spike used
        isoinv (array): the 4 isotopes to use the inversion
        rawspike (array): composition of single spikes available
        errormodel (dict): dictionary describing the errormodel
    """

    def __init__(self, element):
        """Given an element, initialise with some sensible default values."""
        if element in default_data.keys():
            default = default_data[element]
            self.element = default["element"]
            self.isonum = default["isonum"]
            self.mass = default["mass"]
            self.standard = default["standard"]
            self.rawspike = default["rawspike"]
        else:
            self.element = element
            self.isonum = None
            self.mass = None
            self.standard = None
            self.rawspike = None
        self.spike = None
        if self.nisos == 4:
            self.isoinv = np.array(self.isonum)  # copy: isoinv and isonum must stay independent
        else:
            self.isoinv = None
        self.errormodel = {}
        self.set_errormodel()

    def copy(self):
        """Return an independent deep-enough copy of this isotope system.

        Useful in exploratory work where the same element is evaluated with
        several standard compositions, spikes or error models.
        """
        other = IsoData(self.element)
        for attr in ("isonum", "mass", "standard", "spike", "isoinv", "rawspike"):
            value = getattr(self, attr)
            setattr(other, attr, None if value is None else np.array(value))
        other.errormodel = {
            k: {kk: (np.array(vv) if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()}
            for k, v in self.errormodel.items()
        }
        return other

    def __repr__(self):
        return "IsoData()"

    def __str__(self):
        txt = "\n ".join(
            f"{item[0].strip('_')}: {item[1]}" for item in vars(self).items()
        )
        return txt

    @property
    def mass(self):
        """Atomic masses of the isotopes."""
        return self._mass

    @mass.setter
    def mass(self, value):
        if value is None:
            self._mass = None
        else:
            self._mass = np.array(value)

    @property
    def isonum(self):
        "Isotope numbers."
        return self._isonum

    @isonum.setter
    def isonum(self, value):
        if value is None:
            self._isonum = None
        else:
            self._isonum = np.array(value, dtype=int)

    @property
    def isoinv(self):
        "Inversion isotopes."
        return self._isoinv

    @isoinv.setter
    def isoinv(self, value):
        if value is None:
            self._isoinv = None
        else:
            self._isoinv = np.array(value, dtype=int)

    @property
    def standard(self):
        "Composition of standard."
        return self._standard

    @standard.setter
    def standard(self, value):
        if value is None:
            self._standard = None
        else:
            std = np.array(value)
            self._standard = std / sum(std)

    @property
    def spike(self):
        "Composition of double spike."
        return self._spike

    @spike.setter
    def spike(self, value):
        if value is None:
            self._spike = None
        else:
            spk = np.array(value)
            self._spike = spk / sum(spk)

    @property
    def rawspike(self):
        "Compositions of single spikes."
        return self._rawspike

    @rawspike.setter
    def rawspike(self, value):
        if value is None:
            self._rawspike = None
        else:
            rs = np.array(value).T
            self._rawspike = (rs / rs.sum(axis=0)).T

    def isoindex(self, ix):
        """Give the data index corresponding to a given isotope number e.g. 56->1."""
        if ix is None:
            return None
        arr = np.asarray(ix)
        lut = self._isoindex_map
        if arr.ndim == 0:
            k = int(arr)
            return lut.get(k, k)
        values = [lut.get(int(v), int(v)) for v in arr.ravel()]
        out = np.array(values, dtype=int).reshape(arr.shape)
        if isinstance(ix, list):
            return out.tolist()
        return out

    @property
    def _isoindex_map(self):
        """Cached mapping from isotope number to data index.

        The original implementation used ``np.vectorize`` here, which cost more
        than the whole rest of the error calculation: it is called several times
        per error estimate.  A plain dict lookup keyed on the current ``isonum``
        array is around two orders of magnitude cheaper.
        """
        cached = self.__dict__.get("_isoindex_cache")
        if cached is None or cached[0] is not self.isonum:
            cached = (self.isonum, {int(k): i for i, k in enumerate(self.isonum)})
            self.__dict__["_isoindex_cache"] = cached
        return cached[1]

    @property
    def isoname(self):
        """Names of the isotopes."""
        return [self.element + str(i) for i in self.isonum]

    @property
    def isolabel(self):
        """Isotope labels for plotting."""
        return ["$^{" + str(i) + "}$" + self.element for i in self.isonum]

    @property
    def rawspikelabel(self):
        """Single spike labels for plotting."""
        return ["spike " + str(i + 1) for i in range(self.rawspike.shape[0])]

    @property
    def nisos(self):
        """Number of isotopes in system."""
        if self.isonum is None:
            return 0
        return len(self.isonum)

    @property
    def nratios(self):
        """Number of isotope ratios to describe system."""
        return self.nisos - 1

    @property
    def nrawspikes(self):
        """Number of single spikes available."""
        if self.rawspike is None:
            return 0
        return self.rawspike.shape[0]

    def set_errormodel(
        self,
        intensity=10.0,
        deltat=8.0,
        R=1e11,
        T=300.0,
        radiogenic=None,
        measured_type="fixed-total",
        R_reference=1e11,
    ):
        """Set the error model used for error estimates and monte carlo runs.

        Args:
            intensity (float): Total beam intensity in volts
            deltat (float): Integration time in seconds
            R (float/array): resistance in Ohms
                If a float then resistors are identical for all isotopes
                If an array then individual resistances for each isotope
            T (float): temperature in K
            radiogenic (str): If True put errors on the standard (unspiked) run, if False do not.
                If None, then 'Pb','Sr','Hf','Os','Nd' are assumed radiogenic, others not.
            measured_type (str): If 'fixed-total' then total beam intensity of mixture is fixed,
                If 'fixed-sample' then beam current from the sample is fixed.
            R_reference (float): reference resistance used for describing beam intensity.
                The total beam current is intensity/R_reference
                e.g. the 10 V default beam corresponds to 100 pA with R_reference=1e11 Ohms.
        """
        if radiogenic is None:
            if self.element in ["Pb", "Sr", "Hf", "Os", "Nd"]:
                radiogenic = True
            else:
                radiogenic = False

        nisos = self.nisos

        if isinstance(R, float):
            # if a float given for R assume resistors are the same for all beams
            R = R * np.ones(nisos)

        if len(R) != nisos:
            raise Exception("Must have same length for R array as number of isotopes.")

        R = np.array(R)  # ensure working with a numpy array

        # Equation (35), modified to allow setting of different resistors for different beams as suggested by Chris Coath
        a = 4 * k * T * (R_reference**2) / (deltat * R)  # Johnson-Nyquist noise
        b = elementarycharge * R_reference / deltat  # Counting statistics

        # by default assume Johnson noise and counting statistics
        self.errormodel["measured"] = {
            "type": measured_type,
            "intensity": intensity,
            "a": a * np.ones(nisos),
            "b": b * np.ones(nisos),
            "c": 0.0 * np.ones(nisos),
        }
        self.errormodel["spike"] = {
            "type": "fixed-total",
            "intensity": intensity,
            "a": 0.0 * np.ones(nisos),
            "b": 0.0 * np.ones(nisos),
            "c": 0.0 * np.ones(nisos),
        }
        if radiogenic:
            # if a radiogenic isotope then put errors on the standard (unspiked) run
            self.errormodel["standard"] = {
                "type": "fixed-total",
                "intensity": intensity,
                "a": a * np.ones(nisos),
                "b": b * np.ones(nisos),
                "c": 0.0 * np.ones(nisos),
            }
        else:
            # otherwise assume no errors on standard composition
            self.errormodel["standard"] = {
                "type": "fixed-total",
                "intensity": intensity,
                "a": 0.0 * np.ones(nisos),
                "b": 0.0 * np.ones(nisos),
                "c": 0.0 * np.ones(nisos),
            }

    def ratio(self, composition, denominator_isotope):
        """Convert a compositional array into array of isotopic ratios."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos)
        ni = ni[ni != di]
        return composition[..., ni] / composition[..., di, np.newaxis]

    def composition(self, data, denominator_isotope):
        """Convert an array of isotopic ratios to an array of compositional vectors."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos)
        ni = ni[ni != di]
        comp_shape = list(data.shape)
        comp_shape[-1] += 1
        comp = np.ones(comp_shape)
        comp[ni] = data
        comp = normalise_composition(comp)
        return comp

    def invrat(self, isoinv=None):
        """Indices of isotope ratios used in inversion."""
        if isoinv is None:
            isoinv = self.isoinv

        isoinv = self.isoindex(isoinv)  # convert to indices

        isonum = np.arange(self.nisos)
        isonum = isonum[isonum != isoinv[0]]
        isonum = np.concatenate((np.array([isoinv[0]]), isonum))

        invrat = np.array([np.where(isonum == i)[0][0] for i in isoinv])
        invrat = invrat[1:] - 1
        return invrat

    def rationame(self, denominator_isotope):
        """Names of the isotopic ratios."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos)
        ni = ni[ni != di]
        return np.array(
            [
                str(i) + self.element + "/" + str(self.isonum[di]) + self.element
                for i in self.isonum[ni]
            ]
        )

    def ratioidx(self, numerator_isotope, denominator_isotope):
        """The index corresponding to a particular isotopic ratio."""
        di = self.isoindex(denominator_isotope)
        ni = self.isoindex(numerator_isotope)
        if ni < di:
            return ni
        if ni == di:
            return None
        if ni > di:
            return ni - 1


def normalise_composition(comp):
    """Normalise rows of an array to unit sum, i.e. rows are compositional vectors.

    Works for any number of leading dimensions, including a bare 1-D array, and
    no longer relies on ``type(...) is float`` (which silently fails for numpy
    scalars).
    """
    comp = np.asarray(comp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return comp / comp.sum(axis=-1, keepdims=True)


def ratioproptorealprop(lambda_, ratio_a, ratio_b):
    """Convert a proportion in ratio space to one per mole."""
    a = 1 + np.sum(ratio_a, axis=-1)
    b = 1 + np.sum(ratio_b, axis=-1)
    return lambda_ * a / (lambda_ * a + (1 - lambda_) * b)  # equation (9)


def realproptoratioprop(prop, ratio_a, ratio_b):
    """Convert a proportion per mole into ratio space."""
    a = 1 + np.sum(ratio_a, axis=-1)
    b = 1 + np.sum(ratio_b, axis=-1)
    return prop * b / (prop * b + (1 - prop) * a)  # equation (9)


def ratio(data, isoidx):
    """Convert data to isotope ratios based on choice of isotopes. First index is the denominator index."""
    di = isoidx[0]
    ni = isoidx[1:]
    return data[..., ni] / data[..., di, np.newaxis]
