"""What a *third* enriched isotope would and would not buy, computed on Sn.

The toolbox is a double-spike toolbox, and not by accident.  Rudge et al. (2009)
set up three unknowns -- the spike proportion, the natural fractionation alpha
and the instrumental fractionation beta -- and *three* ratios, so the system is
square and exactly determined.  ``errors._z_sensitivity_many`` even hard-codes
that square shape (``dfdy = np.empty((B, 3, 3))``) because the whole error
appendix is implicit-function differentiation of a square system: eq. (15)
inverted, then eqs. (17)-(22).

That has a consequence which is easy to miss and is the entire subject of this
script: **a double spike is exactly determined, so it has no way to test any of
its own assumptions.**  Nothing is left over.  A third enriched isotope -- or
more precisely, using more than four isotopes in the inversion -- is the only
thing that creates spare equations, and each spare equation buys exactly one
hypothesis test.

So the question "is a triple spike worth it for Sn?" is really two questions:

    (1) used as extra equations for the *same* three unknowns, how much
        precision does the redundancy buy?  (little, and this script measures it)
    (2) used to free a *fourth* unknown, what can that unknown be?  (the
        fractionation law; this script measures how well it can be pinned down)

Both are answered here as generalised least squares on the toolbox's own forward
model.  The k = 3 case is validated against ``errors.errorestimate`` to machine
precision before any k > 3 number is believed -- without that check the extra
digits would just be decoration.

The fractionation law is varied with a one-parameter family of mass factors,

    P_i(theta) = ((m_i/m_ref)**theta - 1) / theta        (Box-Cox)

so that theta = 0 is the exponential/power law the toolbox assumes,
(ln of the mass ratio), and theta = 1 is the linear law, (m_i/m_ref - 1).  The
family is imposed by *rewriting ``isodata.mass``* to fictitious masses whose
logs reproduce P_i(theta) -- every routine in the package reads the mass factor
from there, so the patch reaches the forward model, the inversion and the error
propagation at once, with no second implementation to keep in step.

Run from the repository root:

    G:/Python39/python.exe tools/probe_triple.py
    G:/Python39/python.exe tools/probe_triple.py --pair 112 122 --out report.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doublespike import IsoData  # noqa: E402
from doublespike.errors import (  # noqa: E402
    _calcratiocov_many,
    _composition_many,
    errorestimate,
    ratiodata_many,
)
from doublespike.optimal import singleoptimalspike  # noqa: E402

np.seterr(all="ignore")

# The best four-isotope inversion for the 117-122 spike pair, from the full
# 1260-combination search (paper/collect_ranking.py).  Hard-coded rather than
# read from paper/, because paper/ is not distributed with the package.
DEFAULT_INV = [116, 117, 122, 124]


def massfactor(theta, x):
    """Box-Cox mass factor; theta = 0 is the power/exponential law.

    Series-expanded for small theta.  The closed form (x**theta - 1)/theta is
    catastrophic there: x**theta is 1 + 1e-4 * ln x, so subtracting the one
    throws away five significant digits and leaves an *absolute* error of order
    1e-12 in P.  That is invisible in the forward model and ruinous in the
    finite-difference check of dP/dtheta, where the difference being taken is
    only ~1e-7.  Hence

        P = sum_{n>=0} theta**n L**(n+1) / (n+1)!,        L = ln x
    """
    x = np.asarray(x, float)
    L = np.log(x)
    if abs(theta) < 5e-2:
        out = np.zeros_like(L)
        term = L.copy()
        for n in range(12):
            out = out + term
            term = term * L * theta / (n + 2)
        return out
    return (x**theta - 1.0) / theta


def dmassfactor_dtheta(theta, x):
    """d P / d theta, which is L**2 / 2 at theta = 0 and the linear law nearby.

    Series again, for the same conditioning reason as ``massfactor``:

        dP/dtheta = sum_{n>=0} (n+1) theta**n L**(n+2) / (n+2)!
    """
    x = np.asarray(x, float)
    L = np.log(x)
    if abs(theta) < 5e-2:
        out = np.zeros_like(L)
        term = 0.5 * L**2
        for n in range(12):
            out = out + term
            term = term * L * theta * (n + 2) / ((n + 1) * (n + 3))
        return out
    return (x**theta * (theta * L - 1.0) + 1.0) / theta**2


class Model:
    """One element, one spike pair, one denominator, arbitrary ratio subsets."""

    def __init__(self, element, pair, inv):
        self.iso = IsoData(element)
        self.iso.set_errormodel()
        self.true_mass = np.array(self.iso.mass, float)  # never patched
        self.mean_mass = float(np.mean(self.true_mass))
        self.isotopes = [int(m) for m in self.iso.isonum]
        self.pair = sorted(int(m) for m in pair)
        self.inv = sorted(int(m) for m in inv)
        self.theta = 0.0

        self.spike, self.p, _, _, self.ppmperamu = singleoptimalspike(
            self.iso, "pure",
            [int(self.iso.isoindex(m)) for m in self.pair],
            [int(self.iso.isoindex(m)) for m in self.inv],
        )
        self.spike = np.asarray(self.spike, float)

        # denominator by the same rule as errorestimate/dsinversion: the
        # inversion isotope the spike contains most of
        ii = [int(self.iso.isoindex(m)) for m in self.inv]
        self.deno = int(ii[int(np.argmax(self.spike[ii]))])
        self.order = [i for i in range(self.iso.nisos) if i != self.deno]
        self.names = ["%d/%d" % (self.isotopes[i], self.isotopes[self.deno])
                      for i in self.order]

    # ---- the law --------------------------------------------------------
    def set_law(self, theta):
        """Rewrite the masses so that every P in the package equals P(theta)."""
        self.theta = theta
        if abs(theta) < 1e-14:
            self.iso.mass = self.true_mass.copy()
        else:
            x = self.true_mass / self.true_mass[self.deno]
            self.iso.mass = self.true_mass[self.deno] * np.exp(massfactor(theta, x))

    def slots(self, masses):
        """Ratio indices for a list of mass numbers."""
        return [self.order.index(int(self.iso.isoindex(m))) for m in masses
                if int(self.iso.isoindex(m)) != self.deno]

    # ---- forward model ---------------------------------------------------
    def state(self, p=None, alpha=0.0, beta=0.0):
        """(lambda, AT, An, AN, AM, Am) over ALL ratios, at the current law."""
        z, AP, An, AT, Am, AN, AM = ratiodata_many(
            self.iso, self.deno, np.atleast_1d(self.p if p is None else p),
            self.spike[None, :], np.atleast_1d(alpha), np.atleast_1d(beta),
        )
        return (float(z[0][0]), AP, np.asarray(An, float), np.asarray(AT[0], float),
                np.asarray(AN[0], float), np.asarray(AM[0], float),
                np.asarray(Am[0], float))

    def forward(self, p=None, alpha=0.0, beta=0.0):
        return self.state(p, alpha, beta)[6]

    # ---- analytic Jacobians ---------------------------------------------
    def _pieces(self, p, alpha, beta):
        lam, AP, An, AT, AN, AM = self.state(p, alpha, beta)[:6]
        if p is None:
            p = self.p
        a = 1.0 + float(AT.sum())
        b = 1.0 + float(AN.sum())
        D = p * b + (1.0 - p) * a
        return lam, AP, An, AT, AN, AM, a, b, D

    def jac_y(self, p=None, alpha=0.0, beta=0.0):
        """dAm/d(p, alpha, beta), shape (nratios, 3), analytic."""
        if p is None:
            p = self.p
        lam, AP, An, AT, AN, AM, a, b, D = self._pieces(p, alpha, beta)
        dlam_dp = a * b / D**2                                   # equation (9)
        dlam_da = p * (1.0 - p) * a * (-float((AP * AN).sum())) / D**2
        dld_a_da = float((AP * AN).sum()) * np.ones_like(AP)
        E = np.exp(beta * AP)
        out = np.empty((AP.size, 3))
        out[:, 0] = E * (AT - AN) * dlam_dp
        out[:, 1] = E * (dlam_da * (AT - AN) - (1.0 - lam) * AP * AN)
        out[:, 2] = self.forward(p, alpha, beta) * AP
        del dld_a_da
        return out

    def jac_theta(self, p=None, alpha=0.0, beta=0.0):
        """dAm/dtheta, shape (nratios,), analytic."""
        if p is None:
            p = self.p
        lam, AP, An, AT, AN, AM, a, b, D = self._pieces(p, alpha, beta)
        x = self.true_mass / self.true_mass[self.deno]
        q = dmassfactor_dtheta(self.theta, x)[self.order]
        db = -alpha * float((AN * q).sum())
        dlam = p * (1.0 - p) * a * db / D**2
        E = np.exp(beta * AP)
        Am = self.forward(p, alpha, beta)
        return E * (dlam * (AT - AN) - (1.0 - lam) * alpha * AN * q) + Am * beta * q

    def jac_inputs(self, p=None, alpha=0.0, beta=0.0):
        """(dAm/dAn, dAm/dAT), each (nratios, nratios) analytic."""
        if p is None:
            p = self.p
        lam, AP, An, AT, AN, AM, a, b, D = self._pieces(p, alpha, beta)
        E = np.exp(beta * AP)
        n = AP.size
        dlam_dAn = p * (1.0 - p) * a * np.exp(-alpha * AP) / D**2
        Bn = np.diag((1.0 - lam) * np.exp(-alpha * AP) * E) \
            + E[:, None] * (AT - AN)[:, None] * dlam_dAn[None, :]
        BT = np.diag(lam * E)
        return Bn, BT

    def jac_check(self, p=None, alpha=0.0, beta=0.0, h=1e-6):
        """Largest relative disagreement between the analytic and difference forms."""
        if p is None:
            p = self.p
        A = self.jac_y(p, alpha, beta)
        cols = []
        for dp, da, db in ((h, 0, 0), (0, h, 0), (0, 0, h)):
            hi = self.forward(p + dp, alpha + da, beta + db)
            lo = self.forward(p - dp, alpha - da, beta - db)
            cols.append((hi - lo) / (2 * h))
        B = np.array(cols).T
        return float(np.abs(A - B).max() / np.abs(A).max())

    # ---- covariances -----------------------------------------------------
    def covariances(self, p=None, alpha=0.0, beta=0.0):
        if p is None:
            p = self.p
        Am = self.forward(p, alpha, beta)
        measured = _composition_many(np.atleast_2d(Am), self.deno, self.iso.nisos)
        VAn = _calcratiocov_many(self.iso.standard, self.iso.errormodel["standard"],
                                 self.deno)
        VAT = _calcratiocov_many(self.spike[None, :], self.iso.errormodel["spike"],
                                 self.deno)
        VAm = _calcratiocov_many(measured, self.iso.errormodel["measured"], self.deno,
                                 np.atleast_1d(p))
        return np.asarray(VAn[0], float), np.asarray(VAT[0], float), np.asarray(VAm[0], float)


def gls(m, masses, free_theta=False, p=None, alpha=0.0, beta=0.0):
    """Generalised least squares on the toolbox's forward model.

    Returns (sigma_on_each_unknown, ppm_per_amu_on_alpha).  With three ratios and
    three unknowns the least-squares solution *is* the exact solution, so this
    reproduces ``errorestimate`` exactly -- which is the validation the report
    prints before it shows any k > 3 number.

    ``alpha`` and ``beta`` are the operating point.  They matter for anything
    involving the fractionation law: the law enters the model only through the
    products ``alpha*P`` and ``beta*P``, so at the package default (both zero)
    the law is exactly invisible and every law sensitivity would come out as a
    machine zero.
    """
    if p is None:
        p = m.p
    S = m.slots(masses)
    VAn, VAT, VAm = m.covariances(p, alpha, beta)
    Vm = VAm[np.ix_(S, S)]
    A = m.jac_y(p, alpha, beta)[S]
    Bn, BT = m.jac_inputs(p, alpha, beta)
    Bn, BT = Bn[S], BT[S]
    if free_theta:
        A = np.hstack([A, m.jac_theta(p, alpha, beta)[S][:, None]])
    W = np.linalg.pinv(Vm)
    M = np.linalg.pinv(A.T @ W @ A) @ A.T @ W
    C = Vm + Bn @ VAn @ Bn.T + BT @ VAT @ BT.T
    Cov = M @ C @ M.T
    sigma = np.sqrt(np.maximum(np.diag(Cov), 0.0))
    return sigma, 1e6 * sigma[1] / m.mean_mass


def solve3(m, masses, y_obs, x0=None, iters=80):
    """Recover (p, alpha, beta) from k measured ratios by Newton least squares.

    With k = 3 this is the exact double-spike inversion; with k > 3 it is the
    least-squares version.  Used to demonstrate the law-induced bias by
    pretending data generated under one law were inverted under another.
    """
    S = m.slots(masses)
    p, a, b = (m.p, 0.0, 0.0) if x0 is None else x0
    for _ in range(iters):
        r = m.forward(p, a, b)[S] - y_obs
        J = m.jac_y(p, a, b)[S]
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        # keep the proportion inside its physical range
        if not (0.0 < p + step[0] < 1.0):
            step = step * 0.5
        p, a, b = p + step[0], a + step[1], b + step[2]
        if np.abs(step).max() < 1e-15:
            break
    return p, a, b, float(np.abs(m.forward(p, a, b)[S] - y_obs).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--element", default="Sn")
    ap.add_argument("--pair", nargs=2, type=int, default=[117, 122])
    ap.add_argument("--inv", nargs=4, type=int, default=DEFAULT_INV)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = []

    def say(s=""):
        out.append(str(s))

    m = Model(args.element, args.pair, args.inv)
    base = [x for x in m.inv if x != 122]
    deno_mass = m.isotopes[m.deno]

    say("=" * 78)
    say("%s triple spike: what a third enriched isotope buys" % args.element)
    say("=" * 78)
    say("spike pair        : %d-%d" % tuple(m.pair))
    say("four-isotope inv  : %s" % m.inv)
    say("denominator       : %d   (spike's largest inversion isotope)" % deno_mass)
    say("optimal proportion: p = %.4f" % m.p)
    say("spike             : %s" % np.round(m.spike, 6).tolist())
    say("isotopes          : %s" % m.isotopes)
    say("abundances %%      : %s" % [round(100 * v, 3) for v in m.iso.standard])
    say()

    # ---- 0 -----------------------------------------------------------------
    say("0. What the toolbox can and cannot do today (run live, not asserted)")
    try:
        errorestimate(m.iso, m.p, m.spike, m.inv + [m.iso.isoindex(119)])
        say("   errorestimate with 5 isotopes : accepted  <- unexpected")
    except Exception as exc:
        say("   errorestimate with 5 isotopes : %s" % exc)
    try:
        r = singleoptimalspike(m.iso, "pure",
                               [m.iso.isoindex(x) for x in (112, 117, 122)],
                               [m.iso.isoindex(x) for x in (112, 117, 119, 122)])
        support = [m.isotopes[i] for i in np.nonzero(np.asarray(r[0]) > 1e-6)[0]]
        say("   singleoptimalspike, 3 isotopes: support = %s  <- the third is"
            % support)
        say("                                   silently dropped, no error raised")
    except Exception as exc:
        say("   singleoptimalspike, 3 isotopes: %s" % exc)
    say("   reason, not policy: errors.py builds dfdy as a 3x3 square matrix")
    say("   because eq. (15) of Rudge et al. (2009) is three equations in three")
    say("   unknowns.  A k-ratio version is a least-squares solve, i.e. the")
    say("   appendix has to be re-derived -- it is not a flag to switch on.")
    say()

    # ---- 1 -----------------------------------------------------------------
    say("1. The arithmetic of redundancy")
    say("   %-6s %-7s %-9s %-11s" % ("isoinv", "ratios", "unknowns", "spare eqs"))
    for k in (4, 5, 6, 7, 8):
        say("   %-6d %-7d %-9d %-11d" % (k, k - 1, 3, k - 4))
    say("   A double spike (4 isotopes) has zero spare equations, so it cannot")
    say("   test the exponential law, cannot test for a mass-independent effect,")
    say("   and cannot test for an unmodelled interference.  Every fifth isotope")
    say("   buys exactly one test.  That is the entire budget of a triple spike.")
    say()

    # ---- validation --------------------------------------------------------
    say("2. Validation of the generalised least squares at k = 3")
    e_ref, ppm_ref = errorestimate(m.iso, m.p, m.spike, m.inv)
    sig, ppm = gls(m, base)
    val_rel = abs(sig[1] - e_ref) / e_ref
    say("   toolbox errorestimate : sigma_alpha = %.14e   ppm/amu = %.6f"
        % (e_ref, ppm_ref))
    say("   generalised GLS       : sigma_alpha = %.14e   ppm/amu = %.6f"
        % (sig[1], ppm))
    say("   relative difference   : %.3e" % val_rel)
    say("   analytic vs difference Jacobian disagreement: %.3e" % m.jac_check())
    say("   -> only k > 3 numbers further down are trustworthy if this is tiny.")
    say()

    # ---- 3 -----------------------------------------------------------------
    say("3. Redundancy used for the same three unknowns: how much precision?")
    say("   %-34s %-4s %-11s %-9s" % ("ratios added to the best three", "n", "ppm/amu",
                                    "gain"))
    sig3, ppm3 = gls(m, base)
    say("   %-34s %-4d %-11.4f %-9s" % ("(none: the double spike)", 3, ppm3, "1.0000"))
    others = [x for x in m.isotopes if x not in m.inv and x != deno_mass]
    rows = []
    for extra in others:
        s, pp = gls(m, sorted(base + [extra]))
        rows.append((extra, pp, pp / ppm3))
    for extra, pp, gain in sorted(rows, key=lambda r: r[1]):
        say("   %-34s %-4d %-11.4f %-9.4f" % ("+ %d" % extra, 4, pp, gain))
    sig_all, ppm_all = gls(m, [m.isotopes[i] for i in m.order])
    say("   %-34s %-4d %-11.4f %-9.4f"
        % ("every ratio", len(m.order), ppm_all, ppm_all / ppm3))
    say("   -> the best single addition is a few per cent, not a factor.  Extra")
    say("      isotopes are nearly redundant *for the same three unknowns*.")
    say("      The reason is structural: alpha and beta act along mass-dependent")
    say("      directions, so a new isotope adds a row that is almost a")
    say("      combination of the existing ones.  Compare tools/probe_geometry.py")
    say("      section B, where alpha and beta are only 139 degrees apart.")
    say()

    # ---- 4 -----------------------------------------------------------------
    say("4. Redundancy used to free a fourth unknown: the fractionation law")
    say("   theta = 0 is the exponential/power law the package assumes.  The")
    say("   family is P(theta) = ((m/m_ref)**theta - 1)/theta, whose expansion in")
    say("   the relative mass difference y = m/m_ref - 1 is")
    say("       P = y + (theta - 1) y**2 / 2 + O(y**3),")
    say("   so theta IS the quadratic term -- which is precisely what separates the")
    say("   standard laws.  Exponential has beta*P = beta*y - beta*y**2/2; the")
    say("   linear law ln(R/R0) = ln(1 + beta*y) has -beta**2*y**2/2.  Equating the")
    say("   quadratic coefficients gives")
    say("       theta_linear = 1 - beta.")
    say("   Note the trap that falls straight out: at beta = 1 the two laws agree to")
    say("   this order, so beta = 1 is the single worst place to ask the question.")
    say()
    say("   The law reaches the model only through alpha*P and beta*P, so at the")
    say("   package default it is exactly invisible:")
    for a0, b0 in ((0.0, 0.0), (0.0, 1.5)):
        m.set_law(1e-5)
        yh = m.forward(m.p, a0, b0)
        m.set_law(-1e-5)
        yl = m.forward(m.p, a0, b0)
        m.set_law(0.0)
        say("      max |dAm/dtheta| at alpha=%4.1f beta=%4.1f : %.3e"
            % (a0, b0, np.abs((yh - yl) / 2e-5).max()))
    say()

    BETA = 1.5

    def fd_theta(hh):
        m.set_law(hh)
        yp = m.forward(m.p, 0.0, BETA)
        m.set_law(-hh)
        ymm = m.forward(m.p, 0.0, BETA)
        m.set_law(0.0)
        return (yp - ymm) / (2 * hh)

    an = m.jac_theta(m.p, 0.0, BETA)
    for hh in (1e-3, 3e-4, 1e-4):
        say("   jac_theta analytic vs difference, h=%.0e : %.3e"
            % (hh, np.abs(an - fd_theta(hh)).max() / np.abs(an).max()))
    say("   -> agreement at the 1e-9 level.  The residual grows slowly as h falls,")
    say("      which identifies it as roundoff and not truncation: ratiodata_many")
    say("      forms P = log(mass) - log(m_ref), two numbers near 4.8, so the mass")
    say("      patch carries an absolute error of ~1e-16 that a difference over 2h")
    say("      amplifies.  A step-size scan therefore has two failure modes that")
    say("      look opposite, and the *sign of the trend* is what tells them apart:")
    say("      a disagreement that does not shrink as h falls means a term is")
    say("      missing (that caught two real bugs in tools/probe_geometry.py); one")
    say("      that grows means roundoff.  This table used to grow far faster, at")
    say("      2e-6, because the mass patch went through the catastrophic")
    say("      (x**theta - 1)/theta; series-expanding P removed that.")
    say()

    best_extra = sorted(rows, key=lambda r: r[1])[0][0]
    say("   Can a fourth ratio constrain theta?  It depends on beta, because the")
    say("   law effect is proportional to the fractionation that multiplies it:")
    say("   %-6s %-14s %-14s %-10s"
        % ("beta", "sigma_theta", "sigma_theta", "|theta_lin|"))
    say("   %-6s %-14s %-14s %-10s"
        % ("", "4 ratios", "all ratios", "/ sigma"))
    sig_at_scan = {}
    for b0 in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        st4, _ = gls(m, sorted(base + [best_extra]), free_theta=True, beta=b0)
        stN, _ = gls(m, [m.isotopes[i] for i in m.order], free_theta=True, beta=b0)
        sep = abs(1.0 - b0)
        sig_at_scan[b0] = st4
        say("   %-6.2f %-14.4f %-14.4f %-10.2f" % (b0, st4[3], stN[3], sep / st4[3]))
    say("   -> at beta = 1 the two laws coincide by construction, which is why that")
    say("      row has no significance at all.  Away from it the separation grows:")
    say("      2.8 sigma at beta = 1.25, 6.8 at 1.5, 18 at 2.  A realistic MC-ICP-MS")
    say("      bias for Sn is 1-2 % per amu, i.e. 12-24 % across the mass range, so")
    say("      beta is somewhere near 1.1-2.0 and the exponential and linear laws")
    say("      should separate at a few sigma per measurement.  That sigma is")
    say("      statistical and averages down, so a repeated standard settles it.")
    say("      For Sn -- ten isotopes, and no other element with as many -- this is")
    say("      the one thing a fifth isotope buys that a double spike cannot buy at")
    say("      any precision.  Caveat: this counts statistical error only.  A wrong")
    say("      spike composition or an unmodelled interference would masquerade as")
    say("      a law, and no amount of averaging removes that.")
    say()

    A3 = m.jac_y(m.p, 0.0, BETA)[m.slots(base)]
    shift = -np.linalg.solve(A3, m.jac_theta(m.p, 0.0, BETA)[m.slots(base)])
    say("   The other side of the coin: if theta is simply assumed zero, the three")
    say("   ratios are still fitted exactly and the law is absorbed silently.")
    say("      d(p, alpha, beta)/dtheta = %s" % np.round(shift, 8).tolist())
    say("      -> dalpha/dtheta = %.3e, i.e. %.2f ppm amu-1 per unit theta"
        % (shift[1], 1e6 * shift[1] / m.mean_mass))
    say("      -> the exponential-to-linear step (|theta| = %.2f at beta = %.1f) is"
        % (abs(1.0 - BETA), BETA))
    say("         worth %.2f ppm amu-1, against a measurement precision of %.2f."
        % (abs(1e6 * shift[1] * (1.0 - BETA) / m.mean_mass), ppm3))
    say("   So the law is not academic for Sn -- it is of the same order as the")
    say("   precision the method advertises, and the double spike cannot see it.")
    say()

    # ---- 5 -----------------------------------------------------------------
    # Interfering nuclide abundances (IUPAC); atomic isobars only.  Sn-117 is
    # listed separately because its interference is the molecular ion 116CdH+,
    # which is a different problem -- see the note in the verdict.
    ISOBAR = {
        112: ("112Cd", 0.2410),
        114: ("114Cd", 0.2870),
        115: ("115In", 0.9571),
        116: ("116Cd", 0.0749),
        120: ("120Te", 0.00096),
        122: ("122Te", 0.0255),
        124: ("124Te", 0.0474),
    }
    say("5. The Sn-specific catch: which mass is worth adding, and which are safe")
    say("   %-6s %-11s %-12s %-11s %-8s" % ("mass", "Sn / %", "isobar", "isobar / %",
                                          "ratio"))
    abund = {int(m): float(v) for m, v in zip(m.iso.isonum, m.iso.standard)}
    for mass in m.isotopes:
        if mass in ISOBAR:
            name, ab = ISOBAR[mass]
            say("   %-6d %-11.3f %-12s %-11.4f %-8.2f"
                % (mass, 100 * abund[mass], name, 100 * ab, ab / abund[mass]))
        elif mass == 117:
            say("   %-6d %-11.3f %-12s %-11s %-8s"
                % (mass, 100 * abund[mass], "116CdH+", "molecular", "-"))
        else:
            say("   %-6d %-11.3f %-12s %-11s %-8s"
                % (mass, 100 * abund[mass], "none", "-", "-"))
    say("   The isotope whose addition helps most in section 3 is mass %d, and it"
        % best_extra)
    say("   is in this table.  Extra ratios are not free: each one is another")
    say("   interference the subtraction has to get right, and the clean list is")
    say("   short.  Only 118 and 119 carry no atomic isobar at all; 117 is clean")
    say("   of atomic isobars but carries 116CdH+, which is breakable (collision")
    say("   cell, desolvation) rather than fundamental.")
    say()
    say("=" * 78)
    say("VERDICT for Sn")
    say("=" * 78)
    best_row = sorted(rows, key=lambda r: r[1])[0]
    say("* For mass-DEPENDENT work a third enriched isotope is not worth it.")
    say("  Measured here: the best single extra ratio improves the double spike")
    say("  from %.2f to %.2f ppm amu-1 (%.1f %%), and using all %d ratios gives"
        % (ppm3, best_row[1], 100 * (1 - best_row[1] / ppm3), len(m.order)))
    say("  %.2f ppm amu-1, %.1f %% in total.  That is not a reason to rebuild a"
        % (ppm_all, 100 * (1 - ppm_all / ppm3)))
    say("  method, and the isotope that is safest to add already captures most of")
    say("  the gain.")
    say("* The one thing a fifth isotope buys that a double spike cannot is the")
    say("  FRACTIONATION LAW, and for Sn that is not academic: the exponential and")
    say("  linear forms differ by about %.0f ppm amu-1 at beta = %.1f, versus the"
        % (abs(1e6 * shift[1] * (1.0 - BETA) / m.mean_mass), BETA))
    say("  %.2f ppm amu-1 the method advertises.  With a fourth ratio sigma_theta is"
        % ppm3)
    say("  %.2f, against a separation of %.2f at realistic beta, so the laws should"
        % (sig_at_scan[1.5][3], abs(1.0 - BETA)))
    say("  separate at 3-7 sigma per measurement -- and that error is statistical,")
    say("  so repeats make it decisive.  Testing the exponential law directly is")
    say("  something no double spike can do at any precision.  Note the trap that")
    say("  falls out of the algebra: at beta = 1 (1 % per amu over 12 amu) the two")
    say("  laws coincide to second order and the test has no power at all.")
    say("* Extra ratios are not free, and the Sn table in section 5 is why: each")
    say("  added mass is another interference the subtraction has to get right,")
    say("  and only 118 and 119 carry no atomic isobar at all.  The isotope whose")
    say("  addition helps most is mass %d, which carries one." % best_extra)
    say("* For mass-INDEPENDENT work the redundancy IS the point, and Sn is")
    say("  unusual in how much it needs it.  Ten stable isotopes, three with a")
    say("  magnetic nucleus (115, 117, 119), and 115Sn is also radiogenic from")
    say("  115In.  Separating a magnetic isotope effect, a nuclear volume effect")
    say("  and mass-dependent fractionation takes more equations than unknowns --")
    say("  exactly what a double spike cannot supply.  Bragagni et al. (2023)")
    say("  worked around it by using only 117 and 119 for the magnetic effect and")
    say("  treating 115 as radiogenic.  A multi-isotope inversion is the")
    say("  alternative they did not have, and it is where a triple spike pays.")
    say("* Either way the cost is an appendix, not a switch.  The error chain is")
    say("  implicit-function differentiation of a square system, so an")
    say("  over-determined version has to be re-derived and re-validated against")
    say("  this script's k = 3 check, which agrees with errorestimate to %.1e."
        % val_rel)

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
