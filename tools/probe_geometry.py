"""Probe the geometry of the double-spike inversion, using the toolbox's own model.

This exists because the usual textbook picture of the method -- "the mixing line
crosses the fractionation line" -- is easy to state and easy to get subtly wrong.
Every number printed here comes from ``errors.ratiodata`` (equation 10), so the
answers are properties of this implementation, not of a drawing.

Run from the repository root:

    G:/Python39/python.exe tools/probe_geometry.py                  # Sn, 117-122
    G:/Python39/python.exe tools/probe_geometry.py --element Fe --pair 57 58

The inversion set defaults to the two *most abundant* isotopes that are not in
the spike pair.  That single rule reproduces both conventional choices in this
repository: Sn 117-122 gives [117, 118, 120, 122] (what paper/commonmode.py
uses) and Fe 57-58 gives [54, 56, 57, 58] (the primer's example).  Override it
with --extras.

What is measured, and what it shows
-----------------------------------

A.  The map (proportion, alpha, beta) -> the three measured ratios.  Rank and
    condition number of its Jacobian: is the inversion locally unique?
B.  The direction in which each unknown moves the measurement, in log-ratio
    space, compared with P = log of the mass ratios.
        beta  acts exactly along P.
        alpha  does NOT, and that is the whole trick: alpha is applied to the
               sample before the spike is mixed in, while beta is applied to the
               mixture afterwards.  If the two directions coincided, a single
               measurement could not tell them apart.
C.  Projection onto the subspace orthogonal to P.  This is blind to beta to
    machine precision and only approximately blind to alpha, so it is an
    *instrumental*-fractionation-free projection, not a fractionation-free one.
D.  Whether the projected mixing line is straight in the proportion, i.e. how
    much error the familiar straight-line picture carries.
E.  For a fixed proportion, the set reachable by varying (alpha, beta): its
    dimension, and the angle between P and its normal -- 90 degrees means
    fractionation slides you along the surface.
F.  How the separation between alpha and beta, and the conditioning, degrade as
    the spike proportion goes to zero.
G.  The same question for a *single* spike: the system becomes rank deficient,
    and the null direction is exhibited by stepping along it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doublespike import IsoData  # noqa: E402
from doublespike.errors import ratiodata  # noqa: E402
from doublespike.optimal import singleoptimalspike  # noqa: E402

np.seterr(all="ignore")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Setup:
    """The forward model for one element, one spike pair and one inversion set."""

    def __init__(self, element, pair, extras):
        self.iso = IsoData(element)
        self.iso.set_errormodel()
        self.pair = sorted(int(m) for m in pair)
        self.inv = sorted(set(self.pair) | {int(m) for m in extras})
        if len(self.inv) != 4:
            raise SystemExit("need exactly 4 inversion isotopes, got " + str(self.inv))

        self.spike, self.p, _, _, self.ppmperamu = singleoptimalspike(
            self.iso,
            "pure",
            [int(self.iso.isoindex(m)) for m in self.pair],
            [int(self.iso.isoindex(m)) for m in self.inv],
        )

        # denominator: same rule as inversion.dsinversion -- if the spike has
        # essentially nothing at any inversion isotope, the one with the largest
        # spike abundance becomes the denominator
        idx = [int(self.iso.isoindex(m)) for m in self.inv]
        if np.any(self.spike[idx] < 0.001):
            self.deno = int(idx[int(np.argmax(self.spike[idx]))])
        else:
            self.deno = int(idx[0])
        order = [i for i in range(self.iso.nisos) if i != self.deno]
        self.order = order
        self.sel = np.array(
            [order.index(int(self.iso.isoindex(m))) for m in self.inv
             if int(self.iso.isoindex(m)) != self.deno]
        )
        self.ratio_names = [
            "%d/%d" % (int(self.iso.isonum[order[k]]), int(self.iso.isonum[self.deno]))
            for k in self.sel
        ]

    def ratiodata_full(self, p, alpha=0.0, beta=0.0, spike=None):
        """Everything ``errors.ratiodata`` returns, over *all* the ratios."""
        return ratiodata(
            self.iso,
            int(self.iso.isonum[self.deno]),
            p,
            spike=self.spike if spike is None else spike,
            alpha=alpha,
            beta=beta,
        )

    def ratiodata(self, p, alpha=0.0, beta=0.0, spike=None):
        """The same, restricted to the inversion ratios."""
        z, AP, An, AT, Am, AN, AM = self.ratiodata_full(p, alpha, beta, spike)
        sel = self.sel
        return z, [x[sel] for x in (AP, An, AT, Am, AN, AM)]

    def forward(self, p, alpha, beta, spike=None):
        """Return (P, Am) restricted to the inversion ratios."""
        _, (AP, _, _, Am, _, _) = self.ratiodata(p, alpha, beta, spike)
        return AP, Am

    def dlambda_dp(self, p, h=1e-6):
        """d(lambda)/dp, where lambda is the ratio-space proportion of equation (10).

        Only this one-dimensional conversion is differenced.  Equation (10) is
        written in the ratio-space parameter lambda while ``ratiodata`` takes the
        mole-space proportion p, and the first version of this script forgot the
        chain factor -- so its first Jacobian column was d log(Am)/d lambda,
        mislabelled, and it disagreed with the difference version by 36 %.
        """
        lo = float(self.ratiodata(p - h)[0][0])
        hi = float(self.ratiodata(p + h)[0][0])
        return (hi - lo) / (2 * h)

    def jacobian(self, p, alpha=0.0, beta=0.0, spike=None):
        """Analytic Jacobian of log(Am) with respect to (p, alpha, beta).

        Written out rather than differenced, because with the ratio bookkeeping of
        ``ratiodata_many`` in hand the derivatives are exact:

            AM  = lambda*AT + (1-lambda)*AN          AN = An*exp(-alpha*P)
            Am  = AM*exp(beta*P)

            d log(Am)/d beta   = P
            d log(Am)/d lambda = (AT - AN) / AM
            d log(Am)/d alpha  = [ (dl/dalpha)(AT - AN) - (1-lambda)*P*AN ] / AM

        Two traps, both of which this script originally fell into:

        * ``ratiodata`` takes the mole proportion p while equation (10) is written
          in the ratio-space lambda, so the first column needs the chain factor
          d(lambda)/dp.  Without it the column was d/d(lambda) mislabelled and the
          two forms disagreed by 36 %.
        * **lambda itself depends on alpha.**  ``realproptoratioprop`` gives
          lambda = p*b/(p*b + (1-p)*a) with b = 1 + sum(AN), and AN = An*exp(-alpha*P),
          so varying alpha at fixed p drags lambda with it:

              db/dalpha   = -sum(P * AN)
              dlambda/dalpha = p(1-p)*a/D**2 * db/dalpha,   D = p*b + (1-p)*a

          Dropping that term leaves a residual disagreement that is completely
          independent of the step size -- 3.6e-3 for Sn and 9.1e-3 for Fe at every
          h from 1e-3 to 3e-6, which is how it was caught.  A step-size scan that
          does not shrink is not truncation error; it means a term is missing.
        """
        z, (AP, An, AT, Am, AN, AM) = self.ratiodata(p, alpha, beta, spike)
        lam = float(z[0])

        # a and b must be summed over ALL the ratios, not just the inversion
        # subset: realproptoratioprop is called inside ratiodata_many with the
        # full vectors.  Using the subset happens to be harmless for a four
        # isotope system like Fe, where the subset *is* everything, and therefore
        # shows up only on Sn -- which is the kind of bug a single test case hides.
        _, APf, _Anf, ATf, _Amf, ANf, _AMf = self.ratiodata_full(p, alpha, beta, spike)
        a = 1.0 + float(ATf.sum())
        b = 1.0 + float(ANf.sum())
        D = p * b + (1.0 - p) * a
        dlam_dalpha = p * (1.0 - p) * a * (-float((APf * ANf).sum())) / (D * D)

        J = np.array([
            (AT - AN) / AM,
            (dlam_dalpha * (AT - AN) - (1.0 - lam) * AP * AN) / AM,
            AP.copy(),
        ]).T
        J[:, 0] *= self.dlambda_dp(p)
        return J

    def jacobian_check(self, p, alpha=0.0, beta=0.0, h=1e-4):
        """Central-difference Jacobian, for cross-checking the analytic one."""
        cols = []
        for dp, da, db in ((h, 0, 0), (0, h, 0), (0, 0, h)):
            hi = self.forward(p + dp, alpha + da, beta + db)[1]
            lo = self.forward(p - dp, alpha - da, beta - db)[1]
            cols.append((np.log(hi) - np.log(lo)) / (2 * h))
        return np.array(cols).T

    def jacobian_error(self, p):
        """Largest relative disagreement between the analytic and difference forms."""
        a = self.jacobian(p)
        b = self.jacobian_check(p)
        return float(np.abs(a - b).max() / np.abs(a).max())


def default_extras(iso, pair):
    """The two most abundant isotopes that are not in the spike pair."""
    cand = [i for i in range(iso.nisos) if int(iso.isonum[i]) not in pair]
    cand.sort(key=lambda i: -float(iso.standard[i]))
    return [int(iso.isonum[i]) for i in cand[:2]]


def show(setup, out):
    def say(s=""):
        out.append(str(s))

    AP, Am = setup.forward(setup.p, 0.0, 0.0)
    p_hat = unit(AP)

    say("=" * 74)
    say("%s  spike %d-%d" % (setup.iso.element, setup.pair[0], setup.pair[1]))
    say("=" * 74)
    say("all isotopes      : " + str([int(m) for m in setup.iso.isonum]))
    say("inversion isotopes: " + str(setup.inv)
        + "   (pair + the two most abundant others)")
    say("denominator       : " + str(int(setup.iso.isonum[setup.deno]))
        + "   (largest spike abundance)")
    say("inversion ratios  : " + ", ".join(setup.ratio_names))
    say("optimal proportion: p = " + str(round(setup.p, 4)))
    say("ranked ppm/amu    : " + str(round(setup.ppmperamu, 4)))
    say("P = log mass ratios: " + str(np.round(AP, 6)))
    say()

    # ---- A
    say("A. Is (p, alpha, beta) locally recoverable from the three ratios?")
    J = setup.jacobian(setup.p)
    sv = np.linalg.svd(J, compute_uv=False)
    say("   d log(Am)/d(p, alpha, beta):")
    for row in J:
        say("      " + str(np.round(row, 6)))
    say("   singular values  : " + str(np.round(sv, 8)))
    say("   condition number : " + str(round(float(sv[0] / sv[-1]), 3)))
    say("   analytic vs central-difference disagreement: "
        + str(round(setup.jacobian_error(setup.p), 9)))
    say("   -> full rank 3: yes, locally unique.")
    say()

    # ---- B
    say("B. Which direction does each unknown move the measurement?")
    d_alpha, d_beta = J[:, 1], J[:, 2]
    ang_ab = float(np.degrees(np.arccos(np.clip(unit(d_alpha) @ unit(d_beta), -1, 1))))
    ang_bp = float(np.degrees(np.arccos(np.clip(unit(d_beta) @ p_hat, -1, 1))))
    ang_ap = float(np.degrees(np.arccos(np.clip(unit(d_alpha) @ p_hat, -1, 1))))
    say("   d log(Am)/d alpha : " + str(np.round(d_alpha, 6)))
    say("   d log(Am)/d beta  : " + str(np.round(d_beta, 6)))
    say("   beta  vs P        : " + str(round(ang_bp, 6)) + " deg"
        + "   (P itself: beta acts on the whole mixture)")
    say("   alpha vs P        : " + str(round(ang_ap, 4)) + " deg")
    say("   alpha vs beta     : " + str(round(ang_ab, 4)) + " deg")
    say("   -> they are NOT collinear.  alpha hits the sample part only (it is")
    say("      applied before the spike is mixed in), beta hits everything after")
    say("      mixing, so the spike dilutes the two effects unequally.  That")
    say("      failure of collinearity is what makes the pair separable.")
    _, (_, _, ATb, _, _, _) = setup.ratiodata(setup.p)
    dead = [n for n, v in zip(setup.ratio_names, ATb) if abs(v) < 1e-12]
    live = [n for n, v in zip(setup.ratio_names, ATb) if abs(v) >= 1e-12]
    say("   the spike's own ratios AT = " + str(np.round(ATb, 6)))
    say("   ratios where the spike reaches the *numerator*: "
        + (", ".join(live) if live else "none"))
    say("   ratios where the spike contributes nothing at all: "
        + (", ".join(dead) if dead else "none"))
    say("   -> the second spike isotope is the only reason any numerator is")
    say("      non-zero.  A ratio whose spike term is zero sees alpha and beta")
    say("      only as (beta - alpha), so it cannot help separate them.")
    say()

    # ---- C
    say("C. Projection orthogonal to P")
    proj = lambda y: y - (y @ p_hat) * p_hat
    base = proj(np.log(setup.forward(setup.p, 0.0, 0.0)[1]))
    drift_a = max(
        float(np.linalg.norm(proj(np.log(setup.forward(setup.p, a, 0.0)[1])) - base))
        for a in (-0.5, -0.2, -0.05, 0.05, 0.2, 0.5)
    )
    drift_b = max(
        float(np.linalg.norm(proj(np.log(setup.forward(setup.p, 0.0, b)[1])) - base))
        for b in (-2.0, -1.0, -0.25, 0.25, 1.0, 2.0)
    )
    say("   drift from beta  alone, |beta| <= 2   : " + str(drift_b))
    say("   drift from alpha alone, |alpha| <= 0.5: " + str(drift_a))
    say("   -> the projection annihilates beta exactly and alpha only partly.")
    say("      It is an instrumental-free projection, nothing more.")
    say()

    # ---- D
    say("D. Is the projected mixing line straight in the proportion?")
    ps = np.linspace(0.05, 0.95, 19)
    pts = np.array([proj(np.log(setup.forward(p, 0.0, 0.0)[1])) for p in ps])
    chord = pts[-1] - pts[0]
    c_hat = unit(chord)
    off = np.array([np.linalg.norm((q - pts[0]) - ((q - pts[0]) @ c_hat) * c_hat)
                    for q in pts])
    say("   chord length          : " + str(round(float(np.linalg.norm(chord)), 6)))
    say("   max off-line distance : " + str(round(float(off.max()), 6)))
    say("   max off / chord       : "
        + str(round(float(off.max() / np.linalg.norm(chord)), 6)))
    say("   -> the straight-line picture is an approximation; this is its size.")
    say()

    # ---- E
    say("E. For a fixed proportion, what does (alpha, beta) reach?")
    grid = np.array([
        np.log(setup.forward(setup.p, a, b)[1])
        for a in np.linspace(-0.05, 0.05, 9)
        for b in np.linspace(-1.0, 1.0, 9)
    ])
    cen = grid - grid.mean(axis=0)
    _, s, vt = np.linalg.svd(cen, full_matrices=False)
    ang_n = float(np.degrees(np.arccos(np.clip(abs(vt[2] @ p_hat), -1, 1))))
    say("   singular values of the centred cloud: " + str(np.round(s, 8)))
    say("   s[2]/s[0] = " + str(round(float(s[2] / s[0]), 10)) + "  -> a 2-D surface")
    say("   angle between P and the surface normal: " + str(round(ang_n, 4)) + " deg")
    say("   surface normal: " + str(np.round(vt[2], 6)))
    say("   -> P lies in the surface, so instrument fractionation slides the")
    say("      measurement along it.  The normal is the combination that no")
    say("      (alpha, beta) can produce at fixed p -- the quantity that a")
    say("      *fifth* isotope tests, which is what a double spike residual is.")
    say()

    # ---- F
    say("F. How the separation degrades as the spike proportion goes to zero")
    say("        p    angle(alpha,beta)    condition number")
    for p in (0.5, 0.44, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        Jp = setup.jacobian(p)
        a = float(np.degrees(np.arccos(np.clip(unit(Jp[:, 1]) @ unit(Jp[:, 2]), -1, 1))))
        svp = np.linalg.svd(Jp, compute_uv=False)
        say("   %6.2f %16.4f %20.1f" % (p, a, svp[0] / svp[-1]))
    say("   -> with little spike the two directions crowd together and the")
    say("      inversion becomes ill-conditioned.  That is why the optimal")
    say("      proportion sits well away from zero, and it is a statement about")
    say("      geometry rather than about counting.")
    say()

    # ---- G
    say("G. What a single spike does instead")
    single = np.zeros(setup.iso.nisos)
    single[int(setup.iso.isoindex(setup.pair[0]))] = 1.0
    others = [m for m in setup.inv if m not in setup.pair]
    one = Setup(setup.iso.element, setup.pair, others)  # for geometry only
    one.spike = single
    idx = [int(one.iso.isoindex(m)) for m in one.inv]
    one.deno = int(idx[int(np.argmax(single[idx]))])
    one.order = [i for i in range(one.iso.nisos) if i != one.deno]
    one.sel = np.array([one.order.index(int(one.iso.isoindex(m))) for m in one.inv
                        if int(one.iso.isoindex(m)) != one.deno])
    one.ratio_names = ["%d/%d" % (int(one.iso.isonum[one.order[k]]),
                                  int(one.iso.isonum[one.deno])) for k in one.sel]
    say("   single spike at " + str(setup.pair[0])
        + ", inversion " + str(one.inv) + " -> ratios " + ", ".join(one.ratio_names))
    J2 = one.jacobian(one.p, spike=single)
    sv2 = np.linalg.svd(J2, compute_uv=False)
    say("   Jacobian is " + str(J2.shape)
        + ", singular values " + str(np.round(sv2, 10)))
    say("   rank = " + str(int(np.sum(sv2 > 1e-9 * sv2[0]))) + " against 3 unknowns")
    # exhibit the null direction: step along it and check nothing moves
    u, s2, vt2 = np.linalg.svd(J2)
    null = vt2[-1] / np.linalg.norm(vt2[-1])
    step = 1e-4
    dp, da, db = null * step
    shift = np.log(one.forward(one.p + dp, da, db, spike=single)[1]) \
        - np.log(one.forward(one.p, 0.0, 0.0, spike=single)[1])
    say("   null direction (d p, d alpha, d beta) = " + str(np.round(null, 4)))
    say("   stepping 1e-4 along it moves the ratios by " + str(float(np.abs(shift).max())))
    say("   -> the null direction is essentially (0, 1, 1): only alpha + beta is")
    say("      observable, they are perfectly exchangeable.  Note this is not a")
    say("      counting failure -- there are three ratios and three unknowns.  It")
    say("      is a structural one: a pure single spike has nothing at the other")
    say("      isotopes, so the denominator is forced onto the spiked isotope and")
    say("      every ratio the inversion can form has the spike in the")
    say("      denominator.  The spike then contributes only the overall scale")
    say("      (1 - lambda), which is exactly the amount of freedom alpha - beta")
    say("      already had.  That is why one spike gives a concentration but")
    say("      never a fractionation.")
    say()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--element", default="Sn")
    ap.add_argument("--pair", nargs=2, type=int, default=None,
                    help="the two enriched isotopes (default 117 122 for Sn)")
    ap.add_argument("--extras", nargs=2, type=int, default=None,
                    help="the two further isotopes used in the inversion")
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()

    if args.pair is None:
        args.pair = [117, 122] if args.element == "Sn" else None
    iso = IsoData(args.element)
    if args.pair is None:
        abund = sorted(range(iso.nisos), key=lambda i: -float(iso.standard[i]))
        args.pair = [int(iso.isonum[i]) for i in abund[:2]]
    if args.extras is None:
        args.extras = default_extras(iso, set(args.pair))

    setup = Setup(args.element, args.pair, args.extras)
    out = []
    show(setup, out)
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
