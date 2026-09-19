"""Regenerate the figures used by docs/double-spike-primer.html.

The document is a single self contained HTML file, so the figures are produced as
SVG fragments and inlined by ``tools/build_primer.py``.  Everything here is drawn
from the package's own ``errorcurve`` / ``errorcurve2d`` / ``optimalspike``, so a
figure can never disagree with the code.

Each drawing function takes an optional destination:

* ``dest=None`` (what ``build_primer`` uses) renders into memory and returns the
  SVG as a string.  Nothing touches the disk, which matters here: an earlier
  version wrote the SVGs into a scratch directory next to the repository, and
  on Windows that directory sometimes survived the cleanup, where ``git add -A``
  would have picked it up.
* ``dest=<a directory>`` writes ``<name>.svg`` files there, which is what the
  command line below does.

Figure text is deliberately ASCII (element symbols, numbers): matplotlib has no
CJK font configured on this machine, and the prose around the figure is Chinese
anyway.

    G:/Python39/python.exe tools/dspike_figures.py [output_dir]
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doublespike import IsoData, errorcurve, errorcurve2d  # noqa: E402
from doublespike.optimal import singleoptimalspike  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: selectable, small
matplotlib.rcParams["font.size"] = 9

#: the same 10 V / 8 s / 1e11 ohm / 300 K model the catalogue uses
FE_PAIRS = [
    ("pure", (56, 58), "pure $^{56}$Fe-$^{58}$Fe", "#c0392b", "-"),
    ("pure", (57, 58), "pure $^{57}$Fe-$^{58}$Fe", "#e67e22", "-"),
    ("pure", (54, 58), "pure $^{54}$Fe-$^{58}$Fe", "#7f8c8d", "-"),
    ("real", (56, 58), "ORNL $^{56}$Fe*-$^{58}$Fe*", "#922b21", "--"),
    ("real", (57, 58), "ORNL $^{57}$Fe*-$^{58}$Fe*", "#b9770e", "--"),
    ("real", (54, 58), "ORNL $^{54}$Fe*-$^{58}$Fe*", "#566573", "--"),
]


def emit(fig, dest, name):
    """Save ``fig`` to ``dest``/name.svg, or return the SVG text if dest is None."""
    if dest is None:
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)
        return buf.getvalue()
    path = Path(dest) / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def fe_error_curves(dest=None):
    """Error on alpha against spike:sample proportion, for the Fe double spikes.

    Each curve is drawn at *that pair's own* optimal double spike composition,
    which is what makes the comparison fair.
    """
    iso = IsoData("Fe")
    iso.set_errormodel()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    summary = []
    for type_, (a, b), label, colour, style in FE_PAIRS:
        isospike = [iso.isoindex(a), iso.isoindex(b)]
        spike, p_opt, err_opt, _, ppm_opt = singleoptimalspike(iso, type_, isospike)
        errorcurve(iso, spike, isoinv=[54, 56, 57, 58], plottype="ppmperamu",
                   ax=ax, color=colour, linestyle=style, label=label)
        summary.append(
            f"{label}: q({a}Fe)={spike[iso.isoindex(a)]:.1%}, "
            f"spike:sample={p_opt:.1%}:{1 - p_opt:.1%}, {ppm_opt:.2f} ppm/amu at optimum"
        )
    # the 54Fe-58Fe curve shoots up very steeply on either side of its optimum;
    # clip the axis so that the shapes near the minima stay readable
    ax.set_ylim(0, 220)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fe: error on alpha vs spike:sample proportion (10 V, 8 s, 1e11 ohm)",
                 fontsize=9)
    fig.tight_layout()
    print("\n".join("    " + s for s in summary))
    return emit(fig, dest, "fig_fe_errorcurves")


def fe_error_surface(dest=None):
    """2D map of the error for the ORNL 57Fe*-58Fe* spike, over (p, q)."""
    iso = IsoData("Fe")
    iso.set_errormodel()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    errorcurve2d(
        iso,
        "real",
        isospike=[iso.isoindex(57), iso.isoindex(58)],
        isoinv=[54, 56, 57, 58],
        plottype="ppmperamu",
        resolution=160,
        threshold=0.5,
        ax=ax,
    )
    spike, p_opt, err_opt, _, ppm_opt = singleoptimalspike(
        iso, "real", [iso.isoindex(57), iso.isoindex(58)]
    )
    ax.plot(p_opt, spike[iso.isoindex(57)], "ks", ms=5,
            label=f"optimum = {ppm_opt:.2f} ppm/amu")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return emit(fig, dest, "fig_fe_errorsurface")


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fe_error_curves(out_dir)
    fe_error_surface(out_dir)


if __name__ == "__main__":
    main()
