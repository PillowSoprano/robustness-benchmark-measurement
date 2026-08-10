"""Shared figure style for every figure in the paper.

Two rules this module exists to enforce.

Draw at final print size. A figure created 17 in wide and scaled into a
7.16 in text block shrinks every glyph by 2.4x, so a nominal 9 pt tick label
prints at under 4 pt. Use COL and TEXT below as the figure width and set no
scaling in LaTeX, then the point sizes here are the point sizes on paper.

Lock the colour semantics. Persistence is grey in every figure because it is
the uninformed baseline; each learned model keeps the same colour and the same
marker everywhere, so a reader who learns the mapping once keeps it.

Palette is Paul Tol's bright qualitative scheme, which stays separable under
the common colour-vision deficiencies. Markers differ as well, so the figures
survive greyscale printing.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep SVG text editable
import matplotlib.pyplot as plt

# Final print widths, inches. IEEEtran letterpaper two-column.
COL = 3.40      # one column
TEXT = 7.16     # full text width (figure*)

# Paul Tol bright + a semantic grey for the baseline.
GREY = "#666666"
BLUE = "#4477AA"
RED = "#EE6677"
GREEN = "#228833"

PURPLE = "#AA3377"
TEAL = "#009988"

MODEL = {
    "persistence": dict(color=GREY, marker=None, lw=1.6, zorder=2),
    "ridge":       dict(color=BLUE, marker="o", lw=1.6, zorder=3),
    "mlp":         dict(color=RED, marker="s", lw=1.6, zorder=3),
    "gru":         dict(color=GREEN, marker="^", lw=1.6, zorder=3),
    "dlinear":     dict(color=PURPLE, marker="D", lw=1.6, zorder=3),
    "patchtf":     dict(color=TEAL, marker="v", lw=1.6, zorder=3),
}
LABEL = {"persistence": "persistence", "ridge": "ridge",
         "mlp": "MLP", "gru": "GRU",
         "dlinear": "DecompLinear", "patchtf": "PatchAttn"}

# Dead zone below the baseline, used by the retention figures.
DEAD = "#EE6677"


def apply():
    """Install the shared rcParams. Call once at the top of a plot script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.markersize": 2.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def baseline(ax, y=1.0, label=None, dead_below=True):
    """Draw the persistence reference and shade the region below it.

    Below the line a predictor is worse than holding the last observation,
    which is the one thing a reader should see without consulting a caption.
    """
    if dead_below:
        lo = ax.get_ylim()[0]
        ax.axhspan(lo, y, color=DEAD, alpha=0.055, lw=0, zorder=0)
    ax.axhline(y, color="black", lw=1.3, zorder=4)
    if label:
        # left edge: the right edge belongs to the end-of-curve labels
        ax.text(0.015, y, label, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=6.5, color="0.25",
                zorder=5)


def end_labels(ax, entries, pad=0.26, fontsize=7.5):
    """Label curves at their right-hand end instead of using a legend box.

    entries: list of (x_last, y_last, text, colour).
    Widens the x axis by `pad` of its range to make room, then places each
    label just past its curve, nudging apart any that would collide.
    """
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + pad * (x1 - x0))
    lo, hi = ax.get_ylim()
    span = hi - lo
    placed = []
    for x, y, text, colour in sorted(entries, key=lambda e: -e[1]):
        yy = y
        for py in placed:
            if abs(yy - py) < 0.085 * span:
                yy = py - 0.085 * span
        placed.append(yy)
        ax.annotate(text, xy=(x, yy), xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=fontsize, color=colour, va="center",
                    annotation_clip=False)


def family_crossing(results, family, seeds, domain):
    """Utility crossing for a model family, only when most seeds show one.

    A crossing carried by one seed out of three is a seed artifact, which is
    the seventh measurement choice this paper is about. Reporting it as a
    family property in a figure would contradict the text.
    """
    names = [family] if family == "ridge" else [f"{family}_s{s}" for s in seeds]
    vals = [results[n][domain]["utility_crossing"] for n in names]
    have = [v for v in vals if v is not None]
    return sum(v is not None for v in vals) * 2 > len(vals) and \
        sum(have) / len(have) or None


def save(fig, out):
    """Write PDF for LaTeX, SVG for hand polish, PNG for quick viewing."""
    from pathlib import Path
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out.with_suffix("." + ext),
                    dpi=200 if ext == "png" else None)
    print("wrote", out.with_suffix(".pdf"), "+ .svg + .png")
