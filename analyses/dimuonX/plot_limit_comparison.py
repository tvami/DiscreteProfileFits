"""Overlay the median expected limits of several models on one axis.

Reads the .json files written by make_limit_plot.py -- every one of them must be the
same quantity (the "quantity" field is checked), otherwise the curves are not
comparable and the script refuses.

Usage:
    python3 plot_limit_comparison.py work_.../permodel/limits_*.json -o work_.../comparison
"""
import argparse
import json
import os

import numpy as np

from make_limit_plot import MODEL_LABELS, SEAM, split_at_seam


def short_label(family):
    """One-line legend label -- MODEL_LABELS is two lines for the T' families."""
    return MODEL_LABELS.get(family, family).replace("\n", ", ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="+", help="limits_<family>.json from make_limit_plot")
    ap.add_argument("-o", "--outbase", required=True)
    ap.add_argument("--lumi", type=float, default=96.94)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    curves, quantities = [], set()
    for path in sorted(args.jsons):
        with open(path) as f:
            d = json.load(f)
        quantities.add(d["quantity"])
        curves.append((d.get("family") or os.path.basename(path),
                       np.array([p["mass"] for p in d["points"]]),
                       np.array([p["exp_lim_events"] for p in d["points"]])))
    if len(quantities) != 1:
        raise SystemExit("refusing to overlay different quantities: " + str(quantities))
    quantity = quantities.pop()

    # Order the legend by sensitivity so it reads top-to-bottom like the curves.
    curves.sort(key=lambda c: np.median(c[2]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (family, masses, med) in enumerate(curves):
        for j, sl in enumerate(split_at_seam(masses)):
            s = np.array(sl)
            ax.plot(masses[s], med[s], lw=1.9, color=cmap(i % 10),
                    label=short_label(family) if j == 0 else None)

    below = curves[0][1][curves[0][1] < SEAM[0]]
    above = curves[0][1][curves[0][1] > SEAM[1]]
    if len(below) and len(above):
        ax.axvspan(below.max(), above.min(), color="0.88", zorder=0)

    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if 0.1 <= v < 10 else ""))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel(r"$\sigma\,\mathcal{B}\,\epsilon_{\mathrm{anom}}f_{\mathrm{win}}$ [fb]"
                  if "eps_anom" in quantity else r"$\sigma\,\mathcal{B}$ [fb]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9,
              title="Median expected, 95% CL")

    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax.text(0.085, 1.02, "Work in Progress", transform=ax.transAxes, style="italic",
            fontsize=11)
    ax.text(1.0, 1.02, f"{args.lumi:.2f} fb$^{{-1}}$ (13.6 TeV)",
            transform=ax.transAxes, ha="right", fontsize=11)
    if args.label:
        ax.text(0.03, 0.04, args.label, transform=ax.transAxes, fontsize=9, color="0.3")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{args.outbase}.{ext}", dpi=130)
    print(f"wrote {args.outbase}.pdf/.png with {len(curves)} models")


if __name__ == "__main__":
    main()
