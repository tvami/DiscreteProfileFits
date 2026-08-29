"""
Brazil-band upper-limit plot from a scan summary JSON.

Model-independent form (what this produces today):

    sigma_vis = sigma * B * A * eps  =  N_lim / L

i.e. a limit on the *selected* signal yield per unit luminosity, in fb. It needs no
acceptance x efficiency chain -- see section 5.4 of
Analysis/HANDOVER_20260730_upper_limits_sigmaB.md. Divide by A*eps(model, m) later to
get a model-specific sigma*B; pass --ae-json once that exists.

The expected limit and its bands come from the background-only Asimov dataset, so this
runs on a signal-free sample (SR background MC, same-sign VR) with no unblinding. The
observed curve on such a sample is a closure check, not a result, and is off by default.

Sanity checks, all from the handover's item E, are applied before plotting:
  * refuse any point whose limit fields are -1 (doFit.py's failure sentinel for
    AsymptoticLimits -- it would otherwise become a negative cross section);
  * require exp_2sig_low < exp_1sig_low < exp < exp_1sig_high < exp_2sig_high;
  * flag r_at_bound points -- a limit pinned at R_MAX is a bound, not a measurement.

Usage:
    python3 make_limit_plot.py --summary work_.../srmc_scan_summary.json \
        --label "SR background MC" -o work_.../limits
"""
import argparse
import json
import os

import numpy as np

# The 47.5-52.5 GeV DY mass-binned stitch seam is excluded from the scan. The curve
# must show a gap there rather than interpolate across it.
SEAM = (47.5, 52.5)

# CMS mplhep recommended band colours, matching the EarthAsDM limit plots
# (exp_lim/set_limit_alphaMax.py): gold for the inner 68%, blue for the outer 95%.
# These are the *central quantiles of the upper-limit distribution* under the
# background-only hypothesis -- NOT +-1/+-2 sigma, and the legend must not say sigma.
COLOR_68 = "#FFDF7F"
COLOR_95 = "#85D1FB"

# Family key (as in signal_loosemass_manifest.json / presel_efficiency_*.json) -> the
# decay chain the MC was generated with. The limit is on sigma times THAT chain.
MODEL_LABELS = {
    "TpTpTo2T2STo2Mu2B_MTp1000":
        r"$T'\bar{T}'\to tS\,\bar{t}S$, $S\to\mu\mu$, $S\to b\bar{b}$"
        "\n" r"$M_{T'} = 1000$ GeV",
    "TpTpTo2T2STo2Mu2G_MTp1000":
        r"$T'\bar{T}'\to tS\,\bar{t}S$, $S\to\mu\mu$, $S\to\gamma\gamma$"
        "\n" r"$M_{T'} = 1000$ GeV",
    "TpTpTo2T2STo2MuInv_MTp1000":
        r"$T'\bar{T}'\to tS\,\bar{t}S$, $S\to\mu\mu$, $S\to$ inv."
        "\n" r"$M_{T'} = 1000$ GeV",
    "H2toH1H3to2Mu_MH2-250":
        r"$H_2\to H_1H_3$, $H_3\to\mu\mu$, $M_{H_2} = 250$ GeV",
    "H2toH1H3to2Mu_MH2-500":
        r"$H_2\to H_1H_3$, $H_3\to\mu\mu$, $M_{H_2} = 500$ GeV",
    "H2toH1toInvH3to2Mu_MH2-250":
        r"$H_2\to H_1H_3$, $H_1\to$ inv., $H_3\to\mu\mu$, $M_{H_2} = 250$ GeV",
    "H2toH1toInvH3to2Mu_MH2-500":
        r"$H_2\to H_1H_3$, $H_1\to$ inv., $H_3\to\mu\mu$, $M_{H_2} = 500$ GeV",
    "TTH2to2Mu": r"$t\bar{t}H_2$, $H_2\to\mu\mu$",
    "VLLVLLToZHTo2MuInv_MVLL-250":
        r"VLL pair $\to ZH$, $H\to\mu\mu$, $M_{\mathrm{VLL}} = 250$ GeV",
    "VLLVLLToZHTo2MuInv_MVLL-500":
        r"VLL pair $\to ZH$, $H\to\mu\mu$, $M_{\mathrm{VLL}} = 500$ GeV",
}

LIMIT_KEYS = ("exp_lim_2sig_low", "exp_lim_1sig_low", "exp_lim_events",
              "exp_lim_1sig_high", "exp_lim_2sig_high")


def load_points(summary_path, want_observed):
    """Flatten the summary into a mass-ordered list of validated limit points."""
    with open(summary_path) as f:
        summary = json.load(f)

    pts, dropped, at_bound = [], [], []
    for window, masses in summary.items():
        for mass, rec in masses.items():
            if rec.get("status") != "ok":
                dropped.append((float(mass), "fit failed"))
                continue
            vals = [rec.get(k, -1.0) for k in LIMIT_KEYS]
            if any(v is None or v <= 0 for v in vals):
                dropped.append((float(mass), "AsymptoticLimits returned the -1 sentinel"))
                continue
            if not all(a < b for a, b in zip(vals, vals[1:])):
                dropped.append((float(mass), "limit quantiles out of order"))
                continue
            obs = rec.get("obs_lim_events", -1.0)
            if want_observed and (obs is None or obs <= 0):
                dropped.append((float(mass), "observed limit missing"))
                continue
            if rec.get("r_at_bound"):
                at_bound.append(float(mass))
            pts.append({"mass": float(mass), "window": window, "obs": obs,
                        **dict(zip(LIMIT_KEYS, vals))})

    pts.sort(key=lambda p: p["mass"])
    return pts, dropped, at_bound


def split_at_seam(masses):
    """Index slices either side of the excluded band, so nothing is drawn across it."""
    below = [i for i, m in enumerate(masses) if m < SEAM[0]]
    above = [i for i, m in enumerate(masses) if m > SEAM[1]]
    return [s for s in (below, above) if s]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="scan summary JSON")
    ap.add_argument("-o", "--outbase", required=True,
                    help="output path prefix (.pdf/.png/.json/.tex are appended)")
    ap.add_argument("--lumi", type=float, default=96.94,
                    help="integrated luminosity in /fb (AN/3Samples.tex:39)")
    ap.add_argument("--label", default="",
                    help="sample label drawn on the plot, e.g. 'SR background MC'")
    ap.add_argument("--draw-observed", action="store_true",
                    help="also draw the observed curve. On a signal-free sample this "
                         "is a closure check, not a result")
    ap.add_argument("--ae-json", default=None,
                    help="JSON of the FULL A*eps per mass to divide out, giving a "
                         "model-specific sigma*B. Needs eps_anom, which does not exist "
                         "yet (item D2 of the handover)")
    ap.add_argument("--presel-json", default=None,
                    help="Analysis/presel_efficiency_sr_v3.3.0.json. Divides out the "
                         "trigger+preselection efficiency only, leaving "
                         "sigma*B*eps_anom*f_win. Requires --family")
    ap.add_argument("--family", default=None,
                    help="family key inside --presel-json, e.g. TpTpTo2T2STo2Mu2G_MTp1000")
    ap.add_argument("--shape-note", default=None,
                    help="extra text after the signal-shape note in the on-plot caption")
    ap.add_argument("--table-step", type=float, default=5.0,
                    help="quote roughly one table row per this many GeV")
    args = ap.parse_args()

    if args.presel_json and not args.family:
        raise SystemExit("--presel-json requires --family")

    pts, dropped, at_bound = load_points(args.summary, args.draw_observed)
    if not pts:
        raise SystemExit("no usable limit points in " + args.summary)

    print(f"{len(pts)} usable mass points, {len(dropped)} dropped")
    for mass, why in dropped:
        print(f"  dropped m={mass}: {why}")
    if at_bound:
        print(f"  WARNING: r_at_bound at {len(at_bound)} point(s) -- these are bounds, "
              f"not measurements: {at_bound}")

    masses = np.array([p["mass"] for p in pts])
    scale = np.full_like(masses, 1.0 / args.lumi)  # events -> fb

    # What is actually on the y axis depends on how much of A*eps we can divide out.
    #   N_lim = sigma * B * L * eps_presel * eps_anom * f_win
    # so dividing by L alone leaves the fully model-independent visible cross section,
    # and dividing additionally by eps_presel leaves sigma*B*eps_anom*f_win.
    ylabel = r"$\sigma\,\mathcal{B}\,A\epsilon$ [fb]"
    quantity = "sigma*B*A*eps [fb]"
    if args.ae_json:
        with open(args.ae_json) as f:
            ae = json.load(f)
        scale = scale / np.array([ae[str(p["mass"])] for p in pts])
        ylabel = r"$\sigma\,\mathcal{B}$ [fb]"
        quantity = "sigma*B [fb]"
    elif args.presel_json:
        with open(args.presel_json) as f:
            table = json.load(f)["families"][args.family]
        # eps_presel is on a 5 GeV grid; the hypotheses are not. Interpolate, and do
        # not extrapolate -- np.interp clamps to the end values, which is what we want
        # for the 13.8 GeV hypothesis sitting just below the 15 GeV node.
        grid_m = np.array(sorted(float(m) for m in table))
        grid_e = np.array([table[str(int(m))]["eff_presel"] for m in grid_m])
        scale = scale / np.interp(masses, grid_m, grid_e)
        ylabel = r"$\sigma\,\mathcal{B}\,\epsilon_{\mathrm{anom}}f_{\mathrm{win}}$ [fb]"
        quantity = "sigma*B*eps_anom*f_win [fb] (trigger+preselection divided out)"

    curves = {k: np.array([p[k] for p in pts]) * scale for k in LIMIT_KEYS}
    obs = np.array([p["obs"] for p in pts]) * scale

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for sl in split_at_seam(masses):
        s = np.array(sl)
        first = s[0] == 0
        ax.fill_between(masses[s], curves["exp_lim_2sig_low"][s],
                        curves["exp_lim_2sig_high"][s],
                        color=COLOR_95, label="95% expected" if first else None)
        ax.fill_between(masses[s], curves["exp_lim_1sig_low"][s],
                        curves["exp_lim_1sig_high"][s],
                        color=COLOR_68, label="68% expected" if first else None)
        ax.plot(masses[s], curves["exp_lim_events"][s], "k--", lw=2.5,
                label="Median expected" if first else None)
        if args.draw_observed:
            ax.plot(masses[s], obs[s], "k-", lw=2, marker="o", ms=3.5,
                    label="Observed" if first else None)

    # Shade the whole hypothesis gap, not just the seam: --exclude-band drops every
    # hypothesis whose +/-7 sigma window *overlaps* [47.5, 52.5], so the real gap is
    # much wider than the seam and shading only the seam would misread as a plotting
    # artefact between the band edge and the first surviving point.
    below = masses[masses < SEAM[0]]
    above = masses[masses > SEAM[1]]
    if len(below) and len(above):
        gap_lo, gap_hi = below.max(), above.min()
        ax.axvspan(gap_lo, gap_hi, color="0.88", zorder=0)
        ax.text(0.5 * (gap_lo + gap_hi), 0.03,
                "DY stitch seam\nexcluded", fontsize=8, color="0.35",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())

    ax.set_yscale("log")
    # The band spans well under a decade, so the default log locator labels a single
    # tick. Label the minor ticks with plain numbers instead.
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if 0.1 <= v < 10 else ""))
    ax.tick_params(axis="y", which="minor", labelsize=8)

    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(masses.min() - 1, masses.max() + 1)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", framealpha=0.9, title="95% CL upper limits")

    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax.text(0.085, 1.02, "Work in Progress", transform=ax.transAxes, style="italic",
            fontsize=11)
    ax.text(1.0, 1.02, f"{args.lumi:.2f} fb$^{{-1}}$ (13.6 TeV)",
            transform=ax.transAxes, ha="right", fontsize=11)

    # Name the model on the plot. Which model this is "for" has two independent
    # answers, and both belong here: the signal SHAPE used in the fit (always the 2B
    # template -- see appendix_shape_mismatch.tex) and, when eps_presel is divided out,
    # the family whose efficiency was used.
    caption = [MODEL_LABELS.get(args.family, args.family)] if args.family else []
    caption.append("Fit template: 2B (medium width), used for all models"
                   + (f", {args.shape_note}" if args.shape_note else ""))
    if args.label:
        caption.append(args.label)
    ax.text(0.03, 0.055, "\n".join(c for c in caption if c),
            transform=ax.transAxes, fontsize=9.5, color="0.25", va="bottom")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{args.outbase}.{ext}", dpi=130)
    plt.close(fig)

    payload = {
        "summary": os.path.abspath(args.summary),
        "lumi_invfb": args.lumi,
        "quantity": quantity,
        "family": args.family,
        "seam_excluded_GeV": list(SEAM),
        "r_at_bound_masses": at_bound,
        "dropped": [{"mass": m, "reason": w} for m, w in dropped],
        "points": [{"mass": float(m),
                    **{k: float(curves[k][i]) for k in LIMIT_KEYS},
                    "obs": float(obs[i])} for i, m in enumerate(masses)],
    }
    with open(f"{args.outbase}.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Decimated LaTeX table -- 97 rows is far too many for the note.
    rows, last = [], -1e9
    for i, m in enumerate(masses):
        if m - last < args.table_step:
            continue
        last = m
        rows.append(f"    {m:.1f} & {curves['exp_lim_events'][i]:.2f} & "
                    f"{curves['exp_lim_1sig_low'][i]:.2f} & "
                    f"{curves['exp_lim_1sig_high'][i]:.2f} \\\\")
    with open(f"{args.outbase}.tex", "w") as f:
        f.write("\\begin{tabular}{cccc}\n  \\hline\n"
                "    $m_{\\mu\\mu}$ [GeV] & Expected & $-1\\sigma$ & $+1\\sigma$ \\\\\n"
                "  \\hline\n" + "\n".join(rows) +
                "\n  \\hline\n\\end{tabular}\n")

    best = int(np.argmin(curves["exp_lim_events"]))
    worst = int(np.argmax(curves["exp_lim_events"]))
    print(f"wrote {args.outbase}.pdf/.png/.json/.tex ({len(rows)} table rows)")
    print(f"  best  expected {curves['exp_lim_events'][best]:.3g} fb at "
          f"m = {masses[best]:.1f} GeV")
    print(f"  worst expected {curves['exp_lim_events'][worst]:.3g} fb at "
          f"m = {masses[worst]:.1f} GeV")


if __name__ == "__main__":
    main()
