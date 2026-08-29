"""
Regenerate the scan summary JSON + per-mass table from existing
<outroot>/<window>/M<mass>/best/ results, without re-running any fit. Also
prints the mass points ranked by |significance| (local; look-elsewhere applies),
lists fits whose window was shrunk below the default, and writes four plots
into <outroot>:
  scan_significance.png  -- signed local significance + fitted N_sig
  scan_bkg_gof.png       -- background-only goodness of fit
  scan_window_sizes.png  -- adaptive fit-window size vs mass
  scan_func_forms.png    -- winning bkg function family + # parameters vs mass
The GoF plot uses the BACKGROUND-ONLY fit (this is a control region, so no
signal is expected and the bkg-only GoF is the relevant check).

Defaults reproduce the Results_vr_dimuon_combined_v1 scan; pass
--plan/--outroot/--summary/--label for another dataset.
"""
import argparse
import glob
import json
import os

import numpy as np

from Utils import get_nPars  # order -> # free shape parameters (per functional form)

NSIG_START = 7.0    # default half-width; anything smaller means the window was shrunk
PVAL_THRESH = 0.05  # S+B GoF threshold; below this the fit is "poor GoF" (kept at the floor)

# Background function families in the discrete-profiling envelope, drawn in a
# fixed order/colour so the func-form plot is consistent across datasets.
FAM_ORDER = ["exp", "expPoly", "polyExp", "bern"]
FAM_COLOR = {"exp": "#2ca02c", "expPoly": "#1f77b4", "polyExp": "#d62728", "bern": "#9467bd"}
NPAR_FLEX = 4       # n_par >= this => flexible form worth eyeballing (lots of curvature)

# Limit fields written by doFit.py that the summary carries through to the sigma*B
# conversion. All are in events at the r=1 normalization (sig_norm).
LIMIT_FIELDS = ("exp_lim_events", "exp_lim_1sig_low", "exp_lim_1sig_high",
                "exp_lim_2sig_low", "exp_lim_2sig_high", "sig_norm")


def best_form(r):
    """(family, n_par) of the winning background-only pdf, or (None, None).

    The chosen family is `bkgfit_best_pdf` (e.g. 'polyExp_shape'). `final_func_forms`
    stores the F-test-selected *index* into that family's tested-order list, so the
    fit's polynomial ORDER is func_forms[family][index]; the number of free shape
    parameters is then get_nPars(order, family) (order and #params differ, e.g.
    polyExp has order+1 params, exp has 2*order).
    """
    pdf = r.get("bkgfit_best_pdf")
    if not pdf:
        return None, None
    fam = pdf.replace("_shape", "")
    idx = (r.get("final_func_forms") or {}).get(fam)
    forms = r.get("func_forms") or (r.get("script_options") or {}).get("func_forms") or {}
    if idx is None or fam not in forms or idx >= len(forms[fam]):
        return fam, None
    order = forms[fam][idx]
    return fam, get_nPars(order, fam)


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="scan_plan.json", help="scan-plan JSON")
    ap.add_argument("--outroot", default="data_fits", help="fit-results root (plots written here)")
    ap.add_argument("--summary", default="data_scan_summary.json", help="summary JSON path")
    ap.add_argument("-s", "--sigdir", default="signal_fits/2B_loosemass", help="signal template dir")
    ap.add_argument("--label", default="same-sign VR dimuon", help="dataset label for plot titles")
    return ap.parse_args()


def main():
    args = get_args()
    plan = json.load(open(args.plan))

    def collect(tag, mass):
        rj = os.path.join(args.outroot, tag, "M%s" % mass, "best", "fit_results_%s.json" % mass)
        return json.load(open(rj)) if os.path.exists(rj) else None

    # signal sigma(mass) from the 2B interpolation nodes, to recover the winning
    # half-width as n_sigma = (mass - m_min) / sigma (no data-boundary flooring
    # in this scan, so this is exact up to the 0.5-sigma adaptive step).
    sigmap = {}
    for f in glob.glob(os.path.join(args.sigdir, "case_interpolation_M*.json")):
        sigmap[float(f.split("_M")[-1][:-5])] = json.load(open(f))["sigma"]

    def winning_nsigma(mass, m_min):
        if m_min is None or mass not in sigmap:
            return None
        return round((mass - m_min) / sigmap[mass] / 0.5) * 0.5

    summary = {}
    rows = []
    npass = ntot = 0
    for tag, info in plan.items():
        for m in info["masses"]:
            ntot += 1
            r = collect(tag, m)
            row = summary.setdefault(tag, {})
            if r is None:
                row[str(m)] = {"status": "fail"}
                rows.append((tag, m, None))
                continue
            npass += 1
            ndof = r.get("bkgfit_ndof") or 0
            bsz = (r.get("script_options") or {}).get("bin_size")
            binfrac = round(bsz / sigmap[float(m)], 2) if (bsz and float(m) in sigmap) else None
            fam, npar = best_form(r)
            rec = {"status": "ok", "center": info["center"],
                   "best_pdf": fam, "n_par": npar,
                   "n_sig": r.get("obs_excess_events", 0.0),
                   "n_sig_unc": r.get("obs_excess_events_unc", 0.0),
                   "signif": r.get("signif", float("nan")),
                   "sbfit_prob": r.get("sbfit_prob", -1),
                   "bkgfit_prob": r.get("bkgfit_prob", -1),
                   "bkgfit_chi2ndof": (r.get("bkgfit_chi2", float("nan")) / ndof) if ndof else float("nan"),
                   "sbfit_chi2ndof": (r.get("sbfit_chi2", float("nan")) / r.get("sbfit_ndof"))
                                     if r.get("sbfit_ndof") else float("nan"),
                   "obs_lim_events": r.get("obs_lim_events", float("nan")),
                   "n_sigma": winning_nsigma(float(m), r.get("m_min")),
                   "bin_frac": binfrac,
                   "gof_ok": r.get("sbfit_prob", -1) >= PVAL_THRESH,
                   "m_min": r.get("m_min"), "m_max": r.get("m_max")}
            # Expected limit + bands and the normalization convention, needed to turn
            # the event limit into a sigma*B limit. doFit.py leaves these at -1 when
            # AsymptoticLimits fails, so keep -1 rather than papering over it.
            rec.update({k: r.get(k, -1.0) for k in LIMIT_FIELDS})
            row[str(m)] = rec
            rows.append((tag, m, rec))

    sdir = os.path.dirname(args.summary)
    if sdir:
        os.makedirs(sdir, exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)

    print("=" * 92)
    print("%-9s %-6s %-6s %-17s %-7s %-8s %-9s %s" %
          ("window", "mass", "stat", "N_sig +- unc", "signif", "sbProb", "obs_lim", "window[GeV]"))
    print("-" * 92)
    for tag, m, rec in rows:
        if rec is None:
            print("%-9s %-6s %-6s" % (tag, m, "no fit"))
            continue
        stat = "OK" if rec["gof_ok"] else "POOR"  # POOR = kept at floor, S+B GoF < thresh
        print("%-9s %-6s %-6s %7.1f +- %-6.1f %-7.2f %-8.3f %-9.1f [%.2f,%.2f]" %
              (tag, m, stat, rec["n_sig"], rec["n_sig_unc"], rec["signif"],
               rec["sbfit_prob"], rec["obs_lim_events"], rec["m_min"], rec["m_max"]))
    print("-" * 92)
    npoor = sum(1 for _, _, r in rows if r is not None and not r["gof_ok"])
    print("fits with a usable best/: %d / %d   (of which %d have poor S+B GoF, p<%.2f, kept at the floor)"
          % (npass, ntot, npoor, PVAL_THRESH))

    print("\nTop 10 by |local significance| (look-elsewhere NOT applied):")
    ok = [(t, m, r) for (t, m, r) in rows if r is not None]
    ok.sort(key=lambda x: -abs(x[2]["signif"]))
    for t, m, r in ok[:10]:
        sign = "excess" if r["n_sig"] >= 0 else "deficit"
        print("  M%-6s %-9s signif=%.2f  (%s, N_sig=%.1f +- %.1f)"
              % (m, t, r["signif"], sign, r["n_sig"], r["n_sig_unc"]))

    # --- fits that needed the window shrunk below the +/-7 sigma default ---
    shrunk = [(t, m, r) for (t, m, r) in rows
              if r is not None and r["n_sigma"] is not None and r["n_sigma"] < NSIG_START - 1e-6]
    shrunk.sort(key=lambda x: x[2]["n_sigma"])
    print("\nFits that required shrinking the window below the +-%.0f sigma default: %d / %d"
          % (NSIG_START, len(shrunk), npass))
    if shrunk:
        print("  %-7s %-9s %-8s %-9s %s" % ("mass", "window", "n_sigma", "bkgProb", "window[GeV]"))
        for t, m, r in shrunk:
            print("  M%-6s %-9s +-%-5.1f  %-9.3f [%.2f,%.2f]"
                  % (m, t, r["n_sigma"], r["bkgfit_prob"], r["m_min"], r["m_max"]))

    # --- fits that needed coarser bins than the 0.5 sigma default ---
    coarse = [(t, m, r) for (t, m, r) in rows
              if r is not None and r["bin_frac"] is not None and r["bin_frac"] > 0.5 + 1e-6]
    coarse.sort(key=lambda x: -x[2]["bin_frac"])
    print("\nFits that needed coarser than 0.5 sigma bins (bin-size escalation): %d" % len(coarse))
    if coarse:
        print("  %-7s %-9s %-8s %-9s %s" % ("mass", "window", "bin_frac", "sbProb", "window[GeV]"))
        for t, m, r in coarse:
            print("  M%-6s %-9s %-8.1f %-9.3f [%.2f,%.2f]"
                  % (m, t, r["bin_frac"], r["sbfit_prob"], r["m_min"], r["m_max"]))

    # --- fits kept at the floor with poor S+B GoF (p < threshold) ---
    poor = [(t, m, r) for (t, m, r) in rows if r is not None and not r["gof_ok"]]
    poor.sort(key=lambda x: x[2]["sbfit_prob"])
    print("\nFits kept at the floor with poor S+B GoF (p < %.2f): %d"
          % (PVAL_THRESH, len(poor)))
    if poor:
        print("  %-7s %-9s %-9s %-9s %s" % ("mass", "window", "sbProb", "signif", "N_sig +- unc"))
        for t, m, r in poor:
            print("  M%-6s %-9s %-9.3f %-9.2f %.1f +- %.1f"
                  % (m, t, r["sbfit_prob"], r["signif"], r["n_sig"], r["n_sig_unc"]))

    # --- winning background function form + number of parameters ---
    from collections import Counter
    formed = [(t, m, r) for (t, m, r) in rows if r is not None and r.get("best_pdf")]
    fam_ct = Counter(r["best_pdf"] for _, _, r in formed)
    npar_ct = Counter(r["n_par"] for _, _, r in formed)
    print("\nWinning background function form (bkg-only pdf) over %d fits:" % len(formed))
    print("  by family : " + ", ".join("%s=%d" % (f, fam_ct.get(f, 0)) for f in FAM_ORDER
                                        if fam_ct.get(f)))
    print("  by n_par  : " + ", ".join("%dpar=%d" % (k, npar_ct[k])
                                        for k in sorted(x for x in npar_ct if x is not None)))
    flex = sorted((t, m, r) for (t, m, r) in formed if (r["n_par"] or 0) >= NPAR_FLEX)
    print("\nFlexible fits (>=%d bkg parameters -- most curvature, worth eyeballing): %d"
          % (NPAR_FLEX, len(flex)))
    if flex:
        print("  %-7s %-9s %-12s %-6s %-9s %s" %
              ("mass", "window", "form", "n_par", "bkgProb", "window[GeV]"))
        for t, m, r in flex:
            print("  M%-6s %-9s %-12s %-6d %-9.3f [%.2f,%.2f]"
                  % (m, t, r["best_pdf"], r["n_par"], r["bkgfit_prob"], r["m_min"], r["m_max"]))
    print("\nwrote %s" % args.summary)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    okrows = [(float(m), r) for (_, m, r) in rows if r is not None]
    okrows.sort()
    mass = np.array([m for m, _ in okrows])
    lab = args.label

    # 1) signed local significance + fitted N_sig vs mass
    sig = np.array([r["signif"] * (1 if r["n_sig"] >= 0 else -1) for _, r in okrows])
    nsig = np.array([r["n_sig"] for _, r in okrows])
    nunc = np.array([min(r["n_sig_unc"], 60) for _, r in okrows])  # clip pathological bars
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].axhline(0, color="k", lw=0.8)
    for y, c in [(2, "0.6"), (-2, "0.6"), (3, "0.8"), (-3, "0.8")]:
        ax[0].axhline(y, color=c, ls="--", lw=0.8)
    ax[0].plot(mass, sig, "o-", ms=4, color="#1f77b4", lw=0.8)
    # flag points kept at the floor with poor S+B GoF (fit not trustworthy)
    poor_mask = np.array([not r["gof_ok"] for _, r in okrows])
    if poor_mask.any():
        ax[0].plot(mass[poor_mask], sig[poor_mask], "X", ms=9, color="#d62728",
                   label="poor S+B GoF (p<%.2f, kept at floor)" % PVAL_THRESH)
        ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_ylabel("signed local significance [$\\sigma$]")
    ax[0].set_ylim(-3.5, 3.5)
    ax[0].set_title("%s scan  (2B signal, %d mass points)" % (lab, len(mass)))
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].errorbar(mass, nsig, yerr=nunc, fmt="o", ms=3, color="#333", elinewidth=0.8, capsize=2)
    ax[1].set_ylabel("fitted $N_\\mathrm{sig}$")
    ax[1].set_xlabel("dimuon mass [GeV]")
    ax[1].set_ylim(-70, 70)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outroot, "scan_significance.png"), dpi=110)
    plt.close(fig)

    # 2) goodness of fit: background-only (control region) overlaid with S+B (the fit
    # actually used for the result) -- a bkg-only dip below p=0.05 is only expected to be
    # benign if the S+B fit at the same point stays >= 0.05 (adaptive wrapper's real gate).
    pval = np.array([r["bkgfit_prob"] for _, r in okrows])
    sbval = np.array([r["sbfit_prob"] for _, r in okrows])
    chn = np.array([r["bkgfit_chi2ndof"] for _, r in okrows])
    sbchn = np.array([r["sbfit_chi2ndof"] for _, r in okrows])
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[2.4, 1])
    axp = fig.add_subplot(gs[0, :])
    axp.axhline(0.05, color="r", ls="--", lw=1, label="p = 0.05")
    axp.plot(mass, pval, "o-", ms=4, color="#22aa77", lw=0.8, label="background-only")
    axp.plot(mass, sbval, "s-", ms=3.5, color="#aa44cc", lw=0.8, alpha=0.85, label="S+B")
    # points where bkg-only dips below 0.05 but S+B recovers above it (counted below, not marked)
    recovered = (pval < 0.05) & (sbval >= 0.05)
    axp.set_ylabel("goodness-of-fit p-value")
    axp.set_ylim(-0.02, 1.02)
    axp.set_title("Goodness of fit (bkg-only vs S+B) -- %s scan (%d fits)" % (lab, len(mass)))
    axp.legend(loc="upper right", fontsize=8)
    axc = fig.add_subplot(gs[1, 0], sharex=axp)
    axc.axhline(1.0, color="0.5", ls="--", lw=1)
    axc.plot(mass, chn, "s-", ms=3, color="#22aa77", lw=0.8, label="bkg-only")
    axc.plot(mass, sbchn, "o-", ms=3, color="#aa44cc", lw=0.8, alpha=0.85, label="S+B")
    axc.set_ylabel(r"$\chi^2/\mathrm{ndof}$")
    axc.set_xlabel("dimuon mass [GeV]")
    axc.legend(loc="upper right", fontsize=7)
    axh = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 11)
    axh.hist(pval, bins=bins, color="#22aa77", edgecolor="k", alpha=0.55, label="bkg-only")
    axh.hist(sbval, bins=bins, color="#aa44cc", edgecolor="k", alpha=0.55, label="S+B")
    axh.axhline(len(mass) / 10.0, color="r", ls="--", lw=1)
    axh.set_xlabel("p-value")
    axh.set_ylabel("N fits")
    axh.set_title("p-value dist. (flat = good)", fontsize=9)
    axh.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outroot, "scan_bkg_gof.png"), dpi=110)
    plt.close(fig)

    n_bkg_bad = int(np.sum(pval < 0.05))
    n_recovered = int(np.sum(recovered))
    n_not_recovered = int(np.sum((pval < 0.05) & (sbval < 0.05)))
    print("bkg-only GoF: p<0.05 in %d/%d fits (mean p=%.3f, median chi2/ndof=%.2f); "
          "of those, %d recovered by S+B (p>=0.05), %d did NOT (sbfit_prob<0.05, real gate violation)"
          % (n_bkg_bad, len(mass), pval.mean(), np.median(chn), n_recovered, n_not_recovered))

    # 3) adaptive fit-window sizes vs mass
    nsg = np.array([r["n_sigma"] if r["n_sigma"] is not None else np.nan for _, r in okrows])
    width = np.array([r["m_max"] - r["m_min"] for _, r in okrows])
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].axhline(NSIG_START, color="0.5", ls="--", lw=1, label="default $\\pm%.0f\\sigma$" % NSIG_START)
    ax[0].axhline(4.0, color="r", ls="--", lw=1, label="min $\\pm4\\sigma$ (else fail)")
    ax[0].plot(mass, nsg, "o-", ms=4, color="#8844cc", lw=0.8)
    ax[0].set_ylabel("winning half-width [$\\sigma$]")
    ax[0].set_ylim(3.5, 7.5)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_title("Adaptive fit-window size -- %s scan (%d fits)" % (lab, len(mass)))
    ax[1].plot(mass, width, "s-", ms=3, color="#cc7722", lw=0.8)
    ax[1].set_ylabel("full window width [GeV]")
    ax[1].set_xlabel("dimuon mass [GeV]")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outroot, "scan_window_sizes.png"), dpi=110)
    plt.close(fig)

    # 4) winning background function form + number of parameters vs mass
    fams = [r.get("best_pdf") for _, r in okrows]
    npars = [r.get("n_par") for _, r in okrows]
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], width_ratios=[2.4, 1])
    axn = fig.add_subplot(gs[0, :])
    for fm in FAM_ORDER:
        xs = [m for (m, r) in okrows if r.get("best_pdf") == fm and r.get("n_par") is not None]
        ys = [r["n_par"] for (m, r) in okrows if r.get("best_pdf") == fm and r.get("n_par") is not None]
        if xs:
            axn.scatter(xs, ys, s=48, color=FAM_COLOR[fm], label=fm, zorder=3,
                        edgecolor="k", linewidth=0.4)
    axn.axhspan(NPAR_FLEX - 0.5, max([p for p in npars if p] + [NPAR_FLEX]) + 0.5,
                color="0.9", zorder=0, label="flexible ($\\geq%d$ par)" % NPAR_FLEX)
    axn.set_ylabel("# background shape parameters")
    axn.set_ylim(0.5, max([p for p in npars if p] + [4]) + 0.5)
    axn.set_yticks(range(1, max([p for p in npars if p] + [4]) + 1))
    axn.set_xlabel("dimuon mass [GeV]")
    axn.set_title("Winning background function form -- %s scan (%d fits)" % (lab, len(mass)))
    axn.legend(loc="upper left", fontsize=8, ncol=2)
    # bottom-left: stacked histogram of n_par, split by family
    axh = fig.add_subplot(gs[1, 0])
    pvals = sorted(set(p for p in npars if p is not None))
    bottom = np.zeros(len(pvals))
    for fm in FAM_ORDER:
        cnt = np.array([sum(1 for (m, r) in okrows
                            if r.get("best_pdf") == fm and r.get("n_par") == pv) for pv in pvals])
        if cnt.any():
            axh.bar(pvals, cnt, bottom=bottom, color=FAM_COLOR[fm], label=fm,
                    edgecolor="k", linewidth=0.3)
            bottom += cnt
    axh.set_xlabel("# background shape parameters")
    axh.set_ylabel("N fits")
    axh.set_xticks(pvals)
    # bottom-right: family usage counts
    axf = fig.add_subplot(gs[1, 1])
    used = [f for f in FAM_ORDER if any(r.get("best_pdf") == f for _, r in okrows)]
    cts = [sum(1 for _, r in okrows if r.get("best_pdf") == f) for f in used]
    axf.bar(range(len(used)), cts, color=[FAM_COLOR[f] for f in used], edgecolor="k", linewidth=0.3)
    axf.set_xticks(range(len(used)))
    axf.set_xticklabels(used, rotation=30, ha="right", fontsize=8)
    axf.set_ylabel("N fits")
    axf.set_title("family usage", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outroot, "scan_func_forms.png"), dpi=110)
    plt.close(fig)

    print("wrote %s, %s, %s and %s" % (os.path.join(args.outroot, "scan_significance.png"),
                                       os.path.join(args.outroot, "scan_bkg_gof.png"),
                                       os.path.join(args.outroot, "scan_window_sizes.png"),
                                       os.path.join(args.outroot, "scan_func_forms.png")))


if __name__ == "__main__":
    main()
