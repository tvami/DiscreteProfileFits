"""
Build the data-fit scan plan:
  - global mass grid at ~1 sigma fidelity (step = local signal sigma),
    snapped to the 0.2 GeV interpolation-template nodes
  - assign each mass to the window whose 5 GeV center is closest
    (each window owns its central +/-1.25 GeV -> contiguous, non-overlapping)
Prints a per-window summary and writes the scan-plan JSON.

Windows are discovered by globbing the collaborator's results directory; the
sideband-included spectrum (<subdir>/global_anomalous_masses.h5, key 'mass') is
fit directly (doFit's loader accepts key 'mass' or 'masses', so no local copy
step is needed).

Defaults reproduce the Results_vr_dimuon_combined_v1 scan; pass --base/--series
(and -o) to build a plan for a different dataset.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def win_center(tag):
    lo, hi = [int(x) / 10.0 for x in tag.split("_")]
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/eos/uscms/store/user/tvami/CATHODE/"
                    "Flow_Learning/Results_vr_dimuon_combined_v1",
                    help="dataset results directory")
    ap.add_argument("--series", default="nsplit_dimuon_combined_v1_datavr_stage4_20260710_190828",
                    help="window-directory series prefix (before the _<LO>_<HI> tag)")
    ap.add_argument("--subdir", default="stage3_classifier",
                    help="per-window subdirectory holding the spectrum")
    ap.add_argument("--specfile", default="global_anomalous_masses.h5",
                    help="sideband-included spectrum filename")
    ap.add_argument("--sigdir", default="signal_fits/2B_loosemass",
                    help="2B signal interpolation-node directory")
    ap.add_argument("-o", "--out", default="scan_plan.json",
                    help="output scan-plan JSON path")
    ap.add_argument("--exclude-band", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None, help="drop any mass hypothesis whose +/-NSIGMA fit "
                    "window overlaps this GeV band (e.g. a masked/empty region in the MC)")
    ap.add_argument("--m-hypothesis-max", type=float, default=None,
                    help="drop hypotheses above this mass (GeV). The grid otherwise "
                         "runs to the highest window center + 1.25 GeV, which for the "
                         "SR-MC scan reaches 78.7 GeV -- close enough to the 80 GeV "
                         "data ceiling that the nominal +/-7 sigma window is "
                         "truncated. 75.0 reproduces the 97-hypothesis scan in the AN")
    ap.add_argument("--exclude-nsigma", type=float, default=7.0,
                    help="fit-window half-width (in sigma) used for --exclude-band overlap "
                    "test; match the adaptive wrapper's nominal start (default 7)")
    args = ap.parse_args()

    def win_file(tag):
        return os.path.join(args.base, "%s_%s" % (args.series, tag), args.subdir, args.specfile)

    # discover windows (edges in GeV*10) that actually exist on disk
    pat = re.compile(re.escape(args.series) + r"_(\d+_\d+)$")
    win_tags = sorted(
        (m.group(1) for d in glob.glob(os.path.join(args.base, args.series + "_*"))
         for m in [pat.search(os.path.basename(d.rstrip("/")))] if m),
        key=win_center)
    if not win_tags:
        raise SystemExit("No window directories found under %s" % args.base)

    # sigma(m) from the 2B interpolation nodes
    nodes = {}
    for f in glob.glob(os.path.join(args.sigdir, "case_interpolation_M*.json")):
        nodes[float(f.split("_M")[-1][:-5])] = json.load(open(f))["sigma"]
    node_masses = np.array(sorted(nodes))

    def sigma(m):
        return nodes[node_masses[np.argmin(np.abs(node_masses - m))]]

    def snap_down(m):
        """Largest available template node <= m."""
        i = np.searchsorted(node_masses, m + 1e-9, side="right") - 1
        return float(node_masses[max(i, 0)])

    centers = [win_center(t) for t in win_tags]
    OWN = 1.25  # each window owns +/-1.25 GeV around its center
    m_lo = max(min(centers) - OWN, float(node_masses.min()))
    # cap at the highest 2B signal template node (no signal shape exists above it)
    m_hi = min(max(centers) + OWN, float(node_masses.max()))
    if args.m_hypothesis_max is not None:
        m_hi = min(m_hi, args.m_hypothesis_max)

    # Global grid, stepped by the local resolution.
    #
    # Each hypothesis must sit on an available template, so the ideal step of
    # sigma(m) has to be snapped onto the node grid. Snap DOWN, not to nearest:
    # snapping to the nearest node can place the next point up to half a node
    # spacing ABOVE m+sigma, which makes the realized gap exceed 1 sigma no
    # matter how fine the nodes are. Snapping down guarantees gap <= sigma.
    # Iterate on the realized (snapped) points rather than on an unsnapped
    # running sum, so the guarantee applies to the gaps that actually occur.
    grid = [snap_down(m_lo + 1e-9)]
    if grid[0] < m_lo - 1e-9:                       # never start below the range
        grid[0] = float(node_masses[np.searchsorted(node_masses, m_lo - 1e-9)])
    while True:
        cur = grid[-1]
        nxt = snap_down(cur + sigma(cur))
        if nxt <= cur + 1e-9:
            # sigma is smaller than the node spacing: cannot honour <= 1 sigma.
            # Advance one node and let the check below report the violation.
            j = np.searchsorted(node_masses, cur + 1e-9, side="right")
            if j >= len(node_masses):
                break
            nxt = float(node_masses[j])
        if nxt > m_hi + 1e-6:
            break
        grid.append(nxt)
    grid = sorted(set(round(x, 4) for x in grid))

    # Verify the requirement the grid exists to satisfy: no gap wider than the
    # local resolution, or a signal falling between two hypotheses is described
    # by neither. Loud rather than silent -- this is a sensitivity issue.
    gaps = np.diff(grid)
    ratios = np.array([g / sigma(m) for g, m in zip(gaps, grid[:-1])])
    if len(ratios) and ratios.max() > 1.0 + 1e-6:
        bad = [(grid[i], grid[i + 1], ratios[i]) for i in np.where(ratios > 1.0 + 1e-6)[0]]
        print("\n*** WARNING: %d gap(s) exceed 1 sigma -- signal templates are too "
              "coarsely spaced here:" % len(bad))
        for lo, hi, r in bad[:10]:
            print("      %.1f -> %.1f GeV : %.2f sigma" % (lo, hi, r))
        print("    Generate interpolation nodes on a finer grid to fix.\n")

    # optionally drop hypotheses whose fit window overlaps a masked/empty band
    excluded = []
    if args.exclude_band is not None:
        blo, bhi = sorted(args.exclude_band)
        k = args.exclude_nsigma
        keep = []
        for m in grid:
            s = sigma(m)
            if (m + k * s < blo) or (m - k * s > bhi):
                keep.append(m)
            else:
                excluded.append(m)
        grid = keep
        print("excluding band [%.2f, %.2f] GeV (+/-%.1f sigma window): dropped %d "
              "hypotheses -> %s" % (blo, bhi, k, len(excluded),
                                    " ".join("%.1f" % x for x in excluded)))

    # assign each mass to nearest center (ties -> lower center)
    plan = {t: [] for t in win_tags}
    for mass in grid:
        ci = int(np.argmin([abs(mass - c) + 1e-6 * c for c in centers]))
        plan[win_tags[ci]].append(mass)

    # ---- report ----
    print("=" * 70)
    print("DATA-FIT SCAN PLAN  (2B signal, closest-center assignment)")
    print("base: %s" % args.base)
    print("=" * 70)
    print("%-9s %-8s %-9s %-7s  %s" % ("window", "center", "owns", "Nfit", "masses (GeV)"))
    print("-" * 70)
    total = 0
    out = {}
    for t in win_tags:
        c = win_center(t)
        ms = plan[t]
        total += len(ms)
        out[t] = {"center": c, "file": win_file(t), "masses": ms}
        owns = "[%.2f,%.2f]" % (c - OWN, c + OWN)
        mstr = " ".join("%.1f" % x for x in ms)
        print("%-9s %-8.1f %-9s %-7d  %s" % (t, c, owns, len(ms), mstr))
    print("-" * 70)
    print("total mass points: %d   range [%.1f, %.1f] GeV" % (total, grid[0], grid[-1]))
    print("signal templates cover up to %.1f GeV; windows centered above that own no masses"
          % float(node_masses.max()))
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
