"""
Driver: run the adaptive fit for every (window, mass) in a scan-plan JSON.

Uses the validated configuration (dimuonX_config func-forms, +/-7 sigma start,
0.5 sigma bins, adaptive shrink to 4 sigma on GoF failure), 2B signal shapes,
and each window's sideband-included spectrum read straight from EOS (the path in
the plan; doFit's loader accepts the key 'mass' directly).

Runs concurrent adaptive fits. Results under <outroot>/<window>/M<mass>/.
Writes a summary JSON + prints a per-mass table. Defaults reproduce the
Results_vr_dimuon_combined_v1 scan; pass --plan/--outroot/--summary for another.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# run_fit_adaptive.py (and doFit.py, which it calls) live in the general
# framework, one level up from this analysis directory.
FRAMEWORK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Limit fields written by doFit.py that the summary carries through to the sigma*B
# conversion. All are in events at the r=1 normalization (sig_norm).
LIMIT_FIELDS = ("exp_lim_events", "exp_lim_1sig_low", "exp_lim_1sig_high",
                "exp_lim_2sig_low", "exp_lim_2sig_high", "sig_norm")


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="scan_plan.json", help="scan-plan JSON")
    ap.add_argument("--outroot", default="data_fits", help="output root for fit dirs")
    ap.add_argument("--summary", default="data_scan_summary.json", help="summary JSON path")
    ap.add_argument("-c", "--config", default="dimuonX_config.json", help="fit config JSON")
    ap.add_argument("-s", "--sigdir", default="signal_fits/2B_loosemass", help="signal template dir")
    ap.add_argument("-j", "--workers", type=int,
                    default=int(os.environ.get("DS_WORKERS", "3")), help="concurrent fits")
    ap.add_argument("--n-sigma-start", type=float, default=7.0)
    ap.add_argument("--n-sigma-min", type=float, default=4.0)
    ap.add_argument("--bin-frac", type=float, default=0.5)
    ap.add_argument("--m-data-min", type=float, default=None,
                    help="floor the fit window at this mass (default: none)")
    ap.add_argument("--m-data-max", type=float, default=80.0,
                    help="cap the fit window at this mass to avoid the data edge (default 80)")
    return ap.parse_args()


def run_one(job, args):
    tag, mass = job
    outdir = os.path.join(args.outroot, tag, "M%s" % mass)
    plan = run_one.plan
    cmd = [sys.executable, os.path.join(FRAMEWORK_DIR, "run_fit_adaptive.py"),
           "-M", str(mass), "-c", args.config, "-s", args.sigdir,
           "-o", outdir,
           "--n-sigma-start", str(args.n_sigma_start), "--n-sigma-min", str(args.n_sigma_min),
           "--bin-frac", str(args.bin_frac),
           "-i", plan[tag]["file"]]
    if args.m_data_min is not None:
        cmd += ["--m-data-min", str(args.m_data_min)]
    if args.m_data_max is not None:
        cmd += ["--m-data-max", str(args.m_data_max)]
    log = os.path.join(outdir, "adaptive_log.txt")
    os.makedirs(outdir, exist_ok=True)
    with open(log, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=f)
    return tag, mass, ret.returncode


def collect(outroot, tag, mass):
    rj = os.path.join(outroot, tag, "M%s" % mass, "best", "fit_results_%s.json" % mass)
    return json.load(open(rj)) if os.path.exists(rj) else None


def main():
    args = get_args()
    plan = json.load(open(args.plan))
    run_one.plan = plan

    jobs = sorted((tag, m) for tag, info in plan.items() for m in info["masses"])
    print("Running %d adaptive fits (%d concurrent) -> %s ..."
          % (len(jobs), args.workers, args.outroot), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tag, mass, rc in ex.map(lambda j: run_one(j, args), jobs):
            done += 1
            print("  [%d/%d] %s M%s  rc=%d" % (done, len(jobs), tag, mass, rc), flush=True)

    # summarize
    summary = {}
    print("\n" + "=" * 92)
    print("%-9s %-6s %-6s %-17s %-7s %-8s %-9s %s" %
          ("window", "mass", "stat", "N_sig +- unc", "signif", "sbProb", "obs_lim", "window[GeV]"))
    print("-" * 92)
    npass = 0
    for tag, info in plan.items():
        for m in info["masses"]:
            r = collect(args.outroot, tag, m)
            row = summary.setdefault(tag, {})
            if r is None:
                print("%-9s %-6s %-6s" % (tag, m, "FAIL"))
                row[str(m)] = {"status": "fail"}
                continue
            npass += 1
            ns = r.get("obs_excess_events", 0.0)
            nsu = r.get("obs_excess_events_unc", 0.0)
            sig = r.get("signif", float("nan"))
            p = r.get("sbfit_prob", -1)
            lim = r.get("obs_lim_events", float("nan"))
            mlo, mhi = r.get("m_min"), r.get("m_max")
            print("%-9s %-6s %-6s %7.1f +- %-6.1f %-7.2f %-8.3f %-9.1f [%.2f,%.2f]" %
                  (tag, m, "OK", ns, nsu, sig, p, lim, mlo, mhi))
            row[str(m)] = {"status": "ok", "center": info["center"],
                           "n_sig": ns, "n_sig_unc": nsu, "signif": sig,
                           "sbfit_prob": p, "obs_lim_events": lim,
                           "m_min": mlo, "m_max": mhi}
            # Expected limit + bands and the normalization convention, needed to turn
            # the event limit into a sigma*B limit. doFit.py leaves these at -1 when
            # AsymptoticLimits fails, so keep -1 rather than papering over it.
            row[str(m)].update({k: r.get(k, -1.0) for k in LIMIT_FIELDS})
    print("-" * 92)
    print("passed %d / %d" % (npass, len(jobs)))
    sdir = os.path.dirname(args.summary)
    if sdir:
        os.makedirs(sdir, exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print("wrote %s" % args.summary)


if __name__ == "__main__":
    main()
