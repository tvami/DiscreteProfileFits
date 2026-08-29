#!/usr/bin/env python3
"""
convert_signal_h5.py — turn a raw phase-space signal H5 into the single-'masses'-key
format the Stage-4 mass fitters (fit_signalshapes.py / doFit.py) expect.

Stage 4 only ever needs the per-event invariant (dimuon) mass. In the phase-space
files that mass is stored under the key 'Background_dimuon_mass' (see
shared/process_h5_segmented.py / run_stage0_full.py). This script extracts that column
and writes an H5 containing only a 1-D float64 dataset called 'masses'.

Idempotent: if the input already has a 'masses' dataset it is passed straight through,
so it is safe to run on already-converted files.

Usage:
    python convert_signal_h5.py -i raw_signal.h5 -o signal_masses.h5
    python convert_signal_h5.py -i raw_signal.h5 -o out.h5 --mass-key Signal_dimuon_mass
"""
import argparse
import sys

import h5py
import numpy as np


def find_mass_array(f, explicit_key=None):
    """Return (key, 1-D mass array) from an open H5 file.

    Preference order:
      1. an existing 'masses' dataset (passthrough),
      2. an explicitly requested --mass-key,
      3. 'Background_dimuon_mass' (the phase-space convention),
      4. any key ending in 'dimuon_mass'.
    """
    keys = list(f.keys())

    if "masses" in keys:
        return "masses", np.asarray(f["masses"][:]).ravel()

    candidates = []
    if explicit_key:
        candidates.append(explicit_key)
    candidates.append("Background_dimuon_mass")
    candidates.extend(k for k in keys if k.endswith("dimuon_mass"))

    for key in candidates:
        if key in keys:
            return key, np.asarray(f[key][:]).ravel()

    raise KeyError(
        "No 'masses' or '*_dimuon_mass' dataset found. "
        f"Available keys: {sorted(keys)}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Raw phase-space signal H5")
    p.add_argument("-o", "--output", required=True, help="Output masses-key H5")
    p.add_argument("--mass-key", default=None,
                   help="Override the dimuon-mass dataset name to extract")
    args = p.parse_args()

    with h5py.File(args.input, "r") as f:
        src_key, masses = find_mass_array(f, args.mass_key)

    masses = masses.astype(np.float64)
    n_total = len(masses)
    finite = np.isfinite(masses)
    if not finite.all():
        print(f"  Dropping {np.sum(~finite)} non-finite mass values")
        masses = masses[finite]

    if len(masses) == 0:
        print("ERROR: no finite mass values to write", file=sys.stderr)
        sys.exit(1)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("masses", data=masses)

    tag = "passthrough" if src_key == "masses" else f"from '{src_key}'"
    print(f"  Converted {args.input} ({tag})")
    print(f"    {len(masses)}/{n_total} events  range=[{masses.min():.3f}, {masses.max():.3f}]")
    print(f"  Wrote {args.output}  (key 'masses')")


if __name__ == "__main__":
    main()
