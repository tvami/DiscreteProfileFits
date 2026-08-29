# Fitting framework for F-test + Discrete profile fits

The goal of this repo is to provide a universal, easy-to-use framework for performing parametric fits with multiple families of functional forms.
Signal events are fit to a parametric shape (double crystal-ball) and saved.
An input dataset along with a mass range to be fit is provided.
The data is then fit with the various choices of functional forms describing the background shape.
For each functional form an F-test is performed to select the optimal number of parameters.
Then, a signal plus background fit is performed using Combine. Discrete profiling is used to include all functional forms, each with their determined optimal number of parameters.

## Repository layout

This repo tracks **only the general fitting framework**: `Fitter.py`,
`DataCardMaker.py`, `Utils.py`, `doFit.py`, `fit_signalshapes.py`,
`interpolation.py`, `makePostfitPlot.py`, `run_fit_adaptive.py`, `CMS_lumi.py`,
`tdrstyle.py`, and `dimuonX_config.json` as a worked example config.

Particular applications of the framework (a specific search's automation,
one-off validation tooling, signal shapes, the analysis note, etc.) live under
`analyses/<name>/`, each self-contained with its own `.gitignore` for its own
bulk outputs. See [`analyses/dimuonX/README.md`](analyses/dimuonX/README.md)
for the dimuon resonance search, the framework's reference application.

## Installation

This framework relies on Combine. Follow the latest recommendations at https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#within-cmssw-recommended-for-cms-users

The instructions for the current recommended version (v10.4.2) are:

```bash
cmsrel CMSSW_14_1_8
cd CMSSW_14_1_8/src
cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v10.4.2 \
    https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b -j$(nproc --ignore=2)
```

Clone this repo inside the `src/` directory:

```bash
git clone git@github.com:OzAmram/DiscreteProfileFits.git
```

## Inputs

Input files containing masses to be fit must be HDF5 files with a single dataset,
named either `masses` or `mass`.

Signal shapes are stored in `.json` files produced by `fit_signalshapes.py`.

## Fit configuration file

Both `doFit.py` and `fit_signalshapes.py` accept a JSON config file (`-c / --config`) that sets
all fit parameters. Command-line arguments override config values when both are given.

Example config (`dimuonX_config.json`):

```json
{
    "inputFile":    "bkg_mc_masses.h5",
    "m_min":        30.0,
    "m_max":        36.1,
    "bin_size":     0.05,
    "dcb_model":    true,
    "sqrts":        "13.6",
    "lumi":         "",
    "scale_j_unc":  0.01,
    "res_j_unc":    0.035,
    "lumi_unc":     0.016,
    "sig_norm":     100.0,
    "ftest_thresh": 0.05,
    "err_thresh":   0.5,
    "bkg_name":       "CMS_EXO26006_bkg",
    "pdf_index_name": "CMS_EXO26006_bkg_pdfindex",
    "bkg_par_prefix": "CMS_EXO26006_bkg_",
    "lumi_syst_name": "lumi_13p6TeV_2024",
    "func_forms": {
        "bern":    [2, 3, 4],
        "polyExp": [1, 2, 3],
        "exp":     [1, 2, 3],
        "expPoly": [1, 2, 3, 4]
    }
}
```

`func_forms` lists the polynomial **order** to try for each family, not the number
of free parameters — the two differ per family (`Utils.get_nPars`): `bern` → order,
`polyExp` → order+1, `expPoly` → order, `exp` → 2*order-1 (`order` slopes plus
order-1 coefficients, the last fraction being fixed by normalization). The orders
above cap every family at 4 free parameters, except `exp` at 5. Allowing too much
flexibility lets the background absorb genuine signal-like features, so prefer
capping these rather than adding orders.

Key fields:

| Field | Description |
|-------|-------------|
| `inputFile` | HDF5 file with a `masses` (or `mass`) dataset |
| `m_min`, `m_max` | Fit window boundaries (GeV) |
| `bin_size` | Histogram bin width (GeV). A good rule of thumb is `0.25 × σ_signal` |
| `dcb_model` | Use double crystal-ball signal model (recommended) |
| `sig_norm` | Signal yield corresponding to signal strength `r = 1` |
| `lumi_unc` | Fractional integrated-luminosity uncertainty, written as an lnN on the signal only (the background is data-driven). `0.016` for 2024, CERN-CMS-DP-2026-003. Set `<= 0` to drop the nuisance |
| `bkg_name` | Name of the background envelope (`RooMultiPdf`) in the workspace and the card. Default `multi_pdf`. Its yield is always `<bkg_name>_norm`, since combine looks the normalization up by that name |
| `pdf_index_name` | Name of the discrete-profiling index. Default `pdf_index` |
| `bkg_par_prefix` | Prefix for the background pdf and shape-parameter names, e.g. `CMS_EXO26006_bkg_` gives `CMS_EXO26006_bkg_bern_p0`. Default empty, i.e. `bern_p0` |
| `lumi_syst_name` | Name of the luminosity lnN nuisance. Default `lumi_13p6TeV` |
| `ftest_thresh` | p-value threshold for the F-test to prefer a higher-order function |
| `err_thresh` | Fractional fit-error threshold; models with large errors are excluded from F-test. Currently inert — `doFit.py` doesn't populate per-order fit errors, so this guard never fires; kept for when that's restored |
| `func_forms` | Dict mapping functional-form name → list of orders to try. Available forms: `bern` (Bernstein polynomial), `exp` (sum of exponentials), `polyExp` (polynomial × exponential), `expPoly` (exponential of a polynomial) |

### Nuisance parameter naming

`bkg_name`, `pdf_index_name`, `bkg_par_prefix` and `lumi_syst_name` exist so that the
released cards can follow the
[CMS systematics naming conventions](https://gitlab.cern.ch/cms-analysis/general/systematics),
which require analysis-specific parameters to start with `CMS_<analysisID>_` and the
luminosity nuisance to carry its era. The defaults reproduce the historical generic
names (`multi_pdf`, `pdf_index`, `bern_p0`, `lumi_13p6TeV`), so existing configs are
unaffected; `analyses/dimuonX/dimuonX_config.json` sets the EXO-26-006 names.

`bkg_par_prefix` is applied to both the F-test fits and the final workspace shapes,
because the F-test best-fit values are cached under the parameter names and re-read as
starting points for the envelope. Changing only one of the two would silently drop the
seeding.

## Available background functional forms

| Name | Formula |
|------|---------|
| `bern` | Bernstein polynomial of degree `order−1` |
| `exp` | Sum of `order` exponentials |
| `polyExp` | `(1 + Σ pᵢ xⁱ) × exp(c x)` |
| `expPoly` | `exp(Σ pᵢ xⁱ)` |

## Signal shape fitting

Fit signal MC masses at each mass point:

```bash
python3 fit_signalshapes.py \
    -i signal_data/2G/signal_mass_40.h5 \
    -M 40 \
    --dcb-model \
    --no-error-band \
    --m-data-max 80.0 \
    -o signal_fits/2G/
```

This produces `signal_fits/2G/sig_fit_40.json` with the fitted DCB parameters.

### Signal shape interpolation

After fitting all mass points, interpolate to produce signal templates at arbitrary mass
spacing:

```bash
python3 interpolation.py \
    -s case \
    --json-dir signal_fits/2G/ \
    --masses $(python3 -c "import numpy as np; print(*np.round(np.arange(15,80.01,0.2),1))") \
    -m 15 -M 75 \
    -o signal_fits/2G/
```

This produces one JSON per mass point, e.g. `signal_fits/2G/case_interpolation_M40.0.json`.

Use `--fit-mass-min`/`--fit-mass-max` to exclude `sig_fit_*.json` points outside a
mass range from the parameterization (e.g. when the highest generated masses
sit on a kinematic edge and are poorly constrained) without deleting them.

## Running a single fit

```bash
python3 doFit.py \
    -c dimuonX_config.json \
    -M 40 \
    --m-min 34.7 --m-max 45.3 \
    --bin-size 0.166 \
    -s signal_fits/2G/case_interpolation_M40.0.json \
    -o fits/M40/
```

A good starting point for the window and bin size at mass `M` with signal width `σ`:
- Window: `[M − 8σ, M + 8σ]` (floor at the data lower edge)
- Bin size: `0.25 × σ`

Results and postfit plots are saved in the output directory. The JSON
`fit_results_{M}.json` contains fit quality metrics including `sbfit_prob` and
`bkgfit_prob` (chi²/ndof p-values for the s+b and background-only postfit plots).

## Adaptive fit wrapper

`run_fit_adaptive.py` automates window-size and bin-size selection. It starts at
±8σ and 0.25σ bins; if the s+b postfit chi² p-value is below threshold, it
shrinks the window down to ±4σ (0.5σ steps). If no window size passes at the
starting bin fraction, it coarsens the bin size (0.1σ steps, up to 1.0σ by
default) and repeats the window scan at each — coarser bins are less sensitive
to a narrow statistical fluctuation landing in one bin. If nothing passes
outright, it falls back to the highest-p attempt across the whole (bin size,
window) grid rather than dropping the mass point.

```bash
python3 run_fit_adaptive.py \
    -M 72 \
    -c dimuonX_config.json \
    -s signal_fits/2G \
    -o fits/M72/
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-M` | required | Signal mass hypothesis (GeV) |
| `-c` | `dimuonX_config.json` | Config file |
| `-s` | required | Directory containing `case_interpolation_M{mass}.json` |
| `-o` | `bkg_mc_fits/M{mass}` | Output directory |
| `-i` | config's `inputFile` | Override the input h5 file (e.g. a mock signal-injected dataset) |
| `--pval-thresh` | `0.05` | Minimum acceptable s+b chi² p-value |
| `--n-sigma-start` | `8.0` | Starting half-width in σ units |
| `--n-sigma-min` | `4.0` | Minimum half-width before declaring failure |
| `--n-sigma-step` | `0.5` | Step size for shrinking the window |
| `--bin-frac` | `0.25` | Starting bin size as a fraction of σ |
| `--bin-frac-max` | `1.0` | Max bin size (in σ) to escalate to if every window fails at the starting bin size |
| `--bin-frac-step` | `0.1` | Bin-size escalation step, in σ |
| `--m-data-min` | none | Floor the fit window at this mass (data lower edge) |
| `--m-data-max` | none | Cap the fit window at this mass (data upper edge) |

Each attempted (bin size, window size) combination is saved in a subdirectory
`bin{f}_window{N}sig/` for review. The best passing attempt is copied to
`best/`. The script exits with code 0 on success and 1 on failure.
