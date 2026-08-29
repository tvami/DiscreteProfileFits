import os
import pickle
import json
import random
import optparse
import math
import types
from Fitter import Fitter, shape_map
from DataCardMaker import DataCardMaker
from Utils import *
from array import array
from fit_signalshapes import  fit_signalmodel
from makePostfitPlot import make_postfit_plot

# signal-strength range passed to combine's FitDiagnostics/AsymptoticLimits below;
# also used to recognize a MINOS error that hit this wall rather than a real crossing.
R_MIN = -5
R_MAX = 10

def dofit(options):

    label = options.label
    plot_dir = options.outDir
    if(plot_dir[-1] != '/'):
        plot_dir += '/'

    if(not os.path.exists(plot_dir)):
        os.system("mkdir %s" % plot_dir)

    if(os.path.exists(plot_dir + "fit_results_{}.json".format(options.mass))):
        #remove old results
        os.system("rm %s" % plot_dir + "fit_results_{}.json".format(options.mass))

    mass = options.mass 
    binsx = list(np.arange(options.m_min, options.m_max, options.bin_size))


    nbins_fine = int((options.m_max - options.m_min) / options.bin_size)
    print('Num bins', nbins_fine)

    histos_sb = ROOT.TH1F("m_sb", "m_sb" ,nbins_fine, 0., 1.)

    load_h5_sb(options.inputFile, histos_sb, xmin=options.m_min, xmax=options.m_max)
    print("************ Found %i total events \n" % histos_sb.GetEntries())
    print(histos_sb.Integral())

    if(not os.path.exists(options.sig_shape)):
        print("Sig file %s doesn't exist" % options.sig_shape)
        exit(1)

    sig_file_name = options.sig_shape

    sb_fname = plot_dir + "sb_fit.root"
    sb_outfile = ROOT.TFile(sb_fname, 'RECREATE')
    sb_outfile.cd()
    histos_sb.Write("m_sb")
    sb_outfile.Close()
    sig_data_name = 'm_sb'
    fit_label = "mumu"
    poi_name = "m"

    print(" CREATE WORKSPACE ")
    card = DataCardMaker(fit_label, outdir=plot_dir,
                         bkg_name=options.bkg_name,
                         pdf_index_name=options.pdf_index_name)

    card.importBinnedData(sb_fname, sig_data_name,
                          poi_name, 'data_obs', 1.0)

    if options.dcb_model:
        card.addDCBSignalShape('model_signal_m', sig_file_name,
                               {'CMS_scale_j': 1.0}, {'CMS_res_j': 1.0}, 
                               xmin=options.m_min, xmax=options.m_max)
    else:
        card.addSignalShape('model_signal_m',  sig_file_name,
                            {'CMS_scale_j': 1.0}, {'CMS_res_j': 1.0},
                            xmin=options.m_min, xmax=options.m_max)


    sig_norm = card.addFixedYieldFromFile('model_signal_m', 0, sig_file_name,
                                          "m_sig", norm = options.sig_norm)
    card.addSystematic("CMS_scale_j", "param", [0.0, options.scale_j_unc])
    card.addSystematic("CMS_res_j", "param", [0.0, options.res_j_unc])

    if options.blinded:
        print("BLIND FIT TO DO ")
        exit(1)

    fitting_histogram = histos_sb
    data_name = "data_bkg"

    func_forms = options.func_forms
    final_func_forms = dict()
    for func_form, orderToTry in func_forms.items():
        print("\n \n Fitting with functional form %s " % func_form)

        # Names of this family's pdf and shape parameters in the released workspace.
        # bkg_par_prefix is empty by default, giving the historical 'bern_p0' etc.
        par_label = options.bkg_par_prefix + func_form

        chi2s = [0]*len(orderToTry)
        fit_params = [0] * len(orderToTry)
        ndofs = [0]*len(orderToTry)
        probs = [0]*len(orderToTry)
        fit_errs = [0]*len(orderToTry)
        bkg_fnames = [""]*len(orderToTry)

        for i, order in enumerate(orderToTry):
            print("Trying %i parameter background fit" % order)
            bkg_fnames[i] = plot_dir + func_form + "_" + str(order) + 'par_bkg_fit%i.json' % i

            model_name = "model_b" + str(i)
            fitter_bkg = Fitter(['m_fine'], debug = False, outdir=plot_dir)
            # Ensure the scratch cache is always removed, even if the fit raises.
            try:
                fitter_bkg.importBinnedData(fitting_histogram, 'm_fine', data_name)
                # par_label must match the one used for the final workspace shape below,
                # since these fit values are cached and re-read there as starting points.
                fitter_bkg.bkgShape(name=model_name, poi='m_fine', order=order,
                                    func_form=func_form, par_label=par_label)

                fres = fitter_bkg.fit(model_name, data_name, options=[ROOT.RooFit.Save(1), ROOT.RooFit.Verbose(0),  ROOT.RooFit.Minos(1), ROOT.RooFit.Minimizer("Minuit2")])

                chi2_fine, ndof_fine = fitter_bkg.projection(
                    model=model_name, data=data_name, poi="m_fine",
                    filename=plot_dir +func_form + "_" + str(order) + "par_bkg_fit.png", binning=0, logy=False,
                    show_error=False)

                chi2_prob = ROOT.TMath.Prob(chi2_fine, ndof_fine)

                bkg_fit_params = dict()
                for parName in fitter_bkg.par_names:
                    value, error = fitter_bkg.fetch(parName)
                    bkg_fit_params[parName] = (value, error)

                bkg_fit_params['par_names'] = fitter_bkg.par_names
                bkg_fit_params['cov'] = convert_matrix(fres.covarianceMatrix())


                with open(bkg_fnames[i], "w") as jsonfile:
                    json.dump(bkg_fit_params, jsonfile, indent=4)


                print("#############################")
                print("Order %i results: " % order)
                print("bkg fit chi2/nbins (fine binning) ", chi2_fine, ndof_fine, chi2_fine/ndof_fine, chi2_prob)
                print("#############################")

                chi2s[i] = chi2_fine
                ndofs[i] = ndof_fine
                probs[i] = chi2_prob
                fit_params[i] = bkg_fit_params
                fit_errs[i] = 0. # Deprecated
            finally:
                fitter_bkg.delete()

        #F-test on this functional for to determine best num of parameters
        best_i = f_test(orderToTry, ndofs, chi2s, fit_errs, thresh = options.ftest_thresh, err_thresh = options.err_thresh)
        best_order = orderToTry[best_i]
        print("\n Chose order %i based on F-test ! \n" % best_order)

        #save func and order
        final_func_forms[func_form] = best_i

        #Add this functional form to workspace of the final fit,
        # seeding with the Fitter's best-fit values so combine starts from a
        # good initial point for every pdf_index (not just the selected one).
        shape_builder = shape_map[func_form]
        bkg_model,_,bkg_pars = shape_builder(par_label, card.poi, order=best_order,
                                              par_label=par_label,
                                              start_vals=fit_params[best_i])

        card.bkg_shapes.append(bkg_model)
        card.bkg_pars.extend(bkg_pars)
        func_name = func_form + "_order" + str(best_order)
        card.bkg_shape_names.append(func_name)


    print("#### Building multi pdf ### ")
    card.buildBkgShape()

    if(options.sig_norm_unc > 0):
        card.addSystematic("SigEff", "lnN", values = {"model_signal_m" : 1. + options.sig_norm_unc})

    # Integrated-luminosity uncertainty. The background is data-driven, so this hits
    # the signal normalization only. 1.6% for the 2024 dataset, CERN-CMS-DP-2026-003.
    if(options.lumi_unc > 0):
        card.addSystematic(options.lumi_syst_name, "lnN",
                           values = {"model_signal_m" : 1. + options.lumi_unc})

    # Signal efficiency terms. All are signal only, like the luminosity: the
    # background is data-driven and carries the efficiencies of the data itself.
    if(options.trig_eff_unc > 0):
        card.addSystematic(options.trig_eff_name, "lnN",
                           values = {"model_signal_m" : 1. + options.trig_eff_unc})

    if(options.muon_id_eff_unc > 0):
        card.addSystematic(options.muon_id_eff_name, "lnN",
                           values = {"model_signal_m" : 1. + options.muon_id_eff_unc})

    if(options.anom_eff_unc > 0):
        card.addSystematic(options.anom_eff_name, "lnN",
                           values = {"model_signal_m" : 1. + options.anom_eff_unc})

    print("making card")
    card.makeCard()
    #card.delete()

    cmd = (
        " cd {plot_dir} ; "
        + "text2workspace.py datacard_mass_{l2}.txt -o workspace_{l1}_{l2}.root; "
        + "combine -M FitDiagnostics workspace_{l1}_{l2}.root -m {mass} -n _{l1}_{l2} --robustFit 1 --cminDefaultMinimizerStrategy {strat} --saveWorkspace --rMin {rmin} --rMax {rmax} {extra}; "
        + "combine -M AsymptoticLimits workspace_{l1}_{l2}.root -m {mass} -n lim_{l1}_{l2} --rMax {rmax}; "
        ).format(plot_dir=plot_dir, mass=mass, l1=label, l2=fit_label, rmin=R_MIN, rmax=R_MAX,
                 strat=getattr(options, "min_strategy", 0),
                 extra=getattr(options, "combine_extra", "") or "")
    print(cmd)
    os.system(cmd)
    workspace_name = 'workspace_{l1}_{l2}.root'.format(l1=label, l2=fit_label)

    # combine formats the -m value with %g (e.g. 15 -> mH15, 15.2 -> mH15.2),
    # so match that here rather than rounding to an int (breaks non-integer masses).
    f_limit_name = (plot_dir + 'higgsCombinelim_{l1}_{l2}.'
                    + 'AsymptoticLimits.mH{mass:g}.root'
                    ).format(mass=mass, l1=label, l2=fit_label)
    f_diagnostics_name = (plot_dir + 'fitDiagnostics_{l1}_{l2}.root'
                   ).format(l1=label, l2=fit_label)


    f_diagnostics = ROOT.TFile(f_diagnostics_name, "READ")
    params_sb = f_diagnostics.Get("tree_fit_sb")
    params_sb.GetEntry(0)
    sig_strength = params_sb.r
    sig_strength_unc = params_sb.rErr
    # robustFit + cminDefaultMinimizerStrategy 0 can leave the symmetric rErr at 0;
    # fall back to the MINOS errors (rLoErr/rHiErr) which combine still computes.
    # A MINOS scan that never crosses the 1-sigma NLL threshold before hitting the
    # r range wall (R_MIN/R_MAX) reports the distance to that wall instead, which is
    # not a real uncertainty (seen when the likelihood is essentially flat in that
    # direction -- background can absorb a large signal-strength shift there). Drop
    # any wall-pinned side from the fallback rather than averaging in that artifact.
    try:
        rlo = abs(params_sb.rLoErr)
        rhi = abs(params_sb.rHiErr)
        eps = 1e-3
        lo_at_wall = rlo > 0 and abs((sig_strength - rlo) - R_MIN) < eps
        hi_at_wall = rhi > 0 and abs((sig_strength + rhi) - R_MAX) < eps
        if sig_strength_unc <= 0:
            if lo_at_wall and hi_at_wall:
                sig_strength_unc = -1.0  # both sides degenerate: uncertainty undefined
            elif lo_at_wall:
                sig_strength_unc = rhi
            elif hi_at_wall:
                sig_strength_unc = rlo
            elif rlo > 0 or rhi > 0:
                sig_strength_unc = 0.5 * (rlo + rhi)
    except Exception:
        pass
    nll_sb = params_sb.nll_min


    print("Signal strength is %.3f +/- %.3f" % (sig_strength, sig_strength_unc))

    params_b = f_diagnostics.Get("tree_fit_b")
    params_b.GetEntry(0)
    nll_b = params_b.nll_min

    delta_ll = nll_b - nll_sb
    print("dLL is %.2e" % delta_ll)

    delta_ll = max(delta_ll, 0.)
    signif = np.sqrt(2 * delta_ll)
    pval = 0.5 * math.erfc(signif / math.sqrt(2))

    print("Asymptotic significance (from dLL) is %.2f" % (signif))


    print('sig_norm %.3f' % sig_norm)



    # Count over the actual fit range, not binsx[0]..binsx[-1]: np.arange stops one
    # bin short of m_max (39.8 for [30, 40] in 0.2 bins) while the histogram's last
    # bin does extend to m_max, so the old bounds undercounted the truth signal and
    # made true_sig_strength come out slightly below the value it is normalised to.
    n_sig_true_in_window = get_sig_in_window(options.inputFile,
                                             options.m_min, options.m_max)
    true_sig_strength = n_sig_true_in_window /  sig_norm
    print("True sig strength %.3f (%i truth signal events in [%.2f, %.2f])"
          % (true_sig_strength, n_sig_true_in_window, options.m_min, options.m_max))

    #expected significance TODO
    exp_signif = 0.5
    exp_pval = 1.0


    f_limit = ROOT.TFile(f_limit_name, "READ")
    res2 = f_limit.Get("limit")
    eps = 0.01
    obs_limit = -1
    exp_limit = exp_low = exp_high = exp_two_low = exp_two_high = -1

    
    for i in range(6):
        res2.GetEntry(i)
        if(res2.quantileExpected == -1):  # obs limit
            obs_limit = res2.limit
        elif(abs(res2.quantileExpected - 0.5) < eps):  # exp limit
            exp_limit = res2.limit
        elif(abs(res2.quantileExpected - 0.025) < eps):  # 2sigma, low
            exp_two_low = res2.limit
        elif(abs(res2.quantileExpected - 0.16) < eps):  # 1sigma, low
            exp_low = res2.limit
        elif(abs(res2.quantileExpected - 0.84) < eps):  # 1sigma, high
            exp_high = res2.limit
        elif(abs(res2.quantileExpected - 0.975) < eps):  # 2sigma, high
            exp_two_high = res2.limit

    print("Obs limit is %.3f (%.1f events)" % (obs_limit, obs_limit*sig_norm))
    print("Expected was %.3f (%.1f events)" % (exp_limit, exp_limit*sig_norm))
    print("Expected range %.1f-%.1f (one sigma), %.1f-%.1f (two sigma)" % (exp_low * sig_norm, exp_high*sig_norm, exp_two_low * sig_norm, exp_two_high * sig_norm))


    #signal yield in +/- 2 sigma
    check_rough_sig(options.inputFile, options.mass*0.9, options.mass*1.1)


    f_limit.Close()
    f_diagnostics.Close()

    results = dict()

    # bkg fit results
    results['signif'] = signif
    # signif is unsigned (delta_ll is clipped at 0), so a downward fluctuation
    # reads as an excess.  Carry the sign of the fitted yield alongside it, the
    # same convention as summarize_data_scan.py's data scan.
    results['signif_signed'] = signif * (1.0 if sig_strength >= 0 else -1.0)
    results['asimov_signif'] = exp_signif
    results['asimov_pval'] = exp_pval
    results['pval'] = pval
    results['func_forms'] = func_forms
    results['final_func_forms'] = final_func_forms
    results['sig_strength'] = sig_strength
    results['sig_strength_unc'] = sig_strength_unc
    results['sig_norm'] = sig_norm
    # Closure quantities: r_true is what the fit should recover if the chain is
    # unbiased.  Both are 0 when the input h5 carries no truth_label dataset.
    results['true_sig_strength'] = true_sig_strength
    results['n_sig_true_in_window'] = int(n_sig_true_in_window)
    # r pinned at the R_MIN/R_MAX wall is a bound, not a measurement -- flag it so
    # downstream summaries can refuse to quote the number.
    results['r_at_bound'] = bool(abs(sig_strength - R_MAX) < 1e-3
                                 or abs(sig_strength - R_MIN) < 1e-3)
    results['r_min'] = R_MIN
    results['r_max'] = R_MAX
    results['obs_excess_events'] = sig_strength*sig_norm
    results['obs_excess_events_unc'] = (-1.0 if sig_strength_unc == -1.0
                                         else sig_strength_unc*sig_norm)
    results['obs_lim_events'] = obs_limit*sig_norm
    results['exp_lim_events'] = exp_limit*sig_norm
    results['exp_lim_1sig_low'] = exp_low * sig_norm
    results['exp_lim_2sig_low'] = exp_two_low * sig_norm
    results['exp_lim_1sig_high'] = exp_high * sig_norm
    results['exp_lim_2sig_high'] = exp_two_high* sig_norm
    results['sig_norm_unc'] = options.sig_norm_unc
    results['mass'] = options.mass
    results['m_min'] = options.m_min
    results['m_max'] = options.m_max
    results['script_options'] = vars(options)

    print("Saving fit results to %s" % plot_dir + "fit_results_{}.json".format(options.mass))

    with open(plot_dir + "fit_results_{}.json".format(options.mass), "w") as jsonfile:
        json.dump(results, jsonfile, indent=4)

    # Postfit signal + background plot
    postfit_opt = types.SimpleNamespace(
        inputWSFile = plot_dir + "datacardInputs_mass_%s.root" % fit_label,
        fitDiagFile = f_diagnostics_name,
        cat         = "mass_%s" % fit_label,
        mass        = options.mass,
        mMin        = options.m_min,
        mMax        = options.m_max,
        nBins       = nbins_fine,
        pdfNBins    = 600,
        sigNorm     = options.sig_norm,
        poiName     = poi_name,
        outDir      = plot_dir,
        ext         = "",
        lumi        = options.lumi,
        sqrts       = options.sqrts,
        drawSignal  = True,
        jsonFile    = plot_dir + "fit_results_{}.json".format(options.mass),
    )
    make_postfit_plot(postfit_opt)

    # Background-only postfit plot
    postfit_opt_bkg = types.SimpleNamespace(**vars(postfit_opt))
    postfit_opt_bkg.bkgOnly    = True
    postfit_opt_bkg.drawSignal = False
    make_postfit_plot(postfit_opt_bkg)

    return results


def fitting_options():
    parser = optparse.OptionParser()
    parser.add_option("-c", "--config", dest="config", default=None,
                      help="JSON config file; keys become defaults overrideable by CLI args")
    parser.add_option("-M", "-M", dest="mass", type=float, default=15.,
                      help="Signal mass hypothesis")
    parser.add_option("-i", "--inputFile", dest="inputFile",
                      default='fit_inputs/no_selection_03p.h5',
                      help="input h5 file")
    parser.add_option("-o", "--outDir", dest="outDir", default='plots/',
                      help="Where to put the output")
    parser.add_option("-s", "--sig_shape", default="sig_shape_M15.json",
                      help="Pre-saved signal shape")

    parser.add_option("--m-min", type=float, default=11.0,
                      help="Minimum m for the fit")
    parser.add_option("--m-max", type=float, default=20.0,
                      help="Maximum m for the fit")
    parser.add_option("--bin-size", type=float, default=0.2,
                      help="Size of bins")

    parser.add_option("--scale_j_unc", type=float, default=0.01,
                      help="Uncertainty on signal mean from JES")
    parser.add_option("--res_j_unc", type=float, default=0.035,
                      help="Uncertainty on signal width from JER")
    parser.add_option("--sig_norm", type=float, default=100.0,
                      help="Signal normalization (definition of mu=1)")
    parser.add_option("--ftest_thresh", type=float, default=0.05,
                      help="Threshold to prefer a function in the f-test")
    parser.add_option("--err_thresh", type=float, default=0.5,
                      help="Threshold on fit unc to be included in f-test")
    parser.add_option("--refit_sig", default=False, action="store_true",
                      help="""Fit the signal events (using truth labels)
                      to get signal shape""")
    parser.add_option("--rebin", default=False, action="store_true",
                      help="""Rebin  to make sure no bins less than 5 evts""")
    parser.add_option("-l", "--label", dest="label", default='test',
                      help="Label for file names")
    parser.add_option("--no_draw_sig", dest="draw_sig", action = 'store_false', help="Don't draw separate signal and bkg contribution on S+B fit plots")
    parser.set_defaults(draw_sig = True)
    parser.add_option("-b", "--blinded", dest="blinded", action="store_true",
                      default=False,
                      help="Blinding the signal region for the fit.")
    parser.add_option("--dcb-model", dest="dcb_model", action="store_true",
                      default=False,
                      help="""Whether to use double crystal ball model for signal shape instead
                      of default model (gaussian core with single crystal ball)""")
    parser.add_option("--sig_norm_unc", dest="sig_norm_unc", type=float, default=-1.0,
                      help="Fractional uncertainty on signal normalization (for limits)")
    # 2024 integrated luminosity, CERN-CMS-DP-2026-003 (https://cds.cern.ch/record/2952191).
    # Applied as a lnN on the signal only; the background estimate is data-driven.
    parser.add_option("--lumi_unc", dest="lumi_unc", type=float, default=0.016,
                      help="Fractional integrated-luminosity uncertainty applied as a "
                           "lnN on the signal yield (default 0.016 = 1.6%%, 2024, "
                           "CERN-CMS-DP-2026-003). Set <= 0 to disable.")
    parser.add_option("--lumi", dest="lumi", default="",
                      help="Luminosity string for plot label, e.g. '27.0 fb^{-1}'")
    # Nuisance/object naming. The defaults reproduce the historical generic names;
    # set these (most easily in the config JSON) to follow the CMS systematics naming
    # conventions, which require analysis-specific parameters to start with
    # CMS_<analysisID>_ and the luminosity nuisance to carry its era.
    parser.add_option("--bkg-name", dest="bkg_name", default="multi_pdf",
                      help="Name of the background envelope (RooMultiPdf) in the "
                           "workspace; its yield is always <bkg_name>_norm")
    parser.add_option("--pdf-index-name", dest="pdf_index_name", default="pdf_index",
                      help="Name of the discrete profiling index in the workspace and card")
    parser.add_option("--bkg-par-prefix", dest="bkg_par_prefix", default="",
                      help="Prefix prepended to the background shape parameter and pdf "
                           "names, e.g. 'CMS_EXO26006_bkg_' gives CMS_EXO26006_bkg_bern_p0")
    parser.add_option("--lumi-syst-name", dest="lumi_syst_name", default="lumi_13p6TeV",
                      help="Name of the integrated-luminosity lnN nuisance")
    parser.add_option("--trig_eff_unc", dest="trig_eff_unc", type=float, default=-1.0,
                      help="Fractional trigger-efficiency scale-factor uncertainty, applied "
                           "as a lnN on the signal yield. Set <= 0 to disable (default).")
    parser.add_option("--trig-eff-name", dest="trig_eff_name", default="CMS_eff_m_trigger",
                      help="Name of the trigger-efficiency lnN nuisance. Use an analysis-specific "
                           "name (CMS_<analysisID>_eff_m_trigger) when the scale factors are "
                           "derived in the analysis rather than provided centrally.")
    parser.add_option("--muon_id_eff_unc", dest="muon_id_eff_unc", type=float, default=-1.0,
                      help="Fractional muon identification scale-factor uncertainty, applied as a "
                           "lnN on the signal yield. Set <= 0 to disable (default).")
    parser.add_option("--muon-id-eff-name", dest="muon_id_eff_name", default="CMS_eff_m_id",
                      help="Name of the muon-ID efficiency lnN nuisance.")
    parser.add_option("--anom_eff_unc", dest="anom_eff_unc", type=float, default=-1.0,
                      help="Fractional anomaly-classifier efficiency uncertainty, applied as a "
                           "lnN on the signal yield. Set <= 0 to disable (default).")
    parser.add_option("--anom-eff-name", dest="anom_eff_name", default="CMS_eff_anom",
                      help="Name of the anomaly-classifier efficiency lnN nuisance.")
    parser.add_option("--sqrts", dest="sqrts", default="13.6",
                      help="Centre-of-mass energy label (TeV)")
    # Strategy 0 is fast and fine for most points, but on a large injected signal
    # it can end with a non-positive-definite covariance ("MnPosDef ... non-positive
    # diagonal element"), after which combine reports "Fit failed", writes no fit_s
    # to fitDiagnostics, and leaves rErr AND the MINOS errors at 0. The postfit chi2
    # is then computed from pre-fit shape parameters and is meaningless. Raising the
    # strategy costs time but recomputes the Hessian properly. Default unchanged.
    parser.add_option("--min-strategy", dest="min_strategy", type=int, default=0,
                      help="combine --cminDefaultMinimizerStrategy (0, 1 or 2). "
                           "Raise to 1/2 if the S+B fit fails with a non-positive-"
                           "definite covariance matrix.")
    # Escape hatch for envelope (RooMultiPdf) convergence failures, e.g.
    # --combine-extra="--cminRunAllDiscreteCombinations". Appended to the
    # FitDiagnostics call only; empty by default so nothing else changes.
    parser.add_option("--combine-extra", dest="combine_extra", default="",
                      help="Extra options appended verbatim to the combine "
                           "FitDiagnostics command.")
    return parser


_DEFAULT_FUNC_FORMS = {
    "bern":    [2, 3, 4, 5, 6],
    "polyExp": [1, 2, 3],
    "exp":     [1, 2, 3, 4],
}

if __name__ == "__main__":
    parser = fitting_options()

    # First pass: find --config only (ignore unknown/positional args)
    (pre_opts, _) = parser.parse_args()

    func_forms = _DEFAULT_FUNC_FORMS.copy()
    if pre_opts.config:
        with open(pre_opts.config) as f:
            cfg = json.load(f)
        # func_forms is a structured dict — handle separately
        func_forms = cfg.pop("func_forms", func_forms)
        # Remaining flat keys become parser defaults; CLI args will override them
        parser.set_defaults(**cfg)

    # Second parse: CLI args take priority over config-sourced defaults
    (options, args) = parser.parse_args()
    options.func_forms = func_forms

    dofit(options)

