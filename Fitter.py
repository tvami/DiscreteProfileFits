import ROOT
import json
from numpy import random
from array import array
import os,sys
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

def scaledVariable(m, name='x'):
    #rescale range of variable to 0 to 1

    mini = m.getMin()
    maxi = m.getMax()
    old_name = m.GetName()
    x = ROOT.RooFormulaVar (name, "(%s - %.10f) / (%.10f - %.10f)" % (old_name, mini, maxi, mini), ROOT.RooArgList(m));

    return x


def polyShape(name = 'model', poi = None, order=4, par_label='poly', start_vals=None):
    #sum of bernstein polynomials

    #initial values
    start_val = 1.0
    pmin = -20.0
    pmax = 20.0

    par_list = ROOT.RooArgList()
    p_objs = []
    par_names = []
    for i in range(order):
        pname = f"{par_label}_p{i}"
        init = start_vals[pname][0] if (start_vals and pname in start_vals) else start_val
        poly_par = ROOT.RooRealVar(pname, pname, init, pmin, pmax)
        par_list.add(poly_par)
        par_names.append(pname)
        p_objs.append(poly_par)

    shape = ROOT.RooPolynomial(name+"_shape", name+"_shape", poi, par_list)

    return shape, par_names, p_objs


def bernShape(name = 'model', poi = None, order=4, par_label='bern', start_vals=None):
    #sum of bernstein polynomials

    pmin = 0.0
    pmax = 20.0

    par_list = ROOT.RooArgList()
    p_objs = []
    par_names = []
    for i in range(order):
        pname = f"{par_label}_p{i}"
        init = start_vals[pname][0] if (start_vals and pname in start_vals) else 1.0
        poly_par = ROOT.RooRealVar(pname, pname, init, pmin, pmax)
        par_list.add(poly_par)
        par_names.append(pname)
        p_objs.append(poly_par)


    shape = ROOT.RooBernstein(name+"_shape", name+"_shape", poi, par_list)

    return shape, par_names, p_objs


def bernPowerShape(name = 'model', poi = None, order=4, par_label='bernPower', start_vals=None):
    # Bernstein polynomial (degree order-1) times a power law: bern(x) * x^c.
    # Implemented as a single RooGenericPdf so only one normalization integral
    # is needed.  Using RooAddPdf(bern, power_law) fails because the power_law
    # PDF diverges at x=0 for c<0, causing RooFit's numerical integrator to
    # hang or report "integral does not converge".
    # The product bern(x)*x^c is finite and well-behaved for x in [0,1]:
    # Bernstein polynomials are 0 at the endpoints for interior basis terms,
    # and the overall product is integrable even for c > -1.

    from math import comb

    cname      = f"{par_label}_pow_coeff"
    coeff_init = start_vals[cname][0] if (start_vals and cname in start_vals) else 0.0
    # Lower bound -1: x^c is integrable on [0,1] only for c > -1; preventing
    # c < -1 stops the optimizer finding the degenerate spike-at-zero solution.
    coeff = ROOT.RooRealVar(cname, cname, coeff_init, -1.0, 5.0)

    bern_pars = []
    par_names = [cname]
    p_objs    = [coeff]

    for i in range(order):
        pname  = f"{par_label}_p{i}"
        p_init = start_vals[pname][0] if (start_vals and pname in start_vals) else 1.0
        p = ROOT.RooRealVar(pname, pname, p_init, 0.01, 20.0)
        bern_pars.append(p)
        p_objs.append(p)
        par_names.append(pname)

    # Build Bernstein polynomial of degree (order-1) as a formula string.
    # B_{i,n}(x) = C(n,i) * x^i * (1-x)^(n-i),  n = order-1
    # args: @0=poi, @1=coeff, @2..@(order+1)=bern control points
    n = order - 1
    terms = []
    for i in range(order):
        c_ni = comb(n, i)
        x_pow   = "" if i == 0   else ("*@0"         if i == 1   else f"*pow(@0,{i})")
        omx_pow = "" if n-i == 0 else ("*(1-@0)"     if n-i == 1 else f"*pow(1-@0,{n-i})")
        terms.append(f"@{i+2}*{c_ni}{x_pow}{omx_pow}")

    #add the power law part
    bern_str = "+".join(terms)
    # Small offset in the power term keeps the integrand finite at x=0
    # while being negligible (< 0.01%) across the physical range.
    formula = f"({bern_str})*pow(@0+1e-4,@1)"

    arg_list = ROOT.RooArgList()
    arg_list.add(poi)
    arg_list.add(coeff)
    for p in bern_pars:
        arg_list.add(p)
    p_objs.append(arg_list)

    shape = ROOT.RooGenericPdf(name+"_shape", name+"_shape", formula, arg_list)

    return shape, par_names, p_objs


def expShape(name, poi = None, order=4, par_label='exp', start_vals=None):
    #Sum of exponentials

    pmin = -20.0
    pmax = 20.0
    cmin = 0.
    cmax = 1.

    p_objs = []
    par_names = []
    e_list = ROOT.RooArgList()
    c_list = ROOT.RooArgList()

    for i in range(order):
        pname = f"{par_label}_p{i}"
        cname = f"{par_label}_c{i}"
        p_init = start_vals[pname][0] if (start_vals and pname in start_vals) else -0.5
        c_init = start_vals[cname][0] if (start_vals and cname in start_vals) else 0.2
        exp_par = ROOT.RooRealVar(pname, pname, p_init, pmin, pmax)
        exp = ROOT.RooExponential(f"{par_label}_{i}", f"{par_label}_{i}", poi, exp_par)
        coef = ROOT.RooRealVar(cname, cname, c_init, cmin, cmax)

        e_list.add(exp)
        par_names.append(pname)

        if(i<(order-1)): # don't include last coeff because redundant
            c_list.add(coef)
            par_names.append(cname)

        p_objs.extend([exp_par, exp, coef])

    recursiveFraction=True
    shape = ROOT.RooAddPdf(name+"_shape", name+"_shape", e_list, c_list, recursiveFraction)

    return shape, par_names, p_objs


def expPolyShape(name='model', poi=None, order=2, par_label='expPoly', start_vals=None):
    # Exponential of a polynomial: exp(p0*x + p1*x^2 + ... + p_{n-1}*x^n)
    # Implemented as RooGenericPdf; RooFit normalizes numerically over the fit range.

    p_objs = []
    par_names = []
    poly_pars = []

    for i in range(order):
        pname = f"{par_label}_p{i}"
        # Good starting point: mild falling slope for p0, small corrections for higher orders
        p_init = start_vals[pname][0] if (start_vals and pname in start_vals) else (-0.05 if i == 0 else 0.0)
        p = ROOT.RooRealVar(pname, pname, p_init, -10.0, 10.0)
        poly_pars.append(p)
        p_objs.append(p)
        par_names.append(pname)

    arg_list = ROOT.RooArgList()
    arg_list.add(poi)
    for p in poly_pars:
        arg_list.add(p)

    exponent_terms = "+".join(
        "@%d*pow(@0,%d)" % (i + 1, i + 1) for i in range(order)
    )
    formula = "exp(%s)" % exponent_terms

    shape = ROOT.RooGenericPdf(name + "_shape", name + "_shape", formula, arg_list)
    p_objs.append(arg_list)

    return shape, par_names, p_objs


def polyExpShape(name = 'model', poi = None, order=4, par_label='polyExp', start_vals=None):
    #Polynomial times exponential
    # Implemented as RooGenericPdf so the product is self-normalizing.
    # RooProdPdf of two PDFs over the same observable does NOT renormalize,
    # producing an invalid PDF with integral != 1.

    pmin = -20.0
    pmax = 20.0

    ec_name = f"{par_label}_exp_c"
    ec_init = start_vals[ec_name][0] if (start_vals and ec_name in start_vals) else -0.5
    exp_par = ROOT.RooRealVar(ec_name, ec_name, ec_init, pmin, pmax)

    par_list = ROOT.RooArgList()
    p_objs = [exp_par]
    par_names = []

    # Build the polynomial part as a sum: 1 + p0*x + p1*x^2 + ...
    # using RooFormulaVar terms captured in the RooGenericPdf expression.
    # Simpler: build a RooArgList of [poi, exp_par, poly_p0, poly_p1, ...]
    # and write the formula as exp(@1*@0) * (1 + @2*@0 + @3*@0^2 + ...)
    poly_pars = []
    for i in range(order):
        pp_name = f"{par_label}_poly_p{i}"
        pp_init = start_vals[pp_name][0] if (start_vals and pp_name in start_vals) else 0.5/(i+1)**2
        p = ROOT.RooRealVar(pp_name, pp_name, pp_init, -20.0, 20.0)
        poly_pars.append(p)
        p_objs.append(p)
        par_names.append(pp_name)
    par_names.append(ec_name)

    # formula arguments: @0=poi, @1=exp_c, @2..=poly pars
    arg_list = ROOT.RooArgList()
    arg_list.add(poi)
    arg_list.add(exp_par)
    for p in poly_pars:
        arg_list.add(p)

    poly_terms = "+".join(
        "@%d*pow(@0,%d)" % (i + 2, i + 1) for i in range(order)
    )
    formula = "exp(@1*@0)*(1+%s)" % poly_terms

    shape = ROOT.RooGenericPdf(name+"_shape", name+"_shape", formula, arg_list)
    p_objs.append(arg_list)

    return shape, par_names, p_objs


shape_map = {
        "bern":      bernShape,
        "poly":      polyShape,
        "exp":       expShape,
        "polyExp":   polyExpShape,
        "expPoly":   expPolyShape,
        "bernPower": bernPowerShape,
        }



class Fitter(object):
    def __init__(self,poi = ['x'], debug = False, outdir=""):
        self.outdir = outdir
        # Keep the scratch cache file inside outdir (not the cwd) so that if a
        # fit crashes before cleanup, the orphan lands in the job's own results
        # dir rather than polluting the repo root.
        self.cache_name = os.path.join(outdir, "cache%i.root"%(random.randint(0, 1e+6)))
        print("Making cache %s "  % self.cache_name)
        self.cache=ROOT.TFile(self.cache_name,"RECREATE")
        self.cache.cd()
        self.debug = debug
        self.cleanedup = False
        self.objs = []
        self.par_names = []

        self.w=ROOT.RooWorkspace("w","w")
        self.dimensions = len(poi)
        self.poi=poi
        for v in poi:
            self.w.factory(v+"[1,161]")


    def __del__(self):
        # Backstop only; callers should invoke delete() explicitly (ideally in a
        # finally:) since __del__ is not guaranteed to run if the worker is killed.
        if(not getattr(self, "cleanedup", True)): self.delete()

    def delete(self):
        # Close the TFile first so ROOT releases the handle, then remove it.
        try:
            if getattr(self, "cache", None) is not None and self.cache.IsOpen():
                self.cache.Close()
        except Exception:
            pass
        if os.path.exists(self.cache_name):
            os.remove(self.cache_name)
        self.cleanedup = True

    def importBinnedData(self,histogram,poi = "x",name = "data", regions=[]):
        cList = ROOT.RooArgList()
        var = self.w.var(poi)
        cList.add(var)
        axis=histogram.GetXaxis()
        mini=axis.GetXmin()
        maxi=axis.GetXmax()
        bins=axis.GetNbins()
        binningx =[]
        for i in range(1,bins+2):
            binningx.append(axis.GetBinLowEdge(i))

        if(len(regions) == 0):
            var.setMax(maxi)
            var.setMin(mini)
        else:
            for reg_name,reg_low,reg_high in regions:
                var.setRange(reg_name, reg_low, reg_high)
        if(self.debug): 
            print (" set binning "+str(binningx)) 
        var.setBinning(ROOT.RooBinning(len(binningx)-1,array("d",binningx)))
        #a = self.w.var(p).getBinning()
        #for b in range(0,a.numBins()+1):
            #print a.binLow(b)
        dataHist=ROOT.RooDataHist(name,name,cList,histogram)
        getattr(self.w,'import')(dataHist)

        self.objs.append(dataHist)

        self.w.Print()
        return dataHist

    def fetch(self,var):
        #import pdb; pdb.set_trace()
        self.w.var(var).Print()
        print("Fetching value " ,self.w.var(var).getVal()  )
        print("Fetching error " ,self.w.var(var).getError())
        return (self.w.var(var).getVal(), self.w.var(var).getError())

    def getFunc(self,model = "model"):
        return self.w.pdf(model)

    def getData(self,data = "data"):
        return self.w.data(data)

    def getVar(self,var = "mjj"):
        return self.w.var(var)

    def getW(self):
        return self.w

    def fit(self,model = "model",data="data",options=[]):
        fit_data = self.w.data(data)
        print(fit_data)
        print(options, options[0])
        self.w.Print()
        self.w.pdf(model).Print()

        if len(options)==0:
            fitresults = self.w.pdf(model).fitTo(fit_data)
        if len(options)==1:
            fitresults = self.w.pdf(model).fitTo(fit_data,options[0])	    
        if len(options)==2:
            fitresults = self.w.pdf(model).fitTo(fit_data,options[0],options[1])
        if len(options)==3:
            fitresults = self.w.pdf(model).fitTo(fit_data,options[0],options[1],options[2])
        if len(options)==4:
            fitresults = self.w.pdf(model).fitTo(fit_data,options[0],options[1],options[2],options[3])

        if fitresults:
            fitresults.Print() 
            f = ROOT.TFile.Open(self.outdir + 'fitresults.root','RECREATE')
            fitresults.SetName("fitresults")
            fitresults.Write()
            f.Close()	 
        return fitresults 

    def getLegend(self):
        self.legend = ROOT.TLegend(0.7510112,0.7183362,0.8502143,0.919833)
        self.legend.SetTextSize(0.032)
        self.legend.SetLineColor(0)
        self.legend.SetShadowColor(0)
        self.legend.SetLineStyle(1)
        self.legend.SetLineWidth(1)
        self.legend.SetFillColor(0)
        self.legend.SetFillStyle(0)
        self.legend.SetMargin(0.35)
        return self.legend

    def projection(self,model="model",data="data",poi="x",filename="fit.root",binning=0,logy=False,xtitle='x',show_error=True, mass=1000):


        f = ROOT.TFile.Open(self.outdir + "fitresults.root",'READ')

        linear_errors = False
        ndata = self.w.data(data).numEntries()
        self.frame=self.w.var(poi).frame(ROOT.RooFit.Bins(ndata))
        nbins = self.w.var(poi).getBinning().numBins()
        if(binning == 0): binning = self.w.var(poi).getBinning()


        if f: 
            fr = f.Get('fitresults')
        else:
            fr = 0
            print("No fit result found (fitresults.root), plotting model only")

        if binning:
            self.w.data(data).plotOn(self.frame,ROOT.RooFit.Invisible())
            if fr: 
                if(show_error):
                    self.w.pdf(model).plotOn(self.frame,ROOT.RooFit.VisualizeError(fr,1, linear_errors),ROOT.RooFit.FillColor(ROOT.kRed-7),ROOT.RooFit.LineColor(ROOT.kRed-7),ROOT.RooFit.Name(fr.GetName()))
                self.w.pdf(model).plotOn(self.frame,ROOT.RooFit.LineColor(ROOT.kRed+1))	 
            self.w.data(data).plotOn(self.frame, ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson))

        self.legend = self.getLegend()
        self.legend.AddEntry( self.w.pdf(model)," Full PDF","l")

        # Unique name per canvas: reusing "c" makes ROOT auto-delete the
        # previous same-named canvas, which can use-after-free and segfault
        # after many projection() calls in one process (crashed M19.6/M26.2).
        self.c=ROOT.TCanvas("c_%i" % random.randint(0, 1000000000),"c")
        if logy:
            self.frame.SetMinimum(1)
            self.frame.SetMaximum(1e+7)
            self.c.SetLogy()
        self.c.cd()
        self.frame.Draw()
        self.frame.GetYaxis().SetTitle('events')
        self.frame.GetXaxis().SetTitle(xtitle)
        self.frame.SetTitle('')
        self.c.Draw()

        self.legend.Draw("same")	    
        self.c.SaveAs(filename)

        chi2 = ROOT.RooChi2Var("chi2", "chi2", self.w.pdf(model), self.w.data(data))
        chi2_val = chi2.getVal()

        pullDist = self.frame.pullHist()
        nfloat = self.w.pdf(model).getParameters(self.w.data(data)).selectByAttrib("Constant", False).getSize()

        ndof = nbins - nfloat

        return chi2_val, ndof

    def signalGaus(self,name = 'model',poi="MVV",mass=0):
        #single crystall ball plus gaussian

        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

        self.w.factory("MH[1000]")
        self.w.factory("mean[%.1f,%.1f,%.1f]"%(mass,0.8*mass,1.2*mass))
        self.w.factory("sigma[%.1f,%.1f,%.1f]"%(mass*0.02,mass*0.005,mass*0.10))
        self.w.factory("Gaussian::"+name+"(%s,mean,gsigma)"%poi)

        self.sig_shape_params = ['mean', 'sigma'] 
    

    def signalCB(self,name = 'model',poi="MVV",mass=0):
        #single crystall ball plus gaussian

        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

        self.w.factory("MH[1000]")
        self.w.factory("mean[%.1f,%.1f,%.1f]"%(mass,0.8*mass,1.2*mass))
        self.w.factory("sigma[%.1f,%.1f,%.1f]"%(mass*0.02,mass*0.005,mass*0.10))
        self.w.factory("alpha[0.85,0.60,1.20]")
        self.w.factory("sign[6,0.1,150]")
        self.w.factory("CBShape::"+name+"(%s,mean,sigma,alpha,sign)"%poi)

        self.sig_shape_params = ['mean', 'sigma', 'alpha', 'sign']



    def signalCBGaus(self,name = 'model',poi="MVV",mass=0):
        #single crystall ball plus gaussian

        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

        self.w.factory("MH[1000]")
        self.w.factory("mean[%.1f,%.1f,%.1f]"%(mass,0.8*mass,1.2*mass))
        self.w.factory("sigma[%.1f,%.1f,%.1f]"%(mass*0.02,mass*0.005,mass*0.10))
        self.w.factory("alpha[0.85,0.60,1.20]")
        self.w.factory("sign[6,0.1,150]")
        self.w.factory("scalesigma[2.0,1.2,3.6]")
        gsigma = ROOT.RooFormulaVar("gsigma","@0*@1", ROOT.RooArgList(self.w.var("sigma"),self.w.var("scalesigma")))
        getattr(self.w,'import')(gsigma)
        self.w.factory("Gaussian::gauss(%s,mean,gsigma)"%poi)
        self.w.factory("CBShape::cb(%s,mean,sigma,alpha,sign)"%poi)
        self.w.factory('SUM::'+name+'(sigfrac[0.0,0.0,0.850]*gauss,cb)')

        self.sig_shape_params = ['mean', 'sigma', 'alpha', 'sign', 'scalesigma', 'sigfrac']
    
    def signalDCB(self, name='model', poi="MVV", mass=0):
        #double crystall ball
    
        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
        
        self.w.factory("MH[1000]")
        self.w.factory("mean[%.4f,%.4f,%.4f]"%(mass,0.8*mass,1.2*mass))
        # Use %g/%.4f precision: %.1f rounded the lower bound mass*0.005 up (e.g. M10:
        # 0.05 -> 0.1), which forced a spurious 0.1 GeV sigma floor on low-mass fits.
        self.w.factory("sigma[%.4f,%.4f,%.4f]"%(mass*0.02,mass*0.005,mass*0.10))
        self.w.factory("alpha[1.2,0.0,18]")
        self.w.factory("alpha2[1.2,0.0,10]")
        self.w.factory("sign[5,0,600]")
        self.w.factory("sign2[5,0,50]")
        self.w.factory("DoubleCB::"+name+"(%s,mean,sigma,alpha,sign,alpha2,sign2)"%poi)

        self.sig_shape_params = ['mean', 'sigma', 'alpha', 'sign', 'alpha2', 'sign2']

    def signalShape(self, name='model', poi='MVV', mass=0, shape='gaus'):
        if(shape == 'gaus'):
            self.signalGaus(name,poi,mass)
        elif(shape == 'CB'):
            self.signalCB(name,poi,mass)
        elif(shape == 'DCB'):
            self.signalDCB(name,poi,mass)
        elif(shape == 'CBgaus'):
            self.signalCBGaus(name,poi,mass)
        else:
            print("Unknown signal shape %s" % shape)
            exit(1)
        return




    def bkgShape(self, func_form='bern', name = 'model',poi="m",order=2, par_label=None):
        # par_label sets the prefix of the shape parameter names; it defaults to the
        # family name ('bern_p0', ...). Pass it to follow a naming convention, and pass
        # the same value here and to the final workspace build, since the F-test fit
        # values are cached under these names and re-read as starting points.

        if(type(poi) == str): poi = self.w.var(poi)

        if par_label is None: par_label = func_form

        if(func_form in shape_map.keys()):
            shape, par_names, objs = shape_map[func_form](name = name, poi=poi, order=order, par_label=par_label)
        else:
            print("Shape %s not implemented!" % shape)
            exit(1)
        
        self.objs.extend(objs)
        self.par_names.extend(par_names)

        norm = 10000.
        norm_name = name + "_norm"
        norm_var = ROOT.RooRealVar(norm_name, norm_name, norm, 0., 1e8)

        model = ROOT.RooExtendPdf(name, name, shape, norm_var)

        getattr(self.w,'import')(model)
        self.objs.append((shape, norm_var, model))

        return model


    def addDCBSignalShape(self, name, poi, jsonFile, scale={},
                          resolution={}):

        
        pdfName="_".join([name,])
        
        scaleStr='0'
        resolutionStr='0'

        scaleSysts=[]
        resolutionSysts=[]
        for syst,factor in scale.items():
            self.w.factory(syst+"[0,-0.1,0.1]")
            scaleStr=scaleStr+"+{factor}*{syst}".format(factor=factor,syst=syst)
            scaleSysts.append(syst)
        for syst,factor in resolution.items():
            self.w.factory(syst+"[0,-0.5,0.5]")
            resolutionStr=resolutionStr+"+{factor}*{syst}".format(factor=factor,syst=syst)
            resolutionSysts.append(syst)
            
        if(type(poi) == str): 
            poi = self.w.var(poi)
    
        f = ROOT.TFile(jsonFile,'READ')
        meanG = f.Get('mean')
        sigmaG = f.Get('sigma')
        alpha_oneG = f.Get('alpha')
        sign_oneG = f.Get('sign')
        alpha_twoG = f.Get('alpha2')
        sign_twoG = f.Get('sign2')

        x = ROOT.Double(0.)
        mean = ROOT.Double(0.)
        meanG.GetPoint(0,x,mean)
        sigma = ROOT.Double(0.)
        sigmaG.GetPoint(0,x,sigma)
        alpha_one = ROOT.Double(0.)
        alpha_oneG.GetPoint(0, x, alpha_one)
        alpha_two = ROOT.Double(0.)
        alpha_twoG.GetPoint(0, x, alpha_two)
        sign_one = ROOT.Double(0.)
        sign_oneG.GetPoint(0,x,sign_one)
        sign_two = ROOT.Double(0.)
        sign_twoG.GetPoint(0,x,sign_two)

        meanVar = "_".join(["MEAN", name, ])
        self.w.factory(
            "expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(
                name=meanVar, param=mean, vv_syst=scaleStr,
                vv_systs=','.join(scaleSysts)))

        sigmaVar = "_".join(["SIGMA", name, ])
        self.w.factory(
            "expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(
                name=sigmaVar, param=sigma, vv_syst=resolutionStr,
                vv_systs=','.join(resolutionSysts)))

        alphaOneVar = "_".join(["ALPHAONE", name, ])        
        alpha_one = ROOT.RooRealVar(alphaOneVar,alphaOneVar,alpha_one)
        getattr(self.w, 'import')(alpha_one)

        alphaTwoVar = "_".join(["ALPHATWO", name, ])        
        alpha_two = ROOT.RooRealVar(alphaTwoVar, alphaTwoVar, alpha_two)
        getattr(self.w, 'import')(alpha_two)
        
        signOneVar = "_".join(["SIGNONE", name, ])
        sign_one = ROOT.RooRealVar(signOneVar, signOneVar, sign_one)    
        getattr(self.w, 'import')(sign_one)
        
        signTwoVar = "_".join(["SIGNTWO", name, ])
        sign_two = ROOT.RooRealVar(signTwoVar, signTwoVar, sign_two)    
        getattr(self.w, 'import')(sign_two)
        
        #dcbFunc = "_".join(["dcb", name, ])
        #import pdb; pdb.set_trace()
        dcb = ROOT.RooDoubleCB(pdfName, pdfName, poi,
                                      self.w.function(meanVar),
                                      self.w.function(sigmaVar), alpha_one,
                                      sign_one, alpha_two, sign_two)

        getattr(self.w, 'import')(dcb)

        self.objs.append(dcb)

        return dcb, [alpha_one, alpha_two, sign_one, sign_two]


