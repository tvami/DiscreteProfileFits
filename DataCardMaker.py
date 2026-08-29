import ROOT
ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
import json
import sys
import ctypes  
from array import array
from Fitter import Fitter
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

class DataCardMaker:
    # bkg_name / pdf_index_name set the names of the background envelope (RooMultiPdf)
    # and of its discrete index in the workspace and the datacard. The defaults keep the
    # historical generic names; pass CMS-convention names (e.g. 'CMS_<analysisID>_bkg')
    # when the cards are released. The envelope's yield is always <bkg_name>_norm, since
    # combine looks the normalization up by that name.
    def __init__(self,tag, outdir="", bkg_name="multi_pdf", pdf_index_name="pdf_index"):
        self.systematics=[]
        self.tag="mass"+"_"+tag
        self.outdir = outdir
        self.bkg_name = bkg_name
        self.pdf_index_name = pdf_index_name
        self.rootFile = ROOT.TFile(self.outdir + "datacardInputs_%s.root"%self.tag,"RECREATE")
        self.rootFile.cd()
        self.w=ROOT.RooWorkspace("w","w")
        self.luminosity = 1.0
        self.contributions=[]
        self.systematics=[]
        self.bkg_shapes = []
        self.bkg_pars = []
        self.bkg_shape_names = []

    def delete(self):
        #if self.w: 
        #    try:
        #        self.w.Delete()
        #        self.rootFile.Close()
        #        self.rootFile.Delete()
        #    except:
        #        print("Delete ws")
        return
      
    def makeCard(self):

        f = open(self.outdir + "datacard_"+self.tag+'.txt','w')
        f.write('imax 1\n')
        f.write('jmax {n}\n'.format(n=len(self.contributions)-1))
        f.write('kmax *\n')
        f.write('-------------------------\n')
        for c in self.contributions:
            f.write('shapes {name} {channel} {file}.root w:{pdf}\n'.format(name=c['name'],channel=self.tag,file="datacardInputs_"+self.tag,pdf=c['pdf']))
        f.write('shapes {name} {channel} {file}.root w:{name}\n'.format(name="data_obs",channel=self.tag,file="datacardInputs_"+self.tag))
        f.write('-------------------------\n')
        f.write('bin '+self.tag+'\n')
        f.write('observation  -1\n')
        f.write('-------------------------\n')
        f.write('bin\t') 

        for shape in self.contributions: f.write(self.tag+'\t')
        f.write('\n')

        #Sort the shapes by ID 
 
        shapes = sorted(self.contributions,key=lambda x: x['ID'])
        #print names
        f.write('process\t')
        for shape in shapes:
            f.write(shape['name']+'\t')
        f.write('\n')

        #Print ID
        f.write('process\t')
        for shape in shapes:
            f.write(str(shape['ID'])+'\t')
        f.write('\n')

        #print rates
        f.write('rate\t')
        for shape in shapes:
            f.write(str(shape['yield'])+'\t')
        f.write('\n')


        #Now systematics
        for syst in self.systematics:
            if syst['kind'] == 'param':
                f.write(syst['name']+'\t'+'param\t' +str(syst['values'][0])+'\t'+str(syst['values'][1])+'\n')
            elif syst['kind'] == 'flatParam':
                f.write(syst['name']+'\t'+'flatParam\n')
            elif 'rateParam' in syst['kind']:
                line = syst['name']+'\t'+str(syst['kind'])+"\t"+str(syst['bin'])+"\t"+str(syst['process'])+"\t"+str(syst['values'])+"\t"+str(syst['variables'])
                line+='\n' 
                f.write(line)
                
            elif syst['kind'] == 'discrete':
                f.write(syst['name']+'\t'+'discrete\n')

            elif syst['kind'] == 'lnN' :
                f.write(syst['name']+'\t'+ 'lnN\t' )
                for shape in shapes:
                    has=False
                    for name,v in syst['values'].items():
                        if shape['name']==name:
                            f.write(str(v)+'\t' )
                            has=True
                            break;
                    if not has:
                            f.write('-\t' )
                f.write('\n' )
            elif syst['kind'] == 'lnU': 
                f.write(syst['name']+'\t'+ 'lnU\t' )
                for shape in shapes:
                    has=False
                    for name,v in syst['values'].items():
                        if shape['name']==name:
                            f.write(str(v)+'\t' )
                            has=True
                            break;
                    if not has:
                            f.write('-\t' )
                f.write('\n' )  

        #include discrete profiling nuisance
        f.write("%s discrete" % self.pdf_index_name)
                        
        f.close()



        self.rootFile.cd()
        self.w.Write()
        self.rootFile.Close()
    
    def importBinnedData(self,filename,histoname,poi_name,name = "data_obs",scale=1):
        f=ROOT.TFile(filename)
        histogram=f.Get(histoname)
        histogram.Scale(scale)

        self.nData = histogram.Integral();
        cList = ROOT.RooArgList()

        axis=histogram.GetXaxis()
        mini=axis.GetXmin()
        maxi=axis.GetXmax()
        bins=axis.GetNbins()

        self.w.factory(poi_name + "[%.0f,%.0f]" % (mini, maxi))  
        self.poi = self.w.var(poi_name)

        self.poi.setMin(mini)
        self.poi.setMax(maxi)
        self.poi.setBins(bins)
        self.poi.setBins(bins,"cache")
        dataHist = ROOT.RooDataHist(name, name, self.poi, histogram)

        getattr(self.w,'import')(dataHist)
    
    def addSystematic(self,name,kind,values,bin="",process="",variables="",addPar = ""):
        if kind != 'rateParam': self.systematics.append({'name':name,'kind':kind,'values':values })
        else: self.systematics.append({'name':name,'kind':kind,'bin':bin,'process':process,'values':values,'variables':variables})
    
    def addFixedYieldFromFile(self,name,ID,filename,histoName,norm = 1680.0):
        pdfName="_".join([name,self.tag])

        self.contributions.append({'name':name,'pdf':pdfName,'ID':ID,'yield':norm})
        return norm

    def addFloatingYield(self,name,ID,filename,histoName,mini=0,maxi=1e+9,constant=False):
        pdfName="_".join([name,self.tag])
        pdfNorm="_".join([name,self.tag,"norm"])
        events=100.0
        self.w.factory("{name}[{val},{mini},{maxi}]".format(name=pdfNorm,val=events,mini=mini,maxi=maxi))       
        if constant:
            self.w.var(pdfNorm).setConstant(1)
        self.contributions.append({'name':name,'pdf':pdfName,'ID':ID,'yield':events})
        return events

    def addSigShape(name, jsonFile, **args):
        with open(jsonFile, 'r') as f:
            params = json.load(f)

        if(params['shape'] == 'DCB'):
            self.addDCBSignalShape(name, params, **args)

    def addDCBSignalShape(self, name, jsonFile, scale={}, resolution={}, xmin=-1.0, xmax = -1.0):

        pdfName="_".join([name,self.tag])

        with open(jsonFile, 'r') as f:
            params = json.load(f)

        
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
            
        x = ctypes.c_double(0.0)
        mean = params['mean']
        sigma = params['sigma']

        if(xmin > 0. and xmax > 0.): #account for rescaling
            mean = (mean - xmin)/(xmax - xmin)
            sigma = sigma / (xmax - xmin)

        print("adding mean")
        meanVar = "_".join(["MEAN",name,self.tag])
        meanRaw = "_".join(["MEANRAW",name,self.tag])
        meanRaw_root = ROOT.RooRealVar(meanRaw, meanRaw,mean)
        getattr(self.w, "import")(meanRaw_root)
        self.w.factory("expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(name=meanVar,param=meanRaw,vv_syst=scaleStr,vv_systs=','.join(scaleSysts)))

        print("adding sigma")
        sigmaVar = "_".join(["SIGMA",name,self.tag])
        sigmaRaw = "_".join(["sigmaRAW",name,self.tag])
        sigmaRaw_root = ROOT.RooRealVar(sigmaRaw, sigmaRaw,sigma)
        getattr(self.w, "import")(sigmaRaw_root)
        self.w.factory("expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(name=sigmaVar,param=sigmaRaw,vv_syst=resolutionStr,vv_systs=','.join(resolutionSysts)))

        alphaOneVar = "_".join(["ALPHAONE", name, self.tag])        
        alpha_one = ROOT.RooRealVar(alphaOneVar,alphaOneVar,params['alpha'])
        getattr(self.w, 'import')(alpha_one)

        alphaTwoVar = "_".join(["ALPHATWO", name, self.tag])        
        alpha_two = ROOT.RooRealVar(alphaTwoVar, alphaTwoVar, params['alpha2'])
        getattr(self.w, 'import')(alpha_two)
        
        signOneVar = "_".join(["SIGNONE", name, self.tag])
        sign_one = ROOT.RooRealVar(signOneVar, signOneVar, params['sign'])    
        getattr(self.w, 'import')(sign_one)
        
        signTwoVar = "_".join(["SIGNTWO", name, self.tag])
        sign_two = ROOT.RooRealVar(signTwoVar, signTwoVar, params['sign2'])    
        getattr(self.w, 'import')(sign_two)
        
        #dcbFunc = "_".join(["dcb", name, self.tag])
        #import pdb; pdb.set_trace()
        dcb = ROOT.RooDoubleCB(pdfName, pdfName, self.poi,
                                      self.w.function(meanVar),
                                      self.w.function(sigmaVar), alpha_one,
                                      sign_one, alpha_two, sign_two)

        getattr(self.w, 'import')(dcb)
        return dcb


    def addSignalShape(self,name,jsonFile,scale ={},resolution={}, xmin=-1.0, xmax = -1.0):
    
        pdfName="_".join([name,self.tag])

        with open(jsonFile, 'r') as f:
            params = json.load(f)

    
        #self.w.factory("MH[3000]")
        #self.w.var("MH").setConstant(1)
       
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
       
    
        f = ROOT.TFile(jsonFile,'READ')
        meanG = f.Get('mean')
        sigmaG = f.Get('sigma')
        alphaG = f.Get('alpha')
        scalesigmaG = f.Get('scalesigma')
        sigfracG = f.Get('sigfrac')
        signG = f.Get('sign')
        
        x = ctypes.c_double(0.0)
        mean = ctypes.c_double(0.0)
        meanG.GetPoint(0,x,mean)

        sigma = ctypes.c_double(0.0)
        sigmaG.GetPoint(0,x,sigma)  
        alpha = ctypes.c_double(0.0)
        alphaG.GetPoint(0,x,alpha)
        scalesigma = ctypes.c_double(0.0)
        scalesigmaG.GetPoint(0,x,scalesigma)
        sigfrac = ctypes.c_double(0.0)
        sigfracG.GetPoint(0,x,sigfrac)
        sign = ctypes.c_double(0.0)
        signG.GetPoint(0,x,sign)


        if(xmin > 0. and xmax > 0.): #account for rescaling
            mean = (mean - xmin)/(xmax - xmin)
            sigma = sigma / (xmax - xmin)
            
        
        print("adding mean")
        meanVar = "_".join(["MEAN",name,self.tag])
        meanRaw = "_".join(["MEANRAW",name,self.tag])
        meanRaw_root = ROOT.RooRealVar(meanRaw, meanRaw,mean)
        self.w.factory("expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(name=meanVar,param=meanRaw,vv_syst=scaleStr,vv_systs=','.join(scaleSysts)))

        print("adding sigma")
        sigmaVar = "_".join(["SIGMA",name,self.tag])
        sigmaRaw = "_".join(["sigmaRAW",name,self.tag])
        sigmaRaw_root = ROOT.RooRealVar(sigmaRaw, sigmaRaw,sigma)
        self.w.factory("expr::{name}('{param}*(1+{vv_syst})',{vv_systs},{param})".format(name=sigmaVar,param=sigmaRaw,vv_syst=resolutionStr,vv_systs=','.join(resolutionSysts)))
            
        print("adding others")
        alphaVar = "_".join(["ALPHA",name,self.tag])        
        alpha = ROOT.RooRealVar(alphaVar,alphaVar,alpha)
        getattr(self.w,'import')(alpha)
        
        sigfracVar = "_".join(["SIGFRAC",name,self.tag])
        sigfrac = ROOT.RooRealVar(sigfracVar,sigfracVar,sigfrac)
        getattr(self.w,'import')(sigfrac)
        
        scalesigmaVar = "_".join(["SCALESIGMA",name,self.tag])
        scalesigma = ROOT.RooRealVar(scalesigmaVar,scalesigmaVar,scalesigma)
        getattr(self.w,'import')(scalesigma)
        
        signVar = "_".join(["SIGN",name,self.tag])
        sign = ROOT.RooRealVar(signVar,signVar,sign)    
        getattr(self.w,'import')(sign)

        gsigmaVar = "_".join(["GSIGMA",name,self.tag])      
        gsigma = ROOT.RooFormulaVar(gsigmaVar,"@0*@1", ROOT.RooArgList(self.w.function(sigmaVar),scalesigma))
        #getattr(self.w,'import')(gsigma)

        gaussFunc = "_".join(["gauss",name,self.tag])   
        gauss = ROOT.RooGaussian(gaussFunc, gaussFunc, self.poi, self.w.function(meanVar), gsigma)
        cbFunc = "_".join(["cb",name,self.tag])
        cb    = ROOT.RooCBShape(cbFunc, cbFunc,  self.poi, self.w.function(meanVar), self.w.function(sigmaVar), alpha, sign)
        model = ROOT.RooAddPdf(pdfName, pdfName, gauss, cb, self.w.var(sigfracVar)) 
        getattr(self.w,'import')(model)
        return model

    def buildBkgShape(self):
        #Make a RooCategory object. This will control which of the pdfs is "active"
        cat = ROOT.RooCategory(self.pdf_index_name,"Index of Pdf which is active")


        rList = ROOT.RooArgList()
        for shape in self.bkg_shapes:
            rList.add(shape)
            #getattr(self.w,'import')(shape)

        multi_pdf = ROOT.RooMultiPdf(self.bkg_name, "All pdfs", cat, rList)
        #automatically adds the -0.5 penalty to likelihood per degree of freedom

        norm_name = self.bkg_name + "_norm"
        norm_var = ROOT.RooRealVar(norm_name,"Number of background events", self.nData,0,1e8);

        self.addSystematic(norm_name, "flatParam", [])

        getattr(self.w,'import')(cat)
        getattr(self.w,'import')(multi_pdf)
        getattr(self.w,'import')(norm_var)

        self.contributions.append({'name':'background','pdf':self.bkg_name,'ID':1,'yield':1})


    def addBkgShapeNoTag(self,name,variable, fname, func_form = "bern",   order=4):

        pdfName=name+"_"+self.tag
    
        if self.w.var(MVV) == None: self.w.factory(MVV+"[0,10000]")

        poi = self.w.var(MVV)

        par_list = ROOT.RooArgList(poi)
    
        print("npars", nPars)

        print("fname ", fname)

        with open(fname, 'r') as f:
            params = json.load(f)

        for i in range(nPars):
            x = ctypes.c_double(0.)
            parsG[i].GetPoint(0,x,pars_val[i])
            pName="CMS_bkg_p%i"%i
            pars_val[i] = pars_val[i].value
            errUp=pars_val[i]+parsG[i].GetErrorYhigh(0)*100.
            errDown=pars_val[i]-parsG[i].GetErrorYlow(0)*100.
            print (i,pName,pars_val[i],parsG[i].GetErrorYhigh(0),parsG[i].GetErrorYlow(0),errUp,errDown)

        #TODO
        #model = bernShape(self.w, name=pdfName, poi=self.w.var(MVV), order = order, start_vals = pars_vals)
        model = None

        getattr(self.w,'import')(model)
