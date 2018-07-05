from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile, KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize
from numpy import asarray
import george
from george import kernels
from scipy.signal import savgol_filter as sf

class strPlot:

    def neg_ln_like(self, p):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(self.flux)

    def grad_neg_ln_like(self, p):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(self.flux)


    def findflaremax(self, flux):
        return max(flux)

    def findflaretime(self, flarepeak, flux, time): # retrieves the time of the flare
        tof = time
        for i, flare in enumerate(flux):
            if flare == flarepeak:
                return tof[i]

    def getmodel(self, p, data): # computes the model of the flares using appaloosa's aflare1 function
        time, y, nflares = data
        p = np.exp(p)
        model = np.zeros_like([time])
        p = np.reshape(p, (nflares, 3))
        for i in range(nflares):
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=True, uptime=10)
        return model

    def ng_ln_like(self, p, data):
        _, y, _ = data
        model = self.getmodel(p, data)
        return np.sum((model - y) ** 2)

    def __init__(self, star, range1, range2): # cleans the list of flux, normalizes to 0
        sap = star.remove_outliers().remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1
        self.flux = [number / scipy.std(self.flux) for number in self.flux]
        self.time = self.time[np.isfinite(self.flux)]
        self.flux = asarray(self.flux)
        self.flux = self.flux[np.isfinite(self.flux)]


    def guesspeaks(self): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetect(self.flux)
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findflaretime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.004, self.flarepeak]
            self.params[i, :] = p
        return np.log(self.params)


    def fit(self, p, bounds):
        result = minimize(self.ng_ln_like, p, args=[self.time, self.flux, self.nflares], method='L-BFGS-B', bounds=bounds)
        return result.x

    def min(self):
        result = minimize(self.ng_ln_like, np.log(self.params), args=[self.time, self.flux, self.nflares], method='L-BFGS-B', bounds=self.bounds)
        return result.x

    def setbounds (self, p):
        p = asarray(p).ravel()
        bounds = np.zeros([len(p), 2])
        for i in range(len(p)):
            for j in range(2):
                if j < 1:
                    bounds[i][j] = p[i]
                else:
                    bounds[i][j] = p[i] + p[i] ** 1/6
        self.bounds = bounds
        return bounds

    def computegeorge (self):
        testmodel = 1 * np.sin(2*np.pi/ 2.5) * self.flux

        kernel = np.var(self.flux) * kernels.Matern52Kernel(metric=0.5) * kernels.CosineKernel(log_period=2.5) # a * cos(2pi/T * (t-3082) ) *
        self.gp = george.GP(kernel)
        self.gp.compute(self.time, self.flux)
        print(kernel)
        #result = minimize(self.neg_ln_like, self.gp.get_parameter_vector(), jac=self.grad_neg_ln_like)
        #self.gp.set_parameter_vector(result.x)
        pred_mean, pred_var = self.gp.predict(self.flux, self.time, return_var=True)
        return pred_mean




def run():
    # stars to test george on: 206208968

    w359 = KeplerTargetPixelFile.from_archive(201885041)
    lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
    steps, iters = 300, 2000
    for slice in range(0, iters, steps):
        flare = strPlot(lc359, slice, slice + steps)
        sub_flares(flare)
        #make_george(flare)

def checkzero(l):
    if len(l) == 0:
        return False
    else:
        return True

def plotflares(flare): # run the flux list through the modeling process, then subtract models from original flux

    flareorig = flare.flux
    finalmodel = 0
    count = 0
    model_list = []
    while len(fd.flaredetect(flare.flux)) > 0: # while flares are still being detected, compute its model and plot
        tempmodel = get_model(flare)
        finalmodel += tempmodel

        figplot = pl
        figplot.plot(flare.time, tempmodel.flatten(), color = 'Black', linestyle= '--', label = 'Model')
        figplot.plot(flare.time, flare.flux, color = 'Red', label = 'Flux')
        figplot.legend(loc = 'upper left')
        figplot.xlabel('Time - BJD')
        figplot.ylabel('Flux - Normalized 0')
        figplot.show() # plot the model for each set of flux
        figplot.clf()
        model_list.append(tempmodel)


        flare.flux = flare.flux-tempmodel.flatten()
        count+=1

    # multi_model = pl
    # pl.plot(flare.time, flareorig, color="Grey")
    # #pl.show()
    # pl.clf()
    # for model in model_list:
    #     multi_model.plot(flare.time, model.flatten(), color='Black')
    # multi_model.show()


    # figplotmodel = pl
    # figplotmodel.plot(flare.time, flare.flux, color = 'Black', linestyle = '--', label ='Flares subtracted')
    # figplotmodel.plot(flare.time, finalmodel.flatten(), color = 'Blue', label = 'Flare model')
    # figplotmodel.legend(loc = 'upper left')
    # figplotmodel.xlabel('Time - BJD')
    # figplotmodel.ylabel('Flux - Normalized 0')
    # figplotmodel.clf()


    if finalmodel is not 0: # if the model was computed then plot
        finalplot = pl
        finalplot.plot(flare.time, flareorig, color = 'Blue', label = 'Original flux')
        finalplot.plot(flare.time, finalmodel.flatten(), color = 'Black', linestyle= '--', label = 'Final model')
        #print("Run")
        finalplot.show()



def get_model(flare):
    guessparams = flare.guesspeaks() # returns the parameters of where and when each flare occured
    notempty = checkzero(flare.detflares) # checks for flares to exit in the data set before modeling
    if notempty:
        bounds = flare.setbounds(guessparams)
        georgemodel = flare.computegeorge()
        #bounds = flare.setbounds(guessparams) # set the bounds for each parameter

        fitparams = flare.fit(guessparams, bounds) # fit the parameters with a minimization process
        model = flare.getmodel(fitparams, [flare.time, flare.flux, flare.nflares]) + georgemodel # model the flares using appaloosa's aflare

        return model


def make_period_fit(flare):

    period = sf(flare.flux, 51, 3)
    periodclean = sf(period, 51, 3)
    periodcleanclean = sf(periodclean, 51, 3) # lol

    dx = np.diff(flare.time) / np.diff(periodcleanclean)
    period_change = sign_change(dx)
    print(period_change)
    print("Rotational period: " + str(flare.time[period_change[1]] - flare.time[period_change[0]]))
    return periodcleanclean

def sign_change(model): # returns index where the first derivative changes sign from positive to negative
    change_sign = []
    j = 0
    while j < len(model)-1:
        if model[j] > 0:
            while model[j] > 0 and j < len(model)-1:
                j+=1
            else:
                change_sign.append(j)
        else:
            j+=1
    return change_sign


def sub_flares(flare):
    flare_orig = flare.flux
    while len(fd.flaredetect(flare.flux)) > 0: # while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        flare.flux = flare.flux-tempmodel.flatten()
    period = make_period_fit(flare) # then model this subtracted flare flux set
    pl.plot(flare.time, flare_orig)
    pl.plot(flare.time, period)
    pl.show()

def sub_flare_model(flare):
    guessparams = flare.guesspeaks()  # returns the parameters of where and when each flare occured
    notempty = checkzero(flare.detflares)  # checks for flares to exit in the data set before modeling
    if notempty:
        bounds = flare.setbounds(guessparams)

        fitparams = flare.fit(guessparams, bounds)  # fit the parameters with a minimization process
        model = flare.getmodel(fitparams, [flare.time, flare.flux,
                                           flare.nflares])

        return model
