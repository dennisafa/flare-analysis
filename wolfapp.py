from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile, KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize
import time
import os
from numpy import asarray
import george
from george import kernels

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
        sap = star.remove_outliers()
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

    def setbounds (self, p):
        p = asarray(p).ravel()
        bounds = np.zeros([len(p), 2])
        k = 0
        for i in range(len(p)):
            for j in range(2):
                if j < 1:
                    bounds[i][j] = p[i]
                else:
                    bounds[i][j] = p[i] + p[i] ** 1/6
        return bounds

    def computegeorge (self):
        kernel = np.var(self.flux) * kernels.ExpSquaredKernel(0.5) * kernels.ExpSine2Kernel(log_period = 0.5, gamma=1)
        self.gp = george.GP(kernel) # testing
        self.gp.compute(self.time, self.flux)
        result = minimize(self.neg_ln_like, self.gp.get_parameter_vector(), jac=self.grad_neg_ln_like)
        self.gp.set_parameter_vector(result.x)
        pred_mean, pred_var = self.gp.predict(self.flux, self.time, return_var=True)
        return pred_mean




def run():
    w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
    lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
    steps = 100
    for slice in range(0, len(lc359.flux), steps):
        flare = strPlot(lc359, slice, slice + steps)
        plotflares(flare)

def checkzero(l):
    if len(l) == 0:
        return False
    else:
        return True

def plotflares(flare):

    flareorig = flare.flux
    finalmodel = 0
    count = 0
    list_plots = []
    while len(fd.flaredetect(flare.flux)) > 0:
        tempmodel = loopcomp(flare)
        finalmodel += tempmodel

        # figplot.plot(flare.time, (flare.flux-tempmodel.flatten()), color = 'Black', linestyle= '--', label = 'Flares subtracted')
        # figplot.plot(flare.time, flare.flux, color = 'Red', label = 'Original')
        # figplot.legend(loc = 'upper left')
        # figplot.xlabel('Time - BJD')
        # figplot.ylabel('Flux - Normalized 0')
        # list_plots.append(figplot)
        # figplot.clf()

        flare.flux = flare.flux-tempmodel.flatten()
        count+=1

    # figplotmodel = pl
    # figplotmodel.plot(flare.time, flare.flux, color = 'Black', linestyle = '--', label ='Flares subtracted')
    # figplotmodel.plot(flare.time, finalmodel.flatten(), color = 'Blue', label = 'Flare model')
    # figplotmodel.legend(loc = 'upper left')
    # figplotmodel.xlabel('Time - BJD')
    # figplotmodel.ylabel('Flux - Normalized 0')
    # figplotmodel.clf()

    if finalmodel is not 0:
        finalplot = pl
        finalplot.plot(flare.time, flareorig, color = 'Red', label = 'Original flux')
        finalplot.plot(flare.time, finalmodel.flatten(), color = 'Black', linestyle= '--', label = 'Final model')
        finalplot.show()



def loopcomp(flare):
    guessparams = flare.guesspeaks()
    notempty = checkzero(flare.detflares)
    if notempty:
        #georgemodel = flare.computegeorge()
        bounds = flare.setbounds(guessparams)

        fitparams = flare.fit(guessparams, bounds)
        model = flare.getmodel(fitparams, [flare.time, flare.flux, flare.nflares])

        return model